from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .contract import FEATURE_CONTRACT_VERSION
from .data import (
    FEATURE_ARRAY_FILES,
    PEER_ARRAY_FILES,
    CacheWarmupReport,
    load_sample_index,
    validate_feature_store,
    warm_feature_store_cache,
)
from .run_profiles import RunProfile

SESSION_PREPARATION_VERSION = "B3_FEATURE_SESSION_PREPARATION_V1"
SESSION_PREPARATION_FILENAME = "session_preparation.json"
_IDENTITY_FILES = (
    "manifest.json",
    "sample_index.parquet",
    "date_index.parquet",
    "equity_index.parquet",
    "context_index.parquet",
    "global_context_index.parquet",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _boot_identity() -> str:
    linux_boot_id = Path("/proc/sys/kernel/random/boot_id")
    if linux_boot_id.is_file():
        return linux_boot_id.read_text(encoding="utf-8").strip()
    return f"{platform.node()}:{platform.system()}:{platform.release()}"


def _environment() -> dict[str, object]:
    return {
        "boot_identity": _boot_identity(),
        "hostname": socket.gethostname(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
    }


def _file_snapshot(store: Path) -> dict[str, dict[str, object]]:
    names = (*_IDENTITY_FILES, *FEATURE_ARRAY_FILES, *PEER_ARRAY_FILES)
    snapshot: dict[str, dict[str, object]] = {}
    for name in names:
        path = store / name
        stat = path.stat()
        row: dict[str, object] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
            "device": stat.st_dev,
            "inode": stat.st_ino,
        }
        if name in _IDENTITY_FILES:
            row["sha256"] = _sha256(path)
        snapshot[name] = row
    return snapshot


def _canonical_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def prepare_feature_store_session(
    output_path: Path,
    feature_store: Path,
    git_commit: str,
    run_profile: RunProfile,
) -> dict[str, object]:
    feature_store = feature_store.resolve()
    started = time.perf_counter()
    sample_index = validate_feature_store(feature_store)
    validation_seconds = time.perf_counter() - started
    validated_snapshot = _file_snapshot(feature_store)
    warmup = warm_feature_store_cache(feature_store, "selected")
    if _file_snapshot(feature_store) != validated_snapshot:
        raise RuntimeError("Feature store mutated during session preparation")
    total_seconds = time.perf_counter() - started
    body: dict[str, object] = {
        "version": SESSION_PREPARATION_VERSION,
        "status": "passed",
        "git_commit_sha": git_commit,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "resolved_feature_store": str(feature_store),
        "run_profile": run_profile.metadata(),
        "run_profile_identity_sha256": run_profile.identity_sha256,
        "environment": _environment(),
        "sample_count": sample_index.height,
        "file_snapshot": validated_snapshot,
        "cache_warmup": asdict(warmup),
        "performance": {
            "full_validation_seconds": validation_seconds,
            "cache_warmup_seconds": warmup.seconds,
            "total_preparation_seconds": total_seconds,
        },
        "validation": {
            "full_feature_store_validation": True,
            "peer_arrays_validated": True,
            "cache_warmed_once_for_session": True,
            "child_validation": "exact identity hashes plus immutable file snapshot",
        },
    }
    body["identity_sha256"] = _canonical_hash(body)
    _atomic_write_json(output_path, body)
    return body


def validate_session_preparation(
    path: Path,
    feature_store: Path,
    git_commit: str,
    run_profile: RunProfile,
) -> tuple[Any, CacheWarmupReport]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Session-preparation artifact is unreadable") from error
    if not isinstance(payload, dict):
        raise ValueError("Session-preparation artifact must be an object")
    recorded_hash = payload.pop("identity_sha256", None)
    if recorded_hash != _canonical_hash(payload):
        raise ValueError("Session-preparation artifact identity hash is invalid")
    payload["identity_sha256"] = recorded_hash
    expected_environment = _environment()
    if (
        payload.get("version") != SESSION_PREPARATION_VERSION
        or payload.get("status") != "passed"
        or payload.get("git_commit_sha") != git_commit
        or payload.get("feature_contract_version") != FEATURE_CONTRACT_VERSION
        or payload.get("resolved_feature_store") != str(feature_store.resolve())
        or payload.get("run_profile") != run_profile.metadata()
        or payload.get("run_profile_identity_sha256") != run_profile.identity_sha256
        or payload.get("environment") != expected_environment
        or payload.get("file_snapshot") != _file_snapshot(feature_store.resolve())
    ):
        raise ValueError("Session-preparation artifact is stale or identity-mismatched")
    sample_index = load_sample_index(feature_store)
    if sample_index.height != payload.get("sample_count"):
        raise ValueError("Prepared feature-store sample count changed")
    warmup = payload.get("cache_warmup")
    if not isinstance(warmup, dict):
        raise ValueError("Session-preparation cache warmup metadata is missing")
    return sample_index, CacheWarmupReport(**warmup)
