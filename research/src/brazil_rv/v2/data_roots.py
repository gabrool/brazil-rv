from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Mapping, Sequence

from .artifacts import sha256_file

DATA_ROOTS_ENV = "BRAZIL_RV_DATA_ROOTS"
DATA_ROOTS_SCHEMA = "BRAZIL_RV_DATA_ROOTS_V1"


@dataclass(frozen=True)
class ExternalFileResolution:
    recorded_path: str
    resolved_path: str
    bytes: int
    sha256: str
    recorded_root_prefix: str | None
    local_root_prefix: str | None
    override_file: str | None
    override_file_sha256: str | None

    def payload(self) -> dict[str, object]:
        return asdict(self)


def portable_name(path: str) -> str:
    """Return a basename without assuming the recorded path's host OS."""

    if "\\" in path or PureWindowsPath(path).drive:
        return PureWindowsPath(path).name
    return PurePosixPath(path).name


def _normalized(path: str) -> str:
    return path.replace("\\", "/").rstrip("/")


def _load_override(
    override_path: str | Path | None = None,
) -> tuple[Path, str, dict[str, str]]:
    configured = override_path or os.environ.get(DATA_ROOTS_ENV)
    if not configured:
        raise FileNotFoundError(
            f"foreign recorded data path requires {DATA_ROOTS_ENV}"
        )
    path = Path(configured).resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != DATA_ROOTS_SCHEMA:
        raise ValueError("data-root override schema is not recognized")
    roots = payload.get("roots")
    if not isinstance(roots, Mapping) or not roots:
        raise ValueError("data-root override must contain a nonempty roots mapping")
    mappings: dict[str, str] = {}
    for recorded, local in roots.items():
        if (
            not isinstance(recorded, str)
            or not recorded
            or not isinstance(local, str)
            or not local
        ):
            raise ValueError("data-root prefixes must be nonempty strings")
        normalized = _normalized(recorded)
        if normalized.casefold() in {value.casefold() for value in mappings}:
            raise ValueError("data-root override contains duplicate recorded prefixes")
        mappings[normalized] = local
    return path, sha256_file(path), mappings


def _matching_mapping(recorded: str, mappings: Mapping[str, str]) -> tuple[str, str]:
    normalized = _normalized(recorded)
    matches = [
        (prefix, local)
        for prefix, local in mappings.items()
        if normalized.casefold() == prefix.casefold()
        or normalized.casefold().startswith(prefix.casefold() + "/")
    ]
    if not matches:
        raise FileNotFoundError(f"no data-root override matches recorded path: {recorded}")
    return max(matches, key=lambda item: len(item[0]))


@lru_cache(maxsize=128)
def _verified_sha256(path: str, size: int, modified_ns: int) -> str:
    del size, modified_ns
    return sha256_file(Path(path))


def _verify(path: Path, record: Mapping[str, object]) -> tuple[int, str]:
    expected_size = int(record["bytes"])
    expected_sha = str(record["sha256"]).casefold()
    stat = path.stat()
    if stat.st_size != expected_size:
        raise ValueError(f"external artifact size mismatch: {path}")
    actual_sha = _verified_sha256(str(path), stat.st_size, stat.st_mtime_ns)
    if actual_sha.casefold() != expected_sha:
        raise ValueError(f"external artifact SHA-256 mismatch: {path}")
    return stat.st_size, actual_sha


def resolve_external_file(
    record: Mapping[str, object],
    *,
    override_path: str | Path | None = None,
    local_path: str | Path | None = None,
) -> tuple[Path, ExternalFileResolution]:
    """Resolve one immutable manifest record and verify its content identity."""

    recorded = record.get("path")
    if not isinstance(recorded, str) or not recorded:
        raise ValueError("external artifact record lacks its recorded path")
    if local_path is not None:
        resolved = Path(local_path).resolve(strict=True)
        size, digest = _verify(resolved, record)
        return resolved, ExternalFileResolution(
            recorded_path=recorded,
            resolved_path=str(resolved),
            bytes=size,
            sha256=digest,
            recorded_root_prefix=None,
            local_root_prefix=str(resolved.parent),
            override_file=None,
            override_file_sha256=None,
        )
    native = Path(recorded)
    if native.is_absolute() and native.is_file():
        resolved = native.resolve(strict=True)
        size, digest = _verify(resolved, record)
        return resolved, ExternalFileResolution(
            recorded_path=recorded,
            resolved_path=str(resolved),
            bytes=size,
            sha256=digest,
            recorded_root_prefix=None,
            local_root_prefix=None,
            override_file=None,
            override_file_sha256=None,
        )

    override, override_sha, mappings = _load_override(override_path)
    recorded_prefix, local_prefix = _matching_mapping(recorded, mappings)
    normalized = _normalized(recorded)
    suffix = normalized[len(recorded_prefix) :].lstrip("/")
    parts = PurePosixPath(suffix).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("external artifact path escapes its mapped data root")
    local_root = Path(local_prefix).resolve(strict=True)
    resolved = local_root.joinpath(*parts).resolve(strict=True)
    if not resolved.is_relative_to(local_root):
        raise ValueError("resolved external artifact escapes its mapped data root")
    size, digest = _verify(resolved, record)
    return resolved, ExternalFileResolution(
        recorded_path=recorded,
        resolved_path=str(resolved),
        bytes=size,
        sha256=digest,
        recorded_root_prefix=recorded_prefix,
        local_root_prefix=str(local_root),
        override_file=str(override),
        override_file_sha256=override_sha,
    )


def resolve_external_files(
    records: Sequence[Mapping[str, object]],
    required_names: Sequence[str],
    *,
    override_path: str | Path | None = None,
    local_root: str | Path | None = None,
) -> tuple[dict[str, Path], tuple[ExternalFileResolution, ...]]:
    by_name = {
        portable_name(str(record.get("path", ""))): record for record in records
    }
    if len(by_name) != len(records):
        raise ValueError("external artifact records contain duplicate basenames")
    paths: dict[str, Path] = {}
    resolutions: list[ExternalFileResolution] = []
    for name in required_names:
        record = by_name.get(name)
        if record is None:
            raise FileNotFoundError(f"external artifact manifest lacks {name}")
        path, resolution = resolve_external_file(
            record,
            override_path=override_path,
            local_path=None if local_root is None else Path(local_root) / name,
        )
        paths[name] = path
        resolutions.append(resolution)
    return paths, tuple(resolutions)
