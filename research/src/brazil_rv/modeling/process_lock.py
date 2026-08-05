from __future__ import annotations

import csv
import json
import os
import socket
import subprocess
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .contract import RUN_OUTPUT_BASE

PRODUCTION_TRAINING_LOCK = RUN_OUTPUT_BASE / "_ops" / "production_training.lock"
LOCK_SCHEMA = "brazil_rv_process_lock"
LOCK_VERSION = 1
MALFORMED_INITIALIZING_SECONDS = 60.0
_BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")


@dataclass(frozen=True)
class HostIdentity:
    hostname: str
    boot_id: str | None


def _current_host_identity() -> HostIdentity:
    try:
        boot_id = _BOOT_ID_PATH.read_text(encoding="utf-8").strip() or None
    except OSError:
        boot_id = None
    return HostIdentity(socket.gethostname(), boot_id)


def _pid_is_active(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "nt":
        return Path(f"/proc/{pid}").exists()
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return True
    rows = list(csv.reader(result.stdout.splitlines()))
    return any(len(row) >= 2 and row[1] == str(pid) for row in rows)


def _owner_details(
    path: Path, payload: dict[str, object] | None, status: str
) -> dict[str, object]:
    payload = payload or {}
    return {
        "status": status,
        "path": str(path),
        "schema": payload.get("schema"),
        "version": payload.get("version"),
        "pid": payload.get("pid"),
        "hostname": payload.get("hostname"),
        "boot_id": payload.get("boot_id"),
        "purpose": payload.get("purpose"),
        "token": payload.get("token"),
        "created_at_utc": payload.get("created_at_utc"),
    }


def describe_lock_owner(path: Path, owner: dict[str, object]) -> str:
    fields = (
        "status",
        "pid",
        "hostname",
        "boot_id",
        "purpose",
        "token",
        "created_at_utc",
        "schema",
        "version",
    )
    details = " ".join(f"{field}={owner.get(field)!r}" for field in fields)
    return f"path={path} {details}"


def _lock_age_seconds(path: Path) -> float | None:
    try:
        return time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return None


def _is_valid_current_payload(payload: dict[str, object]) -> bool:
    boot_id = payload.get("boot_id")
    return (
        payload.get("schema") == LOCK_SCHEMA
        and type(payload.get("version")) is int
        and payload.get("version") == LOCK_VERSION
        and type(payload.get("pid")) is int
        and int(payload["pid"]) > 0
        and isinstance(payload.get("hostname"), str)
        and bool(payload["hostname"])
        and "boot_id" in payload
        and (boot_id is None or isinstance(boot_id, str) and bool(boot_id))
        and isinstance(payload.get("token"), str)
        and bool(payload["token"])
        and isinstance(payload.get("purpose"), str)
        and bool(payload["purpose"])
        and isinstance(payload.get("created_at_utc"), str)
        and bool(payload["created_at_utc"])
    )


def _unlink_if_owned(path: Path, token: str) -> bool:
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False
    if not isinstance(current, dict) or current.get("token") != token:
        return False
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def active_lock_owner(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        age_seconds = _lock_age_seconds(path)
        if age_seconds is None:
            return None
        status = (
            "initializing"
            if age_seconds < MALFORMED_INITIALIZING_SECONDS
            else "malformed_requires_operator_cleanup"
        )
        return _owner_details(path, None, status)

    if not isinstance(payload, dict):
        age_seconds = _lock_age_seconds(path)
        status = (
            "initializing"
            if age_seconds is not None and age_seconds < MALFORMED_INITIALIZING_SECONDS
            else "malformed_requires_operator_cleanup"
        )
        return _owner_details(path, None, status)

    if "pid" in payload and "schema" not in payload and "version" not in payload:
        return _owner_details(path, payload, "legacy_ambiguous")

    if not _is_valid_current_payload(payload):
        age_seconds = _lock_age_seconds(path)
        status = (
            "initializing"
            if age_seconds is not None and age_seconds < MALFORMED_INITIALIZING_SECONDS
            else "malformed_requires_operator_cleanup"
        )
        return _owner_details(path, payload, status)

    current_host = _current_host_identity()
    lock_hostname = str(payload["hostname"])
    lock_boot_id = payload["boot_id"]
    if lock_hostname != current_host.hostname:
        return _owner_details(path, payload, "foreign")
    if current_host.boot_id is None or lock_boot_id is None:
        return _owner_details(path, payload, "ambiguous_host_identity")
    if lock_boot_id != current_host.boot_id:
        return _owner_details(path, payload, "foreign")

    pid = int(payload["pid"])
    if _pid_is_active(pid):
        return _owner_details(path, payload, "active_local")
    if _unlink_if_owned(path, str(payload["token"])):
        return None
    return active_lock_owner(path)


@contextmanager
def exclusive_process_lock(path: Path, purpose: str) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    host_identity = _current_host_identity()
    token = uuid.uuid4().hex
    payload = {
        "schema": LOCK_SCHEMA,
        "version": LOCK_VERSION,
        "pid": os.getpid(),
        "hostname": host_identity.hostname,
        "boot_id": host_identity.boot_id,
        "purpose": purpose,
        "token": token,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    encoded = json.dumps(payload, sort_keys=True).encode()
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            owner = active_lock_owner(path)
            if owner is None:
                continue
            details = describe_lock_owner(path, owner)
            raise RuntimeError(f"{purpose} is already active: {details}") from None
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            _unlink_if_owned(path, token)
            raise
        break
    try:
        yield
    finally:
        _unlink_if_owned(path, token)
