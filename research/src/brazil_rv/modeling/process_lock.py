from __future__ import annotations

import csv
import json
import os
import subprocess
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .contract import RUN_OUTPUT_BASE

PRODUCTION_TRAINING_LOCK = RUN_OUTPUT_BASE / "_ops" / "production_training.lock"


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


def active_lock_owner(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        try:
            age_seconds = time.time() - path.stat().st_mtime
        except FileNotFoundError:
            return None
        if age_seconds < 60.0:
            return {"status": "initializing"}
        path.unlink(missing_ok=True)
        return None
    try:
        pid = int(payload["pid"])
    except (KeyError, TypeError, ValueError):
        path.unlink(missing_ok=True)
        return None
    if _pid_is_active(pid):
        return payload
    path.unlink(missing_ok=True)
    return None


@contextmanager
def exclusive_process_lock(path: Path, purpose: str) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    payload = {
        "pid": os.getpid(),
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
            raise RuntimeError(f"{purpose} is already active: {owner}") from None
        try:
            os.write(descriptor, encoded)
        finally:
            os.close(descriptor)
        break
    try:
        yield
    finally:
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            current = None
        if isinstance(current, dict) and current.get("token") == token:
            path.unlink(missing_ok=True)
