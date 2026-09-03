from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")


def write_json_atomic(path: Path, payload: Any, *, write_sha256: bool = True) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes(payload)
    digest = hashlib.sha256(content).hexdigest()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if write_sha256:
        sha_path = path.with_suffix(path.suffix + ".sha256")
        sha_path.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return digest


def inventory(root: Path, *, exclude: set[str] | None = None) -> list[dict[str, object]]:
    excluded = exclude or set()
    rows: list[dict[str, object]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def verify_inventory(root: Path, rows: list[dict[str, object]]) -> None:
    expected = {str(row["path"]): row for row in rows}
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_paths != set(expected):
        missing = sorted(set(expected) - actual_paths)
        extra = sorted(actual_paths - set(expected))
        raise ValueError(f"Inventory mismatch: missing={missing}, extra={extra}")
    for relative, row in expected.items():
        path = root / relative
        if path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"Size mismatch for {relative}")
        if sha256_file(path) != str(row["sha256"]):
            raise ValueError(f"SHA-256 mismatch for {relative}")
