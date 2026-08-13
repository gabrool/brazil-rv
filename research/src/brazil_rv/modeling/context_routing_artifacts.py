from __future__ import annotations

import hashlib
import os
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member(root: Path, member: str) -> tuple[str, Path]:
    if not member or "\\" in member:
        raise ValueError(f"Unsafe archive member: {member!r}")
    relative = PurePosixPath(member)
    if relative.is_absolute() or any(
        part in ("", ".", "..") for part in relative.parts
    ):
        raise ValueError(f"Unsafe archive member: {member!r}")
    normalized = relative.as_posix()
    path = root.joinpath(*relative.parts)
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"Archive member escapes root: {member}")
    current = resolved_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Archive member traverses a symlink: {member}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Archive member is missing or not a file: {member}")
    return normalized, resolved


def _member_snapshots(
    root: Path, members: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for member in members:
        normalized, path = _safe_member(root, member)
        identity = normalized.casefold()
        if identity in seen:
            raise ValueError(f"Duplicate archive member: {normalized}")
        seen.add(identity)
        details = path.stat()
        snapshots[normalized] = {
            "path": path,
            "size": details.st_size,
            "mtime_ns": details.st_mtime_ns,
            "sha256": sha256_file(path),
        }
    return snapshots


def validate_archive(
    archive_path: Path, expected_sha256: dict[str, str]
) -> dict[str, str]:
    if not archive_path.is_file() or archive_path.is_symlink():
        raise FileNotFoundError(f"Validated archive does not exist: {archive_path}")
    with zipfile.ZipFile(archive_path, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len({name.casefold() for name in names}):
            raise ValueError("Archive contains duplicate members")
        if set(names) != set(expected_sha256):
            raise ValueError("Archive member list does not match the explicit contract")
        observed: dict[str, str] = {}
        for info in infos:
            relative = PurePosixPath(info.filename)
            if (
                relative.is_absolute()
                or "\\" in info.filename
                or any(part in ("", ".", "..") for part in relative.parts)
            ):
                raise ValueError(f"Archive contains unsafe member: {info.filename}")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Archive contains a symlink: {info.filename}")
            digest = hashlib.sha256()
            with archive.open(info, "r") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            observed[info.filename] = digest.hexdigest()
        if observed != expected_sha256:
            raise ValueError("Archive content hashes do not match source artifacts")
        return observed


def create_validated_archive(
    root: Path, members: tuple[str, ...], archive_path: Path
) -> dict[str, str]:
    root = root.resolve(strict=True)
    snapshots = _member_snapshots(root, members)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_name(f"{archive_path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name, snapshot in snapshots.items():
                archive.write(snapshot["path"], arcname=name)
        for name, snapshot in snapshots.items():
            path = snapshot["path"]
            details = path.stat()
            if (
                details.st_size != snapshot["size"]
                or details.st_mtime_ns != snapshot["mtime_ns"]
                or sha256_file(path) != snapshot["sha256"]
            ):
                raise RuntimeError(f"Archive input mutated during creation: {name}")
        expected = {name: str(row["sha256"]) for name, row in snapshots.items()}
        validate_archive(temporary, expected)
        os.replace(temporary, archive_path)
        validate_archive(archive_path, expected)
        return expected
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_archive_sha256(archive_path: Path, sidecar_path: Path) -> str:
    digest = sha256_file(archive_path)
    temporary = sidecar_path.with_name(f"{sidecar_path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(f"{digest}  {archive_path.name}\n", encoding="utf-8")
        os.replace(temporary, sidecar_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return digest


def validate_archive_sha256(archive_path: Path, sidecar_path: Path) -> str:
    if not sidecar_path.is_file() or sidecar_path.is_symlink():
        raise FileNotFoundError(f"Archive SHA-256 sidecar is missing: {sidecar_path}")
    parts = sidecar_path.read_text(encoding="utf-8").strip().split()
    if len(parts) != 2 or parts[1] != archive_path.name:
        raise ValueError("Archive SHA-256 sidecar is malformed")
    digest = sha256_file(archive_path)
    if parts[0] != digest:
        raise ValueError("Archive SHA-256 sidecar does not match the archive")
    return digest


def publish_output_pointer(
    pointer_path: Path,
    output_dir: Path,
    archive_path: Path,
    sidecar_path: Path,
    expected_members: dict[str, str],
) -> None:
    output_dir = output_dir.resolve(strict=True)
    validate_archive(archive_path, expected_members)
    validate_archive_sha256(archive_path, sidecar_path)
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = pointer_path.with_name(f"{pointer_path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(f"{output_dir}\n", encoding="utf-8")
        os.replace(temporary, pointer_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
