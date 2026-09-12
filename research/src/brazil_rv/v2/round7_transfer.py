"""Transfer only changed store files; reconstruct and verify the immutable root."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path

from .artifacts import sha256_file, write_json_atomic
from .data_roots import resolve_external_root
from .round7_data import PROJECT


def pack(output):
    accepted = json.loads(
        (PROJECT / "docs/v2_round7_inputs.json").read_text(encoding="utf-8")
    )
    base = json.loads(
        (PROJECT / "docs/v2_round6_inputs.json").read_text(encoding="utf-8")
    )
    source, _ = resolve_external_root(accepted["store"]["root"])
    previous, _ = resolve_external_root(base["store"]["root"])
    if (
        sha256_file(source / "manifest.json") != accepted["store"]["manifest_sha256"]
        or sha256_file(previous / "manifest.json") != base["store"]["manifest_sha256"]
    ):
        raise ValueError("transfer inputs changed after acceptance")
    files, reused_bytes, sent_bytes = {}, 0, 0
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(
        output / "changed_files.tar.gz", "w:gz", compresslevel=3
    ) as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source).as_posix()
            digest = sha256_file(path)
            old = previous / relative
            reusable = old.is_file() and sha256_file(old) == digest
            size = path.stat().st_size
            files[relative] = {"sha256": digest, "bytes": size, "from_base": reusable}
            if reusable:
                reused_bytes += size
            else:
                archive.add(path, arcname=relative, recursive=False)
                sent_bytes += size
    result = {
        "base_store": base["store"],
        "store": accepted["store"],
        "files": files,
        "archive_sha256": sha256_file(output / "changed_files.tar.gz"),
        "archive_bytes": (output / "changed_files.tar.gz").stat().st_size,
        "reused_bytes": reused_bytes,
        "changed_bytes": sent_bytes,
    }
    write_json_atomic(output / "transfer.json", result)
    return result


def materialize(package, output):
    record = json.loads((package / "transfer.json").read_text(encoding="utf-8"))
    archive_path = package / "changed_files.tar.gz"
    if sha256_file(archive_path) != record["archive_sha256"]:
        raise ValueError("store delta archive changed")
    previous, _ = resolve_external_root(record["base_store"]["root"])
    if (
        sha256_file(previous / "manifest.json")
        != record["base_store"]["manifest_sha256"]
    ):
        raise ValueError("remote reusable store differs")
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(output, filter="data")
    for relative, entry in record["files"].items():
        destination = output / relative
        if entry["from_base"]:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(previous / relative, destination)
        if (
            destination.stat().st_size != entry["bytes"]
            or sha256_file(destination) != entry["sha256"]
        ):
            raise ValueError(f"reconstructed store file differs: {relative}")
    if sha256_file(output / "manifest.json") != record["store"]["manifest_sha256"]:
        raise ValueError("reconstructed store manifest differs")
    return {
        "status": "all_files_verified",
        "root": str(output),
        "files": len(record["files"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("pack", "materialize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package", type=Path)
    args = parser.parse_args()
    result = (
        pack(args.output)
        if args.action == "pack"
        else materialize(args.package, args.output)
    )
    print(json.dumps({k: v for k, v in result.items() if k != "files"}))


if __name__ == "__main__":
    main()
