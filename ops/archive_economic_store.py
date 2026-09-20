"""Recover the complete new store from sparse changes and sealed dependencies."""

import base64
import io
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from time import perf_counter

import numpy as np

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.store import close_memmap
from assemble_economic_store import inputs

PROJECT = Path(__file__).resolve().parents[1]


def main():
    start = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    acceptance = bound_json(run["derived_data_inputs"])
    contract = bound_json(acceptance["contract"])
    manifests, replacements, _, _ = inputs(contract)
    root = Path(acceptance["store"]["root"])
    parent = Path(acceptance["parent"]["root"])
    original = json.loads((parent / "manifest.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    archive = Path(run["root"]) / f"economic_store_{commit[:7]}.zip"
    if archive.exists():
        raise FileExistsError(archive)
    restore = Path(run["root"]) / f"economic_store_recovery_{commit[:7]}"
    restore.mkdir(exist_ok=False)
    members, recipes = {}, {"arrays": {}, "tables": {}, "indices": {}}
    known = {}

    def collect(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                known.setdefault(value["sha256"], value["path"])
            for v in value.values():
                collect(v)
        elif isinstance(value, list):
            for v in value:
                collect(v)

    collect(manifests)
    for group in ("arrays", "tables"):
        for rec in original[group].values():
            known.setdefault(rec["sha256"], str(parent / rec["path"]))
    for name, rec in original["indices"].items():
        known.setdefault(rec["sha256"], str(parent / name))

    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as z:

        def add_bytes(name, data):
            import hashlib

            if name in members:
                raise ValueError("Duplicate archive member: " + name)
            z.writestr(name, data)
            members[name] = {
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }

        def add_file(name, path):
            add_bytes(name, Path(path).read_bytes())

        research = subprocess.check_output(
            ["git", "ls-files", "research"], cwd=PROJECT, text=True
        ).splitlines()
        local = [
            "PROJECT_CONTEXT.md",
            "docs/v2_DERIVED_STORE_ACCEPTANCE.md",
            "docs/v2_derived_store_contract.json",
            "docs/v2_economic_data_inputs.json",
            "docs/v2_economic_data_scaling_run.json",
            "docs/v2_economic_data_scaling_progress.md",
        ]
        local += [
            "ops/" + n
            for n in (
                "assemble_economic_store.py",
                "resolve_corporate_claim_rows.py",
                "qualify_economic_store.py",
                "audit_economic_store_boundaries.py",
                "archive_economic_store.py",
            )
        ]
        for path in sorted(set(research + local)):
            add_file("project/" + path, PROJECT / path)
        patch = subprocess.check_output(["git", "diff", "HEAD^", "HEAD"], cwd=PROJECT)
        add_bytes("implementation.patch", patch)
        for key in (
            "corporate_claim_rows",
            "derived_store_assembly",
            "derived_store_boundaries",
        ):
            directory = Path(run[key]["path"]).parent
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    add_file(
                        "evidence/"
                        + key
                        + "/"
                        + path.relative_to(directory).as_posix(),
                        path,
                    )
        add_file("store/manifest.json", root / "manifest.json")
        add_file("store/manifest.sha256", root / "manifest.sha256")
        for key, rec in manifest["arrays"].items():
            if rec["sha256"] in known:
                recipes["arrays"][key] = {
                    "source": known[rec["sha256"]],
                    "expected": rec,
                }
                continue
            source = (
                Path(replacements[key]["path"])
                if key in replacements
                else parent / original["arrays"][key]["path"]
            )
            before = np.load(source, mmap_mode="r")
            after = np.load(root / rec["path"], mmap_mode="r")
            assert (
                before.offset == after.offset
                and before.flags.c_contiguous
                and after.flags.c_contiguous
            )
            indices, values = [], []
            stride = int(np.prod(after.shape[1:]))
            for t in range(0, len(after), 64):
                x, y = before[t : t + 64], after[t : t + 64]
                # Preserve NaN payloads and signed zero as well as numerical equality.
                dtype = np.dtype(f"u{after.dtype.itemsize}")
                ix = np.flatnonzero(x.view(dtype).ravel() != y.view(dtype).ravel())
                indices.append((ix + t * stride).astype(np.uint32))
                values.append(y.ravel()[ix].copy())
            indices, values = np.concatenate(indices), np.concatenate(values)
            assert after.size < 2**32
            b = io.BytesIO()
            np.savez_compressed(b, indices=indices, values=values)
            name = "store/deltas/" + key + ".npz"
            add_bytes(name, b.getvalue())
            with (root / rec["path"]).open("rb") as handle:
                header = base64.b64encode(handle.read(after.offset)).decode("ascii")
            source_hash = (
                replacements[key]["sha256"]
                if key in replacements
                else original["arrays"][key]["sha256"]
            )
            recipes["arrays"][key] = {
                "source": str(source),
                "source_sha256": source_hash,
                "patch": name,
                "patch_cells": len(indices),
                "header": header,
                "expected": rec,
            }
            close_memmap(before)
            close_memmap(after)
        for group in ("tables", "indices"):
            for key, rec in manifest[group].items():
                path = rec.get("path", key)
                if rec["sha256"] in known:
                    recipes[group][key] = {
                        "source": known[rec["sha256"]],
                        "expected": rec,
                    }
                else:
                    member = "store/files/" + path
                    add_file(member, root / path)
                    recipes[group][key] = {"member": member, "expected": rec}
        dependencies = {
            k: run[k]
            for k in (
                "rename_history_recovery",
                "rename_dependency_recovery",
                "fca_issuer_peer_recovery",
                "lending_feature_recovery",
                "rename_m1_recovery",
                "m1_scalar_recovery",
                "to_close_clock_recovery",
                "remaining_auxiliary_recovery",
                "bvbg_target_recovery",
            )
        }
        for rec in dependencies.values():
            assert bound_json(rec)
        recipe = {
            "implementation_commit": commit,
            "store": acceptance["store"],
            "parent": acceptance["parent"],
            "dependencies": dependencies,
            "recipes": recipes,
            "recovery_contract": "Restore dependencies if unavailable, then use bound source bytes and sparse amendments; no immutable raw archives or old fits are duplicated.",
        }
        add_bytes(
            "recovery_recipe.json", (json.dumps(recipe, indent=2) + "\n").encode()
        )
        z.writestr("members.json", json.dumps(members, indent=2))

    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            target = (restore / info.filename).resolve()
            if not target.is_relative_to(restore.resolve()):
                raise ValueError("Unsafe archive member")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(info))
    for name, rec in members.items():
        assert sha256_file(restore / name) == rec["sha256"], name
    recovered = json.loads((restore / "recovery_recipe.json").read_text())
    scratch = restore / "restored_array.npy"
    cells = 0
    for group, records in recovered["recipes"].items():
        for key, rec in records.items():
            if "member" in rec:
                shutil.copyfile(restore / rec["member"], scratch)
            else:
                shutil.copyfile(rec["source"], scratch)
            if "patch" in rec:
                with scratch.open("r+b") as handle:
                    handle.write(base64.b64decode(rec["header"]))
                a = np.load(scratch, mmap_mode="r+")
                with np.load(restore / rec["patch"]) as patch:
                    a.reshape(-1)[patch["indices"]] = patch["values"]
                close_memmap(a)
            assert sha256_file(scratch) == rec["expected"]["sha256"], (group, key)
            if group == "arrays":
                cells += int(np.prod(rec["expected"]["shape"]))
            scratch.unlink()
    result = {
        "status": "verified",
        "implementation_commit": commit,
        "archive": {**binding(archive), "bytes": archive.stat().st_size},
        "members": len(members),
        "member_index": binding(restore / "members.json"),
        "store": acceptance["store"],
        "reconstructed_array_cells": cells,
        "reconstructed_arrays": len(recipes["arrays"]),
        "reconstructed_tables": len(recipes["tables"]),
        "recipes": binding(restore / "recovery_recipe.json"),
        "dependencies": dependencies,
        "immutable_raw_or_old_fits_duplicated": False,
        "seconds": perf_counter() - start,
    }
    record = Path(run["root"]) / "economic_store_recovery.json"
    write_json_atomic(record, result)
    run["economic_store_recovery"] = binding(record)
    write_json_atomic(pointer, run)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
