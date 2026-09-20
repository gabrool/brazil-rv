"""Archive new evidence and verify complete-store recovery without raw duplication."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
import zipfile

import numpy as np

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.store import close_memmap

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    bound_json(run["composed_data_inputs"])
    assembly = bound_json(run["composed_store_assembly"])
    contract = bound_json(assembly["contract"])
    root = Path(assembly["store"]["root"])
    parent = Path(assembly["parent_store"]["root"])
    m = json.loads((root / "manifest.json").read_text())
    old = json.loads((parent / "manifest.json").read_text())
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    base = Path(run["root"])
    archive = base / f"composed_store_{commit[:7]}.zip"
    restored = base / f"composed_store_recovery_{commit[:7]}"
    assert not archive.exists() and not restored.exists()
    restored.mkdir()
    dependencies = {
        k: run[k]
        for k in (
            "surviving_rename_market_recovery",
            "surviving_rename_dependency_recovery",
            "surviving_rename_recovery",
            "event_composition_recovery",
            "natura_store_recovery",
            "spot_invoice_recovery",
        )
    }
    for rec in dependencies.values():
        bound_json(rec)
    members, aliases, hashes, locations = {}, {}, {}, {}
    recipes = {"arrays": {}, "tables": {}, "indices": {}}
    patches = contract["layers"]
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as z:

        def add(name, data):
            assert name not in members
            digest = hashlib.sha256(data).hexdigest()
            members[name] = {"sha256": digest, "bytes": len(data)}
            if digest in hashes:
                aliases[name] = hashes[digest]
            else:
                hashes[digest] = name
                z.writestr(name, data)

        def file(name, path):
            path = Path(path)
            locations[str(path.resolve())] = name
            add(name, path.read_bytes())

        project_paths = subprocess.check_output(
            ["git", "ls-files", "research"], cwd=PROJECT, text=True
        ).splitlines()
        project_paths += [
            "PROJECT_CONTEXT.md",
            "docs/PROJECT_CONTEXT_HISTORY.md",
            "docs/v2_COMPOSED_DATA_STORE.md",
            "docs/v2_composed_store_contract.json",
            "docs/v2_composed_data_inputs.json",
            "docs/v2_economic_data_scaling_run.json",
            "docs/v2_economic_data_scaling_progress.md",
            "docs/v2_STAGE_A_CLOSEOUT.md",
            "docs/v2_SURVIVING_RENAME_PROPAGATION.md",
            "ops/compose_surviving_store.py",
            "ops/qualify_composed_store.py",
            "ops/archive_composed_store.py",
            "ops/propagate_corporate_targets.py",
            "ops/verify_corporate_replay.py",
            "ops/audit_corporate_target_inputs.py",
        ]
        for name in sorted(set(project_paths)):
            file("project/" + name, PROJECT / name)
        add(
            "implementation.patch",
            subprocess.check_output(["git", "diff", "HEAD^", "HEAD"], cwd=PROJECT),
        )
        for directory in ("surviving_store", "surviving_store_initial_failure"):
            folder = base / directory
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    file(
                        "evidence/"
                        + directory
                        + "/"
                        + path.relative_to(folder).as_posix(),
                        path,
                    )
        file("store/manifest.json", root / "manifest.json")
        file("store/manifest.sha256", root / "manifest.sha256")
        for group in ("tables", "indices"):
            for key, rec in m[group].items():
                path = root / (key if group == "indices" else rec["path"])
                previous = old[group].get(key)
                if previous and previous["sha256"] == rec["sha256"]:
                    recipes[group][key] = {
                        "dependency": str(
                            parent / (key if group == "indices" else previous["path"])
                        ),
                        "expected": rec,
                    }
                else:
                    name = "store/" + rec["path"]
                    file(name, path)
                    recipes[group][key] = {"member": name, "expected": rec}
        for key, rec in m["arrays"].items():
            layers = []
            for record in patches:
                with np.load(record["path"]) as delta:
                    for label in delta.files:
                        if (
                            label.endswith("__indices")
                            and label.split("__")[-2] == key
                            and len(delta[label])
                        ):
                            name = locations.get(str(Path(record["path"]).resolve()))
                            layers.append(
                                {
                                    "member": name,
                                    "dependency": None if name else record,
                                    "indices": label,
                                    "values": label.replace("__indices", "__values"),
                                }
                            )
            recipes["arrays"][key] = {
                "source": str(parent / old["arrays"][key]["path"]),
                "source_sha256": old["arrays"][key]["sha256"],
                "layers": layers,
                "expected": rec,
            }
            if not layers:
                assert old["arrays"][key]["sha256"] == rec["sha256"]
        add(
            "recipes.json",
            json.dumps(
                {
                    "parent": assembly["parent_store"],
                    "dependencies": dependencies,
                    "recipes": recipes,
                },
                indent=2,
            ).encode(),
        )
        z.writestr("aliases.json", json.dumps(aliases, indent=2))
        z.writestr("inventory.json", json.dumps(members, indent=2))
    # Restore/hash every new logical member, including aliases. Original source
    # and accepted baseline stores remain explicit previously verified dependencies.
    with zipfile.ZipFile(archive) as z:
        for name, record in members.items():
            data = z.read(aliases.get(name, name))
            assert (
                len(data) == record["bytes"]
                and hashlib.sha256(data).hexdigest() == record["sha256"]
            )
            target = restored / name
            assert target.resolve().is_relative_to(restored.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    saved = json.loads((restored / "recipes.json").read_text())["recipes"]
    rebuilt, inherited, cells = [], [], 0
    scratch = Path("C:/quant-data/b3/interim") / f"composed_recovery_{commit[:7]}"
    scratch.mkdir(exist_ok=False)
    work = scratch / "one_array.npy"
    for key, recipe in saved["arrays"].items():
        if not recipe["layers"]:
            assert recipe["source_sha256"] == recipe["expected"]["sha256"]
            inherited.append(key)
            continue
        shutil.copyfile(recipe["source"], work)
        a = np.load(work, mmap_mode="r+")
        for layer in recipe["layers"]:
            if layer["member"]:
                path = restored / layer["member"]
            else:
                path = Path(layer["dependency"]["path"])
                assert sha256_file(path) == layer["dependency"]["sha256"]
            with np.load(path) as delta:
                a[tuple(delta[layer["indices"]].T)] = delta[layer["values"]]
        cells += a.size
        close_memmap(a)
        assert sha256_file(work) == recipe["expected"]["sha256"]
        rebuilt.append(key)
        work.unlink()
    for group in ("tables", "indices"):
        for recipe in saved[group].values():
            if "member" in recipe:
                assert (
                    sha256_file(restored / recipe["member"])
                    == recipe["expected"]["sha256"]
                )
    scratch.rmdir()
    result = {
        "status": "verified_new_evidence_and_complete_store_recovery",
        "acceptance": run["composed_data_inputs"],
        "archive": {**binding(archive), "bytes": archive.stat().st_size},
        "restored_root": str(restored),
        "unique_members": len(members) - len(aliases),
        "logical_members": len(members),
        "deduplicated_aliases": len(aliases),
        "arrays_rebuilt": rebuilt,
        "arrays_inherited_from_verified_parent": inherited,
        "reconstructed_array_cells": cells,
        "all_arrays": len(saved["arrays"]),
        "tables": len(saved["tables"]),
        "indices": len(saved["indices"]),
        "dependencies": dependencies,
        "implementation_commit": commit,
        "limits": "Accepted parent and original source recovery dependencies are required; no raw archives, old fits or accepted-store arrays duplicated in the ZIP. New bounded audit views are reproducible from saved recipes/row outputs and sparse amendments; no model scoring.",
        "seconds": perf_counter() - started,
    }
    output = base / "composed_store_recovery.json"
    write_json_atomic(output, result)
    run["composed_store_recovery"] = binding(output)
    write_json_atomic(pointer, run)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
