"""Recover incremental matched-data evidence and every changed store array."""

import hashlib
import io
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
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    acceptance = bound_json(run["matched_data_inputs"])
    assembly = bound_json(run["stage_c_event_store_assembly"])
    contract = bound_json(assembly["contract"])
    root, parent = Path(acceptance["store"]["root"]), Path(contract["parent"]["root"])
    current = json.loads((root / "manifest.json").read_text())
    original = json.loads((parent / "manifest.json").read_text())
    previous = bound_json(run["stage_c_account_recovery"])
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    archive = Path(run["root"]) / f"matched_store_{commit[:7]}.zip"
    assert not archive.exists()
    base = Path(run["stage_c_root"])
    members, aliases, unique, inherited, locations = {}, {}, {}, {}, {}
    recipes = {"arrays": {}, "tables": {}, "indices": {}}
    with zipfile.ZipFile(previous["archive"]["path"]) as prior:
        prior_members = json.loads(prior.read("members.json"))
        prior_aliases = json.loads(prior.read("aliases.json"))
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:

            def add(name, data):
                digest = hashlib.sha256(data).hexdigest()
                record = dict(sha256=digest, bytes=len(data))
                if prior_members.get(name) == record:
                    inherited[name] = record
                    return
                assert name not in members
                members[name] = record
                if digest in unique:
                    aliases[name] = unique[digest]
                else:
                    unique[digest] = name
                    z.writestr(name, data)

            def file(name, path):
                locations[str(path.resolve())] = name
                add(name, path.read_bytes())

            project_paths = subprocess.check_output(
                [
                    "git",
                    "ls-files",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                    "research",
                    "ops",
                ],
                cwd=PROJECT,
                text=True,
            ).splitlines()
            project_paths += [
                "PROJECT_CONTEXT.md",
                "docs/v2_MATCHED_ECONOMIC_REPLAYS.md",
                "docs/v2_economic_data_scaling_progress.md",
                "docs/v2_economic_data_scaling_run.json",
                "docs/v2_matched_store_contract.json",
                "docs/v2_matched_data_inputs.json",
                "docs/v2_matched_economics_recovery.json",
                "research/preregistrations/v2_economic_data_scaling.md",
            ]
            for name in sorted(set(project_paths)):
                file("project/" + name, PROJECT / name)
            add(
                "implementation.patch",
                subprocess.check_output(
                    ["git", "diff", "7fa42fc", "HEAD"], cwd=PROJECT
                ),
            )
            for path in sorted((base / "event_data").rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    assert not path.is_symlink() and path.resolve().is_relative_to(
                        base.resolve()
                    )
                    file("evidence/" + path.relative_to(base).as_posix(), path)
            for name in ("manifest.json", "manifest.sha256"):
                file("store/" + name, root / name)
            for group in ("tables", "indices"):
                for key, rec in current[group].items():
                    old = original[group].get(key)
                    if old and old["sha256"] == rec["sha256"]:
                        recipes[group][key] = dict(parent=old, expected=rec)
                    else:
                        path = root / (key if group == "indices" else rec["path"])
                        name = "store/" + path.relative_to(root).as_posix()
                        file(name, path)
                        recipes[group][key] = dict(member=name, expected=rec)
            for key, rec in current["arrays"].items():
                layers = []
                for record in contract["layers"]:
                    with np.load(record["path"]) as delta:
                        for label in delta.files:
                            if (
                                label.endswith("__indices")
                                and label.split("__")[-2] == key
                                and len(delta[label])
                            ):
                                name = locations[str(Path(record["path"]).resolve())]
                                layers.append(
                                    dict(
                                        member=name,
                                        indices=label,
                                        values=label.replace("__indices", "__values"),
                                    )
                                )
                recipes["arrays"][key] = dict(
                    parent=original["arrays"][key], layers=layers, expected=rec
                )
                if not layers:
                    assert original["arrays"][key]["sha256"] == rec["sha256"]
            add(
                "recipes.json",
                json.dumps(
                    dict(parent=contract["parent"], recipes=recipes), indent=2
                ).encode(),
            )
            add(
                "dependencies.json",
                json.dumps(
                    dict(
                        account=run["stage_c_account_recovery"],
                        parent=run["composed_store_recovery"],
                    ),
                    indent=2,
                ).encode(),
            )
            z.writestr("inherited_members.json", json.dumps(inherited, indent=2))
            z.writestr("aliases.json", json.dumps(aliases, indent=2))
            z.writestr("members.json", json.dumps(members, indent=2))
        archive_seconds = perf_counter() - tick
        print(
            json.dumps(
                dict(
                    archive_bytes=archive.stat().st_size,
                    new_members=len(members),
                    inherited_members=len(inherited),
                )
            ),
            flush=True,
        )
        scratch = base / f"matched_store_recovery_scratch_{commit[:7]}"
        scratch.mkdir(exist_ok=False)
        work = scratch / "one_file.bin"
        rebuilt, kept, cells = [], [], 0
        with zipfile.ZipFile(archive) as z:

            def restored(name):
                if name in inherited:
                    data = prior.read(prior_aliases.get(name, name))
                    record = inherited[name]
                else:
                    data = z.read(aliases.get(name, name))
                    record = members[name]
                assert (
                    len(data) == record["bytes"]
                    and hashlib.sha256(data).hexdigest() == record["sha256"]
                )
                return data

            for name in members:
                work.write_bytes(restored(name))
                assert sha256_file(work) == members[name]["sha256"]
                work.unlink()
            # Only layers actually required for the new store are extracted from
            # the previously verified archive. No old source/book census repeats.
            saved = json.loads(restored("recipes.json"))["recipes"]
            for key, recipe in saved["arrays"].items():
                if not recipe["layers"]:
                    kept.append(key)
                    continue
                shutil.copyfile(parent / recipe["parent"]["path"], work)
                array = np.load(work, mmap_mode="r+")
                for layer in recipe["layers"]:
                    with np.load(io.BytesIO(restored(layer["member"]))) as delta:
                        array[tuple(delta[layer["indices"]].T)] = delta[layer["values"]]
                cells += array.size
                close_memmap(array)
                assert sha256_file(work) == recipe["expected"]["sha256"], key
                rebuilt.append(key)
                work.unlink()
        scratch.rmdir()
    report = dict(
        status="verified_incremental_evidence_and_complete_store_recovery",
        implementation_commit=commit,
        acceptance=run["matched_data_inputs"],
        archive={**binding(archive), "bytes": archive.stat().st_size},
        logical_members=len(members),
        unique_members=len(unique),
        aliases=len(aliases),
        members_reused_from_prior_verified_archive=len(inherited),
        arrays_rebuilt=rebuilt,
        arrays_inherited=kept,
        reconstructed_array_cells=cells,
        tables=len(recipes["tables"]),
        indices=len(recipes["indices"]),
        dependencies=dict(
            account=run["stage_c_account_recovery"],
            parent=run["composed_store_recovery"],
        ),
        archive_seconds=archive_seconds,
        recovery_seconds=perf_counter() - tick - archive_seconds,
        scope="Restored/hash-checked every new logical member and reconstructed all changed complete-store NPY bytes one file at a time. Prior unchanged evidence is referenced by exact archive member/hash. Original sources, accepted arrays and old fits are not copied into the archive. Recovery scratch only is removed; original evidence remains. No new consumer/history/model proof is claimed.",
    )
    output = Path(run["root"]) / "matched_store_recovery.json"
    write_json_atomic(output, report)
    write_json_atomic(PROJECT / "docs/v2_matched_store_recovery.json", report)
    run = json.loads(pointer.read_text())
    run["stage_c_matched_store_recovery"] = binding(output)
    write_json_atomic(pointer, run)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
