"""Recover new scaling evidence and reconstruct changed store bytes from deltas."""

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
    acceptance = bound_json(run["scaling_data_inputs"])
    contract = bound_json(acceptance["contract"])
    root, parent = Path(acceptance["store"]["root"]), Path(acceptance["parent"]["root"])
    current = json.loads((root / "manifest.json").read_text())
    original = json.loads((parent / "manifest.json").read_text())
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    archive = Path(run["root"]) / f"scaling_inputs_{commit[:7]}.zip"
    assert not archive.exists()
    evidence_roots = {
        "metadata": Path(run["scaling_data_plan"]["path"]).parent,
        "derived": Path(bound_json(run["scaling_data_workspace"])["root"]),
    }
    members, aliases, unique, locations = {}, {}, {}, {}
    recipes = {"arrays": {}, "tables": {}, "indices": {}}
    dependencies = {
        key: run[key]
        for key in ("stage_c_matched_store_recovery", "scaling_expanded_recovery")
    }
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:

        def add(name, data):
            digest = hashlib.sha256(data).hexdigest()
            assert name not in members
            members[name] = dict(sha256=digest, bytes=len(data))
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
            "docs/v2_economic_data_scaling_progress.md",
            "docs/v2_economic_data_scaling_run.json",
            "docs/v2_SCALING_DATA_REPAIRS.md",
            "docs/v2_scaling_data_inputs.json",
            "docs/v2_scaling_store_contract.json",
        ]
        for name in sorted(set(project_paths)):
            file("project/" + name, PROJECT / name)
        add(
            "implementation.patch",
            subprocess.check_output(["git", "diff", "dc519f2", "HEAD"], cwd=PROJECT),
        )
        for label, base in evidence_roots.items():
            for path in sorted(base.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    assert not path.is_symlink() and path.resolve().is_relative_to(
                        base.resolve()
                    )
                    file(label + "/" + path.relative_to(base).as_posix(), path)
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
        # Read each sparse layer once; full accepted arrays never enter the ZIP.
        layers_by_key = {}
        for rec in contract["layers"]:
            with np.load(rec["path"]) as delta:
                for label in delta.files:
                    if label.endswith("__indices") and len(delta[label]):
                        key = label.split("__")[-2]
                        layers_by_key.setdefault(key, []).append(
                            dict(
                                member=locations[str(Path(rec["path"]).resolve())],
                                indices=label,
                                values=label.replace("__indices", "__values"),
                            )
                        )
        for key, rec in current["arrays"].items():
            layers = layers_by_key.get(key, [])
            recipes["arrays"][key] = dict(
                parent=original["arrays"][key], layers=layers, expected=rec
            )
            if not layers:
                assert original["arrays"][key]["sha256"] == rec["sha256"]
        add(
            "recipes.json",
            json.dumps(
                dict(parent=acceptance["parent"], recipes=recipes), indent=2
            ).encode(),
        )
        add("dependencies.json", json.dumps(dependencies, indent=2).encode())
        z.writestr("aliases.json", json.dumps(aliases, indent=2))
        z.writestr("members.json", json.dumps(members, indent=2))
    archive_seconds = perf_counter() - tick
    print(
        json.dumps(
            dict(archive_bytes=archive.stat().st_size, logical_members=len(members))
        ),
        flush=True,
    )
    scratch = (
        evidence_roots["derived"].parent / f"scaling_recovery_scratch_{commit[:7]}"
    )
    scratch.mkdir(exist_ok=False)
    work = scratch / "one_file.bin"
    rebuilt, kept, cells = [], [], 0
    with zipfile.ZipFile(archive) as z:

        def restored(name):
            data = z.read(aliases.get(name, name))
            rec = members[name]
            assert (
                len(data) == rec["bytes"]
                and hashlib.sha256(data).hexdigest() == rec["sha256"]
            )
            return data

        for name in members:
            work.write_bytes(restored(name))
            assert sha256_file(work) == members[name]["sha256"]
            work.unlink()
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
        status="verified_scaling_evidence_and_complete_store_recovery",
        implementation_commit=commit,
        acceptance=run["scaling_data_inputs"],
        archive={**binding(archive), "bytes": archive.stat().st_size},
        logical_members=len(members),
        unique_members=len(unique),
        aliases=len(aliases),
        arrays_rebuilt=rebuilt,
        arrays_inherited=kept,
        reconstructed_array_cells=cells,
        tables=len(recipes["tables"]),
        indices=len(recipes["indices"]),
        dependencies=dependencies,
        archive_seconds=archive_seconds,
        recovery_seconds=perf_counter() - tick - archive_seconds,
        scope="Every logical member restored/hash-checked; all changed store arrays reconstructed byte-exactly from immutable parent plus restored sparse layers. Unchanged parent bytes reuse completed checks. No immutable sources, accepted full arrays or old fits copied. Recovery bytes are not a repeated numerical/history proof.",
    )
    destination = PROJECT / "docs/v2_scaling_input_recovery.json"
    write_json_atomic(destination, report)
    run = json.loads(pointer.read_text())
    run["scaling_input_recovery"] = binding(destination)
    write_json_atomic(pointer, run)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
