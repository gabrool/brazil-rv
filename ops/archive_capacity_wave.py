"""Archive a completed capacity wave without copying its saved controls."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
import zipfile

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main(wave):
    started = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    prefix = f"stage_d_{wave}"
    root = Path(run[prefix + "_plan"]["path"]).parent
    fits = bound_json(binding(root / "refits.json"))
    assert fits["status"] == "complete" and len(fits["completed"]) == fits["planned"]
    result = bound_json(run[prefix + "_results"])
    assert bound_json(result["qualification"])["status"] == "complete"
    for source in result["sensitivities"]["sources"]:
        assert bound_json(source["qualification"])["status"] == "complete"
    previous_key = {
        "width": "stage_c_refit_recovery",
        "depth": "stage_d_width_recovery",
        "lstm": "stage_d_depth_recovery",
    }[wave]
    previous = bound_json(run[previous_key])
    with zipfile.ZipFile(previous["archive"]["path"]) as z:
        known = json.loads(z.read("inherited_members.json"))
        known.update(json.loads(z.read("members.json")))
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    archive = Path(run["root"]) / f"capacity_{wave}_{commit[:7]}.zip"
    assert not archive.exists()
    members, inherited, aliases, unique = {}, {}, {}, {}
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as z:

        def add(name, path):
            record = dict(sha256=sha256_file(path), bytes=path.stat().st_size)
            if known.get(name) == record:
                inherited[name] = record
                return
            assert name not in members
            members[name] = record
            if record["sha256"] in unique:
                aliases[name] = unique[record["sha256"]]
            else:
                unique[record["sha256"]] = name
                z.write(path, name)

        names = subprocess.check_output(
            ["git", "ls-files", "research", "ops", "docs", "PROJECT_CONTEXT.md"],
            cwd=PROJECT,
            text=True,
        ).splitlines()
        for name in names:
            add("project/" + name, PROJECT / name)
        evidence = [(root, "capacity_" + wave)]
        if wave == "width" and "stage_c_account_composition" in run:
            evidence.append(
                (
                    Path(run["stage_c_account_composition"]["path"]).parent,
                    "matched_account_composition",
                )
            )
        if wave == "width" and "stage_d_lstm_engineering" in run:
            evidence.append(
                (
                    Path(run["stage_d_lstm_engineering"]["path"]).parent,
                    "capacity_lstm_engineering",
                )
            )
        for folder, label in evidence:
            for directory, children, filenames in os.walk(folder):
                children[:] = [name for name in children if name != "__pycache__"]
                for name in sorted(filenames):
                    path = Path(directory) / name
                    assert path.resolve().is_relative_to(folder.resolve())
                    add(
                        "evidence/" + label + "/" + path.relative_to(folder).as_posix(),
                        path,
                    )
        for name, value in (
            ("members", members),
            ("inherited_members", inherited),
            ("aliases", aliases),
        ):
            z.writestr(name + ".json", json.dumps(value, indent=2))
        z.writestr(
            "dependencies.json", json.dumps({previous_key: run[previous_key]}, indent=2)
        )
    archive_seconds = perf_counter() - started
    scratch = Path(run["stage_c_root"]) / f"capacity_{wave}_recovery_{commit[:7]}"
    scratch.mkdir(exist_ok=False)
    target = scratch / "one_file.bin"
    assert shutil.disk_usage(scratch).free > max(x["bytes"] for x in members.values())
    with zipfile.ZipFile(archive) as z:
        for name, record in members.items():
            with (
                z.open(aliases.get(name, name)) as source,
                target.open("wb") as destination,
            ):
                shutil.copyfileobj(source, destination, 1024 * 1024)
            assert (
                target.stat().st_size == record["bytes"]
                and sha256_file(target) == record["sha256"]
            ), name
            target.unlink()
    scratch.rmdir()
    report = dict(
        status="verified_capacity_wave_artifact_recovery",
        wave=wave,
        implementation_commit=commit,
        archive={**binding(archive), "bytes": archive.stat().st_size},
        logical_members=len(members),
        unique_members=len(unique),
        aliases=len(aliases),
        inherited_members=len(inherited),
        fits=len(fits["completed"]),
        dependencies={previous_key: run[previous_key]},
        archive_seconds=archive_seconds,
        recovery_seconds=perf_counter() - started - archive_seconds,
        scope="All new wave checkpoints, predictions, primary/scenario books, frozen plans, executed and failed recipes, runtime/source and documentation restore/hash-check. Matched controls and immutable inputs/accepted stores/old fits are inherited, never copied. One-file scratch is removed only after its exact hash check. This verifies bytes, not a repeated source, history, model or economic experiment.",
    )
    output = Path(run["root"]) / f"capacity_{wave}_recovery.json"
    write_json_atomic(output, report)
    write_json_atomic(PROJECT / "docs" / f"v2_capacity_{wave}_recovery.json", report)
    run[prefix + "_recovery"] = binding(output)
    write_json_atomic(pointer, run)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave", choices=("width", "depth", "lstm"), required=True)
    main(parser.parse_args().wave)
