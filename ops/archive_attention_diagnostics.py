"""Recover completed reversal diagnostics while broader fits continue separately."""

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


def main():
    started = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    assert bound_json(run["scaling_parent_patience_fits"])["status"] == "complete"
    bound_json(run["scaling_parent_patience_results"])
    root = Path(run["scaling_investigation"]["path"]).parent
    prior = bound_json(run["stage_d_depth_recovery"])
    with zipfile.ZipFile(prior["archive"]["path"]) as source:
        known = json.loads(source.read("inherited_members.json"))
        known.update(json.loads(source.read("members.json")))
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    archive = Path(run["root"]) / f"attention_diagnostics_{commit[:7]}.zip"
    assert not archive.exists()
    members, inherited, aliases, unique = {}, {}, {}, {}
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as output:

        def add(name, path):
            receipt = dict(sha256=sha256_file(path), bytes=path.stat().st_size)
            if known.get(name) == receipt:
                inherited[name] = receipt
                return
            assert name not in members
            members[name] = receipt
            if receipt["sha256"] in unique:
                aliases[name] = unique[receipt["sha256"]]
            else:
                unique[receipt["sha256"]] = name
                output.write(path, name)

        names = subprocess.check_output(
            ["git", "ls-files", "research", "ops", "docs", "PROJECT_CONTEXT.md"],
            cwd=PROJECT,
            text=True,
        ).splitlines()
        for name in names:
            add("project/" + name, PROJECT / name)
        folders = [
            (root, "attention_diagnostics"),
            (Path(run["stage_d_lstm_storage"]["path"]).parent, "capacity_lstm_storage"),
            (Path(run["stage_d_lstm_plan"]["path"]).parent, "capacity_lstm_plans"),
        ]
        for folder, label in folders:
            for directory, children, filenames in os.walk(folder):
                children[:] = [
                    n
                    for n in children
                    if n not in {"expanded_folds", "__pycache__"}
                    and not (Path(directory) / n).is_junction()
                ]
                for name in sorted(filenames):
                    path = Path(directory) / name
                    assert path.resolve().is_relative_to(folder.resolve())
                    add(
                        "evidence/" + label + "/" + path.relative_to(folder).as_posix(),
                        path,
                    )
        expanded = Path(run["scaling_expanded_fit_plan"]["path"]).parent
        for name in ("plan.json", "frozen_design.json", "evaluation/plan.json"):
            add("evidence/expanded_frozen/" + name, expanded / name)
        # Original training source is immutable in its clean worktree and already
        # archive-dependent; preserve the complete specific runtime without data.
        runtime = Path(
            bound_json(run["scaling_parent_patience_plan"])["runtime"]["training"][
                "path"
            ]
        ).parents[4]
        runtime_names = subprocess.check_output(
            ["git", "ls-files", "research"], cwd=runtime, text=True
        ).splitlines()
        for name in runtime_names:
            add("runtime/attention_diagnostics/" + name, runtime / name)
        for name, value in (
            ("members", members),
            ("inherited_members", inherited),
            ("aliases", aliases),
        ):
            output.writestr(name + ".json", json.dumps(value, indent=2))
        output.writestr(
            "dependencies.json",
            json.dumps(
                dict(
                    stage_d_depth_recovery=run["stage_d_depth_recovery"],
                    stage_c_refit_recovery=run["stage_c_refit_recovery"],
                ),
                indent=2,
            ),
        )
    archive_seconds = perf_counter() - started
    scratch = Path(run["stage_c_root"]) / f"attention_diagnostics_recovery_{commit[:7]}"
    scratch.mkdir(exist_ok=False)
    target = scratch / "one_file.bin"
    assert shutil.disk_usage(scratch).free > max(x["bytes"] for x in members.values())
    with zipfile.ZipFile(archive) as source:
        for name, receipt in members.items():
            with (
                source.open(aliases.get(name, name)) as compressed,
                target.open("wb") as restored,
            ):
                shutil.copyfileobj(compressed, restored, 1024 * 1024)
            assert (
                target.stat().st_size == receipt["bytes"]
                and sha256_file(target) == receipt["sha256"]
            ), name
            target.unlink()
    scratch.rmdir()
    report = dict(
        status="verified_completed_attention_diagnostic_recovery",
        implementation_commit=commit,
        archive={**binding(archive), "bytes": archive.stat().st_size},
        logical_members=len(members),
        unique_members=len(unique),
        aliases=len(aliases),
        inherited_members=len(inherited),
        archive_seconds=archive_seconds,
        recovery_seconds=perf_counter() - started - archive_seconds,
        dependencies=dict(
            stage_d_depth_recovery=run["stage_d_depth_recovery"],
            stage_c_refit_recovery=run["stage_c_refit_recovery"],
        ),
        scope="All completed fit/forecast/portfolio/position/input-contract/training-numerics/parent-patience diagnostics, exact executed/failure recipes, plans and source/docs restore/hash-check. No original source/store/old-fit copies. Broader eight-fold fits are actively growing and excluded, except their three immutable frozen plans. This is byte recovery, not repeated numerical proof. Prior depth/width/C archives also restore removed redundant epoch copies; recorded relocation and directory-junction maps govern physical placement.",
    )
    destination = Path(run["root"]) / "attention_diagnostics_recovery.json"
    write_json_atomic(destination, report)
    write_json_atomic(PROJECT / "docs/v2_attention_diagnostics_recovery.json", report)
    run = json.loads(pointer.read_text())
    run["scaling_diagnostic_recovery"] = binding(destination)
    write_json_atomic(pointer, run)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
