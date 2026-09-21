"""Recover completed matched refits, fresh portfolio inputs and economic books."""

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
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    base = Path(run["stage_c_root"])
    fit_root = Path(run["stage_c_refit_root"])
    fit_progress = bound_json(binding(fit_root / "refits.json"))
    primary = Path(run["stage_c_data_replay_plan"]["path"]).parent
    sensitivity = Path(run["stage_c_refit_sensitivity_plan"]["path"]).parent
    assert fit_progress["status"] == "complete"
    for key in (
        "stage_c_data_replay_qualification",
        "stage_c_refit_sensitivity_qualification",
        "stage_c_refit_debit_qualification",
    ):
        assert bound_json(run[key])["status"] == "complete"
    bound_json(run["stage_c_refit_results"])
    for root in (primary, sensitivity):
        assert bound_json(binding(root / "replays.json"))["status"] == "complete"
    previous = bound_json(run["stage_c_matched_store_recovery"])
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    archive = Path(run["root"]) / f"matched_refits_{commit[:7]}.zip"
    assert not archive.exists()
    with zipfile.ZipFile(previous["archive"]["path"]) as prior:
        inherited_inventory = json.loads(prior.read("inherited_members.json"))
        inherited_inventory.update(json.loads(prior.read("members.json")))
    members, aliases, unique, inherited = {}, {}, {}, {}
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as z:

        def file(name, path):
            rec = dict(sha256=sha256_file(path), bytes=path.stat().st_size)
            if inherited_inventory.get(name) == rec:
                inherited[name] = rec
                return
            assert name not in members
            members[name] = rec
            if rec["sha256"] in unique:
                aliases[name] = unique[rec["sha256"]]
            else:
                unique[rec["sha256"]] = name
                z.write(path, name)

        project_files = subprocess.check_output(
            ["git", "ls-files", "research", "ops"], cwd=PROJECT, text=True
        ).splitlines()
        project_files += [
            "PROJECT_CONTEXT.md",
            "docs/v2_MATCHED_REFITS.md",
            "docs/v2_economic_data_scaling_progress.md",
            "docs/v2_economic_data_scaling_run.json",
            "docs/v2_matched_data_inputs.json",
            "docs/v2_matched_store_contract.json",
        ]
        for name in sorted(set(project_files)):
            file("project/" + name, PROJECT / name)
        for root, label in (
            (base / "refit_economics", "refit_economics"),
            (fit_root, "data_refits"),
            (primary, "data_refit_replays"),
            (sensitivity, "matched_refit_sensitivities"),
            (
                Path(run["stage_c_refit_debit_plan"]["path"]).parent,
                "matched_refit_debit_bounds",
            ),
        ):
            for directory, children, names in os.walk(root):
                children[:] = [d for d in children if d not in {"__pycache__", "fits"}]
                for name in sorted(names):
                    path = Path(directory) / name
                    assert path.resolve().is_relative_to(root.resolve())
                    file(
                        "evidence/" + label + "/" + path.relative_to(root).as_posix(),
                        path,
                    )
        physical_fits = Path(run["root"]) / "matched_refit_fit_storage"
        for record in fit_progress["completed"]:
            manifest = Path(record["manifest"]["path"])
            bound_json(record["manifest"])
            root = manifest.parent
            for path in sorted(root.rglob("*")):
                if not path.is_file() or "__pycache__" in path.parts:
                    continue
                resolved = path.resolve()
                assert resolved.is_relative_to(
                    fit_root.resolve()
                ) or resolved.is_relative_to(physical_fits.resolve())
                file(
                    "evidence/data_refits/" + path.relative_to(fit_root).as_posix(),
                    path,
                )
        z.writestr("members.json", json.dumps(members, indent=2))
        z.writestr("aliases.json", json.dumps(aliases, indent=2))
        z.writestr("inherited_members.json", json.dumps(inherited, indent=2))
        z.writestr(
            "dependencies.json",
            json.dumps(
                {
                    k: run[k]
                    for k in (
                        "stage_c_matched_store_recovery",
                        "stage_c_account_recovery",
                    )
                },
                indent=2,
            ),
        )
    archive_seconds = perf_counter() - tick
    scratch = base / f"refit_recovery_scratch_{commit[:7]}"
    scratch.mkdir(exist_ok=False)
    target = scratch / "one_file.bin"
    assert shutil.disk_usage(scratch).free > max(r["bytes"] for r in members.values())
    with zipfile.ZipFile(archive) as z:
        for name, record in members.items():
            with z.open(aliases.get(name, name)) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            assert target.stat().st_size == record["bytes"]
            assert sha256_file(target) == record["sha256"], name
            target.unlink()
    scratch.rmdir()
    report = dict(
        status="verified_matched_refit_and_economic_artifact_recovery",
        implementation_commit=commit,
        archive={**binding(archive), "bytes": archive.stat().st_size},
        logical_members=len(members),
        unique_members=len(unique),
        aliases=len(aliases),
        inherited_members=len(inherited),
        fits=len(fit_progress["completed"]),
        dependencies={
            k: run[k]
            for k in ("stage_c_matched_store_recovery", "stage_c_account_recovery")
        },
        archive_seconds=archive_seconds,
        recovery_seconds=perf_counter() - tick - archive_seconds,
        scope="Every new logical artifact restored and hash-checked one file at a time, including all epoch checkpoints, fresh derived portfolio cache, failed/executed recipes, storage journals, forecasts, primary and sensitivity books. Unchanged project source inherits the prior verified archive chain. Original sources, accepted store NPY files and old fits are not copied. Canonical fit paths and physical junction targets remain documented. This is byte recovery, not a new model/source/history proof or completion of conditional later experiments.",
    )
    output = Path(run["root"]) / "matched_refit_recovery.json"
    write_json_atomic(output, report)
    write_json_atomic(PROJECT / "docs/v2_matched_refit_recovery.json", report)
    run["stage_c_refit_recovery"] = binding(output)
    write_json_atomic(pointer, run)
    print(json.dumps({k: v for k, v in report.items() if k != "scope"}), flush=True)


if __name__ == "__main__":
    main()
