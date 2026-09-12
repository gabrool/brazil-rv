"""Frozen Round-7 job inventory, reusing the existing isolated-process launcher."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import research_rounds as rr
from .artifacts import sha256_file, write_json_atomic
from .contract import DEVELOPMENT_FOLDS, RUN_MANY_PLAN_SCHEMA
from .data_roots import resolve_external_root
from .round6 import training_command as old_training_command
from .round7 import CELLS, SCREEN_FOLDS, SEEDS, calibrate_budget, pretrain_key
from .round7_data import PROJECT


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pretraining(root, cell, seed):
    return root / "pretraining" / pretrain_key(cell) / f"seed_{seed}"


def trajectory(root, cell, fold, seed):
    if (
        cell == "B4"
        and (root / "budget.json").exists()
        and read(root / "budget.json")["B"] == 60
        and fold in SCREEN_FOLDS
        and seed in SEEDS
    ):
        return root / "calibration" / f"{fold}_seed_{seed}"
    return root / "trajectories" / cell / f"{fold}_seed_{seed}"


def freeze(root):
    code = rr._git_identity()
    inputs_path = PROJECT / "docs/v2_round7_inputs.json"
    inputs = read(inputs_path)
    if inputs["status"] != "accepted_cpu_preflight":
        raise ValueError("Round-7 CPU input acceptance is required")
    diagnostics = PROJECT / "docs/v2_round7_archived_diagnostics.json"
    if read(diagnostics)["status"] != "complete":
        raise ValueError("archived diagnostics must be reported before paid work")
    store, _ = resolve_external_root(inputs["store"]["root"])
    if sha256_file(store / "manifest.json") != inputs["store"]["manifest_sha256"]:
        raise ValueError("accepted repaired store changed")
    design = {
        "schema": "BRAZIL_RV_ROUND7_FROZEN_V1",
        "implementation": code,
        "store": {**inputs["store"], "root": str(store)},
        "feature_schema_sha256": read(store / "manifest.json")["feature_schema_sha256"],
        "protocol": read(PROJECT / "research/preregistrations/v2_round7.json"),
        "bindings": {
            str(p.relative_to(PROJECT)): sha256_file(p)
            for p in [
                inputs_path,
                diagnostics,
                PROJECT / "research/preregistrations/v2_round7.md",
                PROJECT / "research/preregistrations/v2_round7_implementation.md",
                PROJECT / "research/preregistrations/v2_round7.json",
            ]
        },
        "evaluation_design": inputs["evaluation_design"],
        "fast_initialization": {
            "mode": "native_fresh",
            "transfer_chronology_clean": True,
        },
        "forward_capture": False,
        "heldout_access": False,
    }
    root.mkdir(parents=True, exist_ok=False)
    write_json_atomic(root / "frozen_design.json", design)
    return design


def design_at(root):
    design = read(root / "frozen_design.json")
    if rr._git_identity()["commit"] != design["implementation"]["commit"]:
        raise ValueError("compute code differs from frozen Round-7 implementation")
    design["store"]["root"] = str(resolve_external_root(design["store"]["root"])[0])
    return design


def parent_binding(root, cell, seed):
    directory = pretraining(root, cell, seed)
    manifest = read(directory / "run_manifest.json")
    name = "raw_patience.pt" if pretrain_key(cell) == "s0_slow" else "tail_average.pt"
    path = directory / name
    if (
        manifest["status"] != "completed"
        or sha256_file(path) != manifest["artifacts"][name]
    ):
        raise ValueError("pretraining did not complete with its registered weights")
    return path, manifest["artifacts"][name]


def phase_tasks(root, phase):
    cells = {c["cell"]: c for c in CELLS}
    if phase == "anchor_p":
        return [("A0", "P", "pretrain_internal", s) for s in SEEDS]
    if phase == "anchor_f":
        return [("A0", "F", f, s) for f in DEVELOPMENT_FOLDS for s in SEEDS]
    if phase == "pretrain_r":
        seen, representatives = {"s0_slow"}, []
        for cell in CELLS:
            if pretrain_key(cell) not in seen:
                seen.add(pretrain_key(cell))
                representatives.append(cell["cell"])
        return [
            (c, "P", "pretrain_internal", s) for c in representatives for s in SEEDS
        ]
    if phase == "calibration":
        return [("B4", "F", f, s) for f in SCREEN_FOLDS for s in SEEDS]
    if phase == "screen":
        budget = read(root / "budget.json")["B"]
        dropped = (
            read(root / "budget_cut.json")["dropped_cells"]
            if (root / "budget_cut.json").exists()
            else []
        )
        if dropped != ["B11", "B7", "B8", "B2", "A3"][: len(dropped)]:
            raise ValueError("screen cuts must follow the registered order")
        return [
            (c["cell"], "F", f, s)
            for c in CELLS
            if c["cell"] != "A0"
            and c["cell"] not in dropped
            and not (c["cell"] == "B4" and budget == 60)
            for f in SCREEN_FOLDS
            for s in SEEDS
        ]
    if phase == "confirmation":
        advanced = read(root / "advancement.json")["cells"]
        return [
            (c, "F", f, s)
            for c in advanced
            for f in DEVELOPMENT_FOLDS
            if f not in SCREEN_FOLDS
            for s in SEEDS
        ]
    if phase in {"six_seed_p", "six_seed_f"}:
        leader = read(root / "confirmation_leader.json")["cell"]
        candidates = list(dict.fromkeys(["A0", leader]))
        if phase == "six_seed_p":
            representatives = {pretrain_key(cells[c]): c for c in candidates}
            # S0 R recipe variants still inherit old-recipe S0 P.
            representatives["s0_slow"] = "A0"
            return [
                (c, "P", "pretrain_internal", s)
                for c in representatives.values()
                for s in (61, 79, 97)
            ]
        return [
            (c, "F", f, s)
            for c in candidates
            for f in DEVELOPMENT_FOLDS
            for s in (61, 79, 97)
        ]
    raise ValueError(f"unknown phase: {phase}")


def plan(root, phase, parallel):
    design = design_at(root)
    engineering = read(root / "gpu_engineering.json")
    expected_cases = {"A1", "A3", "B1", "B4", "B6", "B7", "B8", "B9", "B10", "B11"}
    if (
        engineering["device"] != "cuda"
        or engineering["store_manifest_sha256"] != design["store"]["manifest_sha256"]
        or {r["cell"] for r in engineering["results"] if r["passed"]} != expected_cases
    ):
        raise ValueError(
            "all registered CUDA engineering cases must pass before real fits"
        )
    if (
        phase == "screen"
        and read(root / "anchor_diagnostics/summary.json")["status"] != "complete"
    ):
        raise ValueError("repaired S0 diagnostics must be reported before screening")
    cells = {c["cell"]: c for c in CELLS}
    tasks = phase_tasks(root, phase)
    jobs = []
    for name, stage, fold, seed in tasks:
        cell = cells[name]
        output = (
            pretraining(root, cell, seed)
            if stage == "P"
            else root / "calibration" / f"{fold}_seed_{seed}"
            if phase == "calibration"
            else trajectory(root, name, fold, seed)
        )
        parent, digest = (
            parent_binding(root, cell, seed) if stage == "F" else (None, None)
        )
        if name == "A0":
            command = old_training_command(
                design,
                output,
                "S0",
                seed,
                fold,
                stage,
                checkpoint=parent,
                digest=digest,
            )
            expected = {
                "status": "completed",
                "stage": stage,
                "transfer_chronology_clean": True,
                "compiled_graphs": {"training": 1, "selection": 1, "total": 2},
                "official_validation_accessed": False,
                "test_accessed": False,
            }
            if stage == "F":
                expected["scoring_complete"] = True
        else:
            epochs = (
                60
                if stage == "P" or phase == "calibration"
                else read(root / "budget.json")["B"]
            )
            command = [
                sys.executable,
                "-m",
                "brazil_rv.v2.round7_training",
                "--store",
                design["store"]["root"],
                "--output",
                str(output),
                "--cell",
                name,
                "--stage",
                stage,
                "--fold",
                fold,
                "--seed",
                str(seed),
                "--epochs",
                str(epochs),
            ]
            if parent:
                command += ["--parent", str(parent), "--parent-sha256", digest]
            if phase == "calibration":
                command.append("--calibration")
            elif stage == "F":
                command.append("--export-scores")
            expected = {
                "schema": "BRAZIL_RV_ROUND7_FIT_V1",
                "status": "completed",
                "stage": stage,
                "contract": {
                    "store_manifest_sha256": design["store"]["manifest_sha256"],
                    "pretrain_key": pretrain_key(cell),
                    "epochs": epochs,
                    "parent_sha256": digest,
                },
                "compiled_graphs": {"training": 1, "selection": 1},
            }
        jobs.append(
            {
                "name": f"{name}_{stage}_{fold}_{seed}",
                "seed": seed,
                "fold": fold,
                "run_dir": str(output),
                "cwd": str(PROJECT),
                "command": command,
                "expected_manifest": expected,
            }
        )
    result = {
        "schema": RUN_MANY_PLAN_SCHEMA,
        "phase": phase,
        "max_parallel": parallel,
        "frozen_design_sha256": sha256_file(root / "frozen_design.json"),
        "jobs": jobs,
    }
    path = root / f"plan_{phase}.json"
    if path.exists():
        raise FileExistsError(path)
    write_json_atomic(path, result)
    return {"plan": str(path), "jobs": len(jobs), "max_parallel": parallel}


def freeze_budget(root):
    design_at(root)
    path = root / "budget.json"
    if path.exists():
        raise FileExistsError(path)
    histories, bindings = [], {}
    for fold in SCREEN_FOLDS:
        for seed in SEEDS:
            directory = root / "calibration" / f"{fold}_seed_{seed}"
            manifest = read(directory / "run_manifest.json")
            history = read(directory / "history.json")
            if (
                manifest["status"] != "completed"
                or manifest["epochs_completed"] != 60
                or not all(h["clean_fit"] for h in history)
            ):
                raise ValueError(
                    "calibration needs sixty complete fit/selection readouts"
                )
            histories.append([h["selection"]["mean_ic"] for h in history])
            bindings[f"{fold}_{seed}"] = sha256_file(directory / "run_manifest.json")
    result = {
        "B": calibrate_budget(histories),
        "selection_curves": histories,
        "fit_manifest_hashes": bindings,
        "evaluation_scores_consumed": False,
        "rule": "registered equal-fit trailing-five mean, earliest multiple of five within .001 of maximum",
    }
    write_json_atomic(path, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "plan", "budget"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase")
    parser.add_argument("--parallel", type=int, default=6)
    args = parser.parse_args()
    if args.action == "freeze":
        result = freeze(args.root)
    elif args.action == "budget":
        result = freeze_budget(args.root)
    else:
        result = plan(args.root, args.phase, args.parallel)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
