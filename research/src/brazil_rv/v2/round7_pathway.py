"""Separately frozen pathway factorial using the existing Round-7 fit engine."""

from __future__ import annotations

import argparse
import json
import sys
import subprocess
from dataclasses import asdict
from pathlib import Path

from .artifacts import sha256_file, write_json_atomic
from .characteristic_model import CharacteristicModel
from .contract import DEVELOPMENT_FOLDS, RUN_MANY_PLAN_SCHEMA
from .data_roots import resolve_external_root
from .research_rounds import _git_identity
from .round7 import (
    CELLS,
    PATHWAY_CELLS,
    SCREEN_FOLDS,
    SEEDS,
    configuration,
    pretrain_key,
)
from .round7_data import PROJECT
from .round7_program import read, pretraining, trajectory
from .round7_program import parent_binding
from .round6 import training_command as old_training_command
from .round7_seed_audit import run as omission_audit

ORDER = tuple(c["cell"] for c in PATHWAY_CELLS)
PAIRS = (("GL", "GE"), ("TL", "TE"))
IC = "primary_neutral_target_ic"
NET = "headline_net_excess_bps"


def register():
    inputs = read(PROJECT / "docs/v2_round7_inputs.json")
    store = resolve_external_root(inputs["store"]["root"])[0]
    names = read(store / "manifest.json")["feature_names"]
    cells = []
    for cell in PATHWAY_CELLS:
        config = configuration(cell, names)
        model = CharacteristicModel(config)
        cells.append(
            {
                **cell,
                "config": asdict(config),
                "parameters": sum(p.numel() for p in model.parameters()),
            }
        )
    result = {
        "schema": "BRAZIL_RV_PATHWAY_PROTOCOL_V1",
        "cells": cells,
        "store": inputs["store"],
        "screen_folds": SCREEN_FOLDS,
        "seeds": SEEDS,
        "registration_sha256": sha256_file(
            PROJECT / "research/preregistrations/v2_round7_pathway.md"
        ),
        "original_protocol_sha256": sha256_file(
            PROJECT / "research/preregistrations/v2_round7.json"
        ),
        "primary_contrasts": ["GE_minus_GL", "TE_minus_TL"],
        "secondary_contrasts": ["TL_minus_GL", "TE_minus_GE", "interaction"],
        "stage_p_epochs": 60,
        "fine_tune_budget": "inherit_original_B4_calibration",
        "rho": 0.05,
        "forward_capture": False,
        "heldout_access": False,
    }
    write_json_atomic(
        PROJECT / "research/preregistrations/v2_round7_pathway.json", result
    )
    return {c["cell"]: c["parameters"] for c in cells}


def freeze(root, original):
    code = _git_identity()
    original = original.resolve()
    base = read(original / "frozen_design.json")
    protocol = read(PROJECT / "research/preregistrations/v2_round7_pathway.json")
    if protocol["registration_sha256"] != sha256_file(
        PROJECT / "research/preregistrations/v2_round7_pathway.md"
    ):
        raise ValueError("machine registration must match the current written protocol")
    if base["store"]["manifest_sha256"] != protocol["store"]["manifest_sha256"]:
        raise ValueError("original and extension stores must match")
    root.mkdir(parents=True, exist_ok=False)
    result = {
        **base,
        "implementation": code,
        "pathway_extension": True,
        "pathway_protocol": protocol,
        "original_run": {
            "root": str(original),
            "frozen_design_sha256": sha256_file(original / "frozen_design.json"),
        },
        "comparison_roots": {c["cell"]: str(original) for c in CELLS},
        "pathway_registration_sha256": sha256_file(
            PROJECT / "research/preregistrations/v2_round7_pathway.md"
        ),
    }
    write_json_atomic(root / "frozen_design.json", result)
    return result["implementation"]


def original_at(design):
    original = resolve_external_root(design["original_run"]["root"])[0]
    if (
        sha256_file(original / "frozen_design.json")
        != design["original_run"]["frozen_design_sha256"]
    ):
        raise ValueError("bound original experiment changed")
    return original


def advance(result):
    if set(result["folds"]) != set(SCREEN_FOLDS) or tuple(result["seeds"]) != SEEDS:
        raise ValueError(
            "pathway screen requires four registered folds and three seeds"
        )

    def delta(cell):
        return result["paired"][f"{cell}_minus_B4"]

    def fold_delta(cell, fold):
        row = delta(cell)
        if "opposite_of" in row:
            return -result["paired"][row["opposite_of"]]["folds"][fold][IC]["estimate"]
        return row["folds"][fold][IC]["estimate"]

    points = {c: delta(c)["pooled"][IC]["estimate"] for c in ORDER}
    qualified = [
        pair
        for pair in PAIRS
        if any(
            points[c] >= 0.002 and sum(fold_delta(c, f) > 0 for f in SCREEN_FOLDS) >= 3
            for c in pair
        )
    ]
    if qualified:
        leader = max(points.values())
        qualified = [
            pair
            for pair in PAIRS
            if pair in qualified
            or any(points[c] > 0 and points[c] >= leader - 0.002 for c in pair)
        ]
    return {
        "cells": [c for pair in qualified for c in pair],
        "screen_deltas": points,
        "interpretation": "screening decision, not a test of absence or a held-out result",
    }


def confirmation_choice(result):
    if (
        set(result["folds"]) != set(DEVELOPMENT_FOLDS)
        or tuple(result["seeds"]) != SEEDS
    ):
        raise ValueError(
            "confirmation requires all fourteen folds and three matched seeds"
        )
    candidates = [c for c in ORDER if c in result["cells"]]
    return max(
        candidates, key=lambda c: result["readouts"][c]["pooled"][IC]["estimate"]
    )


def select(result, omissions, reference):
    candidate = confirmation_choice(result)
    if (
        omissions["candidate"] != candidate
        or omissions["reference"] != reference
        or tuple(omissions["seeds"]) != SEEDS
    ):
        raise ValueError(
            "omissions must use the highest-IC candidate and original designation"
        )
    pair = result["paired"][f"{candidate}_minus_{reference}"]["pooled"]
    net = result["readouts"][candidate]["pooled"][NET]["estimate"]
    eligible = (
        pair[IC]["estimate"] is not None
        and pair[IC]["estimate"] > 0
        and pair[IC]["lower_95"] is not None
        and pair[IC]["lower_95"] > 0
        and omissions["all_omissions_positive"]
        and net is not None
        and net >= 0
        and pair[NET]["estimate"] is not None
        and pair[NET]["estimate"] >= 0
    )
    matched = {"GE": "GL", "TE": "TL"}.get(candidate)
    return {
        "cell": candidate,
        "reference": reference,
        "eligible": bool(eligible),
        "extension_cells": [candidate] + ([matched] if matched else []),
        "paired": pair,
        "all_omissions_positive": omissions["all_omissions_positive"],
        "interpretation": "candidate frozen before fresh seeds; nominal intervals follow development selection",
    }


def finalize(decision, six=None, omissions=None):
    candidate, reference = decision["cell"], decision["reference"]
    accepted = False
    if decision["eligible"]:
        if (
            tuple(six["seeds"]) != (11, 29, 47, 61, 79, 97)
            or set(six["folds"]) != set(DEVELOPMENT_FOLDS)
            or omissions["candidate"] != candidate
            or omissions["reference"] != reference
            or tuple(omissions["seeds"]) != tuple(six["seeds"])
        ):
            raise ValueError(
                "finalization requires the frozen candidate and six matched seeds"
            )
        pair = six["paired"][f"{candidate}_minus_{reference}"]["pooled"]
        net = six["readouts"][candidate]["pooled"][NET]["estimate"]
        accepted = (
            pair[IC]["estimate"] is not None
            and pair[IC]["estimate"] > 0
            and pair[NET]["estimate"] is not None
            and pair[NET]["estimate"] >= 0
            and net is not None
            and net >= 0
            and omissions["all_omissions_positive"]
        )
    return {
        "status": "complete",
        "original_designation": reference,
        "designation": candidate if accepted else reference,
        "extension_candidate": candidate,
        "extension_accepted": bool(accepted),
        "reason": "registered paired gain and seed agreement"
        if accepted
        else "no qualifying stable extension gain",
        "interpretation": "development research; no untouched holdout or deployment claim",
    }


def plan(root, phase, parallel):
    design = read(root / "frozen_design.json")
    if design["implementation"]["commit"] != _git_identity()["commit"]:
        raise ValueError("extension code differs from freeze")
    original = original_at(design)
    evidence = read(root / "gpu_engineering.json")
    if (
        evidence["device"] != "cuda"
        or evidence["store_manifest_sha256"] != design["store"]["manifest_sha256"]
        or {r["cell"] for r in evidence["results"] if r["passed"]} != set(ORDER)
    ):
        raise ValueError("all four extension graphs require CUDA acceptance")
    if read(root / "pathway_acceptance.json")["status"] != "passed":
        raise ValueError("pathway-specific correctness acceptance required")
    budget = 60 if phase == "pretrain" else read(original / "budget.json")["B"]
    cells = {c["cell"]: c for c in (*CELLS, *PATHWAY_CELLS)}
    if phase == "pretrain":
        tasks = [(c, "P", "pretrain_internal", s) for c in ORDER for s in SEEDS]
    elif phase == "screen":
        tasks = [(c, "F", f, s) for c in ORDER for f in SCREEN_FOLDS for s in SEEDS]
    elif phase == "confirmation":
        selected = read(root / "advancement.json")["cells"]
        tasks = (
            [
                (c, "F", f, s)
                for c in [*selected, "B4"]
                for f in DEVELOPMENT_FOLDS
                if f not in SCREEN_FOLDS
                for s in SEEDS
            ]
            if selected
            else []
        )
    elif phase in ("six_seed_p", "six_seed_f"):
        decision = read(root / "confirmation_leader.json")
        selected = (
            decision["extension_cells"] + [decision["reference"]]
            if decision["eligible"]
            else []
        )
        tasks = (
            [(c, "P", "pretrain_internal", s) for c in selected for s in (61, 79, 97)]
            if phase == "six_seed_p"
            else [
                (c, "F", f, s)
                for c in selected
                for f in DEVELOPMENT_FOLDS
                for s in (61, 79, 97)
            ]
        )
    else:
        raise ValueError("unknown extension phase")
    jobs = []
    for name, stage, fold, seed in tasks:
        cell = cells[name]
        fit_root, checkout, executable = root, PROJECT, sys.executable
        if name not in ORDER:
            # Supplementary references use the original frozen engine and
            # directory, preserving exact compatibility and completed work.
            fit_root, checkout = original, PROJECT.with_name("b3-quant")
            actual = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
            ).strip()
            if (
                actual
                != read(original / "frozen_design.json")["implementation"]["commit"]
            ):
                raise ValueError("original reference checkout differs from its freeze")
            executable = (
                str(checkout / "research/.venv/bin/python")
                if sys.platform != "win32"
                else sys.executable
            )
            source = (
                pretraining(original, cell, seed)
                if stage == "P"
                else trajectory(original, name, fold, seed)
            )
            if (source / "run_manifest.json").exists():
                if read(source / "run_manifest.json")["status"] != "completed":
                    raise ValueError("reference trajectory is incomplete")
                continue
        output = (
            pretraining(fit_root, cell, seed)
            if stage == "P"
            else trajectory(fit_root, name, fold, seed)
        )
        parent, digest = None, None
        if stage == "F":
            parent, digest = parent_binding(fit_root, cell, seed)
        epochs = 60 if stage == "P" else budget
        command = [
            executable,
            "-m",
            "brazil_rv.v2.round7_training",
            "--store",
            str(resolve_external_root(design["store"]["root"])[0]),
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
            command += [
                "--parent",
                str(parent),
                "--parent-sha256",
                digest,
                "--export-scores",
            ]
        expected = {
            "status": "completed",
            "stage": stage,
            "contract": {
                "pretrain_key": pretrain_key(cell),
                "epochs": epochs,
                "parent_sha256": digest,
            },
            "compiled_graphs": {"training": 1, "selection": 1},
        }
        if name == "A0" or (stage == "P" and pretrain_key(cell) == "s0_slow"):
            base = read(original / "frozen_design.json")
            base["store"]["root"] = str(resolve_external_root(base["store"]["root"])[0])
            command = old_training_command(
                base, output, "S0", seed, fold, stage, checkpoint=parent, digest=digest
            )
            command[0] = executable
            expected = {
                "status": "completed",
                "stage": stage,
                "compiled_graphs": {"training": 1, "selection": 1, "total": 2},
                "transfer_chronology_clean": True,
            }
            if stage == "F":
                expected["scoring_complete"] = True
        jobs.append(
            {
                "name": f"{name}_{stage}_{fold}_{seed}",
                "seed": seed,
                "fold": fold,
                "run_dir": str(output),
                "cwd": str(checkout),
                "command": command,
                "expected_manifest": expected,
            }
        )
    result = {
        "schema": RUN_MANY_PLAN_SCHEMA,
        "phase": phase,
        "max_parallel": parallel,
        "jobs": jobs,
        "frozen_design_sha256": sha256_file(root / "frozen_design.json"),
    }
    path = root / f"plan_{phase}.json"
    if path.exists():
        raise FileExistsError(path)
    write_json_atomic(path, result)
    return {"jobs": len(jobs), "plan": str(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("register", "freeze", "plan", "advance", "select", "finalize"),
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--phase")
    parser.add_argument("--parallel", type=int, default=3)
    args = parser.parse_args()
    if args.action == "register":
        result = register()
    elif args.action == "freeze":
        result = freeze(args.root, args.original)
    elif args.action == "plan":
        result = plan(args.root, args.phase, args.parallel)
    elif args.action == "advance":
        result = advance(read(args.root / "pathway_screen_result.json"))
        path = args.root / "advancement.json"
        if path.exists():
            raise FileExistsError(path)
        write_json_atomic(path, result)
    else:
        root = args.root
        original = original_at(read(root / "frozen_design.json"))
        reference = read(original / "decision.json")["designation"]
        path = root / (
            "confirmation_leader.json" if args.action == "select" else "decision.json"
        )
        if path.exists():
            raise FileExistsError(path)
        sources = {"original_decision": sha256_file(original / "decision.json")}
        if args.action == "select":
            source = root / "pathway_confirmation_result.json"
            panel = read(source)
            omissions = omission_audit(
                root,
                candidate=confirmation_choice(panel),
                reference=reference,
                seeds=SEEDS,
                group="pathway_confirmation",
            )
            result = select(panel, omissions, reference)
            sources["confirmation"] = sha256_file(source)
            sources["omissions"] = sha256_file(root / "pathway_confirmation_loso.json")
        else:
            if not read(root / "advancement.json")["cells"]:
                result = {
                    "status": "complete",
                    "original_designation": reference,
                    "designation": reference,
                    "extension_candidate": None,
                    "extension_accepted": False,
                    "reason": "no encoder pair met the fixed screen advancement rule",
                    "interpretation": "no demonstrated benefit under this screen, not proof of absence",
                    "source_hashes": {
                        **sources,
                        "advancement": sha256_file(root / "advancement.json"),
                        "screen": sha256_file(root / "pathway_screen_result.json"),
                    },
                }
                write_json_atomic(path, result)
                print(json.dumps(result))
                return
            decision = read(root / "confirmation_leader.json")
            six, omissions = None, None
            if decision["eligible"]:
                six = read(root / "pathway_six_seed_result.json")
                omissions = omission_audit(
                    root,
                    candidate=decision["cell"],
                    reference=reference,
                    group="pathway_six_seed",
                )
                sources["six_seed"] = sha256_file(root / "pathway_six_seed_result.json")
                sources["omissions"] = sha256_file(root / "pathway_six_seed_loso.json")
            result = finalize(decision, six, omissions)
            sources["confirmation_leader"] = sha256_file(
                root / "confirmation_leader.json"
            )
        result["source_hashes"] = sources
        write_json_atomic(path, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
