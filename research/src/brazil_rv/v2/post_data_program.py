"""Bounded post-data calibration and matched screen using the existing job runner."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .artifacts import sha256_file, write_json_atomic
from .contract import RUN_MANY_PLAN_SCHEMA
from .data_roots import resolve_external_root
from .research_rounds import _git_identity
from .round6 import training_command as incumbent_command
from .round7 import SCREEN_FOLDS, SEEDS, configuration
from .round7_data import PROJECT
from .round7_training import TrainingRecipe

CELLS = {
    name: {
        "cell": name,
        "graph": graph,
        "inputs": inputs,
        "film": film,
        "temporal_encoder": encoder,
        "peer_timing": peer,
    }
    for name, graph, inputs, film, encoder, peer in (
        ("S0", "s0", "slow", False, "gru", "none"),
        ("S0_common", "s0", "slow", False, "gru", "none"),
        ("C1_slow", "c1", "slow", False, "gru", "none"),
        ("TE_slow", "c1", "slow", False, "attention", "early"),
        ("TL_slow", "c1", "slow", False, "attention", "late"),
        ("C1_family", "c1", "all", False, "gru", "none"),
        ("C1_all", "c1", "all", True, "gru", "none"),
        ("TE_family", "c1", "all", False, "attention", "early"),
        ("TE_all", "c1", "all", True, "attention", "early"),
    )
}
RECIPES = {
    "sam125": TrainingRecipe(),
    "asam20": TrainingRecipe(rho=0.2, adaptive=True),
    "asam50": TrainingRecipe(rho=0.5, adaptive=True),
    "sam125_lr3": TrainingRecipe(learning_rate=3e-4),
    "sam05": TrainingRecipe(rho=0.05),
    "sam125_transfer1": TrainingRecipe(transferred_multiplier=1.0),
}
CALIBRATION_CELLS = ("TE_slow", "C1_all")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def freeze(root):
    pointer = PROJECT / "docs/v2_data_inputs.json"
    inputs = read(pointer)
    store, resolution = resolve_external_root(inputs["store"]["root"])
    if (
        not inputs["accepted"]
        or sha256_file(store / "manifest.json") != inputs["store"]["manifest_sha256"]
    ):
        raise ValueError("current data acceptance does not bind this store")
    manifest = read(store / "manifest.json")
    if manifest["axes"]["date_end"] > "2024-12-30":
        raise PermissionError("protected history is outside this research program")
    root.mkdir(parents=True, exist_ok=False)
    design = {
        "implementation": _git_identity(),
        "store": {**inputs["store"], "root": str(store)},
        "store_resolution": resolution.payload(),
        "pointer_sha256": sha256_file(pointer),
        "registration_sha256": sha256_file(
            PROJECT / "research/preregistrations/v2_post_data.md"
        ),
        "feature_names": manifest["feature_names"],
        "feature_schema_sha256": manifest["feature_schema_sha256"],
        "economic_source": read(PROJECT / "docs/v2_round7_inputs.json")[
            "evaluation_design"
        ],
        "cells": {
            name: {
                "spec": cell,
                "config": asdict(configuration(cell, manifest["feature_names"])),
            }
            for name, cell in CELLS.items()
        },
        "recipes": {name: asdict(recipe) for name, recipe in RECIPES.items()},
        "screen_folds": SCREEN_FOLDS,
        "seeds": SEEDS,
        "parent_recipe": "sam125; selected independently per graph/input/seed",
        "calibration_rule": "equal-fit selected selection IC; settings within .001 of maximum tie by declared recipe order; full trajectories also reported",
        "fast_initialization": {
            "mode": "native_fresh",
            "transfer_chronology_clean": True,
        },
        "forward_capture": False,
        "heldout_access": False,
    }
    for name, cell in CELLS.items():
        write_json_atomic(root / "specs" / f"{name}.json", cell)
    for name, recipe in RECIPES.items():
        write_json_atomic(root / "recipes" / f"{name}.json", asdict(recipe))
    write_json_atomic(root / "frozen_design.json", design)
    return design


def parent_path(root, cell, seed):
    return root / "parents" / cell / f"seed_{seed}"


def fit_path(root, cell, recipe, fold, seed):
    return root / "fits" / cell / recipe / f"{fold}_seed_{seed}"


def plan(root, phase, *, max_parallel=2):
    design = read(root / "frozen_design.json")
    if design["implementation"]["commit"] != _git_identity()["commit"]:
        raise ValueError("training checkout differs from the frozen implementation")
    for name, recipe in design["recipes"].items():
        if read(root / "recipes" / f"{name}.json") != recipe:
            raise ValueError("recipe changed after freeze")
    for name, item in design["cells"].items():
        if read(root / "specs" / f"{name}.json") != item["spec"]:
            raise ValueError("cell inputs or graph changed after freeze")
    tasks = []
    if phase == "b_parents":
        tasks = [
            (c, "sam125", "P", "pretrain_internal", s)
            for c in CALIBRATION_CELLS
            for s in SEEDS[:2]
        ]
    elif phase == "b_calibration":
        tasks = [
            (c, r, "F", f, s)
            for c in CALIBRATION_CELLS
            for r in RECIPES
            for f in (("F14",) if r.endswith("transfer1") else ("F2", "F14"))
            for s in SEEDS[:2]
        ]
    elif phase == "c_parents":
        tasks = [
            (c, "sam125", "P", "pretrain_internal", s) for c in CELLS for s in SEEDS
        ]
    elif phase == "c_screen":
        choices = read(root / "calibration_choice.json")
        tasks = [
            (c, "incumbent" if c == "S0" else choices["recipes"][c], "F", f, s)
            for c in CELLS
            for f in SCREEN_FOLDS
            for s in SEEDS
        ]
    else:
        raise ValueError("unknown post-data phase")
    jobs = []
    for name, recipe, stage, fold, seed in tasks:
        output = (
            parent_path(root, name, seed)
            if stage == "P"
            else fit_path(root, name, recipe, fold, seed)
        )
        parent = digest = None
        if stage == "F":
            parent_root = parent_path(root, name, seed)
            parent = parent_root / (
                "raw_patience.pt" if name == "S0" else "selected.pt"
            )
            accepted = read(parent_root / "run_manifest.json")
            digest = accepted["artifacts"][parent.name]
            if accepted["status"] != "completed" or sha256_file(parent) != digest:
                raise ValueError("parent has not completed with its bound weights")
        if name == "S0":
            command = incumbent_command(
                design,
                output,
                "S0",
                seed,
                fold,
                stage,
                checkpoint=parent,
                digest=digest,
            )
        else:
            command = [
                sys.executable,
                "-m",
                "brazil_rv.v2.round7_training",
                "--store",
                design["store"]["root"],
                "--output",
                str(output),
                "--cell-spec",
                str(root / "specs" / f"{name}.json"),
                "--recipe",
                str(root / "recipes" / f"{recipe}.json"),
                "--stage",
                stage,
                "--fold",
                fold,
                "--seed",
                str(seed),
                "--epochs",
                "60",
            ]
            if stage == "F":
                command += ["--parent", str(parent), "--parent-sha256", digest]
            if name.startswith("C1") or name == "S0_common":
                # Full-batch GH200 checks show no steady-state GRU benefit;
                # retain compilation for the temporal-attention graphs.
                command.append("--eager")
            if phase == "c_screen":
                command.append("--export-scores")
        # Existing complete B fits are submitted once more only to attach scores;
        # their trainer verifies/reuses the saved fit instead of training again.
        expected = {"status": "completed", "stage": stage}
        if name != "S0":
            expected["contract"] = {
                "cell": CELLS[name],
                "recipe": design["recipes"][recipe],
                "store_manifest_sha256": design["store"]["manifest_sha256"],
            }
            if phase == "c_screen":
                expected["scoring_complete"] = True
        jobs.append(
            {
                "name": f"{name}_{recipe}_{stage}_{fold}_{seed}",
                "seed": seed,
                "fold": fold,
                "run_dir": str(output),
                "cwd": str(PROJECT),
                "command": command,
                "expected_manifest": expected,
                "resume": name != "S0",
            }
        )
    path = root / f"{phase}_plan.json"
    write_json_atomic(
        path,
        {"schema": RUN_MANY_PLAN_SCHEMA, "max_parallel": max_parallel, "jobs": jobs},
    )
    return {"path": str(path), "jobs": len(jobs)}


def choose(root):
    summaries, choices = {}, {}
    for cell in CALIBRATION_CELLS:
        by_recipe = {}
        # The transfer1 bridge has only F14 and cannot win a four-fit comparison.
        for recipe in tuple(RECIPES)[:-1]:
            records = [
                read(fit_path(root, cell, recipe, fold, seed) / "run_manifest.json")
                for fold in ("F2", "F14")
                for seed in SEEDS[:2]
            ]
            values = [r["selection_ic"] for r in records]
            if not np.isfinite(values).all():
                raise ValueError("undefined calibration selection score")
            by_recipe[recipe] = {
                "mean_ic": float(np.mean(values)),
                "fit_ic": values,
                "selected_epochs": [r["selected_epoch"] for r in records],
                "completed_epochs": [r["epochs_completed"] for r in records],
            }
        maximum = max(r["mean_ic"] for r in by_recipe.values())
        choices[cell] = next(
            r for r, v in by_recipe.items() if v["mean_ic"] >= maximum - 0.001
        )
        summaries[cell] = by_recipe
    mapped = {
        name: choices["TE_slow"]
        if name.startswith(("TE", "TL"))
        else choices["C1_all"]
        if name.startswith("C1")
        else "sam125"
        for name in CELLS
        if name != "S0"
    }
    result = {
        "recipes": mapped,
        "calibration": summaries,
        "decision_basis": "fit/selection only; independent recipe tuning by rich-GRU and slow-attention lanes",
        "evaluation_scores_read": False,
    }
    path = root / "calibration_choice.json"
    if path.exists() and read(path) != result:
        raise ValueError("calibration choice is already frozen")
    write_json_atomic(path, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "plan", "choose"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("b_parents", "b_calibration", "c_parents", "c_screen")
    )
    parser.add_argument("--max-parallel", type=int, default=2)
    args = parser.parse_args()
    result = (
        freeze(args.root)
        if args.command == "freeze"
        else choose(args.root)
        if args.command == "choose"
        else plan(args.root, args.phase, max_parallel=args.max_parallel)
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
