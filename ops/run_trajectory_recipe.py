"""Trajectory-recipe wave: same cells, same data, only the stopping/selection rule.

Contrast registered by ``docs/v2_SHARPE_TWO_PLAN.md`` (R1). Every cell keeps its
Stage C architecture, inputs, optimizer, loss, selector population and store. The
only changes are the checkpoint trajectory:

* P stage: patience 5 -> ``--p-patience`` (default 20) and trailing-window
  smoothed selection (``--smoothing``, default 3) instead of the raw best epoch.
* F stage: the 60-epoch cosine schedule with patience 5 is replaced by a short
  fully annealed schedule (``schedule_epochs = epochs = --f-epochs``, default 8),
  patience equal to the epoch budget (no early stop) and the same smoothed
  selection, so the selected child is a low-learning-rate iterate near the centre
  of the best trailing window rather than a near-peak-LR draw.

Freeze once, then execute (resumable, one fit at a time on the RTX 2060):

    uv run --project research python ops/run_trajectory_recipe.py --freeze \
        --cell TE_full --cell C6 --seed 11 --seed 29 --seed 47 --all-folds
    uv run --project research python ops/run_trajectory_recipe.py

Fits land under ``<run root>/trajectory_recipe/fits/<cell>/...`` with the same
layout as the Stage C refits, so the existing replay and account tooling can
score them paired against the Stage C controls fold by fold and seed by seed.
Nothing here reads evaluation labels or held-out consumers.
"""

import argparse
import gc
import inspect
import json
from pathlib import Path
from time import perf_counter

import torch

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.characteristic_model import CharacteristicModel
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.research_rounds import _git_identity
from brazil_rv.v2.round7 import configuration
from brazil_rv.v2.round7_training import TrainingRecipe, train
from brazil_rv.v2.splits import _SELECTION_WINDOWS
from brazil_rv.v2.train import compile_forward

PROJECT = Path(__file__).resolve().parents[1]
POINTER = PROJECT / "docs/v2_economic_data_scaling_run.json"
PLAN_KEY = "trajectory_recipe_plan"


def trajectory_recipes(p_recipe, f_recipe, *, p_patience, f_epochs, smoothing):
    """Derive the P and F trajectory recipes from the Stage C recipes.

    Only stopping and selection fields change; learning rate, SAM radius,
    adaptivity and the transferred-parameter multiplier are inherited unchanged.
    ``f_epochs=None`` keeps the source F schedule and patience (the fallback arm)
    and adds only the smoothed selection.
    """
    parent = {
        **p_recipe,
        "patience": int(p_patience),
        "selection_smoothing": int(smoothing),
    }
    child = {**f_recipe, "selection_smoothing": int(smoothing)}
    if f_epochs is not None:
        child["schedule_epochs"] = int(f_epochs)
        child["patience"] = int(f_epochs)
    for recipe in (parent, child):
        TrainingRecipe(**recipe)  # every key must be a real recipe field
    return parent, child


def freeze(run, args):
    source = bound_json(run["stage_c_refit_plan"])
    cells = args.cell or sorted(source["cells"])
    unknown = set(cells) - set(source["cells"])
    if unknown:
        raise SystemExit(f"cells not in the Stage C plan: {sorted(unknown)}")
    seeds = args.seed or list(source["seeds"])
    folds = list(_SELECTION_WINDOWS) if args.all_folds else list(source["folds"])
    f_epochs = None if args.keep_f_schedule else int(args.f_epochs)
    p_recipe, f_recipes = None, {}
    for cell in cells:
        parent, child = trajectory_recipes(
            source["p_recipe"],
            source["f_recipes"][cell],
            p_patience=args.p_patience,
            f_epochs=f_epochs,
            smoothing=args.smoothing,
        )
        p_recipe, f_recipes[cell] = parent, child
    f_budget = source["maximum_epochs"] if f_epochs is None else f_epochs
    f_contrast = (
        "F schedule and patience unchanged, smoothed selection only"
        if f_epochs is None
        else (
            f"F 60-epoch cosine/patience "
            f"{next(iter(source['f_recipes'].values()))['patience']} -> fully "
            f"annealed {f_epochs}-epoch schedule, no early stop, same smoothed "
            "selection"
        )
    )
    controls = {}
    for cell in cells:
        for seed in source["seeds"]:
            for fold in source["folds"]:
                path = (
                    Path(run["stage_c_refit_root"])
                    / "fits"
                    / cell
                    / f"{fold}_seed_{seed}"
                    / "run_manifest.json"
                )
                if path.exists():
                    controls[f"{cell}/{fold}/{seed}"] = binding(path)
    root = Path(run["root"]) / "trajectory_recipe"
    root.mkdir(exist_ok=False)
    plan = {
        "status": "frozen_before_trajectory_outcomes",
        "source": run["stage_c_refit_plan"],
        "store": source["store"],
        "cells": {cell: source["cells"][cell] for cell in cells},
        "controls": controls,
        "seeds": seeds,
        "folds": folds,
        "p_recipe": p_recipe,
        "f_recipes": f_recipes,
        "maximum_epochs": source["maximum_epochs"],
        "f_epochs": int(f_budget),
        "ema_half_life_epochs": source.get("attention_gru_ema_half_life_epochs"),
        "planned_fits": len(cells) * len(seeds) * (1 + len(folds)),
        "driver": binding(Path(__file__)),
        "runtime": {
            "git": _git_identity(),
            "files": {
                "training": binding(Path(inspect.getfile(train))),
                "model": binding(Path(inspect.getfile(CharacteristicModel))),
                "configuration": binding(Path(inspect.getfile(configuration))),
                "compiler": binding(Path(inspect.getfile(compile_forward))),
            },
        },
        "contrast": (
            "One-factor trajectory contrast against the Stage C fits: identical cells, "
            "store, optimizer, loss and selector population; P patience "
            f"{source['p_recipe']['patience']}->{args.p_patience} with trailing-"
            f"{args.smoothing} smoothed centre-epoch selection; {f_contrast}."
        ),
        "gate": (
            "Primary: paired per-fold, per-seed common-population evaluation IC delta "
            "versus the Stage C control over all executed folds, Newey-West lag 10 "
            "on the pooled daily differences; adopt only if the lower 95% bound "
            "exceeds zero and at least 2/3 of seeds and 2/3 of folds are positive. "
            "Economics (R$10m neutral and flexible-net books) are reported for the "
            "retained recipe only, never used to pick it. No per-fold or per-seed "
            "recipe choice."
        ),
    }
    write_json_atomic(root / "plan.json", plan)
    run[PLAN_KEY] = binding(root / "plan.json")
    write_json_atomic(POINTER, run)
    print(
        json.dumps(
            {"frozen": str(root / "plan.json"), "planned_fits": plan["planned_fits"]}
        )
    )


def execute(run, arms=None):
    torch.set_num_threads(1)
    reference = run[PLAN_KEY]
    plan = bound_json(reference)
    assert plan["driver"]["sha256"] == sha256_file(Path(__file__))
    imported = {
        "training": Path(inspect.getfile(train)),
        "model": Path(inspect.getfile(CharacteristicModel)),
        "configuration": Path(inspect.getfile(configuration)),
        "compiler": Path(inspect.getfile(compile_forward)),
    }
    for name, record in plan["runtime"]["files"].items():
        assert sha256_file(imported[name]) == record["sha256"], name
    selected_arms = set(arms or plan["cells"])
    assert selected_arms <= set(plan["cells"])
    root = Path(reference["path"]).parent
    store = Path(plan["store"]["root"])
    assert sha256_file(store / "manifest.json") == plan["store"]["manifest_sha256"]
    progress = root / "refits.json"
    completed = (
        json.loads(progress.read_text())["completed"] if progress.exists() else []
    )
    done = {r["key"] for r in completed}
    jobs = [
        ("P", "pretrain_internal", seed, arm)
        for seed in plan["seeds"]
        for arm in plan["cells"]
    ]
    jobs += [
        ("F", fold, seed, arm)
        for fold in plan["folds"]
        for seed in plan["seeds"]
        for arm in plan["cells"]
    ]
    for stage, fold, seed, arm in jobs:
        if arm not in selected_arms:
            continue
        key = f"{arm}/{stage}/{fold}/{seed}"
        if key in done:
            bound_json(next(r["manifest"] for r in completed if r["key"] == key))
            continue
        output = (
            root
            / "fits"
            / arm
            / (f"P_seed_{seed}" if stage == "P" else f"{fold}_seed_{seed}")
        )
        parent = root / "fits" / arm / f"P_seed_{seed}" / "selected.pt"
        tick = perf_counter()
        print(
            json.dumps(
                {"starting": key, "completed": len(completed), "planned": len(jobs)}
            ),
            flush=True,
        )
        torch._dynamo.reset()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        train(
            store,
            output,
            cell=plan["cells"][arm],
            stage=stage,
            fold=fold,
            seed=seed,
            epochs=plan["maximum_epochs"] if stage == "P" else plan["f_epochs"],
            recipe=TrainingRecipe(
                **(plan["p_recipe"] if stage == "P" else plan["f_recipes"][arm])
            ),
            parent=parent if stage == "F" else None,
            parent_sha256=sha256_file(parent) if stage == "F" else None,
            compiled=True,
            export_scores=stage == "F",
            ema_half_life_epochs=plan["ema_half_life_epochs"] if stage == "F" else None,
        )
        completed.append(
            {
                "key": key,
                "manifest": binding(output / "run_manifest.json"),
                "seconds": perf_counter() - tick,
            }
        )
        done.add(key)
        write_json_atomic(
            progress,
            {
                "status": "complete" if len(completed) == len(jobs) else "running",
                "plan": reference,
                "completed": completed,
                "planned": len(jobs),
            },
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--cell", action="append", help="Stage C cell name; repeatable")
    parser.add_argument("--seed", type=int, action="append", help="seed; repeatable")
    parser.add_argument(
        "--all-folds", action="store_true", help="every development fold"
    )
    parser.add_argument("--p-patience", type=int, default=20)
    parser.add_argument("--f-epochs", type=int, default=8)
    parser.add_argument(
        "--keep-f-schedule",
        action="store_true",
        help="fallback arm: keep the F schedule and patience, add smoothing only",
    )
    parser.add_argument("--smoothing", type=int, default=3)
    parser.add_argument("--arm", action="append", help="execute only these cells")
    args = parser.parse_args()
    run = json.loads(POINTER.read_text())
    freeze(run, args) if args.freeze else execute(run, args.arm)
