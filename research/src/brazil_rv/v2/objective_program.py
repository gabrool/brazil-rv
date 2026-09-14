"""Matched economic-objective fits with explicit compatible parent adaptation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys

import numpy as np
import torch

from .artifacts import sha256_file, write_json_atomic
from .contract import ALLOWED_SEEDS, DEVELOPMENT_FOLDS, RUN_MANY_PLAN_SCHEMA
from .data_roots import resolve_external_root
from .economic_objective import prepare_targets
from .model import DailyMultiHorizonModel
from .portfolio_program import PROJECT, read
from .portfolio_training import load_data, save_checkpoint
from .post_data_program import CELLS as POST_DATA_CELLS
from .research_rounds import _git_identity
from .round7 import configuration, pretrain_key, SCREEN_FOLDS
from .round7_preprocessing import Round7Preprocessing
from .round7_training import CHECKPOINT_SCHEMA, TrainingRecipe, train
from .train import _verified_checkpoint_input_contract, model_config_contract

CELLS = {
    "TE_all": POST_DATA_CELLS["TE_all"],
    "C6": {
        "cell": "C6",
        "graph": "s0",
        "inputs": "c6",
        "families": ["fundamentals", "magnitudes"],
    },
}
RECIPES = {
    "TE_all": TrainingRecipe(rho=0.2, adaptive=True),
    "C6": TrainingRecipe(rho=0.125),
}


def adapt_c6_parent(source, destination, digest, seed, store_manifest, store_sha):
    """Exact S0 tensors plus zero cold-family projections, with a new contract."""
    if sha256_file(source) != digest:
        raise ValueError("C6 source parent changed")
    parent = torch.load(source, map_location="cpu", weights_only=True)
    original = _verified_checkpoint_input_contract(parent)
    if (
        parent["stage"] != "P"
        or parent["seed"] != seed
        or not parent["transfer_chronology_clean"]
    ):
        raise ValueError("C6 source is not its clean P parent")
    if original["training"]["store"]["manifest_sha256"] != store_sha:
        raise ValueError("C6 parent is bound to another store")
    names = store_manifest["feature_names"]
    if original["training"]["features"]["ordered_slow_names"] != names["slow"]:
        raise ValueError("C6 parent slow coordinates differ")
    config = configuration(CELLS["C6"], names)
    plain = replace(config, sidecar_feature_counts=())
    if original["model_config"] != model_config_contract(plain):
        raise ValueError("C6 source graph differs beyond cold families")
    model = DailyMultiHorizonModel(config)
    state = model.state_dict()
    source_state = parent["model_state_dict"]
    new_keys = {key for key in state if key.startswith("sidecar_projections.")}
    if set(state) - new_keys != set(source_state):
        raise ValueError("C6 parent tensor identity differs")
    for key in new_keys:
        if torch.count_nonzero(state[key]):
            raise ValueError("cold C6 projection is not zero")
    state.update(source_state)
    model.load_state_dict(state, strict=True)
    original_model = DailyMultiHorizonModel(plain)
    original_model.load_state_dict(source_state, strict=True)
    generator = torch.Generator().manual_seed(101)
    values = torch.randn(1, 8, 60, config.slow_feature_count, generator=generator)
    valid = torch.ones_like(values, dtype=torch.bool)
    active = torch.ones(1, 8, dtype=torch.bool)
    history = torch.ones(1, 8, 60, dtype=torch.bool)
    families = {
        name: (
            torch.ones(1, 8, count),
            torch.ones(1, 8, count, dtype=torch.bool),
            torch.zeros(1, 8, count),
        )
        for name, count in config.sidecar_feature_counts
    }
    model.eval()
    original_model.eval()
    with torch.no_grad():
        a = original_model(
            values,
            valid,
            history,
            active,
            slow_feature_age_sessions=torch.zeros_like(values),
        )
        b = model(
            values,
            valid,
            history,
            active,
            slow_feature_age_sessions=torch.zeros_like(values),
            sidecars=families,
        )
    if not torch.equal(a, b):
        raise ValueError("zero-family adaptation changes the inherited predictor")
    contract = {
        "code": _git_identity(),
        "config": asdict(config),
        "pretrain_key": pretrain_key(CELLS["C6"]),
        "store_manifest_sha256": store_sha,
        "preprocessing": Round7Preprocessing({}, feature_names={}).payload(),
        "unexposed_families": list(families),
        "parent_adaptation": {
            "source_sha256": digest,
            "source_input_contract_sha256": original["sha256"],
            "zero_initialized": sorted(new_keys),
            "shared_tensors_exact": True,
            "prediction_max_abs_error": 0.0,
        },
    }
    save_checkpoint(
        destination,
        {
            "schema": CHECKPOINT_SCHEMA,
            "stage": "P",
            "seed": seed,
            "fold": "pretrain_internal",
            "contract": contract,
            "model_state_dict": model.state_dict(),
        },
    )
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "adaptation": contract["parent_adaptation"],
    }


def prepare(root):
    torch.set_num_threads(1)
    frozen = read(root / "frozen_design.json")
    store, resolution = resolve_external_root(frozen["store"]["root"])
    store_sha = sha256_file(store / "manifest.json")
    if store_sha != frozen["store"]["manifest_sha256"]:
        raise ValueError("accepted objective store changed")
    manifest = read(store / "manifest.json")
    sources = read(root / "phase3/parents.json")
    parents = {arm: {} for arm in CELLS}
    for seed in ALLOWED_SEEDS:
        for arm, name in (("S0", "raw_patience.pt"), ("TE_all", "selected.pt")):
            relative = f"parents/{arm}/seed_{seed}/{name}"
            source = root / "phase3" / relative
            digest = sources[relative]["sha256"]
            if sha256_file(source) != digest:
                raise ValueError("recovered parent differs")
            if arm == "S0":
                parents["C6"][str(seed)] = adapt_c6_parent(
                    source,
                    root / "phase3/parents/C6" / f"seed_{seed}/selected.pt",
                    digest,
                    seed,
                    manifest,
                    store_sha,
                )
            else:
                parents[arm][str(seed)] = {"path": str(source), "sha256": digest}
    target = root / "phase3/economic_targets.npz"
    if not target.exists():
        data, binding = load_data(root, "C6")
        prepare_targets(
            root, data, binding, np.load(store / "date_index.npy", allow_pickle=False)
        )
    target_record = read(target.with_suffix(".json"))
    if sha256_file(target) != target_record["sha256"]:
        raise ValueError("economic target cache changed")
    design = {
        "implementation": _git_identity(),
        "store": {"root": str(store), "manifest_sha256": store_sha},
        "store_resolution": resolution.payload(),
        "parents": parents,
        "cells": CELLS,
        "recipes": {k: asdict(v) for k, v in RECIPES.items()},
        "economic_targets": target_record,
        "economic_weight": 0.25,
        "registration_sha256": sha256_file(
            PROJECT / "research/preregistrations/v2_decision_phase3.md"
        ),
        "heldout_accessed": False,
    }
    write_json_atomic(root / "phase3/frozen_design.json", design)
    return design


def fit(root, arm, fold, seed, variant, *, smoke=False):
    design = read(root / "phase3/frozen_design.json")
    if design["implementation"] != _git_identity():
        raise ValueError("objective checkout differs from frozen engineering")
    store, _ = resolve_external_root(design["store"]["root"])
    from .data_roots import resolve_external_file

    parent, _ = resolve_external_file(design["parents"][arm][str(seed)])
    return train(
        store,
        root
        / "phase3"
        / ("smoke" if smoke else "fits")
        / arm
        / variant
        / f"{fold}_seed_{seed}",
        cell=design["cells"][arm],
        stage="F",
        fold=fold,
        seed=seed,
        epochs=1 if smoke else 60,
        recipe=TrainingRecipe(**design["recipes"][arm]),
        parent=parent,
        parent_sha256=design["parents"][arm][str(seed)]["sha256"],
        economic_targets=root / "phase3/economic_targets.npz",
        economic_weight=design["economic_weight"] if variant == "economic" else 0,
        export_scores=not smoke,
    )


def plan(root, *, smoke=False, confirmation=False, max_parallel=2):
    design = read(root / "phase3/frozen_design.json")
    arms = list(CELLS)
    if confirmation:
        arms = read(root / "phase3/screen_summary.json")["survivors"]
    folds = (
        ["F2"]
        if smoke
        else [f for f in DEVELOPMENT_FOLDS if f not in SCREEN_FOLDS]
        if confirmation
        else SCREEN_FOLDS
    )
    jobs = []
    for fold in folds:
        for arm in arms:
            for variant in ("neutral", "economic"):
                for seed in ALLOWED_SEEDS[:1] if smoke else ALLOWED_SEEDS:
                    command = [
                        sys.executable,
                        "-m",
                        "brazil_rv.v2.objective_program",
                        "fit",
                        "--root",
                        str(root),
                        "--arm",
                        arm,
                        "--fold",
                        fold,
                        "--seed",
                        str(seed),
                        "--variant",
                        variant,
                    ]
                    if smoke:
                        command.append("--smoke")
                    jobs.append(
                        {
                            "name": f"{arm}_{variant}_{fold}_{seed}",
                            "seed": seed,
                            "fold": fold,
                            "run_dir": str(
                                root
                                / "phase3"
                                / ("smoke" if smoke else "fits")
                                / arm
                                / variant
                                / f"{fold}_seed_{seed}"
                            ),
                            "cwd": str(PROJECT),
                            "command": command,
                            "resume": True,
                            "expected_manifest": {
                                "contract": {"code": design["implementation"]}
                            },
                        }
                    )
    destination = (
        root
        / "phase3"
        / f"{'smoke' if smoke else 'confirmation' if confirmation else 'screen'}_plan.json"
    )
    write_json_atomic(
        destination,
        {"schema": RUN_MANY_PLAN_SCHEMA, "max_parallel": max_parallel, "jobs": jobs},
    )
    return {"plan": str(destination), "fits": len(jobs)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "fit", "plan"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=CELLS)
    parser.add_argument("--fold", choices=DEVELOPMENT_FOLDS)
    parser.add_argument("--seed", type=int, choices=ALLOWED_SEEDS)
    parser.add_argument("--variant", choices=("neutral", "economic"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=2)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.root)
    elif args.command == "plan":
        print(
            plan(
                args.root,
                smoke=args.smoke,
                confirmation=args.confirmation,
                max_parallel=args.max_parallel,
            )
        )
    else:
        fit(args.root, args.arm, args.fold, args.seed, args.variant, smoke=args.smoke)


if __name__ == "__main__":
    main()
