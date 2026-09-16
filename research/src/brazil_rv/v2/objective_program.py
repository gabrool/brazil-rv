"""Matched economic-objective fits with explicit compatible parent adaptation."""

from __future__ import annotations

import argparse
import gc
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
    if destination.exists():
        saved = torch.load(destination, map_location="cpu", weights_only=True)
        if (
            saved["contract"]["parent_adaptation"] != contract["parent_adaptation"]
            or saved["contract"]["config"] != contract["config"]
        ):
            raise ValueError("existing adapted parent has a different contract")
        if any(
            not torch.equal(value, saved["model_state_dict"][key])
            for key, value in model.state_dict().items()
        ):
            raise ValueError("existing adapted parent tensors differ")
        return {
            "path": str(destination),
            "sha256": sha256_file(destination),
            "adaptation": contract["parent_adaptation"],
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
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
    for records in parents.values():
        for record in records.values():
            record["bytes"] = Path(record["path"]).stat().st_size
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
        "execution": {
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "capability": list(torch.cuda.get_device_capability())
            if torch.cuda.is_available()
            else None,
        },
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
    torch.set_num_threads(1)
    design = read(root / "phase3/frozen_design.json")
    if design["implementation"] != _git_identity():
        raise ValueError("objective checkout differs from frozen engineering")
    if design["execution"]["torch"] != str(torch.__version__):
        raise ValueError("objective runtime differs from frozen engineering")
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
        epochs=2 if smoke else 60,
        recipe=TrainingRecipe(**design["recipes"][arm]),
        parent=parent,
        parent_sha256=design["parents"][arm][str(seed)]["sha256"],
        economic_targets=root / "phase3/economic_targets.npz",
        economic_weight=design["economic_weight"] if variant == "economic" else 0,
        export_scores=not smoke,
    )


def run_local(root, *, smoke=False, confirmation=False):
    """One GPU worker; retain verified store hashes and loaded CUDA libraries."""
    plan(root, smoke=smoke, confirmation=confirmation, max_parallel=1)
    kind = "smoke" if smoke else "confirmation" if confirmation else "screen"
    jobs = read(root / "phase3" / f"{kind}_plan.json")["jobs"]
    completed = []
    for job in jobs:
        arguments = job["command"]

        def value(name):
            return arguments[arguments.index(name) + 1]

        # Independent fits retain their own RNG, optimizer, contract and resume
        # files; only import/verification caches survive this boundary.
        torch._dynamo.reset()
        result = fit(
            root,
            value("--arm"),
            value("--fold"),
            int(value("--seed")),
            value("--variant"),
            smoke=smoke,
        )
        completed.append(
            {
                "name": job["name"],
                "selected_epoch": result["selected_epoch"],
                "manifest_sha256": sha256_file(
                    Path(job["run_dir"]) / "run_manifest.json"
                ),
            }
        )
        write_json_atomic(
            root / "phase3" / f"{kind}_local_progress.json",
            {
                "completed": completed,
                "planned": len(jobs),
                "status": "completed" if len(completed) == len(jobs) else "running",
            },
        )
        gc.collect()
        torch.cuda.empty_cache()
    return completed


def plan(root, *, smoke=False, confirmation=False, max_parallel=2):
    design = read(root / "phase3/frozen_design.json")
    if not smoke:
        acceptance = read(root / "phase3/gpu_acceptance.json")
        if not acceptance["passed"] or acceptance["design_sha256"] != sha256_file(
            root / "phase3/frozen_design.json"
        ):
            raise ValueError("financial objective fits require matched GPU acceptance")
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


def accept_smoke(root):
    """Validate disposable compiled CUDA fits before financial dispatch."""
    design = read(root / "phase3/frozen_design.json")
    if not read(root / "phase3/cpu_acceptance.json")["passed"]:
        raise ValueError("fit-only CPU gradient acceptance is missing")
    results = {}
    local = design.get("execution", {}).get("capability", [8])[0] < 8
    if local:
        engineering = root / "phase3/local_engineering"
        if not read(engineering / "input_verification.json")["passed"]:
            raise ValueError("local input verification is missing")
        for arm in CELLS:
            for fold in ("F2", "F14"):
                check = read(engineering / f"{arm}_{fold}.json")
                if (
                    not check["passed"]
                    or not check["cache_exact"]
                    or check["torch"] != design["execution"]["torch"]
                ):
                    raise ValueError("local full-population engineering failed")
    for arm in CELLS:
        matched = []
        for variant in ("neutral", "economic"):
            path = root / "phase3/smoke" / arm / variant / "F2_seed_11"
            record = read(path / "run_manifest.json")
            contract = record["contract"]
            history = read(path / "history.json")
            if (
                record["status"] != "completed"
                or contract["code"] != design["implementation"]
                or not contract["compile"]
                or not contract["date_tensor_cache"]
                or record["peak_cuda_bytes"] <= 0
                or len(history) != 2
                or not all(
                    np.isfinite(
                        [v["training_loss"], v["selection"]["mean_ic"], v["sam_gap"]]
                    ).all()
                    and v["updates"] > 0
                    for v in history
                )
                or bool(contract["economic_auxiliary"]) != (variant == "economic")
                or local
                and (
                    contract["runtime"]["autocast_dtype"] != "torch.float16"
                    or not contract["runtime"]["gradient_scaling"]
                )
            ):
                raise ValueError(
                    f"compiled CUDA objective smoke failed: {arm}/{variant}"
                )
            for name, digest in record["artifacts"].items():
                if sha256_file(path / name) != digest:
                    raise ValueError("GPU smoke artifact changed")
            matched.append(
                (
                    contract["padded_name_count"],
                    contract["fit_target_window"],
                    [v["updates"] for v in history],
                    contract["runtime"],
                )
            )
            results[f"{arm}/{variant}"] = {
                "manifest_sha256": sha256_file(path / "run_manifest.json"),
                "epoch_seconds": [v["seconds"] for v in history],
                "peak_cuda_bytes": record["peak_cuda_bytes"],
                "padded_names": contract["padded_name_count"],
                "compiled_graphs": record["compiled_graphs"],
                "runtime": contract["runtime"],
                "cache": record["date_tensor_cache_resources"],
            }
        if matched[0] != matched[1]:
            raise ValueError(
                "matched objective smoke populations or update counts differ"
            )
    output = {
        "passed": True,
        "design_sha256": sha256_file(root / "phase3/frozen_design.json"),
        "results": results,
        "scope": "two compiled mixed-precision CUDA epochs per arm/objective; CPU gradient and unchanged initialization checked separately",
    }
    write_json_atomic(root / "phase3/gpu_acceptance.json", output)
    return output


def cpu_acceptance(root):
    """Fit-only real-data shape/gradient checks; no forecast experiment or scores."""
    from functools import partial
    import time

    from .characteristic_model import CharacteristicModel
    from .data import V2DailyDataset, stage_name_count
    from .economic_objective import (
        EconomicCollator,
        attach_economic_head,
        economic_loss,
    )
    from .round7_training import TrainingObjective, forward, model_batch
    from .train import _cli_stage_indices

    torch.set_num_threads(2)
    frozen = read(root / "phase3/frozen_design.json")
    store, _ = resolve_external_root(frozen["store"]["root"])
    fit_rows, _, _, fit_window = _cli_stage_indices(store, "F", "F2")
    manifest = read(store / "manifest.json")
    results = {}
    for arm, cell in CELLS.items():
        started = time.monotonic()
        characteristic = arm == "TE_all"
        config = configuration(cell, manifest["feature_names"])
        families = tuple(
            name
            for name, _ in (
                config.family_counts
                if characteristic
                else config.sidecar_feature_counts
            )
        )
        parent_path = root / "phase3/parents" / arm / "seed_11/selected.pt"
        if sha256_file(parent_path) != frozen["parents"][arm]["11"]["sha256"]:
            raise ValueError("CPU acceptance parent changed")
        parent = torch.load(parent_path, map_location="cpu", weights_only=True)
        dataset = V2DailyDataset(
            store,
            fit_rows,
            target_window_indices=fit_window,
            stage="finetune",
            purpose="training",
            lookback=60,
            enabled_sidecars=families,
            include_intraday=False,
            include_fast=False,
            include_common_state=characteristic,
            compact_names=True,
        )
        try:
            preparation = Round7Preprocessing.fit(
                dataset,
                split_common=characteristic,
                parent=Round7Preprocessing.from_payload(
                    parent["contract"]["preprocessing"]
                ),
            )
            width = stage_name_count(dataset)
            collator = EconomicCollator(
                partial(preparation.collate, fixed_name_count=width),
                root / "phase3/economic_targets.npz",
                fit_rows,
                fit_window,
                dataset.store.isins,
            )
            cpu = collator([dataset[0], dataset[len(dataset) - 1]])
            batch = model_batch(cpu, torch.device("cpu"))
            model = (
                CharacteristicModel(config)
                if characteristic
                else DailyMultiHorizonModel(config)
            )
            model.load_state_dict(parent["model_state_dict"], strict=True)
            model.eval()
            with torch.no_grad():
                baseline = forward(model, batch, characteristic=characteristic)
            attach_economic_head(model)
            with torch.no_grad():
                current = forward(model, batch, characteristic=characteristic)
            if not torch.equal(current, baseline):
                raise ValueError("auxiliary attachment changes the neutral predictor")
            from .contract import HORIZONS

            horizons = config.horizons if characteristic else HORIZONS
            objective = TrainingObjective(
                model,
                characteristic=characteristic,
                head_indices=[HORIZONS.index(h) for h in horizons],
                loss_kind="soft_spearman",
                cuda=False,
                economic_weight=0.25,
            )
            objective_value = objective(batch)
            objective_value.backward()
            head_norm = float(model.economic_head.weight.grad.norm())
            with torch.no_grad():
                model.economic_head.weight.add_(
                    model.economic_head.weight.grad, alpha=-0.01
                )
            model.zero_grad(set_to_none=True)
            _, hidden = forward(
                model, batch, characteristic=characteristic, return_hidden=True
            )
            auxiliary = economic_loss(
                model.economic_head(hidden).squeeze(-1),
                batch["economic_target"],
                batch["economic_mask"],
            )
            auxiliary.backward()
            encoder_norm = float(model.slow_input_projection.weight.grad.norm())
            if (
                not np.isfinite(
                    [head_norm, encoder_norm, float(objective_value.detach())]
                ).all()
                or min(head_norm, encoder_norm) <= 0
            ):
                raise ValueError("economic auxiliary did not reach the actual encoder")
            results[arm] = {
                "parent_sha256": sha256_file(parent_path),
                "parameters": sum(p.numel() for p in model.parameters()),
                "probe_date_indices": [int(fit_rows[0]), int(fit_rows[-1])],
                "lookback": 60,
                "padded_names": width,
                "active_names": batch["active_mask"].sum(1).tolist(),
                "economic_label_counts": batch["economic_mask"].sum(1).tolist(),
                "objective_value": float(objective_value.detach()),
                "economic_head_gradient_norm": head_norm,
                "auxiliary_only_encoder_gradient_norm": encoder_norm,
                "neutral_initialization_exact": True,
                "preprocessing": collator.contract,
                "access": dataset.access_ledger.payload(),
                "seconds": time.monotonic() - started,
            }
        finally:
            dataset.store.close()
    output = {
        "passed": True,
        "implementation": _git_identity(),
        "results": results,
        "scope": "two full eligible cross-sections per architecture, F2 fit only; discard probe updates; no selected fit or evaluation scores",
        "gpu_compile_amp_acceptance_still_required": True,
        "heldout_accessed": False,
    }
    write_json_atomic(root / "phase3/cpu_acceptance.json", output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("prepare", "fit", "plan", "cpu-check", "accept-smoke", "run-local"),
    )
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
    elif args.command == "cpu-check":
        cpu_acceptance(args.root)
    elif args.command == "accept-smoke":
        accept_smoke(args.root)
    elif args.command == "run-local":
        run_local(args.root, smoke=args.smoke, confirmation=args.confirmation)
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
