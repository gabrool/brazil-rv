"""Bounded model-input, averaging and component experiments on accepted data."""

from __future__ import annotations

import argparse
import gc
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .artifacts import sha256_file, write_json_atomic
from .contract import ALLOWED_SEEDS, DEVELOPMENT_FOLDS
from .data_roots import resolve_external_root
from .portfolio_program import PROJECT, read
from .post_data_program import CELLS
from .research_rounds import _git_identity
from .round7 import SCREEN_FOLDS, configuration
from .round7_score import score
from .round7_training import TrainingRecipe, train


def input_cells(names):
    families = sorted(
        k.removeprefix("sidecar_") for k in names if k.startswith("sidecar_")
    )
    base = {**CELLS["TE_all"], "excluded_fields": {"options": ["uncovered_call_share"]}}
    return {
        name: {
            **base,
            "cell": name,
            "families": [f for f in families if f not in removed],
        }
        for name, removed in (
            ("TE_full", ()),
            ("TE_no_micro", ("microstructure",)),
            ("TE_no_weak", ("oddlot", "sector", "rebalance")),
        )
    }


def prepare(root):
    pointer = read(PROJECT / "docs/v2_data_inputs.json")
    store, resolution = resolve_external_root(pointer["store"]["root"])
    manifest = read(store / "manifest.json")
    if (
        sha256_file(store / "manifest.json") != pointer["store"]["manifest_sha256"]
        or not pointer["accepted"]
        or manifest["axes"]["date_end"] > "2024-12-30"
    ):
        raise ValueError("foundation requires the accepted development-only store")
    decision = read(PROJECT / "docs/v2_decision_run.json")
    prior, _ = resolve_external_root(decision["root"])
    design = {
        "implementation": _git_identity(),
        "registration_sha256": sha256_file(
            PROJECT / "research/preregistrations/v2_foundation.md"
        ),
        "store": {
            "root": str(store),
            "manifest_sha256": pointer["store"]["manifest_sha256"],
        },
        "store_resolution": resolution.payload(),
        "prior_decision_root": str(prior),
        "prior_design_sha256": sha256_file(prior / "phase3/frozen_design.json"),
        "seeds": list(ALLOWED_SEEDS),
        "screen_folds": list(SCREEN_FOLDS),
        "p_recipe": asdict(TrainingRecipe()),
        "f_recipe": asdict(TrainingRecipe(rho=0.2, adaptive=True)),
        "ema_half_life_epochs": 1.0,
        "runtime": {
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(),
            "precision": "FP16 neural, scaled gradients, FP32 loss/weights",
        },
        "forward_capture": False,
        "heldout_accessed": False,
    }
    cells = input_cells(manifest["feature_names"])
    wave = {
        "name": "input",
        "control": "TE_full",
        "cells": cells,
        "purpose": "matched P+F dataset removal; full cleaned roster control",
        "configurations": {
            name: asdict(configuration(cell, manifest["feature_names"]))
            for name, cell in cells.items()
        },
    }
    if (root / "frozen_design.json").exists():
        if (
            read(root / "frozen_design.json") != design
            or read(root / "waves/input.json") != wave
        ):
            raise ValueError("existing foundation registration differs")
        return design
    write_json_atomic(root / "frozen_design.json", design)
    write_json_atomic(root / "waves/input.json", wave)
    write_json_atomic(
        PROJECT / "docs/v2_foundation_run.json",
        {
            "root": str(root),
            "design_sha256": sha256_file(root / "frozen_design.json"),
            "status": "engineering_and_inventory",
            "financial_source_commit": design["implementation"]["commit"],
            "registration": "research/preregistrations/v2_foundation.md",
        },
    )
    return design


def inventory(root):
    """No labels: date support and exact original neutral F state availability."""
    design = read(root / "frozen_design.json")
    store = Path(design["store"]["root"])
    manifest = read(store / "manifest.json")
    active = np.load(store / "active.npy", mmap_mode="r")
    dates = np.load(store / "date_index.npy", mmap_mode="r")
    fields = {}
    for family, names in manifest["feature_names"].items():
        if not family.startswith("sidecar_"):
            continue
        arrays = {
            suffix: np.load(
                store / manifest["arrays"][family + "_" + suffix]["path"], mmap_mode="r"
            )
            for suffix in ("values", "valid", "age_sessions")
        }
        for i, name in enumerate(names):
            mask = arrays["valid"][..., i] & active
            known = (arrays["age_sessions"][..., i] >= 0) & active
            per_date = mask.sum(1)
            fields[f"{family}:{name}"] = {
                "active_stock_days": int(mask.sum()),
                "distinct_dates": int((per_date > 0).sum()),
                "known_age_without_value_stock_days": int((known & ~mask).sum()),
                "first_date": str(dates[np.flatnonzero(per_date)[0]])
                if mask.any()
                else None,
                "last_date": str(dates[np.flatnonzero(per_date)[-1]])
                if mask.any()
                else None,
                "median_names_on_observed_date": float(
                    np.median(per_date[per_date > 0])
                )
                if mask.any()
                else 0,
                "distinct_dates_by_year": {
                    str(year): int(
                        (
                            (per_date > 0)
                            & (dates.astype("datetime64[Y]").astype(int) + 1970 == year)
                        ).sum()
                    )
                    for year in range(2010, 2025)
                },
            }
    prior = Path(design["prior_decision_root"])
    trajectories = {}
    for arm in ("C6", "TE_all"):
        for fold in DEVELOPMENT_FOLDS:
            for seed in ALLOWED_SEEDS:
                path = prior / "phase3/fits" / arm / "neutral" / f"{fold}_seed_{seed}"
                record = read(path / "run_manifest.json")
                epoch = record["selected_epoch"]
                selected = path / "selected.pt"
                if sha256_file(selected) != record["artifacts"]["selected.pt"]:
                    raise ValueError("sealed neutral checkpoint changed")
                states = []
                for e in range(max(1, epoch - 2), epoch + 1):
                    relative = f"epochs/epoch_{e:03d}.pt"
                    expected = record["artifacts"].get(
                        relative, record["artifacts"].get(relative.replace("/", "\\"))
                    )
                    file = path / relative
                    if not file.exists():
                        states.append({"epoch": e, "available": False})
                        continue
                    digest = sha256_file(file)
                    if digest != expected:
                        raise ValueError("sealed neutral epoch changed")
                    states.append(
                        {
                            "epoch": e,
                            "available": True,
                            "path": str(file),
                            "sha256": digest,
                        }
                    )
                trajectories[f"{arm}/{fold}/{seed}"] = {
                    "selected_epoch": epoch,
                    "fit_manifest_sha256": sha256_file(path / "run_manifest.json"),
                    "selected": {
                        "path": str(selected),
                        "sha256": record["artifacts"]["selected.pt"],
                    },
                    "trailing_states": states,
                    "available": all(s["available"] for s in states),
                }
    output = {
        "design_sha256": sha256_file(root / "frozen_design.json"),
        "active_stock_days": int(active.sum()),
        "maximum_active_names": int(active.sum(1).max()),
        "fields": fields,
        "trajectories": trajectories,
        "labels_accessed": False,
    }
    write_json_atomic(root / "inventory.json", output)
    print(
        {
            "trajectories": len(trajectories),
            "average_available": sum(v["available"] for v in trajectories.values()),
            "active_stock_days": int(active.sum()),
        },
        flush=True,
    )
    return output


def fit_wave(root, wave_name, *, smoke=False):
    torch.set_num_threads(1)
    design = read(root / "frozen_design.json")
    wave = read(root / "waves" / f"{wave_name}.json")
    if _git_identity() != design["implementation"]:
        raise ValueError("run from the frozen foundation source")
    store = Path(design["store"]["root"])
    if sha256_file(store / "manifest.json") != design["store"]["manifest_sha256"]:
        raise ValueError("foundation store changed")
    if not smoke and not read(root / "engineering_acceptance.json")["passed"]:
        raise ValueError("foundation GPU admission has not passed")
    destination = root / ("smoke" if smoke else "fits")
    seeds = design["seeds"][:1] if smoke else design["seeds"]
    folds = ["F2"] if smoke else design["screen_folds"]
    jobs = [
        (name, "P", "pretrain_internal", seed)
        for seed in seeds
        for name in wave["cells"]
    ]
    jobs += [
        (name, "F", fold, seed)
        for fold in folds
        for seed in seeds
        for name in wave["cells"]
    ]
    progress = []
    for name, stage, fold, seed in jobs:
        path = (
            destination
            / name
            / (f"P_seed_{seed}" if stage == "P" else f"{fold}_seed_{seed}")
        )
        parent = destination / name / f"P_seed_{seed}/selected.pt"
        torch._dynamo.reset()
        torch.cuda.reset_peak_memory_stats()
        print(
            {"starting": str(path), "job": len(progress) + 1, "jobs": len(jobs)},
            flush=True,
        )
        result = train(
            store,
            path,
            cell=wave["cells"][name],
            stage=stage,
            fold=fold,
            seed=seed,
            epochs=1 if smoke else 60,
            recipe=TrainingRecipe(**design["p_recipe" if stage == "P" else "f_recipe"]),
            parent=parent if stage == "F" else None,
            parent_sha256=sha256_file(parent) if stage == "F" else None,
            compiled=True,
            export_scores=not smoke and stage == "F",
            ema_half_life_epochs=design["ema_half_life_epochs"]
            if stage == "F"
            else None,
        )
        if not smoke and stage == "F":
            score(
                store,
                path / "selected_ema.pt",
                path / "ema_scores",
                expected_sha256=result["artifacts"]["selected_ema.pt"],
                compiled=True,
                fixed_name_count=result["contract"]["padded_name_count"],
            )
        progress.append(
            {
                "cell": name,
                "stage": stage,
                "fold": fold,
                "seed": seed,
                "manifest_sha256": sha256_file(path / "run_manifest.json"),
                "selected_epoch": result["selected_epoch"],
                "epochs": result["epochs_completed"],
                "seconds": result["seconds_this_process"],
            }
        )
        write_json_atomic(
            root / f"{wave_name}_{'smoke' if smoke else 'training'}_progress.json",
            {
                "status": "complete" if len(progress) == len(jobs) else "running",
                "planned": len(jobs),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "completed": progress,
            },
        )
        gc.collect()
        torch.cuda.empty_cache()
    if smoke:
        evidence = {}
        for name, stage, fold, seed in jobs:
            path = (
                destination
                / name
                / (f"P_seed_{seed}" if stage == "P" else f"{fold}_seed_{seed}")
            )
            record = read(path / "run_manifest.json")
            history = read(path / "history.json")
            if (
                record["epochs_completed"] != 1
                or record["peak_cuda_bytes"] <= 0
                or record["contract"]["runtime"]["autocast_dtype"] != "torch.float16"
                or not all(
                    np.isfinite([h["training_loss"], h["selection"]["mean_ic"]]).all()
                    for h in history
                )
                or stage == "F"
                and "selected_ema.pt" not in record["artifacts"]
            ):
                raise ValueError("foundation full-population GPU admission failed")
            evidence[f"{name}/{stage}"] = {
                "manifest_sha256": sha256_file(path / "run_manifest.json"),
                "peak_cuda_bytes": record["peak_cuda_bytes"],
                "padded_names": record["contract"]["padded_name_count"],
                "seconds": record["seconds_this_process"],
            }
        write_json_atomic(
            root / "engineering_acceptance.json",
            {
                "passed": True,
                "fits": evidence,
                "scope": "one full P epoch and one full F2 epoch per input cell; disposable; no evaluation read",
                "design_sha256": sha256_file(root / "frozen_design.json"),
            },
        )
    return progress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "inventory", "run"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--wave", default="input")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.root)
    elif args.command == "inventory":
        inventory(args.root)
    else:
        fit_wave(args.root, args.wave, smoke=args.smoke)


if __name__ == "__main__":
    main()
