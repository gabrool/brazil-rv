"""Fit-only RTX engineering: exact cache, precision and real SAM throughput."""

from __future__ import annotations

import argparse
import copy
from functools import partial
from pathlib import Path
import time

import numpy as np
import torch

from .artifacts import sha256_file, write_json_atomic
from .characteristic_model import CharacteristicModel
from .contract import HORIZONS
from .data import V2DailyDataset, stage_name_count
from .economic_objective import EconomicCollator, attach_economic_head
from .model import DailyMultiHorizonModel
from .objective_program import CELLS, RECIPES
from .portfolio_program import read
from .round7 import configuration
from .round7_preprocessing import Round7Preprocessing
from .round7_training import (
    DateTensorCache,
    TrainingObjective,
    autocast_dtype,
    forward,
    model_batch,
    optimizer_step,
    recipe_optimizer,
)
from .train import _cli_stage_indices, compile_forward, set_deterministic_seed


def verify_inputs(root):
    from .portfolio_program import PROJECT

    checkpoint = read(PROJECT / "docs/v2_decision_cpu_checkpoint.json")
    design = read(root / "phase3/frozen_design.json")
    store = Path(design["store"]["root"])
    manifest = read(store / "manifest.json")
    records = {store / "manifest.json": design["store"]["manifest_sha256"]}
    records.update(
        {store / v["path"]: v["sha256"] for v in manifest["arrays"].values()}
    )
    records.update(
        {root / name: digest for name, digest in checkpoint["cache_files"].items()}
    )
    for parents in checkpoint["phase3_parents"].values():
        records.update({Path(v["path"]): v["sha256"] for v in parents.values()})
    records[root / "phase3/economic_targets.npz"] = checkpoint["economic_target_sha256"]
    records.update(
        {
            root / "phase3/mappings" / f"{fold}.json": digest
            for fold, digest in checkpoint["screen_mapping_sha256"].items()
        }
    )
    for path, digest in records.items():
        if sha256_file(path) != digest:
            raise ValueError(f"accepted input changed: {path}")
    result = {
        "passed": True,
        "files_verified": len(records),
        "bindings": {str(p): d for p, d in records.items()},
        "store_end": str(np.load(store / "date_index.npy", allow_pickle=False)[-1]),
        "forward_capture": False,
        "heldout_accessed": False,
        "filesystem_compression": "NTFS lossless; every accepted array hash unchanged",
    }
    write_json_atomic(root / "phase3/local_engineering/input_verification.json", result)
    print("Verified", len(records), "accepted input files", flush=True)
    return result


def check(root, arm, fold):
    torch.set_num_threads(1)
    set_deterministic_seed(11)
    device = torch.device("cuda")
    design = read(root / "phase3/frozen_design.json")
    store = Path(design["store"]["root"])
    if sha256_file(store / "manifest.json") != design["store"]["manifest_sha256"]:
        raise ValueError("engineering store differs")
    parent_path = Path(design["parents"][arm]["11"]["path"])
    if sha256_file(parent_path) != design["parents"][arm]["11"]["sha256"]:
        raise ValueError("engineering parent differs")
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    config = configuration(CELLS[arm], read(store / "manifest.json")["feature_names"])
    characteristic = arm == "TE_all"
    families = tuple(
        n
        for n, _ in (
            config.family_counts if characteristic else config.sidecar_feature_counts
        )
    )
    fit, _, _, window = _cli_stage_indices(store, "F", fold)
    dataset = V2DailyDataset(
        store,
        fit,
        target_window_indices=window,
        stage="finetune",
        purpose="training",
        lookback=60,
        enabled_sidecars=families,
        include_fast=False,
        include_intraday=False,
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
            fit,
            window,
            dataset.store.isins,
        )
        indices = np.linspace(0, len(dataset) - 1, 16, dtype=int)
        cpu = collator([dataset[int(i)] for i in indices])
        cache = DateTensorCache([cpu], fit[indices], len(dataset.store.isins), device)
        canonical = model_batch(cpu, device)
        gathered = cache.gather(list(range(16)))
        if any(not torch.equal(canonical[k], gathered[k]) for k in canonical):
            raise ValueError("factorized cache changes canonical observations")
        del gathered, cache
        torch.cuda.empty_cache()
        model = (
            CharacteristicModel(config)
            if characteristic
            else DailyMultiHorizonModel(config)
        ).to(device)
        model.load_state_dict(parent["model_state_dict"], strict=True)
        attach_economic_head(model)
        model.eval()
        with torch.no_grad():
            reference = forward(model, canonical, characteristic=characteristic).float()
            with torch.autocast("cuda", dtype=autocast_dtype(device)):
                mixed = forward(model, canonical, characteristic=characteristic).float()
        mask = canonical["active_mask"][..., None, None].expand_as(reference)
        error = float(
            (mixed[mask] - reference[mask]).square().mean().sqrt()
            / reference[mask].std().clamp_min(1e-12)
        )
        if not torch.isfinite(mixed).all() or error > 0.08:
            raise ValueError(f"FP16 differs excessively from FP32: {error}")
        original = copy.deepcopy(model.state_dict())
        result = {
            "arm": arm,
            "fold": fold,
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(),
            "dates": fit[indices].tolist(),
            "fit_dates": len(fit),
            "padded_names": width,
            "active_names": canonical["active_mask"].sum(1).tolist(),
            "lookback": 60,
            "cache_exact": True,
            "fp16_error_over_fp32_std": error,
            "cases": {},
            "store_manifest_sha256": design["store"]["manifest_sha256"],
            "parent_sha256": sha256_file(parent_path),
            "heldout_accessed": False,
        }
        destination = root / "phase3/local_engineering" / f"{arm}_{fold}.json"
        write_json_atomic(destination, result)
        for compiled in (False, True):
            model.load_state_dict(original)
            model.train()
            set_deterministic_seed(11)
            optimizer = recipe_optimizer(model, cuda=True, learning_rate=1e-4)
            objective = TrainingObjective(
                model,
                characteristic=characteristic,
                head_indices=[
                    HORIZONS.index(h)
                    for h in (config.horizons if characteristic else HORIZONS)
                ],
                loss_kind="soft_spearman",
                cuda=True,
                economic_weight=0.25,
            )
            call = compile_forward(objective) if compiled else objective
            scaler = torch.amp.GradScaler("cuda", init_scale=256)
            timings, losses, retry_counts = [], [], []
            torch.cuda.reset_peak_memory_stats()
            for step in range(4):
                torch.cuda.synchronize()
                started = time.perf_counter()
                diagnostic = {}
                loss, gap = optimizer_step(
                    model,
                    optimizer,
                    lambda: call(canonical),
                    RECIPES[arm].rho,
                    adaptive=RECIPES[arm].adaptive,
                    scaler=scaler,
                    diagnostics=diagnostic,
                )
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - started)
                losses.append([loss, gap])
                retry_counts.append(diagnostic["loss_scale_retries"])
                print(arm, fold, compiled, step, timings[-1], losses[-1], flush=True)
            result["cases"]["compiled" if compiled else "eager"] = {
                "seconds": timings,
                "loss_and_sam_gap": losses,
                "scale_retries": retry_counts,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            }
            write_json_atomic(destination, result)
        result["passed"] = True
        write_json_atomic(destination, result)
        return result
    finally:
        dataset.store.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=CELLS)
    parser.add_argument("--fold", choices=("F2", "F14"))
    parser.add_argument("--verify-inputs", action="store_true")
    args = parser.parse_args()
    if args.verify_inputs:
        verify_inputs(args.root)
    else:
        check(args.root, args.arm, args.fold)


if __name__ == "__main__":
    main()
