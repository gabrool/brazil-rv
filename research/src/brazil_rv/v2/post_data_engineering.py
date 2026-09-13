"""Bounded real-input numerical and throughput checks for post-data research."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .artifacts import sha256_file, write_json_atomic
from .characteristic_model import CharacteristicModel
from .data import V2DailyDataset, stage_name_count
from .data_roots import resolve_external_root
from .model import DailyMultiHorizonModel
from .post_data_program import CELLS
from .round7 import configuration
from .round7_data import PROJECT
from .round7_preprocessing import Round7Preprocessing
from .round7_training import (
    DateTensorCache,
    TrainingObjective,
    forward,
    model_batch,
    optimizer_step,
    recipe_optimizer,
)
from .train import (
    _cli_stage_indices,
    _unique_compiled_graphs,
    compile_forward,
    set_deterministic_seed,
)


def check(output, *, cuda=False, cache_only=False):
    source_hashes = {
        p.name: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
        if p.name
        in {
            "post_data_engineering.py",
            "round7_training.py",
            "characteristic_model.py",
            "temporal_pathway.py",
            "training_diagnostics.py",
        }
    }
    torch.set_num_threads(6)
    device = torch.device("cuda" if cuda else "cpu")
    pointer = json.loads(
        (PROJECT / "docs/v2_data_inputs.json").read_text(encoding="utf-8")
    )
    root, _ = resolve_external_root(pointer["store"]["root"])
    if sha256_file(root / "manifest.json") != pointer["store"]["manifest_sha256"]:
        raise ValueError("engineering store differs from current acceptance")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    families = tuple(
        n.removeprefix("sidecar_")
        for n in manifest["feature_names"]
        if n.startswith("sidecar_")
    )
    results, parent_preprocessing = [], None
    for stage, fold in (("P", "pretrain_internal"), ("F", "F2"), ("F", "F14")):
        fit, _, _, window = _cli_stage_indices(root, stage, fold)
        data = V2DailyDataset(
            root,
            fit,
            stage="pretrain" if stage == "P" else "finetune",
            target_window_indices=window,
            purpose="training",
            enabled_sidecars=families,
            include_fast=False,
            include_intraday=False,
            include_common_state=True,
            compact_names=True,
            lookback=60,
        )
        try:
            preparation = Round7Preprocessing.fit(
                data, split_common=True, parent=parent_preprocessing
            )
            if stage == "P":
                parent_preprocessing = preparation
            width = stage_name_count(data)
            size = 16 if cuda and (fold == "F14" or cache_only) else 2
            indices = np.linspace(len(data) // 3, 2 * len(data) // 3, size, dtype=int)
            start = time.perf_counter()
            cpu = preparation.collate(
                [data[int(i)] for i in indices], fixed_name_count=width
            )
            batch = model_batch(cpu, device)
            collation_seconds = time.perf_counter() - start
            if cache_only:
                cache = DateTensorCache([cpu], size, device)
                order = list(reversed(range(size)))
                gathered = cache.gather(order)
                for key, value in batch.items():
                    if not torch.equal(gathered[key], value[order]):
                        raise ValueError(f"cached canonical tensor changed: {key}")
                if cuda:
                    torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(50):
                    cache.gather(order)
                if cuda:
                    torch.cuda.synchronize()
                results.append(
                    {
                        "stage": stage,
                        "fold": fold,
                        "dates": fit[indices].tolist(),
                        "padded_names": width,
                        "lookback": 60,
                        "all_model_tensors_exact": True,
                        "cache_bytes": cache.bytes,
                        "canonical_collation_seconds": collation_seconds,
                        "cached_gather_seconds": (time.perf_counter() - start) / 50,
                    }
                )
                del cache, gathered
                continue
            for name in ("S0_common", "C1_all", "TE_slow", "TL_slow"):
                if cuda and fold != "F14":
                    continue
                characteristic = name != "S0_common"
                config = configuration(CELLS[name], manifest["feature_names"])
                horizons = config.horizons if characteristic else (1, 2, 3, 5, 10)
                set_deterministic_seed(11)
                model = (
                    CharacteristicModel(config)
                    if characteristic
                    else DailyMultiHorizonModel(config)
                ).to(device)
                if name.endswith("slow") or name == "S0_common":
                    inputs = {
                        k: v
                        for k, v in batch.items()
                        if not k.startswith(("sidecar_", "common_state_"))
                    }
                else:
                    inputs = batch
                model.eval()
                with torch.no_grad():
                    reference = forward(
                        model, inputs, characteristic=characteristic
                    ).float()
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        mixed = forward(
                            model, inputs, characteristic=characteristic
                        ).float()
                valid = inputs["active_mask"][..., None, None].expand_as(reference)
                error = float(
                    (mixed[valid] - reference[valid]).square().mean().sqrt()
                    / reference[valid].std().clamp_min(1e-12)
                )
                if not torch.isfinite(mixed).all() or error > 0.08:
                    raise FloatingPointError(
                        f"AMP forward discrepancy for {name}: {error}"
                    )
                original = copy.deepcopy(model.state_dict())
                record = {
                    "stage": stage,
                    "fold": fold,
                    "cell": name,
                    "dates": fit[indices].tolist(),
                    "padded_names": width,
                    "active_names": int(inputs["active_mask"].sum()),
                    "lookback": 60,
                    "amp_error_over_fp32_std": error,
                    "collation_seconds": collation_seconds,
                }
                for compiled in (False, True) if cuda else (False,):
                    model.load_state_dict(original)
                    model.train()
                    set_deterministic_seed(11)
                    optimizer = recipe_optimizer(model, cuda=cuda, learning_rate=1e-4)
                    objective = TrainingObjective(
                        model,
                        characteristic=characteristic,
                        head_indices=[(1, 2, 3, 5, 10).index(h) for h in horizons],
                        loss_kind="soft_spearman",
                        cuda=cuda,
                    )
                    callable_objective = (
                        compile_forward(objective) if compiled else objective
                    )
                    timings, losses, graphs = [], [], _unique_compiled_graphs()
                    for step in range(4 if cuda else 1):
                        if cuda:
                            torch.cuda.synchronize()
                        start = time.perf_counter()
                        loss, _ = optimizer_step(
                            model,
                            optimizer,
                            lambda: callable_objective(inputs),
                            0.2,
                            adaptive=True,
                        )
                        if cuda:
                            torch.cuda.synchronize()
                        timings.append(time.perf_counter() - start)
                        losses.append(loss)
                    if (
                        not all(torch.isfinite(p).all() for p in model.parameters())
                        or not np.isfinite(losses).all()
                    ):
                        raise FloatingPointError(
                            "nonfinite full adaptive optimizer step"
                        )
                    record["compiled" if compiled else "eager"] = {
                        "step_seconds": timings,
                        "losses": losses,
                        "graphs": _unique_compiled_graphs() - graphs,
                        "peak_cuda_bytes": torch.cuda.max_memory_allocated()
                        if cuda
                        else 0,
                    }
                results.append(record)
                print(json.dumps(record), flush=True)
        finally:
            data.store.close()
    report = {
        "status": "passed",
        "store": pointer["store"],
        "device": str(device),
        "torch": torch.__version__,
        "checks": results,
        "source_hashes_lf": source_hashes,
        "evaluation_scores_read": False,
        "heldout_access": False,
        "cache_only": cache_only,
    }
    write_json_atomic(output, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    check(args.output, cuda=args.cuda, cache_only=args.cache_only)


if __name__ == "__main__":
    main()
