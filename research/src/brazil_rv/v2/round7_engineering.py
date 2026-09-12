"""Real-input, synthetic-label learning and full-graph acceptance; no alpha scores."""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

from .artifacts import sha256_file, write_json_atomic
from .characteristic_model import CharacteristicModel
from .contract import HORIZONS
from .data import V2DailyDataset, stage_name_count
from .model import DailyMultiHorizonModel
from .normalization import midrank_unit_interval
from .round7 import CELLS, configuration
from .round7_preprocessing import Round7Preprocessing
from .round7_training import (
    TrainingObjective,
    daily_primary_ic,
    forward,
    model_batch,
    optimizer_step,
    recipe_optimizer,
)
from .train import _cli_stage_indices, _unique_compiled_graphs, compile_forward


def synthetic_samples(sample):
    """Replace every outcome with a known function of permitted input values."""
    active = sample["active_mask"].astype(bool)
    core = sample["slow_features"][:, -1]
    signal = core[:, 0] + 0.8 * core[:, 1] * core[:, 2] + 0.4 * core[:, 3] ** 2
    target = np.zeros((len(active), 5), np.float32)
    target[active] = midrank_unit_interval(signal[active])[:, None]
    sample["targets"] = target
    sample["target_mask"] = np.broadcast_to(active[:, None], target.shape).copy()
    family_signal = np.where(
        sample["sidecar_magnitudes_valid"][:, 0] & sample["sidecar_oddlot_valid"][:, 0],
        sample["sidecar_magnitudes_values"][:, 0]
        * sample["sidecar_oddlot_values"][:, 0],
        0.0,
    )
    native = sample["sidecar_fundamentals_native_values"][:, 2]
    native_valid = sample["sidecar_fundamentals_native_valid"][:, 2]
    family_signal += np.where(native_valid, 0.5 * (native > 0.2), 0.0)
    family_target = np.zeros_like(target)
    family_target[active] = midrank_unit_interval(family_signal[active])[:, None]
    keep = {
        "schema",
        "date_index",
        "trade_date",
        "name_index",
        "active_mask",
        "slow_features",
        "slow_feature_mask",
        "slow_history_mask",
        "slow_feature_age_sessions",
        "targets",
        "target_mask",
    }
    sample = {
        k: v
        for k, v in sample.items()
        if k in keep or k.startswith(("sidecar_", "common_state_"))
    }
    return (
        {
            k: v
            for k, v in sample.items()
            if not k.startswith(("sidecar_", "common_state_"))
        },
        {**sample, "targets": family_target},
    )


def run(store_root, output, *, device, cells, maximum_steps=120):
    torch.set_num_threads(6)
    manifest = json.loads((store_root / "manifest.json").read_text(encoding="utf-8"))
    all_config = configuration(
        next(c for c in CELLS if c["cell"] == "B4"), manifest["feature_names"]
    )
    engineering_fold = "F14" if device.type == "cuda" else "F2"
    fit, _, _, window = _cli_stage_indices(store_root, "F", engineering_fold)
    dataset = V2DailyDataset(
        store_root,
        fit,
        stage="finetune",
        enabled_sidecars=tuple(n for n, _ in all_config.family_counts),
        include_fast=False,
        include_intraday=False,
        include_common_state=True,
        compact_names=True,
        target_window_indices=window,
        purpose="training",
    )
    try:
        preparation = Round7Preprocessing.fit(dataset, split_common=True)
        dataset.magnitude_clip = preparation.magnitude
        width = stage_name_count(dataset)
        # CUDA acceptance uses a complete 16-date batch, with distinct real dates.
        # CPU keeps the bounded one-date engineering fixture already reported.
        samples = [
            synthetic_samples(dataset[-1 - k])
            for k in range(16 if device.type == "cuda" else 1)
        ]
        results = []
        for cell_name in cells:
            torch.manual_seed(11)
            torch._dynamo.reset()
            torch._dynamo.utils.counters.clear()
            cell = next(c for c in CELLS if c["cell"] == cell_name)
            characteristic = cell["graph"] != "s0"
            config = configuration(cell, manifest["feature_names"])
            inputs = [s[int(cell["inputs"] == "all")] for s in samples]
            prep = (
                preparation
                if characteristic and cell["inputs"] == "all"
                else Round7Preprocessing(
                    preparation.native if cell["inputs"] == "all" else None,
                    None,
                    preparation.magnitude if cell["inputs"] == "all" else None,
                )
            )
            batch = model_batch(prep.collate(inputs, fixed_name_count=width), device)
            model = (
                CharacteristicModel(config)
                if characteristic
                else DailyMultiHorizonModel(config)
            ).to(device)
            horizons = config.horizons if characteristic else HORIZONS
            objective = TrainingObjective(
                model,
                characteristic=characteristic,
                head_indices=[HORIZONS.index(h) for h in horizons],
                loss_kind=cell.get("loss", "soft_spearman"),
                cuda=device.type == "cuda",
            )
            precision = None
            if device.type == "cuda":
                model.eval()
                gradients, losses = [], []
                for bf16 in (False, True):
                    objective.cuda = bf16
                    model.zero_grad(set_to_none=True)
                    loss = objective(batch)
                    loss.backward()
                    losses.append(float(loss.detach()))
                    gradients.append(
                        torch.cat(
                            [
                                p.grad.flatten()
                                for p in model.parameters()
                                if p.grad is not None
                            ]
                        )
                    )
                cosine = torch.nn.functional.cosine_similarity(
                    gradients[0], gradients[1], dim=0
                )
                precision = {
                    "mode": "evaluation-mode loss and gradients; identical fixed inputs/weights",
                    "fp32_loss": losses[0],
                    "bf16_loss": losses[1],
                    "gradient_cosine": float(cosine),
                    "absolute_loss_difference": abs(losses[0] - losses[1]),
                }
                precision["passed"] = (
                    precision["gradient_cosine"] >= 0.99
                    and precision["absolute_loss_difference"] <= 0.01
                )
                if not precision["passed"]:
                    raise RuntimeError(f"BF16 engineering bridge failed: {precision}")
                del gradients, loss
                model.zero_grad(set_to_none=True)
            compile_model = partial(
                compile_forward,
                backend="inductor" if device.type == "cuda" else "eager",
                mode="max-autotune" if device.type == "cuda" else None,
            )
            training = compile_model(objective)
            inference = compile_model(model)
            optimizer = recipe_optimizer(model, cuda=device.type == "cuda")
            started = time.perf_counter()
            training_graphs = evaluation_graphs = 0
            for step in range(1, maximum_steps + 1):
                model.train()
                before = _unique_compiled_graphs()
                optimizer_step(model, optimizer, partial(training, batch), cell["rho"])
                training_graphs += _unique_compiled_graphs() - before
                if step % 10:
                    continue
                model.eval()
                before = _unique_compiled_graphs()
                with (
                    torch.no_grad(),
                    torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=device.type == "cuda",
                    ),
                ):
                    scores = (
                        forward(inference, batch, characteristic=characteristic)
                        .float()
                        .mean(dim=2)
                    )
                evaluation_graphs += _unique_compiled_graphs() - before
                target_heads = [HORIZONS.index(h) for h in horizons]
                ic = float(
                    np.mean(
                        daily_primary_ic(
                            scores.cpu().numpy(),
                            batch["targets"][..., target_heads].cpu().numpy(),
                            batch["target_mask"][..., target_heads].cpu().numpy(),
                            batch["active_mask"].cpu().numpy(),
                        )
                    )
                )
                if ic > 0.5:
                    break
            result = {
                "cell": cell_name,
                "steps": step,
                "synthetic_ic": ic,
                "seconds_including_compile": time.perf_counter() - started,
                "graphs": {
                    "training": training_graphs,
                    "evaluation": evaluation_graphs,
                },
                "parameters": sum(p.numel() for p in model.parameters()),
                "precision_bridge": precision,
                "engineering_fold": engineering_fold,
                "synthetic_target": (
                    "masked magnitudes[0] * oddlot[0] + .5 * (native profitability > .2)"
                    if cell["inputs"] == "all"
                    else "core[0] + .8 * core[1] * core[2] + .4 * core[3]^2"
                ),
                "history_sessions": 60,
                "dates_per_batch": len(inputs),
                "actual_names_by_date": [int(s["active_mask"].sum()) for s in inputs],
                "padded_names": width,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated()
                if device.type == "cuda"
                else None,
            }
            result["passed"] = ic > 0.5 and training_graphs == evaluation_graphs == 1
            results.append(result)
            write_json_atomic(
                output,
                {
                    "schema": "BRAZIL_RV_ROUND7_ENGINEERING_V1",
                    "store_manifest_sha256": sha256_file(store_root / "manifest.json"),
                    "synthetic_labels_only": True,
                    "device": str(device),
                    "compile_backend": "inductor" if device.type == "cuda" else "eager",
                    "results": results,
                },
            )
            print(json.dumps(result), flush=True)
            if not result["passed"]:
                raise RuntimeError("registered engineering acceptance failed")
            del model, objective, training, inference, optimizer, batch
        return results
    finally:
        dataset.store.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--cells",
        nargs="+",
        default=["A1", "A3", "B1", "B4", "B6", "B7", "B8", "B9", "B10", "B11"],
    )
    args = parser.parse_args()
    run(args.store, args.output, device=torch.device(args.device), cells=args.cells)


if __name__ == "__main__":
    main()
