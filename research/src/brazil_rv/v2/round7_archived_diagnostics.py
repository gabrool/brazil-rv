"""Read archived weights on their original store; never train or select a new model."""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .artifacts import sha256_file, write_json_atomic
from .config import ModelConfig
from .data import V2DailyDataset, collate_v2_daily, stage_name_count
from .model import DailyMultiHorizonModel
from .round7_data import registered_sources
from .round7_training import daily_primary_ic, forward, member_loss, model_batch
from .train import _cli_stage_indices


def paired_mean_se(raw, ema, lags=10):
    difference = np.asarray(ema, float) - np.asarray(raw, float)
    known = np.isfinite(difference)
    count = int(known.sum())
    if count < 2:
        return {"count": count, "mean_ema_minus_raw": None, "newey_west_mean_se": None}
    mean = float(difference[known].mean())
    centered = np.where(known, difference - mean, 0.0)
    variance = float(centered @ centered)
    for lag in range(1, min(lags, len(difference) - 1) + 1):
        variance += (
            2.0 * (1.0 - lag / (lags + 1)) * float(centered[lag:] @ centered[:-lag])
        )
    return {
        "count": count,
        "mean_ema_minus_raw": mean,
        "newey_west_mean_se": float(np.sqrt(max(0.0, variance)) / count),
        "lags_sessions": lags,
    }


def gradient_readout(model, cpu_batch, rho):
    """Deterministic evaluation-mode sharpness and shared head-gradient alignment."""
    model.eval()
    batch = model_batch(cpu_batch, torch.device("cpu"))
    predictions = forward(model, batch, characteristic=False)
    mask = batch["target_mask"] & batch["active_mask"][..., None]
    shared = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and "head" not in name
    ]
    gradients = []
    for h in range(5):
        loss = member_loss(
            predictions[..., h : h + 1],
            batch["targets"][..., h : h + 1],
            mask[..., h : h + 1],
        )
        grad = torch.autograd.grad(loss, shared, retain_graph=True, allow_unused=True)
        gradients.append(
            torch.cat(
                [
                    g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)
                    for p, g in zip(shared, grad, strict=True)
                ]
            )
        )
    stacked = torch.stack(gradients)
    norms = stacked.norm(dim=1)
    cosine = (stacked @ stacked.T) / (norms[:, None] * norms[None]).clamp_min(1e-12)
    first, traded = stacked[:2].mean(0), stacked[2:].mean(0)
    combined = member_loss(predictions, batch["targets"], mask)
    parameters = [p for p in model.parameters() if p.requires_grad]
    all_grad = torch.autograd.grad(combined, parameters, allow_unused=True)
    used = [(p, g) for p, g in zip(parameters, all_grad, strict=True) if g is not None]
    originals = [p.detach().clone() for p, _ in used]
    norm = torch.stack([g.norm().square() for _, g in used]).sum().sqrt()
    try:
        with torch.no_grad():
            for p, g in used:
                p.add_(g * (rho / norm.clamp_min(1e-12)))
            perturbed = member_loss(
                forward(model, batch, characteristic=False), batch["targets"], mask
            )
    finally:
        with torch.no_grad():
            for (p, _), original in zip(used, originals, strict=True):
                p.copy_(original)
    return {
        "mode": "evaluation; dropout disabled; fixed final fit-date cross-section",
        "head_order": [1, 2, 3, 5, 10],
        "shared_gradient_norms": norms.tolist(),
        "cosine_matrix": cosine.tolist(),
        "D1_D2_vs_D3_D5_D10_cosine": float(
            first @ traded / (first.norm() * traded.norm()).clamp_min(1e-12)
        ),
        "loss": float(combined.detach()),
        "sam_gap": float(perturbed - combined.detach()),
        "rho": rho,
    }


def sidecar_jacobian(model, cpu_batch):
    """Exact Frobenius norms of a standardized pre-rank composite Jacobian."""
    model.eval()
    batch = model_batch(cpu_batch, torch.device("cpu"))
    keys = [k for k in batch if k.startswith("sidecar_") and k.endswith("_values")]
    if not keys:
        return {}
    for key in keys:
        batch[key] = batch[key].detach().requires_grad_(True)
    scores = forward(model, batch, characteristic=False)[:, :, 0, 2:5]
    active = batch["active_mask"]
    count = active.sum(dim=1, keepdim=True)[..., None].clamp_min(1)
    centered = scores - (scores * active[..., None]).sum(dim=1, keepdim=True) / count
    variance = (centered.square() * active[..., None]).sum(dim=1, keepdim=True) / count
    composite = (centered / (variance + 1e-8).sqrt()).mean(dim=-1)
    squares = [0.0 for _ in keys]
    for name in torch.nonzero(active[0]).flatten().tolist():
        gradients = torch.autograd.grad(
            composite[0, name],
            [batch[k] for k in keys],
            retain_graph=True,
            allow_unused=True,
        )
        for j, grad in enumerate(gradients):
            if grad is not None:
                valid = batch[keys[j].removesuffix("_values") + "_valid"]
                squares[j] += float((grad * valid).square().sum())
    return {
        "composite": "mean of cross-sectionally standardized raw D3/D5/D10 scores; hard ranks have no useful local derivative",
        "date_index": int(cpu_batch["date_index"][0]),
        "output_names": int(active.sum()),
        "family_frobenius_norm": {
            k.removeprefix("sidecar_").removesuffix("_values"): float(np.sqrt(v))
            for k, v in zip(keys, squares, strict=True)
        },
    }


def run_one(source_root, output, *, arm, fold, seed, store_root=None):
    started = time.perf_counter()
    path = source_root / "trajectories" / arm / f"{fold}_seed_{seed}"
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    if store_root is None:
        store_root, _, _, accepted = registered_sources()
        store_hash = accepted["store"]["manifest_sha256"]
    else:
        store_hash = sha256_file(store_root / "manifest.json")
    if (
        manifest["checkpoint_input_contract"]["training"]["store"]["manifest_sha256"]
        != store_hash
    ):
        raise ValueError("archive checkpoint binds a different store")
    config = ModelConfig(**manifest["model_config"])
    model_list, hashes = [], {}
    for name in ("raw_patience", "final_ema"):
        checkpoint = path / (name + ".pt")
        hashes[name] = sha256_file(checkpoint)
        if hashes[name] != manifest["artifacts"][checkpoint.name]:
            raise ValueError("archived weights differ from the sealed run")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = DailyMultiHorizonModel(config)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        model_list.append(model)
    fit, selection, evaluation, fit_window = _cli_stage_indices(store_root, "F", fold)
    data = {}
    final_fit = None
    for label, rows, target_window in (
        ("fit", fit, fit_window),
        ("selection", selection, selection),
        ("evaluation", evaluation, evaluation),
    ):
        dataset = V2DailyDataset(
            store_root,
            rows,
            stage="finetune",
            enabled_sidecars=tuple(n for n, _ in config.sidecar_feature_counts),
            include_intraday=False,
            include_fast=False,
            compact_names=True,
            target_window_indices=target_window,
            purpose="training" if label == "fit" else label,
        )
        try:
            loader = DataLoader(
                dataset,
                batch_size=16,
                collate_fn=partial(
                    collate_v2_daily, fixed_name_count=stage_name_count(dataset)
                ),
            )
            series = [[], []]
            with torch.no_grad():
                for cpu_batch in loader:
                    batch = model_batch(cpu_batch, torch.device("cpu"))
                    for j, model in enumerate(model_list):
                        prediction = forward(model, batch, characteristic=False)[
                            :, :, 0, 2:5
                        ]
                        series[j].extend(
                            daily_primary_ic(
                                prediction.numpy(),
                                batch["targets"][..., 2:5].numpy(),
                                batch["target_mask"][..., 2:5].numpy(),
                                batch["active_mask"].numpy(),
                            ).tolist()
                        )
            data[label] = {
                "date_indices": rows.tolist(),
                "raw_patience_daily_ic": [
                    x if np.isfinite(x) else None for x in series[0]
                ],
                "final_ema_daily_ic": [
                    x if np.isfinite(x) else None for x in series[1]
                ],
                "raw_patience_mean_ic": float(np.nanmean(series[0])),
                "final_ema_mean_ic": float(np.nanmean(series[1])),
                "paired": paired_mean_se(*series),
            }
            if label == "fit":
                final_fit = collate_v2_daily(
                    [dataset[-1]], fixed_name_count=stage_name_count(dataset)
                )
        finally:
            dataset.store.close()
    result = {
        "schema": "BRAZIL_RV_ROUND7_ARCHIVED_DIAGNOSTICS_V1",
        "arm": arm,
        "fold": fold,
        "seed": seed,
        "store": {"root": str(store_root), "manifest_sha256": store_hash},
        "checkpoint_hashes": hashes,
        "source_manifest_sha256": sha256_file(path / "run_manifest.json"),
        "selected_epoch": manifest["selected_epoch"],
        "epochs_completed": manifest["epochs_completed"],
        "panels": data,
        "head_gradients": gradient_readout(model_list[0], final_fit, 0.125),
        "sidecar_sensitivity": sidecar_jacobian(model_list[0], final_fit),
        "precision": "CPU FP32 inference; both archived states compared in the same mode",
        "epoch_checkpoints": "unavailable in original archive; no trajectory reconstruction",
        "seconds": time.perf_counter() - started,
    }
    write_json_atomic(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--arms", nargs="+", default=["S0", "fundamentals"])
    parser.add_argument("--folds", nargs="+", default=[f"F{i}" for i in range(1, 15)])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        for fold in args.folds:
            for seed in args.seeds:
                output = args.output / f"{arm}_{fold}_{seed}.json"
                if output.exists():
                    continue
                result = run_one(
                    args.source_root,
                    output,
                    arm=arm,
                    fold=fold,
                    seed=seed,
                    store_root=args.store,
                )
                print(
                    json.dumps(
                        {k: result[k] for k in ("arm", "fold", "seed", "seconds")}
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
