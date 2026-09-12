"""Fixed-budget Round-7 recipe, independent member losses and uniform tail weights."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from brazil_rv.modeling.engine import _soft_spearman_group_losses

from .artifacts import sha256_file, write_json_atomic
from .characteristic_model import CharacteristicModel
from .contract import HORIZONS, RAW_PATIENCE_SCHEMA
from .data import V2DailyDataset, stage_name_count
from .model import DailyMultiHorizonModel
from .normalization import average_ranks
from .research_rounds import _git_identity
from .round7 import CELLS, configuration, pretrain_key
from .round7_preprocessing import Round7Preprocessing
from .train import (
    DateBatchSampler,
    _atomic_torch_save,
    _cli_stage_indices,
    _canonical_payload_sha256,
    _model_forward,
    _rng_state,
    _restore_rng,
    _unique_compiled_graphs,
    compile_forward,
    set_deterministic_seed,
)

CHECKPOINT_SCHEMA = "BRAZIL_RV_ROUND7_CHECKPOINT_V1"


def member_loss(scores, targets, mask, *, kind="soft_spearman"):
    """Each date/member/head ranks only securities; then equally average heads."""
    b, n, members, heads = scores.shape
    predictions = scores.permute(0, 2, 1, 3).reshape(b * members, n, heads).float()
    targets = (
        targets[:, None].expand(-1, members, -1, -1).reshape_as(predictions).float()
    )
    mask = mask[:, None].expand(-1, members, -1, -1).reshape_as(predictions).bool()
    if kind == "soft_spearman":
        losses, groups = _soft_spearman_group_losses(predictions, targets, mask)
    elif kind == "pearson_on_ranks":
        count = mask.sum(dim=1)
        groups = count >= 2
        x = (
            predictions
            - (predictions * mask).sum(dim=1, keepdim=True)
            / count[:, None].clamp_min(1)
        ) * mask
        y = (
            targets
            - (targets * mask).sum(dim=1, keepdim=True) / count[:, None].clamp_min(1)
        ) * mask
        denominator = (
            (x.square().sum(dim=1) * y.square().sum(dim=1)).clamp_min(1e-12).sqrt()
        )
        losses = (1.0 - (x * y).sum(dim=1) / denominator) * groups
    else:
        raise ValueError("unknown registered loss")
    return (losses.sum(dim=0) / groups.sum(dim=0).clamp_min(1)).mean()


def recipe_optimizer(model, *, cuda):
    """One peak LR for all layers; LayerNorm and every bias have zero decay."""
    no_decay = {
        id(p)
        for module in model.modules()
        if isinstance(module, nn.LayerNorm)
        for p in module.parameters(recurse=False)
    }
    decay, excluded = [], []
    for name, parameter in model.named_parameters():
        (
            excluded if name.endswith("bias") or id(parameter) in no_decay else decay
        ).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.01},
            {"params": excluded, "weight_decay": 0.0},
        ],
        lr=3e-4,
        fused=cuda,
    )


def learning_rate_fraction(update, total_updates):
    """Five-percent warmup, cosine to five-percent peak at the final update."""
    warmup = max(1, math.ceil(0.05 * total_updates))
    if update < warmup:
        return (update + 1) / warmup
    progress = (update - warmup + 1) / max(1, total_updates - warmup)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))


def optimizer_step(model, optimizer, closure, rho):
    """Exact-restore SAM with reused dropout RNG; None means one-pass AdamW."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    start_rng = _rng_state() if rho is not None else None
    loss = closure()
    loss.backward()
    first_norm = torch.nn.utils.clip_grad_norm_(
        parameters, float("inf"), error_if_nonfinite=True
    )
    if rho is None:
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        return float(loss.detach()), 0.0
    used = [p for p in parameters if p.grad is not None]
    originals = [p.detach().clone() for p in used]
    try:
        with torch.no_grad():
            perturbations = torch._foreach_mul(
                [p.grad for p in used], rho / (first_norm + 1e-12)
            )
            torch._foreach_add_(used, perturbations)
        optimizer.zero_grad(set_to_none=True)
        _restore_rng(start_rng)
        second = closure()
        second.backward()
    finally:
        with torch.no_grad():
            torch._foreach_copy_(used, originals)
    torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), float((second - loss).detach())


class TailAverage:
    def __init__(self):
        self.count = 0
        self.state = {}

    def update(self, model):
        self.count += 1
        for name, value in model.state_dict().items():
            value = value.detach().cpu()
            if name not in self.state:
                self.state[name] = value.clone()
            elif value.is_floating_point():
                self.state[name].add_((value - self.state[name]) / self.count)
            else:
                self.state[name] = value.clone()


def model_batch(cpu_batch, device):
    keys = {
        "slow_features",
        "slow_feature_mask",
        "slow_history_mask",
        "slow_feature_age_sessions",
        "active_mask",
        "common_state_features",
        "common_state_feature_mask",
        "common_state_age_sessions",
        "targets",
        "target_mask",
    }
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in cpu_batch.items()
        if isinstance(value, torch.Tensor)
        and (key in keys or key.startswith("sidecar_"))
    }


def forward(model, batch, *, characteristic):
    if not characteristic:
        return _model_forward(model, batch)[..., : len(HORIZONS)].unsqueeze(2)
    return model(
        batch["slow_features"],
        batch["slow_feature_mask"],
        batch["slow_history_mask"],
        batch["active_mask"],
        slow_feature_age_sessions=batch["slow_feature_age_sessions"],
        sidecars={
            key.removeprefix("sidecar_").removesuffix("_values"): (
                value,
                batch[key.removesuffix("_values") + "_valid"],
                batch[key.removesuffix("_values") + "_age_sessions"],
            )
            for key, value in batch.items()
            if key.startswith("sidecar_") and key.endswith("_values")
        },
        common_state=(
            batch["common_state_features"],
            batch["common_state_feature_mask"],
            batch["common_state_age_sessions"],
        )
        if "common_state_features" in batch
        else None,
    )


def daily_primary_ic(predictions, targets, mask, active):
    """Same equal-head, common-population daily IC as the standing selector."""
    daily = np.full(len(predictions), np.nan)
    for day in range(len(predictions)):
        common = (
            active[day]
            & mask[day].all(axis=1)
            & np.isfinite(predictions[day]).all(axis=1)
            & np.isfinite(targets[day]).all(axis=1)
        )
        if common.sum() < 20:
            continue
        correlations = []
        for h in range(predictions.shape[-1]):
            x, y = (
                average_ranks(v[day, common, h].astype(np.float64))
                for v in (predictions, targets)
            )
            x, y = x - x.mean(), y - y.mean()
            scale = np.linalg.norm(x) * np.linalg.norm(y)
            correlations.append(float(x @ y / scale) if scale > 0 else np.nan)
        daily[day] = np.mean(correlations)
    return daily


class TrainingObjective(nn.Module):
    """Compile model and FP32 ranking loss together, fusing pairwise reductions."""

    def __init__(self, model, *, characteristic, head_indices, loss_kind, cuda):
        super().__init__()
        self.model = model
        self.characteristic = characteristic
        self.head_indices = head_indices
        self.loss_kind = loss_kind
        self.cuda = cuda

    def forward(self, batch):
        with torch.autocast(
            device_type="cuda" if self.cuda else "cpu",
            dtype=torch.bfloat16,
            enabled=self.cuda,
        ):
            scores = forward(self.model, batch, characteristic=self.characteristic)
        return member_loss(
            scores,
            batch["targets"][..., self.head_indices],
            batch["target_mask"][..., self.head_indices]
            & batch["active_mask"][..., None],
            kind=self.loss_kind,
        )


def selection_readout(model, loader, device, *, characteristic, model_horizons):
    model.eval()
    model_heads = [model_horizons.index(h) for h in (3, 5, 10)]
    target_heads = [HORIZONS.index(h) for h in (3, 5, 10)]
    dates, daily = [], []
    with torch.no_grad():
        for cpu_batch in loader:
            batch = model_batch(cpu_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                scores = (
                    forward(model, batch, characteristic=characteristic)
                    .float()
                    .mean(dim=2)[..., model_heads]
                )
            daily.extend(
                daily_primary_ic(
                    scores.cpu().numpy(),
                    batch["targets"][..., target_heads].cpu().numpy(),
                    batch["target_mask"][..., target_heads].cpu().numpy(),
                    batch["active_mask"].cpu().numpy(),
                ).tolist()
            )
            dates.extend(cpu_batch["date_index"].tolist())
    if not np.isfinite(daily).any():
        raise ValueError("selection has no defined common-head IC")
    return {
        "date_indices": dates,
        "daily_ic": [float(x) if np.isfinite(x) else None for x in daily],
        "mean_ic": float(np.nanmean(daily)),
    }


def sequential_batches(count, size=16):
    """Cover all dates in order without a singleton tail that recompiles."""
    return [
        part.tolist()
        for part in np.array_split(np.arange(count), math.ceil(count / size))
    ]


def train(
    store_root,
    output,
    *,
    cell_name,
    stage,
    fold,
    seed,
    epochs,
    parent=None,
    parent_sha256=None,
    device=None,
    compiled=True,
    calibration=False,
    export_scores=False,
):
    code = _git_identity()
    cell = next(c for c in CELLS if c["cell"] == cell_name)
    if cell["recipe"] != "R":
        raise ValueError("A0 uses the unchanged Round-6 training path")
    if stage == "P":
        if pretrain_key(cell) == "s0_slow":
            raise ValueError("S0-slow Stage P must use the registered old recipe")
        epochs, rho, loss_kind = 60, 0.05, "soft_spearman"
    else:
        rho, loss_kind = cell["rho"], cell.get("loss", "soft_spearman")
    if calibration and (stage != "F" or cell_name != "B4" or epochs != 60):
        raise ValueError("calibration is the fixed 60-epoch B4 experiment")
    manifest = json.loads((store_root / "manifest.json").read_text(encoding="utf-8"))
    if "round7_repair" not in manifest["metadata"]:
        raise ValueError("Round-7 training requires the repaired store")
    config = configuration(cell, manifest["feature_names"])
    characteristic = cell["graph"] != "s0"
    horizons = config.horizons if characteristic else HORIZONS
    families = tuple(
        n
        for n, _ in (
            config.family_counts if characteristic else config.sidecar_feature_counts
        )
    )
    fit, select, _, fit_window = _cli_stage_indices(store_root, stage, fold)
    split_common = characteristic and bool(families)
    options = dict(
        stage="pretrain" if stage == "P" else "finetune",
        lookback=60,
        enabled_sidecars=families,
        include_intraday=False,
        include_fast=False,
        include_common_state=split_common,
        compact_names=True,
    )
    training = V2DailyDataset(
        store_root, fit, target_window_indices=fit_window, purpose="training", **options
    )
    selection = V2DailyDataset(store_root, select, purpose="selection", **options)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        preparation = Round7Preprocessing.fit(training, split_common=split_common)
        training.magnitude_clip = selection.magnitude_clip = preparation.magnitude
        width = stage_name_count(training, selection)
        collator = partial(preparation.collate, fixed_name_count=width)
        sampler = DateBatchSampler(fit, seed=seed)
        train_loader = DataLoader(
            training,
            batch_sampler=sampler,
            collate_fn=collator,
            pin_memory=device.type == "cuda",
        )
        selection_loader = DataLoader(
            selection,
            batch_sampler=sequential_batches(len(selection)),
            collate_fn=collator,
            pin_memory=device.type == "cuda",
        )
        clean_fit_loader = DataLoader(
            training,
            batch_sampler=sequential_batches(len(training)),
            collate_fn=collator,
            pin_memory=device.type == "cuda",
        )
        contract = {
            "code": code,
            "store_manifest_sha256": sha256_file(store_root / "manifest.json"),
            "config": asdict(config),
            "pretrain_key": pretrain_key(cell),
            "stage": stage,
            "fold": fold,
            "seed": seed,
            "epochs": epochs,
            "rho": rho,
            "loss": loss_kind,
            "preprocessing": preparation.payload(),
            "fit_target_window": fit_window.tolist(),
            "access": {
                "training": training.access_ledger.payload(),
                "selection": selection.access_ledger.payload(),
            },
            "padded_name_count": width,
            "compile": compiled,
            "parent_sha256": parent_sha256,
        }
        set_deterministic_seed(seed)
        model = (
            CharacteristicModel(config)
            if characteristic
            else DailyMultiHorizonModel(config)
        ).to(device)
        if stage == "F":
            if parent is None or parent_sha256 != sha256_file(parent):
                raise ValueError("F requires a hash-bound compatible Stage-P parent")
            payload = torch.load(parent, map_location="cpu", weights_only=True)
            if payload["stage"] != "P" or payload["seed"] != seed:
                raise ValueError("parent stage/seed differs")
            if payload.get("schema") == CHECKPOINT_SCHEMA:
                parent_contract = payload["contract"]
                if (
                    parent_contract["pretrain_key"] != pretrain_key(cell)
                    or parent_contract["store_manifest_sha256"]
                    != contract["store_manifest_sha256"]
                ):
                    raise ValueError("parent graph/input/store contract differs")
            else:
                # Only S0-slow inherits the explicitly registered old-recipe P.
                if (
                    pretrain_key(cell) != "s0_slow"
                    or payload.get("schema") != RAW_PATIENCE_SCHEMA
                    or payload.get("transfer_chronology_clean") is not True
                    or payload["input_contract"]["training"]["store"]["manifest_sha256"]
                    != contract["store_manifest_sha256"]
                ):
                    raise ValueError("old-recipe parent is not repaired-store S0")
            model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer = recipe_optimizer(model, cuda=device.type == "cuda")
        tail = TailAverage()
        start_epoch, history = 1, []
        previous_compilation_sessions = []
        resume_path = output / "resume.pt"
        if (output / "run_manifest.json").exists():
            finished = json.loads(
                (output / "run_manifest.json").read_text(encoding="utf-8")
            )
            if _canonical_payload_sha256(
                finished["contract"]
            ) != _canonical_payload_sha256(contract):
                raise ValueError("completed fit differs from requested contract")
            for name, digest in finished["artifacts"].items():
                if sha256_file(output / name) != digest:
                    raise ValueError(f"completed fit artifact changed: {name}")
            if (
                finished.get("score_manifest_sha256")
                and sha256_file(output / "scores/score_manifest.json")
                != finished["score_manifest_sha256"]
            ):
                raise ValueError("completed score manifest changed")
            return finished
        if resume_path.exists():
            payload = torch.load(resume_path, map_location=device, weights_only=True)
            if _canonical_payload_sha256(
                payload["contract"]
            ) != _canonical_payload_sha256(contract):
                raise ValueError("resume differs from the frozen training contract")
            model.load_state_dict(payload["model_state_dict"])
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            tail.state = {k: v.cpu() for k, v in payload["tail_state"].items()}
            tail.count, history, start_epoch = (
                payload["tail_count"],
                payload["history"],
                payload["epoch"] + 1,
            )
            checkpoint = {
                k: payload[k]
                for k in (
                    "schema",
                    "stage",
                    "seed",
                    "fold",
                    "epoch",
                    "contract",
                    "model_state_dict",
                )
            }
            previous_compilation_sessions = payload.get("compilation_sessions", [])
            _restore_rng(
                (
                    payload["cpu_rng"].cpu(),
                    [v.cpu() for v in payload["cuda_rng"]]
                    if payload["cuda_rng"] is not None
                    else None,
                )
            )
        else:
            # run_many creates only its two log files before starting the child.
            if output.exists() and any(
                p.name not in {"launcher.stdout.log", "launcher.stderr.log"}
                for p in output.iterdir()
            ):
                raise ValueError("incomplete fit has no resumable epoch checkpoint")
            output.mkdir(parents=True, exist_ok=True)
            (output / "epochs").mkdir()
        objective = TrainingObjective(
            model,
            characteristic=characteristic,
            head_indices=[HORIZONS.index(h) for h in horizons],
            loss_kind=loss_kind,
            cuda=device.type == "cuda",
        )
        training_forward = compile_forward(objective) if compiled else objective
        forward_model = compile_forward(model) if compiled else model
        total_updates = epochs * len(train_loader)
        train_graphs = selection_graphs = 0
        run_start = time.perf_counter()
        for epoch in range(start_epoch, epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            sampler.set_epoch(epoch)
            losses, gaps = [], []
            for batch_number, cpu_batch in enumerate(train_loader):
                lr = 3e-4 * learning_rate_fraction(
                    (epoch - 1) * len(train_loader) + batch_number, total_updates
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                batch = model_batch(cpu_batch, device)

                before = _unique_compiled_graphs()
                loss, gap = optimizer_step(
                    model, optimizer, lambda: training_forward(batch), rho
                )
                train_graphs += _unique_compiled_graphs() - before
                losses.append(loss)
                gaps.append(gap)
            before = _unique_compiled_graphs()
            readout = selection_readout(
                forward_model,
                selection_loader,
                device,
                characteristic=characteristic,
                model_horizons=horizons,
            )
            fit_readout = None
            if calibration or epoch == epochs:
                # Diagnostics must not shift future dropout RNG or date visits.
                rng = _rng_state()
                try:
                    fit_readout = selection_readout(
                        forward_model,
                        clean_fit_loader,
                        device,
                        characteristic=characteristic,
                        model_horizons=horizons,
                    )
                finally:
                    _restore_rng(rng)
            selection_graphs += _unique_compiled_graphs() - before
            if epoch > epochs - math.ceil(epochs / 4):
                tail.update(model)
            record = {
                "epoch": epoch,
                "training_loss": float(np.mean(losses)),
                "sam_gap": float(np.mean(gaps)),
                "selection": readout,
                "clean_fit": fit_readout,
                "updates": len(losses),
                "seconds": time.perf_counter() - epoch_start,
                "final_learning_rate": lr,
            }
            history.append(record)
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            checkpoint = {
                "schema": CHECKPOINT_SCHEMA,
                "stage": stage,
                "seed": seed,
                "fold": fold,
                "epoch": epoch,
                "contract": contract,
                "model_state_dict": state,
            }
            _atomic_torch_save(output / "epochs" / f"epoch_{epoch:03d}.pt", checkpoint)
            rng, cuda_rng = _rng_state()
            _atomic_torch_save(
                resume_path,
                {
                    **checkpoint,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "tail_count": tail.count,
                    "tail_state": tail.state,
                    "history": history,
                    "cpu_rng": rng,
                    "cuda_rng": cuda_rng,
                    "compilation_sessions": [
                        *previous_compilation_sessions,
                        {
                            "start_epoch": start_epoch,
                            "end_epoch": epoch,
                            "training": train_graphs,
                            "selection": selection_graphs,
                        },
                    ],
                },
            )
            write_json_atomic(output / "history.json", history)
            print(
                json.dumps(
                    {
                        "cell": cell_name,
                        "stage": stage,
                        "fold": fold,
                        "seed": seed,
                        "epoch": epoch,
                        "budget": epochs,
                        "selection_ic": readout["mean_ic"],
                        "seconds": record["seconds"],
                    }
                ),
                flush=True,
            )
        _atomic_torch_save(
            output / "tail_average.pt",
            {
                **checkpoint,
                "model_state_dict": tail.state,
                "tail_epochs": math.ceil(epochs / 4),
                "tail_count": tail.count,
            },
        )
        report = {
            "schema": "BRAZIL_RV_ROUND7_FIT_V1",
            "status": "completed",
            "seed": seed,
            "fold": fold,
            "stage": stage,
            "contract": contract,
            "epochs_completed": epochs,
            "compiled_graphs": {
                "training": max(
                    [
                        train_graphs,
                        *(s["training"] for s in previous_compilation_sessions),
                    ]
                ),
                "selection": max(
                    [
                        selection_graphs,
                        *(s["selection"] for s in previous_compilation_sessions),
                    ]
                ),
            },
            "compiled_graph_count_rule": "maximum per process; complete session records retained",
            "compilation_sessions": [
                *previous_compilation_sessions,
                {
                    "start_epoch": start_epoch,
                    "end_epoch": epochs,
                    "training": train_graphs,
                    "selection": selection_graphs,
                },
            ],
            "seconds_this_process": time.perf_counter() - run_start,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated()
            if device.type == "cuda"
            else 0,
            "artifacts": {
                str(p.relative_to(output)): sha256_file(p)
                for p in [
                    output / "tail_average.pt",
                    output / "history.json",
                    *sorted((output / "epochs").glob("*.pt")),
                ]
            },
        }
        if export_scores:
            from .round7_score import score

            score(
                store_root,
                output / "tail_average.pt",
                output / "scores",
                expected_sha256=report["artifacts"]["tail_average.pt"],
                compiled=compiled,
                device=device,
                reusable_models=(model, forward_model),
                fixed_name_count=width,
            )
            report["score_manifest_sha256"] = sha256_file(
                output / "scores/score_manifest.json"
            )
        write_json_atomic(output / "run_manifest.json", report)
        # A sealed fit resumes from its hash-checked artifacts. The large transient
        # optimizer/RNG checkpoint is useful only while training is incomplete.
        resume_path.unlink(missing_ok=True)
        return report
    finally:
        training.store.close()
        selection.store.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cell",
        choices=[c["cell"] for c in CELLS if c["recipe"] == "R"],
        required=True,
    )
    parser.add_argument("--stage", choices=("P", "F"), required=True)
    parser.add_argument("--fold", default="pretrain_internal")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--parent-sha256")
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--export-scores", action="store_true")
    args = parser.parse_args()
    train(
        args.store,
        args.output,
        cell_name=args.cell,
        stage=args.stage,
        fold=args.fold,
        seed=args.seed,
        epochs=args.epochs,
        parent=args.parent,
        parent_sha256=args.parent_sha256,
        calibration=args.calibration,
        export_scores=args.export_scores,
    )


if __name__ == "__main__":
    main()
