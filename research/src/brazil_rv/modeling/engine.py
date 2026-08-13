from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .contract import (
    GH200_RUNTIME,
    GRADIENT_CLIP,
    HUBER_DELTA,
    NEURAL_OBJECTIVES,
    SAM_NORM_EPS,
    SAM_RHOS,
    SOFT_RANK_STANDARDIZATION_EPS,
    SOFT_RANK_TEMPERATURES,
    SOFT_SPEARMAN_CORRELATION_EPS,
    NeuralArchitecture,
    RuntimeSettings,
    TCNArchitecture,
    TCNSettings,
    context_routing_metadata,
    peer_feature_metadata,
)
from .metrics import create_metric_table


@dataclass(frozen=True)
class EvaluationObservations:
    sample_id: np.ndarray
    predictions: np.ndarray
    targets: np.ndarray
    raw_returns: np.ndarray
    label_mask: np.ndarray
    date_idx: np.ndarray
    decision_idx: np.ndarray


def compile_model(
    model: nn.Module, runtime: RuntimeSettings = GH200_RUNTIME
) -> nn.Module:
    return torch.compile(
        model,
        backend=runtime.compile_backend,
        mode=runtime.compile_mode,
        fullgraph=runtime.compile_fullgraph,
        dynamic=runtime.compile_dynamic,
    )


def experiment_decimal(value: float, minimum_fraction_digits: int) -> str:
    whole, fraction = f"{value:.10f}".rstrip("0").split(".")
    return f"{whole}p{fraction.ljust(minimum_fraction_digits, '0')}"


def validate_neural_objective(
    objective: str, temperature: float | None
) -> tuple[str, float | None]:
    if objective not in NEURAL_OBJECTIVES:
        raise ValueError(f"Unknown objective: {objective}")
    if objective == "soft_spearman":
        if temperature not in SOFT_RANK_TEMPERATURES:
            raise ValueError(f"Temperature must be one of {SOFT_RANK_TEMPERATURES}")
    elif temperature is not None:
        raise ValueError("rank_huber does not accept a temperature")
    return objective, temperature


def objective_metadata(objective: str, temperature: float | None) -> dict[str, object]:
    objective, temperature = validate_neural_objective(objective, temperature)
    return {"name": objective, "temperature": temperature}


def sam_metadata(optimizer_variant: str, rho: float | None) -> dict[str, object] | None:
    if optimizer_variant == "adamw":
        if rho is not None:
            raise ValueError("AdamW does not accept SAM rho")
        return None
    if optimizer_variant != "sam_adamw" or rho not in SAM_RHOS:
        raise ValueError(f"SAM rho must be one of {SAM_RHOS}")
    return {
        "rho": rho,
        "norm": "l2",
        "adaptive": False,
        "same_batch_replay": True,
        "same_rng_replay": True,
    }


def _soft_spearman_loss_sum(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    label_mask: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    validate_neural_objective("soft_spearman", temperature)
    with torch.autocast(device_type=predictions.device.type, enabled=False):
        scores = predictions.float().transpose(1, 2)
        target_ranks = targets.float().transpose(1, 2)
        mask = label_mask.bool().transpose(1, 2)
        counts = mask.sum(-1)
        valid_groups = counts >= 2
        safe_counts = counts.clamp_min(1)
        centered = (
            scores - (scores * mask).sum(-1).div(safe_counts).unsqueeze(-1)
        ) * mask
        variance = centered.square().sum(-1) / safe_counts
        standardized = centered / torch.sqrt(
            variance.unsqueeze(-1) + SOFT_RANK_STANDARDIZATION_EPS
        )
        equity_count = scores.shape[-1]
        not_self = ~torch.eye(
            equity_count, dtype=torch.bool, device=scores.device
        ).reshape(1, 1, equity_count, equity_count)
        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2) & not_self
        pairwise = (
            standardized.unsqueeze(-1) - standardized.unsqueeze(-2)
        ) / temperature
        soft_ranks = (1 + (torch.sigmoid(pairwise) * pair_mask).sum(-1)) * mask
        soft_centered = (
            soft_ranks - soft_ranks.sum(-1).div(safe_counts).unsqueeze(-1)
        ) * mask
        target_centered = (
            target_ranks - (target_ranks * mask).sum(-1).div(safe_counts).unsqueeze(-1)
        ) * mask
        covariance = (soft_centered * target_centered).sum(-1)
        denominator = (
            (soft_centered.square().sum(-1) * target_centered.square().sum(-1))
            .clamp_min(SOFT_SPEARMAN_CORRELATION_EPS)
            .sqrt()
        )
        losses = (1 - covariance / denominator) * valid_groups
        return losses.sum(), valid_groups.sum()


def soft_spearman_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    label_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    total, count = _soft_spearman_loss_sum(
        predictions, targets, label_mask, temperature
    )
    return total / count.clamp_min(1)


def _rank_huber_loss_sum(
    predictions: torch.Tensor, targets: torch.Tensor, label_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    difference = predictions.float() - targets.float()
    absolute = difference.abs()
    values = torch.where(
        absolute <= HUBER_DELTA,
        0.5 * difference.square(),
        HUBER_DELTA * (absolute - 0.5 * HUBER_DELTA),
    )
    mask = label_mask.bool()
    counts = mask.sum(1)
    valid_horizons = counts > 0
    horizon_loss = (values * mask).sum(1) / counts.clamp_min(1)
    horizon_counts = valid_horizons.sum(1)
    valid_samples = horizon_counts > 0
    sample_loss = (horizon_loss * valid_horizons).sum(1) / horizon_counts.clamp_min(1)
    return sample_loss[valid_samples].sum(), valid_samples.sum()


def rank_huber_loss(
    predictions: torch.Tensor, targets: torch.Tensor, label_mask: torch.Tensor
) -> torch.Tensor:
    total, count = _rank_huber_loss_sum(predictions, targets, label_mask)
    return total / count.clamp_min(1)


def _objective_loss_sum(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    label_mask: torch.Tensor,
    objective: str,
    temperature: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    objective, temperature = validate_neural_objective(objective, temperature)
    if objective == "soft_spearman":
        assert temperature is not None
        return _soft_spearman_loss_sum(predictions, targets, label_mask, temperature)
    return _rank_huber_loss_sum(predictions, targets, label_mask)


def objective_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    label_mask: torch.Tensor,
    objective: str,
    temperature: float | None,
) -> torch.Tensor:
    total, count = _objective_loss_sum(
        predictions, targets, label_mask, objective, temperature
    )
    return total / count.clamp_min(1)


def _model_transfer_keys(batch: dict[str, torch.Tensor]) -> tuple[str, ...]:
    if "tabular_features" in batch:
        keys = ("tabular_features", "equity_mask")
    else:
        keys = (
            "patches",
            "history_patch_mask",
            "instrument_mask",
            "slow_features",
            "state_position",
        )
    if "peer_state" in batch:
        keys = (*keys, "peer_state")
    return (*keys, "targets", "label_mask")


def _to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device, non_blocking=device.type == "cuda")
        for key in _model_transfer_keys(batch)
    }


def _predict(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if "tabular_features" in batch:
        return model(batch["tabular_features"], batch["equity_mask"])
    arguments = [
        batch[name]
        for name in (
            "patches",
            "history_patch_mask",
            "instrument_mask",
            "slow_features",
            "state_position",
        )
    ]
    if "peer_state" in batch:
        arguments.append(batch["peer_state"])
    return model(*arguments)


def _autocast(device: torch.device):
    return torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    )


def _loss_count(effective_batch: list[dict[str, torch.Tensor]], objective: str) -> int:
    if objective == "soft_spearman":
        return sum(
            int((batch["label_mask"].sum(1) >= 2).sum()) for batch in effective_batch
        )
    return sum(
        int(batch["label_mask"].any(dim=(1, 2)).sum()) for batch in effective_batch
    )


def _accumulate_gradients(
    model: nn.Module,
    batches: list[dict[str, torch.Tensor]],
    objective: str,
    temperature: float | None,
    count: int,
) -> torch.Tensor:
    total = next(model.parameters()).new_zeros((), dtype=torch.float32)
    for batch in batches:
        with _autocast(next(model.parameters()).device):
            predictions = _predict(model, batch)
        loss_sum, _ = _objective_loss_sum(
            predictions, batch["targets"], batch["label_mask"], objective, temperature
        )
        if not torch.isfinite(loss_sum):
            raise FloatingPointError("Training loss is non-finite")
        total += loss_sum.detach()
        (loss_sum / count).backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    if not gradients or not all(
        bool(torch.isfinite(value).all()) for value in gradients
    ):
        raise FloatingPointError("Training gradients are non-finite")
    return total


def _optimizer_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> torch.Tensor:
    try:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        if not torch.isfinite(norm):
            raise FloatingPointError("Gradient norm is non-finite")
        optimizer.step()
        if not all(
            bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
        ):
            raise FloatingPointError("Optimizer produced non-finite parameters")
        if scheduler is not None:
            scheduler.step()
        return norm.detach()
    finally:
        optimizer.zero_grad(set_to_none=True)


def _rng_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    return torch.get_rng_state(), torch.cuda.get_rng_state(
        device
    ) if device.type == "cuda" else None


def _restore_rng(
    state: tuple[torch.Tensor, torch.Tensor | None], device: torch.device
) -> None:
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state(state[1], device)


def _run_sam_update(
    model: nn.Module,
    batches: list[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    objective: str,
    temperature: float | None,
    rho: float,
    count: int,
    observer: Callable[[str, nn.Module], None] | None = None,
) -> dict[str, float | int | bool]:
    parameters = tuple(model.parameters())
    originals = [parameter.detach().clone() for parameter in parameters]
    device = parameters[0].device
    start_rng = _rng_state(device)
    optimizer.zero_grad(set_to_none=True)
    try:
        first_loss = _accumulate_gradients(
            model, batches, objective, temperature, count
        )
        first_end_rng = _rng_state(device)
        first_norm = torch.sqrt(
            sum(
                parameter.grad.detach().float().square().sum()
                for parameter in parameters
                if parameter.grad is not None
            )
        )
        if observer:
            observer("first_gradients", model)
        scale = rho / (first_norm + SAM_NORM_EPS)
        with torch.no_grad():
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.add_(parameter.grad.detach().float() * scale)
        if observer:
            observer("perturbed_parameters", model)
        optimizer.zero_grad(set_to_none=True)
        _restore_rng(start_rng, device)
        try:
            _accumulate_gradients(model, batches, objective, temperature, count)
            second_end_rng = _rng_state(device)
            second_norm = torch.sqrt(
                sum(
                    parameter.grad.detach().float().square().sum()
                    for parameter in parameters
                    if parameter.grad is not None
                )
            )
        finally:
            with torch.no_grad():
                for parameter, original in zip(parameters, originals, strict=True):
                    parameter.copy_(original)
        if not all(
            torch.equal(parameter, original)
            for parameter, original in zip(parameters, originals, strict=True)
        ):
            raise RuntimeError("SAM parameter restoration was not exact")
        if not torch.equal(first_end_rng[0], second_end_rng[0]) or (
            first_end_rng[1] is not None
            and not torch.equal(first_end_rng[1], second_end_rng[1])
        ):
            raise RuntimeError("SAM RNG replay diverged")
        if observer:
            observer("second_gradients", model)
        clipped = _optimizer_update(model, optimizer, scheduler)
        if observer:
            observer("updated_parameters", model)
        return {
            "loss_sum": float(first_loss),
            "loss_count": count,
            "gradient_norm": float(clipped),
            "first_pass_gradient_norm": float(first_norm),
            "second_pass_gradient_norm": float(second_norm),
            "perturbation_norm": rho,
            "backward_passes": 2 * len(batches),
            "rng_replay_exact": True,
            "all_finite": True,
        }
    except BaseException:
        with torch.no_grad():
            for parameter, original in zip(parameters, originals, strict=True):
                parameter.copy_(original)
        optimizer.zero_grad(set_to_none=True)
        raise


def run_effective_batch_update(
    model: nn.Module,
    effective_batch: list[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    runtime: RuntimeSettings,
    optimizer_variant: str,
    objective: str,
    temperature: float | None,
    sam_rho: float | None,
    *,
    sam_observer: Callable[[str, nn.Module], None] | None = None,
) -> dict[str, object]:
    if len(effective_batch) != runtime.accumulation_steps:
        raise ValueError(f"Expected {runtime.accumulation_steps} microbatches")
    objective_metadata(objective, temperature)
    sam_metadata(optimizer_variant, sam_rho)
    device = next(model.parameters()).device
    batches = [_to_device(batch, device) for batch in effective_batch]
    count = _loss_count(batches, objective)
    if not count:
        raise ValueError("Effective batch has no valid objective group")
    if optimizer_variant == "sam_adamw":
        assert sam_rho is not None
        return _run_sam_update(
            model,
            batches,
            optimizer,
            scheduler,
            objective,
            temperature,
            sam_rho,
            count,
            sam_observer,
        )
    optimizer.zero_grad(set_to_none=True)
    total = _accumulate_gradients(model, batches, objective, temperature, count)
    norm = _optimizer_update(model, optimizer, scheduler)
    return {
        "loss_sum": float(total),
        "loss_count": count,
        "gradient_norm": float(norm),
        "backward_passes": len(batches),
        "all_finite": True,
    }


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    runtime: RuntimeSettings,
    optimizer_variant: str,
    objective: str,
    temperature: float | None,
    sam_rho: float | None,
) -> dict[str, object]:
    model.train()
    started = time.perf_counter()
    batches: list[dict[str, torch.Tensor]] = []
    updates: list[dict[str, object]] = []
    for batch in loader:
        batches.append(batch)
        if len(batches) == runtime.accumulation_steps:
            updates.append(
                run_effective_batch_update(
                    model,
                    batches,
                    optimizer,
                    scheduler,
                    runtime,
                    optimizer_variant,
                    objective,
                    temperature,
                    sam_rho,
                )
            )
            batches = []
    if batches:
        raise ValueError("Training epoch ended with an incomplete effective batch")
    loss_sum = sum(float(update["loss_sum"]) for update in updates)
    loss_count = sum(int(update["loss_count"]) for update in updates)
    return {
        "objective_loss": loss_sum / loss_count,
        "optimizer_steps": len(updates),
        "backward_passes": sum(int(update["backward_passes"]) for update in updates),
        "mean_gradient_norm": float(
            np.mean([update["gradient_norm"] for update in updates])
        ),
        "epoch_seconds": time.perf_counter() - started,
        "all_finite": True,
    }


def _filter_evaluation_rows(
    predictions: torch.Tensor, cpu_batch: dict[str, torch.Tensor]
) -> dict[str, np.ndarray]:
    valid = cpu_batch["sample_valid_mask"].numpy().astype(bool, copy=False)
    return {
        "predictions": predictions[: int(valid.sum())].float().cpu().numpy(),
        **{
            name: cpu_batch[name].numpy()[valid]
            for name in (
                "sample_id",
                "targets",
                "raw_returns",
                "label_mask",
                "date_idx",
                "decision_idx",
            )
        },
    }


def collect_evaluation_observations(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    objective: str,
    temperature: float | None,
) -> tuple[EvaluationObservations, dict[str, object], list[dict[str, object]]]:
    objective_metadata(objective, temperature)
    device = next(model.parameters()).device
    model.eval()
    collected = {
        name: []
        for name in (
            "sample_id",
            "predictions",
            "targets",
            "raw_returns",
            "label_mask",
            "date_idx",
            "decision_idx",
        )
    }
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for cpu_batch in loader:
            batch = _to_device(cpu_batch, device)
            valid_count = int(cpu_batch["sample_valid_mask"].sum())
            with _autocast(device):
                predictions = _predict(model, batch)
            loss_sum, count = _objective_loss_sum(
                predictions[:valid_count],
                batch["targets"][:valid_count],
                batch["label_mask"][:valid_count],
                objective,
                temperature,
            )
            total_loss += float(loss_sum)
            total_count += int(count)
            for name, values in _filter_evaluation_rows(predictions, cpu_batch).items():
                collected[name].append(values)
    arrays = {name: np.concatenate(parts) for name, parts in collected.items()}
    summary, daily = create_metric_table(
        arrays["predictions"],
        arrays["targets"],
        arrays["raw_returns"],
        arrays["label_mask"],
        arrays["date_idx"],
        arrays["decision_idx"],
    )
    if not total_count:
        raise ValueError("Evaluation has no valid objective groups")
    summary["objective"] = objective_metadata(objective, temperature)
    summary["objective_loss"] = total_loss / total_count
    observations = EvaluationObservations(
        **{name: arrays[name] for name in EvaluationObservations.__dataclass_fields__}
    )
    return observations, summary, daily


def evaluate_model(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    objective: str,
    temperature: float | None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    _, summary, daily = collect_evaluation_observations(
        model, loader, objective, temperature
    )
    return summary, daily


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    model_name: str,
    architecture: NeuralArchitecture,
    tcn_settings: TCNSettings | None,
    optimizer_variant: str,
    objective: str,
    temperature: float | None,
    sam_rho: float | None,
    seed: int,
    epoch: int,
    validation_score: float,
    feature_store: Path,
    global_context: str | None,
    peer_features: str,
) -> dict[str, object]:
    if getattr(model, "model_name", None) != model_name:
        raise ValueError("Checkpoint model name does not match the model")
    return {
        "model_name": model_name,
        "architecture": asdict(architecture),
        "tcn_settings": None if tcn_settings is None else asdict(tcn_settings),
        "peer_features": peer_feature_metadata(model_name, architecture, peer_features),
        "context_routing": context_routing_metadata(architecture)
        if isinstance(architecture, TCNArchitecture)
        else None,
        "optimizer_variant": optimizer_variant,
        "objective": objective_metadata(objective, temperature),
        "sam": sam_metadata(optimizer_variant, sam_rho),
        "seed": seed,
        "epoch": epoch,
        "validation_score": validation_score,
        "feature_store": str(feature_store),
        "global_context": global_context,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
