from __future__ import annotations

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
from .metrics import create_metric_table, primary_validation_score

TrainingObjective = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


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
    validate_neural_objective("soft_spearman", temperature)
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
    return (sample_loss * valid_samples).sum(), valid_samples.sum()


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


def eager_training_objective(
    objective: str, temperature: float | None
) -> TrainingObjective:
    objective, temperature = validate_neural_objective(objective, temperature)
    if objective == "soft_spearman":
        assert temperature is not None

        def loss(
            predictions: torch.Tensor,
            targets: torch.Tensor,
            label_mask: torch.Tensor,
        ) -> torch.Tensor:
            return _soft_spearman_loss_sum(
                predictions, targets, label_mask, temperature
            )[0]

        return loss

    def loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        label_mask: torch.Tensor,
    ) -> torch.Tensor:
        return _rank_huber_loss_sum(predictions, targets, label_mask)[0]

    return loss


def compile_training_objective(
    objective: str,
    temperature: float | None,
    runtime: RuntimeSettings = GH200_RUNTIME,
) -> TrainingObjective:
    return torch.compile(
        eager_training_objective(objective, temperature),
        backend=runtime.compile_backend,
        mode=runtime.compile_mode,
        fullgraph=runtime.compile_fullgraph,
        dynamic=runtime.compile_dynamic,
    )


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


def _concatenate_batches(values: list[torch.Tensor]) -> torch.Tensor:
    first = values[0]
    output = torch.empty(
        (sum(value.shape[0] for value in values), *first.shape[1:]),
        dtype=first.dtype,
        pin_memory=first.is_pinned(),
    )
    return torch.cat(values, out=output)


def _combine_effective_batch(
    batches: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    return {
        key: _concatenate_batches([batch[key] for batch in batches])
        for key in _model_transfer_keys(batches[0])
    }


def _loss_count(label_mask: torch.Tensor, objective: str) -> int:
    if objective == "soft_spearman":
        return int((label_mask.sum(1) >= 2).sum())
    return int(label_mask.any(dim=(1, 2)).sum())


def _split_microbatches(
    batch: dict[str, torch.Tensor], microbatch_size: int
) -> list[dict[str, torch.Tensor]]:
    batch_size = next(iter(batch.values())).shape[0]
    return [
        {
            name: values[start : start + microbatch_size]
            for name, values in batch.items()
        }
        for start in range(0, batch_size, microbatch_size)
    ]


def _accumulate_gradients(
    model: nn.Module,
    batches: list[dict[str, torch.Tensor]],
    loss_function: TrainingObjective,
    count: int,
) -> torch.Tensor:
    total = next(model.parameters()).new_zeros((), dtype=torch.float32)
    for batch in batches:
        with _autocast(next(model.parameters()).device):
            predictions = _predict(model, batch)
        loss_sum = loss_function(predictions, batch["targets"], batch["label_mask"])
        total += loss_sum.detach()
        (loss_sum / count).backward()
    return total


def _gradient_norm(parameters: Iterable[nn.Parameter], maximum: float) -> torch.Tensor:
    try:
        return torch.nn.utils.clip_grad_norm_(
            tuple(parameters), maximum, error_if_nonfinite=True
        ).detach()
    except RuntimeError as error:
        raise FloatingPointError("Training gradients are non-finite") from error


def _optimizer_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> torch.Tensor:
    try:
        norm = _gradient_norm(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        return norm
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
    loss_function: TrainingObjective,
    rho: float,
    count: int,
    observer: Callable[[str, nn.Module], None] | None = None,
) -> dict[str, object]:
    parameters = tuple(model.parameters())
    originals = [parameter.detach().clone() for parameter in parameters]
    device = parameters[0].device
    start_rng = _rng_state(device)
    optimizer.zero_grad(set_to_none=True)
    try:
        first_loss = _accumulate_gradients(model, batches, loss_function, count)
        first_norm = _gradient_norm(parameters, float("inf"))
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
            _accumulate_gradients(model, batches, loss_function, count)
        finally:
            with torch.no_grad():
                for parameter, original in zip(parameters, originals, strict=True):
                    parameter.copy_(original)
        if observer:
            observer("second_gradients", model)
        clipped = _optimizer_update(model, optimizer, scheduler)
        if observer:
            observer("updated_parameters", model)
        return {
            "loss_sum": first_loss,
            "loss_count": count,
            "gradient_norm": clipped,
            "backward_passes": 2 * len(batches),
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
    training_objective: TrainingObjective | None = None,
    sam_observer: Callable[[str, nn.Module], None] | None = None,
) -> dict[str, object]:
    if len(effective_batch) != runtime.accumulation_steps:
        raise ValueError(f"Expected {runtime.accumulation_steps} microbatches")
    objective_metadata(objective, temperature)
    sam_metadata(optimizer_variant, sam_rho)
    cpu_batch = _combine_effective_batch(effective_batch)
    count = _loss_count(cpu_batch["label_mask"], objective)
    if not count:
        raise ValueError("Effective batch has no valid objective group")
    device = next(model.parameters()).device
    batch = _to_device(cpu_batch, device)
    batches = _split_microbatches(batch, runtime.microbatch_size)
    loss_function = training_objective or eager_training_objective(
        objective, temperature
    )
    if optimizer_variant == "sam_adamw":
        assert sam_rho is not None
        return _run_sam_update(
            model,
            batches,
            optimizer,
            scheduler,
            loss_function,
            sam_rho,
            count,
            sam_observer,
        )
    optimizer.zero_grad(set_to_none=True)
    total = _accumulate_gradients(model, batches, loss_function, count)
    norm = _optimizer_update(model, optimizer, scheduler)
    return {
        "loss_sum": total,
        "loss_count": count,
        "gradient_norm": norm,
        "backward_passes": len(batches),
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
    training_objective: TrainingObjective | None = None,
) -> dict[str, object]:
    model.train()
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
                    training_objective=training_objective,
                )
            )
            batches = []
    if batches:
        raise ValueError("Training epoch ended with an incomplete effective batch")
    loss_count = sum(int(update["loss_count"]) for update in updates)
    loss_sum = torch.stack([update["loss_sum"] for update in updates]).sum()
    mean_gradient_norm = torch.stack(
        [update["gradient_norm"] for update in updates]
    ).mean()
    objective_loss, mean_gradient_norm_value = (
        torch.stack((loss_sum / loss_count, mean_gradient_norm)).detach().cpu().tolist()
    )
    return {
        "objective_loss": objective_loss,
        "optimizer_steps": len(updates),
        "backward_passes": sum(int(update["backward_passes"]) for update in updates),
        "mean_gradient_norm": mean_gradient_norm_value,
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


def collect_validation_observations(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    objective: str,
    temperature: float | None,
) -> tuple[EvaluationObservations, float]:
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
    loss_sums: list[torch.Tensor] = []
    total_count = 0
    with torch.inference_mode():
        for cpu_batch in loader:
            valid_count = int(cpu_batch["sample_valid_mask"].sum())
            total_count += _loss_count(cpu_batch["label_mask"][:valid_count], objective)
            batch = _to_device(cpu_batch, device)
            with _autocast(device):
                predictions = _predict(model, batch)
            loss_sum, _ = _objective_loss_sum(
                predictions[:valid_count],
                batch["targets"][:valid_count],
                batch["label_mask"][:valid_count],
                objective,
                temperature,
            )
            loss_sums.append(loss_sum.detach())
            for name, values in _filter_evaluation_rows(predictions, cpu_batch).items():
                collected[name].append(values)
    if not total_count:
        raise ValueError("Evaluation has no valid objective groups")
    arrays = {name: np.concatenate(parts) for name, parts in collected.items()}
    order = np.argsort(arrays["sample_id"], kind="stable")
    observations = EvaluationObservations(
        **{
            name: arrays[name][order]
            for name in EvaluationObservations.__dataclass_fields__
        }
    )
    objective_loss_value = (
        float(torch.stack(loss_sums).sum().detach().cpu()) / total_count
    )
    return observations, objective_loss_value


def validation_primary_metric(observations: EvaluationObservations) -> float:
    return primary_validation_score(
        observations.predictions,
        observations.targets,
        observations.label_mask,
        observations.date_idx,
    )


def summarize_evaluation_observations(
    observations: EvaluationObservations,
    objective: str,
    temperature: float | None,
    objective_loss_value: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    summary, daily = create_metric_table(
        observations.predictions,
        observations.targets,
        observations.raw_returns,
        observations.label_mask,
        observations.date_idx,
        observations.decision_idx,
    )
    summary["objective"] = objective_metadata(objective, temperature)
    summary["objective_loss"] = objective_loss_value
    return summary, daily


def collect_evaluation_observations(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    objective: str,
    temperature: float | None,
) -> tuple[EvaluationObservations, dict[str, object], list[dict[str, object]]]:
    observations, objective_loss_value = collect_validation_observations(
        model, loader, objective, temperature
    )
    summary, daily = summarize_evaluation_observations(
        observations, objective, temperature, objective_loss_value
    )
    return observations, summary, daily


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
