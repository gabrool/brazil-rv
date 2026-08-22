from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .contract import (
    DYNAMIC_CHANNEL_COUNT,
    EQUITY_COUNT,
    GH200_RUNTIME,
    GRADIENT_CLIP,
    SAM_NORM_EPS,
    SAM_RHO,
    SOFT_RANK_STANDARDIZATION_EPS,
    SOFT_RANK_TEMPERATURE,
    SOFT_SPEARMAN_CORRELATION_EPS,
    TCN_ARCHITECTURE,
    RuntimeSettings,
)
from .metrics import create_metric_table, primary_validation_score
from .provenance import model_metadata

TrainingObjective = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
UpdateCallback = Callable[[], None]


@dataclass(frozen=True)
class EvaluationObservations:
    predictions: np.ndarray
    targets: np.ndarray
    raw_returns: np.ndarray
    label_mask: np.ndarray
    sample_id: np.ndarray
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


def objective_metadata() -> dict[str, object]:
    return {
        "name": "soft_spearman",
        "temperature": SOFT_RANK_TEMPERATURE,
    }


def sam_metadata() -> dict[str, object]:
    return {"rho": SAM_RHO, "base_optimizer": "adamw"}


def _soft_spearman_loss_sum(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    label_mask: torch.Tensor,
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
        ) / SOFT_RANK_TEMPERATURE
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
) -> torch.Tensor:
    total, count = _soft_spearman_loss_sum(predictions, targets, label_mask)
    return total / count.clamp_min(1)


def eager_training_objective() -> TrainingObjective:
    def loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        label_mask: torch.Tensor,
    ) -> torch.Tensor:
        return _soft_spearman_loss_sum(predictions, targets, label_mask)[0]

    return loss


def compile_training_objective(
    runtime: RuntimeSettings = GH200_RUNTIME,
) -> TrainingObjective:
    return torch.compile(
        eager_training_objective(),
        backend=runtime.compile_backend,
        mode=runtime.compile_mode,
        fullgraph=runtime.compile_fullgraph,
        dynamic=runtime.compile_dynamic,
    )


def _model_transfer_keys(batch: Mapping[str, torch.Tensor]) -> tuple[str, ...]:
    keys = (
        "patches",
        "history_patch_mask",
        "instrument_mask",
        "slow_features",
        "state_position",
        "targets",
        "label_mask",
    )
    if "sidecar_features" in batch:
        return (*keys, "sidecar_features")
    return keys


def _to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device, non_blocking=device.type == "cuda")
        for key in _model_transfer_keys(batch)
    }


def _predict(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    inputs = (
        batch["patches"],
        batch["history_patch_mask"],
        batch["instrument_mask"],
        batch["slow_features"],
        batch["state_position"],
    )
    if "sidecar_features" in batch:
        return model(*inputs, batch["sidecar_features"])
    return model(*inputs)


def _autocast(device: torch.device):
    return torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    )


def _loss_count(label_mask: torch.Tensor) -> float:
    return float((label_mask.sum(1) >= 2).sum())


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
    count: float,
) -> torch.Tensor:
    total = next(model.parameters()).new_zeros((), dtype=torch.float32)
    for batch in batches:
        with _autocast(next(model.parameters()).device):
            predictions = _predict(model, batch)
        loss_sum = loss_function(
            predictions,
            batch["targets"],
            batch["label_mask"],
        )
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
    after_update: UpdateCallback | None,
) -> torch.Tensor:
    try:
        norm = _gradient_norm(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if after_update is not None:
            after_update()
        return norm
    finally:
        optimizer.zero_grad(set_to_none=True)


def _rng_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    return (
        torch.get_rng_state(),
        torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    )


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
    count: float,
    after_update: UpdateCallback | None,
) -> dict[str, object]:
    parameters = tuple(model.parameters())
    originals = [parameter.detach().clone() for parameter in parameters]
    device = parameters[0].device
    start_rng = _rng_state(device)
    optimizer.zero_grad(set_to_none=True)
    try:
        first_loss = _accumulate_gradients(model, batches, loss_function, count)
        first_norm = _gradient_norm(parameters, float("inf"))
        scale = SAM_RHO / (first_norm + SAM_NORM_EPS)
        with torch.no_grad():
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.add_(parameter.grad.detach().float() * scale)
        optimizer.zero_grad(set_to_none=True)
        _restore_rng(start_rng, device)
        try:
            _accumulate_gradients(model, batches, loss_function, count)
        finally:
            with torch.no_grad():
                for parameter, original in zip(parameters, originals, strict=True):
                    parameter.copy_(original)
        clipped = _optimizer_update(model, optimizer, scheduler, after_update)
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
    runtime: RuntimeSettings = GH200_RUNTIME,
    *,
    training_objective: TrainingObjective | None = None,
    after_update: UpdateCallback | None = None,
) -> dict[str, object]:
    if len(effective_batch) != runtime.loader_batches_per_effective_batch:
        raise ValueError("Effective batch has the wrong loader-batch count")
    count = sum(_loss_count(batch["label_mask"]) for batch in effective_batch)
    if count <= 0:
        raise ValueError("Effective batch has no valid objective group")
    device = next(model.parameters()).device
    batches = [
        microbatch
        for cpu_batch in effective_batch
        for microbatch in _split_microbatches(
            _to_device(cpu_batch, device), runtime.microbatch_size
        )
    ]
    return _run_sam_update(
        model,
        batches,
        optimizer,
        scheduler,
        training_objective or eager_training_objective(),
        count,
        after_update,
    )


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    runtime: RuntimeSettings = GH200_RUNTIME,
    training_objective: TrainingObjective | None = None,
    after_update: UpdateCallback | None = None,
) -> dict[str, object]:
    model.train()
    batches: list[dict[str, torch.Tensor]] = []
    updates: list[dict[str, object]] = []
    for batch in loader:
        batches.append(batch)
        if len(batches) == runtime.loader_batches_per_effective_batch:
            updates.append(
                run_effective_batch_update(
                    model,
                    batches,
                    optimizer,
                    scheduler,
                    runtime,
                    training_objective=training_objective,
                    after_update=after_update,
                )
            )
            batches = []
    if batches:
        raise ValueError("Training epoch ended with an incomplete effective batch")
    loss_count = sum(float(update["loss_count"]) for update in updates)
    loss_sum = torch.stack([update["loss_sum"] for update in updates]).sum()
    gradient_norm = torch.stack([update["gradient_norm"] for update in updates]).mean()
    objective_value, norm_value = (
        torch.stack((loss_sum / loss_count, gradient_norm)).detach().cpu().tolist()
    )
    if not np.isfinite(objective_value) or not np.isfinite(norm_value):
        raise FloatingPointError("Epoch training statistics are non-finite")
    return {
        "objective_loss": objective_value,
        "optimizer_steps": len(updates),
        "backward_passes": sum(int(update["backward_passes"]) for update in updates),
        "mean_gradient_norm": norm_value,
    }


def _filter_evaluation_metadata(
    cpu_batch: dict[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    valid = cpu_batch["sample_valid_mask"].numpy().astype(bool, copy=False)
    return {
        name: cpu_batch[name].numpy()[valid]
        for name in (
            "sample_id",
            "targets",
            "raw_returns",
            "label_mask",
            "date_idx",
            "decision_idx",
        )
    }


def collect_validation_observations(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
) -> tuple[EvaluationObservations, float]:
    device = next(model.parameters()).device
    model.eval()
    collected = {
        name: []
        for name in (
            "sample_id",
            "targets",
            "raw_returns",
            "label_mask",
            "date_idx",
            "decision_idx",
        )
    }
    prediction_parts: list[torch.Tensor] = []
    loss_sums: list[torch.Tensor] = []
    total_count = 0.0
    with torch.inference_mode():
        for cpu_batch in loader:
            valid_count = int(cpu_batch["sample_valid_mask"].sum())
            total_count += _loss_count(cpu_batch["label_mask"][:valid_count])
            batch = _to_device(cpu_batch, device)
            with _autocast(device):
                predictions = _predict(model, batch)
            loss_sum, _ = _soft_spearman_loss_sum(
                predictions[:valid_count],
                batch["targets"][:valid_count],
                batch["label_mask"][:valid_count],
            )
            loss_sums.append(loss_sum.detach())
            prediction_parts.append(predictions[:valid_count].float())
            for name, values in _filter_evaluation_metadata(cpu_batch).items():
                collected[name].append(values)
    if total_count <= 0:
        raise ValueError("Evaluation has no valid objective groups")
    arrays = {
        "predictions": torch.cat(prediction_parts).cpu().numpy(),
        **{name: np.concatenate(parts) for name, parts in collected.items()},
    }
    order = np.argsort(arrays["sample_id"], kind="stable")
    observations = EvaluationObservations(
        **{
            name: arrays[name][order]
            for name in EvaluationObservations.__dataclass_fields__
        }
    )
    return observations, float(torch.stack(loss_sums).sum().cpu()) / total_count


def collect_equity_input_ablation_predictions(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    *,
    dynamic_channel_count: int,
    slow_field_count: int,
) -> tuple[EvaluationObservations, dict[str, np.ndarray]]:
    """Evaluate every one-field equity-input zeroing with one loader traversal."""
    if dynamic_channel_count != DYNAMIC_CHANNEL_COUNT:
        raise ValueError("Dynamic ablation count differs from the model contract")
    device = next(model.parameters()).device
    model.eval()
    metadata = {
        name: []
        for name in (
            "sample_id",
            "targets",
            "raw_returns",
            "label_mask",
            "date_idx",
            "decision_idx",
        )
    }
    predictions = {
        **{f"dynamic_{index}": [] for index in range(dynamic_channel_count)},
        **{f"slow_{index}": [] for index in range(slow_field_count)},
    }
    with torch.inference_mode():
        for cpu_batch in loader:
            valid_count = int(cpu_batch["sample_valid_mask"].sum())
            batch = _to_device(cpu_batch, device)
            for index in range(dynamic_channel_count):
                patches = batch["patches"].clone()
                patches[:, :EQUITY_COUNT, :, index::dynamic_channel_count] = 0
                modified = {**batch, "patches": patches}
                with _autocast(device):
                    values = _predict(model, modified)
                predictions[f"dynamic_{index}"].append(
                    values[:valid_count].float().cpu().numpy()
                )
            for index in range(slow_field_count):
                slow = batch["slow_features"].clone()
                slow[:, :EQUITY_COUNT, index] = 0
                modified = {**batch, "slow_features": slow}
                with _autocast(device):
                    values = _predict(model, modified)
                predictions[f"slow_{index}"].append(
                    values[:valid_count].float().cpu().numpy()
                )
            for name, values in _filter_evaluation_metadata(cpu_batch).items():
                metadata[name].append(values)
    arrays = {name: np.concatenate(parts) for name, parts in metadata.items()}
    order = np.argsort(arrays["sample_id"], kind="stable")
    reference = EvaluationObservations(
        predictions=np.zeros_like(arrays["targets"])[order],
        **{name: arrays[name][order] for name in metadata},
    )
    return reference, {
        name: np.concatenate(parts)[order] for name, parts in predictions.items()
    }


def assert_observations_aligned(
    reference: EvaluationObservations,
    candidate: EvaluationObservations,
) -> None:
    for name in (
        "sample_id",
        "date_idx",
        "decision_idx",
        "targets",
        "label_mask",
        "raw_returns",
    ):
        if not np.array_equal(getattr(reference, name), getattr(candidate, name)):
            raise ValueError(f"Evaluation observations differ in {name}")


def validation_primary_metric(observations: EvaluationObservations) -> float:
    return primary_validation_score(
        observations.predictions,
        observations.targets,
        observations.label_mask,
        observations.date_idx,
    )


def summarize_evaluation_observations(
    observations: EvaluationObservations,
    objective_loss_value: float | None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    summary, daily = create_metric_table(
        observations.predictions,
        observations.targets,
        observations.raw_returns,
        observations.label_mask,
        observations.date_idx,
        observations.decision_idx,
    )
    summary["objective"] = objective_metadata()
    if objective_loss_value is not None:
        summary["objective_loss"] = objective_loss_value
    return summary, daily


def collect_evaluation_observations(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
) -> tuple[EvaluationObservations, dict[str, object], list[dict[str, object]]]:
    observations, loss = collect_validation_observations(model, loader)
    summary, daily = summarize_evaluation_observations(observations, loss)
    return observations, summary, daily


def state_dict_to_cpu(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state_dict.items()}


def checkpoint_payload(
    model: nn.Module,
    ema_state_dicts: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    seed: int,
    epoch: int,
    validation_scores: Mapping[str, float],
    feature_store: Path,
    run_provenance: dict[str, object],
) -> dict[str, object]:
    metadata = model_metadata(getattr(model, "sidecar_feature_count", None))
    if run_provenance.get("model") != metadata:
        raise ValueError("Run provenance differs from checkpoint model")
    payload = {
        "model": metadata,
        "architecture": asdict(TCN_ARCHITECTURE),
        "objective": objective_metadata(),
        "sam": sam_metadata(),
        "seed": seed,
        "epoch": epoch,
        "validation_scores": dict(validation_scores),
        "feature_store": str(feature_store.resolve()),
        "feature_store_identity": run_provenance["feature_store_identity"],
        "repository_commit": run_provenance["repository_commit"],
        "run_provenance": run_provenance,
        "model_state_dict": state_dict_to_cpu(model.state_dict()),
        "ema_state_dicts": {
            name: state_dict_to_cpu(state) for name, state in ema_state_dicts.items()
        },
    }
    if "external_sidecar" in run_provenance:
        payload["external_sidecar"] = run_provenance["external_sidecar"]
    return payload
