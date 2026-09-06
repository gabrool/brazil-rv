from __future__ import annotations

from collections.abc import Mapping

import torch

from brazil_rv.modeling.contract import SOFT_RANK_STANDARDIZATION_EPS
from brazil_rv.modeling.engine import _soft_spearman_loss_sum

from .contract import SOFT_RANK_TEMPERATURE


def _flatten_date_pairs(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 4:
        return values.reshape(-1, values.shape[-2], values.shape[-1])
    if values.ndim != 3:
        raise ValueError(
            "model arrays must have shape [date, name, head] or [pair, 2, name, head]"
        )
    return values


def _masked_head_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
    normalization_count: torch.Tensor | None = None,
) -> torch.Tensor:
    clean_scores = torch.where(mask, scores, torch.zeros_like(scores))
    clean_targets = torch.where(mask, targets, torch.zeros_like(targets))
    total, count = _soft_spearman_loss_sum(
        clean_scores, clean_targets, mask, temperature
    )
    denominator = count if normalization_count is None else normalization_count
    return total / denominator.clamp_min(1)


def score_persistence_penalty(
    paired_scores: torch.Tensor,
    score_mask: torch.Tensor,
    *,
    horizon_count: int = 5,
    epsilon: float = SOFT_RANK_STANDARDIZATION_EPS,
    normalization_count: torch.Tensor | None = None,
) -> torch.Tensor:
    """Population-z-score persistence over adjacent, full cross-sections."""

    if paired_scores.ndim != 4 or paired_scores.shape[1] != 2:
        raise ValueError("paired_scores must have shape [pair, 2, name, head]")
    # Persistence is deliberately outside autocast. Population moments and the
    # subtraction across adjacent dates are part of the frozen float32 loss.
    with torch.autocast(device_type=paired_scores.device.type, enabled=False):
        scores = paired_scores[..., :horizon_count].float()
        return _score_persistence_penalty_float32(
            scores,
            score_mask,
            epsilon=epsilon,
            normalization_count=normalization_count,
        )


def _score_persistence_penalty_float32(
    scores: torch.Tensor,
    score_mask: torch.Tensor,
    *,
    epsilon: float,
    normalization_count: torch.Tensor | None = None,
) -> torch.Tensor:
    if score_mask.ndim == 3:
        mask = score_mask[..., None].expand_as(scores)
    elif score_mask.ndim == 4:
        mask = score_mask[..., : scores.shape[-1]]
        if mask.shape != scores.shape:
            raise ValueError("score_mask is misaligned with paired_scores")
    else:
        raise ValueError("score_mask must have shape [pair, 2, name] or match scores")
    mask = mask.bool()
    clean = torch.where(mask, scores, torch.zeros_like(scores))
    counts = mask.sum(dim=2)
    safe_counts = counts.clamp_min(1)
    means = clean.sum(dim=2) / safe_counts
    centered = torch.where(mask, scores - means[:, :, None], torch.zeros_like(scores))
    variances = centered.square().sum(dim=2) / safe_counts
    standardized = centered / torch.sqrt(variances[:, :, None] + epsilon)
    common = mask[:, 0] & mask[:, 1]
    valid_groups = (counts[:, 0] >= 2) & (counts[:, 1] >= 2)
    common &= valid_groups[:, None]
    squared_change = (standardized[:, 1] - standardized[:, 0]).square()
    total = torch.where(common, squared_change, torch.zeros_like(squared_change)).sum()
    count = common.sum()
    denominator = count if normalization_count is None else normalization_count
    return total / denominator.clamp_min(1)


def multi_horizon_loss_normalizers(
    target_mask: torch.Tensor,
    *,
    score_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Count the effective full-batch groups used by each loss component."""

    flat_mask = _flatten_date_pairs(target_mask).bool()
    if flat_mask.shape[-1] not in (5, 6):
        raise ValueError("the v2 objective requires five daily horizon heads")
    per_horizon = (flat_mask[..., :5].sum(dim=1) >= 2).sum(dim=0)
    to_close = (
        (flat_mask[..., 5:].sum(dim=1) >= 2).sum()
        if flat_mask.shape[-1] == 6
        else flat_mask.new_zeros((), dtype=torch.int64)
    )
    persistence = flat_mask.new_zeros((), dtype=torch.int64)
    if score_mask is not None:
        if target_mask.ndim != 4 or target_mask.shape[1] != 2:
            raise ValueError("persistence normalization requires adjacent date pairs")
        if score_mask.ndim == 3:
            persistent_mask = score_mask[..., None].expand(
                -1, -1, -1, 5
            )
        elif score_mask.ndim == 4:
            persistent_mask = score_mask[..., :5]
            if persistent_mask.shape != target_mask[..., :5].shape:
                raise ValueError("score_mask is misaligned with target_mask")
        else:
            raise ValueError("score_mask must be paired by date and name")
        persistent_mask = persistent_mask.bool()
        counts = persistent_mask.sum(dim=2)
        valid_groups = (counts[:, 0] >= 2) & (counts[:, 1] >= 2)
        common = persistent_mask[:, 0] & persistent_mask[:, 1]
        persistence = (common & valid_groups[:, None]).sum()
    return {
        "per_horizon": per_horizon,
        "to_close": to_close,
        "persistence": persistence,
    }


def multi_horizon_loss_components(
    scores: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    score_mask: torch.Tensor | None = None,
    persistence_weight: float = 0.0,
    temperature: float = SOFT_RANK_TEMPERATURE,
    to_close_weight: float = 0.0,
    normalization_counts: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    if scores.shape != targets.shape or scores.shape != target_mask.shape:
        raise ValueError("scores, targets, and target_mask must have identical shapes")
    if scores.shape[-1] not in (5, 6):
        raise ValueError("the v2 objective requires five daily horizon heads")
    if to_close_weight and scores.shape[-1] != 6:
        raise ValueError("a weighted to-close auxiliary requires its sixth head")
    flat_scores = _flatten_date_pairs(scores)
    flat_targets = _flatten_date_pairs(targets)
    flat_mask = _flatten_date_pairs(target_mask).bool()
    if normalization_counts is None:
        normalization_counts = multi_horizon_loss_normalizers(
            target_mask,
            score_mask=score_mask if persistence_weight else None,
        )
    per_horizon_counts = normalization_counts.get("per_horizon")
    to_close_count = normalization_counts.get("to_close")
    persistence_count = normalization_counts.get("persistence")
    if (
        not isinstance(per_horizon_counts, torch.Tensor)
        or per_horizon_counts.shape != (5,)
        or not isinstance(to_close_count, torch.Tensor)
        or to_close_count.numel() != 1
        or not isinstance(persistence_count, torch.Tensor)
        or persistence_count.numel() != 1
    ):
        raise ValueError("loss normalization counts are malformed")
    head_losses = torch.stack(
        tuple(
            _masked_head_loss(
                flat_scores[..., head : head + 1],
                flat_targets[..., head : head + 1],
                flat_mask[..., head : head + 1],
                temperature,
                per_horizon_counts[head],
            )
            for head in range(5)
        )
    )
    horizon = head_losses.mean()
    to_close = (
        _masked_head_loss(
            flat_scores[..., 5:],
            flat_targets[..., 5:],
            flat_mask[..., 5:],
            temperature,
            to_close_count,
        )
        if scores.shape[-1] == 6
        else scores.new_zeros((), dtype=torch.float32)
    )
    if persistence_weight:
        if scores.ndim != 4 or score_mask is None:
            raise ValueError(
                "persistence requires paired scores and an explicit score mask"
            )
        persistence = score_persistence_penalty(
            scores,
            score_mask,
            normalization_count=persistence_count,
        )
    else:
        persistence = scores.new_zeros((), dtype=torch.float32)
    total = horizon + to_close_weight * to_close + persistence_weight * persistence
    return {
        "total": total,
        "horizon": horizon,
        "to_close": to_close,
        "persistence": persistence,
        "per_horizon": head_losses,
    }


def multi_horizon_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    score_mask: torch.Tensor | None = None,
    persistence_weight: float = 0.0,
    temperature: float = SOFT_RANK_TEMPERATURE,
    to_close_weight: float = 0.0,
    normalization_counts: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    return multi_horizon_loss_components(
        scores,
        targets,
        target_mask,
        score_mask=score_mask,
        persistence_weight=persistence_weight,
        temperature=temperature,
        to_close_weight=to_close_weight,
        normalization_counts=normalization_counts,
    )["total"]
