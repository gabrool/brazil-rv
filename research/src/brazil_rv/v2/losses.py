from __future__ import annotations

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
) -> torch.Tensor:
    clean_scores = torch.where(mask, scores, torch.zeros_like(scores))
    clean_targets = torch.where(mask, targets, torch.zeros_like(targets))
    total, count = _soft_spearman_loss_sum(
        clean_scores, clean_targets, mask, temperature
    )
    return total / count.clamp_min(1)


def score_persistence_penalty(
    paired_scores: torch.Tensor,
    score_mask: torch.Tensor,
    *,
    horizon_count: int = 5,
    epsilon: float = SOFT_RANK_STANDARDIZATION_EPS,
) -> torch.Tensor:
    """Population-z-score persistence over adjacent, full cross-sections."""

    if paired_scores.ndim != 4 or paired_scores.shape[1] != 2:
        raise ValueError("paired_scores must have shape [pair, 2, name, head]")
    # Persistence is deliberately outside autocast. Population moments and the
    # subtraction across adjacent dates are part of the frozen float32 loss.
    with torch.autocast(device_type=paired_scores.device.type, enabled=False):
        scores = paired_scores[..., :horizon_count].float()
        return _score_persistence_penalty_float32(
            scores, score_mask, epsilon=epsilon
        )


def _score_persistence_penalty_float32(
    scores: torch.Tensor,
    score_mask: torch.Tensor,
    *,
    epsilon: float,
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
    return total / common.sum().clamp_min(1)


def multi_horizon_loss_components(
    scores: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    score_mask: torch.Tensor | None = None,
    persistence_weight: float = 0.0,
    temperature: float = SOFT_RANK_TEMPERATURE,
    to_close_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    if scores.shape != targets.shape or scores.shape != target_mask.shape:
        raise ValueError("scores, targets, and target_mask must have identical shapes")
    if scores.shape[-1] != 6:
        raise ValueError("the v2 objective requires five horizon heads and to-close")
    flat_scores = _flatten_date_pairs(scores)
    flat_targets = _flatten_date_pairs(targets)
    flat_mask = _flatten_date_pairs(target_mask).bool()
    head_losses = torch.stack(
        tuple(
            _masked_head_loss(
                flat_scores[..., head : head + 1],
                flat_targets[..., head : head + 1],
                flat_mask[..., head : head + 1],
                temperature,
            )
            for head in range(5)
        )
    )
    horizon = head_losses.mean()
    to_close = _masked_head_loss(
        flat_scores[..., 5:],
        flat_targets[..., 5:],
        flat_mask[..., 5:],
        temperature,
    )
    if persistence_weight:
        if scores.ndim != 4 or score_mask is None:
            raise ValueError(
                "persistence requires paired scores and an explicit score mask"
            )
        persistence = score_persistence_penalty(scores, score_mask)
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
) -> torch.Tensor:
    return multi_horizon_loss_components(
        scores,
        targets,
        target_mask,
        score_mask=score_mask,
        persistence_weight=persistence_weight,
        temperature=temperature,
    )["total"]
