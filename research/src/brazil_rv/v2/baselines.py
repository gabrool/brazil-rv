from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .normalization import average_ranks, rank_gauss_panel


@dataclass(frozen=True)
class BaselinePanel:
    scores: NDArray[np.float32]
    score_mask: NDArray[np.bool_]


def rank_gaussianize(
    values: NDArray[np.floating],
    mask: NDArray[np.bool_],
    *,
    clip: float = 3.0,
) -> NDArray[np.float32]:
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("values and mask must have shape [date, name]")
    valid = np.asarray(mask, dtype=bool)
    return rank_gauss_panel(values, valid, valid, clip=clip)[0]


def _lagged_return(
    close: NDArray[np.floating],
    observed: NDArray[np.bool_],
    active: NDArray[np.bool_],
    ambiguous_action: NDArray[np.bool_],
    *,
    recent_lag: int,
    distant_lag: int,
    sign: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    ambiguous = np.asarray(ambiguous_action, dtype=np.bool_)
    if (
        close.ndim != 2
        or observed.shape != close.shape
        or active.shape != close.shape
        or ambiguous.shape != close.shape
    ):
        raise ValueError(
            "close, observed, active, and ambiguous_action must have shape [date, name]"
        )
    if not 0 <= recent_lag < distant_lag:
        raise ValueError("baseline lags must satisfy 0 <= recent < distant")
    values = np.zeros(close.shape, dtype=np.float64)
    mask = np.zeros(close.shape, dtype=bool)
    cumulative = np.cumsum(ambiguous, axis=0, dtype=np.int32)
    for date in range(distant_lag, close.shape[0]):
        recent = date - recent_lag
        distant = date - distant_lag
        valid = (
            active[date]
            & observed[recent]
            & observed[distant]
            & np.isfinite(close[recent])
            & np.isfinite(close[distant])
            & (close[distant] > 0)
            & (cumulative[recent] - cumulative[distant] == 0)
        )
        values[date, valid] = sign * (
            close[recent, valid] / close[distant, valid] - 1.0
        )
        mask[date] = valid
    return values, mask


def _rank_panel(values: NDArray[np.floating], mask: NDArray[np.bool_]) -> BaselinePanel:
    scores = np.zeros((*values.shape, 5), dtype=np.float32)
    expanded_mask = np.repeat(mask[..., None], 5, axis=-1)
    for date in range(values.shape[0]):
        valid = mask[date]
        if valid.any():
            ranks = average_ranks(np.asarray(values[date, valid], dtype=np.float64))
            scores[date, valid, :] = ranks[:, None]
    return BaselinePanel(scores=scores, score_mask=expanded_mask)


def build_baselines(
    close: NDArray[np.floating],
    observed: NDArray[np.bool_],
    active: NDArray[np.bool_],
    ambiguous_action: NDArray[np.bool_],
    *,
    slow_lag: int = 1,
) -> dict[str, BaselinePanel]:
    """Build causal ranks; fine-tuning uses t-1 slow data (``slow_lag=1``)."""

    if slow_lag not in (0, 1):
        raise ValueError(
            "slow_lag must be zero for pretraining or one for v2 fine/eval"
        )
    close_values = np.asarray(close, dtype=np.float64)
    observed_mask = np.asarray(observed, dtype=bool)
    active_mask = np.asarray(active, dtype=bool)
    ambiguous = np.asarray(ambiguous_action, dtype=np.bool_)
    reversal_5, mask_5 = _lagged_return(
        close_values,
        observed_mask,
        active_mask,
        ambiguous,
        recent_lag=slow_lag,
        distant_lag=5 + slow_lag,
        sign=-1.0,
    )
    reversal_21, mask_21 = _lagged_return(
        close_values,
        observed_mask,
        active_mask,
        ambiguous,
        recent_lag=slow_lag,
        distant_lag=21 + slow_lag,
        sign=-1.0,
    )
    momentum_12_1, mask_momentum = _lagged_return(
        close_values,
        observed_mask,
        active_mask,
        ambiguous,
        recent_lag=21 + slow_lag,
        distant_lag=252 + slow_lag,
        sign=1.0,
    )
    blend_mask = mask_5 & mask_momentum
    blend = 0.5 * (
        rank_gaussianize(reversal_5, blend_mask)
        + rank_gaussianize(momentum_12_1, blend_mask)
    )
    return {
        "reversal_5": _rank_panel(reversal_5, mask_5),
        "reversal_21": _rank_panel(reversal_21, mask_21),
        "momentum_12_1": _rank_panel(momentum_12_1, mask_momentum),
        "reversal_5_momentum_12_1_blend": _rank_panel(blend, blend_mask),
    }
