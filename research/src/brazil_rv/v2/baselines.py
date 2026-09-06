from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .data import read_scalar_feature_view
from .features import wealth_return_validity
from .normalization import average_ranks, rank_gauss_panel
from .store import V2Store


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
    shareholder_wealth_close: NDArray[np.floating],
    shareholder_wealth_valid: NDArray[np.bool_],
    active: NDArray[np.bool_],
    unresolved_action: NDArray[np.bool_],
    *,
    recent_lag: int,
    distant_lag: int,
    sign: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    ambiguous = np.asarray(unresolved_action, dtype=np.bool_)
    if (
        shareholder_wealth_close.ndim != 2
        or shareholder_wealth_valid.shape != shareholder_wealth_close.shape
        or active.shape != shareholder_wealth_close.shape
        or ambiguous.shape != shareholder_wealth_close.shape
    ):
        raise ValueError(
            "shareholder wealth, validity, active, and unresolved_action "
            "must have shape [date, name]"
        )
    if not 0 <= recent_lag < distant_lag:
        raise ValueError("baseline lags must satisfy 0 <= recent < distant")
    values = np.zeros(shareholder_wealth_close.shape, dtype=np.float64)
    mask = np.zeros(shareholder_wealth_close.shape, dtype=bool)
    span = distant_lag - recent_lag
    endpoint_validity = wealth_return_validity(
        shareholder_wealth_close,
        shareholder_wealth_valid,
        span,
        ambiguous,
    )
    for date in range(distant_lag, shareholder_wealth_close.shape[0]):
        recent = date - recent_lag
        distant = date - distant_lag
        valid = (
            active[date]
            & shareholder_wealth_valid[recent]
            & shareholder_wealth_valid[distant]
            & np.isfinite(shareholder_wealth_close[recent])
            & np.isfinite(shareholder_wealth_close[distant])
            & (shareholder_wealth_close[distant] > 0)
            & endpoint_validity[recent]
        )
        values[date, valid] = sign * (
            shareholder_wealth_close[recent, valid]
            / shareholder_wealth_close[distant, valid]
            - 1.0
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
    shareholder_wealth_close: NDArray[np.floating],
    shareholder_wealth_valid: NDArray[np.bool_],
    active: NDArray[np.bool_],
    unresolved_action: NDArray[np.bool_],
    target_scale_sigma: NDArray[np.floating],
) -> dict[str, BaselinePanel]:
    """Build decision-time baseline ranks from canonical raw inputs.

    Return controls consume the corrected shareholder-wealth series through
    ``t-1``. ``target_scale_sigma`` is already the canonical decision-row value
    lagged by the store builder, so inverse volatility consumes row ``t``
    without applying a second lag.
    """

    source_lag = 1
    wealth_values = np.asarray(shareholder_wealth_close, dtype=np.float64)
    wealth_valid = np.asarray(shareholder_wealth_valid, dtype=bool)
    active_mask = np.asarray(active, dtype=bool)
    ambiguous = np.asarray(unresolved_action, dtype=np.bool_)
    volatility = np.asarray(target_scale_sigma, dtype=np.float64)
    if volatility.shape != wealth_values.shape:
        raise ValueError("target_scale_sigma must have shape [date, name]")
    reversal_5, mask_5 = _lagged_return(
        wealth_values,
        wealth_valid,
        active_mask,
        ambiguous,
        recent_lag=source_lag,
        distant_lag=5 + source_lag,
        sign=-1.0,
    )
    reversal_21, mask_21 = _lagged_return(
        wealth_values,
        wealth_valid,
        active_mask,
        ambiguous,
        recent_lag=source_lag,
        distant_lag=21 + source_lag,
        sign=-1.0,
    )
    momentum_12_1, mask_momentum = _lagged_return(
        wealth_values,
        wealth_valid,
        active_mask,
        ambiguous,
        recent_lag=21 + source_lag,
        distant_lag=252 + source_lag,
        sign=1.0,
    )
    blend_mask = mask_5 & mask_momentum
    blend = 0.5 * (
        rank_gaussianize(reversal_5, blend_mask)
        + rank_gaussianize(momentum_12_1, blend_mask)
    )
    inverse_volatility = np.zeros(wealth_values.shape, dtype=np.float64)
    inverse_volatility_mask = np.zeros(wealth_values.shape, dtype=np.bool_)
    for day in range(wealth_values.shape[0]):
        valid = (
            active_mask[day] & np.isfinite(volatility[day]) & (volatility[day] > 0.0)
        )
        inverse_volatility[day, valid] = -volatility[day, valid]
        inverse_volatility_mask[day] = valid
    return {
        "reversal_5": _rank_panel(reversal_5, mask_5),
        "reversal_21": _rank_panel(reversal_21, mask_21),
        "momentum_12_1": _rank_panel(momentum_12_1, mask_momentum),
        "reversal_5_momentum_12_1_blend": _rank_panel(blend, blend_mask),
        "inverse_volatility_20": _rank_panel(
            inverse_volatility, inverse_volatility_mask
        ),
    }


def build_store_baselines(
    store: V2Store,
    date_indices: Sequence[int] | NDArray[np.integer],
) -> dict[str, BaselinePanel]:
    """Build baselines on the same canonical decision axes as model consumers."""

    # Baselines consume their declared raw controls, not normalized model
    # features.  The zero-family view shares the canonical decision axes and
    # entry eligibility without materializing the full slow feature panel.
    view = read_scalar_feature_view(store, date_indices, ())
    indices = view.date_indices
    return build_baselines(
        store.read("shareholder_wealth_close", indices),
        store.read("shareholder_wealth_valid", indices),
        view.active,
        ~store.read("action_session_resolved", indices),
        store.read("target_scale_sigma", indices),
    )
