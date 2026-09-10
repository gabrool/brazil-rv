"""Physical-scale daily channels and fit-isolated clipping for Round 5."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .features import _rolling_stat, exact_log_return, yang_zhang_volatility

FEATURE_NAMES = (
    "return_1_over_vol_20",
    "daily_vol_20_raw",
    "log_traded_value_20",
    "economic_beta_60",
)


def magnitude_panel(
    *,
    wealth_open: NDArray,
    wealth_high: NDArray,
    wealth_low: NDArray,
    wealth_close: NDArray,
    wealth_valid: NDArray,
    volume_brl: NDArray,
    activity_valid: NDArray,
    daily_feature_valid: NDArray,
    slow_age_sessions: NDArray,
    slow_feature_names: tuple[str, ...],
    economic_beta: NDArray,
    economic_beta_valid: NDArray,
    economic_beta_age_sessions: NDArray,
) -> tuple[NDArray, NDArray, NDArray]:
    """Reuse parent support; daily measurements end at t-1, beta is already dated t.

    Wealth OHLC is the parent's decision-causal wealth chain, never the
    retrospective accounting action arrays. Parent validity retains its action
    and identity boundaries. Nothing is clipped using the complete panel.
    Slow ages are already decision-dated; beta age measures the most recent
    valid adjacent equity/BOVA return pair used by the rolling estimate.
    """
    observed = np.asarray(wealth_valid, bool)
    prices = [
        np.where(observed, value, np.nan)
        for value in (wealth_open, wealth_high, wealth_low, wealth_close)
    ]
    returns, return_valid = exact_log_return(
        prices[3], 1, shareholder_wealth_valid=observed
    )
    volatility, volatility_valid = yang_zhang_volatility(*prices, 20)
    mean_volume, volume_valid = _rolling_stat(
        np.where(activity_valid, volume_brl, np.nan), 20, "mean", minimum=20
    )
    shape = (*observed.shape, len(FEATURE_NAMES))
    values = np.zeros(shape, np.float32)
    valid = np.zeros(shape, bool)
    ages = np.full(shape, -1, np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        source = (returns / volatility, volatility, np.log(mean_volume))
    support = (
        return_valid & volatility_valid & (volatility > 0),
        volatility_valid,
        volume_valid & (mean_volume > 0),
    )
    parent_fields = ("log_return_1", "yang_zhang_vol_20", "log_volume_mean_20")
    for column, (payload, known, parent_field) in enumerate(
        zip(source, support, parent_fields, strict=True)
    ):
        parent_column = slow_feature_names.index(parent_field)
        parent_known = daily_feature_valid[..., parent_column]
        parent_age = slow_age_sessions[..., parent_column]
        if column == 0:
            volatility_column = slow_feature_names.index("yang_zhang_vol_20")
            parent_known = parent_known & daily_feature_valid[..., volatility_column]
            parent_age = np.maximum(
                parent_age, slow_age_sessions[..., volatility_column]
            )
        mask = known[:-1] & parent_known[1:] & np.isfinite(payload[:-1])
        values[1:, :, column] = np.where(mask, payload[:-1], 0)
        valid[1:, :, column] = mask
        ages[1:, :, column] = np.where(mask, parent_age[1:], -1)
    valid[..., 3] = economic_beta_valid & np.isfinite(economic_beta)
    values[..., 3] = np.where(valid[..., 3], economic_beta, 0)
    ages[..., 3] = np.where(valid[..., 3], economic_beta_age_sessions, -1)
    return values, valid, ages


@dataclass(frozen=True)
class FitClip:
    """Per-field 0.5/99.5 percentiles estimated solely from declared fit rows."""

    lower: NDArray
    upper: NDArray
    fit_date_indices: tuple[int, ...]

    @classmethod
    def fit(
        cls, values: NDArray, valid: NDArray, active: NDArray, fit_date_indices: NDArray
    ) -> "FitClip":
        indices = np.asarray(fit_date_indices, np.int64)
        sample = np.asarray(values)[indices]
        known = (
            np.asarray(valid)[indices]
            & np.asarray(active)[indices, :, None]
            & np.isfinite(sample)
        )
        lower, upper = [], []
        for field in range(sample.shape[-1]):
            selected = sample[..., field][known[..., field]]
            bounds = (
                np.quantile(selected, (0.005, 0.995))
                if selected.size
                else (-np.inf, np.inf)
            )
            lower.append(bounds[0])
            upper.append(bounds[1])
        return cls(
            np.array(lower, np.float32),
            np.array(upper, np.float32),
            tuple(indices.tolist()),
        )

    def transform(self, values: NDArray, valid: NDArray) -> NDArray:
        return np.where(valid, np.clip(values, self.lower, self.upper), 0).astype(
            np.float32
        )

    @classmethod
    def from_payload(cls, payload: Mapping) -> "FitClip":
        """Restore frozen checkpoint bounds; absent fit support stays unbounded."""
        lower = np.array(
            [-np.inf if x is None else x for x in payload["lower"]], np.float32
        )
        upper = np.array(
            [np.inf if x is None else x for x in payload["upper"]], np.float32
        )
        if (
            payload.get("quantiles") != [0.005, 0.995]
            or lower.ndim != 1
            or upper.shape != lower.shape
            or np.isnan(lower).any()
            or np.isnan(upper).any()
            or np.any(lower > upper)
        ):
            raise ValueError("checkpoint magnitude clipping bounds are malformed")
        return cls(lower, upper, tuple(int(x) for x in payload["fit_date_indices"]))

    def payload(self) -> dict:
        return {
            "quantiles": [0.005, 0.995],
            "fit_date_indices": list(self.fit_date_indices),
            "lower": [float(x) if np.isfinite(x) else None for x in self.lower],
            "upper": [float(x) if np.isfinite(x) else None for x in self.upper],
            "no_fit_observations_rule": "unclipped, not fit on later observations",
        }
