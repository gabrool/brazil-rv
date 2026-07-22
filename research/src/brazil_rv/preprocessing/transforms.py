from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from numpy.typing import NDArray

from .contract import (
    DECISION_EQUITY_INDICES,
    HORIZONS,
    MAD_NORMALIZATION,
    MIN_ACTIVE_EQUITIES,
    MIN_ADJACENT_RETURNS_PER_DAY,
    PRICE_FEATURE_CLIP,
    PRICE_VOL_FLOOR,
    PRICE_VOL_REFERENCE,
    RATE_VOL_FLOOR_BP,
    RATE_VOL_REFERENCE_BP,
    VOL_EWMA_ALPHA,
    VOL_REGIME_CLIP,
    VOL_WARMUP_VALID_DAYS,
    VOLUME_FEATURE_CLIP,
    VOLUME_LOOKBACK_SESSIONS,
    VOLUME_MAD_FLOOR,
    VOLUME_MIN_OBSERVATIONS,
)


@dataclass(frozen=True)
class InstrumentFeatures:
    dynamic: NDArray[np.float32]
    vol_regime: NDArray[np.float32]
    sigma: NDArray[np.float64]
    data_ready: NDArray[np.bool_]


def build_causal_features(
    raw_grid: NDArray[np.float64],
    observed: NDArray[np.bool_],
    valid_day: NDArray[np.bool_],
    *,
    is_rate: bool,
    extra_ready: NDArray[np.bool_] | None = None,
) -> InstrumentFeatures:
    """Build features using only states computed from strictly earlier dates."""
    date_count, minute_count, field_count = raw_grid.shape
    if field_count != 5 or observed.shape != (date_count, minute_count):
        raise ValueError("raw_grid must be [date, minute, OHLCV]")

    if extra_ready is None:
        extra_ready = np.ones(date_count, dtype=bool)

    dynamic = np.zeros((date_count, minute_count, 6), dtype=np.float32)
    vol_regime = np.zeros(date_count, dtype=np.float32)
    sigma_by_day = np.zeros(date_count, dtype=np.float64)
    data_ready = np.zeros(date_count, dtype=bool)

    warmup_variances: list[float] = []
    ewma_variance: float | None = None
    volatility_floor = RATE_VOL_FLOOR_BP if is_rate else PRICE_VOL_FLOOR
    volatility_reference = RATE_VOL_REFERENCE_BP if is_rate else PRICE_VOL_REFERENCE

    for date_idx in range(date_count):
        ready = (
            ewma_variance is not None and valid_day[date_idx] and extra_ready[date_idx]
        )
        if ready:
            sigma = np.sqrt(max(ewma_variance, volatility_floor**2))
            sigma_by_day[date_idx] = sigma
            data_ready[date_idx] = True
            vol_regime[date_idx] = np.float32(
                np.clip(
                    np.log(sigma / volatility_reference),
                    -VOL_REGIME_CLIP,
                    VOL_REGIME_CLIP,
                )
            )

            positions = np.flatnonzero(observed[date_idx])
            if positions.size:
                prices = raw_grid[date_idx, positions, :4]
                anchors = np.empty(positions.size, dtype=np.float64)
                anchors[0] = prices[0, 0]
                anchors[1:] = raw_grid[date_idx, positions[:-1], 3]
                if is_rate:
                    moves = 100.0 * (prices - anchors[:, None])
                else:
                    moves = np.log(prices / anchors[:, None])
                dynamic[date_idx, positions, :4] = np.clip(
                    moves / sigma, -PRICE_FEATURE_CLIP, PRICE_FEATURE_CLIP
                ).astype(np.float32)

                first_history_date = max(0, date_idx - VOLUME_LOOKBACK_SESSIONS)
                prior_observed = observed[first_history_date:date_idx, positions]
                counts = prior_observed.sum(axis=0)
                baseline_ready = counts >= VOLUME_MIN_OBSERVATIONS
                if baseline_ready.any():
                    prior_volume = raw_grid[first_history_date:date_idx, positions, 4]
                    prior_logs = np.full(prior_volume.shape, np.nan, dtype=np.float64)
                    np.log(prior_volume, out=prior_logs, where=prior_observed)
                    usable = prior_logs[:, baseline_ready]
                    median = np.nanmedian(usable, axis=0)
                    mad = np.nanmedian(np.abs(usable - median), axis=0)
                    scale = np.maximum(MAD_NORMALIZATION * mad, VOLUME_MAD_FLOOR)
                    current_log_volume = np.log(
                        raw_grid[date_idx, positions[baseline_ready], 4]
                    )
                    surprise = np.clip(
                        (current_log_volume - median) / scale,
                        -VOLUME_FEATURE_CLIP,
                        VOLUME_FEATURE_CLIP,
                    )
                    dynamic[date_idx, positions[baseline_ready], 4] = surprise.astype(
                        np.float32
                    )

                dynamic[date_idx, positions, 5] = 1.0

        daily_variance = _daily_variance(
            raw_grid[date_idx, :, 3], observed[date_idx], is_rate=is_rate
        )
        if valid_day[date_idx] and daily_variance is not None:
            if ewma_variance is None:
                warmup_variances.append(daily_variance)
                if len(warmup_variances) == VOL_WARMUP_VALID_DAYS:
                    ewma_variance = float(np.median(warmup_variances))
            else:
                ewma_variance = (
                    1.0 - VOL_EWMA_ALPHA
                ) * ewma_variance + VOL_EWMA_ALPHA * daily_variance

    return InstrumentFeatures(dynamic, vol_regime, sigma_by_day, data_ready)


def _daily_variance(
    close: NDArray[np.float64], observed: NDArray[np.bool_], *, is_rate: bool
) -> float | None:
    adjacent = observed[1:] & observed[:-1]
    if int(adjacent.sum()) < MIN_ADJACENT_RETURNS_PER_DAY:
        return None
    current = close[1:][adjacent]
    previous = close[:-1][adjacent]
    returns = 100.0 * (current - previous) if is_rate else np.log(current / previous)
    return float(np.mean(returns**2))


def build_prior_rate_level(
    daily_close: NDArray[np.float64], observed: NDArray[np.bool_]
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    date_count = daily_close.shape[0]
    scaled = np.zeros(date_count, dtype=np.float32)
    ready = np.zeros(date_count, dtype=bool)
    prior_close: float | None = None
    for date_idx in range(date_count):
        if prior_close is not None:
            scaled[date_idx] = np.float32(np.clip(prior_close / 10.0, -1.0, 3.0))
            ready[date_idx] = True
        if observed[date_idx]:
            prior_close = float(daily_close[date_idx])
    return scaled, ready


def time_to_expiry_scaled(
    market_dates: tuple[date, ...], expiry_date: date
) -> NDArray[np.float32]:
    days = np.array(
        [max((expiry_date - market_date).days, 0) for market_date in market_dates],
        dtype=np.float64,
    )
    return np.clip(np.log1p(days / 365.25) / np.log(11.0), 0.0, 1.0).astype(np.float32)


def build_raw_returns(
    raw_grid: NDArray[np.float64], observed: NDArray[np.bool_]
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    date_count = raw_grid.shape[0]
    raw_returns = np.zeros(
        (date_count, len(DECISION_EQUITY_INDICES), len(HORIZONS)),
        dtype=np.float32,
    )
    endpoint_mask = np.zeros(raw_returns.shape, dtype=bool)
    for decision_idx, entry_index in enumerate(DECISION_EQUITY_INDICES):
        entry_observed = observed[:, entry_index]
        entry_open = raw_grid[:, entry_index, 0]
        for horizon_idx, horizon in enumerate(HORIZONS):
            exit_index = entry_index + horizon - 1
            valid = entry_observed & observed[:, exit_index]
            endpoint_mask[:, decision_idx, horizon_idx] = valid
            raw_returns[valid, decision_idx, horizon_idx] = np.log(
                raw_grid[valid, exit_index, 3] / entry_open[valid]
            ).astype(np.float32)
    return raw_returns, endpoint_mask


def center_cross_section(
    raw_returns: NDArray[np.float32],
    candidate_mask: NDArray[np.bool_],
    sigma: NDArray[np.float64],
) -> tuple[
    NDArray[np.float32],  # masked_raw
    NDArray[np.bool_],  # label_mask
    NDArray[np.float32],  # targets
    NDArray[np.float32],  # medians
    NDArray[np.bool_],  # horizon_mask
]:
    """Center one date of [equity, decision, horizon] raw returns."""
    masked_raw = np.where(candidate_mask, raw_returns, 0.0).astype(np.float32)
    targets = np.zeros(masked_raw.shape, dtype=np.float32)
    label_mask = np.zeros(masked_raw.shape, dtype=bool)
    medians = np.zeros(masked_raw.shape[1:], dtype=np.float32)
    horizon_mask = np.zeros(masked_raw.shape[1:], dtype=bool)

    for decision_idx in range(masked_raw.shape[1]):
        for horizon_idx, horizon in enumerate(HORIZONS):
            valid = candidate_mask[:, decision_idx, horizon_idx]
            if int(valid.sum()) < MIN_ACTIVE_EQUITIES:
                masked_raw[:, decision_idx, horizon_idx] = 0.0
                continue
            median = np.float32(np.median(masked_raw[valid, decision_idx, horizon_idx]))
            medians[decision_idx, horizon_idx] = median
            horizon_mask[decision_idx, horizon_idx] = True
            label_mask[valid, decision_idx, horizon_idx] = True
            targets[valid, decision_idx, horizon_idx] = (
                (
                    masked_raw[valid, decision_idx, horizon_idx].astype(np.float64)
                    - float(median)
                )
                / (sigma[valid] * np.sqrt(horizon))
            ).astype(np.float32)

    return masked_raw, label_mask, targets, medians, horizon_mask
