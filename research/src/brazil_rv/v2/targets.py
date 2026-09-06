from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .contract import DECISION_MINUTE_INDEX, HORIZONS
from .normalization import midrank_unit_interval


@dataclass(frozen=True)
class MultiDayTargets:
    primary: NDArray[np.float32]
    primary_valid: NDArray[np.bool_]
    normalized_residual: NDArray[np.float32]
    raw_midrank: NDArray[np.float32]
    raw_valid: NDArray[np.bool_]
    raw_log_return: NDArray[np.float32]
    horizons: tuple[int, ...] = HORIZONS


@dataclass(frozen=True)
class ToCloseTarget:
    target: NDArray[np.float32]
    valid: NDArray[np.bool_]
    normalized_residual: NDArray[np.float32]
    raw_log_return: NDArray[np.float32]


@dataclass(frozen=True)
class NeutralizedReturns:
    log_return: NDArray[np.float32]
    valid: NDArray[np.bool_]
    cross_sectional_median: NDArray[np.float32]


def build_neutralized_log_returns(
    adjusted_close: NDArray[np.floating],
    observed: NDArray[np.bool_],
    active: NDArray[np.bool_],
    return_neutralized_event: NDArray[np.bool_],
    *,
    minimum_cross_section: int = 20,
) -> NeutralizedReturns:
    """Build daily split-adjusted returns with distribution days market-marked."""

    close = np.asarray(adjusted_close, dtype=np.float64)
    seen = np.asarray(observed, dtype=np.bool_)
    membership = np.asarray(active, dtype=np.bool_)
    neutralize = np.asarray(return_neutralized_event, dtype=np.bool_)
    if close.ndim != 2 or any(
        value.shape != close.shape for value in (seen, membership, neutralize)
    ):
        raise ValueError("neutralized-return inputs must align [date, name]")
    if minimum_cross_section < 1:
        raise ValueError("minimum_cross_section must be positive")
    values = np.zeros(close.shape, dtype=np.float32)
    valid = np.zeros(close.shape, dtype=np.bool_)
    median = np.full(close.shape[0], np.nan, dtype=np.float32)
    for day in range(1, close.shape[0]):
        base_valid = (
            seen[day]
            & seen[day - 1]
            & np.isfinite(close[day])
            & np.isfinite(close[day - 1])
            & (close[day] > 0)
            & (close[day - 1] > 0)
        )
        raw = np.zeros(close.shape[1], dtype=np.float64)
        raw[base_valid] = np.log(close[day, base_valid] / close[day - 1, base_valid])
        non_event = base_valid & membership[day] & ~neutralize[day]
        if int(non_event.sum()) >= minimum_cross_section:
            median[day] = np.float32(np.median(raw[non_event]))
        row_valid = base_valid.copy()
        event = base_valid & neutralize[day]
        if event.any():
            if np.isfinite(median[day]):
                raw[event] = float(median[day])
            else:
                row_valid[event] = False
        values[day, row_valid] = raw[row_valid].astype(np.float32)
        valid[day] = row_valid
    return NeutralizedReturns(values, valid, median)


def _rank_row(
    values: NDArray[np.float64], valid: NDArray[np.bool_]
) -> NDArray[np.float32]:
    output = np.zeros(values.shape, dtype=np.float32)
    if valid.any():
        output[valid] = midrank_unit_interval(values[valid])
    return output


def build_multi_day_targets(
    neutralized_log_return: NDArray[np.floating],
    neutralized_log_return_valid: NDArray[np.bool_],
    active: NDArray[np.bool_],
    yang_zhang_sigma_20: NDArray[np.floating],
    *,
    horizons: tuple[int, ...] = HORIZONS,
    winsor_limit: float = 5.0,
) -> MultiDayTargets:
    """Construct causal-entry, future-realized multi-session targets."""

    daily_return = np.asarray(neutralized_log_return, dtype=np.float64)
    return_valid = np.asarray(neutralized_log_return_valid, dtype=np.bool_)
    membership = np.asarray(active, dtype=np.bool_)
    sigma = np.asarray(yang_zhang_sigma_20, dtype=np.float64)
    if daily_return.ndim != 2 or any(
        value.shape != daily_return.shape
        for value in (return_valid, membership, sigma)
    ):
        raise ValueError("target inputs must be aligned [date, name]")
    if (
        not horizons
        or any(value <= 0 for value in horizons)
        or len(set(horizons)) != len(horizons)
    ):
        raise ValueError("target horizons must be unique and positive")
    shape = (*daily_return.shape, len(horizons))
    primary = np.zeros(shape, dtype=np.float32)
    primary_valid = np.zeros(shape, dtype=np.bool_)
    residual = np.zeros(shape, dtype=np.float32)
    raw_rank = np.zeros(shape, dtype=np.float32)
    raw_valid = np.zeros(shape, dtype=np.bool_)
    raw_return = np.empty(shape, dtype=np.float32)

    build_multi_day_targets_into(
        daily_return,
        return_valid,
        membership,
        sigma,
        primary=primary,
        primary_valid=primary_valid,
        normalized_residual=residual,
        raw_midrank=raw_rank,
        raw_valid=raw_valid,
        raw_log_return=raw_return,
        horizons=horizons,
        winsor_limit=winsor_limit,
    )
    return MultiDayTargets(
        primary=primary,
        primary_valid=primary_valid,
        normalized_residual=residual,
        raw_midrank=raw_rank,
        raw_valid=raw_valid,
        raw_log_return=raw_return,
        horizons=horizons,
    )


def build_multi_day_targets_into(
    neutralized_log_return: NDArray[np.floating],
    neutralized_log_return_valid: NDArray[np.bool_],
    active: NDArray[np.bool_],
    yang_zhang_sigma_20: NDArray[np.floating],
    *,
    primary: NDArray[np.float32],
    primary_valid: NDArray[np.bool_],
    normalized_residual: NDArray[np.float32],
    raw_midrank: NDArray[np.float32],
    raw_valid: NDArray[np.bool_],
    raw_log_return: NDArray[np.float32],
    source_rows: NDArray[np.integer] | None = None,
    horizons: tuple[int, ...] = HORIZONS,
    winsor_limit: float = 5.0,
) -> None:
    """Stream one target horizon at a time into pre-allocated arrays."""

    daily_return = np.asarray(neutralized_log_return)
    return_valid = np.asarray(neutralized_log_return_valid, dtype=np.bool_)
    membership = np.asarray(active, dtype=np.bool_)
    sigma = np.asarray(yang_zhang_sigma_20)
    if daily_return.ndim != 2 or any(
        value.shape != daily_return.shape
        for value in (return_valid, membership, sigma)
    ):
        raise ValueError("target inputs must be aligned [date, name]")
    if (
        not horizons
        or any(value <= 0 for value in horizons)
        or len(set(horizons)) != len(horizons)
    ):
        raise ValueError("target horizons must be unique and positive")
    if source_rows is None:
        rows = np.arange(daily_return.shape[0], dtype=np.int64)
    else:
        raw_rows = np.asarray(source_rows)
        if raw_rows.ndim != 1 or not np.issubdtype(raw_rows.dtype, np.integer):
            raise TypeError("source_rows must be a one-dimensional integer array")
        rows = raw_rows.astype(np.int64, copy=False)
        if np.any(rows < 0) or np.any(rows >= daily_return.shape[0]):
            raise ValueError("source_rows contains an out-of-range index")
    shape = (rows.size, daily_return.shape[1], len(horizons))
    destinations = (
        primary,
        primary_valid,
        normalized_residual,
        raw_midrank,
        raw_valid,
        raw_log_return,
    )
    if any(value.shape != shape for value in destinations):
        raise ValueError("target destinations are misaligned")
    if (
        primary.dtype != np.float32
        or normalized_residual.dtype != np.float32
        or raw_midrank.dtype != np.float32
        or raw_log_return.dtype != np.float32
        or primary_valid.dtype != np.bool_
        or raw_valid.dtype != np.bool_
    ):
        raise TypeError("target destinations must use float32 values and bool masks")

    for destination in destinations:
        destination[...] = False if destination.dtype == np.bool_ else 0.0
    raw_log_return[...] = np.nan

    for horizon_index, horizon in enumerate(horizons):
        for output_day, day in enumerate(rows):
            if day == 0 or day + horizon >= daily_return.shape[0]:
                continue
            path = daily_return[day + 1 : day + horizon + 1]
            path_valid = return_valid[day + 1 : day + horizon + 1].all(axis=0)
            base_valid = membership[day] & path_valid
            row = np.full(daily_return.shape[1], np.nan, dtype=np.float64)
            row[base_valid] = path[:, base_valid].sum(axis=0, dtype=np.float64)
            raw_log_return[output_day, :, horizon_index] = row.astype(np.float32)
            raw_valid[output_day, :, horizon_index] = base_valid
            raw_midrank[output_day, :, horizon_index] = _rank_row(row, base_valid)

            usable_sigma = np.isfinite(sigma[day - 1]) & (sigma[day - 1] > 0)
            target_valid = base_valid & usable_sigma
            if not target_valid.any():
                continue
            median_return = float(np.median(row[base_valid]))
            normalized = (row[target_valid] - median_return) / (
                sigma[day - 1, target_valid] * np.sqrt(horizon)
            )
            normalized = np.clip(normalized, -winsor_limit, winsor_limit)
            normalized_residual[output_day, target_valid, horizon_index] = (
                normalized.astype(np.float32)
            )
            primary[output_day, target_valid, horizon_index] = midrank_unit_interval(
                normalized
            )
            primary_valid[output_day, target_valid, horizon_index] = True


def build_to_close_target(
    entry_open: NDArray[np.floating],
    session_close: NDArray[np.floating],
    realized_daily_vol: NDArray[np.floating],
    active: NDArray[np.bool_],
    fast_present: NDArray[np.bool_],
    *,
    session_minutes: int = 405,
    cutoff: int = DECISION_MINUTE_INDEX,
) -> ToCloseTarget:
    """Return from the 15:45 entry-bar open to close, normalized by time left."""

    entry = np.asarray(entry_open, dtype=np.float64)
    close = np.asarray(session_close, dtype=np.float64)
    sigma = np.asarray(realized_daily_vol, dtype=np.float64)
    membership = np.asarray(active, dtype=np.bool_)
    present = np.asarray(fast_present, dtype=np.bool_)
    if entry.ndim != 2 or any(
        value.shape != entry.shape for value in (close, sigma, membership, present)
    ):
        raise ValueError("to-close arrays must be aligned [date, name]")
    remaining = session_minutes - cutoff
    if remaining <= 0 or cutoff <= 0:
        raise ValueError("cutoff must leave at least one session minute")
    valid = (
        membership
        & present
        & np.isfinite(entry)
        & np.isfinite(close)
        & np.isfinite(sigma)
        & (entry > 0)
        & (close > 0)
        & (sigma > 0)
    )
    raw = np.full(entry.shape, np.nan, dtype=np.float32)
    raw[valid] = np.log(close[valid] / entry[valid]).astype(np.float32)
    residual = np.zeros(entry.shape, dtype=np.float32)
    target = np.zeros(entry.shape, dtype=np.float32)
    scale = np.sqrt(remaining / session_minutes)
    for day in range(entry.shape[0]):
        row_valid = valid[day]
        if not row_valid.any():
            continue
        raw_row = np.log(close[day, row_valid] / entry[day, row_valid])
        normalized = raw_row / (sigma[day, row_valid] * scale)
        normalized -= np.median(normalized)
        normalized = np.clip(normalized, -5.0, 5.0)
        residual[day, row_valid] = normalized.astype(np.float32)
        target[day, row_valid] = midrank_unit_interval(normalized)
    return ToCloseTarget(
        target=target,
        valid=valid,
        normalized_residual=residual,
        raw_log_return=raw,
    )


def target_interval_end_indices(
    date_count: int, horizons: tuple[int, ...] = HORIZONS
) -> NDArray[np.int64]:
    """Return `[date, horizon]` end indices, using -1 when unavailable."""

    if date_count < 0:
        raise ValueError("date_count must be non-negative")
    starts = np.arange(date_count, dtype=np.int64)[:, None]
    ends = starts + np.asarray(horizons, dtype=np.int64)[None, :]
    ends[ends >= date_count] = -1
    return ends
