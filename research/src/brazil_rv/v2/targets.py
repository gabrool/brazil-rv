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
    raw_log_return: NDArray[np.float64]
    horizons: tuple[int, ...] = HORIZONS


@dataclass(frozen=True)
class ToCloseTarget:
    target: NDArray[np.float32]
    valid: NDArray[np.bool_]
    normalized_residual: NDArray[np.float32]
    raw_log_return: NDArray[np.float64]


def _rank_row(
    values: NDArray[np.float64], valid: NDArray[np.bool_]
) -> NDArray[np.float32]:
    output = np.zeros(values.shape, dtype=np.float32)
    if valid.any():
        output[valid] = midrank_unit_interval(values[valid])
    return output


def build_multi_day_targets(
    total_return_close: NDArray[np.floating],
    active: NDArray[np.bool_],
    yang_zhang_sigma_20: NDArray[np.floating],
    unresolved_action: NDArray[np.bool_],
    *,
    horizons: tuple[int, ...] = HORIZONS,
    winsor_limit: float = 5.0,
) -> MultiDayTargets:
    """Construct causal-entry, future-realized multi-session targets."""

    close = np.asarray(total_return_close, dtype=np.float64)
    membership = np.asarray(active, dtype=np.bool_)
    sigma = np.asarray(yang_zhang_sigma_20, dtype=np.float64)
    unresolved = np.asarray(unresolved_action, dtype=np.bool_)
    if close.ndim != 2 or any(
        value.shape != close.shape
        for value in (membership, sigma, unresolved)
    ):
        raise ValueError("target inputs must be aligned [date, name]")
    if not horizons or any(value <= 0 for value in horizons) or len(set(horizons)) != len(horizons):
        raise ValueError("target horizons must be unique and positive")
    shape = (*close.shape, len(horizons))
    primary = np.zeros(shape, dtype=np.float32)
    primary_valid = np.zeros(shape, dtype=np.bool_)
    residual = np.zeros(shape, dtype=np.float32)
    raw_rank = np.zeros(shape, dtype=np.float32)
    raw_valid = np.zeros(shape, dtype=np.bool_)
    raw_return = np.full(shape, np.nan, dtype=np.float64)

    for horizon_index, horizon in enumerate(horizons):
        for day in range(max(0, close.shape[0] - horizon)):
            path = close[day : day + horizon + 1]
            path_valid = np.isfinite(path).all(axis=0) & (path > 0).all(axis=0)
            action_clear = ~unresolved[day + 1 : day + horizon + 1].any(axis=0)
            base_valid = membership[day] & path_valid & action_clear
            row = np.full(close.shape[1], np.nan, dtype=np.float64)
            row[base_valid] = np.log(close[day + horizon, base_valid] / close[day, base_valid])
            raw_return[day, :, horizon_index] = row
            raw_valid[day, :, horizon_index] = base_valid
            raw_rank[day, :, horizon_index] = _rank_row(row, base_valid)

            usable_sigma = np.isfinite(sigma[day]) & (sigma[day] > 0)
            target_valid = base_valid & usable_sigma
            if not target_valid.any():
                continue
            normalized = row[target_valid] / (sigma[day, target_valid] * np.sqrt(horizon))
            normalized -= np.median(normalized)
            normalized = np.clip(normalized, -winsor_limit, winsor_limit)
            residual[day, target_valid, horizon_index] = normalized.astype(np.float32)
            primary[day, target_valid, horizon_index] = midrank_unit_interval(normalized)
            primary_valid[day, target_valid, horizon_index] = True
    return MultiDayTargets(
        primary=primary,
        primary_valid=primary_valid,
        normalized_residual=residual,
        raw_midrank=raw_rank,
        raw_valid=raw_valid,
        raw_log_return=raw_return,
        horizons=horizons,
    )


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
    raw = np.full(entry.shape, np.nan, dtype=np.float64)
    raw[valid] = np.log(close[valid] / entry[valid])
    residual = np.zeros(entry.shape, dtype=np.float32)
    target = np.zeros(entry.shape, dtype=np.float32)
    scale = np.sqrt(remaining / session_minutes)
    for day in range(entry.shape[0]):
        row_valid = valid[day]
        if not row_valid.any():
            continue
        normalized = raw[day, row_valid] / (sigma[day, row_valid] * scale)
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


def target_interval_end_indices(date_count: int, horizons: tuple[int, ...] = HORIZONS) -> NDArray[np.int64]:
    """Return `[date, horizon]` end indices, using -1 when unavailable."""

    if date_count < 0:
        raise ValueError("date_count must be non-negative")
    starts = np.arange(date_count, dtype=np.int64)[:, None]
    ends = starts + np.asarray(horizons, dtype=np.int64)[None, :]
    ends[ends >= date_count] = -1
    return ends
