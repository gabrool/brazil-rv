from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .corporate_actions import AlignedActionTerms, apply_contractual_action
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


@dataclass(frozen=True)
class EconomicMultiDayTargets:
    """Canonical endpoint outcomes and median/volatility-scaled labels."""

    primary: NDArray[np.float32]
    primary_valid: NDArray[np.bool_]
    normalized_residual: NDArray[np.float32]
    normalized_cross_section_valid: NDArray[np.bool_]
    shareholder_midrank: NDArray[np.float32]
    shareholder_valid: NDArray[np.bool_]
    shareholder_simple_return: NDArray[np.float32]
    terminal_wealth: NDArray[np.float32]
    terminal_loss: NDArray[np.bool_]
    price_midrank: NDArray[np.float32]
    price_valid: NDArray[np.bool_]
    price_simple_return: NDArray[np.float32]
    horizons: tuple[int, ...]
    entry_mark_type: str
    exit_mark_type: str


def build_neutralized_log_returns(
    adjusted_close: NDArray[np.floating],
    observed: NDArray[np.bool_],
    active: NDArray[np.bool_],
    return_neutralized_event: NDArray[np.bool_],
    *,
    minimum_cross_section: int = 20,
) -> NeutralizedReturns:
    """Build an explicitly synthetic market-neutralized diagnostic series.

    This is not a corporate-action adjustment, shareholder return, headline
    target, or P&L input.  It is retained only for bounded audit comparisons.
    """

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


def build_economic_multi_day_targets(
    raw_close: NDArray[np.floating],
    close_observed: NDArray[np.bool_],
    active: NDArray[np.bool_],
    sigma_asof: NDArray[np.floating],
    actions: AlignedActionTerms,
    *,
    source_rows: NDArray[np.integer] | None = None,
    horizons: tuple[int, ...] = HORIZONS,
    winsor_limit: float = 5.0,
    minimum_sigma: float = 1e-8,
    entry_mark_type: str = "daily_last_trade_close_proxy",
    exit_mark_type: str = "daily_last_trade_close_proxy",
) -> EconomicMultiDayTargets:
    """Build price and gross shareholder outcomes from verified action terms.

    Entry wealth buys at close(t).  Contractual q/d events are applied only
    over ``(t, t+H]``; cash remains cash/receivable and is not reinvested.
    Only entry and exit quotes are required, so a missing intermediate print
    does not erase an otherwise exact endpoint outcome.  ``sigma_asof[t]`` is
    consumed directly because the canonical risk input is already lagged.
    """

    close = np.asarray(raw_close, dtype=np.float64)
    observed = np.asarray(close_observed, dtype=np.bool_)
    membership = np.asarray(active, dtype=np.bool_)
    sigma = np.asarray(sigma_asof, dtype=np.float64)
    q = np.asarray(actions.shares_per_prior_share, dtype=np.float64)
    d = np.asarray(actions.cash_per_prior_share, dtype=np.float64)
    resolved = np.asarray(actions.session_resolved, dtype=np.bool_)
    successor = (
        np.broadcast_to(np.arange(close.shape[1], dtype=np.int64), close.shape)
        if actions.successor_index is None
        else np.asarray(actions.successor_index, dtype=np.int64)
    )
    if close.ndim != 2 or any(
        value.shape != close.shape
        for value in (observed, membership, sigma, q, d, resolved, successor)
    ):
        raise ValueError("economic-target inputs must align [date, name]")
    if (successor < 0).any() or (successor >= close.shape[1]).any():
        raise ValueError("action successor indices are invalid")
    if (
        not horizons
        or any(value <= 0 for value in horizons)
        or len(set(horizons)) != len(horizons)
    ):
        raise ValueError("target horizons must be unique and positive")
    if not np.isfinite(winsor_limit) or winsor_limit <= 0.0:
        raise ValueError("winsor_limit must be positive and finite")
    if not np.isfinite(minimum_sigma) or minimum_sigma <= 0.0:
        raise ValueError("minimum_sigma must be positive and finite")
    if not entry_mark_type or not exit_mark_type:
        raise ValueError("entry and exit mark types must be explicit")
    invalid_resolved_terms = resolved & (
        ~np.isfinite(q) | ~np.isfinite(d) | (q < 0.0) | (d < 0.0)
    )
    if invalid_resolved_terms.any():
        raise ValueError("resolved action terms must have finite non-negative q/d")

    if source_rows is None:
        rows = np.arange(close.shape[0], dtype=np.int64)
    else:
        raw_rows = np.asarray(source_rows)
        if raw_rows.ndim != 1 or not np.issubdtype(raw_rows.dtype, np.integer):
            raise TypeError("source_rows must be a one-dimensional integer array")
        rows = raw_rows.astype(np.int64, copy=False)
        if np.any(rows < 0) or np.any(rows >= close.shape[0]):
            raise ValueError("source_rows contains an out-of-range index")

    shape = (rows.size, close.shape[1], len(horizons))
    primary = np.zeros(shape, dtype=np.float32)
    primary_valid = np.zeros(shape, dtype=np.bool_)
    normalized = np.zeros(shape, dtype=np.float32)
    normalized_cross_section_valid = np.zeros(
        (rows.size, len(horizons)), dtype=np.bool_
    )
    shareholder_rank = np.zeros(shape, dtype=np.float32)
    shareholder_valid = np.zeros(shape, dtype=np.bool_)
    shareholder_return = np.full(shape, np.nan, dtype=np.float32)
    terminal_wealth = np.full(shape, np.nan, dtype=np.float32)
    terminal_loss = np.zeros(shape, dtype=np.bool_)
    price_rank = np.zeros(shape, dtype=np.float32)
    price_valid = np.zeros(shape, dtype=np.bool_)
    price_return = np.full(shape, np.nan, dtype=np.float32)

    for horizon_index, horizon in enumerate(horizons):
        for output_day, day in enumerate(rows):
            end = day + horizon
            if end >= close.shape[0]:
                continue
            entry_valid = (
                membership[day]
                & observed[day]
                & np.isfinite(close[day])
                & (close[day] > 0.0)
            )
            if not entry_valid.any():
                continue

            shares = np.ones(close.shape[1], dtype=np.float64)
            cash = np.zeros(close.shape[1], dtype=np.float64)
            claim = np.arange(close.shape[1], dtype=np.int64)
            chain_resolved = entry_valid.copy()
            for event_day in range(day + 1, end + 1):
                event_q = q[event_day, claim]
                event_d = d[event_day, claim]
                event_resolved = (
                    resolved[event_day, claim]
                    & np.isfinite(event_q)
                    & np.isfinite(event_d)
                    & (event_q >= 0.0)
                    & (event_d >= 0.0)
                )
                chain_resolved &= event_resolved
                safe_q = np.where(event_resolved, event_q, 1.0)
                safe_d = np.where(event_resolved, event_d, 0.0)
                shares, cash = apply_contractual_action(
                    shares,
                    cash,
                    shares_per_prior_share=safe_q,
                    cash_per_prior_share=safe_d,
                )
                shares = np.asarray(shares, dtype=np.float64)
                cash = np.asarray(cash, dtype=np.float64)
                claim = np.where(
                    event_resolved, successor[event_day, claim], claim
                )

            exit_observed = (
                observed[end, claim]
                & np.isfinite(close[end, claim])
                & (close[end, claim] > 0.0)
            )
            endpoint_known = (shares == 0.0) | exit_observed
            economic_valid = chain_resolved & endpoint_known
            exit_price = np.where(exit_observed, close[end, claim], 0.0)
            terminal_price_per_entry_share = shares * exit_price
            terminal_value_per_entry_share = terminal_price_per_entry_share + cash
            wealth = np.full(close.shape[1], np.nan, dtype=np.float64)
            wealth[economic_valid] = (
                terminal_value_per_entry_share[economic_valid]
                / close[day, economic_valid]
            )
            holding_valid = (
                economic_valid & np.isfinite(wealth) & (wealth >= 0.0)
            )
            price_wealth = np.full(close.shape[1], np.nan, dtype=np.float64)
            price_wealth[economic_valid] = (
                terminal_price_per_entry_share[economic_valid]
                / close[day, economic_valid]
            )
            row_price_valid = (
                economic_valid
                & np.isfinite(price_wealth)
                & (price_wealth >= 0.0)
            )
            holding_simple = wealth - 1.0
            price_simple = price_wealth - 1.0

            shareholder_valid[output_day, :, horizon_index] = holding_valid
            shareholder_return[output_day, holding_valid, horizon_index] = (
                holding_simple[holding_valid].astype(np.float32)
            )
            terminal_wealth[output_day, holding_valid, horizon_index] = wealth[
                holding_valid
            ].astype(np.float32)
            terminal_loss[output_day, holding_valid, horizon_index] = (
                wealth[holding_valid] == 0.0
            )
            shareholder_rank[output_day, :, horizon_index] = _rank_row(
                holding_simple, holding_valid
            )
            price_valid[output_day, :, horizon_index] = row_price_valid
            price_return[output_day, row_price_valid, horizon_index] = price_simple[
                row_price_valid
            ].astype(np.float32)
            price_rank[output_day, :, horizon_index] = _rank_row(
                price_simple, row_price_valid
            )

            if not holding_valid.any():
                continue
            log_wealth = np.full(close.shape[1], np.nan, dtype=np.float64)
            positive = holding_valid & (wealth > 0.0)
            log_wealth[positive] = np.log(wealth[positive])
            log_wealth[holding_valid & (wealth == 0.0)] = -np.inf
            median_return = float(np.median(log_wealth[holding_valid]))
            if not np.isfinite(median_return):
                continue
            normalized_cross_section_valid[output_day, horizon_index] = True
            usable_sigma = np.isfinite(sigma[day]) & (sigma[day] > minimum_sigma)
            row_target_valid = holding_valid & usable_sigma
            if not row_target_valid.any():
                continue
            row_normalized = np.full(close.shape[1], np.nan, dtype=np.float64)
            row_positive = row_target_valid & (wealth > 0.0)
            row_normalized[row_positive] = (
                (log_wealth[row_positive] - median_return)
                / (sigma[day, row_positive] * np.sqrt(horizon))
            )
            row_normalized[row_target_valid & (wealth == 0.0)] = -winsor_limit
            row_normalized[row_target_valid] = np.clip(
                row_normalized[row_target_valid], -winsor_limit, winsor_limit
            )
            normalized[output_day, row_target_valid, horizon_index] = (
                row_normalized[row_target_valid].astype(np.float32)
            )
            primary[output_day, row_target_valid, horizon_index] = (
                midrank_unit_interval(row_normalized[row_target_valid])
            )
            primary_valid[output_day, row_target_valid, horizon_index] = True

    return EconomicMultiDayTargets(
        primary=primary,
        primary_valid=primary_valid,
        normalized_residual=normalized,
        normalized_cross_section_valid=normalized_cross_section_valid,
        shareholder_midrank=shareholder_rank,
        shareholder_valid=shareholder_valid,
        shareholder_simple_return=shareholder_return,
        terminal_wealth=terminal_wealth,
        terminal_loss=terminal_loss,
        price_midrank=price_rank,
        price_valid=price_valid,
        price_simple_return=price_return,
        horizons=horizons,
        entry_mark_type=entry_mark_type,
        exit_mark_type=exit_mark_type,
    )


def build_multi_day_targets(
    neutralized_log_return: NDArray[np.floating],
    neutralized_log_return_valid: NDArray[np.bool_],
    active: NDArray[np.bool_],
    yang_zhang_sigma_20: NDArray[np.floating],
    *,
    horizons: tuple[int, ...] = HORIZONS,
    winsor_limit: float = 5.0,
) -> MultiDayTargets:
    """Construct the legacy synthetic-return diagnostic targets.

    The accepted economic path uses :func:`build_economic_multi_day_targets`.
    This compatibility function remains while existing store callers migrate.
    """

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
            if day + horizon >= daily_return.shape[0]:
                continue
            path = daily_return[day + 1 : day + horizon + 1]
            path_valid = return_valid[day + 1 : day + horizon + 1].all(axis=0)
            base_valid = membership[day] & path_valid
            row = np.full(daily_return.shape[1], np.nan, dtype=np.float64)
            row[base_valid] = path[:, base_valid].sum(axis=0, dtype=np.float64)
            raw_log_return[output_day, :, horizon_index] = row.astype(np.float32)
            raw_valid[output_day, :, horizon_index] = base_valid
            raw_midrank[output_day, :, horizon_index] = _rank_row(row, base_valid)

            usable_sigma = np.isfinite(sigma[day]) & (sigma[day] > 1e-8)
            target_valid = base_valid & usable_sigma
            if not target_valid.any():
                continue
            median_return = float(np.median(row[base_valid]))
            normalized = (row[target_valid] - median_return) / (
                sigma[day, target_valid] * np.sqrt(horizon)
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
