from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LedgerConfig:
    k_per_side: int = 30
    buffer_per_side: int = 30
    gross_target: float = 2.0
    cost_bps_per_side: float = 4.0
    annual_borrow_rate: float = 0.02
    short_proceeds_remuneration: float = 0.0
    ex_date_marking: Literal["neutral", "raw"] = "neutral"
    forced_liquidation_haircut: float = 0.0
    max_missing_sessions: int = 10
    annual_sessions: int = 252

    def __post_init__(self) -> None:
        if self.k_per_side <= 0 or self.buffer_per_side < 0:
            raise ValueError("ledger K must be positive and its buffer non-negative")
        if self.gross_target <= 0 or self.cost_bps_per_side < 0:
            raise ValueError("ledger gross/cost controls are invalid")
        if self.annual_borrow_rate < 0 or self.annual_sessions <= 0:
            raise ValueError("ledger financing controls are invalid")
        if not 0 <= self.short_proceeds_remuneration <= 1:
            raise ValueError("short-proceeds remuneration must be in [0, 1]")
        if self.ex_date_marking not in ("neutral", "raw"):
            raise ValueError("ex-date marking must be neutral or raw")
        if not 0 <= self.forced_liquidation_haircut < 1:
            raise ValueError("forced-liquidation haircut must be in [0, 1)")
        if self.max_missing_sessions < 1:
            raise ValueError("max-missing sessions must be positive")


@dataclass(frozen=True)
class StatefulLedgerResult:
    dates: tuple[date, ...]
    nav: NDArray[np.float64]
    daily_net_return: NDArray[np.float64]
    net_excess_all_cash_bps: NDArray[np.float64]
    gross_pnl_bps: NDArray[np.float64]
    neutral_flow_bps: NDArray[np.float64]
    interest_bps: NDArray[np.float64]
    cost_bps: NDArray[np.float64]
    borrow_bps: NDArray[np.float64]
    gross_fraction_nav: NDArray[np.float64]
    turnover_fraction_nav: NDArray[np.float64]
    stale_mark_name_days: NDArray[np.int64]
    neutral_marked_name_days: NDArray[np.int64]
    forced_liquidation_count: NDArray[np.int64]
    position_sign: NDArray[np.int8]

    def summary(self) -> dict[str, float | int]:
        excess = self.net_excess_all_cash_bps
        standard_deviation = float(np.std(excess, ddof=1)) if excess.size > 1 else 0.0
        sharpe = (
            float(np.mean(excess) / standard_deviation * np.sqrt(252.0))
            if standard_deviation > 0
            else 0.0
        )
        mean_gross = float(np.mean(self.gross_fraction_nav))
        mean_turnover = float(np.mean(self.turnover_fraction_nav))
        return {
            "mean_net_excess_bps_per_day": float(np.mean(excess)),
            "annualized_net_excess_sharpe": sharpe,
            "mean_gross_fraction_nav": mean_gross,
            "mean_turnover_fraction_nav": mean_turnover,
            "average_holding_sessions": (
                2.0 * mean_gross / mean_turnover if mean_turnover > 0 else 0.0
            ),
            "stale_mark_name_days": int(self.stale_mark_name_days.sum()),
            "neutral_marked_name_days": int(self.neutral_marked_name_days.sum()),
            "forced_liquidation_count": int(self.forced_liquidation_count.sum()),
            "neutral_flow_nav": float(
                np.sum(self.neutral_flow_bps * np.r_[1.0, self.nav[:-1]]) / 10_000.0
            ),
            "terminal_nav": float(self.nav[-1]),
        }


def _validate_inputs(
    dates: Sequence[date],
    scores: NDArray[np.floating],
    score_mask: NDArray[np.bool_],
    active: NDArray[np.bool_],
    adjusted_close: NDArray[np.floating],
    neutralized_log_return: NDArray[np.floating],
    neutralized_log_return_valid: NDArray[np.bool_],
    return_neutralized_event_mask: NDArray[np.bool_],
    cross_sectional_median_log_return: NDArray[np.floating],
    cdi_returns: NDArray[np.floating],
) -> tuple[np.ndarray, ...]:
    date_values = tuple(dates)
    if not date_values or tuple(sorted(date_values)) != date_values:
        raise ValueError("ledger dates must be nonempty and chronological")
    score = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(score_mask, dtype=np.bool_)
    membership = np.asarray(active, dtype=np.bool_)
    close = np.asarray(adjusted_close, dtype=np.float64)
    neutral_return = np.asarray(neutralized_log_return, dtype=np.float64)
    neutral_valid = np.asarray(neutralized_log_return_valid, dtype=np.bool_)
    event = np.asarray(return_neutralized_event_mask, dtype=np.bool_)
    median = np.asarray(cross_sectional_median_log_return, dtype=np.float64)
    cdi = np.asarray(cdi_returns, dtype=np.float64)
    matrix_shape = (len(date_values), score.shape[1] if score.ndim == 2 else 0)
    if score.ndim != 2 or any(
        value.shape != matrix_shape
        for value in (mask, membership, close, neutral_return, neutral_valid, event)
    ):
        raise ValueError("ledger name panels must align [date, name]")
    if median.shape != (len(date_values),) or cdi.shape != (len(date_values),):
        raise ValueError("ledger date-only arrays are misaligned")
    if not np.isfinite(cdi).all():
        raise ValueError("ledger CDI returns must be finite")
    return (
        score,
        mask,
        membership,
        close,
        neutral_return,
        neutral_valid,
        event,
        median,
        cdi,
    )


def simulate_stateful_ledger(
    *,
    dates: Sequence[date],
    scores: NDArray[np.floating],
    score_mask: NDArray[np.bool_],
    active: NDArray[np.bool_],
    adjusted_close: NDArray[np.floating],
    neutralized_log_return: NDArray[np.floating],
    neutralized_log_return_valid: NDArray[np.bool_],
    return_neutralized_event_mask: NDArray[np.bool_],
    cross_sectional_median_log_return: NDArray[np.floating],
    cdi_returns: NDArray[np.floating],
    config: LedgerConfig = LedgerConfig(),
) -> StatefulLedgerResult:
    """Replay the registered close-filled, buffered, stateful v2 policy."""

    (
        score,
        score_valid,
        membership,
        close,
        neutral_return,
        neutral_valid,
        event,
        median,
        cdi,
    ) = _validate_inputs(
        dates,
        scores,
        score_mask,
        active,
        adjusted_close,
        neutralized_log_return,
        neutralized_log_return_valid,
        return_neutralized_event_mask,
        cross_sectional_median_log_return,
        cdi_returns,
    )
    del neutral_return, neutral_valid
    day_count, name_count = score.shape
    nav = np.empty(day_count, dtype=np.float64)
    daily_return = np.empty(day_count, dtype=np.float64)
    excess_bps = np.empty(day_count, dtype=np.float64)
    gross_pnl_bps = np.zeros(day_count, dtype=np.float64)
    neutral_flow_bps = np.zeros(day_count, dtype=np.float64)
    interest_bps = np.zeros(day_count, dtype=np.float64)
    cost_bps = np.zeros(day_count, dtype=np.float64)
    borrow_bps = np.zeros(day_count, dtype=np.float64)
    gross_fraction = np.zeros(day_count, dtype=np.float64)
    turnover_fraction = np.zeros(day_count, dtype=np.float64)
    stale_days = np.zeros(day_count, dtype=np.int64)
    neutral_days = np.zeros(day_count, dtype=np.int64)
    forced_count = np.zeros(day_count, dtype=np.int64)
    position_history = np.zeros((day_count, name_count), dtype=np.int8)

    shares = np.zeros(name_count, dtype=np.float64)
    marks = np.full(name_count, np.nan, dtype=np.float64)
    pending_exit = np.zeros(name_count, dtype=np.bool_)
    missing_sessions = np.zeros(name_count, dtype=np.int64)
    previous_nav = 1.0

    for day in range(day_count):
        start_nav = previous_nav
        prior_shares = shares.copy()
        prior_marks = marks.copy()
        held = prior_shares != 0
        prior_values = np.zeros(name_count, dtype=np.float64)
        prior_values[held] = np.abs(prior_shares[held] * prior_marks[held])
        long_value = float(prior_values[prior_shares > 0].sum())
        short_value = float(prior_values[prior_shares < 0].sum())

        printed = np.isfinite(close[day]) & (close[day] > 0)
        mark_pnl = 0.0
        neutral_flow = 0.0
        for name in np.flatnonzero(held):
            if printed[name]:
                mark_pnl += shares[name] * (close[day, name] - marks[name])
                if (
                    config.ex_date_marking == "neutral"
                    and day > 0
                    and event[day, name]
                    and np.isfinite(close[day - 1, name])
                    and close[day - 1, name] > 0
                    and np.isfinite(median[day])
                ):
                    neutral_flow += shares[name] * (
                        close[day - 1, name] * np.exp(median[day]) - close[day, name]
                    )
                    neutral_days[day] += 1
                elif (
                    config.ex_date_marking == "neutral"
                    and day > 0
                    and event[day, name]
                    and np.isfinite(close[day - 1, name])
                    and close[day - 1, name] > 0
                ):
                    raise ValueError(
                        "neutral event marking requires a finite market return"
                    )
                marks[name] = close[day, name]
                missing_sessions[name] = 0
            else:
                missing_sessions[name] += 1
                stale_days[day] += 1

        traded_notional = 0.0
        forced_pnl = 0.0
        must_force = (
            held
            & ~printed
            & ((~membership[day]) | (missing_sessions >= config.max_missing_sessions))
        )
        for name in np.flatnonzero(must_force):
            liquidation_mark = marks[name] * (
                1.0 - config.forced_liquidation_haircut
                if shares[name] > 0
                else 1.0 + config.forced_liquidation_haircut
            )
            forced_pnl += shares[name] * (liquidation_mark - marks[name])
            traded_notional += abs(shares[name] * liquidation_mark)
            shares[name] = 0.0
            marks[name] = np.nan
            pending_exit[name] = False
            missing_sessions[name] = 0
            forced_count[day] += 1

        eligible = score_valid[day] & membership[day] & np.isfinite(score[day])
        eligible_names = np.flatnonzero(eligible)
        ranks = np.full(name_count, -1, dtype=np.int64)
        if eligible_names.size:
            order = eligible_names[
                np.argsort(score[day, eligible_names], kind="stable")
            ]
            ranks[order] = np.arange(order.size)
        else:
            order = eligible_names
        width = config.k_per_side + config.buffer_per_side
        for name in np.flatnonzero(shares != 0):
            kept = (
                eligible[name]
                and not pending_exit[name]
                and (
                    ranks[name] >= len(order) - width
                    if shares[name] > 0
                    else ranks[name] <= width - 1
                )
            )
            if not kept:
                pending_exit[name] = True

        for name in np.flatnonzero((shares != 0) & pending_exit & printed):
            traded_notional += abs(shares[name] * close[day, name])
            shares[name] = 0.0
            marks[name] = np.nan
            pending_exit[name] = False
            missing_sessions[name] = 0

        if day < day_count - 1 and len(order) >= 2 * config.k_per_side:
            slot_notional = start_nav * config.gross_target / (2 * config.k_per_side)
            occupied = shares != 0
            long_slots = config.k_per_side - int((shares > 0).sum())
            short_slots = config.k_per_side - int((shares < 0).sum())
            for name in order[::-1]:
                if long_slots <= 0:
                    break
                if occupied[name] or not printed[name]:
                    continue
                shares[name] = slot_notional / close[day, name]
                marks[name] = close[day, name]
                occupied[name] = True
                traded_notional += slot_notional
                long_slots -= 1
            for name in order:
                if short_slots <= 0:
                    break
                if occupied[name] or not printed[name]:
                    continue
                shares[name] = -slot_notional / close[day, name]
                marks[name] = close[day, name]
                occupied[name] = True
                traded_notional += slot_notional
                short_slots -= 1

        if day == day_count - 1:
            for name in np.flatnonzero(shares != 0):
                if printed[name]:
                    liquidation_mark = close[day, name]
                else:
                    liquidation_mark = marks[name] * (
                        1.0 - config.forced_liquidation_haircut
                        if shares[name] > 0
                        else 1.0 + config.forced_liquidation_haircut
                    )
                    forced_pnl += shares[name] * (liquidation_mark - marks[name])
                    forced_count[day] += 1
                traded_notional += abs(shares[name] * liquidation_mark)
                shares[name] = 0.0
                marks[name] = np.nan
                pending_exit[name] = False
                missing_sessions[name] = 0

        interest = cdi[day] * (
            start_nav - long_value + config.short_proceeds_remuneration * short_value
        )
        borrow = config.annual_borrow_rate / config.annual_sessions * short_value
        costs = config.cost_bps_per_side / 10_000.0 * traded_notional
        pnl = mark_pnl + forced_pnl + neutral_flow + interest - borrow - costs
        current_nav = start_nav + pnl
        if not np.isfinite(current_nav):
            raise ValueError("ledger produced a non-finite NAV")
        nav[day] = current_nav
        daily_return[day] = current_nav / start_nav - 1.0
        excess_bps[day] = 10_000.0 * (daily_return[day] - cdi[day])
        gross_pnl_bps[day] = 10_000.0 * (mark_pnl + forced_pnl) / start_nav
        neutral_flow_bps[day] = 10_000.0 * neutral_flow / start_nav
        interest_bps[day] = 10_000.0 * interest / start_nav
        cost_bps[day] = 10_000.0 * costs / start_nav
        borrow_bps[day] = 10_000.0 * borrow / start_nav
        turnover_fraction[day] = traded_notional / start_nav
        held_now = shares != 0
        gross_fraction[day] = (
            float(np.abs(shares[held_now] * marks[held_now]).sum()) / current_nav
            if held_now.any() and current_nav != 0
            else 0.0
        )
        position_history[day, shares > 0] = 1
        position_history[day, shares < 0] = -1
        previous_nav = current_nav

    arrays = (
        nav,
        daily_return,
        excess_bps,
        gross_pnl_bps,
        neutral_flow_bps,
        interest_bps,
        cost_bps,
        borrow_bps,
        gross_fraction,
        turnover_fraction,
    )
    if not all(np.isfinite(value).all() for value in arrays):
        raise RuntimeError("ledger output contains a non-finite day")
    return StatefulLedgerResult(
        tuple(dates),
        nav,
        daily_return,
        excess_bps,
        gross_pnl_bps,
        neutral_flow_bps,
        interest_bps,
        cost_bps,
        borrow_bps,
        gross_fraction,
        turnover_fraction,
        stale_days,
        neutral_days,
        forced_count,
        position_history,
    )


def ledger_sensitivity_grid(
    **inputs: object,
) -> dict[str, StatefulLedgerResult]:
    """Run the registered financing/cost grid and policy sensitivities."""

    return {
        name: simulate_stateful_ledger(config=config, **inputs)  # type: ignore[arg-type]
        for name, config in ledger_configurations().items()
    }


def ledger_configurations() -> dict[str, LedgerConfig]:
    """Return the named registered headline grid and policy sensitivities."""

    headline = LedgerConfig()
    configurations = {
        f"cost_{cost:g}_borrow_{borrow:g}": replace(
            headline, cost_bps_per_side=cost, annual_borrow_rate=borrow
        )
        for cost in (2.0, 4.0, 7.0)
        for borrow in (0.02, 0.04)
    }
    configurations.update(
        {
            "sensitivity_buffer_0": replace(headline, buffer_per_side=0),
            "sensitivity_buffer_2k": replace(
                headline, buffer_per_side=2 * headline.k_per_side
            ),
            "sensitivity_short_proceeds_full": replace(
                headline, short_proceeds_remuneration=1.0
            ),
            "sensitivity_raw_marking": replace(headline, ex_date_marking="raw"),
            "sensitivity_haircut_30pct": replace(
                headline, forced_liquidation_haircut=0.30
            ),
        }
    )
    return configurations
