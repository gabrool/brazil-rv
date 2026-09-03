from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from itertools import product
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DailySwingConfig:
    k_per_side: int = 30
    rank_band: float = 0.3
    gross_target: float = 2.0
    cost_bps_per_side: float = 4.0
    annual_borrow_rate: float = 0.02
    margin_fraction_of_gross: float = 0.5
    annual_sessions: int = 252
    terminal_liquidation: bool = True

    def __post_init__(self) -> None:
        if self.k_per_side <= 0 or self.annual_sessions <= 0:
            raise ValueError("K and annual session count must be positive")
        finite_nonnegative = (
            self.rank_band,
            self.gross_target,
            self.cost_bps_per_side,
            self.annual_borrow_rate,
            self.margin_fraction_of_gross,
        )
        if not all(
            math.isfinite(value) and value >= 0.0 for value in finite_nonnegative
        ):
            raise ValueError(
                "swing configuration values must be finite and non-negative"
            )
        if self.gross_target <= 0.0:
            raise ValueError("gross target must be positive")


@dataclass(frozen=True)
class DailySwingResult:
    annual_sessions: int
    signal_dates: tuple[date, ...]
    exit_dates: tuple[date, ...]
    weights: NDArray[np.float64]
    interval_valid: NDArray[np.bool_]
    missing_exit_position_count: NDArray[np.int64]
    gross_pnl_bps: NDArray[np.float64]
    turnover_fraction_nav: NDArray[np.float64]
    turnover_cost_bps: NDArray[np.float64]
    borrow_cost_bps: NDArray[np.float64]
    cdi_earned_bps: NDArray[np.float64]
    all_cash_cdi_bps: NDArray[np.float64]
    net_pnl_bps: NDArray[np.float64]
    net_excess_all_cash_bps: NDArray[np.float64]
    deployed_gross_fraction_nav: NDArray[np.float64]
    terminal_liquidation_valid: bool

    def summary(self) -> dict[str, float | int]:
        valid = self.interval_valid

        def mean(values: NDArray[np.float64]) -> float:
            selected = values[valid & np.isfinite(values)]
            return float(np.mean(selected)) if selected.size else math.nan

        def sharpe(values: NDArray[np.float64]) -> float:
            selected = values[valid & np.isfinite(values)]
            standard_deviation = (
                float(np.std(selected, ddof=1)) if selected.size > 1 else math.nan
            )
            return (
                float(
                    np.sqrt(float(self.annual_sessions))
                    * np.mean(selected)
                    / standard_deviation
                )
                if np.isfinite(standard_deviation) and standard_deviation > 0.0
                else math.nan
            )

        gross_sum = float(self.deployed_gross_fraction_nav[valid].sum())
        turnover_sum = float(self.turnover_fraction_nav[valid].sum())
        average_holding_sessions = (
            2.0 * gross_sum / turnover_sum
            if self.terminal_liquidation_valid and turnover_sum > 0.0
            else math.nan
        )
        return {
            "date_count": int(valid.sum()),
            "invalid_interval_count": int((~valid).sum()),
            "missing_exit_position_count": int(self.missing_exit_position_count.sum()),
            "mean_net_excess_all_cash_bps": mean(self.net_excess_all_cash_bps),
            "annualized_net_sharpe": sharpe(self.net_pnl_bps),
            "annualized_net_excess_sharpe": sharpe(self.net_excess_all_cash_bps),
            "mean_turnover_fraction_nav": mean(self.turnover_fraction_nav),
            "average_holding_sessions": average_holding_sessions,
            "mean_deployed_gross_fraction_nav": mean(self.deployed_gross_fraction_nav),
            "terminal_liquidation_valid": self.terminal_liquidation_valid,
        }


def _tail_weights(
    scores: NDArray[np.float64],
    eligible: NDArray[np.bool_],
    *,
    k_per_side: int,
    gross_target: float,
) -> NDArray[np.float64]:
    names = np.flatnonzero(eligible & np.isfinite(scores))
    result = np.zeros(scores.shape, dtype=np.float64)
    if names.size < 2 * k_per_side:
        return result
    order = names[np.argsort(scores[names], kind="stable")]
    side_gross = 0.5 * gross_target
    result[order[:k_per_side]] = -side_gross / k_per_side
    result[order[-k_per_side:]] = side_gross / k_per_side
    return result


def build_daily_weights(
    scores: NDArray[np.floating],
    score_mask: NDArray[np.bool_],
    active: NDArray[np.bool_],
    config: DailySwingConfig,
) -> NDArray[np.float64]:
    """Form each close-auction target from decision-time information only."""
    score_values = np.asarray(scores, dtype=np.float64)
    valid_scores = np.asarray(score_mask, dtype=bool)
    membership = np.asarray(active, dtype=bool)
    if (
        score_values.ndim != 2
        or valid_scores.shape != score_values.shape
        or membership.shape != score_values.shape
    ):
        raise ValueError("daily swing score inputs must share a date-by-name shape")
    weights = np.zeros_like(score_values, dtype=np.float64)
    for day in range(score_values.shape[0]):
        eligible = (
            valid_scores[day]
            & membership[day]
            & np.isfinite(score_values[day])
        )
        effective = score_values[day].copy()
        if config.rank_band > 0.0 and day > 0:
            both = (
                valid_scores[day - 1]
                & valid_scores[day]
                & np.isfinite(score_values[day - 1])
                & np.isfinite(score_values[day])
            )
            unchanged = both & (
                np.abs(score_values[day] - score_values[day - 1]) <= config.rank_band
            )
            effective[unchanged] = score_values[day - 1, unchanged]
        weights[day] = _tail_weights(
            effective,
            eligible,
            k_per_side=config.k_per_side,
            gross_target=config.gross_target,
        )
    return weights


def _turnover(
    weights: NDArray[np.float64], *, terminal_liquidation: bool
) -> NDArray[np.float64]:
    prior = np.vstack((np.zeros((1, weights.shape[1])), weights[:-1]))
    turnover = np.abs(weights - prior)
    if terminal_liquidation and len(turnover):
        turnover[-1] += np.abs(weights[-1])
    return turnover


def simulate_daily_swing(
    *,
    dates: Sequence[date],
    scores: NDArray[np.floating],
    score_mask: NDArray[np.bool_],
    active: NDArray[np.bool_],
    total_return_close: NDArray[np.floating],
    cdi_returns: NDArray[np.floating],
    config: DailySwingConfig = DailySwingConfig(),
) -> DailySwingResult:
    """Replay close-to-close daily targets with explicit full-cost accounting."""
    date_axis = tuple(dates)
    if (
        not date_axis
        or date_axis != tuple(sorted(date_axis))
        or len(set(date_axis)) != len(date_axis)
    ):
        raise ValueError(
            "daily swing dates must be nonempty, unique, and chronological"
        )
    close = np.asarray(total_return_close, dtype=np.float64)
    cdi = np.asarray(cdi_returns, dtype=np.float64)
    if (
        close.ndim != 2
        or close.shape[0] != len(date_axis)
        or cdi.shape != (len(date_axis),)
    ):
        raise ValueError("daily swing price or CDI axis differs from dates")
    if not np.isfinite(cdi).all():
        raise ValueError("daily CDI returns must be finite")
    # Freeze positions using only information available at the 15:45 decision.
    # Auction-close availability is an execution outcome, not an eligibility
    # input: a missing print invalidates the interval without substituting a
    # lower-ranked name after the fact.
    all_weights = build_daily_weights(scores, score_mask, active, config)
    weights = all_weights[:-1]
    turnover_by_name = _turnover(
        weights, terminal_liquidation=config.terminal_liquidation
    )
    held = weights != 0.0
    valid_endpoint = (
        np.isfinite(close[:-1])
        & (close[:-1] > 0.0)
        & np.isfinite(close[1:])
        & (close[1:] > 0.0)
    )
    missing = held & ~valid_endpoint
    interval_valid = ~missing.any(axis=1)
    terminal_liquidation_valid = bool(
        not config.terminal_liquidation
        or not len(weights)
        or not missing[-1].any()
    )
    returns = np.zeros_like(weights)
    np.divide(
        close[1:] - close[:-1],
        close[:-1],
        out=returns,
        where=valid_endpoint,
    )
    gross = (weights * returns).sum(axis=1) * 10_000.0
    turnover = turnover_by_name.sum(axis=1)
    turnover_cost = turnover * config.cost_bps_per_side
    borrow = (
        np.maximum(-weights, 0.0).sum(axis=1)
        * (config.annual_borrow_rate / config.annual_sessions)
        * 10_000.0
    )
    deployed_gross = np.abs(weights).sum(axis=1)
    cdi_earned = (
        cdi[1:]
        * np.maximum(
            1.0 - config.margin_fraction_of_gross * deployed_gross,
            0.0,
        )
        * 10_000.0
    )
    all_cash_cdi = cdi[1:] * 10_000.0
    net = gross - turnover_cost - borrow + cdi_earned
    excess = net - all_cash_cdi
    for values in (gross, turnover_cost, borrow, cdi_earned, net, excess):
        values[~interval_valid] = np.nan
    return DailySwingResult(
        annual_sessions=config.annual_sessions,
        signal_dates=date_axis[:-1],
        exit_dates=date_axis[1:],
        weights=weights,
        interval_valid=interval_valid,
        missing_exit_position_count=missing.sum(axis=1).astype(np.int64),
        gross_pnl_bps=gross,
        turnover_fraction_nav=turnover,
        turnover_cost_bps=turnover_cost,
        borrow_cost_bps=borrow,
        cdi_earned_bps=cdi_earned,
        all_cash_cdi_bps=all_cash_cdi,
        net_pnl_bps=net,
        net_excess_all_cash_bps=excess,
        deployed_gross_fraction_nav=deployed_gross,
        terminal_liquidation_valid=terminal_liquidation_valid,
    )


def swing_sensitivity_grid(
    *,
    dates: Sequence[date],
    scores: NDArray[np.floating],
    score_mask: NDArray[np.bool_],
    active: NDArray[np.bool_],
    total_return_close: NDArray[np.floating],
    cdi_returns: NDArray[np.floating],
    costs_bps: Sequence[float] = (2.0, 4.0, 7.0),
    annual_borrow_rates: Sequence[float] = (0.02, 0.04),
    base_config: DailySwingConfig = DailySwingConfig(),
) -> dict[tuple[float, float], DailySwingResult]:
    from dataclasses import replace

    return {
        (float(cost), float(borrow)): simulate_daily_swing(
            dates=dates,
            scores=scores,
            score_mask=score_mask,
            active=active,
            total_return_close=total_return_close,
            cdi_returns=cdi_returns,
            config=replace(
                base_config,
                cost_bps_per_side=float(cost),
                annual_borrow_rate=float(borrow),
            ),
        )
        for cost, borrow in product(costs_bps, annual_borrow_rates)
    }
