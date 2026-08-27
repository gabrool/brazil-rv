from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from brazil_rv.execution.experiment54 import (
    ENTRIES,
    FOLDS,
    GROSS_LIMIT,
    HORIZONS,
    LIMIT_VARIANTS,
    NAME_CAP_FRACTION_OF_GROSS,
    NAV_BRL,
    THRESHOLDS_BPS,
    WAITS,
    _allocate_frontier,
    _maker_metrics,
    forward_edge_bps,
    quantile_edges,
    strict_through_fill_minutes,
    taker_decision,
)


def _events() -> dict[str, np.ndarray]:
    return {
        "day": np.array([0], dtype=np.int32),
        "refresh": np.array([0], dtype=np.int16),
        "name": np.array([0], dtype=np.int16),
        "minute": np.array([2], dtype=np.int16),
        "direction": np.array([1], dtype=np.int8),
    }


def test_experiment54_grid_constants_are_exact() -> None:
    assert HORIZONS == (15, 30, 60, 120)
    assert ENTRIES == (
        "decision_open",
        "next_open",
        "mean_open_10",
        "mean_open_30",
    )
    assert THRESHOLDS_BPS == (4.5, 7.0, 10.0)
    assert WAITS == (5, 15, 30)
    assert LIMIT_VARIANTS == ("last_close", "improved_half_half_spread")
    assert NAME_CAP_FRACTION_OF_GROSS * GROSS_LIMIT * NAV_BRL == 1_000_000.0


def test_quantile_edges_are_deterministic_and_ordered() -> None:
    values = np.arange(10, dtype=np.float64)
    assert np.allclose(
        quantile_edges(values, (0.2, 0.4, 0.6, 0.8)), [1.8, 3.6, 5.4, 7.2]
    )


def test_forward_entries_hold_after_each_completed_entry() -> None:
    open_price = np.arange(100.0, 150.0)[None, :, None]
    close = open_price + 1.0
    observed = np.ones_like(open_price, dtype=bool)
    sums = np.concatenate((np.zeros((1, 1, 1)), np.cumsum(open_price, axis=1)), axis=1)
    counts = np.concatenate(
        (
            np.zeros((1, 1, 1), dtype=np.int16),
            np.cumsum(observed, axis=1, dtype=np.int16),
        ),
        axis=1,
    )

    decision = forward_edge_bps(
        _events(),
        open_price=open_price,
        close_price=close,
        observed=observed,
        horizon=3,
        entry="decision_open",
    )
    next_open = forward_edge_bps(
        _events(),
        open_price=open_price,
        close_price=close,
        observed=observed,
        horizon=3,
        entry="next_open",
    )
    mean_10 = forward_edge_bps(
        _events(),
        open_price=open_price,
        close_price=close,
        observed=observed,
        horizon=3,
        entry="mean_open_10",
        open_sums=sums,
        open_counts=counts,
    )

    assert np.isclose(decision[0], (105.0 / 102.0 - 1.0) * 10_000.0)
    assert np.isclose(next_open[0], (106.0 / 103.0 - 1.0) * 10_000.0)
    assert np.isclose(
        mean_10[0], (115.0 / np.mean(np.arange(103.0, 113.0)) - 1.0) * 10_000.0
    )


def test_strict_through_rule_rejects_touch_and_uses_first_cross() -> None:
    high = np.full((1, 8, 2), 10.0)
    low = np.full((1, 8, 2), 10.0)
    observed = np.ones_like(high, dtype=bool)
    low[0, 2, 0] = 9.0
    low[0, 3, 0] = 8.9
    high[0, 2, 1] = 11.0
    high[0, 3, 1] = 11.1

    fill = strict_through_fill_minutes(
        direction=np.array([1, -1]),
        limit_price=np.array([9.0, 11.0]),
        day=np.array([0, 0]),
        decision_minute=np.array([1, 1]),
        name=np.array([0, 1]),
        high_price=high,
        low_price=low,
        observed=observed,
        wait=3,
    )

    assert fill.tolist() == [3, 3]


def test_frontier_respects_per_name_and_total_gross_capacity() -> None:
    names = 25
    events = {
        "day": np.zeros(names, dtype=np.int16),
        "refresh": np.zeros(names, dtype=np.int16),
        "name": np.arange(names),
        "capacity_brl": np.full(names, 12_000_000.0),
    }
    rows = _allocate_frontier(
        events=events,
        expected_net_bps=np.full(names, 2.0),
        eligible=np.ones(names, dtype=bool),
        dates=(date(2024, 1, 2),),
    )

    assert rows[0]["allocated_notional_brl"] == 20_000_000.0
    assert rows[0]["allocated_event_count"] == 20
    assert rows[0]["expected_net_pnl_brl"] == 4_000.0


def test_taker_decision_uses_best_registered_horizon_per_fold() -> None:
    rows = []
    for fold, best in zip(FOLDS, (7.9, 7.8, 7.7), strict=True):
        for horizon in HORIZONS:
            rows.append(
                {
                    "fold": fold,
                    "horizon_minutes": horizon,
                    "threshold_bps": 7.0,
                    "mean_net_nav_bps_per_day": best - horizon / 10_000,
                }
            )
    decision = taker_decision(pl.DataFrame(rows))
    assert decision["outcome"] == "CLOSED"

    rows[0]["mean_net_nav_bps_per_day"] = 8.1
    rows[4]["mean_net_nav_bps_per_day"] = 8.2
    decision = taker_decision(pl.DataFrame(rows))
    assert decision["outcome"] == "VIABLE"


def test_maker_metrics_use_strict_fill_and_unconditional_matched_edge() -> None:
    session_minutes = 405
    shape = (1, session_minutes, 1)
    open_price = np.full(shape, 100.0)
    close_price = np.full(shape, 101.0)
    high_price = np.full(shape, 101.0)
    low_price = np.full(shape, 100.0)
    low_price[0, 4, 0] = 99.0
    observed = np.ones(shape, dtype=bool)
    state = {
        "state_cell_id": np.array([0]),
        "rank_decile": np.array([9]),
        "delta_quintile": np.array([4]),
        "tail_entry": np.array([True]),
        "liquidity_tercile": np.array([2]),
        "tod_bucket": np.array([0]),
        "head_agreement": np.array([True]),
        "spread_tercile": np.array([1]),
        "sigma_tercile": np.array([1]),
    }
    events = {
        **state,
        "day": np.array([0]),
        "refresh": np.array([0]),
        "name": np.array([0]),
        "minute": np.array([2]),
        "direction": np.array([1]),
        "tail": np.array([True]),
        "full_spread_bps": np.array([10.0]),
        "taker_cost_measured_bps": np.array([7.0]),
        "capacity_brl": np.array([1_000_000.0]),
    }

    grouped, _, frontier = _maker_metrics(
        fold="C",
        dates=(date(2024, 1, 2),),
        events=events,
        unconditional_edge=np.array([50.0]),
        open_price=open_price,
        close_price=close_price,
        high_price=high_price,
        low_price=low_price,
        observed=observed,
        last_close=np.full(shape, 100.0),
        horizon=15,
        wait=5,
        variant="last_close",
    )

    row = grouped.row(0, named=True)
    assert row["fill_rate"] == 1.0
    assert row["mean_time_to_fill_minutes"] == 2.0
    assert np.isclose(row["mean_unconditional_matched_alpha_bps"], 50.0)
    assert frontier["mean_allocated_notional_brl_per_day"].item() == 1_000_000.0
