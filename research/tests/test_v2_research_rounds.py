from __future__ import annotations

import numpy as np

from brazil_rv.v2.research_rounds import (
    RUNG_GROUPS,
    _folded_bootstrap,
    _paired_readouts,
)


def _report(offset: float) -> dict[str, object]:
    dates = [f"2024-01-{day:02d}" for day in range(1, 26)]
    metrics = []
    persistence = []
    for index, value in enumerate(dates):
        for horizon in (1, 2, 3, 5, 10):
            metrics.append(
                {
                    "date": value,
                    "horizon_sessions": horizon,
                    "raw_rank_ic": offset + index / 100.0,
                    "decile_spread_bps_per_holding_session": offset + index,
                }
            )
            for lag in (1, 5):
                persistence.append(
                    {
                        "date": value,
                        "horizon_sessions": horizon,
                        "lag_sessions": lag,
                        "spearman": offset + index / 1000.0,
                    }
                )
    return {
        "daily_primary_ic": [
            {"date": value, "mean_primary_horizon_ic": offset + index / 10.0}
            for index, value in enumerate(dates)
        ],
        "daily_metric_table": metrics,
        "persistence_table": persistence,
        "economics": {
            "daily_table": [
                {
                    "exit_date": value,
                    "cost_bps_per_side": 4.0,
                    "annual_borrow_rate": 0.02,
                    "net_excess_all_cash_bps": offset + index,
                }
                for index, value in enumerate(dates)
            ]
        },
        "input_hashes": {"scores": str(offset), "targets": "same"},
    }


def test_folded_bootstrap_never_crosses_fold_boundaries() -> None:
    result = _folded_bootstrap(
        (np.arange(25, dtype=np.float64), np.arange(25, 50, dtype=np.float64)),
        replications=100,
    )
    assert result["fold_boundary_preserved"] is True
    assert result["estimate"] == 24.5
    assert result["finite_observations"] == 50


def test_paired_readouts_cover_all_registered_families() -> None:
    baseline = {fold: _report(0.0) for fold in ("F1", "F2", "F3")}
    candidate = {fold: _report(1.0) for fold in ("F1", "F2", "F3")}
    paired = _paired_readouts(candidate, baseline)
    assert set(paired["pooled"]) == {
        "residual_ic",
        "raw_rank_ic",
        "persistence_1",
        "persistence_5",
        "decile_spread_bps_per_holding_session",
        "headline_net_excess_bps",
    }
    assert all(value["estimate"] == 1.0 for value in paired["pooled"].values())


def test_registered_gbdt_ladder_is_exact_and_cumulative() -> None:
    assert tuple(RUNG_GROUPS) == (
        "a_slow",
        "b_intraday",
        "c_lending",
        "d_all_sidecars",
    )
    assert RUNG_GROUPS["c_lending"] == ("lending",)
    assert RUNG_GROUPS["d_all_sidecars"] == (
        "lending",
        "oddlot",
        "options",
        "rebalance",
        "events",
        "fundamentals",
    )
