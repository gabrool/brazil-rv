from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from brazil_rv.v2.research_rounds import (
    RUNG_GROUPS,
    _economics_not_worse,
    _folded_bootstrap,
    _paired_readouts,
    _point_is_negative,
    seal_root,
    _small_interval_spanning_zero,
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


def test_folded_bootstrap_records_undefined_readout_as_null() -> None:
    result = _folded_bootstrap(
        (np.full(25, np.nan), np.full(25, np.nan)),
        replications=100,
    )
    assert result["estimate"] is None
    assert result["lower_95"] is None
    assert result["upper_95"] is None
    assert result["finite_observations"] == 0


def test_undefined_economics_cannot_establish_not_worse() -> None:
    undefined = {"estimate": None, "lower_95": None, "upper_95": None}
    positive = {"estimate": 1.0, "lower_95": 0.5, "upper_95": 1.5}
    assert _point_is_negative(undefined) is False
    assert _economics_not_worse(undefined, positive) is False
    assert _economics_not_worse(positive, undefined) is False
    assert _small_interval_spanning_zero(undefined) is False


def test_paired_readouts_cover_all_registered_families() -> None:
    baseline = {fold: _report(0.0) for fold in ("F1", "F2", "F3")}
    candidate = {fold: _report(1.0) for fold in ("F1", "F2", "F3")}
    paired = _paired_readouts(candidate, baseline)
    assert set(paired["pooled"]) == {
        "median_residual_ic",
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


def test_superseded_root_seals_without_a_research_claim(tmp_path: Path) -> None:
    (tmp_path / "superseded.json").write_text(
        json.dumps(
            {
                "status": "superseded_by_fix_pass_3",
                "research_claim": False,
                "official_validation_accessed": False,
                "test_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    seal_root(root=tmp_path, research_claim=False)
    access = json.loads((tmp_path / "access_audit.json").read_text(encoding="utf-8"))
    inventory = json.loads(
        (tmp_path / "artifact_inventory.json").read_text(encoding="utf-8")
    )
    assert access["research_claim"] is False
    assert inventory["research_claim"] is False
