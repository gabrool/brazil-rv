from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from brazil_rv.execution.experiment52 import (
    FOLDS,
    _load_existing_report,
    daily_readout,
    execution_cells,
    rotation_designation,
    stored_daily_volatility,
)
from brazil_rv.execution.config import ExecutionConfig
from brazil_rv.execution.report import DailyExecutionResult
from brazil_rv.execution.report import write_execution_report
from brazil_rv.preprocessing.contract import (
    EQUITY_SESSION_MINUTES,
    PRICE_VOL_REFERENCE,
)


def test_frozen_grid_contains_only_requested_twelve_cells() -> None:
    cells = execution_cells()
    assert len(cells) == 12
    assert {float(cell["band"]) for cell in cells} == {0.0, 0.5, 1.0, 2.0}
    assert {cell["blend_name"] for cell in cells} == {
        "equal",
        "h30_only",
        "front_loaded",
    }
    for cell in cells:
        config = cell["config"]
        assert config["nav_brl"] == 10_000_000.0
        assert config["gross_target"] == 2.0
        assert config["participation_rate"] == 0.10
        assert config["fee_bps"] == 2.0
        assert config["taper_minutes"] == 30


def test_store_volatility_is_inverted_to_dimensionless_daily_units() -> None:
    regime = np.asarray([[0.0, np.log(2.0)]])
    expected = PRICE_VOL_REFERENCE * np.sqrt(EQUITY_SESSION_MINUTES)
    np.testing.assert_allclose(
        stored_daily_volatility(regime), [[expected, 2 * expected]]
    )


def test_daily_readout_uses_sample_standard_deviation_and_exact_cost_drag() -> None:
    daily = []
    start = date(2024, 1, 2)
    for index, net in enumerate((80.0, 100.0, 120.0)):
        daily.append(
            DailyExecutionResult(
                trade_date=start + timedelta(days=index),
                net_pnl_brl=net,
                gross_pnl_brl=net + 10.0,
                spread_cost_brl=6.0,
                fees_brl=4.0,
                cdi_earned_brl=0.0,
                turnover_brl=1_000.0,
                max_intraday_gross_brl=2_000.0,
                forced_fill_count=0,
            )
        )
    result = daily_readout(daily, 10_000.0)
    assert result["mean_daily_net_pnl_brl"] == 100.0
    assert result["std_daily_net_pnl_brl"] == 20.0
    assert result["daily_cost_drag_bps_of_nav"] == 10.0
    assert result["annualized_net_sharpe"] == np.sqrt(252.0) * 5.0


def test_rotation_designation_uses_other_folds_and_win_count() -> None:
    sharpes = {
        "steady": (2.0, 2.0, 2.0),
        "late_star": (3.0, 3.0, -10.0),
    }
    rows = [
        {
            "cell_id": cell,
            "fold": fold,
            "annualized_net_sharpe": values[index],
            "net_pnl_brl": 100.0 * values[index],
        }
        for cell, values in sharpes.items()
        for index, fold in enumerate(FOLDS)
    ]
    table, designation = rotation_designation(rows)
    assert len(table) == 6
    assert designation["c0_cell_id"] == "steady"
    assert designation["rotation_win_count"] == 2
    winners = {
        row["heldout_fold"]: row["cell_id"] for row in designation["rotation_winners"]
    }
    assert winners == {
        "fold_c": "steady",
        "fold_a": "steady",
        "fold_b": "late_star",
    }


def test_rotation_tie_break_uses_all_heldout_folds() -> None:
    rows = [
        {
            "cell_id": cell,
            "fold": fold,
            "annualized_net_sharpe": sharpe,
            "net_pnl_brl": sharpe,
        }
        for cell, values in {
            "left": (10.0, 0.0, -1.0),
            "right": (-1.0, 10.0, 1.0),
            "third": (0.0, -1.0, 10.0),
        }.items()
        for fold, sharpe in zip(FOLDS, values, strict=True)
    ]
    _, designation = rotation_designation(rows)
    assert designation["rotation_win_count"] == 1
    assert designation["c0_cell_id"] == "right"


def test_existing_report_resume_requires_exact_hashes(tmp_path) -> None:
    config = ExecutionConfig()
    inputs = {"source": "a" * 64}
    daily = [
        DailyExecutionResult(
            trade_date=date(2024, 1, 2),
            net_pnl_brl=1.0,
            gross_pnl_brl=2.0,
            spread_cost_brl=0.5,
            fees_brl=0.5,
            cdi_earned_brl=0.0,
            turnover_brl=10.0,
            max_intraday_gross_brl=20.0,
            forced_fill_count=1,
        )
    ]
    path = tmp_path / "report.json"
    expected = write_execution_report(
        path, config=config, input_sha256=inputs, daily=daily
    )

    loaded, artifact = _load_existing_report(path, config=config, input_sha256=inputs)

    assert loaded == daily
    assert artifact["sha256"] == expected["sha256"]
