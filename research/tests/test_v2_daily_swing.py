from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from brazil_rv.execution.daily_swing import (
    DailySwingConfig,
    build_daily_weights,
    simulate_daily_swing,
    swing_sensitivity_grid,
)
from brazil_rv.execution.experiment58 import (
    _cell_weights,
    _turnover_with_terminal_liquidation,
)


def test_daily_weights_match_experiment58_on_fully_observed_fixture() -> None:
    days = 4
    names = 60
    scores = np.stack(
        [np.linspace(-1.0, 1.0, names) + day * 0.05 for day in range(days)]
    )
    daily_move = np.linspace(-0.001, 0.001, names)
    close = 100.0 * (1.0 + np.arange(days)[:, None] * daily_move[None, :])
    active = np.ones_like(close, dtype=bool)
    score_mask = np.ones_like(close, dtype=bool)
    config = DailySwingConfig(k_per_side=30, rank_band=0.3)
    dates = tuple(date(2024, 1, 2) + timedelta(days=day) for day in range(days))
    cdi = np.asarray([0.0001, 0.0002, 0.0003, 0.0004])

    actual = build_daily_weights(scores, score_mask, active, config)
    experiment58 = _cell_weights(scores, close, active, 30, 0.3)
    replay = simulate_daily_swing(
        dates=dates,
        scores=scores,
        score_mask=score_mask,
        active=active,
        total_return_close=close,
        cdi_returns=cdi,
        config=config,
    )
    turnover = _turnover_with_terminal_liquidation(experiment58).sum(axis=1)
    returns = close[1:] / close[:-1] - 1.0
    gross = (experiment58 * returns).sum(axis=1) * 10_000.0
    borrow = np.maximum(-experiment58, 0.0).sum(axis=1) * 0.02 / 252 * 10_000
    deployed = np.abs(experiment58).sum(axis=1)
    cdi_earned = cdi[1:] * np.maximum(1.0 - 0.5 * deployed, 0.0) * 10_000
    net = gross - 4.0 * turnover - borrow + cdi_earned

    np.testing.assert_array_equal(actual[:-1], experiment58)
    np.testing.assert_array_equal(replay.weights, experiment58)
    np.testing.assert_allclose(replay.gross_pnl_bps, gross)
    np.testing.assert_allclose(replay.turnover_fraction_nav, turnover)
    np.testing.assert_allclose(replay.borrow_cost_bps, borrow)
    np.testing.assert_allclose(replay.cdi_earned_bps, cdi_earned)
    np.testing.assert_allclose(replay.net_pnl_bps, net)
    np.testing.assert_allclose(replay.net_excess_all_cash_bps, net - cdi[1:] * 10_000)


def test_daily_swing_full_cost_accounting_matches_hand_calculation() -> None:
    dates = (date(2024, 1, 2), date(2024, 1, 3))
    scores = np.asarray([[-2.0, -1.0, 1.0, 2.0]] * 2)
    close = np.asarray([[100.0] * 4, [90.0, 100.0, 100.0, 110.0]])
    mask = np.ones_like(scores, dtype=bool)
    config = DailySwingConfig(k_per_side=1, rank_band=0.0)

    result = simulate_daily_swing(
        dates=dates,
        scores=scores,
        score_mask=mask,
        active=mask,
        total_return_close=close,
        cdi_returns=np.asarray([0.0, 0.001]),
        config=config,
    )

    expected_borrow = 0.02 / 252 * 10_000
    np.testing.assert_allclose(result.weights, [[-1.0, 0.0, 0.0, 1.0]])
    np.testing.assert_allclose(result.gross_pnl_bps, [2_000.0])
    np.testing.assert_allclose(result.turnover_fraction_nav, [4.0])
    np.testing.assert_allclose(result.turnover_cost_bps, [16.0])
    np.testing.assert_allclose(result.borrow_cost_bps, [expected_borrow])
    np.testing.assert_allclose(result.cdi_earned_bps, [0.0])
    np.testing.assert_allclose(result.all_cash_cdi_bps, [10.0])
    np.testing.assert_allclose(
        result.net_excess_all_cash_bps,
        [2_000.0 - 16.0 - expected_borrow - 10.0],
    )
    assert result.summary()["average_holding_sessions"] == 1.0


def test_future_close_missingness_does_not_change_decision_weights() -> None:
    dates = tuple(date(2024, 1, 2) + timedelta(days=index) for index in range(3))
    scores = np.asarray(
        [
            [-2.0, -1.0, 1.0, 2.0],
            [-2.0, -1.0, 1.0, 2.0],
            [-2.0, -1.0, 1.0, 2.0],
        ]
    )
    mask = np.ones_like(scores, dtype=bool)
    close = np.full_like(scores, 100.0)
    config = DailySwingConfig(k_per_side=1, rank_band=0.0)

    before = build_daily_weights(scores, mask, mask, config)
    missing = close.copy()
    missing[1, 0] = np.nan
    after = build_daily_weights(scores, mask, mask, config)
    replay = simulate_daily_swing(
        dates=dates,
        scores=scores,
        score_mask=mask,
        active=mask,
        total_return_close=missing,
        cdi_returns=np.zeros(3),
        config=config,
    )

    np.testing.assert_array_equal(before[0], after[0])
    assert replay.interval_valid.tolist() == [False, False]
    assert replay.missing_exit_position_count.tolist() == [1, 1]
    assert np.isnan(replay.net_excess_all_cash_bps[0])
    assert replay.terminal_liquidation_valid is False


def test_same_day_close_outcome_cannot_change_decision_weights() -> None:
    dates = tuple(date(2024, 1, 2) + timedelta(days=index) for index in range(3))
    scores = np.tile(np.asarray([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]), (3, 1))
    mask = np.ones_like(scores, dtype=bool)
    close = np.full_like(scores, 100.0)
    config = DailySwingConfig(k_per_side=1, rank_band=0.0)

    expected = build_daily_weights(scores, mask, mask, config)
    changed_close = close.copy()
    changed_close[0, 0] = np.nan
    changed_close[1, -1] = -1.0
    actual = build_daily_weights(scores, mask, mask, config)
    replay = simulate_daily_swing(
        dates=dates,
        scores=scores,
        score_mask=mask,
        active=mask,
        total_return_close=changed_close,
        cdi_returns=np.zeros(3),
        config=config,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(replay.weights, expected[:-1])
    assert replay.interval_valid.tolist() == [False, False]
    assert replay.terminal_liquidation_valid is False


def test_sensitivity_grid_contains_the_frozen_six_cells() -> None:
    dates = tuple(date(2024, 1, 2) + timedelta(days=index) for index in range(3))
    scores = np.tile(np.linspace(-1.0, 1.0, 60), (3, 1))
    mask = np.ones_like(scores, dtype=bool)
    grid = swing_sensitivity_grid(
        dates=dates,
        scores=scores,
        score_mask=mask,
        active=mask,
        total_return_close=np.full_like(scores, 100.0),
        cdi_returns=np.zeros(3),
    )

    assert set(grid) == {
        (2.0, 0.02),
        (2.0, 0.04),
        (4.0, 0.02),
        (4.0, 0.04),
        (7.0, 0.02),
        (7.0, 0.04),
    }
