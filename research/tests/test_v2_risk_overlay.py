import numpy as np
import pytest

from brazil_rv.v2.performance import sharpe
from brazil_rv.v2.risk_overlay import (
    apply_scale,
    circular_block_indices,
    combine_scales,
    evaluate_overlay,
    paired_bootstrap,
    summarize,
    trailing_external_volatility,
    trailing_volatility,
    volatility_target_scale,
)


def test_trailing_volatility_uses_only_sessions_strictly_before_each_day():
    rng = np.random.default_rng(5)
    returns = rng.normal(0, 0.01, size=120)
    vol = trailing_volatility(returns, 20)
    assert np.isnan(vol[:20]).all() and np.isfinite(vol[20:]).all()
    assert vol[20] == pytest.approx(returns[:20].std(ddof=1) * np.sqrt(252))
    shocked = returns.copy()
    shocked[50] = 0.5
    changed = trailing_volatility(shocked, 20)
    np.testing.assert_array_equal(changed[:51], vol[:51])
    assert changed[51] > vol[51]


def test_scale_is_bounded_and_unknown_volatility_keeps_full_exposure():
    vol = np.array([np.nan, 0.05, 0.10, 0.20, 0.40])
    scale = volatility_target_scale(vol, 0.10, floor=0.25)
    np.testing.assert_allclose(scale, [1.0, 1.0, 1.0, 0.5, 0.25])
    np.testing.assert_allclose(
        combine_scales(scale, np.full(5, 0.75)), [0.75, 0.75, 0.75, 0.5, 0.25]
    )
    with pytest.raises(ValueError):
        volatility_target_scale(vol, 0.10, floor=1.5)


def test_full_exposure_is_the_identity_and_rescaling_is_charged_once():
    excess = np.array([10.0, -5.0, 3.0, 2.0])
    same, turnover = apply_scale(excess, np.ones(4), previous_gross=2.0, cost_bps=4.0)
    np.testing.assert_allclose(same, excess)
    assert turnover.sum() == 0.0
    scaled, turnover = apply_scale(
        excess, np.array([1.0, 0.5, 0.5, 1.0]), previous_gross=2.0, cost_bps=4.0
    )
    np.testing.assert_allclose(turnover, [0.0, 1.0, 0.0, 1.0])
    np.testing.assert_allclose(scaled, [10.0, -2.5 - 4.0, 1.5, 2.0 - 4.0])
    with pytest.raises(ValueError):
        apply_scale(excess, np.array([1.0, 1.5, 0.5, 1.0]))


def test_summary_matches_repository_sharpe_conventions():
    rng = np.random.default_rng(8)
    excess = rng.normal(5, 80, size=500)
    cdi = np.full(500, 3.0)
    summary = summarize(excess, cdi)
    assert summary["sharpe_zero_rate"] == pytest.approx(sharpe((excess + cdi) / 1e4))
    assert summary["sharpe_cdi_excess"] == pytest.approx(sharpe(excess / 1e4))
    assert summary["mean_excess_bps"] == pytest.approx(excess.mean())
    assert summary["sessions"] == 500


def test_paired_bootstrap_covers_every_session_and_brackets_constant_deltas():
    indices = circular_block_indices(101, block=40, draws=50, seed=1)
    assert indices.shape == (50, 101)
    assert indices.min() >= 0 and indices.max() <= 100
    base = np.random.default_rng(2).normal(0, 50, size=300)
    cdi = np.full(300, 3.0)
    result = paired_bootstrap(base, base + 1.5, cdi, block=20, draws=200, seed=3)
    assert result["mean_excess_delta_bps"] == pytest.approx(1.5)
    assert result["mean_excess_delta_interval"] == pytest.approx([1.5, 1.5])
    lower, upper = result["sharpe_zero_rate_delta_interval"]
    assert lower <= upper


def test_overlay_evaluation_reports_identity_at_unit_scale_and_derisks_otherwise():
    rng = np.random.default_rng(9)
    daily = {
        "net_excess_bps": rng.normal(6, 80, size=400),
        "cdi_bps": np.full(400, 3.0),
        "gross": np.full(400, 2.0),
    }
    unit = evaluate_overlay(daily, np.ones(400), draws=100)
    assert unit["overlay"] == unit["base"]
    assert unit["sharpe_zero_rate_delta"] == 0.0
    assert unit["mean_rescale_cost_bps"] == 0.0
    scale = volatility_target_scale(
        trailing_volatility(daily["net_excess_bps"] / 1e4, 20), 0.05, floor=0.25
    )
    derisked = evaluate_overlay(daily, scale, draws=100)
    assert (
        derisked["overlay"]["annualized_volatility"]
        < unit["base"]["annualized_volatility"]
    )
    assert 0.0 < derisked["mean_scale"] < 1.0
    assert derisked["paired"]["block_sessions"] == 40


def test_external_volatility_aligns_strictly_before_each_book_date():
    book_dates = np.array(
        ["2020-01-06", "2020-01-07", "2020-01-08"], dtype="datetime64[D]"
    )
    series_dates = np.array(
        ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"], dtype="datetime64[D]"
    )
    returns = np.array([0.01, -0.02, 0.03, -0.01])
    vol = trailing_external_volatility(book_dates, series_dates, returns, 2)
    assert vol[0] == pytest.approx(returns[:2].std(ddof=1) * np.sqrt(252))
    assert vol[1] == pytest.approx(returns[1:3].std(ddof=1) * np.sqrt(252))
    assert vol[2] == pytest.approx(returns[2:4].std(ddof=1) * np.sqrt(252))
