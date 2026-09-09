from dataclasses import replace

import numpy as np
import pytest

from brazil_rv.execution.stateful_ledger import (
    LedgerConfig,
    _scaled_group_bands,
    simulate_stateful_ledger,
)
from brazil_rv.v2.evaluate import (
    _aligned_action_terms,
    _economics_signal,
    _input_hashes,
    _ledger_inputs,
)
from brazil_rv.v2.execution_policy import (
    ExecutionPolicy,
    original_trade_attribution,
    prior_median_volume,
    smooth_and_rank,
    traded_readouts,
    traded_signal,
)
from test_v2_evaluate import _fixture


def test_smoothing_is_causal_and_expires_missing_names():
    values = np.tile([-0.5, 0.0, 0.5], (10, 1))
    valid = np.ones_like(values, dtype=bool)
    valid[1:8, 1] = False
    score, mask = smooth_and_rank(values, valid, theta=0.5)
    assert mask[5, 1] and not mask[6, 1] and mask[8, 1]
    changed = values.copy()
    changed[5:] *= -100
    after, _ = smooth_and_rank(changed, valid, theta=0.5)
    np.testing.assert_array_equal(score[:5], after[:5])
    # Carried state participates in ranking; a re-entering expired name starts fresh.
    assert score[8, 1] == 0


def test_volume_screen_uses_only_twenty_completed_valid_sessions():
    volume = np.full((25, 2), 30e6)
    volume[20:] = 0
    valid = np.ones_like(volume, dtype=bool)
    valid[5, 1] = False
    median = prior_median_volume(volume, valid, [19, 20, 24])
    assert np.isnan(median[0]).all()
    assert median[1, 0] == 30e6 and np.isnan(median[1, 1])
    changed = volume.copy()
    changed[20:] = 1e30
    np.testing.assert_array_equal(
        median[:2], prior_median_volume(changed, valid, [19, 20])
    )


def test_default_policy_preserves_book_and_protected_inputs():
    inputs = _fixture()
    before = _input_hashes(inputs)
    scores, mask = traded_signal(inputs, ExecutionPolicy())
    config = LedgerConfig()
    kwargs = dict(
        config=config, shortable=inputs.shortable_by_borrow_source["borrow_balance"]
    )
    baseline = simulate_stateful_ledger(
        **_ledger_inputs(inputs, *_economics_signal(inputs)), **kwargs
    )
    replay = simulate_stateful_ledger(
        **_ledger_inputs(inputs, scores, mask), **kwargs, capacity_buffer_per_side=30
    )
    assert baseline.summary() == replay.summary()
    assert baseline.intended_orders == replay.intended_orders
    np.testing.assert_array_equal(
        baseline.net_excess_all_cash_bps, replay.net_excess_all_cash_bps
    )
    traded_readouts(inputs, scores, mask)
    assert before == _input_hashes(inputs)


def test_buffer_capacity_is_fixed_and_retention_bands_do_not_overlap():
    sizes = np.full(5, 25)
    result = [
        _scaled_group_bands(
            group_sizes=sizes,
            threshold_multiple=2,
            k_eff=30,
            buffer=buffer,
            capacity_buffer=30,
        )
        for buffer in (15, 30, 45)
    ]
    for quota, retention in result:
        np.testing.assert_array_equal(quota, np.full(5, 6))
        assert np.all(2 * (quota + retention) <= sizes)


@pytest.mark.parametrize("liquid", [False, True])
def test_attribution_reconciles_actions_partial_fills_and_shared_costs(liquid):
    inputs = _fixture()
    close = inputs.raw_close.copy()
    close *= 1 + np.arange(len(close))[:, None] * 0.001
    inputs.action_has_action[5, :] = True
    inputs.action_shares_per_prior_share[5, :] = 2
    inputs.action_cash_per_prior_share[5, :] = 1
    close[5:] = (close[5:] - 1) / 2
    inputs = replace(inputs, raw_close=close, cdi_returns=np.full(len(close), 0.0003))
    config = LedgerConfig()
    result = simulate_stateful_ledger(
        **_ledger_inputs(inputs, *_economics_signal(inputs)),
        config=config,
        shortable=inputs.shortable_by_borrow_source["borrow_balance"],
        fill_fraction=np.full(close.shape, 0.7),
    )
    attribution = original_trade_attribution(
        result,
        entry_liquid=np.full(close.shape, liquid),
        entry_known=np.ones_like(close, dtype=bool),
        action_terms=_aligned_action_terms(inputs),
        annual_borrow_rate_by_name=inputs.annual_borrow_rate_by_name,
        config=config,
    )
    expected = result.net_excess_all_cash_bps if liquid else np.zeros(len(close))
    np.testing.assert_allclose(
        attribution["daily"]["net_excess_contribution_bps"], expected, atol=1e-7
    )


def test_inverse_sigma_preserves_side_budget_before_name_caps_and_is_causal():
    inputs = _fixture()
    # Enough names for a uniform four-slot book, varied sigma and no binding name cap.
    config = replace(
        LedgerConfig(),
        k_per_side=2,
        buffer_per_side=2,
        volatility_balanced_entries=False,
        beta_hedge=False,
        planned_absolute_net_cap=2.0,
        planned_name_weight_cap=1.0,
    )
    sigma = np.broadcast_to(np.linspace(0.01, 0.04, 60), inputs.active.shape).copy()
    kwargs = {
        **_ledger_inputs(inputs, *_economics_signal(inputs)),
        "config": config,
        "shortable": inputs.shortable_by_borrow_source["borrow_balance"],
        "initial_reference_price": np.full(60, 100.0),
    }
    result = simulate_stateful_ledger(**kwargs, entry_sizing_volatility=sigma)
    initial = [
        order
        for order in result.intended_orders
        if order.purpose == "entry" and order.decision_session == 0
    ]
    for side in ("buy", "sell"):
        orders = [order for order in initial if order.side == side]
        assert sum(order.planned_notional for order in orders) == pytest.approx(
            result.start_nav[0]
        )
        weights = [
            order.planned_notional * sigma[0, order.security_index] for order in orders
        ]
        assert weights[0] == pytest.approx(weights[1])
    changed = sigma.copy()
    changed[6:] *= 100
    after = simulate_stateful_ledger(**kwargs, entry_sizing_volatility=changed)
    assert [o for o in result.intended_orders if o.decision_session < 6] == [
        o for o in after.intended_orders if o.decision_session < 6
    ]


def test_entry_liquidity_is_not_reclassified_on_later_partial_fills():
    inputs = _fixture()
    score = np.broadcast_to(np.arange(60, dtype=float), inputs.active.shape)
    kwargs = {
        **_ledger_inputs(inputs, score, inputs.active),
        "initial_reference_price": np.full(60, 100.0),
    }
    config = LedgerConfig()
    result = simulate_stateful_ledger(
        **kwargs,
        config=config,
        shortable=inputs.shortable_by_borrow_source["borrow_balance"],
        fill_fraction=np.full(inputs.active.shape, 0.7),
    )
    tags = np.zeros_like(inputs.active)
    tags[0] = True
    kwargs = dict(
        result=result,
        entry_known=np.ones_like(tags),
        action_terms=_aligned_action_terms(inputs),
        annual_borrow_rate_by_name=inputs.annual_borrow_rate_by_name,
        config=config,
    )
    before = original_trade_attribution(**kwargs, entry_liquid=tags)
    after = original_trade_attribution(**kwargs, entry_liquid=np.ones_like(tags))
    # First order partially fills on both sessions; day-1's changed classification
    # cannot relabel its second fill or the inventory and costs it generates.
    assert (
        before["daily"]["net_excess_contribution_bps"][:2]
        == after["daily"]["net_excess_contribution_bps"][:2]
    )


def test_sweep_baseline_serialization_matches_evaluation():
    from brazil_rv.v2.evaluate import evaluate_scores
    from brazil_rv.v2.execution_sweep import _panel

    inputs = _fixture()
    source = evaluate_scores(inputs, window_name="F1").report
    panel = _panel(
        inputs, source, ExecutionPolicy(), np.full(inputs.active.shape, 30e6)
    )
    assert not any("baseline_daily" in failure for failure in panel["failed_gates"])
