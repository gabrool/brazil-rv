import numpy as np
import pytest

from brazil_rv.execution.stateful_ledger import PortfolioTarget
from test_v2_stateful_ledger import _config, _run


def replay(close, policy, **kwargs):
    return _run(
        np.asarray(close, dtype=np.float64),
        np.ones_like(close),
        initial_reference_price=np.full(len(close[0]), 100.0),
        portfolio_policy=policy,
        **kwargs,
    )


def test_cash_earns_cdi_once_and_has_zero_excess():
    cdi = np.array([0.0003, 0.0005, 0.0004])
    result = replay(
        np.full((3, 2), 100.0),
        lambda state: PortfolioTarget(np.zeros(2)),
        cdi=cdi,
    )
    assert result.nav == pytest.approx(np.cumprod(1 + cdi), abs=1e-12)
    assert result.net_excess_all_cash_bps == pytest.approx(np.zeros(3), abs=1e-10)
    assert not result.fills
    assert not result.economics_unresolved


def test_current_close_mutation_changes_fills_but_not_intentions():
    def policy(state):
        return PortfolioTarget(np.array([0.4, -0.4]))

    prices = np.full((3, 2), 100.0)
    first = replay(prices, policy)
    prices[0] = [150, 80]
    second = replay(prices, policy)
    a = [x for x in first.intended_orders if x.decision_session == 0]
    b = [x for x in second.intended_orders if x.decision_session == 0]
    assert a == b
    assert first.fills[0].quantity != second.fills[0].quantity


def test_partial_reversal_cannot_open_before_exit_fills():
    states = []

    def policy(state):
        states.append(state)
        return PortfolioTarget(np.array([0.4, -0.4]) * (1 if state.day == 0 else -1))

    fractions = np.ones((4, 2))
    fractions[1] = 0.5
    result = replay(np.full((4, 2), 100.0), policy, fill_fraction=fractions)
    assert result.signed_shares[1] == pytest.approx([0.002, -0.002])
    assert not [x for x in result.fills if x.fill_session == 1 and x.purpose == "entry"]
    assert np.any(states[2].pending_exit_fractions > 0)
    assert result.signed_shares[2] == pytest.approx([-0.004, 0.004])
    assert result.signed_shares[-1] == pytest.approx([0, 0])


def test_missing_price_preserves_marked_inventory_and_costs_are_paid():
    prices = np.full((4, 2), 100.0)
    prices[1, 0] = np.nan

    def policy(state):
        return PortfolioTarget(np.array([0.4, -0.4]) if state.day == 0 else np.zeros(2))

    result = replay(prices, policy, config=_config(cost_bps_per_side=4))
    assert result.signed_shares[1, 0] == pytest.approx(0.004)
    assert result.mark_price[1, 0] == 100
    assert result.nav[-1] == pytest.approx(1 - 0.8 * 0.0004 * 2)
    assert result.reconciliation_error == pytest.approx(np.zeros(4), abs=1e-12)


def test_forbidden_new_risk_is_rejected_but_missing_score_can_reduce():
    mask = np.ones((3, 2), dtype=bool)
    mask[1, 0] = False

    def policy(state):
        return PortfolioTarget(np.array([0.4, -0.4]) if state.day == 0 else np.zeros(2))

    result = replay(np.full((3, 2), 100.0), policy, score_mask=mask)
    assert result.signed_shares[1] == pytest.approx([0, 0])
    mask[0, 0] = False
    with pytest.raises(ValueError, match="unavailable opening"):
        replay(np.full((3, 2), 100.0), policy, score_mask=mask)
