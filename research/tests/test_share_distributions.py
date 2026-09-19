from dataclasses import replace

import numpy as np
import pytest

from brazil_rv.execution.portfolio_account import PortfolioAccount, tensor
from brazil_rv.execution.share_distributions import (
    ShareDelivery,
    ShareDistribution,
    slice_distributions,
)
from brazil_rv.execution.stateful_ledger import PortfolioTarget
from test_portfolio_account import compare
from test_portfolio_ledger import replay
from test_v2_stateful_ledger import _config


def distribution(delivery=3):
    return ShareDistribution(
        source_index=0,
        effective_session=1,
        available_session=0,
        legs=(
            ShareDelivery(1, 1.0, 2, loan_principal_fraction=1 / 3),
            ShareDelivery(2, 2.0, delivery, loan_principal_fraction=2 / 3),
        ),
        cash_per_prior_share=10,
        payment_session=4,
        source="synthetic signed claim fixture",
    )


@pytest.mark.parametrize("sign", [-1, 1])
def test_separate_effective_delivery_and_payment_with_existing_positions(sign):
    close = np.array([[100, 30, 30]] + [[np.nan, 30, 30]] * 4)
    targets = (
        np.array([[0.4, -0.1, 0.1], [0, -0.1, 0.1], [0, 0, 0.1], [0, 0, 0], [0, 0, 0]])
        * sign
    )
    event = distribution()
    compare(close, targets, share_distributions=(event,), config=_config())
    states = []

    def policy(state):
        states.append(state)
        return PortfolioTarget(targets[state.day])

    result = replay(close, policy, share_distributions=(event,))
    assert result.nav == pytest.approx(np.ones(5))
    assert result.undelivered_share_notional == pytest.approx([0, 0.36, 0.24, 0, 0])
    assert result.receivables - result.payables == pytest.approx(
        np.array([0, 0.04, 0.04, 0.04, 0]) * sign
    )
    assert not [fill for fill in result.fills if fill.fill_session == 1]
    assert result.signed_shares[1, 0] == pytest.approx(sign * 0.004)
    assert result.signed_shares[-1] == pytest.approx(np.zeros(3))
    assert (
        result.free_cash[-1] + result.restricted_cash[-1] + result.unsettled_cash[-1]
    ) == pytest.approx(1)
    assert result.unpriced_inventory_notional == pytest.approx(np.zeros(5))
    assert states[2].locked.tolist() == [True, False, False]
    # The delivered first leg is in the decision state before current prints.
    assert states[2].weights == pytest.approx(np.array([0.24, 0.02, 0.1]) * sign)
    assert result.equity_market_weights[1] == pytest.approx(
        np.array([0, 0.02, 0.34]) * sign
    )
    assert len(result.share_claim_positions) == 3


@pytest.mark.parametrize("sign", [-1, 1])
def test_terminal_and_unknown_delivery_retain_valued_claim_and_financing(sign):
    close = np.array([[100, 30, 30], [np.nan, 31, 32], [np.nan, 32, 34]])
    event = replace(
        distribution(None),
        legs=(
            ShareDelivery(1, 1, None, loan_principal_fraction=1 / 3),
            ShareDelivery(2, 2, 7, loan_principal_fraction=2 / 3),
        ),
    )
    targets = np.array([[0.4, 0, 0], [0, 0, 0], [0, 0, 0]]) * sign
    config = _config(annual_borrow_rate=0.04)
    compare(
        close,
        targets,
        share_distributions=(event,),
        config=config,
        cdi=np.full(3, 0.0004),
    )
    result = replay(
        close,
        lambda state: PortfolioTarget(targets[state.day]),
        share_distributions=(event,),
        config=config,
        cdi=np.full(3, 0.0004),
    )
    assert result.undelivered_share_notional[-1] == pytest.approx(0.4)
    assert result.signed_shares[-1, 0] == pytest.approx(sign * 0.004)
    assert result.restricted_cash[-1] == pytest.approx(0.4 if sign < 0 else 0)
    assert not [fill for fill in result.fills if fill.fill_session > 0]
    assert result.economics_unresolved
    if sign < 0:
        assert result.borrow_bps[-1] > 0


def test_delivery_current_quote_mutation_does_not_change_prior_decision():
    event = distribution()
    states = []
    books = []
    for price in [30, 90]:
        close = np.array([[100.0, 30, 30]] + [[np.nan, 30, 30]] * 4)
        close[2, 1] = price
        snapshots = []

        def policy(state):
            snapshots.append(state)
            return PortfolioTarget(
                np.array([0.4, 0, 0]) if state.day == 0 else np.zeros(3)
            )

        books.append(replay(close, policy, share_distributions=(event,)))
        states.append(snapshots)
    assert states[0][2].weights == pytest.approx(states[1][2].weights)
    assert [
        order for order in books[0].intended_orders if order.decision_session <= 2
    ] == [order for order in books[1].intended_orders if order.decision_session <= 2]
    assert books[0].nav[2] != books[1].nav[2]


def test_basket_gradient_survives_marking_partial_delivery_and_detach():
    def run(weight, detach=False):
        account = PortfolioAccount.empty([100, 30, 30, 100], config=_config())
        import torch

        account.step(
            torch.stack((weight, weight * 0, weight * 0, weight * 0)),
            day=0,
            close=[100, 30, 30, 100],
            cdi=0,
            session_date="2024-01-02",
            annual_borrow=[0] * 4,
            loan_reference=[100] * 4,
        )
        for day, prices in [(1, [np.nan, 31, 32, 100]), (2, [np.nan, 32, 34, 100])]:
            account.prepare_day(day)
            account.step(
                account.weights,
                day=day,
                close=prices,
                cdi=0,
                session_date="2024-01-02",
                annual_borrow=[0] * 4,
                loan_reference=[100] * 4,
                share_distributions=(distribution(),),
            )
            if detach and day == 1:
                account.detach()
        assert account.market_weights[0].item() == 0
        return account.nav

    weight = tensor(0.4).requires_grad_()
    value = run(weight)
    value.backward()
    assert value.item() == pytest.approx(1.04)
    assert weight.grad.item() == pytest.approx(0.1)
    finite_difference = (run(tensor(0.400001)) - run(tensor(0.399999))) / 0.000002
    assert weight.grad.item() == pytest.approx(finite_difference.item(), abs=1e-9)
    assert run(tensor(0.4), detach=True).item() == pytest.approx(value.item())


def test_distribution_clock_rejects_backdating_and_rebases_without_truncating_claims():
    with pytest.raises(ValueError, match="backdated"):
        replace(distribution(), available_session=2)
    with pytest.raises(ValueError, match="precede"):
        replace(distribution(), legs=(ShareDelivery(1, 1, 0),))
    rebased = slice_distributions((distribution(),), 1, 3)[0]
    assert rebased.effective_session == 0
    assert rebased.available_session == -1
    assert rebased.payment_session == 3
    assert rebased.legs[1].delivery_session == 2
    prior = slice_distributions((distribution(),), 2, 5)[0]
    assert prior.effective_session == -1
    assert prior.source_index == 0


def test_flat_start_keeps_prior_cancellation_without_inventing_historical_claims():
    events = slice_distributions((distribution(),), 2, 5)
    states = []

    def policy(state):
        states.append(state)
        return PortfolioTarget(np.zeros(3))

    result = replay(np.full((3, 3), 100.0), policy, share_distributions=events)
    assert not states[0].entry_allowed[0]
    assert result.nav == pytest.approx(np.ones(3))
    assert not result.share_claim_positions
    assert result.receivables == pytest.approx(np.zeros(3))
    compare(np.full((3, 3), 100.0), np.zeros((3, 3)), share_distributions=events)


@pytest.mark.parametrize("on_fraction", [0.0, 0.2, 1.0])
def test_contract_principal_allocation_is_independent_of_constituent_marks(on_fraction):
    import torch

    close = np.array([[100.0, 10.0, 30.0]] + [[np.nan, 10.0, 30.0]] * 7)
    event = ShareDistribution(
        0,
        1,
        0,
        (
            ShareDelivery(1, 1, 2, loan_principal_fraction=on_fraction),
            ShareDelivery(2, 3, 3, loan_principal_fraction=1 - on_fraction),
        ),
        "explicit fixed-principal allocation fixture",
    )
    config = _config(annual_borrow_rate=0.04)
    targets = np.zeros((8, 3))
    targets[0, 0] = -0.4
    compare(close, targets, share_distributions=(event,), config=config)
    result = replay(
        close,
        lambda state: PortfolioTarget(targets[state.day]),
        share_distributions=(event,),
        config=config,
    )
    # Synthetic dates are in 2022: each exit returns on T+2. Both legs inherit
    # the original rate/principal, despite markedly different constituent prices.
    expected_rent = 0.4 * (
        on_fraction * (1.04 ** (4 / 252) - 1)
        + (1 - on_fraction) * (1.04 ** (5 / 252) - 1)
    )
    assert result.nav[-1] == pytest.approx(1 - expected_rent, abs=1e-12)
    scale = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    account = PortfolioAccount.empty(np.r_[close[0], 100.0], config=config)
    for day in range(8):
        target = torch.zeros(4, dtype=torch.float64)
        if day == 0:
            target[0] = -scale
        account.step(
            target,
            day=day,
            close=np.r_[close[day], 100.0],
            cdi=0.0,
            session_date=result.dates[day],
            loan_reference=np.full(4, 100.0),
            annual_borrow=np.full(4, 0.04),
            share_distributions=(event,),
            terminal=day == 7,
        )
    account.nav.backward()
    assert scale.grad.item() == pytest.approx(-expected_rent / 0.4, abs=1e-12)


def test_multi_leg_loan_terms_cannot_silently_use_price_allocations():
    with pytest.raises(ValueError, match="sum to one"):
        ShareDistribution(
            0, 1, 0, (ShareDelivery(1, 1, 2), ShareDelivery(2, 4, 2)), "terms"
        )
