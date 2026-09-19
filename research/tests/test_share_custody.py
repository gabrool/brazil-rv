"""Custody timing oracles: a beneficial offset is not a delivered loan return."""

import numpy as np
import pytest
import torch

from brazil_rv.execution.portfolio_account import PortfolioAccount, tensor
from brazil_rv.execution.share_custody import ShareCustody
from brazil_rv.execution.share_distributions import (
    FractionAuction,
    ShareDelivery,
    ShareDistribution,
)
from brazil_rv.v2.corporate_actions import AlignedActionTerms
from brazil_rv.execution.stateful_ledger import PortfolioTarget
from brazil_rv.v2.portfolio_objective import clone_account
from test_portfolio_account import compare
from test_portfolio_ledger import replay
from test_v2_stateful_ledger import _config


def succession(day=3):
    return ShareDistribution(0, day, 0, (ShareDelivery(1, 1, day),), "custody fixture")


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("long_purchase_day", [0, 2])
def test_offset_waits_for_long_purchase_value_date_in_both_accounts(
    sign, long_purchase_day
):
    # Short .4 on day 0; opposing .4 purchase either already delivered on day 2
    # or not delivered until day 4. Succession itself is on day 3.
    close = np.full((6, 2), 100.0)
    close[3:, 0] = np.nan
    config = _config(annual_borrow_rate=0.04, short_proceeds_remuneration=1)
    targets = []
    states = []

    def policy(state):
        states.append(state)
        weights = state.weights.copy()
        if state.day == 0:
            weights[0 if sign < 0 else 1] = -0.4
        if state.day == long_purchase_day:
            # Holding the short unchanged isolates custody from daily resizing.
            weights[1 if sign < 0 else 0] = 0.4 / state_nav[state.day]
        targets.append(weights)
        return PortfolioTarget(weights)

    # Prior-day NAV is known analytically before the offset; no CDI in this oracle.
    state_nav = np.array(
        [1.0] + [1 - 0.4 * (1.04 ** ((d - 1) / 252) - 1) for d in range(1, 6)]
    )
    book = replay(close, policy, share_distributions=(succession(),), config=config)
    targets.append(np.zeros(2))  # Terminal liquidation bypasses the policy callback.
    compare(close, targets, share_distributions=(succession(),), config=config)
    due = max(3, long_purchase_day + 2)
    expected_rent = 0.4 * (1.04 ** (due / 252) - 1)
    assert book.loan_payment[due] == pytest.approx(expected_rent, abs=1e-12)
    assert book.loan_payment.sum() == pytest.approx(expected_rent, abs=1e-12)
    assert book.signed_shares[3:] == pytest.approx(np.zeros((3, 2)), abs=1e-12)
    assert book.nav[-1] == pytest.approx(1 - expected_rent, abs=1e-12)
    assert book.restricted_cash[3] == pytest.approx(0.4 if due == 4 else 0)
    assert book.restricted_cash[due] == pytest.approx(0)
    # The recent long's purchase payable keeps its original date despite netting.
    assert book.unsettled_cash[3] == pytest.approx(-0.4 if due == 4 else 0)
    assert book.reconciliation_error == pytest.approx(np.zeros(6), abs=1e-12)
    assert not [f for f in book.fills if f.fill_session >= 3]


def test_sales_consume_settled_inventory_before_pending_receipts_and_transform_identity():
    custody = ShareCustody(3)
    custody.add(3, tensor([2, 0, 0]))
    custody.add(4, tensor([3, 0, 0]))
    custody.fill(tensor([10, 0, 0]), tensor([0, 0, 0]), tensor([6, 0, 0]), 2, 4)
    # Sell the five settled shares, then one of the earlier pending receipt.
    assert [q[0].item() for _, q in custody.receipts] == [1, 3]
    custody.split(0, 2)
    schedule = custody.deliver(0, 1, 0.5, 4.0, -2.0, 2, final=True)
    assert [(d, float(q[1])) for d, q in schedule] == [(2, 0), (3, 1), (4, 1)]
    assert sum(q[1].item() for _, q in custody.receipts) == 2
    assert sum(q[0].item() for _, q in custody.receipts) == 0
    custody.settle(4)
    assert not custody.receipts


def delayed_account(weight, *, clone=False):
    account = PortfolioAccount.empty(
        [100, 100, 100], config=_config(annual_borrow_rate=0.04)
    )
    for day in range(6):
        account.prepare_day(day)
        target = account.weights
        if day == 0:
            target = torch.stack((-weight, weight * 0, weight * 0))
        if day == 2:
            target = torch.stack((target[0], weight / account.nav, weight * 0))
        account.step(
            target,
            day=day,
            close=[100 if day < 3 else np.nan, 100, 100],
            session_date="2024-01-02",
            annual_borrow=[0.04, 0.04, 0],
            loan_reference=[100] * 3,
            cdi=0,
            share_distributions=(succession(),),
        )
        if clone and day == 2:
            copied = clone_account(account)
            copied.custody.receipts[0][1].add_(7)
            assert float(account.custody.receipts[0][1][1]) == pytest.approx(
                float(weight) / 100
            )
            account.detach()
    return account


def test_delayed_offset_gradient_and_independent_sam_state():
    weight = tensor(0.4).requires_grad_()
    account = delayed_account(weight)
    account.nav.backward()
    expected = -(1.04 ** (4 / 252) - 1)
    assert weight.grad.item() == pytest.approx(expected, abs=1e-12)
    finite = (
        delayed_account(tensor(0.40001)).nav - delayed_account(tensor(0.39999)).nav
    ) / 0.00002
    assert weight.grad.item() == pytest.approx(finite.item(), abs=1e-10)
    assert delayed_account(tensor(0.4), clone=True).nav.item() == pytest.approx(
        account.nav.item()
    )


@pytest.mark.parametrize("sign", [-1, 1])
def test_shareholder_fraction_waits_for_auction_while_loan_keeps_fraction(sign):
    close = np.array([[100, 300]] * 3 + [[np.nan, 300]] * 5)
    event = ShareDistribution(
        0,
        1,
        0,
        (ShareDelivery(1, 0.333, 3, FractionAuction(5, 310, 6)),),
        "separate shareholder auction and fractional loan fixture",
    )
    config = _config(initial_capital_brl=10000)
    targets = [[0.4 * sign, 0]] + [[0, 0]] * 7
    compare(close, targets, share_distributions=(event,), config=config)
    book = replay(
        close,
        lambda s: PortfolioTarget(np.array(targets[s.day])),
        share_distributions=(event,),
        config=config,
    )
    entitlement = 40 * 0.333
    fraction = entitlement - np.floor(entitlement)
    if sign > 0:
        assert book.signed_shares[3, 0] == pytest.approx(fraction / 0.333)
        assert book.undelivered_share_notional[3:5] == pytest.approx(
            [fraction * 300] * 2
        )
        assert book.receivables[5] == pytest.approx(fraction * 310)
        assert book.receivables[6] == pytest.approx(0)
        assert book.nav[-1] == pytest.approx(6000 + 13 * 300 + fraction * 310)
        assert book.free_cash[6] == pytest.approx(book.nav[-1])
    else:
        assert book.undelivered_share_notional[3:].sum() == 0
        assert book.receivables.sum() == 0
        assert book.nav[-1] == pytest.approx(14000 - entitlement * 300)
    fills = [f for f in book.fills if f.security_index == 1]
    assert len(fills) == 1
    assert fills[0].fill_session == 3
    assert fills[0].quantity == pytest.approx(13 if sign > 0 else entitlement)


def test_later_auction_price_does_not_change_prior_claim_values_or_orders():
    close = np.array([[100, 300]] * 3 + [[np.nan, 300]] * 5)
    books = []
    for price in [310, 400]:
        event = ShareDistribution(
            0,
            1,
            0,
            (ShareDelivery(1, 0.333, 3, FractionAuction(5, price, 6)),),
            "sourced fixture",
        )
        books.append(
            replay(
                close,
                lambda s: PortfolioTarget(np.array([0.4 if s.day == 0 else 0, 0])),
                config=_config(initial_capital_brl=10000),
                share_distributions=(event,),
            )
        )
    assert books[0].nav[:5] == pytest.approx(books[1].nav[:5])
    assert books[0].intended_orders == books[1].intended_orders
    assert books[1].nav[-1] - books[0].nav[-1] == pytest.approx(0.32 * 90)


def test_rename_converts_outstanding_return_even_when_net_inventory_is_flat():
    close = np.array([[100, 50], [100, 50], [np.nan, 50], [np.nan, 50], [np.nan, 50]])
    q, d = np.ones_like(close), np.zeros_like(close)
    q[2, 0] = 2
    successor = np.broadcast_to([0, 1], close.shape).copy()
    successor[2, 0] = 1
    actions = AlignedActionTerms(q, d, np.ones_like(close, bool), q != 1, successor)
    config = _config(annual_borrow_rate=0.04)
    targets = [[-0.4, 0]] + [[0, 0]] * 4
    compare(close, targets, actions=actions, config=config)
    result = replay(
        close,
        lambda s: PortfolioTarget(np.array(targets[s.day])),
        actions=actions,
        config=config,
    )
    assert result.nav[-1] == pytest.approx(1 - 0.4 * (1.04 ** (3 / 252) - 1))
    assert result.loan_payment[3] > 0
    assert result.signed_shares[1:] == pytest.approx(np.zeros((4, 2)))
