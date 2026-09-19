from dataclasses import replace

import numpy as np
import pytest
import torch

from brazil_rv.execution.portfolio_account import PortfolioAccount, tensor
from brazil_rv.execution.portfolio_policy import PortfolioTarget
from brazil_rv.execution.share_distributions import (
    FractionAuction,
    ShareDelivery,
    ShareDistribution,
)
from test_portfolio_account import compare
from test_portfolio_ledger import replay
from test_v2_stateful_ledger import _config


def event(delivery=3, *, cash=310, carry_zero=False):
    return ShareDistribution(
        0,
        1,
        0,
        (
            ShareDelivery(
                1, 0.333, delivery, FractionAuction(5, cash, 6, True, carry_zero)
            ),
        ),
        "provisioned fraction fixture with settled source returns",
    )


@pytest.mark.parametrize(
    "capital,carry_zero", [(10000, False), (500, False), (500, True)]
)
def test_signed_fraction_separates_tradable_quantity_original_principal_and_payment(
    capital, carry_zero
):
    close = np.array([[100, 300]] + [[np.nan, 300]] * 7)
    targets = np.array([[-0.4, 0]] + [[0, 0]] * 7)
    config = _config(initial_capital_brl=capital, annual_borrow_rate=0.04)
    terms = event(carry_zero=carry_zero)
    compare(close, targets, share_distributions=(terms,), config=config)
    book = replay(
        close,
        lambda s: PortfolioTarget(targets[s.day]),
        share_distributions=(terms,),
        config=config,
    )
    quantity = 0.4 * capital / 100
    whole = np.floor(quantity * 0.333)
    fraction = quantity * 0.333 - whole
    end = 5 if whole else 6 if carry_zero else 2
    rent = 0.4 * capital * (1.04 ** (end / 252) - 1)
    assert book.loan_payment.sum() == pytest.approx(rent, abs=1e-9)
    assert book.nav[-1] == pytest.approx(
        capital * 1.4 - whole * 300 - fraction * 310 - rent, abs=1e-9
    )
    assert book.signed_shares[3, 0] == pytest.approx(-fraction / 0.333)
    assert book.payables[5] - book.receivables[5] == pytest.approx(fraction * 310)
    assert book.payables[6] == 0
    fills = [f for f in book.fills if f.security_index == 1]
    assert sum(f.quantity for f in fills) == pytest.approx(whole)
    assert book.restricted_cash[5] == pytest.approx(
        0.4 * capital * fraction / (quantity * 0.333)
    )
    assert book.restricted_cash[6] == pytest.approx(0)


def test_fraction_rounding_is_per_original_loan_not_net_holding():
    close = np.array([[100, 300]] * 2 + [[np.nan, 300]] * 6)
    targets = np.array([[-0.2, 0], [-0.4, 0]] + [[0, 0]] * 6)
    terms = replace(event(4), effective_session=2)
    config = _config(initial_capital_brl=10000)
    compare(close, targets, share_distributions=(terms,), config=config)
    book = replay(
        close,
        lambda s: PortfolioTarget(targets[s.day]),
        share_distributions=(terms,),
        config=config,
    )
    # Two 20-share original loans each become 6 whole + .66 provisioned PRIO.
    assert sum(
        f.quantity for f in book.fills if f.security_index == 1
    ) == pytest.approx(12)
    assert book.signed_shares[4, 0] == pytest.approx(-1.32 / 0.333)
    assert book.nav[-1] == pytest.approx(14000 - 12 * 300 - 1.32 * 310)


def account_path(weight, cash=310, *, detach=False):
    account = PortfolioAccount.empty(
        [100, 300, 100],
        config=_config(initial_capital_brl=10000, annual_borrow_rate=0.04),
    )
    for day in range(8):
        target = torch.zeros(3, dtype=torch.float64)
        if day == 0:
            target[0] = -weight
        account.step(
            target,
            day=day,
            close=[100 if day == 0 else np.nan, 300, 100],
            cdi=0.0004,
            session_date="2024-01-02",
            annual_borrow=[0.04, 0.04, 0],
            loan_reference=[100, 300, 100],
            share_distributions=(event(cash=cash),),
        )
        if detach and day == 3:
            account.detach()
        if day == 4:
            prior = float(account.nav.detach())
    return account.nav, prior


def test_auction_value_is_causal_and_fraction_gradient_matches_finite_difference():
    weight = tensor(0.4).requires_grad_()
    value, prior = account_path(weight)
    value.backward()
    finite = (
        account_path(tensor(0.4000001))[0] - account_path(tensor(0.3999999))[0]
    ) / 0.0000002
    assert weight.grad.item() == pytest.approx(float(finite), abs=2e-5)
    assert account_path(tensor(0.4), cash=410)[1] == prior
    assert float(account_path(tensor(0.4), detach=True)[0]) == pytest.approx(
        float(value), abs=1e-10
    )


@pytest.mark.parametrize("sign", [-1, 1])
def test_known_fraction_payment_and_proceeds_release_are_simultaneous_before_decision(
    sign,
):
    close = np.array([[100, 300]] + [[np.nan, 300]] * 7)
    states = []

    def policy(state):
        states.append(state)
        return PortfolioTarget(
            np.array([sign * 0.4, 0]) if state.day == 0 else np.zeros(2)
        )

    replay(
        close,
        policy,
        share_distributions=(event(),),
        config=_config(initial_capital_brl=10000),
    )
    assert states[6].free_cash_fraction == pytest.approx(1)
    assert states[6].restricted_cash_fraction == pytest.approx(0)
