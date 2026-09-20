from dataclasses import replace
from decimal import Decimal

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
from brazil_rv.v2.portfolio_objective import clone_account


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("provision", [False, True])
def test_newly_known_auction_can_pay_after_intentions_without_early_income(
    sign, provision
):
    close = np.array([[100, 300]] + [[np.nan, 300]] * 8)
    targets = np.array([[sign * 0.4, 0]] + [[0, 0]] * 8)
    cdi = np.full(9, 0.001)
    config = _config(initial_capital_brl=1000)
    base = ShareDistribution(
        0,
        1,
        0,
        (ShareDelivery(1, 0.333, 3, FractionAuction(5, 300, 7, provision)),),
        "explicit earliest-known versus deadline fraction settlement fixture",
    )
    paths = []
    states = []
    for payment in (5, 7):
        leg = replace(
            base.legs[0], fractional_auction=FractionAuction(5, 300, payment, provision)
        )
        event = replace(base, legs=(leg,))
        compare(close, targets, cdi=cdi, config=config, share_distributions=(event,))
        snapshots = []

        def policy(state):
            snapshots.append(state)
            return PortfolioTarget(targets[state.day])

        paths.append(
            replay(close, policy, cdi=cdi, config=config, share_distributions=(event,))
        )
        states.append(snapshots)
    early, late = paths
    np.testing.assert_array_equal(early.nav[:6], late.nav[:6])
    assert states[0][5].free_cash_fraction == states[1][5].free_cash_fraction
    assert (
        states[0][5].restricted_cash_fraction == states[1][5].restricted_cash_fraction
    )
    # Four source units create1.332 successors. A continuous short has no auction
    # fraction; positive custody and provisioned original loans have .332 units.
    amount = (
        float(Decimal(sign) * Decimal(".332") * Decimal(300))
        if sign > 0 or provision
        else 0
    )
    assert late.receivables[5] - late.payables[5] == pytest.approx(amount)
    assert early.receivables[5] == early.payables[5] == 0
    assert late.receivables[7] == late.payables[7] == 0
    assert early.nav[6] - late.nav[6] == pytest.approx(amount * cdi[6], abs=1e-10)
    assert early.nav[7] - late.nav[7] == pytest.approx(
        amount * ((1.001**2) - 1), abs=1e-10
    )
    if sign < 0 and provision:
        assert early.restricted_cash[5] == 0
        assert late.restricted_cash[5] > 0
        assert late.restricted_cash[7] == 0


@pytest.mark.parametrize("sign", [-1, 1])
def test_same_day_sweep_gradient_and_copied_claim_state(sign):
    event = ShareDistribution(
        0,
        1,
        0,
        (ShareDelivery(1, 0.333, 3, FractionAuction(5, 310, 5, True)),),
        "same-day signed fraction settlement",
    )

    def run(weight, *, probe=False, detached=False):
        account = PortfolioAccount.empty(
            [100, 300, 100], config=_config(initial_capital_brl=1000)
        )
        for day in range(9):
            target = (
                torch.stack((sign * weight, weight * 0, weight * 0))
                if day == 0
                else torch.zeros(3, dtype=torch.float64)
            )
            account.step(
                target,
                day=day,
                close=[100 if day == 0 else np.nan, 300, 100],
                cdi=0.001,
                session_date="2024-01-02",
                annual_borrow=[0, 0, 0],
                loan_reference=[100, 300, 100],
                share_distributions=(event,),
            )
            if day == 4:
                if probe:
                    cloned = clone_account(account)
                    cloned.distributions[0].clear()
                    cloned.trade_cash += 7
                    assert account.distributions[0]
                    assert float(cloned.trade_cash - account.trade_cash) == 7
                if detached:
                    account.detach()
        return account.nav

    weight = tensor(0.4).requires_grad_()
    nav = run(weight, probe=True)
    nav.backward()
    finite = (run(tensor(0.4000001)) - run(tensor(0.3999999))) / 0.0000002
    assert weight.grad.item() == pytest.approx(float(finite), abs=3e-5)
    assert run(tensor(0.4), detached=True).item() == pytest.approx(
        nav.item(), abs=1e-10
    )
