from datetime import date, timedelta

import numpy as np
import pytest
import torch

from brazil_rv.execution.portfolio_account import PortfolioAccount, tensor
from brazil_rv.execution.stateful_ledger import PortfolioTarget
from test_portfolio_ledger import replay
from test_v2_stateful_ledger import _config


def config():
    return _config(
        beta_hedge=True,
        planned_absolute_beta_cap=1.0,
        initial_capital_brl=10_000_000,
        borrow_fee_multiplier=1.0,
        hedge_cost_bps_per_side=0,
        hedge_annual_borrow_rate=0.02,
    )


@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_no_trade_roundtrip_does_not_open_minimum_fee_hedge(direction):
    days = 8
    cfg = config()
    targets = []

    def policy(state):
        weight = -0.399187832646 if state.day == 0 else state.hedge_weight
        if state.day and direction:
            weight = np.nextafter(weight, direction * np.inf)
        targets.append([0.0, weight])
        return PortfolioTarget(np.zeros(1), weight)

    dates = tuple(date(2019, 1, 2) + timedelta(days=i) for i in range(days))
    closes = np.array([87.9, 88.01, 88.5, 88.48, 88.83, 87.5, 89, 88])
    exact = replay(
        np.full((days, 1), 100.0),
        policy,
        config=cfg,
        hedge_beta=np.ones((days, 1)),
        hedge_close=closes,
        dates=dates,
        cdi=np.full(days, 0.0004),
    )
    hedges = [f for f in exact.fills if f.security_index == 1]
    assert [f.fill_session for f in hedges] == [0, days - 1]
    assert {c.opening_session for c in exact.loan_charges} == {0}
    account = PortfolioAccount.empty([100, 100], config=cfg)
    for day in range(days):
        # Exact historical intentions, allowing normal independent NAV ulps.
        target = tensor(targets[day]) if day < days - 1 else tensor([0, 0])
        row = account.step(
            target,
            day=day,
            close=[100, closes[day]],
            cdi=0.0004,
            session_date=dates[day],
            annual_borrow=[0, 0.02],
            loan_reference=[100, 100],
            terminal=day == days - 1,
        )
        assert float(row["nav"]) == pytest.approx(exact.nav[day], abs=1e-7, rel=0)
        assert set(account.loans.opened) <= {0}
        if day < days - 1:
            assert float(account.shares[-1]) == hedges[0].quantity * -1
    assert float(account.shares[-1]) == 0


@pytest.mark.parametrize("weight", [-1e-20, 1e-20])
def test_genuine_tiny_hedge_open_partial_reverse_and_terminal_survive(weight):
    values = [weight, weight / 2, -weight, 0]
    cfg = config()
    exact = replay(
        np.full((4, 1), 100.0),
        lambda state: PortfolioTarget(np.zeros(1), values[state.day]),
        config=cfg,
        hedge_beta=np.ones((4, 1)),
        hedge_close=np.full(4, 100.0),
    )
    fills = [f for f in exact.fills if f.security_index == 1]
    assert [f.fill_session for f in fills] == [0, 1, 2, 3]
    assert all(f.quantity > 0 for f in fills)
    account = PortfolioAccount.empty([100, 100], config=cfg)
    for day, weight in enumerate(values):
        account.step(
            tensor([0, weight]),
            day=day,
            close=[100, 100],
            cdi=0,
            session_date=exact.dates[day],
            annual_borrow=[0, 0.02],
            loan_reference=[100, 100],
            terminal=day == 3,
        )
        assert float(account.shares[-1]) == pytest.approx(
            exact.hedge_signed_shares[day], abs=0, rel=1e-13
        )


def test_hedge_roundoff_has_zero_local_gradient_but_real_change_keeps_gradient():
    account = PortfolioAccount.empty([100, 100], config=config())
    account.step(
        tensor([0, 0.2]),
        day=0,
        close=[100, 100],
        cdi=0,
        session_date="2024-01-02",
        annual_borrow=[0, 0],
        loan_reference=[100, 100],
    )
    before = account.weights[-1].detach()
    target = before.clone().requires_grad_()
    account.step(
        torch.stack((target * 0, target)),
        day=1,
        close=[100, 100],
        cdi=0,
        session_date="2024-01-03",
        annual_borrow=[0, 0],
        loan_reference=[100, 100],
    )
    assert torch.autograd.grad(account.shares[-1], target)[0].item() == 0
    target2 = (before + 0.01).requires_grad_()
    account.step(
        torch.stack((target2 * 0, target2)),
        day=2,
        close=[100, 100],
        cdi=0,
        session_date="2024-01-04",
        annual_borrow=[0, 0],
        loan_reference=[100, 100],
    )
    assert torch.autograd.grad(account.shares[-1], target2)[0].item() == pytest.approx(
        100000
    )
