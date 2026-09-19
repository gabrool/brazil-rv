"""Value-date money oracles, independent of account-to-account agreement."""

import numpy as np
import pytest
import torch

from brazil_rv.execution.portfolio_account import PortfolioAccount, tensor
from brazil_rv.execution.stateful_ledger import PortfolioTarget
from brazil_rv.v2.corporate_actions import AlignedActionTerms
from brazil_rv.v2.portfolio_objective import clone_account
from test_portfolio_account import compare
from test_portfolio_ledger import replay
from test_v2_stateful_ledger import _config


@pytest.mark.parametrize("sign", [-1, 1])
def test_round_trip_before_settlement_keeps_dated_funding(sign):
    # Open .4 on day 0, close on day 1: money moves on days 2 and 3.
    targets = [[sign * 0.4], [0], [0], [0], [0]]
    cdi = np.array([0, 0, 0, 0.001, 0.001])
    close = np.full((5, 1), 100.0)
    config = _config(short_proceeds_remuneration=1)
    compare(close, targets, cdi=cdi, config=config)
    result = replay(
        close,
        lambda s: PortfolioTarget(np.array(targets[s.day])),
        cdi=cdi,
        config=config,
    )
    assert result.free_cash[:2] == pytest.approx([1, 1])
    assert result.unsettled_cash == pytest.approx([-sign * 0.4, 0, sign * 0.4, 0, 0])
    assert result.restricted_cash == pytest.approx([0, 0, 0.4 if sign < 0 else 0, 0, 0])
    assert result.short_proceeds_interest_base == pytest.approx(
        [0, 0, 0, 0.4 if sign < 0 else 0, 0]
    )
    assert result.nav == pytest.approx(
        [1, 1, 1, 1 + (1 - sign * 0.4) * 0.001, (1 + (1 - sign * 0.4) * 0.001) * 1.001]
    )
    assert result.free_cash[-1] == result.nav[-1]


def test_levered_purchase_debit_interest_starts_after_value_date():
    # Only the .2 actual settled debit is financed; a future purchase payable is not.
    targets = [[0.6, 0.6]] * 4 + [[0, 0]]
    close = np.full((5, 2), 100.0)
    config = _config(annual_debit_spread=0.252, planned_absolute_net_cap=1.3)
    compare(close, targets, config=config)
    result = replay(
        close, lambda s: PortfolioTarget(np.array(targets[s.day])), config=config
    )
    assert result.free_cash_interest_bps[:3] == pytest.approx([0, 0, 0])
    expense = result.free_cash_interest_bps * result.start_nav / 1e4
    assert expense[3] == pytest.approx(-0.2 * 0.001)
    assert expense[4] == pytest.approx(-0.2002 * 0.001)
    assert result.free_cash_income_bps == pytest.approx(np.zeros(5))
    assert result.debit_financing_bps == pytest.approx(-result.free_cash_interest_bps)


def test_same_value_date_sales_and_purchases_net_without_double_cash():
    targets = [[0.5, 0], [0.5, 0], [0, 0.5], [0, 0.5], [0, 0.5], [0, 0]]
    result = replay(
        np.full((6, 2), 100.0), lambda s: PortfolioTarget(np.array(targets[s.day]))
    )
    assert result.free_cash == pytest.approx([1, 1, 0.5, 0.5, 0.5, 0.5])
    assert result.unsettled_cash == pytest.approx([-0.5, -0.5, 0, 0, 0, 0.5])
    assert result.nav == pytest.approx(np.ones(6))
    assert result.signed_shares[-1] == pytest.approx([0, 0])
    assert result.unsettled_cash[-1] == pytest.approx(0.5)


def _differentiable_round_trip(weight):
    account = PortfolioAccount.empty(
        [100, 100], config=_config(short_proceeds_remuneration=1)
    )
    for day in range(5):
        account.step(
            torch.stack((weight if day == 0 else weight * 0, weight * 0)),
            day=day,
            close=[100, 100],
            cdi=0.001 if day >= 3 else 0,
            session_date="2024-01-02",
            annual_borrow=[0, 0],
            loan_reference=[100, 100],
        )
    return account


@pytest.mark.parametrize("weight", [-0.4, 0.4])
def test_settlement_gradients_equal_closed_form_and_finite_difference(weight):
    value = tensor(weight).requires_grad_()
    account = _differentiable_round_trip(value)
    account.nav.backward()
    assert value.grad.item() == pytest.approx(-0.001 * 1.001, abs=1e-12)
    delta = (
        _differentiable_round_trip(tensor(weight + 1e-5)).nav
        - _differentiable_round_trip(tensor(weight - 1e-5)).nav
    ) / 2e-5
    assert value.grad.item() == pytest.approx(delta.item(), abs=1e-10)


def test_sam_clone_and_tbptt_preserve_independent_pending_settlements():
    account = PortfolioAccount.empty([100, 100], config=_config())
    value = tensor(0.4).requires_grad_()
    account.step(
        torch.stack((value, value * 0)),
        day=0,
        close=[100, 100],
        cdi=0,
        session_date="2024-01-02",
        annual_borrow=[0, 0],
        loan_reference=[100, 100],
    )
    clone = clone_account(account)
    assert clone.unsettled_cash.item() == pytest.approx(-0.4)
    clone.settlements[0][1].add_(0.1)
    assert account.unsettled_cash.item() == pytest.approx(-0.4)
    before = account.nav.item(), account.cash.item(), account.unsettled_cash.item()
    account.detach()
    assert (
        account.nav.item(),
        account.cash.item(),
        account.unsettled_cash.item(),
    ) == before
    assert all(
        not f.requires_grad and not r.requires_grad for _, f, r in account.settlements
    )


@pytest.mark.parametrize("sign", [-1, 1])
def test_cash_only_event_keeps_proceeds_until_payment_and_does_not_backdate_income(
    sign,
):
    close = np.full((6, 1), np.nan)
    close[0] = 100
    q, d = np.ones_like(close), np.zeros_like(close)
    q[1], d[1] = 0, 100
    actions = AlignedActionTerms(
        q, d, np.ones_like(q, bool), q != 1, np.zeros_like(q, int)
    )
    payments = np.full(close.shape, -1)
    payments[1] = 4
    targets = [[sign * 0.4]] + [[0]] * 5
    cdi = np.array([0, 0, 0, 0, 0.001, 0.001])
    config = _config(short_proceeds_remuneration=1)
    compare(close, targets, actions=actions, payments=payments, cdi=cdi, config=config)
    result = replay(
        close,
        lambda s: PortfolioTarget(np.array(targets[s.day])),
        actions=actions,
        payment_session=payments,
        cdi=cdi,
        config=config,
    )
    assert result.receivables - result.payables == pytest.approx(
        np.array([0, 0.4, 0.4, 0.4, 0, 0]) * sign
    )
    assert result.restricted_cash == pytest.approx(
        [0, 0, 0.4 if sign < 0 else 0, 0.4 if sign < 0 else 0, 0, 0]
    )
    assert result.nav[-2:] == pytest.approx(
        [1 + (1 - sign * 0.4) * 0.001, (1 + (1 - sign * 0.4) * 0.001) * 1.001]
    )


def test_equal_long_short_principal_does_not_create_extra_cdi_income():
    cdi = np.full(8, 0.0004)
    result = replay(
        np.full((8, 2), 100.0),
        lambda s: PortfolioTarget(np.array([0.4, -0.4]) if s.day == 0 else s.weights),
        cdi=cdi,
        config=_config(short_proceeds_remuneration=1),
    )
    assert result.nav == pytest.approx(np.cumprod(1 + cdi), abs=1e-12)
    assert result.short_proceeds_interest_base[:3] == pytest.approx([0, 0, 0])
    assert result.short_proceeds_interest_base[3:] == pytest.approx([0.4] * 5)


def test_t3_t2_transition_net_value_date_in_the_same_account():
    account = PortfolioAccount.empty([100, 100, 100], config=_config())
    dates = ["2019-05-24", "2019-05-27", "2019-05-28", "2019-05-29"]
    cash = []
    for day, date in enumerate(dates):
        account.step(
            tensor([0.4, 0 if day == 0 else 0.4, 0]),
            day=day,
            session_date=date,
            close=[100, 100, 100],
            cdi=0,
            annual_borrow=[0, 0, 0],
            loan_reference=[100, 100, 100],
        )
        cash.append(account.cash.item())
    assert cash == pytest.approx([1, 1, 1, 0.2])
    assert account.unsettled_cash.item() == 0
    assert account.nav.item() == 1
