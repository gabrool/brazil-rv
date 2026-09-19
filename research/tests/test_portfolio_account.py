import numpy as np
import pytest
import torch

from brazil_rv.execution.portfolio_account import PortfolioAccount, tensor
from brazil_rv.execution.stateful_ledger import (
    PortfolioTarget,
    equity_borrow_registration_fee,
)
from brazil_rv.v2.corporate_actions import AlignedActionTerms
from test_portfolio_ledger import replay
from test_v2_stateful_ledger import _config


def compare(
    close,
    targets,
    *,
    cdi=None,
    actions=None,
    payments=None,
    fractions=None,
    config=None,
):
    close = np.asarray(close, dtype=float)
    days, names = close.shape
    cdi = np.zeros(days) if cdi is None else cdi
    config = _config(cost_bps_per_side=4) if config is None else config

    def target_policy(state):
        return PortfolioTarget(np.asarray(targets[state.day]))

    exact = replay(
        close,
        target_policy,
        cdi=cdi,
        actions=actions,
        payment_session=payments,
        fill_fraction=fractions,
        config=config,
    )
    account = PortfolioAccount.empty(np.full(names + 1, 100.0), config=config)
    records = []
    for day in range(days):
        fee = equity_borrow_registration_fee(config.annual_borrow_rate, config=config)
        borrow = np.expm1(np.log1p(config.annual_borrow_rate) / 252) + np.expm1(
            np.log1p(fee) / 252
        )
        record = account.step(
            tensor([*targets[day], 0.0]),
            day=day,
            close=np.r_[close[day], 100.0],
            cdi=float(cdi[day]),
            daily_borrow=np.r_[np.full(names, borrow), 0.0],
            action_q=None
            if actions is None
            else np.r_[actions.shares_per_prior_share[day], 1.0],
            action_d=None
            if actions is None
            else np.r_[actions.cash_per_prior_share[day], 0.0],
            action_resolved=None
            if actions is None
            else np.r_[actions.session_resolved[day], True],
            successor=None
            if actions is None
            else np.r_[actions.successor_index[day], names],
            payment_session=None if payments is None else np.r_[payments[day], -1],
            fill_fraction=None if fractions is None else np.r_[fractions[day], 1.0],
            terminal=day == days - 1,
        )
        assert account.shares.numpy()[:-1] == pytest.approx(
            exact.signed_shares[day], abs=1e-12
        )
        assert account.cash.item() == pytest.approx(exact.free_cash[day], abs=1e-12)
        assert account.restricted.sum().item() == pytest.approx(
            exact.restricted_cash[day], abs=1e-12
        )
        assert account.claims.sum().item() == pytest.approx(
            exact.receivables[day] - exact.payables[day], abs=1e-12
        )
        assert record["nav"].item() == pytest.approx(exact.nav[day], abs=1e-12)
        assert record["cost"].item() == pytest.approx(
            exact.cost_bps[day] * exact.start_nav[day] / 1e4, abs=1e-12
        )
        records.append(record)
    return records


def test_independent_cash_short_financing_gap_missing_and_reversal_parity():
    close = np.array(
        [[100, 100], [120, 90], [np.nan, 95], [125, 92], [121, 89]], dtype=float
    )
    targets = [[0.4, -0.4], [0.35, -0.35], [-0.3, 0.3], [-0.2, 0.2], [0, 0]]
    fractions = np.ones_like(close)
    fractions[1] = 0.5
    for remuneration in [0.0, 1.0]:
        config = _config(
            cost_bps_per_side=4,
            annual_borrow_rate=0.02,
            short_proceeds_remuneration=remuneration,
            annual_debit_spread=0.03,
        )
        compare(
            close, targets, cdi=np.full(5, 0.0004), fractions=fractions, config=config
        )


def test_split_and_delayed_dividend_claim_parity():
    close = np.array(
        [[100, 100], [50, 100], [48, 100], [49, 100], [51, 100]], dtype=float
    )
    q = np.ones_like(close)
    q[1, 0] = 2
    d = np.zeros_like(close)
    d[2, 0] = 2
    actions = AlignedActionTerms(
        q,
        d,
        np.ones_like(close, bool),
        (q != 1) | (d != 0),
        np.broadcast_to([0, 1], close.shape),
    )
    payments = np.full(close.shape, -1)
    payments[2, 0] = 3
    compare(
        close,
        [[0.4, -0.4]] * 5,
        actions=actions,
        payments=payments,
        cdi=np.full(5, 0.0004),
    )


def test_identity_conversion_transfers_inventory_and_exit_instruction():
    close = np.array(
        [[100, 100, np.nan], [np.nan, 100, 50], [np.nan, 100, 53], [np.nan, 100, 51]]
    )
    q = np.ones_like(close)
    q[1, 0] = 2
    successor = np.broadcast_to(np.arange(3), close.shape).copy()
    successor[1, 0] = 2
    actions = AlignedActionTerms(
        q, np.zeros_like(close), np.ones_like(close, bool), q != 1, successor
    )
    compare(
        close,
        [[0.4, -0.4, 0], [0, -0.2, 0], [0, -0.4, 0.4], [0, 0, 0]],
        actions=actions,
    )


@pytest.mark.parametrize("missing_name", [0, 1])
def test_long_quote_outage_and_terminal_inventory_parity(missing_name):
    close = np.full((13, 2), 100.0)
    close[1:, missing_name] = np.nan
    targets = [[0.4, -0.4]] + [[0, 0]] * 12
    records = compare(
        close,
        targets,
        cdi=np.full(13, 0.0004),
        config=_config(cost_bps_per_side=4, annual_borrow_rate=0.02),
    )
    assert records[-1]["unpriced_inventory_notional"].item() == pytest.approx(0.4)
    if missing_name == 1:
        assert records[-1]["borrow"].item() > 0.0


def test_sequential_gradient_includes_future_inventory_payoff_and_chunk_carry():
    config = _config(cost_bps_per_side=4)

    def run(value, *, detach=False):
        account = PortfolioAccount.empty([100, 100, 100], config=config)
        account.step(
            torch.stack((value, -value, value * 0)),
            day=0,
            close=[100, 100, 100],
            cdi=0.0004,
            daily_borrow=[0, 0, 0],
        )
        if detach:
            account.detach()
        account.step(
            account.weights,
            day=1,
            close=[110, 95, 100],
            cdi=0.0004,
            daily_borrow=[0, 0, 0],
        )
        return account.nav

    value = tensor(0.3).requires_grad_()
    nav = run(value)
    analytical = torch.autograd.grad(nav, value)[0].item()
    numerical = (run(tensor(0.300001)).item() - run(tensor(0.299999)).item()) / 0.000002
    assert analytical == pytest.approx(numerical, rel=1e-6)
    assert analytical > 0.14
    assert run(value, detach=True).item() == pytest.approx(nav.item(), abs=1e-12)


def test_joint_hedge_financing_missing_fill_and_reversal_parity():
    close = np.full((4, 2), 100.0)
    hedge_close = np.array([100, np.nan, 97, 101])
    targets = [[0.4, -0.2, -0.2], [0.2, -0.4, 0.2], [0.2, -0.4, 0.2], [0, 0, 0]]
    config = _config(beta_hedge=True, cost_bps_per_side=4, annual_borrow_rate=0.02)

    def policy(state):
        row = targets[state.day]
        return PortfolioTarget(np.array(row[:-1]), row[-1])

    exact = replay(
        close,
        policy,
        config=config,
        cdi=np.full(4, 0.0004),
        hedge_close=hedge_close,
        hedge_beta=np.ones_like(close),
    )
    account = PortfolioAccount.empty([100, 100, 100], config=config)
    rate = np.expm1(np.log1p(0.02) / 252)
    for day in range(4):
        row = account.step(
            tensor(targets[day]),
            day=day,
            close=[100, 100, hedge_close[day]],
            cdi=0.0004,
            daily_borrow=[rate] * 3,
            terminal=day == 3,
        )
        assert row["nav"].item() == pytest.approx(exact.nav[day], abs=1e-12)
        assert account.cash.item() == pytest.approx(exact.free_cash[day], abs=1e-12)
        assert account.shares[:-1].numpy() == pytest.approx(
            exact.signed_shares[day], abs=1e-12
        )
