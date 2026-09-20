from dataclasses import replace

import numpy as np
import pytest
import torch

from brazil_rv.execution.portfolio_account import PortfolioAccount, tensor
from brazil_rv.execution.share_custody import ShareCustody
from brazil_rv.execution.share_distributions import (
    FractionAuction,
    ShareDelivery,
    ShareDistribution,
    slice_distributions,
)
from brazil_rv.execution.stateful_ledger import PortfolioTarget
from brazil_rv.v2.portfolio_objective import clone_account
from test_portfolio_account import compare
from test_portfolio_ledger import replay
from test_v2_stateful_ledger import _config


def event(*, early=True, basket=False):
    legs = (
        (
            ShareDelivery(
                1,
                1,
                5,
                loan_principal_fraction=0.2,
                disposal_session=3 if early else None,
            ),
            ShareDelivery(
                2,
                4,
                5,
                loan_principal_fraction=0.8,
                disposal_session=3 if early else None,
            ),
        )
        if basket
        else (
            ShareDelivery(
                1,
                0.333,
                5,
                FractionAuction(7, 310, 8),
                disposal_session=3 if early else None,
            ),
        )
    )
    return ShareDistribution(0, 3, 1, legs, "prearranged owned-sale hypothesis")


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("basket", [False, True])
def test_prearranged_disposal_preserves_credit_and_short_conversion(sign, basket):
    close = np.array([[100, 20, 20] if basket else [100, 300, 100]] * 10, float)
    close[3:, 0] = np.nan
    targets = np.zeros((10, 3))
    targets[:3, 0] = 0.4 * sign
    config = _config(initial_capital_brl=10000)
    books = []
    for early in (False, True):
        events = (event(early=early, basket=basket),)
        compare(close, targets, config=config, share_distributions=events)
        books.append(
            replay(
                close,
                lambda s: PortfolioTarget(targets[s.day]),
                config=config,
                share_distributions=events,
            )
        )
    late, early = books
    assert early.nav == pytest.approx(late.nav, abs=1e-10)
    fills = [f for f in early.fills if f.security_index in (1, 2)]
    assert min(f.fill_session for f in fills) == (3 if sign > 0 else 5)
    if sign > 0:
        assert early.free_cash[3:5] == pytest.approx([6000, 6000])
        assert early.unsettled_cash[3] == pytest.approx(4000 if basket else 3900)
        assert early.free_cash[5] == pytest.approx(10000 if basket else 9900)
        assert early.receivables[7] == pytest.approx(0 if basket else 0.32 * 310)
    else:
        np.testing.assert_array_equal(early.nav, late.nav)
        assert early.loan_payment == pytest.approx(late.loan_payment)
    rebased = slice_distributions((event(basket=basket),), 1, 10)[0]
    assert rebased.legs[0].disposal_session == 2
    assert rebased.legs[0].delivery_session == 4


def test_missing_partial_and_terminal_sales_preserve_real_receipts():
    close = np.array([[100, 300, 100]] * 7, float)
    close[3:, 0] = np.nan
    close[3, 1] = np.nan
    fractions = np.ones_like(close)
    fractions[4, 1] = 0.5
    targets = np.zeros_like(close)
    targets[:3, 0] = 0.4
    config = _config(initial_capital_brl=10000)
    compare(
        close,
        targets,
        config=config,
        fractions=fractions,
        share_distributions=(event(),),
    )
    book = replay(
        close,
        lambda s: PortfolioTarget(targets[s.day]),
        config=config,
        fill_fraction=fractions,
        share_distributions=(event(),),
    )
    assert book.signed_shares[3:5, 1] == pytest.approx([13, 6.5])
    assert book.free_cash[5] == pytest.approx(6000)
    assert book.undelivered_share_notional[-1] == pytest.approx(96)
    assert not [f for f in book.fills if f.fill_session == 3]


def test_presale_never_settles_before_credit_or_guesses_credit():
    custody = ShareCustody(2)
    custody.add(5, tensor([0, 10]))
    with pytest.raises(ValueError, match="sale settlement precedes"):
        custody.fill(tensor([0, 10]), tensor([0, 0]), tensor([0, 10]), 2, 4)
    with pytest.raises(ValueError, match="later known custody"):
        replace(event(), legs=(replace(event().legs[0], delivery_session=None),))


def test_early_economic_offset_defers_loan_return_proceeds_and_training_copy():
    def run(weight, copy_probe=False):
        account = PortfolioAccount.empty(
            [100, 100, 100],
            config=_config(annual_borrow_rate=0.04, initial_capital_brl=10000),
        )
        evt = ShareDistribution(
            0, 3, 1, (ShareDelivery(1, 1, 5, disposal_session=3),), "presale"
        )
        for day in range(7):
            account.prepare_day(day)
            target = account.weights.clone()
            if day == 0:
                target = torch.stack((weight, -weight, weight * 0))
            account.step(
                target,
                day=day,
                close=[100 if day < 3 else np.nan, 100, 100],
                cdi=0,
                session_date="2024-01-02",
                annual_borrow=[0.04, 0.04, 0],
                loan_reference=[100, 100, 100],
                share_distributions=(evt,),
            )
            if day == 3:
                assert account.shares[:2].abs().max().item() == pytest.approx(
                    0, abs=1e-10
                )
                assert account.restricted.sum().item() == pytest.approx(
                    float(weight.detach()) * 10000
                )
                assert account.loans.return_day.tolist() == [5]
                if copy_probe:
                    cloned = clone_account(account)
                    cloned.settlements[-1][2].add_(1)
                    assert not torch.equal(
                        cloned.settlements[-1][2], account.settlements[-1][2]
                    )
            if day == 5:
                assert account.restricted.sum().item() == pytest.approx(0)
        return account.nav

    weight = tensor(0.4).requires_grad_()
    nav = run(weight, True)
    nav.backward()
    expected = -10000 * (1.04 ** (5 / 252) - 1)
    assert weight.grad.item() == pytest.approx(expected, abs=1e-8)
    assert nav.item() == pytest.approx(10000 + 0.4 * expected, abs=1e-8)
