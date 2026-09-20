from dataclasses import replace

import numpy as np
import pytest
import torch

from brazil_rv.execution.portfolio_account import PortfolioAccount, tensor
from brazil_rv.execution.share_distributions import (
    ShareDelivery,
    ShareDistribution,
    slice_distributions,
)
from brazil_rv.execution.stateful_ledger import PortfolioTarget
from brazil_rv.v2.portfolio_objective import clone_account
from test_portfolio_account import compare
from test_portfolio_ledger import replay
from test_v2_stateful_ledger import _config


def event(conversion, k=0.2):
    return ShareDistribution(
        0,
        3,
        1,
        tuple(
            ShareDelivery(
                i,
                ratio,
                5,
                loan_principal_fraction=fraction,
                loan_conversion_session=conversion,
            )
            for i, ratio, fraction in ((1, 1, k), (2, 4, 1 - k))
        ),
        "explicit net-borrowed conversion timing; shareholder credit5",
    )


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("conversion", [4, 5, 6])
def test_signed_conversion_clock_preserves_physical_long_credit(sign, conversion):
    close = np.array([[100, 20, 20]] * 10, float)
    close[3:, 0] = np.nan
    targets = np.zeros_like(close)
    targets[:3, 0] = 0.4 * sign
    config = _config(initial_capital_brl=10000, annual_borrow_rate=0.04)
    compare(close, targets, config=config, share_distributions=(event(conversion),))
    book = replay(
        close,
        lambda s: PortfolioTarget(targets[s.day]),
        config=config,
        share_distributions=(event(conversion),),
    )
    delivery = 5 if sign > 0 else conversion
    assert (
        min(f.fill_session for f in book.fills if f.security_index in (1, 2))
        == delivery
    )
    assert all(c.delivery_session == delivery for c in book.share_claim_positions)
    assert not [f for f in book.fills if f.security_index == 0 and f.fill_session >= 3]
    assert (
        book.restricted_cash[delivery] > 0
        if sign < 0
        else book.restricted_cash[delivery] == 0
    )


@pytest.mark.parametrize("owned_day", [0, 3, 4])
def test_converted_loan_offset_uses_owned_receipt_and_independent_copy(owned_day):
    def run(weight, probe=False):
        account = PortfolioAccount.empty(
            [100, 100, 100],
            config=_config(initial_capital_brl=10000, annual_borrow_rate=0.04),
        )
        evt = ShareDistribution(
            0, 3, 1, (ShareDelivery(1, 1, 5, loan_conversion_session=4),), "timing"
        )
        for day in range(8):
            account.prepare_day(day)
            target = account.weights.clone()
            if day == 0:
                target = torch.stack(
                    (-weight, weight if owned_day == 0 else weight * 0, weight * 0)
                )
            if day == owned_day and owned_day > 0:
                if day < 4:
                    # Match original units; current NAV already includes rent.
                    target[1] = -account.weights[0]
                else:
                    target = torch.zeros_like(target)  # Real cover after conversion.
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
            if day == 3 and probe:
                cloned = clone_account(account)
                cloned.distributions[0].clear()
                assert account.distributions[0][0].loan_conversion_session == 4
            if day == 4:
                assert account.restricted.sum().item() == pytest.approx(
                    0 if owned_day == 0 else 10000 * float(weight.detach())
                )
                if owned_day > 0:
                    assert account.loans.return_day.tolist() == [owned_day + 2]
        return account.nav

    weight = tensor(0.4).requires_grad_()
    nav = run(weight, True)
    nav.backward()
    return_day = max(4, owned_day + 2)
    expected = -10000 * (1.04 ** (return_day / 252) - 1)
    assert nav.item() == pytest.approx(10000 + 0.4 * expected, abs=1e-8)
    assert weight.grad.item() == pytest.approx(expected, abs=1e-8)


def test_missing_partial_covers_keep_converted_obligations():
    close = np.array([[100, 20, 20]] * 10, float)
    close[3:, 0] = np.nan
    close[4, 1] = np.nan
    fractions = np.ones_like(close)
    fractions[4, 2] = 0.5
    targets = np.zeros_like(close)
    targets[:3, 0] = -0.4
    config = _config(initial_capital_brl=10000, annual_borrow_rate=0.04)
    compare(
        close,
        targets,
        config=config,
        fractions=fractions,
        share_distributions=(event(4),),
    )
    book = replay(
        close,
        lambda s: PortfolioTarget(targets[s.day]),
        config=config,
        fill_fraction=fractions,
        share_distributions=(event(4),),
    )
    assert book.signed_shares[4, 1] < 0 and book.signed_shares[4, 2] < 0
    assert not [f for f in book.fills if f.fill_session == 4 and f.security_index == 1]
    assert book.restricted_cash[5] > 0
    assert book.loan_liability[-1] == pytest.approx(0)


def test_loader_and_slicing_keep_loan_and_shareholder_dates_separate():
    from brazil_rv.v2.corporate_replay import apply_corporate_replay
    from test_corporate_replay import fixture

    data, _, calendar = fixture()
    terms = dict(
        cash_cancellations=[],
        loan_cash_settlements=[],
        share_distributions=[
            dict(
                isin=data.inputs.security_ids[0],
                effective_date=str(calendar[2]),
                available_date=str(calendar[1]),
                cash_per_prior_share=0,
                payment_date=None,
                legs=[
                    dict(
                        successor_isin=data.inputs.security_ids[1],
                        shares_per_prior_share=1,
                        loan_principal_fraction=1,
                        delivery_date=str(calendar[4]),
                        loan_conversion_date=str(calendar[3]),
                    )
                ],
            )
        ],
    )
    changed = apply_corporate_replay(data.inputs, terms, calendar, "c" * 64)
    evt = slice_distributions(changed.share_distributions, 1, len(calendar))[0]
    assert evt.legs[0].loan_conversion_session == 2
    assert evt.legs[0].delivery_session == 3
    assert (
        changed.raw_close is data.inputs.raw_close
        and changed.scores is data.inputs.scores
    )
    with pytest.raises(ValueError, match="separate loan conversion"):
        replace(evt, legs=(replace(evt.legs[0], delivery_session=None),))
