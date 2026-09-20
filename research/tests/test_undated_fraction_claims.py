from dataclasses import replace

import numpy as np
import pytest

from brazil_rv.execution.share_distributions import (
    FractionAuction,
    ShareDelivery,
    ShareDistribution,
    slice_distributions,
)
from brazil_rv.execution.stateful_ledger import PortfolioTarget
from brazil_rv.v2.targets import _basket_endpoint
from test_portfolio_account import compare
from test_portfolio_ledger import replay
from test_v2_stateful_ledger import _config


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("provision", [False, True])
def test_whole_shares_deliver_unknown_fraction_stays_signed_locked(sign, provision):
    auction = FractionAuction(None, None, None, provision)
    event = ShareDistribution(
        0,
        3,
        1,
        (ShareDelivery(1, 0.805, 5, auction),),
        "original promise; loan hypothesis",
    )
    close = np.array([[100.0, 125.0]] * 3 + [[np.nan, 125.0]] * 6)
    targets = np.zeros((9, 2))
    targets[:3, 0] = sign * 0.4
    config = _config(initial_capital_brl=10000)
    compare(close, targets, share_distributions=(event,), config=config)
    result = replay(
        close,
        lambda state: PortfolioTarget(targets[state.day]),
        share_distributions=(event,),
        config=config,
    )
    residual = sign * 0.2 if sign > 0 or provision else 0.0
    assert result.signed_shares[-1, 0] * 0.805 == pytest.approx(residual, abs=1e-12)
    assert result.undelivered_share_notional[-1] == pytest.approx(abs(residual) * 125)
    assert result.signed_shares[-1, 1] == pytest.approx(0.0)
    assert result.receivables[-1] == result.payables[-1] == 0
    assert not [
        f for f in result.fills if f.security_index == 0 and f.fill_session >= 3
    ]
    rebased = slice_distributions((event,), 1, 9)[0]
    assert rebased.legs[0].fractional_auction == auction
    assert rebased.legs[0].delivery_session == 4


def test_unknown_auction_never_becomes_a_post_delivery_gross_unit_endpoint():
    event = ShareDistribution(
        0,
        1,
        0,
        (ShareDelivery(1, 0.805, 3, FractionAuction(None, None, None)),),
        "source",
    )
    close = np.full((5, 2), 100.0)
    observed = np.ones_like(close, dtype=bool)
    q = np.ones_like(close)
    cash = np.zeros_like(close)
    successor = np.tile(np.arange(2), (5, 1))
    args = (close, observed, q, cash, observed, successor, {(1, 0): event})
    assert _basket_endpoint(0, 2, 0, *args) == pytest.approx((80.5, 0))
    assert _basket_endpoint(0, 3, 0, *args) is None
    with pytest.raises(ValueError, match="fully sourced"):
        FractionAuction(4, None, 5)
    with pytest.raises(ValueError, match="known payment"):
        replace(
            event.legs[0].fractional_auction, zero_quantity_rent_through_payment=True
        )


def test_undated_fraction_gradient_and_independent_training_copy():
    from datetime import date, timedelta
    import torch
    from brazil_rv.execution.portfolio_account import PortfolioAccount
    from brazil_rv.v2.portfolio_objective import clone_account

    config = _config(initial_capital_brl=10000)
    account = PortfolioAccount.empty([100.0, 125.0, 100.0], config=config)
    event = ShareDistribution(
        0,
        3,
        1,
        (ShareDelivery(1, 0.805, 5, FractionAuction(None, None, None)),),
        "source",
    )
    weight = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    for day in range(9):
        target = torch.zeros(3, dtype=torch.float64)
        if day < 3:
            target[0] = weight
        account.step(
            target,
            day=day,
            close=[100.0 if day < 3 else np.nan, 125.0, 100.0],
            cdi=0.0,
            session_date=date(2024, 1, 2) + timedelta(days=day),
            annual_borrow=[0.0, 0.0, 0.0],
            loan_reference=[100.0, 125.0, 100.0],
            share_distributions=(event,),
            terminal=day == 8,
        )
        if day == 5:
            clone = clone_account(account)
            clone.distributions[0].clear()
            assert (
                account.distributions[0][0].fractional_auction.available_session is None
            )
    account.nav.backward()
    assert weight.grad.item() == pytest.approx(62.5, abs=1e-9)


def test_undated_source_loader_retains_unknowns_and_model_objects():
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
                cash_per_prior_share=0.0,
                payment_date=None,
                legs=[
                    dict(
                        successor_isin=data.inputs.security_ids[1],
                        shares_per_prior_share=0.8,
                        delivery_date=str(calendar[4]),
                        loan_principal_fraction=1.0,
                        fractional_auction=dict(
                            available_date=None, payment_date=None, cash_per_share=None
                        ),
                    )
                ],
            )
        ],
    )
    changed = apply_corporate_replay(data.inputs, terms, calendar, "b" * 64)
    assert changed.share_distributions[0].legs[0].fractional_auction == FractionAuction(
        None, None, None
    )
    assert (
        changed.raw_close is data.inputs.raw_close
        and changed.scores is data.inputs.scores
    )
