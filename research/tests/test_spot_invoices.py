from dataclasses import replace
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN

import numpy as np
import pytest
import torch

from brazil_rv.execution.portfolio_account import PortfolioAccount, tensor
from brazil_rv.execution.spot_costs import execution_bps, spot_invoice_adjustment
from brazil_rv.execution.stateful_ledger import PortfolioTarget
from brazil_rv.v2.portfolio_objective import clone_account
from test_portfolio_account import compare
from test_portfolio_ledger import replay
from test_spot_costs import corporate


CONVENTION = "security_day_6dp_cent"


def decimal_invoice(amounts, rates):
    unit = Decimal(".000001")
    return np.array(
        [
            float(
                sum(
                    (
                        (
                            Decimal(str(a)).quantize(unit, rounding=ROUND_HALF_UP)
                            * Decimal(str(r))
                            / 10000
                        ).quantize(unit, rounding=ROUND_HALF_UP)
                        for a, r in zip(amounts, rate)
                    ),
                    Decimal(0),
                ).quantize(Decimal(".01"), rounding=ROUND_DOWN)
            )
            for rate in np.asarray(rates).T
        ]
    )


def test_original_normal_invoice_examples_and_account_grouping():
    rates = np.array([[0, 0.5, 2.5, 0, 1]] * 2)
    for amounts, trading in [
        ([5050, 0], 0.25),
        ([8550.4, 0], 0.42),
        ([5050, 8550.4, 1000.000002], 0.73),
    ]:
        amount = tensor(amounts)
        unrounded = amount.sum() * tensor([0.5, 2.5]) / 10000
        actual = unrounded + spot_invoice_adjustment(amount, rates, CONVENTION)
        expected = decimal_invoice(amounts, [[0.5, 2.5]] * len(amounts))
        assert actual.numpy() == pytest.approx(expected, abs=1e-15)
        assert float(actual[0]) == pytest.approx(trading, abs=1e-15)
    # Final truncation is account-wide, not a separate cent truncation per ISIN.
    amount = tensor([101, 101, 0])
    assert float(
        (
            amount.sum() * 0.5 / 10000
            + spot_invoice_adjustment(amount, rates, CONVENTION)[0]
        )
    ) == pytest.approx(0.01)


@pytest.mark.parametrize("start,lag", [("2019-02-04", 3), ("2024-01-02", 2)])
def test_both_accounts_original_spot_settlement_and_funding(start, lag):
    dates = np.busday_offset(np.datetime64(start), np.arange(7)).astype(object)
    cfg = corporate(
        spot_invoice_convention=CONVENTION, unrecovered_spot_trading_bps=0.5
    )
    close = np.full((7, 2), 100.0)
    targets = [[0.33333, -0.26721]] * 6 + [[0, 0]]
    rows = compare(close, targets, cdi=np.full(7, 0.001), dates=dates, config=cfg)
    result = replay(
        close,
        lambda s: PortfolioTarget(np.array(targets[s.day])),
        dates=dates,
        cdi=np.full(7, 0.001),
        config=cfg,
    )
    old = compare(
        close,
        targets,
        cdi=np.full(7, 0.001),
        dates=dates,
        config=replace(cfg, spot_invoice_convention="unrounded"),
    )
    for day, row in enumerate(rows):
        fills = [f for f in result.fills if f.fill_session == day]
        grouped = np.zeros(3)
        for fill in fills:
            grouped[fill.security_index] += fill.gross_notional
        rates = execution_bps(cfg, dates[day])
        expected = decimal_invoice(grouped, [rates[0, 1:3]] * 2 + [rates[1, 1:3]])
        np.testing.assert_allclose(
            result.execution_charges[day, 1:3], expected, rtol=0, atol=1e-13
        )
        np.testing.assert_allclose(
            row["spot_invoice_adjustment"],
            result.spot_invoice_adjustment[day],
            rtol=0,
            atol=1e-13,
        )
        assert sum(f.cost for f in fills) + result.spot_invoice_adjustment[
            day
        ].sum() == pytest.approx(float(row["cost"]), abs=1e-13)
    # Recognition changes current NAV, while funding remains unchanged until after value date.
    assert float(rows[0]["nav"]) != float(old[0]["nav"])
    for day in range(lag + 1):
        assert float(rows[day]["free_cash_interest"]) == pytest.approx(
            float(old[day]["free_cash_interest"]), rel=0, abs=5e-15
        )
    assert float(rows[lag + 1]["free_cash_interest"]) != float(
        old[lag + 1]["free_cash_interest"]
    )
    assert (
        result.unsettled_cash[-1] != 0
    )  # Terminal fills retain their spot obligations.


def test_exact_local_gradient_and_independent_pending_invoice_copies():
    cfg = corporate(spot_invoice_convention=CONVENTION, execution_shortfall_bps=0)

    def run(weight, probe=False):
        account = PortfolioAccount.empty([100, 100, 100], config=cfg)
        row = account.step(
            torch.stack((weight, -weight * 0.4, weight * 0.2)),
            day=0,
            close=[100, 100, 100],
            cdi=0,
            session_date="2024-01-02",
            annual_borrow=[0, 0, 0],
            loan_reference=[100, 100, 100],
        )
        if probe:
            copied = clone_account(account)
            due, free, restricted = copied.settlements[-1]
            original = account.settlements[-1][1].clone()
            free.add_(7)
            assert torch.equal(account.settlements[-1][1], original)
            copied.detach()
            assert not copied.settlements[-1][1].requires_grad
            assert copied.settlements[-1][0] == due
        return row["cost"]

    weight = tensor(0.31234567).requires_grad_()
    cost = run(weight, True)
    cost.backward()
    assert float(weight.grad) == pytest.approx(0, abs=1e-12)
    assert float(
        (run(tensor(0.31234568)) - run(tensor(0.31234566))) / 2e-8
    ) == pytest.approx(0, abs=1e-7)


def test_bundled_cost_cannot_be_rebilled_as_b3_invoice():
    with pytest.raises(ValueError, match="separated dated"):
        replace(
            corporate(), spot_cost_model="bundled", spot_invoice_convention=CONVENTION
        )
