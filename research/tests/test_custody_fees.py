from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest
import torch

from brazil_rv.execution.custody_fees import (
    CustodyAssessment,
    custody_schedule,
    monthly_custody_fee,
)
from brazil_rv.execution.portfolio_account import PortfolioAccount, tensor
from brazil_rv.execution.stateful_ledger import PortfolioTarget
from brazil_rv.v2.portfolio_objective import clone_account
from brazil_rv.v2.corporate_actions import AlignedActionTerms
from test_portfolio_account import compare
from test_portfolio_ledger import replay
from test_v2_stateful_ledger import _config


def config(pay="2024-01-07", **kw):
    return _config(
        initial_capital_brl=100000,
        custody_assessments=(CustodyAssessment("2024-01-04", pay),),
        **kw,
    )


def test_progressive_fee_and_dated_exemption_no_fund_discount():
    for year, threshold in ((2023, "23084.39"), (2024, "24164.73")):
        assert (
            float(monthly_custody_fee(float(Decimal(threshold) - 1), f"{year}-01-31"))
            == 0
        )
        assert float(
            monthly_custody_fee(float(threshold), f"{year}-01-31")
        ) == pytest.approx(float(Decimal(threshold) * Decimal(".0005") / 12))
        expected = (
            Decimal(100000) * Decimal(".0005")
            + Decimal(100000) * Decimal(".0004")
            + Decimal(100000) * Decimal(".0002")
            + Decimal(1400000) * Decimal(".00013")
            + Decimal(8300000) * Decimal(".000072")
        ) / 12
        assert float(monthly_custody_fee(10000000, f"{year}-12-30")) == pytest.approx(
            float(expected), abs=1e-13
        )
    with pytest.raises(ValueError, match="2023-2024"):
        CustodyAssessment("2022-12-30", "2023-01-02")


def test_calendar_uses_full_known_axis_and_preserves_unpaid_terminal_invoice():
    calendar = np.array(
        ["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02", "2024-02-05"]
    )
    assert custody_schedule(calendar, "2024-01-01", "2024-02-02", 3) == (
        CustodyAssessment("2024-01-31", "2024-02-05"),
    )
    assert custody_schedule(calendar[:1], "2024-01-01", "2024-01-30") == ()
    with pytest.raises(ValueError, match="future calendar"):
        custody_schedule(calendar, "2024-01-01", "2024-02-02", 4)


@pytest.mark.parametrize("payment", ["2024-01-04", "2024-01-07", "2024-02-01"])
def test_both_accounts_physical_sales_retained_until_delivery_and_fee_liability(
    payment,
):
    cfg = config(payment)
    targets = np.array([[0.6, -0.4], [0.6, -0.4], [0, 0], [0, 0], [0, 0], [0, 0]])
    close = np.full((6, 2), 100.0)
    rows = compare(close, targets, config=cfg, cdi=np.zeros(6))
    result = replay(
        close,
        lambda s: PortfolioTarget(targets[s.day]),
        config=cfg,
        cdi=np.zeros(6),
    )
    # The day2 sale has removed economic shares, but those 600 owned shares
    # remain in physical custody until day4. Same-value-day loan cover nets out.
    assert result.physical_custody[2, 0] == pytest.approx(600.0)
    assert result.custody_base[2] == pytest.approx(60000.0)
    assert result.custody_fee[2] == pytest.approx(2.5)
    assert result.custody_liability[-1] == (2.5 if payment == "2024-02-01" else 0)
    for day, row in enumerate(rows):
        assert row["physical_custody"].numpy() == pytest.approx(
            result.physical_custody[day], abs=1e-9
        )
        assert float(row["custody_liability"]) == pytest.approx(
            result.custody_liability[day]
        )
    control = replay(
        close,
        lambda s: PortfolioTarget(targets[s.day]),
        config=replace(cfg, custody_assessments=()),
        cdi=np.zeros(6),
    )
    np.testing.assert_array_equal(result.nav[:2], control.nav[:2])
    assert result.intended_orders[:4] == control.intended_orders[:4]


def test_delivered_new_loan_and_delayed_cover_physical_inventory_and_renewal():
    cfg = config(loan_term_sessions=5)
    account = PortfolioAccount.empty([100.0, 100.0], config=cfg)
    physical = []
    for day in range(8):
        row = account.step(
            tensor([-0.4 if day < 4 else 0.0, 0.0]),
            day=day,
            close=[100.0, 100.0],
            cdi=0.0,
            session_date=f"2024-01-{day + 2:02d}",
            annual_borrow=[0.0, 0.0],
            loan_reference=[100.0, 100.0],
            loan_return_session=8,
        )
        physical.append(float(row["physical_custody"][0]))
    assert physical == pytest.approx([0, 400, 0, 0, 0, 0, 400, 400], abs=1e-8)
    assert len(account.loans.renewals) > 0


def test_custody_gradient_prior_funding_and_independent_sam_tbptt_copies():
    def run(weight, probe=False):
        account = PortfolioAccount.empty([100.0, 100.0], config=config())
        for day in range(7):
            row = account.step(
                torch.stack((weight, weight * 0)),
                day=day,
                close=[100.0, 100.0],
                cdi=0.001,
                session_date=f"2024-01-{day + 2:02d}",
                annual_borrow=[0.0, 0.0],
                loan_reference=[100.0, 100.0],
            )
            if day == 2 and probe:
                copied = clone_account(account)
                copied.custody_fees.invoices[0][1].add_(7)
                assert float(
                    copied.custody_fees.liability - account.custody_fees.liability
                ) == pytest.approx(7)
                if copied.custody_fees.pending:
                    copied.custody_fees.pending[0][1].add_(1)
                    assert not torch.equal(
                        copied.custody_fees.pending[0][1],
                        account.custody_fees.pending[0][1],
                    )
                copied.detach()
                assert not copied.custody_fees.liability.requires_grad
            if day == 5:
                assert float(row["custody_payment"]) > 0
        return account.nav

    weight = tensor(0.6).requires_grad_()
    run(weight, True).backward()
    fd = (run(tensor(0.600001)) - run(tensor(0.599999))) / 0.000002
    assert float(weight.grad) == pytest.approx(float(fd), abs=1e-5)


def test_corporate_custody_is_not_silently_treated_as_ordinary():
    account = PortfolioAccount.empty([100.0, 100.0], config=config())
    account.shares[0] = 10
    with pytest.raises(ValueError, match="separate admission"):
        account.custody_fees.require_ordinary(0, account.shares[0], account.loans)


def test_immediate_split_pending_stock_and_missing_month_end_price():
    q = np.ones((6, 2))
    q[1, 0] = 2
    actions = AlignedActionTerms(
        q,
        np.zeros_like(q),
        np.ones_like(q, bool),
        q != 1,
        np.broadcast_to([0, 1], q.shape),
    )
    close = np.full((6, 2), 100.0)
    close[1:, 0] = 50
    close[2, 0] = np.nan
    rows = compare(
        close, [[0.6, -0.4]] * 2 + [[0, 0]] * 4, config=config(), actions=actions
    )
    assert float(rows[2]["physical_custody"][0]) == pytest.approx(1200.0)
    assert float(rows[2]["custody_base"]) == pytest.approx(60000.0)
    assert float(rows[2]["custody_fee"]) == pytest.approx(2.5)


def test_both_accounts_hedge_custody_includes_pending_sale():
    cfg = config(
        beta_hedge=True, hedge_cost_bps_per_side=0, planned_absolute_beta_cap=0.6
    )
    weight = [0.4, 0.4, 0, 0, 0, 0]
    result = replay(
        np.full((6, 1), 100.0),
        lambda s: PortfolioTarget(np.zeros(1), weight[s.day]),
        config=cfg,
        hedge_close=np.full(6, 100.0),
        hedge_beta=np.ones((6, 1)),
    )
    account = PortfolioAccount.empty([100.0, 100.0], config=cfg)
    for day in range(6):
        row = account.step(
            tensor([0.0, weight[day]]),
            day=day,
            close=[100.0, 100.0],
            cdi=0.0,
            session_date=result.dates[day],
            annual_borrow=[0.0, 0.0],
            loan_reference=[100.0, 100.0],
            terminal=day == 5,
        )
        assert float(row["nav"]) == pytest.approx(result.nav[day], abs=1e-9)
        assert row["physical_custody"].numpy() == pytest.approx(
            result.physical_custody[day], abs=1e-9
        )
    assert result.physical_custody[2, -1] == pytest.approx(400.0)
    assert result.custody_fee[2] == pytest.approx(40000 * 0.0005 / 12)
