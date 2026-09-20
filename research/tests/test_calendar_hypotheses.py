"""Separate uncertain money, delivery and contractual-accrual clocks."""

import numpy as np
import pytest
import torch

from brazil_rv.execution.custody_fees import CustodyAssessment
from brazil_rv.execution.loan_contracts import LoanContracts, LoanSession
from brazil_rv.execution.loan_fees import loan_fee_rates
from brazil_rv.execution.portfolio_account import PortfolioAccount, tensor
from brazil_rv.execution.stateful_ledger import PortfolioTarget
from brazil_rv.v2.portfolio_objective import clone_account
from test_loan_accrual import amount
from test_portfolio_account import compare
from test_portfolio_ledger import replay
from test_v2_stateful_ledger import _config

DATES = np.array(
    [
        "2017-01-19",
        "2017-01-20",
        "2017-01-23",
        "2017-01-24",
        "2017-01-26",
        "2017-01-27",
        "2017-01-30",
        "2017-01-31",
    ],
    dtype="datetime64[D]",
)


@pytest.mark.parametrize("clock", ["cash", "delivery"])
def test_short_cover_cash_and_return_proceeds_have_separate_dates(clock):
    cfg = _config(**{f"spot_{clock}_lag_overrides": (("2017-01-20", 2),)})
    targets = [[-0.4]] + [[0]] * 7
    prices = np.full((8, 1), 100.0)
    compare(prices, targets, dates=DATES, config=cfg)
    result = replay(
        prices,
        lambda s: PortfolioTarget(np.array(targets[s.day])),
        dates=DATES,
        config=cfg,
    )
    assert result.nav == pytest.approx(np.ones(8))
    # Jan19 sale settles Jan24. Jan20 cover's money and loan return differ.
    assert result.free_cash[3] == pytest.approx(0.6 if clock == "cash" else 1.4)
    assert result.restricted_cash[3] == pytest.approx(0.4 if clock == "cash" else 0)
    assert result.unsettled_cash[3] == pytest.approx(0 if clock == "cash" else -0.4)
    assert result.free_cash[4] == pytest.approx(1)
    assert result.restricted_cash[4] == pytest.approx(0)


@pytest.mark.parametrize("clock", ["cash", "delivery"])
def test_owned_stock_physical_date_is_independent_of_purchase_cash(clock):
    cfg = _config(
        custody_assessments=(CustodyAssessment("2017-02-28", "2017-02-28"),),
        **{f"spot_{clock}_lag_overrides": (("2017-01-20", 2),)},
    )
    account = PortfolioAccount.empty([100, 100], config=cfg)
    rows = []
    for day, date in enumerate(DATES):
        rows.append(
            account.step(
                tensor([0 if day == 0 else 0.4, 0]),
                day=day,
                close=[100, 100],
                cdi=0,
                session_date=date,
                annual_borrow=[0, 0],
                loan_reference=[100, 100],
            )
        )
    assert rows[3]["physical_custody"][0].item() == pytest.approx(
        0.004 if clock == "delivery" else 0
    )
    assert rows[4]["physical_custody"][0].item() == pytest.approx(0.004)
    targets = [[0]] + [[0.4]] * 7
    compare(np.full((8, 1), 100.0), targets, dates=DATES, config=cfg)


def test_additional_interval_partial_returns_minimum_and_gradient():
    quantity = tensor(100.0).requires_grad_()
    loans = LoanContracts(1)
    loans.open(quantity.reshape(1), [20], [0.1], 0, "2017-01-19")
    paid_rent = tensor(0.0)
    paid_fee = tensor(0.0)
    for day in range(6):
        if day == 1:
            loans.request_return((quantity * 0.4).reshape(1), 3, request_day=1)
        if day == 4:
            loans.request_return((quantity * 0.6).reshape(1), 5, request_day=4)
        loans.accrue(day, DATES[day], extra_interval=day == 3)
        if day == 3:
            # The Jan24 return gets no extra interval; the surviving part does.
            assert sorted(loans.extra_accrual_days.tolist()) == [0, 1]
        rent, fee = loans.pay(day)
        paid_rent = paid_rent + rent.sum()
        paid_fee = paid_fee + fee.sum()
    expected = amount(800, 0.1, 3) + amount(1200, 0.1, 6)
    assert paid_rent.item() == pytest.approx(expected)
    assert paid_fee.item() == pytest.approx(10.0)  # once per original root
    paid_rent.backward()
    assert quantity.grad.item() == pytest.approx(expected / 100)
    assert loans.liability.item() == 0


def test_extra_interval_fee_compounding_transfer_and_copy():
    loans = LoanContracts(2)
    loans.open([100000, 0], [20, 20], [0.1, 0.1], 0, "2017-01-19")
    for day in range(4):
        loans.accrue(day, DATES[day], extra_interval=day == 3)
    copy = loans.detached_copy()
    copy.extra_accrual_days[0] = 9
    assert loans.extra_accrual_days.tolist() == [1]
    loans.deliver(0, 1, 2, 1, final=True)
    assert loans.extra_accrual_days.tolist() == [1]
    loans.request_return([0, 200000], 5, request_day=4)
    for day in (4, 5):
        loans.accrue(day, DATES[day])
    rent, fee = loans.pay(5)
    assert rent.sum().item() == pytest.approx(amount(2e6, 0.1, 6))
    assert fee.sum().item() == pytest.approx(
        amount(2e6, loan_fee_rates(0.1, "2017-01-19"), 7).sum()
    )


def test_extra_interval_does_not_advance_renewal_session():
    loans = LoanContracts(1)
    loans.open([100], [20], [0.1], 0, "2024-01-02")
    for day in range(4):
        loans.accrue(day, "2024-01-02", extra_interval=day == 2)
    before = loans.opened.copy()
    loans.renew(LoanSession(3, "2024-01-05", np.array([30]), np.array([0.2]), 5), 8)
    np.testing.assert_array_equal(loans.opened, before)
    loans.renew(LoanSession(4, "2024-01-08", np.array([30]), np.array([0.2]), 6), 8)
    assert 4 in loans.opened
    assert loans.extra_accrual_days[loans.opened == 4].tolist() == [0]


def test_both_accounts_extra_interval_and_prior_funding():
    cfg = _config(annual_borrow_rate=0.1, loan_extra_accrual_after=("2017-01-24",))
    targets = [[0.4, -0.4]] * 5 + [[0, 0]] * 3
    compare(
        np.full((8, 2), 100.0), targets, dates=DATES, cdi=np.full(8, 0.0003), config=cfg
    )


def test_sam_and_tbptt_keep_independent_accrual_offsets_and_split_queues():
    cfg = _config(
        spot_cash_lag_overrides=(("2017-01-20", 2),),
        loan_extra_accrual_after=("2017-01-24",),
    )
    account = PortfolioAccount.empty([100, 100], config=cfg)
    weight = tensor(-0.4).requires_grad_()
    for day in range(4):
        account.step(
            torch.stack((weight, weight * 0)),
            day=day,
            close=[100, 100],
            cdi=0,
            session_date=DATES[day],
            annual_borrow=[0.1, 0],
            loan_reference=[100, 100],
        )
    copy = clone_account(account)
    copy.loans.extra_accrual_days[:] = 8
    assert np.max(account.loans.extra_accrual_days) == 1
    before = account.nav.item()
    account.detach()
    assert account.nav.item() == before
    assert not account.loans.rent_due.requires_grad
