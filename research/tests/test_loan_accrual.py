import numpy as np
import pytest
import torch

from brazil_rv.execution.loan_contracts import (
    LoanCashSettlement,
    LoanContracts,
    LoanSession,
)
from brazil_rv.execution.loan_fees import loan_fee_rates
from test_portfolio_account import compare
from test_v2_stateful_ledger import _config


def amount(principal, annual, days):
    return principal * np.expm1(np.log1p(annual) * days / 252)


@pytest.mark.parametrize("lag", [0, 1])
def test_dated_endpoints_partial_returns_and_intermediate_liabilities(lag):
    book = LoanContracts(1, electronic_settlement_days=lag)
    book.open([100], [20], [0.1], 0, "2024-01-02")
    fee_rates = loan_fee_rates(0.1, "2024-01-02")
    expense = paid = 0.0
    for day in range(4):
        if day == 1:
            book.request_return([40], 2, request_day=1)
            book.request_return([60], 3, request_day=1)
        rent, fee = book.accrue(day, "2024-01-02")
        expense += rent.item() + fee.item()
        if day == 0:
            assert rent.item() == pytest.approx(amount(2000, 0.1, 1 - lag))
            assert fee.item() == pytest.approx(amount(2000, fee_rates, 1 - lag).sum())
        rp, fp = book.pay(day)
        paid += rp.item() + fp.item()
        if day in (2, 3):
            principal = 800 if day == 2 else 1200
            assert rp.item() == pytest.approx(amount(principal, 0.1, day))
            assert fp.item() == pytest.approx(
                amount(principal, fee_rates, day + 1 - lag).sum()
            )
        assert expense == pytest.approx(paid + book.liability.item(), abs=1e-12)
    assert book.liability.item() == 0


def test_registered_and_pre_platform_contracts_are_d0_even_with_electronic_d1():
    for date, modality in [("2019-01-02", "normal"), ("2024-01-02", "otc")]:
        book = LoanContracts(1, modality=modality)
        book.open([100000], [20], [0.1], 0, date)
        assert book.value_lag.item() == 0
        fee_rates = loan_fee_rates(0.1, date, modality=modality)
        for day in range(4):
            if day == 1:
                book.request_return([100000], 3, request_day=1)
            book.accrue(day, date)
        rent, fee = book.pay(3)
        assert rent.item() == pytest.approx(amount(2e6, 0.1, 3))
        assert fee.item() == pytest.approx(amount(2e6, fee_rates, 4).sum())


def test_d0_same_registration_day_return_request_is_one_day_of_rates():
    book = LoanContracts(1, electronic_settlement_days=0)
    book.open([100], [20], [0.1], 0, "2024-01-02")
    book.request_return([100], 1, request_day=0)
    r0, f0 = book.accrue(0, "2024-01-02")
    r1, f1 = book.accrue(1, "2024-01-03")
    rent, fee = book.pay(1)
    assert rent.item() == pytest.approx(amount(2000, 0.1, 1))
    assert fee.item() == pytest.approx(
        amount(2000, loan_fee_rates(0.1, "2024-01-02"), 1).sum()
    )
    assert r0.item() == rent.item() and r1.item() == 0
    assert f0.item() == 0 and f1.item() == fee.item()


def test_d0_renewal_pays_old_contract_and_starts_new_date_without_double_rent():
    q = torch.tensor([100.0], dtype=torch.float64, requires_grad=True)
    book = LoanContracts(1, electronic_settlement_days=0)
    book.open(q, [20], [0.1], 0, "2024-01-02")
    paid = torch.tensor(0.0, dtype=torch.float64)
    for day in range(5):
        if day == 4:
            book.renew(
                LoanSession(day, "2024-01-08", np.array([30]), np.array([0.2]), 6), 8
            )
        book.accrue(day, "2024-01-08")
        rent, fee = book.pay(day)
        paid = paid + rent.sum() + fee.sum()
    old = (
        amount(2000, 0.1, 4) + amount(2000, loan_fee_rates(0.1, "2024-01-08"), 5).sum()
    )
    new = (
        amount(3000, 0.2, 1) + amount(3000, loan_fee_rates(0.2, "2024-01-08"), 1).sum()
    )
    assert paid.item() == pytest.approx(old)
    assert book.liability.item() == pytest.approx(new)
    (paid + book.liability).backward()
    assert q.grad.item() == pytest.approx((old + new) / 100)
    copy = book.detached_copy()
    copy.value_lag[0] = 1
    copy.return_requested[0] = 0
    assert book.value_lag.item() == 0 and book.return_requested.item() == -1


def test_d0_tariff_change_retains_opening_day_fee_in_its_own_dated_period():
    book = LoanContracts(1, electronic_settlement_days=0)
    book.open([100000], [20], [0.2], 0, "2022-11-11")
    for day, date in enumerate(["2022-11-11", "2022-11-14", "2022-11-16"]):
        if day == 1:
            book.request_return([100000], 2, request_day=1)
        book.accrue(day, date)
    rent, fee = book.pay(2)
    assert rent.item() == pytest.approx(amount(2e6, 0.2, 2))
    expected = (
        amount(2e6, [0.001, 0.009], 1).sum() + amount(2e6, [0.0007, 0.0063], 2).sum()
    )
    assert fee.item() == pytest.approx(expected)


def test_explicit_corporate_stopped_rent_date_overrides_physical_return_endpoint():
    book = LoanContracts(1, electronic_settlement_days=0)
    book.open([100], [20], [0.1], 0, "2024-01-02")
    event = LoanCashSettlement(
        0, 2, 1, 23.0, "source", rent_payment_session=4, payment_session=5
    )
    for day in range(5):
        if day == 2:
            book.cash_settle(event, day)
        book.accrue(day, "2024-01-02")
        rent, fee = book.pay(day)
        if day == 4:
            assert rent.item() == pytest.approx(amount(2000, 0.1, 3))
            assert fee.item() == pytest.approx(
                amount(2000, loan_fee_rates(0.1, "2024-01-02"), 3).sum()
            )
    assert book.cash_liability.item() == 2300


def test_d0_accounts_reconcile_fills_returns_renewals_and_funding():
    prices = np.full((15, 2), 100.0)
    targets = np.tile([0.25, -0.25], (15, 1))
    targets[11:] = 0
    compare(
        prices,
        targets,
        config=_config(
            electronic_loan_settlement_days=0,
            loan_term_sessions=8,
            annual_borrow_rate=0.2,
        ),
        cdi=np.full(15, 0.0003),
    )
