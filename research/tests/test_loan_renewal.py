import numpy as np
import pytest
import torch

from brazil_rv.execution.loan_contracts import LoanContracts, LoanSession
from brazil_rv.execution.portfolio_policy import PortfolioTarget
from test_portfolio_account import compare
from test_portfolio_ledger import replay
from test_v2_stateful_ledger import _config


def session(day, reference=30, rate=0.2, date="2024-01-15"):
    return LoanSession(day, date, np.array([reference]), np.array([rate]), day + 2)


def test_renewal_pays_old_contract_reprices_only_remaining_quantity_and_keeps_entry():
    book = LoanContracts(1, fee_multiplier=0)
    book.open([100], [20], [0.1], 0, "2024-01-02")
    book.request_return([40], 6)
    for day in range(1, 5):
        book.accrue(day, "2024-01-03")
        book.renew(session(day), 8)
    assert book.active_quantity.item() == 60
    before = book.liability.item()
    paid, fee = book.pay(4)
    assert fee.item() == 0
    assert paid.item() == pytest.approx(1200 * (1.1 ** (4 / 252) - 1), abs=1e-12)
    assert book.liability.item() + paid.item() == pytest.approx(before, abs=1e-12)
    assert book.principal.tolist() == [800, 1800]
    assert book.opened.tolist() == [0, 4]
    assert book.root_opened.tolist() == [0, 0]
    assert len(book.renewals) == 1
    assert book.renewals[0].previous_principal == 1200
    assert book.renewals[0].new_principal == 1800
    for day in (5, 6):
        book.accrue(day, "2024-01-15")
    paid, _ = book.pay(6)
    assert paid.item() == pytest.approx(800 * (1.1 ** (6 / 252) - 1), abs=1e-12)
    assert book.liability.item() == pytest.approx(
        1800 * (1.2 ** (2 / 252) - 1), abs=1e-12
    )


def test_renewal_minimum_is_new_contract_with_no_double_nav_expense():
    book = LoanContracts(1)
    book.open([10], [10], [0], 0, "2019-01-02")
    expense = 0
    paid = 0
    for day in range(1, 7):
        rent, fee = book.accrue(day, "2019-01-03")
        expense += (rent + fee).item()
        book.renew(session(day, 10, 0, "2019-01-03"), 8)
        if day == 6:
            book.request_return([10], 6)
        rent, fee = book.pay(day)
        paid += (rent + fee).item()
        assert expense == pytest.approx(paid + book.liability.item(), abs=1e-12)
    assert expense == pytest.approx(20)
    assert paid == pytest.approx(20)


def test_renewal_basket_legs_preserve_original_entry_and_reset_each_security():
    book = LoanContracts(2, fee_multiplier=0)
    book.open([100, 0], [20, 40], [0.1, 0.2], 0, "2024-01-02")
    book.deliver(0, 1, 2, allocation=1, final=True)
    for day in range(1, 5):
        book.accrue(day, "2024-01-03")
    book.renew(LoanSession(4, "2024-01-15", [20, 40], [0.1, 0.2], 6), 8)
    paid, _ = book.pay(4)
    assert paid.sum().item() == pytest.approx(2000 * (1.1 ** (4 / 252) - 1))
    assert book.principal.tolist() == [8000]
    assert book.root_name.tolist() == [0]
    assert book.renewals[0].security_index == 1
    charges = []
    book.accrue(5, "2024-01-16", charges=charges)
    assert charges[0].security_index == 0
    assert charges[0].opening_session == 0


def test_renewal_gradient_and_sam_state_independence():
    quantity = torch.tensor([100.0], dtype=torch.float64, requires_grad=True)
    book = LoanContracts(1, fee_multiplier=0)
    book.open(quantity, [20], [0.1], 0, "2024-01-02")
    for day in range(1, 5):
        book.accrue(day, "2024-01-03")
    book.renew(session(4), 8)
    paid, _ = book.pay(4)
    fork = book.detached_copy()
    for day in range(5, 9):
        book.accrue(day, "2024-01-16")
    total = paid.sum() + book.liability
    total.backward()
    assert quantity.grad.item() == pytest.approx(
        20 * (1.1 ** (4 / 252) - 1) + 30 * (1.2 ** (4 / 252) - 1), abs=1e-12
    )
    book.renew(session(8), 8)
    assert len(book.renewals) == 2
    assert len(fork.renewals) == 1
    assert fork.opened.tolist() == [4]


def test_missing_renewal_reference_stops_without_mutating_contract():
    book = LoanContracts(1, fee_multiplier=0)
    book.open([100], [20], [0.1], 0, "2024-01-02")
    before = book.detached_copy()
    with pytest.raises(ValueError, match="causal published"):
        book.renew(session(4, np.nan), 8)
    assert book.return_day.tolist() == before.return_day.tolist()
    assert book.principal.tolist() == before.principal.tolist()
    with pytest.raises(ValueError, match="missed"):
        book.renew(session(5), 8)


def test_both_accounts_agree_across_multiple_renewals_and_settled_funding():
    close = np.full((15, 2), 100.0)
    targets = np.tile([0.25, -0.25], (15, 1))
    targets[11:] = 0
    config = _config(loan_term_sessions=8, annual_borrow_rate=0.2)
    compare(close, targets, config=config, cdi=np.full(15, 0.0003))
    book = replay(close, lambda s: PortfolioTarget(targets[s.day]), config=config)
    assert len(book.loan_renewals) >= 2
    assert book.loan_liability[-1] == pytest.approx(0, abs=1e-12)
    assert book.loan_payment.sum() == pytest.approx(
        np.sum(book.borrow_bps * book.start_nav / 1e4), abs=1e-12
    )
