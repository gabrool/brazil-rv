import numpy as np
import pytest
import torch

from brazil_rv.execution.loan_contracts import LoanContracts


def test_fixed_reference_rate_and_compound_liability_survive_partial_delayed_return():
    book = LoanContracts(2, fee_multiplier=0)
    book.open([100, 0], [20, np.nan], [0.1, np.nan], 0, "2024-01-02")
    expenses = 0
    for day in range(1, 6):
        rent, fees = book.accrue(day, "2024-01-03")
        expenses += rent.sum().item()
        assert fees.sum().item() == 0
        if day == 2:
            book.request_return([40, 0], settlement_day=4)
        paid_rent, _ = book.pay(day)
        expected = 800 * ((1.1 ** (4 / 252)) - 1) if day == 4 else 0
        assert paid_rent.sum().item() == pytest.approx(expected, abs=1e-12)
        if day < 4:
            assert book.outstanding_principal.tolist() == [2000, 0]
        else:
            assert book.outstanding_principal.tolist() == [1200, 0]
    assert book.liability.item() == pytest.approx(
        1200 * (1.1 ** (5 / 252) - 1), abs=1e-12
    )
    assert expenses == pytest.approx(
        800 * (1.1 ** (4 / 252) - 1) + 1200 * (1.1 ** (5 / 252) - 1), abs=1e-12
    )


def test_new_borrow_is_a_new_cohort_and_does_not_reprice_old_principal():
    book = LoanContracts(1, fee_multiplier=0)
    book.open([100], [10], [0.01], 0, "2024-01-02")
    book.accrue(1, "2024-01-03")
    book.open([50], [30], [0.8], 1, "2024-01-03")
    book.accrue(2, "2024-01-04")
    assert book.liability.item() == pytest.approx(
        1000 * (1.01 ** (2 / 252) - 1) + 1500 * (1.8 ** (1 / 252) - 1), abs=1e-12
    )
    book.split(0, 2)
    assert book.active_quantity.item() == 300
    assert book.outstanding_principal.item() == 2500
    book.request_return([150], 3)
    book.accrue(3, "2024-01-05")
    before = book.liability.item()
    rent, fee = book.pay(3)
    assert rent.item() == pytest.approx(before / 2)
    assert fee.item() == 0
    assert book.liability.item() == pytest.approx(before / 2)


def test_historical_minimum_is_one_contract_obligation_not_a_partial_return_toll():
    book = LoanContracts(1)
    book.open([100], [10], [0], 0, "2019-01-02")
    _, expense = book.accrue(0, "2019-01-02")
    assert expense.item() == pytest.approx(10)
    book.accrue(1, "2019-01-03")
    book.request_return([25], 2)
    book.accrue(2, "2019-01-04")
    liability = book.liability.item()
    _, first = book.pay(2)
    assert first.item() > 0
    assert first.item() + book.liability.item() == pytest.approx(liability)
    book.request_return([75], 3)
    _, extra = book.accrue(3, "2019-01-07")
    assert extra.item() == pytest.approx(0, abs=1e-12)
    _, last = book.pay(3)
    assert first.item() + last.item() == pytest.approx(10)
    assert book.liability.item() == 0
    assert len(book.name) == len(book.root_name) == 0


def test_large_historical_contract_pays_actual_fee_when_above_minimum():
    book = LoanContracts(1)
    book.open([100000], [10], [0], 0, "2019-01-02")
    expense = 0
    for day in range(4):
        _, fee = book.accrue(day, "2019-01-03")
        expense += fee.item()
    book.request_return([100000], 3)
    _, paid = book.pay(3)
    expected = 1e6 * (1.0025 ** (4 / 252) - 1)
    assert paid.item() == pytest.approx(expected, abs=1e-9)
    assert expense == pytest.approx(expected, abs=1e-9)


def test_return_quantity_gradient_and_detach_preserve_liability():
    q = torch.tensor([100.0], dtype=torch.float64, requires_grad=True)
    fraction = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    book = LoanContracts(1, fee_multiplier=0)
    book.open(q, [20], [0.1], 0, "2024-01-02")
    book.accrue(1, "2024-01-03")
    book.request_return(q * fraction, 2)
    book.accrue(2, "2024-01-04")
    paid, _ = book.pay(2)
    book.accrue(3, "2024-01-05")
    loss = paid.sum() + book.liability
    loss.backward()
    per2, per3 = 20 * (1.1 ** (2 / 252) - 1), 20 * (1.1 ** (3 / 252) - 1)
    assert q.grad.item() == pytest.approx(0.4 * per2 + 0.6 * per3, abs=1e-12)
    assert fraction.grad.item() == pytest.approx(100 * (per2 - per3), abs=1e-12)
    before = book.liability.item()
    book.detach()
    assert book.liability.item() == before
    assert not book.liability.requires_grad


def test_missing_opening_reference_does_not_fabricate_a_loan_price():
    book = LoanContracts(1)
    with pytest.raises(ValueError, match="published-average"):
        book.open([1], [np.nan], [0.1], 0, "2024-01-02")


def test_split_delivery_keeps_origin_charges_and_opening_quality():
    book = LoanContracts(3, fee_multiplier=0)
    book.open(
        [100, 0, 0],
        [20, 0, 0],
        [0.1, 0, 0],
        0,
        "2024-01-02",
        imputed=[True, False, False],
    )
    charges = []
    book.accrue(1, "2024-01-03", charges=charges)
    book.deliver(0, 1, 0.4, 0.25, final=False)
    book.deliver(0, 2, 2, 1, final=True)
    assert book.active_quantity.tolist() == [0, 40, 200]
    assert book.outstanding_principal.tolist() == [0, 500, 1500]
    assert book.imputed.all()
    book.request_return([0, 40, 100], 2)
    rent, fees = book.accrue(2, "2024-01-04", charges=charges)
    paid, _ = book.pay(2)
    assert paid.sum().item() == pytest.approx(1250 * (1.1 ** (2 / 252) - 1))
    assert book.outstanding_principal.tolist() == [0, 0, 750]
    assert {(c.opening_session, c.security_index) for c in charges} == {(0, 0)}
    assert charges[-1].rent == pytest.approx(rent.sum().item())
    assert charges[-1].fee == fees.sum().item() == 0


def test_detached_sam_restart_is_independent_with_pending_returns():
    book = LoanContracts(1)
    quantity = torch.tensor([100.0], requires_grad=True, dtype=torch.float64)
    book.open(quantity, [20], [0.1], 0, "2024-01-02")
    book.accrue(1, "2024-01-03")
    book.request_return([40], 3)
    restart = book.detached_copy()
    value = book.liability.item()
    restart.accrue(2, "2024-01-04")
    restart.split(0, 2)
    assert book.liability.item() == value
    assert book.active_quantity.item() == 60
    assert restart.active_quantity.item() == 120
    assert not restart.liability.requires_grad
    assert book.liability.requires_grad


def test_spot_return_timing_transition_is_explicit():
    from brazil_rv.execution.loan_contracts import spot_settlement_session

    assert spot_settlement_session(20, "2019-05-24") == 23
    assert spot_settlement_session(21, "2019-05-27") == 23
