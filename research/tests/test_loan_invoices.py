from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP

import numpy as np
import pytest
import torch

from brazil_rv.execution.loan_contracts import LoanContracts, LoanSession
from brazil_rv.execution.portfolio_account import PortfolioAccount, tensor
from brazil_rv.v2.portfolio_objective import clone_account
from test_portfolio_account import compare
from test_v2_stateful_ledger import _config


@pytest.mark.parametrize(
    "convention",
    ["contract_nearest", "contract_down", "contract_up", "security_day_nearest"],
)
def test_two_original_contracts_round_only_at_payment_and_group_separately(convention):
    book = LoanContracts(1, fee_multiplier=0, invoice_convention=convention)
    book.open([17], [13], [0.071], 0, "2024-01-02")
    book.accrue(1, "2024-01-03")
    book.open([11], [19], [0.093], 1, "2024-01-03")
    book.accrue(2, "2024-01-04")
    before = book.liability.item()
    assert book.pay(2)[0].item() == 0
    assert book.liability.item() == before
    book.request_return([28], 3)
    book.accrue(3, "2024-01-05")
    raw = [
        Decimal(221) * (Decimal("1.071") ** (Decimal(3) / 252) - 1),
        Decimal(209) * (Decimal("1.093") ** (Decimal(2) / 252) - 1),
    ]
    rounding = (
        ROUND_CEILING
        if convention.endswith("up")
        else ROUND_FLOOR
        if convention.endswith("down")
        else ROUND_HALF_UP
    )
    groups = [sum(raw)] if convention.startswith("security") else raw
    expected = sum(x.quantize(Decimal(".01"), rounding=rounding) for x in groups)
    liability = book.liability.item()
    rent, fee = book.pay(3)
    assert rent.item() == pytest.approx(float(expected), abs=1e-14)
    assert fee.item() == book.liability.item() == 0
    assert book.payment_adjustment.sum().item() == pytest.approx(
        float(expected) - liability, abs=1e-14
    )


@pytest.mark.parametrize("principal", [1000, 200000, 500000])
def test_partial_minimum_credit_preserves_lifetime_maximum_not_an_extra_fee(principal):
    books = [
        LoanContracts(1, minimum_allocation=m, record_payments=True)
        for m in ("final", "pro_rata")
    ]
    cumulative = [0.0, 0.0]
    for book in books:
        book.open([100], [principal / 100], [0], 0, "2019-01-02")
    for day in range(10):
        for k, book in enumerate(books):
            if day in (2, 5, 9):
                book.request_return([25 if day < 9 else 50], day)
            book.accrue(day, "2019-01-02")
            before = book.liability.item()
            _, fee = book.pay(day)
            cumulative[k] += fee.item()
            assert before == pytest.approx(
                fee.item() + book.liability.item(), abs=1e-10
            )
        assert cumulative[0] + books[0].liability.item() == pytest.approx(
            cumulative[1] + books[1].liability.item(), abs=1e-9
        )
        if day == 2:
            assert (
                cumulative[1] > cumulative[0]
                if principal <= 200000
                else cumulative[1] == pytest.approx(cumulative[0])
            )
            if principal == 1000:
                assert books[1].minimum_credit.item() > 0
    # 25% returns at ages2 and5; 50% at9. B3 D0 includes both endpoints.
    raw = Decimal(principal) * sum(
        w * (Decimal("1.0025") ** (Decimal(n) / 252) - 1)
        for w, n in [(Decimal(".25"), 3), (Decimal(".25"), 6), (Decimal(".5"), 10)]
    )
    assert cumulative[0] == pytest.approx(float(max(Decimal(10), raw)), abs=1e-8)
    assert cumulative[1] == pytest.approx(cumulative[0], abs=1e-9)
    assert all(
        book.liability.item() == 0 and len(book.minimum_credit) == 0 for book in books
    )


@pytest.mark.parametrize(
    "convention,minimum",
    [
        ("none", "pro_rata"),
        ("contract_nearest", "final"),
        ("security_day_nearest", "final"),
    ],
)
def test_actual_accounts_keep_invoice_expenses_and_pending_terminal_liabilities(
    convention, minimum
):
    targets = np.array([[-0.4], [-0.5], [-0.25], [-0.25], [0], [0], [0], [0]])
    config = _config(
        initial_capital_brl=1000,
        annual_borrow_rate=0.071,
        borrow_fee_multiplier=1,
        loan_invoice_convention=convention,
        loan_minimum_allocation=minimum,
    )
    compare(np.full((8, 1), 13.0), targets, cdi=np.full(8, 0.001), config=config)


@pytest.mark.parametrize("convention", ["none", "contract_nearest"])
def test_minimum_credit_copy_and_actual_account_gradient_away_from_cent_boundaries(
    convention,
):
    def run(weight, probe=False, detached=False):
        account = PortfolioAccount.empty(
            [13, 100],
            config=_config(
                initial_capital_brl=10000,
                annual_borrow_rate=0,
                borrow_fee_multiplier=1,
                loan_minimum_allocation="pro_rata",
                loan_invoice_convention=convention,
            ),
        )
        account.loans.record_payments = True
        for day in range(9):
            value = -weight if day < 2 else -weight / 2 if day < 5 else weight * 0
            account.step(
                torch.stack((value, weight * 0)),
                day=day,
                close=[13, 100],
                cdi=0.001,
                session_date="2019-01-02",
                annual_borrow=[0, 0],
                loan_reference=[13, 100],
            )
            if day == 5:
                assert account.loans.minimum_credit.sum().item() > 0
                if probe:
                    restart = clone_account(account)
                    restart.loans.minimum_credit[0] += 7
                    assert float(
                        restart.loans.minimum_credit.sum()
                        - account.loans.minimum_credit.sum()
                    ) == pytest.approx(7)
                    restart.loans.last_payment["probe"] = 1
                    assert "probe" not in account.loans.last_payment
                if detached:
                    account.detach()
        return account.nav

    weight = tensor(0.4).requires_grad_()
    nav = run(weight, probe=True)
    nav.backward()
    finite = (run(tensor(0.4000001)) - run(tensor(0.3999999))) / 0.0000002
    assert weight.grad.item() == pytest.approx(float(finite), abs=5e-5)
    assert run(tensor(0.4), detached=True).item() == pytest.approx(
        nav.item(), abs=1e-10
    )


@pytest.mark.parametrize("continuation", ["delivery", "renewal"])
def test_partial_minimum_credit_survives_conversion_and_pays_old_root_once(
    continuation,
):
    book = LoanContracts(3, minimum_allocation="pro_rata")
    book.open([100, 0, 0], [10, 20, 30], [0, 0, 0], 0, "2019-01-02")
    book.accrue(0, "2019-01-02")
    book.request_return([25, 0, 0], 1)
    book.accrue(1, "2019-01-03")
    _, paid = book.pay(1)
    first = paid.sum().item()
    credit = book.minimum_credit.sum().item()
    assert credit > 0
    if continuation == "delivery":
        book.deliver(0, 1, 1, 0.2, final=False)
        book.deliver(0, 2, 4, 1, final=True)
        assert book.minimum_credit.sum().item() == credit
        book.request_return([0, 75, 300], 2)
    else:
        book.renew(
            LoanSession(2, "2019-01-04", np.array([11, 20, 30]), np.zeros(3), 4), 6
        )
        assert book.minimum_credit[0].item() == credit
        assert book.minimum_credit[-1].item() == 0
    book.accrue(2, "2019-01-04")
    _, paid = book.pay(2)
    assert first + paid.sum().item() == pytest.approx(10, abs=1e-12)
    assert book.minimum_credit.sum().item() == 0
    assert book.liability.item() == pytest.approx(
        10 if continuation == "renewal" else 0
    )
