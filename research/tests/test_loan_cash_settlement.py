from dataclasses import replace

import numpy as np
import pytest
import torch

from brazil_rv.execution.loan_contracts import (
    LoanCashSettlement,
    LoanCashValue,
    LoanContracts,
    slice_loan_settlements,
)
from brazil_rv.execution.portfolio_account import PortfolioAccount, tensor
from brazil_rv.execution.portfolio_policy import (
    PreferenceModel,
    account_decision,
    ledger_arguments,
    policy_ledger_config,
)
from brazil_rv.execution.stateful_ledger import PortfolioTarget
from brazil_rv.v2.corporate_actions import AlignedActionTerms
from brazil_rv.v2.evaluate import _input_hashes, _ledger_inputs, _PAIRED_INPUT_KEYS
from brazil_rv.v2.portfolio_inputs import Calibration
from test_portfolio_account import compare
from test_portfolio_ledger import replay
from test_portfolio_policy import policy_fixture
from test_v2_stateful_ledger import _config


def event(price=110):
    return LoanCashSettlement(0, 2, 1, price, "dated B3 fixture")


def cash_action(days=5):
    q, cash = np.ones((days, 1)), np.zeros((days, 1))
    q[4, 0], cash[4, 0] = 0, 120
    actions = AlignedActionTerms(
        q, cash, np.ones_like(q, bool), q != 1, np.zeros_like(q, int)
    )
    payments = np.full((days, 1), -1)
    payments[4, 0] = 4
    return actions, payments


@pytest.mark.parametrize("side", [-1, 1])
def test_loan_cash_closeout_is_distinct_from_shareholder_cash_redemption(side):
    close = np.array([[100.0], [np.nan], [np.nan], [np.nan], [np.nan]])
    targets = np.array([[side * 0.4], [0], [0], [0], [0]])
    actions, payments = cash_action()
    config = _config(annual_borrow_rate=0.04)
    compare(
        close,
        targets,
        config=config,
        actions=actions,
        payments=payments,
        loan_cash_settlements=(event(),),
    )
    book = replay(
        close,
        lambda state: PortfolioTarget(targets[state.day]),
        config=config,
        actions=actions,
        payment_session=payments,
        loan_cash_settlements=(event(),),
    )
    if side < 0:
        rent = 0.4 * ((1.04 ** (2 / 252)) - 1)
        assert book.nav[-1] == pytest.approx(0.96 - rent)
        assert book.loan_payment[2] == pytest.approx(rent)
        assert book.loan_payment[3:].sum() == 0
        assert book.loan_cash_payments[0].cash_paid == pytest.approx(0.44)
        assert book.signed_shares[2, 0] == 0
    else:
        assert book.nav[-1] == pytest.approx(1.08)
        assert book.signed_shares[2, 0] == pytest.approx(0.004)
        assert not book.loan_cash_payments
    assert book.restricted_cash[2:].sum() == 0
    assert not [fill for fill in book.fills if fill.fill_session > 0]
    assert book.cost_bps.sum() == 0


def test_cash_closeout_supersedes_later_physical_return_but_preserves_covering_asset():
    close = np.array([[100.0], [100.0], [np.nan], [np.nan], [np.nan]])
    targets = np.array([[-0.4], [0], [0], [0], [0]])
    actions, payments = cash_action()
    config = _config()
    compare(
        close,
        targets,
        config=config,
        actions=actions,
        payments=payments,
        loan_cash_settlements=(event(),),
    )
    book = replay(
        close,
        lambda state: PortfolioTarget(targets[state.day]),
        config=config,
        actions=actions,
        payment_session=payments,
        loan_cash_settlements=(event(),),
    )
    assert book.signed_shares[:, 0] == pytest.approx([-0.004, 0, 0.004, 0.004, 0])
    assert book.nav == pytest.approx([1, 1, 0.96, 0.96, 1.04])
    assert book.free_cash[2] == pytest.approx(0.96)
    assert book.restricted_cash[2] == 0
    assert book.unsettled_cash[2] == pytest.approx(-0.4)
    assert book.free_cash[3] == pytest.approx(0.56)


def test_settlement_price_mutation_cannot_rewrite_prior_intentions():
    close = np.array([[100.0], [np.nan], [np.nan], [np.nan]])
    books = [
        replay(
            close,
            lambda state: PortfolioTarget(np.array([-0.4 if state.day == 0 else 0])),
            loan_cash_settlements=(event(price),),
        )
        for price in [110, 150]
    ]
    assert [x for x in books[0].intended_orders if x.decision_session <= 2] == [
        x for x in books[1].intended_orders if x.decision_session <= 2
    ]
    assert books[0].nav[2] != books[1].nav[2]


def test_cash_loan_settlement_gradient_matches_closed_form_and_finite_difference():
    def run(weight):
        account = PortfolioAccount.empty(
            [100, 100], config=_config(annual_borrow_rate=0.04)
        )
        for day in range(4):
            target = tensor([0, 0]) if day else torch.stack([-weight, weight * 0])
            account.step(
                target,
                day=day,
                close=[100 if day == 0 else np.nan, 100],
                cdi=0,
                session_date="2024-01-02",
                annual_borrow=[0.04, 0],
                loan_reference=[100, 100],
                loan_cash_settlements=(event(),),
            )
        return account.nav

    weight = tensor(0.4).requires_grad_()
    run(weight).backward()
    expected = -0.1 - (1.04 ** (2 / 252) - 1)
    assert weight.grad.item() == pytest.approx(expected, abs=1e-10)
    diff = (run(tensor(0.400001)) - run(tensor(0.399999))) / 0.000002
    assert weight.grad.item() == pytest.approx(diff.item(), abs=1e-9)


def test_source_clock_duplicate_events_and_flat_start_borrow_prohibition():
    with pytest.raises(ValueError, match="backdated"):
        replace(event(), available_session=3)
    close = np.ones((4, 1)) * 100
    with pytest.raises(ValueError, match="duplicate"):
        replay(
            close,
            lambda state: PortfolioTarget(np.zeros(1)),
            loan_cash_settlements=(event(), event()),
        )
    sliced = slice_loan_settlements((event(),), 3, 7)
    assert sliced[0].effective_session == -1
    states = []

    def policy(state):
        states.append(state)
        return PortfolioTarget(np.zeros(1))

    result = replay(close, policy, loan_cash_settlements=sliced)
    assert not states[0].shortable[0]
    assert result.nav == pytest.approx(np.ones(4))
    assert not result.loan_cash_payments


def test_policy_constraints_hashes_and_slicing_keep_static_coordinates_frozen():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    data = policy_fixture()
    static = data.static.copy()
    model = PreferenceModel(
        data,
        Calibration(np.zeros(3), np.ones(3), np.array([0.0008, 0, 0]), 0),
        np.arange(10),
    )
    account = data.initial_account(0, policy_ledger_config())
    original = account_decision(data, model, account, 0)
    source = int(original[:-1].argmin())
    assert original[source] < -0.001
    terms = (LoanCashSettlement(source, 0, 0, 110, "dated B3 fixture"),)
    old_hash = _input_hashes(data.inputs)
    data.inputs = replace(data.inputs, loan_cash_settlements=terms)
    changed = account_decision(data, model, account, 0)
    assert changed[source] >= -1e-9
    np.testing.assert_array_equal(data.static, static)
    new_hash = _input_hashes(data.inputs)
    assert old_hash["loan_cash_settlements"] != new_hash["loan_cash_settlements"]
    assert "loan_cash_settlements" in _PAIRED_INPUT_KEYS
    arguments = _ledger_inputs(data.inputs, data.inputs.scores[..., 0], data.valid)
    assert arguments["loan_cash_settlements"] == terms
    rebased = ledger_arguments(data, 2, 10)["loan_cash_settlements"]
    assert rebased[0].effective_session == -2


def deferred_event():
    return LoanCashSettlement(
        0,
        2,
        1,
        110,
        "lender election fixture",
        rent_payment_session=4,
        payment_session=6,
        valuations=(LoanCashValue(3, 110.1), LoanCashValue(5, 120)),
        unreturned_only=True,
        prohibit_new_borrow=False,
    )


def test_deferred_election_stops_rent_retains_proceeds_and_revalues_only_when_known():
    close = np.array([[100.0], *[[np.nan]] * 7])
    targets = np.array([[-0.4], *[[0]] * 7])
    config = _config(annual_borrow_rate=0.04)
    records = compare(
        close, targets, config=config, loan_cash_settlements=(deferred_event(),)
    )
    book = replay(
        close,
        lambda state: PortfolioTarget(targets[state.day]),
        config=config,
        loan_cash_settlements=(deferred_event(),),
    )
    rent = 0.4 * (1.04 ** (2 / 252) - 1)
    assert book.nav[2:5] == pytest.approx([0.96 - rent, 0.9596 - rent, 0.9596 - rent])
    assert book.nav[5:] == pytest.approx([0.92 - rent] * 3)
    assert book.loan_payment[4] == pytest.approx(rent)
    assert book.loan_payment.sum() == pytest.approx(rent)
    assert book.restricted_cash[2:6] == pytest.approx([0.4] * 4)
    assert book.restricted_cash[6:].sum() == 0
    assert book.signed_shares[2:, 0] == pytest.approx(np.zeros(6))
    assert book.loan_cash_payments[0].session == 6
    assert book.loan_cash_payments[0].cash_paid == pytest.approx(0.48)
    assert float(records[5]["loan_redemption_liability"]) == pytest.approx(0.48)
    altered = replace(
        deferred_event(), valuations=(LoanCashValue(3, 110.1), LoanCashValue(5, 150))
    )
    other = replay(
        close,
        lambda state: PortfolioTarget(targets[state.day]),
        config=config,
        loan_cash_settlements=(altered,),
    )
    np.testing.assert_array_equal(book.nav[:5], other.nav[:5])
    assert [x for x in book.intended_orders if x.decision_session <= 5] == [
        x for x in other.intended_orders if x.decision_session <= 5
    ]


def test_election_whole_contract_eligibility_and_independent_state_copy():
    loans = LoanContracts(1, fee_multiplier=0)
    loans.open([30], [100], [0.04], 0, "2024-01-02")
    loans.accrue(1, "2024-01-03")
    loans.request_return([15], 4)
    loans.open([20], [90], [0.05], 1, "2024-01-03")
    loans.accrue(2, "2024-01-04")
    loans.open([5], [95], [0.06], 2, "2024-01-04")
    assert float(loans.cash_settle(deferred_event(), 2)) == 20
    assert (
        float(loans.active_quantity[0]) == 20
    )  # 15 excluded-root + 5 newly registered
    other = loans.detached_copy()
    assert other.cash_claims is not loans.cash_claims
    assert other.cash_claims[0][1].data_ptr() != loans.cash_claims[0][1].data_ptr()
    other.settle_cash_claims(5)
    assert float(other.cash_liability) == 2400
    assert float(loans.cash_liability) == 2200
    rebased = slice_loan_settlements((deferred_event(),), 1, 8)[0]
    assert (rebased.effective_session, rebased.rent_day, rebased.cash_day) == (1, 3, 5)
    assert [v.available_session for v in rebased.valuations] == [2, 4]


def test_deferred_cash_gradient_and_payment_date_causality():
    def run(weight, payment=6):
        account = PortfolioAccount.empty(
            [100, 100], config=_config(annual_borrow_rate=0.04)
        )
        history = []
        for day in range(8):
            account.step(
                torch.stack([-weight, weight * 0]) if day == 0 else tensor([0, 0]),
                day=day,
                close=[100 if day == 0 else np.nan, 100],
                cdi=0.001,
                session_date="2024-01-02",
                annual_borrow=[0.04, 0],
                loan_reference=[100, 100],
                loan_cash_settlements=(
                    replace(deferred_event(), payment_session=payment),
                ),
            )
            history.append(account.nav)
        return torch.stack(history)

    weight = tensor(0.4).requires_grad_()
    result = run(weight)
    result[-1].backward()
    diff = (run(tensor(0.400001))[-1] - run(tensor(0.399999))[-1]) / 0.000002
    assert float(weight.grad) == pytest.approx(float(diff), abs=1e-8)
    torch.testing.assert_close(
        result[:6].detach(), run(tensor(0.4), payment=7)[:6], rtol=0, atol=0
    )
