from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest
import torch

from brazil_rv.execution.action_settlement import ActionSettlement
from brazil_rv.execution.loan_contracts import LoanContracts
from brazil_rv.execution.portfolio_account import PortfolioAccount, tensor
from brazil_rv.execution.share_custody import ShareCustody
from brazil_rv.execution.stateful_ledger import PortfolioTarget
from brazil_rv.v2.corporate_actions import AlignedActionTerms
from brazil_rv.v2.portfolio_objective import clone_account
from test_portfolio_account import compare
from test_portfolio_ledger import replay
from test_v2_stateful_ledger import _config


def terms(close, *, q_day=None, cash_day=None, cash=0.0):
    q, d = np.ones_like(close), np.zeros_like(close)
    if q_day is not None:
        q[q_day, 0] = 2
    if cash_day is not None:
        d[cash_day, 0] = cash
    return AlignedActionTerms(
        q, d, np.ones_like(close, bool), (q != 1) | (d != 0), np.zeros_like(close, int)
    )


@pytest.mark.parametrize("sign", [-1, 1])
def test_only_incremental_bonus_is_reserved_until_credit(sign):
    close = np.array([[100], [50], [51], [52], [53]], float)
    event = ActionSettlement(0, 1, 0, "sourced credit, conservative loan return", 3)
    actions = terms(close, q_day=1)
    targets = [[sign * 0.4], [0], [0], [0], [0]]
    config = _config(annual_borrow_rate=0.04)
    result = replay(
        close,
        lambda state: PortfolioTarget(np.array(targets[state.day])),
        actions=actions,
        action_settlements=(event,),
        config=config,
    )
    # Pre-effect .004 shares produces exactly .004 additional shares, signed.
    np.testing.assert_allclose(
        result.signed_shares[:, 0],
        [sign * 0.004, sign * 0.004, sign * 0.004, 0, 0],
        atol=1e-15,
    )
    assert [f.fill_session for f in result.fills] == [0, 1, 3]
    assert result.undelivered_share_notional[:3] == pytest.approx([0, 0.2, 0.204])
    compare(close, targets, actions=actions, action_settlements=(event,), config=config)
    # Changing credit does not change any prefix before its first affected date.
    later = replay(
        close,
        lambda state: PortfolioTarget(np.array(targets[state.day])),
        actions=actions,
        action_settlements=(replace(event, bonus_delivery_session=4),),
        config=config,
    )
    np.testing.assert_allclose(result.nav[:3], later.nav[:3], rtol=0, atol=1e-15)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("short_fraction", [1.0, 0.85])
def test_jcp_cash_claim_is_net_for_long_and_separate_short_compensation(
    sign, short_fraction
):
    gross = Decimal("0.12784527353")
    close = np.full((6, 1), 100.0)
    close[1:] -= float(gross)
    actions = terms(close, cash_day=1, cash=float(gross))
    payments = np.full(close.shape, -1)
    payments[1, 0] = 4
    event = ActionSettlement(
        0,
        1,
        0,
        "issuer withholding; lender compensation bound",
        withholding_rate=0.15,
        short_cash_fraction=short_fraction,
    )
    targets = [[0.4 * sign], [0], [0], [0], [0], [0]]
    result = replay(
        close,
        lambda state: PortfolioTarget(np.array(targets[state.day])),
        actions=actions,
        payment_session=payments,
        action_settlements=(event,),
    )
    expected = float(
        Decimal(".004")
        * gross
        * (Decimal(".85") if sign > 0 else Decimal(str(short_fraction)))
    )
    assert result.receivables[1:4] - result.payables[1:4] == pytest.approx(
        np.full(3, expected * sign)
    )
    assert result.receivables[4] == result.payables[4] == 0
    assert result.withholding_accrual.sum() == pytest.approx(
        float(Decimal(".004") * gross * Decimal(".15")) if sign > 0 else 0
    )
    assert result.lender_compensation.sum() == pytest.approx(
        expected if sign < 0 else 0
    )
    # All trading/spot settlement ended by day3. Only the known payment moves cash.
    assert result.free_cash[4] - result.free_cash[3] == pytest.approx(expected * sign)
    assert result.nav[4] == pytest.approx(result.nav[3])
    compare(
        close,
        targets,
        actions=actions,
        payments=payments,
        action_settlements=(event,),
        config=_config(),
    )


def test_bonus_preserves_original_loan_minimum_and_delays_pending_returns():
    loan = LoanContracts(1)
    loan.open([100], [10], [0], 0, "2019-09-16")
    loan.accrue(0, "2019-09-16")
    loan.request_return([25], 2, request_day=1)
    loan.split(0, 2, bonus_delivery=4)
    assert loan.outstanding_principal.item() == 1000
    assert loan.active_quantity.item() == 150
    assert loan.quantity.sum().item() == 200
    loan.request_return([150], 3, request_day=2)
    assert set(loan.return_day) == {4}
    loan.accrue(3, "2019-09-19")
    assert sum(x.sum().item() for x in loan.pay(3)) == 0
    loan.accrue(4, "2019-09-20")
    assert sum(x.sum().item() for x in loan.pay(4)) == pytest.approx(10)
    assert loan.outstanding_principal.item() == 0


def test_bonus_purchase_custody_preserves_both_dates():
    custody = ShareCustody(1)
    custody.add(2, [100])
    custody.split(0, 2, bonus_delivery=4)
    assert [(day, float(q[0])) for day, q in custody.receipts] == [(2, 100), (4, 100)]
    custody.settle(2)
    assert [(day, float(q[0])) for day, q in custody.consume([200], [200], 2)] == [
        (2, 100),
        (4, 100),
    ]


def test_bonus_gradients_and_training_copies_are_independent():
    account = PortfolioAccount.empty([100, 100], config=_config())
    size = tensor(0.4).requires_grad_()
    event = ActionSettlement(0, 1, 0, "fixture", 3)
    for day in range(2):
        account.step(
            torch.stack([size if day == 0 else size * 0, size * 0]),
            day=day,
            close=[100 if day == 0 else 50, 100],
            cdi=0,
            session_date="2019-09-18",
            annual_borrow=[0, 0],
            loan_reference=[100, 100],
            action_q=[1 if day == 0 else 2, 1],
            action_settlements=(event,),
        )
    assert float(account.bonus_shares[0]) == pytest.approx(0.004)
    account.nav.backward()
    assert torch.isfinite(size.grad)
    twin = clone_account(account)
    account.detach()
    twin.prepare_day(3)
    assert float(twin.bonus_shares[0]) == 0
    assert float(account.bonus_shares[0]) == pytest.approx(0.004)
    twin.loans.bonus_return_floors[0] = 99
    assert account.loans.bonus_return_floors[0] == 3


@pytest.mark.parametrize("remaining", [0.0, -0.2])
def test_bonus_defers_covered_loan_proceeds_without_delaying_spot_cash(remaining):
    close = np.array([[100], [100], [50], [50], [50], [50]], float)
    actions = terms(close, q_day=2)
    event = ActionSettlement(0, 2, 0, "credit and whole-return bound", 4)
    targets = [[-0.4], [remaining], [0], [0], [0], [0]]
    result = replay(
        close,
        lambda state: PortfolioTarget(np.array(targets[state.day])),
        actions=actions,
        action_settlements=(event,),
    )
    compare(
        close, targets, actions=actions, action_settlements=(event,), config=_config()
    )
    # The day1 cover's purchase settles day3, but its proceeds stay restricted
    # until day4. The original .4 short-sale receipt still settles day2.
    assert result.restricted_cash[3] == pytest.approx(0.4)
    assert result.nav == pytest.approx(np.ones(6))
    assert result.free_cash[3] < result.free_cash[4]
    assert result.loan_outstanding_principal[3] > 0


def test_scalar_loader_preserves_gross_labels_and_rebases_known_payment():
    from test_corporate_replay import fixture
    from brazil_rv.v2.corporate_replay import apply_corporate_replay
    from brazil_rv.execution.action_settlement import slice_action_settlements

    data, corporate, calendar = fixture()
    corporate["cash_cancellations"] = []
    corporate["loan_cash_settlements"] = []
    corporate["scalar_actions"] = [
        dict(
            isin=data.inputs.security_ids[0],
            effective_date=str(calendar[2]),
            available_date=str(calendar[1]),
            shares_per_prior_share=1.0,
            gross_cash_per_prior_share=0.12784527353,
            payment_date=str(calendar[7]),
            withholding_rate=0.15,
            short_cash_fraction=1,
        )
    ]
    result = apply_corporate_replay(data.inputs, corporate, calendar, "a" * 64)
    assert result.action_cash_per_prior_share[2, 0] == 0.12784527353
    assert result.action_payment_session[2, 0] == 7
    assert result.scores is data.inputs.scores
    assert result.active is data.inputs.active
    sliced = slice_action_settlements(result.action_settlements, 1, 5)
    assert sliced[0].effective_session == 1
    assert sliced[0].available_session == 0
    corporate["scalar_actions"][0]["available_date"] = str(calendar[3])
    with pytest.raises(ValueError, match="knowledge"):
        apply_corporate_replay(data.inputs, corporate, calendar, "b" * 64)


def test_called_bonus_can_request_zero_while_undelivered_obligation_survives():
    from brazil_rv.execution.loan_contracts import LoanRecall

    close = np.array([[100], [50], [50], [50], [50], [50]], float)
    event = ActionSettlement(0, 1, 0, "bonus loan return hypothesis", 4)
    config = _config(loan_recalls=(LoanRecall(0, 2, 3, "ordinary recall bound"),))
    targets = [[-0.4], [0], [0], [0], [0], [0]]
    compare(
        close,
        targets,
        actions=terms(close, q_day=1),
        action_settlements=(event,),
        config=config,
    )


def test_pending_bonus_proceeds_have_independent_sam_copies():
    account = PortfolioAccount.empty([100, 100], config=_config())
    event = ActionSettlement(0, 2, 0, "fixture", 4)
    for day in range(3):
        account.step(
            tensor([-0.4 if day == 0 else 0, 0]),
            day=day,
            close=[100 if day < 2 else 50, 100],
            cdi=0,
            session_date="2019-09-18",
            annual_borrow=[0, 0],
            loan_reference=[100, 100],
            action_q=[2 if day == 2 else 1, 1],
            action_settlements=(event,),
        )
    assert len(account.bonus_proceeds) == 1
    twin = clone_account(account)
    twin.bonus_proceeds[0][1][0] = 0
    assert float(account.bonus_proceeds[0][1][0]) == pytest.approx(0.4)
    account.detach()
    account.prepare_day(4)
    assert not account.bonus_proceeds
