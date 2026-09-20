from dataclasses import replace

import numpy as np
import pytest
import torch

from brazil_rv.execution.custody_fees import CustodyAssessment
from brazil_rv.execution.portfolio_account import PortfolioAccount, tensor
from brazil_rv.execution.share_distributions import (
    FractionAuction,
    ShareDelivery,
    ShareDistribution,
)
from brazil_rv.execution.stateful_ledger import PortfolioTarget
from brazil_rv.execution.action_settlement import ActionSettlement
from brazil_rv.execution.loan_contracts import LoanCashSettlement
from brazil_rv.v2.corporate_actions import AlignedActionTerms
from brazil_rv.v2.portfolio_objective import clone_account
from test_portfolio_account import compare
from test_portfolio_ledger import replay
from test_v2_stateful_ledger import _config, _run


def config(assessment="2024-01-04", **kw):
    return _config(
        initial_capital_brl=100000,
        custody_assessments=(CustodyAssessment(assessment, "2024-02-01"),),
        **kw,
    )


def checked(close, targets, cfg, **kw):
    records = compare(close, targets, config=cfg, **kw)
    book = replay(
        close, lambda s: PortfolioTarget(np.asarray(targets[s.day])), config=cfg, **kw
    )
    for day, row in enumerate(records):
        for key in (
            "custody_base",
            "custody_physical_base",
            "custody_claim_base",
            "custody_fee",
        ):
            assert float(row[key]) == pytest.approx(getattr(book, key)[day], abs=1e-8)
        assert row["physical_custody"].numpy() == pytest.approx(
            book.physical_custody[day], abs=1e-9
        )
    return book, records


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("delivery", [4, None])
def test_known_or_unknown_custody_keeps_signed_rights_separate(sign, delivery):
    close = np.full((7, 2), 100.0)
    close[1:, 0] = np.nan
    event = ShareDistribution(
        0, 1, 0, (ShareDelivery(1, 1, delivery),), "dated fixture"
    )
    targets = [[sign * 0.4, 0]] + [[0, 0]] * 6
    primary, rows = checked(close, targets, config(), share_distributions=(event,))
    other, _ = checked(
        close, targets, config(custody_claim_fraction=0), share_distributions=(event,)
    )
    assert primary.custody_physical_base[2] == 0
    assert primary.custody_claim_base[2] == (40000 if sign > 0 else 0)
    assert primary.custody_fee[2] == pytest.approx(
        40000 * 0.0005 / 12 if sign > 0 else 0
    )
    assert other.custody_fee.sum() == 0
    if sign < 0:
        np.testing.assert_array_equal(primary.nav, other.nav)
    else:
        np.testing.assert_array_equal(primary.nav[:2], other.nav[:2])
    assert float(rows[2]["physical_custody"][0]) == 0


@pytest.mark.parametrize("early", [False, True])
def test_precredit_owned_sale_keeps_receipt_and_fee_right_until_credit(early):
    close = np.full((8, 2), 100.0)
    close[2:, 0] = np.nan
    event = ShareDistribution(
        0,
        2,
        0,
        (ShareDelivery(1, 1, 4, disposal_session=2 if early else None),),
        "owned receipt",
    )
    book, _ = checked(
        close,
        [[0.4, 0], [0.4, 0]] + [[0, 0]] * 6,
        config(),
        share_distributions=(event,),
    )
    assert book.custody_base[2] == pytest.approx(40000)
    assert book.custody_claim_base[2] == pytest.approx(40000)
    assert book.physical_custody[2].sum() == pytest.approx(0)
    if early:
        assert any(f.security_index == 1 and f.fill_session == 2 for f in book.fills)
        assert book.physical_custody[4].sum() == pytest.approx(0)


@pytest.mark.parametrize("sign", [-1, 1])
def test_fraction_claim_is_a_right_or_obligation_never_negative_physical_stock(sign):
    close = np.full((8, 2), 100.0)
    close[1:, 0] = np.nan
    event = ShareDistribution(
        0,
        1,
        0,
        (ShareDelivery(1, 0.3333, 3, FractionAuction(5, 100, 6, True)),),
        "fraction fixture",
    )
    book, _ = checked(
        close,
        [[sign * 0.4, 0]] + [[0, 0]] * 7,
        replace(config("2024-01-05"), initial_capital_brl=1000),
        share_distributions=(event,),
    )
    assert np.min(book.physical_custody) >= 0
    assert book.custody_claim_base[3] == pytest.approx(
        33.32 if sign > 0 else 0, abs=1e-9
    )


@pytest.mark.parametrize("source_short", [False, True])
def test_successor_offset_respects_actual_pending_owned_purchase(source_short):
    close = np.full((7, 2), 100.0)
    close[3:, 0] = np.nan
    short, long = (0, 1) if source_short else (1, 0)
    targets = np.zeros((7, 2))
    targets[:3, short] = -0.4
    targets[2, long] = 0.4
    event = ShareDistribution(
        0, 3, 0, (ShareDelivery(1, 1, 3),), "owned pending offset"
    )
    book, _ = checked(
        close, targets, config("2024-01-05"), share_distributions=(event,)
    )
    assert book.physical_custody[1, short] == 400  # Loan D1 precedes the spot sale D2.
    assert np.max(np.abs(book.physical_custody[2:])) < 1e-8
    assert book.custody_base[3] == 0
    assert book.loan_outstanding_principal[3] > 0
    assert book.loan_outstanding_principal[4] == 0


def test_multi_leg_conversion_preserves_pending_purchase_and_sale_value_dates():
    close = np.full((7, 3), 20.0)
    close[:3, 0] = 100
    close[3:, 0] = np.nan
    event = ShareDistribution(
        0,
        3,
        0,
        (
            ShareDelivery(1, 1, 3, loan_principal_fraction=0.2),
            ShareDelivery(2, 4, 3, loan_principal_fraction=0.8),
        ),
        "multi-leg fixture",
    )
    targets = [[0, 0, 0]] * 2 + [[0.4, 0, 0]] + [[0, 0, 0]] * 4
    book, _ = checked(
        close, targets, config("2024-01-06"), share_distributions=(event,)
    )
    assert book.physical_custody[3].sum() == pytest.approx(0, abs=1e-9)
    assert book.physical_custody[4, 1:3] == pytest.approx([400, 1600], abs=1e-9)
    assert book.custody_base[4] == pytest.approx(40000)
    assert book.physical_custody[5].sum() == pytest.approx(0, abs=1e-9)


def test_loan_cash_extinction_retains_pending_cover_stock_until_its_sale_settles():
    close = np.full((8, 1), 100.0)
    event = LoanCashSettlement(0, 3, 0, 100, "cash closeout fixture", payment_session=5)
    book, _ = checked(
        close,
        [[-0.4], [-0.4]] + [[0]] * 6,
        config("2024-01-06"),
        loan_cash_settlements=(event,),
    )
    assert book.physical_custody[3, 0] == 0
    assert book.physical_custody[4, 0] == 400
    assert book.custody_physical_base[4] == 40000
    assert book.custody_claim_base[4] == 0
    assert book.physical_custody[6, 0] == 0


def test_delivered_unquoted_stock_uses_explicit_opening_value_for_custody():
    close = np.array(
        [
            [100, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, 120],
            [np.nan, 120],
        ]
    )
    event = ShareDistribution(
        0,
        1,
        0,
        (ShareDelivery(1, 1, 3),),
        "opening value fixture",
        carry_source_value=True,
    )
    cfg = config("2024-01-05")
    targets = [[0.4, 0]] + [[0, 0]] * 5
    book = _run(
        close,
        np.ones_like(close),
        initial_reference_price=np.array([100, np.nan]),
        portfolio_policy=lambda s: PortfolioTarget(np.asarray(targets[s.day])),
        share_distributions=(event,),
        config=cfg,
    )
    account = PortfolioAccount.empty([100, np.nan, 100], config=cfg)
    for day in range(6):
        row = account.step(
            tensor([*targets[day], 0]),
            day=day,
            close=[*close[day], 100],
            cdi=0,
            session_date=book.dates[day],
            annual_borrow=[0, 0, 0],
            loan_reference=[100, np.nan, 100],
            share_distributions=(event,),
            terminal=day == 5,
        )
        assert float(row["nav"]) == pytest.approx(book.nav[day], abs=1e-8)
        assert float(row["custody_base"]) == pytest.approx(
            book.custody_base[day], abs=1e-8
        )
    assert book.custody_base[3] == 40000
    assert not any(f.security_index == 1 and f.fill_session <= 3 for f in book.fills)


def test_delayed_bonus_physical_quantity_and_gross_rights():
    close = np.full((7, 1), 100.0)
    close[2:] = 50
    q = np.ones_like(close)
    q[2, 0] = 2
    actions = AlignedActionTerms(
        q, np.zeros_like(q), np.ones_like(q, bool), q != 1, np.zeros_like(q, int)
    )
    settlement = ActionSettlement(0, 2, 0, "bonus fixture", bonus_delivery_session=4)
    book, _ = checked(
        close,
        [[0.4]] * 2 + [[0]] * 5,
        config(),
        actions=actions,
        action_settlements=(settlement,),
    )
    # The original 400 shares can be sold; physical delivery of that sale is D2.
    assert book.physical_custody[2, 0] == pytest.approx(400)
    assert book.custody_physical_base[2] == pytest.approx(20000)
    assert book.custody_claim_base[2] == pytest.approx(20000)
    assert book.custody_base[2] == pytest.approx(40000)


@pytest.mark.parametrize("cover", [False, True])
def test_bonus_on_pending_owned_purchase_or_cover_has_two_physical_dates(cover):
    close = np.full((7, 1), 100.0)
    close[2:] = 50
    q = np.ones_like(close)
    q[2, 0] = 2
    actions = AlignedActionTerms(
        q, np.zeros_like(q), np.ones_like(q, bool), q != 1, np.zeros_like(q, int)
    )
    settlement = ActionSettlement(0, 2, 0, "pending bonus", bonus_delivery_session=4)
    targets = ([[-0.4]] + [[0]] * 6) if cover else ([[0]] + [[0.4]] * 3 + [[0]] * 3)
    book, _ = checked(
        close,
        targets,
        config("2024-01-05"),
        actions=actions,
        action_settlements=(settlement,),
    )
    assert book.physical_custody[2, 0] == 0
    assert book.physical_custody[3, 0] == pytest.approx(400)
    assert book.custody_physical_base[3] == pytest.approx(20000)
    assert book.custody_claim_base[3] == pytest.approx(20000)
    if cover:
        assert book.physical_custody[4, 0] == 0


def test_corporate_pending_fee_gradient_and_independent_copy():
    event = ShareDistribution(
        0, 2, 0, (ShareDelivery(1, 1, 4, disposal_session=2),), "copy fixture"
    )

    def run(weight, probe=False):
        account = PortfolioAccount.empty([100, 100, 100], config=config())
        for day in range(6):
            row = account.step(
                torch.stack((weight, weight * 0, weight * 0))
                if day < 2
                else weight * torch.zeros(3),
                day=day,
                close=[100 if day < 2 else np.nan, 100, 100],
                cdi=0,
                session_date=f"2024-01-{day + 2:02d}",
                annual_borrow=[0, 0, 0],
                loan_reference=[100, 100, 100],
                share_distributions=(event,),
            )
            if day == 2 and probe:
                copied = clone_account(account)
                copied.custody_fees.corporate_pending[0][2].add_(1)
                assert not torch.equal(
                    copied.custody_fees.corporate_pending[0][2],
                    account.custody_fees.corporate_pending[0][2],
                )
                copied.detach()
                assert not copied.custody_fees.corporate_pending[0][2].requires_grad
                assert float(row["custody_claim_base"]) > 0
        return account.nav

    weight = tensor(0.4).requires_grad_()
    run(weight, True).backward()
    finite = (run(tensor(0.400001)) - run(tensor(0.399999))) / 0.000002
    assert float(weight.grad) == pytest.approx(float(finite), abs=1e-5)
