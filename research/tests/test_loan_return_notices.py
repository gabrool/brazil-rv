from dataclasses import replace

import numpy as np
import pytest
import torch

from brazil_rv.execution.loan_contracts import LoanContracts, LoanRecall, LoanSession
from brazil_rv.execution.portfolio_policy import (
    CalibratedPolicy,
    account_decision,
    exact_replay,
    policy_ledger_config,
)
from brazil_rv.v2.portfolio_inputs import Calibration
from test_portfolio_policy import policy_fixture


def test_denial_preserves_original_terms_pending_return_and_cash_until_delivery():
    q = torch.tensor([100.0], dtype=torch.float64, requires_grad=True)
    book = LoanContracts(1, fee_multiplier=0)
    book.open(q, [20], [0.1], 0, "2024-01-02")
    book.request_return(q * 0.4, 6, request_day=4)
    # Unusable renewal references do not stop a denial or reprice the old loan.
    book.renew(LoanSession(4, "2024-01-08", [np.nan], [np.nan], 6), 8, approved=False)
    assert book.principal.tolist() == [1200, 800]
    assert book.return_day.tolist() == [-1, 6]
    assert book.return_deadline.tolist() == [8, 8]
    assert not book.renewals
    assert book.short_blocked(5).tolist() == [True]
    fork = book.detached_copy()
    book.request_return(q * 0.6, 9, request_day=7)
    assert fork.return_day.tolist() == [-1, 6]
    assert fork.return_notices is not book.return_notices
    paid = q.sum() * 0
    for day in range(10):
        book.accrue(day, "2024-01-02")
        if day == 8:
            assert book.overdue_principal(day).item() == 1200
        rent, fees = book.pay(day)
        paid = paid + rent.sum() + fees.sum()
    expected = 800 * (1.1 ** (6 / 252) - 1) + 1200 * (1.1 ** (9 / 252) - 1)
    assert paid.item() == pytest.approx(expected, abs=1e-11)
    paid.backward()
    assert q.grad.item() == pytest.approx(expected / 100, abs=1e-12)
    assert book.liability.item() == 0
    assert not book.short_blocked(10).any()


def test_recall_follows_split_and_delivery_and_never_pays_on_deadline_alone():
    book = LoanContracts(2)
    book.open([10, 0], [20, 10], [0.1, 0], 0, "2019-01-02")
    event = LoanRecall(0, 2, 4, "four-session fixture hypothesis")
    book.observe_recalls(1, [event])
    assert not book.short_blocked(1, [event]).any()
    book.observe_recalls(2, [event])
    book.split(0, 2)
    book.deliver(0, 1, 0.5, 1.0, final=True)
    assert book.return_deadline.tolist() == [4]
    book.accrue(4, "2019-01-08")
    before = book.liability.item()
    assert book.pay(4)[0].sum() == 0
    assert book.liability.item() == before
    assert book.overdue_principal(4).tolist() == [0, 200]
    assert book.short_blocked(5).tolist() == [False, True]


@pytest.mark.parametrize("mode", ["denied", "recall"])
def test_adaptive_controller_accounts_match_notices_offsets_and_missing_covers(mode):
    torch.set_num_threads(1)
    data = policy_fixture()
    names = len(data.inputs.security_ids)
    start, stop = 3, 18
    model = CalibratedPolicy(
        Calibration(np.zeros(3), np.ones(3), np.array([0.002, 0, 0]), 0)
    )
    recalls = tuple(
        LoanRecall(i, 8, 10, "two-session stress") for i in range(names + 1)
    )
    config = policy_ledger_config(
        loan_term_sessions=8,
        approve_loan_renewals=mode != "denied",
        loan_recalls=recalls if mode == "recall" else (),
    )
    # Missing covers cross the explicit deadline, but never cause synthetic fills.
    close = data.inputs.raw_close.copy()
    close[8:13] = np.nan
    data.inputs = replace(data.inputs, raw_close=close)
    exact, chosen, _ = exact_replay(data, model, start, stop, config=config)
    account = data.initial_account(start, config)
    with torch.no_grad():
        for day in range(start, stop):
            target = account_decision(data, model, account, day)
            if day < stop - 1:
                # Independent free-face QPs retain their numerical solve tolerance;
                # money parity below is tighter than the economic admission scale.
                np.testing.assert_allclose(target, chosen[day - start], atol=2e-7)
            row = data.step(account, target, day, terminal=day == stop - 1)
            assert row["nav"].item() == pytest.approx(exact.nav[day - start], abs=1e-9)
            assert row["loan_overdue_principal"].item() == pytest.approx(
                exact.loan_overdue_principal[day - start], abs=1e-10
            )
    assert exact.loan_return_notices
    assert exact.loan_overdue_principal.max() > 0
    assert exact.economics_unresolved
    assert not [
        x for x in exact.fills if 5 <= x.fill_session < 10 and x.security_index < names
    ]
    # Deleting a future recall cannot alter decisions before its notice.
    if mode == "recall":
        other, targets, _ = exact_replay(
            data, model, start, stop, config=replace(config, loan_recalls=())
        )
        np.testing.assert_array_equal(chosen[:5], targets[:5])
        np.testing.assert_array_equal(exact.nav[:5], other.nav[:5])


def test_denial_is_after_intentions_and_before_next_adaptive_decision():
    torch.set_num_threads(1)
    data = policy_fixture()
    model = CalibratedPolicy(
        Calibration(np.zeros(3), np.ones(3), np.array([0.002, 0, 0]), 0)
    )
    primary = policy_ledger_config(loan_term_sessions=8)
    a, before, _ = exact_replay(data, model, 0, 12, config=primary)
    b, after, _ = exact_replay(
        data, model, 0, 12, config=replace(primary, approve_loan_renewals=False)
    )
    np.testing.assert_array_equal(before[:5], after[:5])
    assert b.loan_return_notices[0].session == 4
    assert a.loan_renewals[0].session == 4
    assert not np.array_equal(before[5], after[5])


def test_full_return_roundoff_never_renews_residue_but_tiny_partial_loans_survive():
    book = LoanContracts(1)
    for day in range(4):
        book.open([0.1], [20], [0.1], day, "2019-01-02")
    total = float(book.active_quantity[0])
    book.request_return([np.nextafter(total, 0)], 6, request_day=4)
    assert book.active_quantity.item() == 0
    assert book.quantity.sum().item() == total
    book.renew(LoanSession(4, "2019-01-02", [20], [0.1], 6), 8)
    assert not book.renewals
    tiny = LoanContracts(1)
    tiny.open([1e-20], [20], [0.1], 0, "2019-01-02")
    tiny.request_return([0.4e-20], 2, request_day=1)
    assert tiny.active_quantity.item() == pytest.approx(0.6e-20, rel=1e-14, abs=0)
    tiny.request_return([0], 2)
    assert tiny.active_quantity.item() > 0


def test_mandatory_cover_closes_a_real_position_below_normal_order_threshold():
    from test_portfolio_account import compare
    from test_portfolio_ledger import replay
    from test_v2_stateful_ledger import _config
    from brazil_rv.execution.stateful_ledger import PortfolioTarget

    targets = np.array([[0.01, -0.01], [0.01, -1e-11], [0.01, 0], [0.01, 0], [0, 0]])
    config = _config(loan_recalls=(LoanRecall(1, 2, 4, "fixture"),))
    prices = np.full((5, 2), 100.0)
    compare(prices, targets, config=config)
    book = replay(
        prices, lambda state: PortfolioTarget(targets[state.day]), config=config
    )
    assert book.signed_shares[1, 1] < 0
    assert book.signed_shares[2, 1] == 0
    assert not book.loan_overdue_principal.any()


@pytest.mark.parametrize("called_name", [0, 1])
def test_morning_delivery_precedes_recall_identity_in_both_accounts(called_name):
    from test_portfolio_account import compare
    from test_portfolio_ledger import replay
    from test_v2_stateful_ledger import _config
    from brazil_rv.execution.share_distributions import ShareDelivery, ShareDistribution
    from brazil_rv.execution.stateful_ledger import PortfolioTarget

    events = (ShareDistribution(0, 1, 1, (ShareDelivery(1, 1, 2),), "fixture"),)
    config = _config(loan_recalls=(LoanRecall(called_name, 2, 4, "fixture"),))
    close = np.array(
        [[100, 100], [np.nan, 100], [np.nan, 100], [np.nan, 100], [np.nan, 100]]
    )
    retained = -0.1 if called_name == 0 else 0
    targets = np.array([[-0.1, 0], [-0.1, 0], [0, retained], [0, retained], [0, 0]])
    compare(close, targets, config=config, share_distributions=events)
    book = replay(
        close,
        lambda state: PortfolioTarget(targets[state.day]),
        config=config,
        share_distributions=events,
    )
    assert len(book.loan_return_notices) == called_name
    if called_name:
        assert book.loan_return_notices[0].security_index == 1
        assert book.signed_shares[2, 1] == 0


def test_recall_cannot_address_negative_or_out_of_axis_security():
    from brazil_rv.execution.portfolio_account import PortfolioAccount

    with pytest.raises(ValueError, match="nonnegative"):
        LoanRecall(-1, 0, 2, "fixture")
    with pytest.raises(ValueError, match="outside"):
        PortfolioAccount.empty(
            [100],
            config=policy_ledger_config(loan_recalls=(LoanRecall(1, 0, 2, "fixture"),)),
        )
