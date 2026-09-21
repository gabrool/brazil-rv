from dataclasses import replace

import numpy as np
import pytest

from brazil_rv.execution.portfolio_account import PortfolioAccount, tensor
from brazil_rv.execution.share_distributions import (
    ShareDelivery,
    ShareDistribution,
    slice_distributions,
)
from brazil_rv.execution.stateful_ledger import PortfolioTarget
from test_portfolio_account import compare
from test_portfolio_ledger import replay
from test_v2_stateful_ledger import _config


@pytest.mark.parametrize("sign", [-1, 1])
def test_dated_cash_revisions_retain_original_units_after_share_delivery(sign):
    event = ShareDistribution(
        0,
        1,
        0,
        (ShareDelivery(1, 0.5, 2),),
        "dated cash fixture",
        cash_per_prior_share=50,
        payment_session=6,
        cash_values=((3, 52), (5, 51)),
    )
    close = np.array([[100, 100]] + [[np.nan, 100]] * 7, float)
    targets = [[sign * 0.4, 0]] + [[0, 0]] * 7
    config = _config(initial_capital_brl=1000)
    result = replay(
        close,
        lambda state: PortfolioTarget(np.asarray(targets[state.day])),
        share_distributions=(event,),
        config=config,
    )
    units = result.signed_shares[0, 0]
    # Successor shares have been sold/covered before the second cash revision.
    assert result.signed_shares[4, 1] == 0
    cash = result.receivables - result.payables
    np.testing.assert_allclose(
        cash[1:6], units * np.array([50, 50, 52, 52, 51]), atol=1e-12
    )
    np.testing.assert_allclose(cash[6:], 0, atol=1e-12)
    compare(close, targets, config=config, share_distributions=(event,))
    changed = replace(event, cash_values=((3, 60), (5, 63)))
    future = replay(
        close,
        lambda state: PortfolioTarget(np.asarray(targets[state.day])),
        share_distributions=(changed,),
        config=config,
    )
    np.testing.assert_array_equal(result.nav[:3], future.nav[:3])
    assert [o for o in result.intended_orders if o.decision_session <= 3] == [
        o for o in future.intended_orders if o.decision_session <= 3
    ]
    # Unit-value gains, not a second principal payment; signed short cash follows.
    assert future.nav[-1] - result.nav[-1] == pytest.approx(units * 12, abs=1e-9)
    sliced = slice_distributions((event,), 2, 8)[0]
    assert sliced.cash_values == ((1, 52), (3, 51))
    with pytest.raises(ValueError, match="ordered known values"):
        replace(event, cash_values=((6, 51),))


def test_cash_revision_gradient_and_copied_pending_payment_are_independent():
    import torch
    from brazil_rv.v2.portfolio_objective import clone_account

    event = ShareDistribution(
        0,
        1,
        0,
        (ShareDelivery(1, 0.5, 2),),
        "dated cash fixture",
        cash_per_prior_share=50,
        payment_session=5,
        cash_values=((3, 52),),
    )
    weight = tensor(0.4).requires_grad_()
    account = PortfolioAccount.empty([100, 100, 100], config=_config())
    for day in range(6):
        account.step(
            torch.stack((weight, weight * 0, weight * 0))
            if day == 0
            else tensor([0, 0, 0]),
            day=day,
            close=[100 if day == 0 else np.nan, 100, 100],
            cdi=0,
            session_date="2024-01-02",
            annual_borrow=[0, 0, 0],
            loan_reference=[100, 100, 100],
            share_distributions=(event,),
        )
        if day == 2:
            copied = clone_account(account)
            copied.payments[0][1][0] += 1
            copied.detach()
            assert account.payments[0][1][0].item() == pytest.approx(0.2)
    account.nav.backward()
    assert account.nav.item() == pytest.approx(1.008)
    assert weight.grad.item() == pytest.approx(0.02)


@pytest.mark.parametrize("sign", [-1, 1])
def test_reused_source_reopens_after_conversion_without_old_inventory(sign):
    event = ShareDistribution(
        0,
        1,
        0,
        (ShareDelivery(1, 1, 1),),
        "dated holding/logistics fixture",
        source_reopens_session=3,
    )
    close = np.array(
        [
            [100, 100],
            [np.nan, 100],
            [999, 100],
            [10, 100],
            [11, 100],
            [12, 100],
            [12, 100],
        ],
        dtype=float,
    )
    targets = [
        [sign * 0.4, 0],
        [0, sign * 0.4],
        [0, sign * 0.4],
        [0, sign * 0.4],
        [sign * 0.2, sign * 0.2],
        [sign * 0.2, sign * 0.2],
        [0, 0],
    ]
    config = _config(initial_capital_brl=1000)
    exact = replay(
        close,
        lambda state: PortfolioTarget(np.asarray(targets[state.day])),
        share_distributions=(event,),
        config=config,
    )
    account = PortfolioAccount.empty([100, 100, 100], config=config)
    for day, target in enumerate(targets):
        account.step(
            tensor([*target, 0]),
            day=day,
            close=np.r_[close[day], 100],
            cdi=0,
            session_date=exact.dates[day],
            annual_borrow=[0, 0, 0],
            loan_reference=[100, 100, 100],
            share_distributions=(event,),
            terminal=day == len(targets) - 1,
        )
        assert account.nav.item() == pytest.approx(exact.nav[day], abs=1e-9)
        np.testing.assert_allclose(
            account.shares.numpy()[:2], exact.signed_shares[day], atol=1e-12
        )
    assert np.all(exact.signed_shares[1:4, 0] == 0)
    assert exact.signed_shares[4, 0] * sign > 0
    assert 0 not in account.retired_sources
    assert not [
        f for f in exact.fills if f.security_index == 0 and f.fill_session in (1, 2, 3)
    ]
    sliced = slice_distributions((event,), 4, 7)[0]
    assert sliced.source_reopens_session == -1
    from brazil_rv.execution.share_distributions import retired_distribution_sources

    assert retired_distribution_sources((sliced,), 0) == set()
    from brazil_rv.v2.portfolio_objective import clone_account

    copied = clone_account(account)
    copied.retired_sources.add(0)
    assert not account.retired_sources
    copied.detach()


def test_reused_identity_rejects_overlapping_old_share_claim():
    with pytest.raises(ValueError, match="prior complete"):
        ShareDistribution(
            0, 1, 0, (ShareDelivery(1, 1, None),), "fixture", source_reopens_session=3
        )


def test_pending_fraction_follows_sourced_unit_rename_without_new_quote():
    from brazil_rv.execution.share_distributions import FractionAuction
    from brazil_rv.v2.corporate_actions import AlignedActionTerms
    from test_v2_stateful_ledger import _run

    close = np.array(
        [
            [100, 50, np.nan],
            [np.nan, 50, np.nan],
            [np.nan, 50, np.nan],
            [np.nan, np.nan, 70],
            [np.nan, np.nan, 70],
        ],
        float,
    )
    q = np.ones_like(close)
    successor = np.broadcast_to(np.arange(3), close.shape).copy()
    successor[3, 1] = 2
    flags = np.zeros(close.shape, bool)
    flags[3, 1] = True
    actions = AlignedActionTerms(
        q, np.zeros_like(close), np.ones(close.shape, bool), flags, successor
    )
    event = ShareDistribution(
        0,
        1,
        0,
        (ShareDelivery(1, 0.3, 2, FractionAuction(None, None, None)),),
        "fixture",
    )
    targets = [[0.4, 0, 0]] + [[0, 0, 0]] * 4
    config = _config(initial_capital_brl=1000)
    result = _run(
        close,
        np.ones_like(close),
        actions=actions,
        initial_reference_price=np.array([100, 50, np.nan]),
        portfolio_policy=lambda s: PortfolioTarget(np.asarray(targets[s.day])),
        share_distributions=(event,),
        config=config,
    )
    future_close = close.copy()
    future_close[3:, 2] = 700
    future = _run(
        future_close,
        np.ones_like(close),
        actions=actions,
        initial_reference_price=np.array([100, 50, np.nan]),
        portfolio_policy=lambda s: PortfolioTarget(np.asarray(targets[s.day])),
        share_distributions=(event,),
        config=config,
    )
    np.testing.assert_array_equal(result.nav[:3], future.nav[:3])
    assert [o for o in result.intended_orders if o.decision_session <= 3] == [
        o for o in future.intended_orders if o.decision_session <= 3
    ]
    account = PortfolioAccount.empty([100, 50, np.nan, 100], config=config)
    for day in range(len(close)):
        account.step(
            tensor([*targets[day], 0]),
            day=day,
            close=np.r_[close[day], 100],
            cdi=0,
            session_date=result.dates[day],
            annual_borrow=[0] * 4,
            loan_reference=[100, 50, np.nan, 100],
            action_q=np.r_[q[day], 1],
            successor=np.r_[successor[day], 3],
            share_distributions=(event,),
            terminal=day == 4,
        )
        assert account.nav.item() == pytest.approx(result.nav[day], abs=1e-9)
        if day == 2:
            from brazil_rv.v2.portfolio_objective import clone_account

            copied = clone_account(account)
            copied.distributions[0][0] = replace(
                copied.distributions[0][0], successor_index=2
            )
            copied.detach()
            assert account.distributions[0][0].successor_index == 1
    assert account.distributions[0][0].successor_index == 2
    assert result.undelivered_share_notional[-1] == pytest.approx(14)
    assert account.shares[0].item() == pytest.approx(0.2 / 0.3)
    assert not any(f.security_index == 2 and f.side == "buy" for f in result.fills)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("delivery", [2, None])
def test_unquoted_single_leg_carries_only_claim_value(sign, delivery):
    from test_v2_stateful_ledger import _run

    close = np.array(
        [[100.0, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, 55.0]]
    )
    event = ShareDistribution(
        0,
        1,
        0,
        (ShareDelivery(1, 2, delivery),),
        "explicit continuity hypothesis",
        carry_source_value=True,
    )
    states = []

    def policy(state):
        states.append(state)
        return PortfolioTarget(
            np.array([sign * 0.4, 0]) if state.day == 0 else np.zeros(2)
        )

    config = _config()
    result = _run(
        close,
        np.ones_like(close),
        initial_reference_price=np.array([100.0, np.nan]),
        portfolio_policy=policy,
        share_distributions=(event,),
        config=config,
    )
    account = PortfolioAccount.empty([100.0, np.nan, 100.0], config=config)
    for day in range(4):
        target = tensor([sign * 0.4, 0.0, 0.0] if day == 0 else [0.0, 0.0, 0.0])
        account.step(
            target,
            day=day,
            close=np.r_[close[day], 100.0],
            cdi=0,
            session_date=result.dates[day],
            annual_borrow=[0.0, 0.0, 0.0],
            loan_reference=[100.0, np.nan, 100.0],
            share_distributions=(event,),
            terminal=day == 3,
        )
        assert account.nav.item() == pytest.approx(result.nav[day], abs=1e-12)
        assert account.shares[:-1].numpy() == pytest.approx(result.signed_shares[day])
        assert account.market_weights[:-1].numpy() == pytest.approx(
            result.equity_market_weights[day]
        )
        if day == 1:
            assert account.marks[1].item() == 0  # No public successor mark seeded.
            assert account.distributions[0][0].opening_mark == 50
        if day == 2 and delivery == 2:
            from brazil_rv.v2.portfolio_objective import clone_account

            allowed, _ = account.eligibility(np.ones(3, bool), np.zeros(3, bool))
            assert not allowed[1]
            copied = clone_account(account)
            copied.unquoted_deliveries.clear()
            assert account.unquoted_deliveries == {1}
    assert result.nav[:3] == pytest.approx([1.0, 1.0, 1.0])
    assert result.unpriced_inventory_notional[1:3] == pytest.approx([0.4, 0.4])
    assert not states[2].entry_allowed[1]
    assert not [f for f in result.fills if f.fill_session in (1, 2)]
    if delivery is None:
        assert result.signed_shares[-1, 0] != 0
        assert result.economics_unresolved


def test_opening_carry_requires_explicit_contract_and_never_uses_future_quote():
    from brazil_rv.execution.share_distributions import (
        recognize_distribution,
        basket_prices,
    )

    event = ShareDistribution(0, 1, 0, (ShareDelivery(1, 0.5, None),), "fixture")
    refs = np.array([80.0, np.nan])
    with pytest.raises(ValueError, match="unpriced"):
        recognize_distribution(event, refs)
    legs = recognize_distribution(replace(event, carry_source_value=True), refs)
    assert np.isnan(refs[1])
    assert basket_prices(legs, refs).tolist() == [160.0]
    assert basket_prices(legs, [80.0, 170.0]).tolist() == [170.0]
    with pytest.raises(ValueError, match="one identifiable"):
        replace(distribution(), carry_source_value=True)


def test_first_successor_quote_cannot_change_current_intentions_or_opening_mark():
    from test_v2_stateful_ledger import _run

    event = ShareDistribution(
        0,
        1,
        0,
        (ShareDelivery(1, 2.0, 2),),
        "continuity fixture",
        carry_source_value=True,
    )
    books = []
    for first in (50.0, 90.0):
        close = np.array([[100.0, np.nan], [np.nan, first], [np.nan, first]])

        def policy(state):
            return PortfolioTarget(
                np.array([0.4, 0.0]) if state.day == 0 else np.zeros(2)
            )

        books.append(
            _run(
                close,
                np.ones_like(close),
                portfolio_policy=policy,
                initial_reference_price=np.array([100.0, np.nan]),
                share_distributions=(event,),
            )
        )
    assert [x for x in books[0].intended_orders if x.decision_session <= 1] == [
        x for x in books[1].intended_orders if x.decision_session <= 1
    ]
    assert books[0].nav[0] == books[1].nav[0]
    assert books[0].nav[1] != books[1].nav[1]


def distribution(delivery=3):
    return ShareDistribution(
        source_index=0,
        effective_session=1,
        available_session=0,
        legs=(
            ShareDelivery(1, 1.0, 2, loan_principal_fraction=1 / 3),
            ShareDelivery(2, 2.0, delivery, loan_principal_fraction=2 / 3),
        ),
        cash_per_prior_share=10,
        payment_session=4,
        source="synthetic signed claim fixture",
    )


@pytest.mark.parametrize("sign", [-1, 1])
def test_separate_effective_delivery_and_payment_with_existing_positions(sign):
    close = np.array([[100, 30, 30]] + [[np.nan, 30, 30]] * 4)
    targets = (
        np.array([[0.4, -0.1, 0.1], [0, -0.1, 0.1], [0, 0, 0.1], [0, 0, 0], [0, 0, 0]])
        * sign
    )
    event = distribution()
    compare(close, targets, share_distributions=(event,), config=_config())
    states = []

    def policy(state):
        states.append(state)
        return PortfolioTarget(targets[state.day])

    result = replay(close, policy, share_distributions=(event,))
    assert result.nav == pytest.approx(np.ones(5))
    assert result.undelivered_share_notional == pytest.approx([0, 0.36, 0.24, 0, 0])
    assert result.receivables - result.payables == pytest.approx(
        np.array([0, 0.04, 0.04, 0.04, 0]) * sign
    )
    assert not [fill for fill in result.fills if fill.fill_session == 1]
    assert result.signed_shares[1, 0] == pytest.approx(sign * 0.004)
    assert result.signed_shares[-1] == pytest.approx(np.zeros(3))
    assert (
        result.free_cash[-1] + result.restricted_cash[-1] + result.unsettled_cash[-1]
    ) == pytest.approx(1)
    assert result.unpriced_inventory_notional == pytest.approx(np.zeros(5))
    assert states[2].locked.tolist() == [True, False, False]
    # The delivered first leg is in the decision state before current prints.
    assert states[2].weights == pytest.approx(np.array([0.24, 0.02, 0.1]) * sign)
    assert result.equity_market_weights[1] == pytest.approx(
        np.array([0, 0.02, 0.34]) * sign
    )
    assert len(result.share_claim_positions) == 3


@pytest.mark.parametrize("sign", [-1, 1])
def test_terminal_and_unknown_delivery_retain_valued_claim_and_financing(sign):
    close = np.array([[100, 30, 30], [np.nan, 31, 32], [np.nan, 32, 34]])
    event = replace(
        distribution(None),
        legs=(
            ShareDelivery(1, 1, None, loan_principal_fraction=1 / 3),
            ShareDelivery(2, 2, 7, loan_principal_fraction=2 / 3),
        ),
    )
    targets = np.array([[0.4, 0, 0], [0, 0, 0], [0, 0, 0]]) * sign
    config = _config(annual_borrow_rate=0.04)
    compare(
        close,
        targets,
        share_distributions=(event,),
        config=config,
        cdi=np.full(3, 0.0004),
    )
    result = replay(
        close,
        lambda state: PortfolioTarget(targets[state.day]),
        share_distributions=(event,),
        config=config,
        cdi=np.full(3, 0.0004),
    )
    assert result.undelivered_share_notional[-1] == pytest.approx(0.4)
    assert result.signed_shares[-1, 0] == pytest.approx(sign * 0.004)
    assert result.restricted_cash[-1] == pytest.approx(0.4 if sign < 0 else 0)
    assert not [fill for fill in result.fills if fill.fill_session > 0]
    assert result.economics_unresolved
    if sign < 0:
        assert result.borrow_bps[-1] > 0


def test_delivery_current_quote_mutation_does_not_change_prior_decision():
    event = distribution()
    states = []
    books = []
    for price in [30, 90]:
        close = np.array([[100.0, 30, 30]] + [[np.nan, 30, 30]] * 4)
        close[2, 1] = price
        snapshots = []

        def policy(state):
            snapshots.append(state)
            return PortfolioTarget(
                np.array([0.4, 0, 0]) if state.day == 0 else np.zeros(3)
            )

        books.append(replay(close, policy, share_distributions=(event,)))
        states.append(snapshots)
    assert states[0][2].weights == pytest.approx(states[1][2].weights)
    assert [
        order for order in books[0].intended_orders if order.decision_session <= 2
    ] == [order for order in books[1].intended_orders if order.decision_session <= 2]
    assert books[0].nav[2] != books[1].nav[2]


def test_basket_gradient_survives_marking_partial_delivery_and_detach():
    def run(weight, detach=False):
        account = PortfolioAccount.empty([100, 30, 30, 100], config=_config())
        import torch

        account.step(
            torch.stack((weight, weight * 0, weight * 0, weight * 0)),
            day=0,
            close=[100, 30, 30, 100],
            cdi=0,
            session_date="2024-01-02",
            annual_borrow=[0] * 4,
            loan_reference=[100] * 4,
        )
        for day, prices in [(1, [np.nan, 31, 32, 100]), (2, [np.nan, 32, 34, 100])]:
            account.prepare_day(day)
            account.step(
                account.weights,
                day=day,
                close=prices,
                cdi=0,
                session_date="2024-01-02",
                annual_borrow=[0] * 4,
                loan_reference=[100] * 4,
                share_distributions=(distribution(),),
            )
            if detach and day == 1:
                account.detach()
        assert account.market_weights[0].item() == 0
        return account.nav

    weight = tensor(0.4).requires_grad_()
    value = run(weight)
    value.backward()
    assert value.item() == pytest.approx(1.04)
    assert weight.grad.item() == pytest.approx(0.1)
    finite_difference = (run(tensor(0.400001)) - run(tensor(0.399999))) / 0.000002
    assert weight.grad.item() == pytest.approx(finite_difference.item(), abs=1e-9)
    assert run(tensor(0.4), detach=True).item() == pytest.approx(value.item())


def test_distribution_clock_rejects_backdating_and_rebases_without_truncating_claims():
    with pytest.raises(ValueError, match="backdated"):
        replace(distribution(), available_session=2)
    with pytest.raises(ValueError, match="precede"):
        replace(distribution(), legs=(ShareDelivery(1, 1, 0),))
    rebased = slice_distributions((distribution(),), 1, 3)[0]
    assert rebased.effective_session == 0
    assert rebased.available_session == -1
    assert rebased.payment_session == 3
    assert rebased.legs[1].delivery_session == 2
    prior = slice_distributions((distribution(),), 2, 5)[0]
    assert prior.effective_session == -1
    assert prior.source_index == 0


def test_flat_start_keeps_prior_cancellation_without_inventing_historical_claims():
    events = slice_distributions((distribution(),), 2, 5)
    states = []

    def policy(state):
        states.append(state)
        return PortfolioTarget(np.zeros(3))

    result = replay(np.full((3, 3), 100.0), policy, share_distributions=events)
    assert not states[0].entry_allowed[0]
    assert result.nav == pytest.approx(np.ones(3))
    assert not result.share_claim_positions
    assert result.receivables == pytest.approx(np.zeros(3))
    compare(np.full((3, 3), 100.0), np.zeros((3, 3)), share_distributions=events)


@pytest.mark.parametrize("on_fraction", [0.0, 0.2, 1.0])
def test_contract_principal_allocation_is_independent_of_constituent_marks(on_fraction):
    import torch

    close = np.array([[100.0, 10.0, 30.0]] + [[np.nan, 10.0, 30.0]] * 7)
    event = ShareDistribution(
        0,
        1,
        0,
        (
            ShareDelivery(1, 1, 2, loan_principal_fraction=on_fraction),
            ShareDelivery(2, 3, 3, loan_principal_fraction=1 - on_fraction),
        ),
        "explicit fixed-principal allocation fixture",
    )
    config = _config(annual_borrow_rate=0.04)
    targets = np.zeros((8, 3))
    targets[0, 0] = -0.4
    compare(close, targets, share_distributions=(event,), config=config)
    result = replay(
        close,
        lambda state: PortfolioTarget(targets[state.day]),
        share_distributions=(event,),
        config=config,
    )
    # Synthetic dates are in 2022: each exit returns on T+2. Both legs inherit
    # the original rate/principal, despite markedly different constituent prices.
    expected_rent = 0.4 * (
        on_fraction * (1.04 ** (4 / 252) - 1)
        + (1 - on_fraction) * (1.04 ** (5 / 252) - 1)
    )
    assert result.nav[-1] == pytest.approx(1 - expected_rent, abs=1e-12)
    scale = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    account = PortfolioAccount.empty(np.r_[close[0], 100.0], config=config)
    for day in range(8):
        target = torch.zeros(4, dtype=torch.float64)
        if day == 0:
            target[0] = -scale
        account.step(
            target,
            day=day,
            close=np.r_[close[day], 100.0],
            cdi=0.0,
            session_date=result.dates[day],
            loan_reference=np.full(4, 100.0),
            annual_borrow=np.full(4, 0.04),
            share_distributions=(event,),
            terminal=day == 7,
        )
    account.nav.backward()
    assert scale.grad.item() == pytest.approx(-expected_rent / 0.4, abs=1e-12)


def test_multi_leg_loan_terms_cannot_silently_use_price_allocations():
    with pytest.raises(ValueError, match="sum to one"):
        ShareDistribution(
            0, 1, 0, (ShareDelivery(1, 1, 2), ShareDelivery(2, 4, 2)), "terms"
        )
