from dataclasses import replace
from datetime import date, timedelta

import numpy as np

from brazil_rv.execution.stateful_ledger import (
    LedgerConfig,
    StatefulLedgerResult,
    simulate_stateful_ledger,
)
from brazil_rv.v2.corporate_actions import AlignedActionTerms


def _config(**changes: object) -> LedgerConfig:
    return replace(
        LedgerConfig(
            k_per_side=1,
            buffer_per_side=1,
            cost_bps_per_side=0.0,
            annual_borrow_rate=0.0,
            planned_absolute_net_cap=1.10,
            planned_name_weight_cap=1.10,
        ),
        **changes,
    )


def _run(
    close: np.ndarray,
    scores: np.ndarray,
    *,
    config: LedgerConfig | None = None,
    active: np.ndarray | None = None,
    fill_fraction: np.ndarray | None = None,
    actions: AlignedActionTerms | None = None,
    payment_session: np.ndarray | None = None,
    cdi: np.ndarray | None = None,
    initial_reference_price: np.ndarray | None = None,
) -> StatefulLedgerResult:
    days, names = close.shape
    dates = tuple(date(2024, 1, 2) + timedelta(days=index) for index in range(days))
    mask = np.ones((days, names), dtype=np.bool_)
    no_actions = AlignedActionTerms(
        shares_per_prior_share=np.ones_like(close),
        cash_per_prior_share=np.zeros_like(close),
        session_resolved=mask.copy(),
        has_action=np.zeros_like(mask),
        successor_index=np.broadcast_to(
            np.arange(names, dtype=np.int64), close.shape
        ).copy(),
    )
    return simulate_stateful_ledger(
        dates=dates,
        scores=scores,
        score_mask=mask,
        active=mask if active is None else active,
        raw_close=close,
        cdi_returns=np.zeros(days) if cdi is None else cdi,
        fill_fraction=fill_fraction,
        action_terms=no_actions if actions is None else actions,
        action_payment_session=(
            np.full(close.shape, -1, dtype=np.int64)
            if payment_session is None
            else payment_session
        ),
        security_ids=tuple(f"SEC-{index}" for index in range(names)),
        initial_reference_price=initial_reference_price,
        config=_config() if config is None else config,
    )


def _action_terms(
    shape: tuple[int, int],
    *,
    day: int,
    q: float = 1.0,
    d: float = 0.0,
) -> AlignedActionTerms:
    share_factor = np.ones(shape, dtype=np.float64)
    cash = np.zeros(shape, dtype=np.float64)
    has_action = np.zeros(shape, dtype=np.bool_)
    share_factor[day] = q
    cash[day] = d
    has_action[day] = True
    return AlignedActionTerms(
        shares_per_prior_share=share_factor,
        cash_per_prior_share=cash,
        session_resolved=np.ones(shape, dtype=np.bool_),
        has_action=has_action,
        successor_index=np.broadcast_to(
            np.arange(shape[1], dtype=np.int64), shape
        ).copy(),
    )


def test_intended_orders_use_prior_marks_and_ignore_current_future_print() -> None:
    close = np.full((4, 3), 100.0)
    scores = np.asarray([[3.0, 0.0, -3.0]] * 4)
    missing = close.copy()
    missing[1, 0] = np.nan
    changed = close.copy()
    changed[1, 0] = 250.0

    first = _run(missing, scores)
    second = _run(changed, scores)
    first_orders = [
        order for order in first.intended_orders if order.decision_session == 1
    ]
    second_orders = [
        order for order in second.intended_orders if order.decision_session == 1
    ]
    assert [
        (order.security_index, order.side, order.quantity, order.reference_price)
        for order in first_orders
    ] == [
        (order.security_index, order.side, order.quantity, order.reference_price)
        for order in second_orders
    ]
    assert first_orders[0].reference_price == 100.0
    assert first_orders[0].quantity == 0.01
    assert not any(
        fill.order_id == first_orders[0].order_id and fill.fill_session == 1
        for fill in first.fills
    )
    assert any(
        fill.order_id == second_orders[0].order_id and fill.price == 250.0
        for fill in second.fills
    )


def test_first_evaluation_decision_uses_explicit_pre_window_reference() -> None:
    close = np.full((3, 3), 100.0)
    scores = np.asarray([[3.0, 0.0, -3.0]] * 3)
    changed = close.copy()
    changed[0, 0] = 250.0
    initial = np.asarray([80.0, 90.0, 120.0])
    first = _run(close, scores, initial_reference_price=initial)
    second = _run(changed, scores, initial_reference_price=initial)
    first_orders = [
        order for order in first.intended_orders if order.decision_session == 0
    ]
    second_orders = [
        order for order in second.intended_orders if order.decision_session == 0
    ]
    assert [
        (order.security_index, order.side, order.quantity, order.reference_price)
        for order in first_orders
    ] == [
        (order.security_index, order.side, order.quantity, order.reference_price)
        for order in second_orders
    ]
    assert [order.reference_price for order in first_orders] == [80.0, 120.0]


def test_action_reconciliation_mask_never_changes_intended_order() -> None:
    close = np.full((3, 2), 100.0)
    scores = np.asarray([[1.0, -1.0]] * 3)
    resolved = np.ones_like(close, dtype=np.bool_)
    unresolved = resolved.copy()
    unresolved[0, 0] = False
    base = _action_terms(close.shape, day=1)
    changed = AlignedActionTerms(
        shares_per_prior_share=base.shares_per_prior_share,
        cash_per_prior_share=base.cash_per_prior_share,
        session_resolved=unresolved,
        has_action=np.zeros_like(unresolved),
        successor_index=base.successor_index,
    )
    initial = np.asarray([100.0, 100.0])
    first = _run(close, scores, actions=base, initial_reference_price=initial)
    second = _run(close, scores, actions=changed, initial_reference_price=initial)
    first_orders = [
        order for order in first.intended_orders if order.decision_session == 0
    ]
    second_orders = [
        order for order in second.intended_orders if order.decision_session == 0
    ]
    assert first_orders == second_orders


def test_unfilled_top_entry_reserves_slot_and_is_not_replaced() -> None:
    close = np.full((5, 3), 100.0)
    close[1:4, 0] = np.nan
    scores = np.asarray(
        [
            [3.0, 0.0, -3.0],
            [3.0, 0.0, -3.0],
            [0.0, 3.0, -3.0],
            [0.0, 3.0, -3.0],
            [0.0, 3.0, -3.0],
        ]
    )
    result = _run(close, scores)

    entries = [order for order in result.intended_orders if order.purpose == "entry"]
    assert {(order.decision_session, order.security_index) for order in entries} == {
        (1, 0),
        (1, 2),
    }
    assert not any(order.security_index == 1 for order in entries)
    assert result.pending_entry_count[1] == 1
    assert result.cancelled_entry_count[3] == 1
    assert result.cancellations[0].reason == "expired"


def test_partial_entry_remains_pending_without_allocating_another_slot() -> None:
    close = np.full((5, 3), 100.0)
    scores = np.asarray([[3.0, 0.0, -3.0]] * 5)
    fill_fraction = np.ones_like(close)
    fill_fraction[1, 0] = 0.5
    result = _run(close, scores, fill_fraction=fill_fraction)

    long_entry = next(
        order
        for order in result.intended_orders
        if order.purpose == "entry" and order.side == "buy"
    )
    long_fills = [fill for fill in result.fills if fill.order_id == long_entry.order_id]
    assert [fill.fill_session for fill in long_fills] == [1, 2]
    np.testing.assert_allclose(sum(fill.quantity for fill in long_fills), 0.01)
    assert (
        len([order for order in result.intended_orders if order.purpose == "entry"])
        == 2
    )


def test_eligibility_loss_creates_pending_exit_until_first_later_print() -> None:
    close = np.full((5, 2), 100.0)
    close[2, 0] = np.nan
    close[3, 0] = 80.0
    scores = np.asarray([[1.0, -1.0]] * 5)
    active = np.ones_like(close, dtype=np.bool_)
    active[2:, 0] = False
    result = _run(close, scores, active=active)

    exit_order = next(
        order
        for order in result.intended_orders
        if order.security_index == 0 and order.purpose == "exit"
    )
    assert exit_order.decision_session == 2
    assert result.position_sign[2, 0] == 1
    assert result.pending_exit_count[2] == 1
    exit_fill = next(
        fill for fill in result.fills if fill.order_id == exit_order.order_id
    )
    assert exit_fill.fill_session == 3
    assert exit_fill.price == 80.0
    assert result.position_sign[3, 0] == 0


def test_t18_raw_share_cash_claim_and_fill_path_reconciles() -> None:
    close = np.asarray(
        [
            [100.0, 100.0],
            [100.0, 100.0],
            [45.0, 45.0],
            [45.0, 45.0],
        ]
    )
    scores = np.asarray([[1.0, -1.0]] * 4)
    actions = _action_terms(close.shape, day=2, q=2.0, d=10.0)
    payment_session = np.full(close.shape, -1, dtype=np.int64)
    payment_session[2] = 3
    result = _run(
        close,
        scores,
        actions=actions,
        payment_session=payment_session,
        config=_config(cost_bps_per_side=10.0),
    )

    np.testing.assert_allclose(result.signed_shares[1], [0.01, -0.01])
    np.testing.assert_allclose(result.signed_shares[2], [0.02, -0.02])
    assert result.receivables[2] == 0.1
    assert result.payables[2] == 0.1
    assert result.receivables[3] == 0.0
    assert result.payables[3] == 0.0
    np.testing.assert_allclose(
        result.nav,
        result.free_cash
        + result.restricted_cash
        + result.marked_signed_holdings
        + result.receivables
        - result.payables,
    )
    np.testing.assert_allclose(result.reconciliation_error, 0.0, atol=1e-15)
    assert len(result.fills) == 4
    assert np.isclose(sum(fill.cost for fill in result.fills), 0.0038)
    assert np.isclose(result.nav[-1], 0.9962)


def test_cash_action_without_ex_date_print_preserves_prior_equity() -> None:
    close = np.full((4, 2), 100.0)
    close[2, 0] = np.nan
    scores = np.asarray([[1.0, -1.0]] * 4)
    q = np.ones_like(close)
    d = np.zeros_like(close)
    has_action = np.zeros_like(close, dtype=np.bool_)
    d[2, 0] = 10.0
    has_action[2, 0] = True
    actions = AlignedActionTerms(
        shares_per_prior_share=q,
        cash_per_prior_share=d,
        session_resolved=np.ones_like(has_action),
        has_action=has_action,
        successor_index=np.broadcast_to(
            np.arange(close.shape[1], dtype=np.int64), close.shape
        ).copy(),
    )
    result = _run(close, scores, actions=actions)
    assert result.signed_shares[2, 0] == 0.01
    assert result.mark_price[2, 0] == 90.0
    assert result.receivables[2] == 0.1
    assert result.nav[2] == 1.0


def test_corporate_action_cancels_pending_entries_before_unit_change() -> None:
    close = np.full((5, 3), 100.0)
    close[1, [0, 2]] = np.nan
    close[2, [0, 2]] = 50.0
    scores = np.asarray([[3.0, 0.0, -3.0]] * 5)
    q = np.ones_like(close)
    has_action = np.zeros_like(close, dtype=np.bool_)
    q[2, [0, 2]] = 2.0
    has_action[2, [0, 2]] = True
    actions = AlignedActionTerms(
        shares_per_prior_share=q,
        cash_per_prior_share=np.zeros_like(close),
        session_resolved=np.ones_like(has_action),
        has_action=has_action,
        successor_index=np.broadcast_to(
            np.arange(close.shape[1], dtype=np.int64), close.shape
        ).copy(),
    )
    result = _run(close, scores, actions=actions)
    original = [
        order
        for order in result.intended_orders
        if order.decision_session == 1 and order.purpose == "entry"
    ]
    replacement = [
        order
        for order in result.intended_orders
        if order.decision_session == 2 and order.purpose == "entry"
    ]
    assert len(original) == len(replacement) == 2
    assert {
        cancellation.order_id
        for cancellation in result.cancellations
        if cancellation.reason == "corporate_action"
    } == {order.order_id for order in original}
    assert not any(
        fill.order_id in {order.order_id for order in original} for fill in result.fills
    )
    assert {order.reference_price for order in replacement} == {50.0}
    assert {order.quantity for order in replacement} == {0.02}


def test_corporate_action_cancels_and_reissues_pending_exit_in_new_units() -> None:
    close = np.full((4, 3), 100.0)
    close[1, 0] = np.nan
    close[2:, 0] = 50.0
    scores = np.asarray([[3.0, 0.0, -3.0]] * 4)
    active = np.ones_like(close, dtype=np.bool_)
    active[1:, 0] = False
    q = np.ones_like(close)
    has_action = np.zeros_like(close, dtype=np.bool_)
    q[2, 0] = 2.0
    has_action[2, 0] = True
    actions = AlignedActionTerms(
        shares_per_prior_share=q,
        cash_per_prior_share=np.zeros_like(close),
        session_resolved=np.ones_like(has_action),
        has_action=has_action,
        successor_index=np.broadcast_to(
            np.arange(close.shape[1], dtype=np.int64), close.shape
        ).copy(),
    )
    result = _run(
        close,
        scores,
        active=active,
        actions=actions,
        initial_reference_price=np.full(3, 100.0),
    )

    original = next(
        order
        for order in result.intended_orders
        if order.decision_session == 1
        and order.security_index == 0
        and order.purpose == "exit"
    )
    replacement = next(
        order
        for order in result.intended_orders
        if order.decision_session == 2
        and order.security_index == 0
        and order.purpose == "exit"
    )
    cancellation = next(
        row for row in result.cancellations if row.order_id == original.order_id
    )
    assert original.quantity == 0.01
    assert original.reference_price == 100.0
    assert cancellation.reason == "corporate_action"
    assert cancellation.unfilled_quantity == 0.01
    assert replacement.quantity == 0.02
    assert replacement.reference_price == 50.0
    assert any(fill.order_id == replacement.order_id for fill in result.fills)
    assert not any(fill.order_id == original.order_id for fill in result.fills)


def test_no_inventory_conversion_transfers_causal_reference_to_successor() -> None:
    close = np.asarray(
        [
            [np.nan, 45.0, 100.0],
            [np.nan, 45.0, 100.0],
            [np.nan, 45.0, 100.0],
        ]
    )
    scores = np.asarray([[0.0, 3.0, -3.0]] * 3)
    q = np.ones_like(close)
    d = np.zeros_like(close)
    has_action = np.zeros_like(close, dtype=np.bool_)
    successor = np.broadcast_to(
        np.arange(close.shape[1], dtype=np.int64), close.shape
    ).copy()
    q[0, 0] = 2.0
    d[0, 0] = 10.0
    has_action[0, 0] = True
    successor[0, 0] = 1
    actions = AlignedActionTerms(
        shares_per_prior_share=q,
        cash_per_prior_share=d,
        session_resolved=np.ones_like(has_action),
        has_action=has_action,
        successor_index=successor,
    )
    result = _run(
        close,
        scores,
        actions=actions,
        initial_reference_price=np.asarray([100.0, np.nan, 100.0]),
    )

    successor_order = next(
        order
        for order in result.intended_orders
        if order.decision_session == 0 and order.security_index == 1
    )
    assert successor_order.reference_price == 45.0
    np.testing.assert_allclose(successor_order.quantity, 1.0 / 45.0)


def test_verified_share_conversion_moves_inventory_to_successor_once() -> None:
    close = np.asarray(
        [
            [100.0, np.nan, 100.0],
            [100.0, np.nan, 100.0],
            [np.nan, 50.0, 100.0],
            [np.nan, 50.0, 100.0],
        ]
    )
    scores = np.asarray(
        [
            [3.0, 0.0, -3.0],
            [3.0, 0.0, -3.0],
            [0.0, 3.0, -3.0],
            [0.0, 3.0, -3.0],
        ]
    )
    active = np.ones_like(close, dtype=np.bool_)
    active[2:, 0] = False
    q = np.ones_like(close)
    has_action = np.zeros_like(close, dtype=np.bool_)
    successor = np.broadcast_to(
        np.arange(close.shape[1], dtype=np.int64), close.shape
    ).copy()
    q[2, 0] = 2.0
    has_action[2, 0] = True
    successor[2, 0] = 1
    actions = AlignedActionTerms(
        shares_per_prior_share=q,
        cash_per_prior_share=np.zeros_like(close),
        session_resolved=np.ones_like(has_action),
        has_action=has_action,
        successor_index=successor,
    )
    result = _run(close, scores, active=active, actions=actions)

    np.testing.assert_allclose(result.signed_shares[1], [0.01, 0.0, -0.01])
    np.testing.assert_allclose(result.signed_shares[2], [0.0, 0.02, -0.01])
    assert not any(
        order.security_index == 1 and order.purpose == "entry"
        for order in result.intended_orders
    )
    np.testing.assert_allclose(result.reconciliation_error, 0.0, atol=1e-15)


def test_verified_cash_settlement_needs_no_trade_print_and_pays_once() -> None:
    close = np.full((4, 2), 100.0)
    close[2:, 0] = np.nan
    scores = np.asarray([[1.0, -1.0]] * 4)
    active = np.ones_like(close, dtype=np.bool_)
    active[2:, 0] = False
    q = np.ones_like(close)
    d = np.zeros_like(close)
    has_action = np.zeros_like(close, dtype=np.bool_)
    q[2, 0] = 0.0
    d[2, 0] = 80.0
    has_action[2, 0] = True
    actions = AlignedActionTerms(
        shares_per_prior_share=q,
        cash_per_prior_share=d,
        session_resolved=np.ones_like(has_action),
        has_action=has_action,
        successor_index=np.broadcast_to(
            np.arange(close.shape[1], dtype=np.int64), close.shape
        ).copy(),
    )
    payment_session = np.full(close.shape, -1, dtype=np.int64)
    payment_session[2, 0] = 3
    result = _run(
        close,
        scores,
        active=active,
        actions=actions,
        payment_session=payment_session,
    )

    assert result.signed_shares[2, 0] == 0.0
    assert result.receivables[2] == 0.8
    assert result.receivables[3] == 0.0
    assert np.isclose(result.free_cash[3], 0.8)
    assert np.isclose(result.nav[2], 0.8)
    assert np.isclose(result.nav[3], 0.8)
    assert result.unresolved_inventory_count == 0
    np.testing.assert_allclose(result.reconciliation_error, 0.0, atol=1e-15)


def test_payment_settles_only_its_claim_and_unknown_payment_stays_unresolved() -> None:
    close = np.full((5, 2), 100.0)
    scores = np.asarray([[1.0, -1.0]] * 5)
    q = np.ones_like(close)
    d = np.zeros_like(close)
    has_action = np.zeros_like(close, dtype=np.bool_)
    d[2, 0] = 10.0
    d[3, 0] = 20.0
    has_action[2:4, 0] = True
    actions = AlignedActionTerms(
        shares_per_prior_share=q,
        cash_per_prior_share=d,
        session_resolved=np.ones_like(has_action),
        has_action=has_action,
        successor_index=np.broadcast_to(
            np.arange(close.shape[1], dtype=np.int64), close.shape
        ).copy(),
    )
    payment_session = np.full(close.shape, -1, dtype=np.int64)
    payment_session[2, 0] = 3
    result = _run(
        close,
        scores,
        actions=actions,
        payment_session=payment_session,
    )

    np.testing.assert_allclose(result.receivables[2:], [0.1, 0.2, 0.2])
    assert np.isclose(result.unresolved_receivable, 0.2)
    assert result.economics_unresolved
    np.testing.assert_allclose(result.reconciliation_error, 0.0, atol=1e-15)


def test_terminal_missing_inventory_is_not_sold_and_has_scenario_views() -> None:
    close = np.full((4, 2), 100.0)
    close[-1, 0] = np.nan
    scores = np.asarray([[1.0, -1.0]] * 4)
    result = _run(
        close,
        scores,
        config=_config(forced_liquidation_haircut=0.30),
    )

    long_terminal_exit = next(
        order
        for order in result.intended_orders
        if order.security_index == 0 and order.purpose == "terminal_exit"
    )
    assert not any(
        fill.order_id == long_terminal_exit.order_id for fill in result.fills
    )
    assert result.unresolved_inventory_count == 1
    assert result.unresolved_inventory_notional == 1.0
    assert result.signed_shares[-1, 0] == 0.01
    assert result.free_cash[-1] == 0.0
    assert result.nav[-1] == 1.0
    assert result.haircut_scenario_nav[-1] == 0.7
    assert result.unresolved_excluded_nav[-1] == 0.0
    assert result.valuation_scenario_count.sum() == 1

    short_missing = np.full((4, 2), 100.0)
    short_missing[-1, 1] = np.nan
    short_result = _run(
        short_missing,
        scores,
        config=_config(forced_liquidation_haircut=0.30),
    )
    short_terminal_exit = next(
        order
        for order in short_result.intended_orders
        if order.security_index == 1 and order.purpose == "terminal_exit"
    )
    assert not any(
        fill.order_id == short_terminal_exit.order_id for fill in short_result.fills
    )
    assert short_result.signed_shares[-1, 1] == -0.01
    assert short_result.free_cash[-1] == 1.0
    assert short_result.restricted_cash[-1] == 1.0
    assert short_result.nav[-1] == 1.0
    assert short_result.haircut_scenario_nav[-1] == 0.7
    assert short_result.unresolved_excluded_nav[-1] == 1.0


def test_unresolved_action_coverage_blocks_fill_and_keeps_inventory() -> None:
    close = np.full((4, 2), 100.0)
    scores = np.asarray([[1.0, -1.0]] * 4)
    resolved = np.ones_like(close, dtype=np.bool_)
    resolved[2:, 0] = False
    actions = AlignedActionTerms(
        shares_per_prior_share=np.ones_like(close),
        cash_per_prior_share=np.zeros_like(close),
        session_resolved=resolved,
        has_action=np.zeros_like(resolved),
        successor_index=np.broadcast_to(
            np.arange(close.shape[1], dtype=np.int64), close.shape
        ).copy(),
    )
    result = _run(close, scores, actions=actions)

    exit_order = next(
        order
        for order in result.intended_orders
        if order.security_index == 0 and order.purpose == "exit"
    )
    assert not any(fill.order_id == exit_order.order_id for fill in result.fills)
    assert result.pending_exit_count[-1] == 1
    assert result.unresolved_action_name_days.sum() == 2
    assert result.unresolved_inventory_count == 1
    assert result.signed_shares[-1, 0] == 0.01
    assert result.economics_unresolved


def test_insolvency_stops_path_and_future_order_generation() -> None:
    close = np.asarray(
        [
            [100.0, 100.0],
            [100.0, 100.0],
            [100.0, 250.0],
            [100.0, 1.0],
            [100.0, 1.0],
        ]
    )
    scores = np.asarray([[1.0, -1.0]] * len(close))
    result = _run(close, scores)

    assert result.insolvent
    assert result.insolvency_date == date(2024, 1, 4)
    assert len(result.dates) == 3
    assert result.nav[-1] == -0.5
    assert max(order.decision_session for order in result.intended_orders) == 1
    assert result.economics_unresolved


def test_small_universe_empty_book_and_compounded_cash_guard() -> None:
    close = np.full((4, 1), 100.0)
    scores = np.ones_like(close)
    cdi = np.asarray([0.001, 0.002, 0.003, 0.004])
    result = _run(close, scores, cdi=cdi)

    assert not result.intended_orders
    assert not result.fills
    assert result.entry_blocked_small_universe.all()
    np.testing.assert_array_equal(result.retention_width, 0)
    np.testing.assert_allclose(result.gross_fraction_nav, 0.0)
    assert result.economics_unresolved
    assert np.isclose(result.summary()["compounded_net_excess_vs_all_cash"], 0.0)
    assert np.isfinite(result.nav).all()


def test_retention_width_cannot_overlap_rank_bands() -> None:
    close = np.full((4, 6), 100.0)
    scores = np.broadcast_to(np.arange(6, dtype=np.float64), close.shape).copy()
    result = _run(
        close,
        scores,
        config=LedgerConfig(
            k_per_side=2,
            buffer_per_side=4,
            cost_bps_per_side=0.0,
            annual_borrow_rate=0.0,
            # Single-name admission is intentional: a 50% slot must pass the
            # net cap before its opposite-side order exists.
            planned_absolute_net_cap=0.50,
            planned_name_weight_cap=0.60,
        ),
    )
    np.testing.assert_array_equal(result.retention_width, 3)
    day_one_entries = [
        order
        for order in result.intended_orders
        if order.decision_session == 1 and order.purpose == "entry"
    ]
    assert {
        order.security_index for order in day_one_entries if order.side == "buy"
    } == {
        4,
        5,
    }
    assert {
        order.security_index for order in day_one_entries if order.side == "sell"
    } == {
        0,
        1,
    }


def test_risk_caps_are_bound_to_prior_positive_nav() -> None:
    close = np.full((4, 4), 100.0)
    scores = np.broadcast_to(np.arange(4, dtype=np.float64), close.shape).copy()
    result = _run(
        close,
        scores,
        config=LedgerConfig(
            k_per_side=2,
            buffer_per_side=0,
            gross_target=2.0,
            planned_gross_cap=2.25,
            # The independent-entry contract checks each 50% slot rather than
            # relying on a paired-order exception.
            planned_absolute_net_cap=0.50,
            planned_name_weight_cap=0.50,
            cost_bps_per_side=0.0,
            annual_borrow_rate=0.0,
        ),
    )
    assert result.planned_gross_fraction_nav[1] == 2.0
    assert result.planned_net_fraction_nav[1] == 0.0
    assert result.planned_name_weight_fraction_nav[1] == 0.5
    assert not result.actual_risk_breach[1]


def test_sixty_slot_book_refills_while_three_names_per_side_rotate() -> None:
    days, names = 120, 250
    generator = np.random.default_rng(57)
    close = 100.0 * np.exp(
        np.cumsum(generator.uniform(0.0, 1e-5, size=(days, names)), axis=0)
    )
    base = np.arange(names, dtype=np.float64)
    scores = np.stack(
        [base if day < 2 else np.roll(base, 3 * (day - 1)) for day in range(days)]
    )
    result = _run(
        close,
        scores,
        config=LedgerConfig(
            k_per_side=30,
            buffer_per_side=0,
            cost_bps_per_side=0.0,
            annual_borrow_rate=0.0,
        ),
        initial_reference_price=np.full(names, 100.0),
    )

    assert np.sum(result.position_sign[1] > 0) == 30
    assert np.sum(result.position_sign[1] < 0) == 30
    assert np.all(result.submitted_entry_count[3:-1] == 6)
    assert np.all(result.gross_fraction_nav[1:-1] >= 1.8 - 1e-12)
    assert np.all(result.gross_fraction_nav[1:-1] <= 2.2 + 1e-12)


def test_asymmetric_exit_fills_trim_only_the_heavy_side() -> None:
    days, names = 6, 250
    base = np.arange(names, dtype=np.float64)
    scores = np.tile(base, (days, 1))
    for day in range(1, days):
        scores[day, 246:250] = 100.0 + np.arange(4)
        scores[day, 100:104] = 300.0 + np.arange(4)
        scores[day, 0] = 120.0
        scores[day, 104] = -10.0
    result = _run(
        np.full((days, names), 100.0),
        scores,
        config=LedgerConfig(
            k_per_side=30,
            buffer_per_side=0,
            planned_absolute_net_cap=0.08,
            cost_bps_per_side=0.0,
            annual_borrow_rate=0.0,
        ),
        initial_reference_price=np.full(names, 100.0),
    )

    risk_orders = [
        order
        for order in result.intended_orders
        if order.decision_session == 2 and order.purpose == "risk_exit"
    ]
    assert risk_orders
    assert all(order.side == "buy" for order in risk_orders)
    assert len(risk_orders) < 29
    assert result.risk_trim_net_notional[2] > 0.0
    assert result.risk_trim_gross_notional[2] == 0.0
    assert result.gross_fraction_nav[2] > 1.8


def test_adverse_fifteen_percent_rally_uses_proportional_gross_trims() -> None:
    days, names = 5, 100
    scores = np.tile(np.arange(names, dtype=np.float64), (days, 1))
    close = np.full((days, names), 100.0)
    close[1:, :30] = 115.0
    result = _run(
        close,
        scores,
        config=LedgerConfig(
            k_per_side=30,
            buffer_per_side=30,
            cost_bps_per_side=0.0,
            annual_borrow_rate=0.0,
        ),
        initial_reference_price=np.full(names, 100.0),
    )

    risk_orders = [
        order
        for order in result.intended_orders
        if order.decision_session == 2 and order.purpose == "risk_exit"
    ]
    long_trim = sum(
        order.quantity * order.reference_price
        for order in risk_orders
        if order.side == "sell"
    )
    short_trim = sum(
        order.quantity * order.reference_price
        for order in risk_orders
        if order.side == "buy"
    )
    assert result.gross_fraction_nav[1] > 2.25
    assert result.gross_fraction_nav[2] <= 2.0 + 1e-12
    np.testing.assert_allclose(long_trim, 0.15, atol=1e-12)
    np.testing.assert_allclose(short_trim, 0.30, atol=1e-12)


def test_missing_reference_skips_candidate_and_admits_the_next_name() -> None:
    names = 100
    scores = np.tile(np.arange(names, dtype=np.float64), (4, 1))
    references = np.full(names, 100.0)
    references[99] = np.nan
    result = _run(
        np.full((4, names), 100.0),
        scores,
        config=LedgerConfig(
            k_per_side=30,
            buffer_per_side=30,
            cost_bps_per_side=0.0,
            annual_borrow_rate=0.0,
        ),
        initial_reference_price=references,
    )

    day_zero_buys = {
        order.security_index
        for order in result.intended_orders
        if order.decision_session == 0
        and order.purpose == "entry"
        and order.side == "buy"
    }
    assert 99 not in day_zero_buys
    assert 98 in day_zero_buys
    assert result.blocked_entry_no_reference_count[0] == 1


def test_full_long_side_does_not_block_empty_short_side() -> None:
    names = 100
    scores = np.tile(np.arange(names, dtype=np.float64), (4, 1))
    references = np.full(names, 100.0)
    references[:30] = np.nan
    result = _run(
        np.full((4, names), 100.0),
        scores,
        config=LedgerConfig(
            k_per_side=30,
            buffer_per_side=30,
            planned_absolute_net_cap=1.10,
            cost_bps_per_side=0.0,
            annual_borrow_rate=0.0,
        ),
        initial_reference_price=references,
    )

    assert np.sum(result.position_sign[0] > 0) == 30
    assert np.sum(result.position_sign[0] < 0) == 0
    assert np.sum(result.position_sign[1] > 0) == 30
    assert np.sum(result.position_sign[1] < 0) == 30
    assert all(
        order.side == "sell"
        for order in result.intended_orders
        if order.decision_session == 1 and order.purpose == "entry"
    )
