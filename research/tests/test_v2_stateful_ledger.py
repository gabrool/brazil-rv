from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest

from brazil_rv.execution.stateful_ledger import (
    LedgerConfig,
    StatefulLedgerResult,
    _scaled_group_bands,
    equity_borrow_registration_fee,
    ledger_configurations,
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
            borrow_source="uniform",
            borrow_registration_fee_fraction=0.0,
            borrow_registration_fee_floor=0.0,
            borrow_registration_fee_cap=0.0,
            volatility_balanced_entries=False,
            beta_hedge=False,
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
    score_mask: np.ndarray | None = None,
    fill_fraction: np.ndarray | None = None,
    actions: AlignedActionTerms | None = None,
    payment_session: np.ndarray | None = None,
    cdi: np.ndarray | None = None,
    initial_reference_price: np.ndarray | None = None,
    annual_borrow_rate_by_name: np.ndarray | None = None,
    borrow_rate_imputed: np.ndarray | None = None,
    borrow_rate_placeholder: np.ndarray | None = None,
    shortable: np.ndarray | None = None,
    selection_volatility: np.ndarray | None = None,
    hedge_beta: np.ndarray | None = None,
    hedge_beta_valid: np.ndarray | None = None,
    initial_hedge_reference_price: float = 100.0,
    hedge_close: np.ndarray | None = None,
    hedge_annual_borrow_rate: np.ndarray | None = None,
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
        score_mask=mask if score_mask is None else score_mask,
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
        annual_borrow_rate_by_name=annual_borrow_rate_by_name,
        borrow_rate_imputed=borrow_rate_imputed,
        borrow_rate_placeholder=(
            np.zeros_like(close, dtype=np.bool_)
            if borrow_rate_placeholder is None
            and config is not None
            and config.borrow_source != "uniform"
            else borrow_rate_placeholder
        ),
        shortable=shortable,
        selection_volatility=selection_volatility,
        hedge_beta=hedge_beta,
        hedge_beta_valid=(
            np.isfinite(hedge_beta)
            if hedge_beta is not None and hedge_beta_valid is None
            else hedge_beta_valid
        ),
        initial_hedge_reference_price=initial_hedge_reference_price,
        hedge_close=hedge_close,
        hedge_annual_borrow_rate=hedge_annual_borrow_rate,
        config=_config() if config is None else config,
    )


def _constructed_inputs(days: int, names: int) -> dict[str, np.ndarray]:
    return {
        "annual_borrow_rate_by_name": np.full((days, names), 0.02),
        "borrow_rate_imputed": np.zeros((days, names), dtype=np.bool_),
        "borrow_rate_placeholder": np.zeros((days, names), dtype=np.bool_),
        "shortable": np.ones((days, names), dtype=np.bool_),
        "selection_volatility": np.broadcast_to(
            np.arange(names, dtype=np.float64), (days, names)
        ).copy(),
        "hedge_beta": np.ones((days, names), dtype=np.float64),
        "hedge_close": np.full(days, 100.0),
    }


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
        (
            order.security_index,
            order.side,
            order.planned_notional,
            order.position_fraction,
            order.reference_price,
        )
        for order in first_orders
    ] == [
        (
            order.security_index,
            order.side,
            order.planned_notional,
            order.position_fraction,
            order.reference_price,
        )
        for order in second_orders
    ]
    assert first_orders[0].reference_price == 100.0
    assert first_orders[0].planned_notional == 1.0
    assert not any(
        fill.order_id == first_orders[0].order_id and fill.fill_session == 1
        for fill in first.fills
    )
    assert any(
        fill.order_id == second_orders[0].order_id and fill.price == 250.0
        for fill in second.fills
    )


def test_lending_borrow_charges_observed_rate_plus_registered_capped_fee() -> None:
    close = np.full((5, 3), 100.0)
    scores = np.asarray([[3.0, 0.0, -3.0]] * 5)
    initial = np.full(3, 100.0)
    uniform = _run(
        close,
        scores,
        config=_config(
            annual_borrow_rate=0.02,
            borrow_registration_fee_fraction=0.20,
            borrow_registration_fee_floor=0.00025,
            borrow_registration_fee_cap=0.007,
        ),
        initial_reference_price=initial,
    )
    lending = _run(
        close,
        scores,
        config=_config(
            annual_borrow_rate=0.02,
            borrow_source="borrow_balance",
            borrow_registration_fee_fraction=0.20,
            borrow_registration_fee_floor=0.00025,
            borrow_registration_fee_cap=0.007,
        ),
        initial_reference_price=initial,
        annual_borrow_rate_by_name=np.full_like(close, 0.40),
        borrow_rate_imputed=np.zeros_like(close, dtype=np.bool_),
        borrow_rate_placeholder=np.zeros_like(close, dtype=np.bool_),
        shortable=np.ones_like(close, dtype=np.bool_),
    )
    charged = uniform.borrow_bps > 0.0
    assert charged.any()
    first_charged = int(np.flatnonzero(charged)[0])
    assert lending.borrow_bps[first_charged] == pytest.approx(
        ((1.40 ** (1 / 252) - 1) + (1.007 ** (1 / 252) - 1)) * 10_000
    )
    assert uniform.borrow_bps[first_charged] == pytest.approx(
        ((1.02 ** (1 / 252) - 1) + (1.004 ** (1 / 252) - 1)) * 10_000
    )
    np.testing.assert_allclose(
        lending.held_short_weighted_annual_borrow_rate[charged], 0.407
    )


def test_equity_borrow_registration_fee_schedule_floors_and_caps() -> None:
    config = LedgerConfig()
    np.testing.assert_array_equal(
        equity_borrow_registration_fee(np.asarray([0.0001, 0.01, 0.10]), config=config),
        np.asarray([0.00025, 0.002, 0.007]),
    )


def test_full_short_proceeds_interest_equals_cdi_times_restricted_base() -> None:
    close = np.full((5, 3), 100.0)
    scores = np.asarray([[3.0, 0.0, -3.0]] * 5)
    cdi = np.asarray([0.001, 0.002, 0.003, 0.004, 0.005])
    result = _run(
        close,
        scores,
        cdi=cdi,
        config=_config(short_proceeds_remuneration=1.0),
        initial_reference_price=np.full(3, 100.0),
    )
    expected = 10_000.0 * cdi * result.short_proceeds_interest_base / result.start_nav
    np.testing.assert_allclose(
        result.short_proceeds_interest_bps, expected, atol=1e-12, rtol=0.0
    )
    np.testing.assert_allclose(
        result.interest_bps,
        result.free_cash_interest_bps + result.short_proceeds_interest_bps,
        atol=1e-12,
        rtol=0.0,
    )


def test_sterile_comparator_removes_only_short_proceeds_interest_setting() -> None:
    cells = ledger_configurations()
    headline = cells["borrow_balance"]
    sterile = cells["comparator_sterile_proceeds"]
    assert sterile == replace(headline, short_proceeds_remuneration=0.0)


def test_lending_unshortable_entry_advances_to_next_candidate() -> None:
    close = np.full((4, 4), 100.0)
    scores = np.asarray([[4.0, 2.0, -4.0, -2.0]] * 4)
    shortable = np.ones_like(close, dtype=np.bool_)
    shortable[:, 2] = False
    result = _run(
        close,
        scores,
        config=replace(
            _config(borrow_source="borrow_balance"),
            volatility_balanced_entries=False,
            beta_hedge=False,
        ),
        initial_reference_price=np.full(4, 100.0),
        annual_borrow_rate_by_name=np.full_like(close, 0.02),
        borrow_rate_imputed=np.zeros_like(close, dtype=np.bool_),
        borrow_rate_placeholder=np.zeros_like(close, dtype=np.bool_),
        shortable=shortable,
    )
    first_sells = [
        order
        for order in result.intended_orders
        if order.decision_session == 0 and order.side == "sell"
    ]
    assert [order.security_index for order in first_sells] == [3]
    assert result.excluded_short_entry_candidate_count[0] == 1


def test_explicit_uniform_borrow_source_is_bit_identical_to_default() -> None:
    close = np.full((4, 3), 100.0)
    scores = np.asarray([[3.0, 0.0, -3.0]] * 4)
    initial = np.full(3, 100.0)
    default = _run(close, scores, initial_reference_price=initial)
    explicit = _run(
        close,
        scores,
        config=_config(borrow_source="uniform"),
        initial_reference_price=initial,
        annual_borrow_rate_by_name=np.full_like(close, 0.75),
        shortable=np.zeros_like(close, dtype=np.bool_),
    )
    for field in (
        "nav",
        "daily_net_return",
        "gross_pnl_bps",
        "interest_bps",
        "cost_bps",
        "borrow_bps",
        "signed_shares",
    ):
        np.testing.assert_array_equal(getattr(default, field), getattr(explicit, field))


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
        (
            order.security_index,
            order.side,
            order.planned_notional,
            order.position_fraction,
            order.reference_price,
        )
        for order in first_orders
    ] == [
        (
            order.security_index,
            order.side,
            order.planned_notional,
            order.position_fraction,
            order.reference_price,
        )
        for order in second_orders
    ]
    assert [order.reference_price for order in first_orders] == [80.0, 120.0]


def test_only_prior_unresolved_action_blocks_entries() -> None:
    close = np.full((3, 3), 100.0)
    scores = np.asarray([[3.0, 0.0, -3.0]] * 3)
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
    initial = np.asarray([100.0, 100.0, 100.0])
    first = _run(close, scores, actions=base, initial_reference_price=initial)
    second = _run(close, scores, actions=changed, initial_reference_price=initial)
    first_orders = [
        order for order in first.intended_orders if order.decision_session == 0
    ]
    second_orders = [
        order for order in second.intended_orders if order.decision_session == 0
    ]
    assert {(order.security_index, order.side) for order in first_orders} == {
        (0, "buy"),
        (2, "sell"),
    }
    assert first_orders == second_orders
    # With no pre-window reference, the first entry opportunity is session 1.
    # The unresolved session-0 terms are then known and block opening this name.
    delayed = _run(close, scores, actions=changed)
    assert not any(
        o.security_index == 0 and o.purpose == "entry" for o in delayed.intended_orders
    )


def test_pending_band_cancellation_off_preserves_old_slot_reservation() -> None:
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
    result = _run(close, scores, config=_config(cancel_pending_outside_retention=False))

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


def test_zero_ineligible_hold_reproduces_pass4g_pending_exit_path() -> None:
    close = np.full((5, 2), 100.0)
    close[2, 0] = np.nan
    close[3, 0] = 80.0
    scores = np.asarray([[1.0, -1.0]] * 5)
    active = np.ones_like(close, dtype=np.bool_)
    active[2:, 0] = False
    result = _run(
        close,
        scores,
        active=active,
        config=_config(ineligible_hold_sessions=0),
    )

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
    assert result.exit_instructions_ineligible_hold_exhausted.tolist() == [
        0,
        0,
        1,
        0,
        0,
    ]


def test_held_long_survives_five_ineligible_sessions_and_resets_on_return() -> None:
    close = np.full((8, 3), 100.0)
    scores = np.asarray([[3.0, 0.0, -3.0]] * 8)
    active = np.ones_like(close, dtype=np.bool_)
    active[1:6, 0] = False
    result = _run(
        close,
        scores,
        active=active,
        initial_reference_price=np.full(3, 100.0),
    )

    assert not any(
        order.security_index == 0 and order.purpose == "exit"
        for order in result.intended_orders
    )
    np.testing.assert_array_equal(result.position_sign[:7, 0], 1)
    np.testing.assert_array_equal(result.ineligible_streak[1:6, 0], [1, 2, 3, 4, 5])
    assert result.ineligible_streak[6, 0] == 0
    np.testing.assert_allclose(result.mark_price[:7, 0], 100.0)
    assert result.summary()["ineligible_held_sessions_by_cause"] == {
        "score_valid_false": 0,
        "membership_false": 5,
        "score_nonfinite": 0,
    }


def test_held_short_exits_on_sixth_ineligible_session() -> None:
    close = np.full((9, 3), 100.0)
    scores = np.asarray([[3.0, 0.0, -3.0]] * 9)
    active = np.ones_like(close, dtype=np.bool_)
    active[1:, 2] = False
    result = _run(
        close,
        scores,
        active=active,
        initial_reference_price=np.full(3, 100.0),
    )

    exit_order = next(
        order
        for order in result.intended_orders
        if order.security_index == 2 and order.purpose == "exit"
    )
    assert exit_order.decision_session == 6
    assert result.exit_instruction_cause[6, 2] == 1
    assert result.exit_instructions_ineligible_hold_exhausted[6] == 1
    assert result.ineligible_exit_within_hold_window.sum() == 0
    exit_fill = next(
        fill for fill in result.fills if fill.order_id == exit_order.order_id
    )
    assert exit_fill.fill_session == 6
    assert result.position_sign[6, 2] == 0


def test_ineligible_nonprinter_stays_held_until_unchanged_settlement_path() -> None:
    close = np.full((13, 3), 100.0)
    close[1:11, 0] = np.nan
    scores = np.asarray([[3.0, 0.0, -3.0]] * 13)
    active = np.ones_like(close, dtype=np.bool_)
    active[1:, 0] = False
    result = _run(
        close,
        scores,
        active=active,
        initial_reference_price=np.full(3, 100.0),
    )

    np.testing.assert_array_equal(result.position_sign[:10, 0], 1)
    assert result.position_sign[10, 0] == 0
    assert result.terminal_settlement_count[10] == 1
    assert not any(
        fill.security_index == 0 and fill.purpose == "exit" for fill in result.fills
    )
    assert any(
        fill.security_index == 0 and fill.purpose == "terminal_settlement"
        for fill in result.fills
    )


def test_ineligible_streak_resets_after_rank_exit_and_reentry() -> None:
    close = np.full((6, 5), 100.0)
    scores = np.asarray(
        [
            [5.0, 4.0, 0.0, -4.0, -5.0],
            [0.0, 5.0, 4.0, -4.0, -5.0],
            [5.0, 0.0, 4.0, -4.0, -5.0],
            [5.0, 0.0, 4.0, -4.0, -5.0],
            [5.0, 0.0, 4.0, -4.0, -5.0],
            [5.0, 0.0, 4.0, -4.0, -5.0],
        ]
    )
    active = np.ones_like(close, dtype=np.bool_)
    active[3, 0] = False
    result = _run(
        close,
        scores,
        active=active,
        initial_reference_price=np.full(5, 100.0),
    )

    assert result.position_sign[0, 0] == 1
    assert result.position_sign[1, 0] == 0
    assert result.position_sign[2, 0] == 1
    assert result.ineligible_streak[3, 0] == 1


def test_ineligible_held_session_diagnostic_splits_each_input_cause() -> None:
    close = np.full((3, 5), 100.0)
    scores = np.asarray([[5.0, 4.0, 0.0, -4.0, -5.0]] * 3)
    scores[1, 4] = np.nan
    score_mask = np.ones_like(close, dtype=np.bool_)
    score_mask[1, 0] = False
    result = _run(
        close,
        scores,
        active=np.ones_like(close, dtype=np.bool_),
        score_mask=score_mask,
        initial_reference_price=np.full(5, 100.0),
    )

    assert result.summary()["ineligible_held_sessions_by_cause"] == {
        "score_valid_false": 1,
        "membership_false": 0,
        "score_nonfinite": 1,
    }


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


def test_split_preserves_pending_notional_entries_in_post_action_units() -> None:
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
    assert len(original) == 2
    assert replacement == []
    assert not result.cancellations
    split_fills = [f for f in result.fills if f.fill_session == 2]
    assert {f.order_id for f in split_fills} == {o.order_id for o in original}
    assert {o.planned_notional for o in original} == {1.0}
    assert {o.reference_price for o in original} == {100.0}
    assert {f.quantity for f in split_fills} == {0.02}
    assert {f.gross_notional for f in split_fills} == {1.0}
    np.testing.assert_array_equal(result.signed_shares[2], [0.02, 0.0, -0.02])
    assert result.same_day_action_sessions.sum() == 1


def test_pending_fraction_exit_sells_all_converted_split_shares() -> None:
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
        config=_config(ineligible_hold_sessions=0),
    )

    original = next(
        order
        for order in result.intended_orders
        if order.decision_session == 1
        and order.security_index == 0
        and order.purpose == "exit"
    )
    assert original.position_fraction == 1.0
    assert original.planned_notional is None
    assert original.reference_price == 100.0
    assert not any(
        o.purpose == "exit" and o.decision_session == 2 for o in result.intended_orders
    )
    exit_fill = next(f for f in result.fills if f.order_id == original.order_id)
    assert exit_fill.fill_session == 2
    assert exit_fill.quantity == 0.02
    assert exit_fill.gross_notional == 1.0
    assert result.signed_shares[2, 0] == 0.0


@pytest.mark.parametrize("split_day", [3, 4])
def test_partial_trim_rebases_remaining_fraction_across_split(split_day) -> None:
    close = np.full((5, 4), 100.0)
    close[1:split_day] = 200.0
    fractions = np.ones_like(close)
    fractions[2:split_day] = 0.5
    result = _run(
        close,
        np.tile([-2.0, -1.0, 1.0, 2.0], (5, 1)),
        initial_reference_price=np.full(4, 100.0),
        fill_fraction=fractions,
        actions=_action_terms(close.shape, day=split_day, q=2.0),
        config=_config(
            k_per_side=2,
            buffer_per_side=0,
            planned_name_weight_cap=0.55,
            planned_gross_cap=10.0,
        ),
    )
    first = next(o for o in result.intended_orders if o.purpose == "risk_exit")
    assert first.position_fraction == pytest.approx(0.45)
    np.testing.assert_allclose(np.abs(result.signed_shares[2]), 0.005 * 0.775)
    if split_day == 3:
        # The remaining half of the 45% trim sells 0.225 * original shares,
        # now doubled by the split, leaving exactly 55% of converted inventory.
        np.testing.assert_allclose(np.abs(result.signed_shares[3]), 0.0055)
    else:
        # Final liquidation supersedes the still-partial trim before the split.
        assert any(
            o.purpose == "terminal_exit" and o.position_fraction == 1.0
            for o in result.intended_orders
            if o.decision_session == 4
        )
    np.testing.assert_array_equal(result.signed_shares[-1], 0.0)


def test_successor_reference_becomes_available_only_after_conversion_session() -> None:
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

    assert not any(
        o.decision_session == 0 and o.security_index == 1
        for o in result.intended_orders
    )
    successor_order = next(
        o
        for o in result.intended_orders
        if o.decision_session == 1 and o.security_index == 1
    )
    assert successor_order.reference_price == 45.0
    assert successor_order.planned_notional == 1.0


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


def test_terminal_missing_inventory_inside_grace_is_not_settled() -> None:
    close = np.full((4, 2), 100.0)
    close[-1, 0] = np.nan
    scores = np.asarray([[1.0, -1.0]] * 4)
    result = _run(close, scores)

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
    assert result.settlement_haircut_scenario_nav[-1] == 1.0
    assert result.unresolved_excluded_nav[-1] == 0.0
    assert result.valuation_scenario_count.sum() == 1

    short_missing = np.full((4, 2), 100.0)
    short_missing[-1, 1] = np.nan
    short_result = _run(short_missing, scores)
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
    assert short_result.settlement_haircut_scenario_nav[-1] == 1.0
    assert short_result.unresolved_excluded_nav[-1] == 1.0


def test_long_settles_at_last_mark_on_tenth_missing_session_with_cost() -> None:
    close = np.full((12, 2), 100.0)
    close[1:, 0] = np.nan
    scores = np.asarray([[1.0, -1.0]] * len(close))
    result = _run(
        close,
        scores,
        initial_reference_price=np.full(2, 100.0),
        config=_config(cost_bps_per_side=10.0),
    )

    settlement = next(
        fill
        for fill in result.fills
        if fill.security_index == 0 and fill.purpose == "terminal_settlement"
    )
    assert settlement.fill_session == 10
    assert settlement.side == "sell"
    assert settlement.price == 100.0
    assert np.isclose(settlement.cost, 0.001)
    assert result.signed_shares[10, 0] == 0.0
    assert result.terminal_settlement_count.sum() == 1
    assert np.isclose(result.terminal_settlement_notional.sum(), 1.0)
    assert not any(
        order.security_index == 0
        and order.purpose == "entry"
        and order.decision_session > 10
        for order in result.intended_orders
    )


def test_short_settles_at_last_mark_on_tenth_missing_session() -> None:
    close = np.full((12, 2), 100.0)
    close[1:, 1] = np.nan
    scores = np.asarray([[1.0, -1.0]] * len(close))
    result = _run(
        close,
        scores,
        initial_reference_price=np.full(2, 100.0),
    )

    settlement = next(
        fill
        for fill in result.fills
        if fill.security_index == 1 and fill.purpose == "terminal_settlement"
    )
    assert settlement.fill_session == 10
    assert settlement.side == "buy"
    assert settlement.price == 100.0
    assert result.signed_shares[10, 1] == 0.0
    assert result.restricted_cash[10] == 0.0


def test_terminal_settlement_haircut_scenario_reconciles() -> None:
    close = np.full((12, 2), 100.0)
    close[1:, 0] = np.nan
    scores = np.asarray([[1.0, -1.0]] * len(close))
    result = _run(
        close,
        scores,
        initial_reference_price=np.full(2, 100.0),
    )

    np.testing.assert_allclose(result.reconciliation_error, 0.0, atol=1e-15)
    assert np.isclose(result.nav[10], 1.0)
    assert np.isclose(result.settlement_haircut_scenario_nav[10], 0.7)
    assert np.isclose(
        result.summary()["compounded_net_excess_terminal_settlement_haircut_scenario"],
        -0.3,
    )


def test_print_on_ninth_missing_session_prevents_terminal_settlement() -> None:
    close = np.full((13, 2), 100.0)
    close[1:10, 0] = np.nan
    scores = np.asarray([[1.0, -1.0]] * len(close))
    result = _run(
        close,
        scores,
        initial_reference_price=np.full(2, 100.0),
    )

    assert not any(fill.purpose == "terminal_settlement" for fill in result.fills)
    assert result.terminal_settlement_count.sum() == 0


def test_settlement_incidence_above_fifteen_percent_marks_economics_unresolved() -> (
    None
):
    close = np.full((12, 2), 100.0)
    close[1:, 0] = np.nan
    scores = np.asarray([[1.0, -1.0]] * len(close))
    result = _run(
        close,
        scores,
        initial_reference_price=np.full(2, 100.0),
    )

    summary = result.summary()
    assert result.economics_unresolved
    assert summary["terminal_settlement_economics_unresolved"] is True
    assert summary["terminal_settlement_notional_fraction_nav"] > 0.15


def test_print_after_terminal_settlement_is_counted_without_reopening() -> None:
    close = np.full((13, 2), 100.0)
    close[1:11, 0] = np.nan
    scores = np.asarray([[1.0, -1.0]] * len(close))
    result = _run(
        close,
        scores,
        initial_reference_price=np.full(2, 100.0),
    )

    assert result.terminal_settlement_count.sum() == 1
    assert result.settled_then_printed_count.sum() == 1
    assert result.signed_shares[10, 0] == 0.0
    assert result.signed_shares[11, 0] == 0.0
    assert not any(
        order.security_index == 0
        and order.purpose == "entry"
        and order.decision_session > 10
        for order in result.intended_orders
    )


def test_unresolved_once_then_resolved_name_exits_on_next_instruction() -> None:
    close = np.full((5, 3), 100.0)
    scores = np.asarray(
        [
            [3.0, 0.0, -3.0],
            [3.0, 0.0, -3.0],
            [0.0, 3.0, -3.0],
            [0.0, 3.0, -3.0],
            [0.0, 3.0, -3.0],
        ]
    )
    resolved = np.ones_like(close, dtype=np.bool_)
    resolved[1, 0] = False
    actions = AlignedActionTerms(
        shares_per_prior_share=np.ones_like(close),
        cash_per_prior_share=np.zeros_like(close),
        session_resolved=resolved,
        has_action=np.zeros_like(resolved),
        successor_index=np.broadcast_to(
            np.arange(close.shape[1], dtype=np.int64), close.shape
        ).copy(),
    )
    result = _run(
        close,
        scores,
        actions=actions,
        initial_reference_price=np.full(close.shape[1], 100.0),
    )

    exit_order = next(
        order
        for order in result.intended_orders
        if order.security_index == 0 and order.purpose == "exit"
    )
    assert exit_order.decision_session == 2
    assert any(
        fill.order_id == exit_order.order_id and fill.fill_session == 2
        for fill in result.fills
    )
    assert result.unresolved_action_name_days[1] == 0
    assert result.unresolved_action_name_days[2] == 0
    assert result.unresolved_claim_inventory_fraction_nav[1] == 0.0
    assert result.stale_mark_inventory_fraction_nav[1] == 0.0
    assert result.signed_shares[2, 0] == 0.0
    assert not result.economics_unresolved


def test_dividend_receivable_does_not_block_later_exit() -> None:
    close = np.full((5, 3), 100.0)
    scores = np.asarray(
        [
            [3.0, 0.0, -3.0],
            [3.0, 0.0, -3.0],
            [0.0, 3.0, -3.0],
            [0.0, 3.0, -3.0],
            [0.0, 3.0, -3.0],
        ]
    )
    actions = _action_terms(close.shape, day=1, d=10.0)
    payment = np.full(close.shape, -1, dtype=np.int64)
    payment[1] = 4
    result = _run(
        close,
        scores,
        actions=actions,
        payment_session=payment,
        initial_reference_price=np.full(close.shape[1], 100.0),
    )

    assert result.receivables[1] > 0.0
    long_exit = next(
        order
        for order in result.intended_orders
        if order.security_index == 0 and order.purpose == "exit"
    )
    assert any(
        fill.order_id == long_exit.order_id and fill.fill_session == 2
        for fill in result.fills
    )
    assert result.signed_shares[2, 0] == 0.0


def test_terminal_liquidates_every_printed_position_regardless_of_action_flag() -> None:
    close = np.full((3, 3), 100.0)
    scores = np.asarray([[3.0, 0.0, -3.0]] * 3)
    resolved = np.ones_like(close, dtype=np.bool_)
    resolved[-1, (0, 2)] = False
    actions = AlignedActionTerms(
        shares_per_prior_share=np.ones_like(close),
        cash_per_prior_share=np.zeros_like(close),
        session_resolved=resolved,
        has_action=np.zeros_like(resolved),
        successor_index=np.broadcast_to(
            np.arange(close.shape[1], dtype=np.int64), close.shape
        ).copy(),
    )
    result = _run(
        close,
        scores,
        actions=actions,
        initial_reference_price=np.full(close.shape[1], 100.0),
    )

    terminal_orders = [
        order for order in result.intended_orders if order.purpose == "terminal_exit"
    ]
    assert {order.security_index for order in terminal_orders} == {0, 2}
    assert all(
        any(fill.order_id == order.order_id for fill in result.fills)
        for order in terminal_orders
    )
    assert not np.any(result.signed_shares[-1])
    assert result.unresolved_inventory_count == 0


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


def test_small_universe_uses_effective_two_sided_k_without_rescaling_slots() -> None:
    close = np.full((4, 10), 100.0)
    scores = np.broadcast_to(np.arange(10, dtype=np.float64), close.shape).copy()
    result = _run(
        close,
        scores,
        config=_config(
            k_per_side=30,
            buffer_per_side=30,
            cost_bps_per_side=0.0,
            annual_borrow_rate=0.0,
        ),
    )

    entries = [order for order in result.intended_orders if order.purpose == "entry"]
    assert sum(order.side == "buy" for order in entries) == 5
    assert sum(order.side == "sell" for order in entries) == 5
    assert all(order.planned_notional == 1.0 / 30.0 for order in entries)
    assert not result.entry_blocked_small_universe.any()
    np.testing.assert_array_equal(result.retention_width, 5)


def test_same_close_exit_frees_slot_and_failed_exit_is_trimmed_next_day() -> None:
    close = np.full((5, 4), 100.0)
    scores = np.asarray(
        [
            [-3.0, -1.0, 1.0, 3.0],
            [-3.0, -1.0, 1.0, 3.0],
            [-2.0, -3.0, 3.0, 2.0],
            [-2.0, -3.0, 3.0, 2.0],
            [-2.0, -3.0, 3.0, 2.0],
        ]
    )
    fill_fraction = np.ones_like(close)
    fill_fraction[2, 3] = 0.0
    result = _run(
        close,
        scores,
        fill_fraction=fill_fraction,
        config=_config(buffer_per_side=0, planned_gross_cap=2.25),
    )

    replacement_entries = [
        order
        for order in result.intended_orders
        if order.decision_session == 2 and order.purpose == "entry"
    ]
    assert {order.security_index for order in replacement_entries} == {1, 2}
    assert result.same_close_replacement_count[2] == 2
    assert result.actual_risk_breach[2]
    assert result.risk_trim_gross_notional[3] > 0.0
    assert result.summary()["mean_daily_exits_per_side"] > 0.0


def test_retention_width_cannot_overlap_rank_bands() -> None:
    close = np.full((4, 6), 100.0)
    scores = np.broadcast_to(np.arange(6, dtype=np.float64), close.shape).copy()
    result = _run(
        close,
        scores,
        config=_config(
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
        config=_config(
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
        config=_config(
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


def test_same_close_replacements_avoid_artificial_net_trim() -> None:
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
        config=_config(
            k_per_side=30,
            buffer_per_side=0,
            planned_absolute_net_cap=0.08,
            cost_bps_per_side=0.0,
            annual_borrow_rate=0.0,
        ),
        initial_reference_price=np.full(names, 100.0),
    )

    assert result.same_close_replacement_count[1] == 4
    assert not result.actual_risk_breach.any()
    assert result.risk_trim_net_notional[2] == 0.0
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
        config=_config(
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
        order.position_fraction
        * abs(result.signed_shares[1, order.security_index])
        * order.reference_price
        for order in risk_orders
        if order.side == "sell"
    )
    short_trim = sum(
        order.position_fraction
        * abs(result.signed_shares[1, order.security_index])
        * order.reference_price
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
        config=_config(
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
        config=_config(
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


def test_pass4g_shortfall_decomposition_and_defect_counters() -> None:
    days, names = 8, 100
    close = np.full((days, names), 100.0)
    scores = np.tile(np.arange(names, dtype=np.float64), (days, 1))
    result = _run(
        close,
        scores,
        config=_config(
            k_per_side=30,
            buffer_per_side=30,
            cost_bps_per_side=0.0,
            annual_borrow_rate=0.0,
        ),
        initial_reference_price=np.full(names, 100.0),
    )

    terms = (
        result.gross_shortfall_small_universe
        + result.gross_shortfall_occupancy_pending
        + result.gross_shortfall_occupancy_band_exhausted
        + result.gross_shortfall_occupancy_blocked
        + result.gross_shortfall_occupancy_exit_gap
        + result.gross_shortfall_occupancy_other
        + result.gross_shortfall_sizing_fill
        + result.gross_shortfall_sizing_mark_drift
        + result.gross_shortfall_sizing_nav_drift
    )
    np.testing.assert_allclose(
        terms, result.gross_target - result.gross_fraction_nav, atol=1e-12, rtol=0.0
    )
    np.testing.assert_allclose(result.gross_fraction_nav[:-1], 2.0, atol=1e-12)
    summary = result.summary()
    assert summary["entry_defect_signatures"] == {
        "D1_entry_pending_printed_unblocked_unfilled": 0,
        "D2_entry_fill_quantity_short": 0,
        "D3_blocked_open_slots": 0,
        "D4_cap_block_defects": 0,
        "D5_ineligible_exit_within_hold_window": 0,
    }


def test_pass4g_pending_nonprinters_are_explained_without_behavior_change() -> None:
    days, names = 9, 100
    close = np.full((days, names), 100.0)
    close[:, 98:] = np.nan
    scores = np.tile(np.arange(names, dtype=np.float64), (days, 1))
    result = _run(
        close,
        scores,
        config=_config(
            k_per_side=30,
            buffer_per_side=30,
            cost_bps_per_side=0.0,
            annual_borrow_rate=0.0,
        ),
        initial_reference_price=np.full(names, 100.0),
    )

    np.testing.assert_array_equal(
        result.pending_entries_without_print_today_long[:8],
        np.asarray([2, 2, 0, 2, 2, 0, 2, 2]),
    )
    assert np.all(result.band_candidates_without_prior_session_print_long[1:8] <= 2)
    assert result.entry_pending_printed_unblocked_unfilled.sum() == 0
    terms = (
        result.gross_shortfall_small_universe
        + result.gross_shortfall_occupancy_pending
        + result.gross_shortfall_occupancy_band_exhausted
        + result.gross_shortfall_occupancy_blocked
        + result.gross_shortfall_occupancy_exit_gap
        + result.gross_shortfall_occupancy_other
        + result.gross_shortfall_sizing_fill
        + result.gross_shortfall_sizing_mark_drift
        + result.gross_shortfall_sizing_nav_drift
    )
    np.testing.assert_allclose(
        terms, result.gross_target - result.gross_fraction_nav, atol=1e-12, rtol=0.0
    )


def test_pass4g_sizing_decomposition_matches_hand_computed_mark_and_nav_drift() -> None:
    close = np.full((3, 3), 100.0)
    close[1, 2] = 110.0
    scores = np.asarray([[-1.0, 0.0, 1.0]] * 3)
    result = _run(
        close,
        scores,
        initial_reference_price=np.full(3, 100.0),
    )

    expected_mark_drift = -0.1 / 1.1
    expected_nav_drift = 2.0 * 0.1 / 1.1
    np.testing.assert_allclose(result.gross_shortfall_sizing_fill[1], 0.0, atol=1e-12)
    np.testing.assert_allclose(
        result.gross_shortfall_sizing_mark_drift[1],
        expected_mark_drift,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.gross_shortfall_sizing_nav_drift[1],
        expected_nav_drift,
        atol=1e-12,
    )


def test_rev4_defaults_bind_lending_volatility_balance_and_beta_hedge() -> None:
    config = LedgerConfig()
    assert config.borrow_source == "borrow_balance"
    assert config.volatility_balanced_entries
    assert config.small_stratum_scaling_threshold_multiple == 2
    assert config.beta_hedge
    assert config.hedge_rebalance_threshold_nav == 0.05
    assert config.hedge_notional_cap_nav == 0.60
    assert config.hedge_cost_bps_per_side == 4.0
    uniform = ledger_configurations()["comparator_uniform_borrow"]
    assert uniform.borrow_source == "uniform"
    assert uniform.volatility_balanced_entries
    assert uniform.beta_hedge


@pytest.mark.parametrize("changed_instrument", ["equity", "hedge"])
def test_rev4f_all_orders_ignore_later_closes(changed_instrument: str) -> None:
    close = np.full((4, 4), 100.0)
    bova = np.full(4, 100.0)
    scores = np.tile([-2.0, -1.0, 1.0, 2.0], (4, 1))
    config = _config(beta_hedge=True, hedge_cost_bps_per_side=4.0)
    kwargs = dict(
        config=config,
        initial_reference_price=np.full(4, 100.0),
        hedge_beta=np.tile([1.0, 1.0, 1.134, 1.134], (4, 1)),
    )
    original = _run(close, scores, hedge_close=bova, **kwargs)
    if changed_instrument == "equity":
        close[0, 3] = 110.0
    else:
        bova[0] = 110.0
    changed = _run(close, scores, hedge_close=bova, **kwargs)
    original_orders = [
        order for order in original.intended_orders if order.decision_session == 0
    ]
    changed_orders = [
        order for order in changed.intended_orders if order.decision_session == 0
    ]
    assert original_orders == changed_orders
    hedge = next(order for order in original_orders if order.purpose == "hedge")
    assert hedge.reference_price == 100.0
    assert hedge.planned_notional == pytest.approx(0.134)
    assert original.hedge_target_notional[0] == pytest.approx(-0.134)
    assert [f for f in original.fills if f.fill_session == 0] != [
        f for f in changed.fills if f.fill_session == 0
    ]
    np.testing.assert_allclose(changed.reconciliation_error, 0.0, atol=1e-15)


def test_rev4f_missing_hedge_print_cancels_and_reconciles_notional() -> None:
    scores = np.tile([-1.0, 1.0], (4, 1))
    result = _run(
        np.full((4, 2), 100.0),
        scores,
        config=_config(beta_hedge=True),
        initial_reference_price=np.full(2, 100.0),
        hedge_beta=np.tile([1.0, 1.2], (4, 1)),
        hedge_close=np.asarray([np.nan, 100.0, 100.0, 100.0]),
    )
    hedges = [order for order in result.intended_orders if order.purpose == "hedge"]
    assert hedges[0].decision_session == 0
    cancellation = next(
        c for c in result.cancellations if c.order_id == hedges[0].order_id
    )
    assert cancellation.reason == "expired"
    for order in hedges:
        if order.position_fraction is not None:
            assert order.position_fraction == 1.0
            continue
        filled = sum(
            f.gross_notional for f in result.fills if f.order_id == order.order_id
        )
        cancelled = sum(
            c.unfilled_notional
            for c in result.cancellations
            if c.order_id == order.order_id
        )
        assert filled + cancelled == pytest.approx(order.planned_notional)
    assert result.hedge_signed_shares[-1] == 0.0


def test_rev4f_hedge_fallback_does_not_block_entries_and_is_reported() -> None:
    result = _run(
        np.full((4, 2), 100.0),
        np.tile([-1.0, 1.0], (4, 1)),
        config=_config(beta_hedge=True),
        initial_reference_price=np.full(2, 100.0),
        hedge_beta=np.zeros((4, 2)),
        hedge_beta_valid=np.zeros((4, 2), dtype=bool),
        hedge_close=np.full(4, 100.0),
    )
    assert result.submitted_entry_count[0] == 2
    assert result.hedge_beta_fallback_sessions.all()
    assert result.summary()["hedge_beta_fallback_sessions"] == 4
    np.testing.assert_array_equal(result.hedge_target_notional, 0.0)


def test_terminal_settlement_intention_precedes_possible_last_print() -> None:
    close = np.full((13, 2), 100.0)
    close[1:, 1] = np.nan
    scores = np.tile([-1.0, 1.0], (13, 1))
    original = _run(close, scores, initial_reference_price=np.full(2, 100.0))
    close[10, 1] = 105.0
    changed = _run(close, scores, initial_reference_price=np.full(2, 100.0))
    assert [o for o in original.intended_orders if o.decision_session == 10] == [
        o for o in changed.intended_orders if o.decision_session == 10
    ]
    assert original.terminal_settlement_count[10] == 1
    assert changed.terminal_settlement_count[10] == 0


def test_ledger_signature_does_not_accept_model_feature_beta() -> None:
    import inspect

    parameters = inspect.signature(simulate_stateful_ledger).parameters
    assert "hedge_beta" in parameters and "hedge_beta_valid" in parameters
    assert "beta_60" not in parameters


def test_rev4f_cost_grid_preserves_headline_construction() -> None:
    headline = LedgerConfig()
    grid = ledger_configurations()
    for cost in (2.0, 4.0, 7.0):
        for borrow in ("balance", "strict", "open"):
            key = (
                f"borrow_{borrow}" if cost == 4.0 else f"cost_{cost:g}_borrow_{borrow}"
            )
            assert grid[key] == replace(
                headline, cost_bps_per_side=cost, borrow_source=f"borrow_{borrow}"
            )


def test_rev4f_pending_entry_cancelled_outside_current_quintile_band() -> None:
    days, names = 4, 100
    scores = np.tile(np.arange(names, dtype=float), (days, 1))
    scores[0, 0] = 1000.0
    scores[1:, 0] = 10.5
    close = np.full((days, names), 100.0)
    close[0, 0] = np.nan
    config = _config(
        k_per_side=5,
        buffer_per_side=5,
        volatility_balanced_entries=True,
        planned_gross_cap=3.0,
        planned_absolute_net_cap=1.1,
    )
    inputs = dict(
        selection_volatility=np.tile(np.repeat(np.arange(5), 20), (days, 1)),
        initial_reference_price=np.full(names, 100.0),
    )
    result = _run(close, scores, config=config, **inputs)
    initial = next(
        o
        for o in result.intended_orders
        if o.security_index == 0 and o.purpose == "entry"
    )
    cancelled = next(c for c in result.cancellations if c.order_id == initial.order_id)
    assert cancelled.reason == "band_exit"
    assert cancelled.cancellation_session == 1
    assert not any(f.order_id == initial.order_id for f in result.fills)
    off = _run(
        close,
        scores,
        config=replace(config, cancel_pending_outside_retention=False),
        **inputs,
    )
    assert any(
        f.order_id == initial.order_id and f.fill_session == 1 for f in off.fills
    )


def test_rev4_entry_scheduler_fills_equal_volatility_quotas() -> None:
    days, names = 4, 300
    score_order = np.asarray(
        [group * 60 + within for within in range(60) for group in range(5)]
    )
    score = np.empty(names, dtype=np.float64)
    score[score_order] = np.arange(names, dtype=np.float64)
    scores = np.broadcast_to(score, (days, names)).copy()
    volatility = np.broadcast_to(
        np.repeat(np.arange(5, dtype=np.float64), 60), (days, names)
    ).copy()
    inputs = _constructed_inputs(days, names)
    inputs["selection_volatility"] = volatility
    result = _run(
        np.full((days, names), 100.0),
        scores,
        config=replace(
            LedgerConfig(),
            beta_hedge=False,
            cost_bps_per_side=0.0,
            hedge_cost_bps_per_side=0.0,
        ),
        initial_reference_price=np.full(names, 100.0),
        **inputs,
    )

    np.testing.assert_array_equal(result.volatility_quota[0], [6, 6, 6, 6, 6])
    np.testing.assert_array_equal(result.volatility_group_size[0], [60, 60, 60, 60, 60])
    assert result.entry_eligible_name_count[0] == 300
    assert result.summary()["mean_entry_eligible_name_count"] == 300.0
    assert result.summary()["mean_volatility_group_size_by_quintile"] == [
        60.0,
        60.0,
        60.0,
        60.0,
        60.0,
    ]
    assert result.summary()["minimum_volatility_group_size_by_quintile"] == [
        60,
        60,
        60,
        60,
        60,
    ]
    np.testing.assert_array_equal(result.volatility_occupancy_long[0], [6, 6, 6, 6, 6])
    np.testing.assert_array_equal(result.volatility_occupancy_short[0], [6, 6, 6, 6, 6])


def test_rev4_monotone_negative_volatility_score_fills_both_sides_per_quintile() -> (
    None
):
    days, names = 4, 300
    volatility = np.broadcast_to(
        np.arange(names, dtype=np.float64), (days, names)
    ).copy()
    inputs = _constructed_inputs(days, names)
    inputs["selection_volatility"] = volatility
    result = _run(
        np.full((days, names), 100.0),
        -volatility,
        config=replace(LedgerConfig(), beta_hedge=False, cost_bps_per_side=0.0),
        initial_reference_price=np.full(names, 100.0),
        **inputs,
    )

    np.testing.assert_array_equal(result.volatility_quota[0], [6, 6, 6, 6, 6])
    np.testing.assert_array_equal(result.volatility_occupancy_long[0], [6, 6, 6, 6, 6])
    np.testing.assert_array_equal(result.volatility_occupancy_short[0], [6, 6, 6, 6, 6])


def test_rev4_retention_uses_the_names_current_volatility_quintile() -> None:
    days, names = 4, 100
    initial_volatility = np.repeat(np.arange(5, dtype=np.float64), 20)
    moved_volatility = initial_volatility.copy()
    moved_volatility[[0, 80]] = moved_volatility[[80, 0]]
    volatility = np.stack(
        [initial_volatility, initial_volatility, moved_volatility, moved_volatility]
    )
    scores = np.broadcast_to(np.arange(names, dtype=np.float64), (days, names)).copy()
    scores[:2, 0] = 1_000.0
    scores[2:, 1:20] = np.arange(81.0, 100.0)
    scores[2:, 80] = 100.0
    scores[2:, 81:100] = np.arange(19.0)
    scores[2:, 0] = 50.0
    inputs = _constructed_inputs(days, names)
    inputs["selection_volatility"] = volatility
    config = replace(
        LedgerConfig(),
        k_per_side=5,
        buffer_per_side=5,
        beta_hedge=False,
        planned_gross_cap=3.0,
        planned_absolute_net_cap=0.5,
        planned_name_weight_cap=0.25,
        cost_bps_per_side=0.0,
    )
    moved = _run(
        np.full((days, names), 100.0),
        scores,
        config=config,
        initial_reference_price=np.full(names, 100.0),
        **inputs,
    )
    inputs["selection_volatility"] = np.broadcast_to(
        initial_volatility, (days, names)
    ).copy()
    stayed = _run(
        np.full((days, names), 100.0),
        scores,
        config=config,
        initial_reference_price=np.full(names, 100.0),
        **inputs,
    )

    assert moved.position_sign[1, 0] == 1
    assert moved.exit_instruction_cause[2, 0] == 0
    assert stayed.exit_instruction_cause[2, 0] == 3


def test_rev4_small_stratum_scales_quota_and_buffer_without_band_overlap() -> None:
    quota, buffer = _scaled_group_bands(
        k_eff=30,
        buffer=30,
        group_sizes=np.asarray([60, 60, 8, 60, 60]),
        threshold_multiple=2,
    )

    assert quota[2] == 2
    assert buffer[2] == 2
    assert 2 * (quota[2] + buffer[2]) <= 8


def test_rev4d_stratum_threshold_preserves_full_and_scaled_disjoint_bands() -> None:
    full_quota, full_buffer = _scaled_group_bands(
        k_eff=30,
        buffer=30,
        group_sizes=np.full(5, 38),
        threshold_multiple=2,
    )
    scaled_quota, scaled_buffer = _scaled_group_bands(
        k_eff=30,
        buffer=30,
        group_sizes=np.full(5, 20),
        threshold_multiple=2,
    )

    np.testing.assert_array_equal(full_quota, [6, 6, 6, 6, 6])
    np.testing.assert_array_equal(full_buffer, [6, 6, 6, 6, 6])
    np.testing.assert_array_equal(scaled_quota, [5, 5, 5, 5, 5])
    np.testing.assert_array_equal(scaled_buffer, [5, 5, 5, 5, 5])
    assert np.all(2 * (full_quota + full_buffer) <= 38)
    assert np.all(2 * (scaled_quota + scaled_buffer) <= 20)


def test_rev4c_small_stratum_fixture_is_exact_when_multiple_is_four() -> None:
    quota, buffer = _scaled_group_bands(
        k_eff=30,
        buffer=30,
        group_sizes=np.asarray([60, 60, 8, 60, 60]),
        threshold_multiple=4,
    )

    np.testing.assert_array_equal(quota, [6, 6, 1, 6, 6])
    np.testing.assert_array_equal(buffer, [6, 6, 1, 6, 6])


def test_rev4_unavailable_short_quota_spills_to_best_remaining_names() -> None:
    days, names = 4, 300
    score_order = np.asarray(
        [group * 60 + within for within in range(60) for group in range(5)]
    )
    score = np.empty(names, dtype=np.float64)
    score[score_order] = np.arange(names, dtype=np.float64)
    scores = np.broadcast_to(score, (days, names)).copy()
    volatility = np.broadcast_to(
        np.repeat(np.arange(5, dtype=np.float64), 60), (days, names)
    ).copy()
    inputs = _constructed_inputs(days, names)
    inputs["selection_volatility"] = volatility
    shortable = inputs["shortable"]
    q5_names = np.arange(240, 300)
    q5_score_order = q5_names[np.argsort(score[q5_names], kind="stable")]
    shortable[:, q5_names] = False
    shortable[:, q5_score_order[:3]] = True
    result = _run(
        np.full((days, names), 100.0),
        scores,
        config=replace(LedgerConfig(), beta_hedge=False, cost_bps_per_side=0.0),
        initial_reference_price=np.full(names, 100.0),
        **inputs,
    )

    assert result.volatility_occupancy_short[0, 4] == 3
    assert result.volatility_occupancy_short[0].sum() == 30
    assert result.volatility_occupancy_short[0].max() > 6


def test_rev4_beta_hedge_is_separate_rebalances_and_carries_missing_print() -> None:
    days, names = 4, 4
    scores = np.broadcast_to(np.asarray([-2.0, -1.0, 1.0, 2.0]), (days, names)).copy()
    beta = np.broadcast_to(np.asarray([0.5, 0.5, 1.5, 1.5]), (days, names)).copy()
    rates = np.full((days, names), 0.02)
    shortable = np.ones((days, names), dtype=np.bool_)
    result = _run(
        np.full((days, names), 100.0),
        scores,
        config=replace(
            LedgerConfig(),
            k_per_side=1,
            buffer_per_side=1,
            volatility_balanced_entries=False,
            planned_gross_cap=4.0,
            planned_absolute_net_cap=1.1,
            planned_name_weight_cap=1.1,
            cost_bps_per_side=0.0,
        ),
        initial_reference_price=np.full(names, 100.0),
        annual_borrow_rate_by_name=rates,
        borrow_rate_imputed=np.zeros_like(shortable),
        borrow_rate_placeholder=np.zeros_like(shortable),
        shortable=shortable,
        hedge_beta=beta,
        hedge_close=np.asarray([100.0, np.nan, 110.0, 110.0]),
    )

    assert result.hedge_signed_notional[0] != 0.0
    assert result.hedge_turnover_fraction_nav[0] > 0.0
    assert result.hedge_signed_shares[1] == result.hedge_signed_shares[0]
    assert result.hedge_mark_price[1] == result.hedge_mark_price[0]
    assert result.hedge_capped[0]
    assert result.ex_ante_beta_after_hedge[0] == pytest.approx(0.4)
    assert abs(result.hedge_target_notional[0]) / result.start_nav[0] == pytest.approx(
        0.60
    )
    assert result.hedge_borrow_bps[1] > 0.0
    assert result.hedge_signed_notional[-1] == 0.0
    assert np.all(
        result.gross_fraction_nav_including_hedge >= result.gross_fraction_nav
    )
    assert np.all(result.planned_gross_fraction_nav <= 4.0 + 1e-12)
    assert np.all(np.abs(result.planned_net_fraction_nav) <= 1.1 + 1e-12)


def test_rev4c_hedge_does_not_consume_equity_caps() -> None:
    days, names = 4, 20
    scores = np.broadcast_to(np.arange(names, dtype=np.float64), (days, names)).copy()
    inputs = _constructed_inputs(days, names)
    inputs["hedge_beta"] = np.broadcast_to(
        np.concatenate((np.full(10, 0.45), np.zeros(10))), (days, names)
    ).copy()
    result = _run(
        np.full((days, names), 100.0),
        scores,
        config=replace(
            LedgerConfig(),
            k_per_side=10,
            buffer_per_side=10,
            volatility_balanced_entries=False,
            cost_bps_per_side=0.0,
            hedge_cost_bps_per_side=0.0,
            planned_name_weight_cap=0.11,
        ),
        initial_reference_price=np.full(names, 100.0),
        **inputs,
    )

    assert result.planned_gross_fraction_nav[0] == pytest.approx(2.0)
    assert result.hedge_target_notional[0] == pytest.approx(0.45)
    assert result.summary()["entry_defect_signatures"]["D4_cap_block_defects"] == 0


def test_rev4c_hedge_requirement_is_capped_and_labelled() -> None:
    days, names = 4, 20
    scores = np.broadcast_to(np.arange(names, dtype=np.float64), (days, names)).copy()
    inputs = _constructed_inputs(days, names)
    inputs["hedge_beta"] = np.broadcast_to(
        np.concatenate((np.zeros(10), np.full(10, 0.8))), (days, names)
    ).copy()
    result = _run(
        np.full((days, names), 100.0),
        scores,
        config=replace(
            LedgerConfig(),
            k_per_side=10,
            buffer_per_side=10,
            volatility_balanced_entries=False,
            cost_bps_per_side=0.0,
            hedge_cost_bps_per_side=0.0,
            planned_name_weight_cap=0.11,
        ),
        initial_reference_price=np.full(names, 100.0),
        **inputs,
    )

    assert result.hedge_unconstrained_target_notional[0] == pytest.approx(-0.8)
    assert result.hedge_target_notional[0] == pytest.approx(-0.6)
    assert result.hedge_capped[0]
    assert result.ex_ante_beta_after_hedge[0] == pytest.approx(0.2)


def test_rev4c_cap_setting_is_bit_identical_when_hedge_is_disabled() -> None:
    close = np.full((5, 6), 100.0)
    scores = np.broadcast_to(np.arange(6, dtype=np.float64), close.shape).copy()
    base = _run(
        close,
        scores,
        config=replace(_config(), hedge_notional_cap_nav=0.60),
        initial_reference_price=np.full(6, 100.0),
    )
    changed = _run(
        close,
        scores,
        config=replace(_config(), hedge_notional_cap_nav=0.25),
        initial_reference_price=np.full(6, 100.0),
    )

    np.testing.assert_array_equal(base.nav, changed.nav)
    np.testing.assert_array_equal(base.signed_shares, changed.signed_shares)
    assert base.intended_orders == changed.intended_orders
    assert base.fills == changed.fills


def test_rev4_uniform_comparator_is_exact_legacy_ledger() -> None:
    days, names = 5, 6
    close = np.full((days, names), 100.0)
    scores = np.broadcast_to(np.arange(names, dtype=np.float64), close.shape).copy()
    legacy = _run(close, scores, initial_reference_price=np.full(names, 100.0))
    comparator = _run(
        close,
        scores,
        config=replace(
            LedgerConfig(),
            k_per_side=1,
            buffer_per_side=1,
            cost_bps_per_side=0.0,
            annual_borrow_rate=0.0,
            borrow_source="uniform",
            borrow_registration_fee_fraction=0.0,
            borrow_registration_fee_floor=0.0,
            borrow_registration_fee_cap=0.0,
            volatility_balanced_entries=False,
            beta_hedge=False,
            planned_absolute_net_cap=1.10,
            planned_name_weight_cap=1.10,
        ),
        initial_reference_price=np.full(names, 100.0),
    )

    np.testing.assert_array_equal(comparator.nav, legacy.nav)
    np.testing.assert_array_equal(comparator.daily_net_return, legacy.daily_net_return)
    assert comparator.intended_orders == legacy.intended_orders
    assert comparator.fills == legacy.fills
