from datetime import date, timedelta

import numpy as np

from brazil_rv.execution.stateful_ledger import LedgerConfig, simulate_stateful_ledger


def _run(
    close: np.ndarray,
    scores: np.ndarray,
    *,
    config: LedgerConfig | None = None,
    active: np.ndarray | None = None,
    events: np.ndarray | None = None,
    median: np.ndarray | None = None,
) -> object:
    days, names = close.shape
    dates = tuple(date(2024, 1, 2) + timedelta(days=index) for index in range(days))
    mask = np.ones((days, names), dtype=np.bool_)
    membership = mask if active is None else active
    event = np.zeros_like(mask) if events is None else events
    medians = np.zeros(days) if median is None else median
    return simulate_stateful_ledger(
        dates=dates,
        scores=scores,
        score_mask=mask,
        active=membership,
        adjusted_close=close,
        neutralized_log_return=np.zeros_like(close),
        neutralized_log_return_valid=mask,
        return_neutralized_event_mask=event,
        cross_sectional_median_log_return=medians,
        cdi_returns=np.zeros(days),
        config=config
        or LedgerConfig(
            k_per_side=1,
            buffer_per_side=1,
            cost_bps_per_side=0,
            annual_borrow_rate=0,
        ),
    )


def test_fixed_book_books_large_move_and_neutral_flow_without_dropping_day() -> None:
    close = np.asarray([[100.0, 100.0], [80.0, 100.0], [80.0, 100.0]])
    scores = np.asarray([[1.0, -1.0], [1.0, -1.0], [1.0, -1.0]])
    raw = _run(close, scores)
    np.testing.assert_allclose(raw.gross_pnl_bps, [0.0, -2_000.0, 0.0])
    assert np.isfinite(raw.nav).all()

    events = np.zeros_like(close, dtype=np.bool_)
    events[1, 0] = True
    neutral = _run(close, scores, events=events)
    np.testing.assert_allclose(neutral.daily_net_return, [0.0, 0.0, 0.0])
    assert neutral.neutral_marked_name_days.sum() == 1


def test_rank_buffer_is_stateful_and_zero_buffer_flips() -> None:
    close = np.full((3, 2), 100.0)
    scores = np.asarray([[1.0, -1.0], [-1.0, 1.0], [-1.0, 1.0]])
    buffered = _run(close, scores)
    assert buffered.position_sign[0].tolist() == [1, -1]
    assert buffered.position_sign[1].tolist() == [1, -1]
    exact = _run(
        close,
        scores,
        config=LedgerConfig(
            k_per_side=1,
            buffer_per_side=0,
            cost_bps_per_side=0,
            annual_borrow_rate=0,
        ),
    )
    assert exact.position_sign[1].tolist() == [-1, 1]


def test_missing_exit_waits_then_exits_or_forces_with_haircut() -> None:
    close = np.full((5, 2), 100.0)
    close[1, 0] = np.nan
    scores = np.asarray(
        [[1.0, -1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]
    )
    waited = _run(
        close,
        scores,
        config=LedgerConfig(
            k_per_side=1,
            buffer_per_side=0,
            cost_bps_per_side=0,
            annual_borrow_rate=0,
        ),
    )
    assert waited.stale_mark_name_days[1] == 1
    assert waited.forced_liquidation_count.sum() == 0
    assert waited.position_sign[1, 0] == 1
    assert waited.position_sign[2, 0] == -1

    missing = np.full((4, 2), 100.0)
    missing[1:, 0] = np.nan
    active = np.ones_like(missing, dtype=np.bool_)
    forced = _run(
        missing,
        scores[:4],
        active=active,
        config=LedgerConfig(
            k_per_side=1,
            buffer_per_side=0,
            cost_bps_per_side=0,
            annual_borrow_rate=0,
            max_missing_sessions=2,
            forced_liquidation_haircut=0.30,
        ),
    )
    assert forced.forced_liquidation_count.sum() == 1
    assert forced.gross_pnl_bps[2] == -3_000.0
    no_haircut = _run(
        missing,
        scores[:4],
        active=active,
        config=LedgerConfig(
            k_per_side=1,
            buffer_per_side=0,
            cost_bps_per_side=0,
            annual_borrow_rate=0,
            max_missing_sessions=2,
            forced_liquidation_haircut=0.0,
        ),
    )
    assert no_haircut.forced_liquidation_count.sum() == 1
    assert no_haircut.gross_pnl_bps[2] == 0.0


def test_turnover_financing_split_and_causality() -> None:
    close = np.full((3, 2), 100.0)
    scores = np.asarray([[1.0, -1.0], [1.0, -1.0], [1.0, -1.0]])
    base = _run(close, scores)
    np.testing.assert_allclose(base.turnover_fraction_nav, [2.0, 0.0, 2.0])
    financed = _run(
        close,
        scores,
        config=LedgerConfig(
            k_per_side=1,
            buffer_per_side=1,
            cost_bps_per_side=0,
            short_proceeds_remuneration=1.0,
            annual_borrow_rate=0,
        ),
    )
    np.testing.assert_allclose(base.interest_bps, 0.0)
    np.testing.assert_allclose(financed.interest_bps, 0.0)

    first = _run(close, scores)
    changed = close.copy()
    changed[2] = [1_000.0, 1.0]
    second = _run(changed, scores)
    np.testing.assert_array_equal(first.position_sign[:2], second.position_sign[:2])


def test_split_adjusted_share_units_create_no_artificial_pnl() -> None:
    # A 2-for-1 raw split is already represented by unchanged adjusted closes;
    # the ledger therefore needs no split flow or position rewrite.
    adjusted_close = np.full((3, 2), 100.0)
    scores = np.asarray([[1.0, -1.0]] * 3)
    result = _run(adjusted_close, scores)
    np.testing.assert_array_equal(result.gross_pnl_bps, np.zeros(3))


def test_full_short_proceeds_earns_cdi_while_zero_fraction_does_not() -> None:
    days = 3
    close = np.full((days, 2), 100.0)
    scores = np.asarray([[1.0, -1.0]] * days)
    dates = tuple(date(2024, 1, 2) + timedelta(days=index) for index in range(days))
    common = dict(
        dates=dates,
        scores=scores,
        score_mask=np.ones_like(close, dtype=np.bool_),
        active=np.ones_like(close, dtype=np.bool_),
        adjusted_close=close,
        neutralized_log_return=np.zeros_like(close),
        neutralized_log_return_valid=np.ones_like(close, dtype=np.bool_),
        return_neutralized_event_mask=np.zeros_like(close, dtype=np.bool_),
        cross_sectional_median_log_return=np.zeros(days),
        cdi_returns=np.asarray([0.0, 0.001, 0.001]),
    )
    zero = simulate_stateful_ledger(
        **common,
        config=LedgerConfig(
            k_per_side=1,
            buffer_per_side=1,
            cost_bps_per_side=0,
            annual_borrow_rate=0,
        ),
    )
    full = simulate_stateful_ledger(
        **common,
        config=LedgerConfig(
            k_per_side=1,
            buffer_per_side=1,
            cost_bps_per_side=0,
            short_proceeds_remuneration=1.0,
            annual_borrow_rate=0,
        ),
    )
    assert zero.interest_bps[1] == 0.0
    assert full.interest_bps[1] > 0.0
