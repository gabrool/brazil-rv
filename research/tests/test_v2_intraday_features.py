from datetime import date, datetime, time, timedelta

import numpy as np
import polars as pl

from brazil_rv.v2.build_store import stream_intraday_from_assignments
from brazil_rv.v2.decision_clock import SessionDefinition
from brazil_rv.v2.intraday_features import (
    _rolling_spread_from_moments,
    build_intraday_daily_features,
    detect_open_gap_boundaries,
    _rolling_summary,
    return_consistency,
)


def _minutes() -> tuple[np.ndarray, ...]:
    days, names, minutes = 25, 2, 405
    base = (
        100.0
        + np.arange(days)[:, None, None] * 0.1
        + np.arange(names)[None, :, None]
        + np.arange(minutes)[None, None, :] * 0.001
    )
    observed = np.ones(base.shape, dtype=bool)
    return (
        base,
        base * 1.001,
        base * 0.999,
        base + 0.0005,
        np.ones(base.shape),
        observed,
    )


def _scheduled_minutes() -> tuple[
    tuple[np.ndarray, ...], tuple[SessionDefinition, ...]
]:
    days, names, minutes = 26, 1, 420
    open_price = np.full((days, names, minutes), 100.0)
    close = np.full_like(open_price, 100.0)
    volume = np.ones_like(open_price)
    observed = np.ones(open_price.shape, dtype=np.bool_)
    sessions = [
        SessionDefinition(
            trade_date=date(2024, 1, 1) + timedelta(days=day),
            continuous_open=time(10, 0),
            decision_time=time(15, 45),
            continuous_close=time(17, 0),
            auction_close=time(17, 15),
            source="test",
        )
        for day in range(days)
    ]
    sessions[24] = SessionDefinition(
        trade_date=sessions[24].trade_date,
        continuous_open=time(10, 30),
        decision_time=time(15, 45),
        continuous_close=time(16, 30),
        auction_close=time(17, 0),
        source="test-shifted",
    )
    open_price[24, 0, 315] = 777.0
    open_price[24, 0, 330] = 110.0
    close[24, 0, 359] = 120.0
    close[24, 0, 360:] = 999.0
    volume[24, 0, 300:360] = 2.0
    volume[24, 0, 360:] = 1_000_000.0
    high = close + 1.0
    low = close - 1.0
    return (
        open_price,
        high,
        low,
        close,
        volume,
        observed,
    ), tuple(sessions)


def _build_scheduled(
    inputs: tuple[np.ndarray, ...], sessions: tuple[SessionDefinition, ...], **kwargs
):
    observed = inputs[-1]
    return build_intraday_daily_features(
        *inputs,
        volume_valid=observed.copy(),
        session_valid=np.ones(observed.shape[:2], dtype=np.bool_),
        sessions=sessions,
        **kwargs,
    )


def _build_minutes(*inputs: np.ndarray, **kwargs):
    sessions = tuple(
        SessionDefinition(
            trade_date=date(2024, 1, 1) + timedelta(days=day),
            continuous_open=time(10, 0),
            decision_time=time(15, 45),
            continuous_close=time(16, 45),
            auction_close=time(17, 0),
            source="test",
        )
        for day in range(inputs[0].shape[0])
    )
    return _build_scheduled(inputs, sessions, **kwargs)


def _source_bars(
    sessions: tuple[SessionDefinition, ...],
    included_dates: frozenset[date],
    *,
    marked_date: date | None = None,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for day, session in enumerate(sessions):
        if session.trade_date not in included_dates:
            continue
        continuous_minutes = (
            session.continuous_close.hour * 60
            + session.continuous_close.minute
            - session.continuous_open.hour * 60
            - session.continuous_open.minute
        )
        decision_index = (
            session.decision_time.hour * 60
            + session.decision_time.minute
            - session.continuous_open.hour * 60
            - session.continuous_open.minute
        )
        start = datetime.combine(session.trade_date, session.continuous_open)
        for minute in range(continuous_minutes):
            price = 100.0 + day * 0.1 + minute * 0.001
            bar_open = price
            bar_close = price + 0.0005
            if session.trade_date == marked_date and minute == decision_index:
                bar_open = 777.0
                bar_close = 777.0
            if session.trade_date == marked_date and minute == continuous_minutes - 1:
                bar_open = 123.45
                bar_close = 123.45
            rows.append(
                {
                    "symbol": "TEST3",
                    "ts_exchange": start + timedelta(minutes=minute),
                    "open": bar_open,
                    "high": max(bar_open, bar_close) + 0.01,
                    "low": min(bar_open, bar_close) - 0.01,
                    "close": bar_close,
                    "real_volume": 1.0,
                }
            )
        for timestamp in (
            start - timedelta(minutes=1),
            datetime.combine(session.trade_date, session.continuous_close),
        ):
            rows.append(
                {
                    "symbol": "TEST3",
                    "ts_exchange": timestamp,
                    "open": 999.0,
                    "high": 1_000.0,
                    "low": 998.0,
                    "close": 999.0,
                    "real_volume": 1.0,
                }
            )
    return pl.DataFrame(rows)


def test_scheduled_intraday_uses_shifted_prefix_and_continuous_close() -> None:
    inputs, sessions = _scheduled_minutes()
    result = _build_scheduled(inputs, sessions)

    assert result.decision_mark[24, 0] == 100.0
    assert result.decision_mark_valid[24, 0]
    assert result.session_close[24, 0] == 120.0
    assert result.session_close[24, 0] != inputs[3][24, 0, -1]
    np.testing.assert_allclose(result.values[25, 0, 0], np.log(100.0 / 120.0))
    expected_last30_share = np.log(120.0 / 110.0) / np.log(120.0 / 100.0)
    np.testing.assert_allclose(result.values[25, 0, 8], expected_last30_share)
    np.testing.assert_allclose(result.values[25, 0, 9], 120.0 / 420.0)
    continuous_vwap = 42_040.0 / 420.0
    np.testing.assert_allclose(
        result.values[25, 0, 10], np.log(120.0 / continuous_vwap)
    )


def test_scheduled_intraday_decision_state_ignores_decision_and_later_rows() -> None:
    inputs, sessions = _scheduled_minutes()
    baseline = _build_scheduled(inputs, sessions)
    changed = [value.copy() for value in inputs]
    for index in range(5):
        changed[index][25, :, 345:] = np.nan
    changed[-1][25, :, 345:] = False

    actual = _build_scheduled(tuple(changed), sessions)

    np.testing.assert_array_equal(actual.values[25], baseline.values[25])
    np.testing.assert_array_equal(actual.valid[25], baseline.valid[25])
    np.testing.assert_array_equal(
        actual.realized_daily_vol[25], baseline.realized_daily_vol[25]
    )
    np.testing.assert_array_equal(actual.fast_present[25], baseline.fast_present[25])
    np.testing.assert_array_equal(actual.decision_mark[25], baseline.decision_mark[25])
    assert not actual.session_close_valid[25, 0]


def test_scheduled_intraday_keeps_price_activity_and_source_masks_independent() -> None:
    inputs, sessions = _scheduled_minutes()
    changed = [value.copy() for value in inputs]
    observed = changed[-1]
    volume_valid = np.ones(observed.shape, dtype=np.bool_)
    session_valid = np.ones(observed.shape[:2], dtype=np.bool_)

    # The price record remains usable while one activity cell is unknown.
    volume_valid[24, 0, 10] = False
    changed[4][24, 0, 10] = np.nan
    first = build_intraday_daily_features(
        *changed,
        volume_valid=volume_valid,
        session_valid=session_valid,
        sessions=sessions,
    )
    changed[4][24, 0, 10] = 1e30
    second = build_intraday_daily_features(
        *changed,
        volume_valid=volume_valid,
        session_valid=session_valid,
        sessions=sessions,
    )
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.valid, second.valid)
    assert first.valid[24, 0, 1]
    assert first.fast_present[24, 0]
    assert not first.valid[24, 0, 11]
    assert not first.valid[24, 0, 19]

    # On a supported session, an explicitly valid no-trade minute may have no
    # price without contaminating VWAP: its exact zero carries zero weight.
    no_trade = [value.copy() for value in inputs]
    no_trade[-1][24, 0, 10] = False
    for price in no_trade[:4]:
        price[24, 0, 10] = np.nan
    no_trade[4][24, 0, 10] = 0.0
    valid_no_trade = build_intraday_daily_features(
        *no_trade,
        volume_valid=np.ones(observed.shape, dtype=np.bool_),
        session_valid=session_valid,
        sessions=sessions,
    )
    for price in no_trade[:4]:
        price[24, 0, 10] = 1e30
    mutated_no_trade = build_intraday_daily_features(
        *no_trade,
        volume_valid=np.ones(observed.shape, dtype=np.bool_),
        session_valid=session_valid,
        sessions=sessions,
    )
    assert valid_no_trade.valid[24, 0, 11]
    np.testing.assert_array_equal(valid_no_trade.values, mutated_no_trade.values)
    np.testing.assert_array_equal(valid_no_trade.valid, mutated_no_trade.valid)

    # A fully absent source session is unknown, never an inferred no-trade day.
    source_gap = [value.copy() for value in inputs]
    source_gap[-1][24] = False
    gap_volume_valid = np.ones(source_gap[-1].shape, dtype=np.bool_)
    gap_volume_valid[24] = False
    gap_session_valid = session_valid.copy()
    gap_session_valid[24] = False
    gap = build_intraday_daily_features(
        *source_gap,
        volume_valid=gap_volume_valid,
        session_valid=gap_session_valid,
        sessions=sessions,
    )
    # Lag-one features 9/10 may still carry a known prior session; every
    # feature that consumes the missing current source row is invalid.
    assert not gap.valid[24, 0, [0, 1, 6, 11, 12, 18, 19]].any()
    assert not gap.fast_present[24, 0]


def test_scheduled_intraday_rejects_malformed_valid_activity() -> None:
    inputs, sessions = _scheduled_minutes()
    changed = [value.copy() for value in inputs]
    changed[4][24, 0, 10] = np.nan
    with np.testing.assert_raises_regex(ValueError, "activity must be finite"):
        _build_scheduled(tuple(changed), sessions)


def test_entry_bar_does_not_control_fast_presence() -> None:
    inputs = list(_minutes())
    inputs[-1][24, 0, 345] = False
    result = _build_minutes(*inputs)
    # Scheduled summaries carry the last completed decision mark here.
    assert result.decision_mark_valid[24, 0]
    assert result.fast_present[24, 0]


def _official_returns(inputs):
    final_close = inputs[3][..., -1]
    result = np.full(final_close.shape, np.nan)
    result[1:] = np.log(final_close[1:] / final_close[:-1])
    return result


def test_return_consistency_accepts_different_levels_and_rejects_return_jump():
    close = np.asarray([[100.0], [101.0], [102.0], [103.0]])
    official_return = np.full(close.shape, np.nan)
    official_return[1:] = np.log(close[1:] / close[:-1])
    observed = np.ones(close.shape, dtype=bool)
    for scale in (0.03, 1.0, 19.0):
        result = return_consistency(close * scale, observed, official_return)
        np.testing.assert_array_equal(result[:, 0], [False, True, True, True])
    official_return[2] += 0.006
    assert not return_consistency(close, observed, official_return)[2, 0]
    observed[1] = False
    assert not return_consistency(close, observed, official_return)[1:3].any()


def test_m1_internal_scalars_keep_units_and_validate_only_completed_history():
    inputs = _minutes()
    reference = _official_returns(inputs)
    original = _build_minutes(*inputs, official_log_return=reference)
    scaled = tuple(x * 0.07 if i < 4 else x.copy() for i, x in enumerate(inputs))
    result = _build_minutes(*scaled, official_log_return=reference)
    np.testing.assert_array_equal(result.valid, original.valid)
    np.testing.assert_allclose(result.values, original.values, atol=1e-7)
    expected = np.log(inputs[0][-1, 0, 0] / inputs[3][-2, 0, -1])
    assert result.values[-1, 0, 0] == np.float32(expected)
    assert result.session_close[-1, 0] == scaled[3][-1, 0, -1]


def test_current_close_and_inferred_action_cannot_change_current_decision():
    inputs = _minutes()
    reference = _official_returns(inputs)
    boundaries = np.zeros(reference.shape, dtype=bool)
    original = _build_minutes(
        *inputs, official_log_return=reference, completed_action_boundary=boundaries
    )
    mutated = tuple(x.copy() for x in inputs)
    for value in mutated[:4]:
        value[23, :, 345:] *= 0.92
    changed_reference = reference.copy()
    changed_reference[23] += 0.02
    boundaries[23] = True
    result = _build_minutes(
        *mutated,
        official_log_return=changed_reference,
        completed_action_boundary=boundaries,
    )
    for field in ("values", "valid", "support_fraction", "source_age_sessions"):
        np.testing.assert_array_equal(
            getattr(result, field)[23], getattr(original, field)[23]
        )
    assert not result.valid[24, :, (0, 6, 8, 9, 10)].any()
    assert result.valid[23, :, 1].all()
    assert not result.return_consistent[23].any()


def test_open_known_boundary_excludes_observation_without_destroying_rolling_window():
    inputs = _minutes()
    known = np.zeros(inputs[0].shape[:2], dtype=bool)
    known[23] = True
    result = _build_minutes(*inputs, same_day_boundary=known)
    assert not result.valid[23, :, (0, 6)].any()
    assert result.valid[23, :, 2].all()
    np.testing.assert_array_equal(result.support_fraction[23, :, 2], np.float32(0.8))
    np.testing.assert_array_equal(result.source_age_sessions[23, :, 2], 1.0)


def test_rolling_support_counts_observations_without_imputation():
    values = np.arange(1, 7, dtype=float)[:, None]
    valid = np.ones(values.shape, dtype=bool)
    valid[4] = False
    sums, accepted, coverage, age = _rolling_summary(values, valid, 5, total=True)
    assert accepted[4, 0] and sums[4, 0] == 10.0
    assert coverage[4, 0] == np.float32(0.8) and age[4, 0] == 1
    valid[3] = False
    assert not _rolling_summary(values, valid, 5)[1][4, 0]
    # Historical close knowledge can reject yesterday, never today's sample.
    valid[:] = True
    completed = valid.copy()
    completed[4] = False
    assert _rolling_summary(values, valid, 5, completed_valid=completed)[2][4, 0] == 1
    assert _rolling_summary(values, valid, 5, completed_valid=completed)[2][
        5, 0
    ] == np.float32(0.8)


def test_open_gap_boundary_is_decision_known_and_close_t_invariant():
    open_ = np.asarray([[100.0], [50.0], [51.0]])
    close = np.asarray([[100.0], [52.0], [53.0]])
    observed = np.ones(close.shape, dtype=bool)
    original = detect_open_gap_boundaries(open_, close, observed)
    close[1] = 500.0
    assert original[1, 0]
    assert detect_open_gap_boundaries(open_, close, observed)[1, 0] == original[1, 0]


def test_fast_presence_ignores_every_entry_bar_field() -> None:
    inputs = _minutes()
    original = _build_minutes(*inputs)
    changed = [value.copy() for value in inputs]
    for index in range(5):
        changed[index][24, :, 345] = np.nan
    changed[-1][24, :, 345] = False
    mutated = _build_minutes(*changed)
    np.testing.assert_array_equal(original.fast_present[24], mutated.fast_present[24])


def test_decision_features_exclude_every_entry_and_later_bar_field() -> None:
    inputs = _minutes()
    original = _build_minutes(*inputs)
    changed = [value.copy() for value in inputs]
    # No field from the decision bar at index 345 or later enters this snapshot.
    for index in range(5):
        changed[index][24, :, 345:] *= 10_000.0
    mutated = _build_minutes(*changed)
    np.testing.assert_array_equal(original.values[24], mutated.values[24])
    np.testing.assert_array_equal(original.valid[24], mutated.valid[24])
    np.testing.assert_array_equal(original.decision_mark[24], mutated.decision_mark[24])
    assert not np.array_equal(original.session_close[24], mutated.session_close[24])


def test_exact_final_m1_close_has_an_independent_observation_mask() -> None:
    inputs = list(_minutes())
    inputs[3][24, 0, -1] = 123.45
    inputs[-1][24, 1, -1] = False
    result = _build_minutes(*inputs)
    assert result.session_close[24, 0] == 123.45
    assert result.session_close_valid[24, 0]
    assert np.isnan(result.session_close[24, 1])
    assert not result.session_close_valid[24, 1]


def test_roll_spread_uses_negative_sample_covariance_and_masks_nonnegative() -> None:
    returns = np.asarray([[[0.01, -0.01, 0.01, -0.01]]])

    def estimate(values: np.ndarray):
        left, right = values[..., :-1], values[..., 1:]
        return _rolling_spread_from_moments(
            np.full(values.shape[:2], values.shape[-1] - 1, dtype=np.int64),
            left.sum(axis=2),
            right.sum(axis=2),
            (left * right).sum(axis=2),
            np.full(values.shape[0], values.shape[-1] - 1, dtype=np.int64),
            1,
        )

    spread, mask = estimate(returns)
    left = returns[0, 0, :-1]
    right = returns[0, 0, 1:]
    expected = 2.0 * np.sqrt(-np.cov(left, right, ddof=1)[0, 1])
    assert mask[0, 0]
    assert spread[0, 0] == expected

    increasing = np.asarray([[[0.01, 0.02, 0.03, 0.04]]])
    spread, mask = estimate(increasing)
    assert not mask[0, 0]
    assert np.isnan(spread[0, 0])


def test_streamed_intraday_carries_exact_observed_final_m1_close(
    tmp_path, monkeypatch
) -> None:
    dates = [date(2024, 1, 1) + timedelta(days=index) for index in range(25)]
    isin = "BRTESTACNOR1"
    isins = ["BRNOM1ACNOR0", isin]
    sessions = list(
        SessionDefinition(
            trade_date=value,
            continuous_open=time(10, 0),
            decision_time=time(15, 45),
            continuous_close=time(17, 0),
            auction_close=time(17, 15),
            source="test",
        )
        for value in dates
    )
    sessions[-1] = SessionDefinition(
        trade_date=dates[-1],
        continuous_open=time(10, 30),
        decision_time=time(15, 45),
        continuous_close=time(16, 30),
        auction_close=time(17, 0),
        source="test-shifted",
    )
    schedule = tuple(sessions)
    source_path = tmp_path / "source.parquet"
    assignments = pl.DataFrame(
        {
            "security_id": ["SEC_TEST"],
            "isin": [isin],
            "source_file": [str(source_path)],
        }
    )
    daily = pl.DataFrame({"isin": [isin] * len(dates), "trade_date": dates})
    source = _source_bars(schedule, frozenset(dates), marked_date=dates[-1])

    import brazil_rv.preprocessing.io as io

    dense_grid = io.dense_grid
    grid_calls: list[tuple[pl.DataFrame, int, int]] = []

    def capture_grid(bars, date_count, minute_count):
        grid_calls.append((bars.clone(), date_count, minute_count))
        return dense_grid(bars, date_count, minute_count)

    source.write_parquet(source_path)
    monkeypatch.setattr(io, "dense_grid", capture_grid)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kept_rows = np.asarray([23, 24], dtype=np.int64)
    result = stream_intraday_from_assignments(
        assignments,
        daily,
        schedule,
        isins,
        sigma_asof=np.full((len(dates), len(isins)), 0.02),
        kept_rows=kept_rows,
        workspace=workspace,
    )

    assert len(grid_calls) == 1
    gridded_bars, date_count, minute_count = grid_calls[0]
    assert (date_count, minute_count) == (25, 420)
    shifted = gridded_bars.filter(pl.col("trade_date") == dates[-1])
    assert shifted.height == 360
    assert shifted.get_column("minute_idx").min() == 0
    assert shifted.get_column("minute_idx").max() == 359
    expected_mark = 100.0 + 24 * 0.1 + 314 * 0.001 + 0.0005
    np.testing.assert_allclose(result.result.decision_mark[24, 1], expected_mark)
    np.testing.assert_allclose(result.to_close_entry[24, 1], 777.0)
    assert result.result.decision_mark[24, 1] != result.to_close_entry[24, 1]
    assert result.result.session_close[24, 1] == np.float32(123.45)
    assert result.result.session_close_valid[24, 1]
    assert not result.to_close_entry_valid[:, 0].any()

    assert result.native_mapping.to_dicts() == [
        {
            "fast_index": 0,
            "store_name_index": 1,
            "isin": isin,
            "security_id": "SEC_TEST",
        }
    ]
    native = result.native_arrays
    assert native["fast_patch_values"].shape == (2, 1, 69, 7)
    np.testing.assert_array_equal(native["fast_patch_mask"][:, 0].sum(axis=1), [69, 63])
    assert native["fast_patch_valid"][1, 0, :63].any()
    assert native["fast_last_price_age_valid"][1, 0, :63].all()
    assert not native["fast_patch_mask"][1, 0, 63:].any()


def test_streamed_intraday_grids_only_the_assignment_date_span(
    tmp_path, monkeypatch
) -> None:
    dates = [date(2024, 1, 1) + timedelta(days=index) for index in range(80)]
    allowed_dates = dates[27:52]
    isin = "BRTESTACNOR1"
    sessions = [
        SessionDefinition(
            trade_date=value,
            continuous_open=time(10, 0),
            decision_time=time(15, 45),
            continuous_close=time(17, 0),
            auction_close=time(17, 15),
            source="test",
        )
        for value in dates
    ]
    sessions[30] = SessionDefinition(
        trade_date=dates[30],
        continuous_open=time(10, 30),
        decision_time=time(15, 45),
        continuous_close=time(16, 30),
        auction_close=time(17, 0),
        source="test-shifted",
    )
    schedule = tuple(sessions)
    source_path = tmp_path / "source.parquet"
    assignments = pl.DataFrame(
        {
            "security_id": ["SEC_TEST"],
            "isin": [isin],
            "source_file": [str(source_path)],
        }
    )
    daily = pl.DataFrame(
        {"isin": [isin] * len(allowed_dates), "trade_date": allowed_dates}
    )
    source = _source_bars(schedule, frozenset(allowed_dates))
    grid_calls: list[tuple[pl.DataFrame, int, int]] = []

    import brazil_rv.preprocessing.io as io

    dense_grid = io.dense_grid

    def capture_grid(bars, date_count, minute_count):
        grid_calls.append((bars.clone(), date_count, minute_count))
        return dense_grid(bars, date_count, minute_count)

    source.write_parquet(source_path)
    monkeypatch.setattr(io, "dense_grid", capture_grid)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kept_rows = np.asarray([27, 30, 51], dtype=np.int64)
    result = stream_intraday_from_assignments(
        assignments,
        daily,
        schedule,
        [isin],
        sigma_asof=np.full((len(dates), 1), 0.02),
        kept_rows=kept_rows,
        workspace=workspace,
    )

    assert len(grid_calls) == 1
    bars, date_count, minute_count = grid_calls[0]
    assert (date_count, minute_count) == (45, 420)
    assert set(bars.get_column("trade_date")) == set(allowed_dates)
    assert set(bars.get_column("date_idx")) == set(range(20, 45))
    shifted = bars.filter(pl.col("trade_date") == dates[30])
    assert shifted.height == 360
    assert shifted.get_column("minute_idx").min() == 0
    assert shifted.get_column("minute_idx").max() == 359
    assert not result.result.fast_present[:27].any()
    assert result.result.fast_present[27:52].all()
    assert not result.result.fast_present[52:].any()
    assert result.to_close_entry_valid[27:52].all()
    assert not result.to_close_entry_valid[:27].any()
    assert not result.to_close_entry_valid[52:].any()
    assert np.all(result.result.decision_mark[27:52] != result.to_close_entry[27:52])
    np.testing.assert_array_equal(
        result.native_arrays["fast_patch_mask"][:, 0].sum(axis=1),
        [69, 63, 69],
    )
    assert result.native_mapping.get_column("security_id").to_list() == ["SEC_TEST"]
