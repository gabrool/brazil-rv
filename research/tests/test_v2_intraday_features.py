from datetime import date, datetime, time, timedelta

import numpy as np
import polars as pl

from brazil_rv.v2.build_store import stream_intraday_from_assignments
from brazil_rv.v2.decision_clock import SessionDefinition
from brazil_rv.v2.intraday_features import (
    _rolling_roll_spread,
    build_intraday_daily_features,
    detect_open_gap_boundaries,
    five_minute_returns,
    mask_action_boundaries,
    replace_daily_close_anchors,
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
    return base, base * 1.001, base * 0.999, base + 0.0005, np.ones(base.shape), observed


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


def test_intraday_features_use_completed_bars_before_cutoff_only() -> None:
    inputs = _minutes()
    original = build_intraday_daily_features(*inputs)
    changed = [value.copy() for value in inputs]
    for index in range(4):
        changed[index][24, :, 346:] *= 10.0
    changed[4][24, :, 346:] *= 1_000.0
    mutated = build_intraday_daily_features(*changed)
    np.testing.assert_array_equal(original.values[24], mutated.values[24])
    np.testing.assert_array_equal(original.valid[24], mutated.valid[24])
    np.testing.assert_array_equal(original.entry_open[24], mutated.entry_open[24])


def test_scheduled_intraday_uses_shifted_prefix_and_continuous_close() -> None:
    inputs, sessions = _scheduled_minutes()
    result = build_intraday_daily_features(*inputs, sessions=sessions)

    assert result.entry_open[24, 0] == 100.0
    assert result.entry_open_valid[24, 0]
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
    baseline = build_intraday_daily_features(*inputs, sessions=sessions)
    changed = [value.copy() for value in inputs]
    for index in range(5):
        changed[index][25, :, 345:] = np.nan
    changed[-1][25, :, 345:] = False

    actual = build_intraday_daily_features(*changed, sessions=sessions)

    np.testing.assert_array_equal(actual.values[25], baseline.values[25])
    np.testing.assert_array_equal(actual.valid[25], baseline.valid[25])
    np.testing.assert_array_equal(
        actual.realized_daily_vol[25], baseline.realized_daily_vol[25]
    )
    np.testing.assert_array_equal(actual.fast_present[25], baseline.fast_present[25])
    np.testing.assert_array_equal(actual.entry_open[25], baseline.entry_open[25])
    assert not actual.session_close_valid[25, 0]


def test_entry_bar_does_not_control_fast_presence() -> None:
    inputs = list(_minutes())
    inputs[-1][24, 0, 345] = False
    result = build_intraday_daily_features(*inputs)
    assert not result.entry_open_valid[24, 0]
    assert result.fast_present[24, 0]


def test_action_boundaries_mask_overnight_and_exact_rolling_dependants() -> None:
    result = build_intraday_daily_features(*_minutes())
    boundaries = np.zeros(result.values.shape[:2], dtype=np.bool_)
    boundaries[20, 0] = True

    masked = mask_action_boundaries(result, boundaries)

    assert not masked.valid[20, 0, 0]
    assert not masked.valid[20, 0, 6]
    assert not masked.valid[24, 0, 2]
    assert not masked.valid[24, 0, 3]
    assert not masked.valid[24, 0, 7]
    assert not masked.valid[24, 0, 17]
    assert masked.valid[24, 0, 19]
    assert masked.valid[24, 1, 0]
    assert masked.valid[24, 0, 1]
    assert masked.valid[24, 0, 4]
    assert masked.valid[24, 0, 18]
    assert np.all(masked.values[~masked.valid] == 0.0)


def test_open_gap_boundary_is_decision_known_and_close_t_invariant() -> None:
    raw_open = np.asarray([[100.0], [50.0], [51.0]])
    raw_close = np.asarray([[100.0], [52.0], [53.0]])
    observed = np.ones_like(raw_open, dtype=bool)
    expected = detect_open_gap_boundaries(raw_open, raw_close, observed)
    assert expected[:, 0].tolist() == [False, True, False]

    changed = raw_close.copy()
    changed[1, 0] = 5_200.0
    actual = detect_open_gap_boundaries(raw_open, changed, observed)
    assert actual[1, 0] == expected[1, 0]
    assert actual[2, 0] != expected[2, 0]


def test_fast_presence_ignores_every_entry_bar_field() -> None:
    inputs = _minutes()
    original = build_intraday_daily_features(*inputs)
    changed = [value.copy() for value in inputs]
    for index in range(5):
        changed[index][24, :, 345] = np.nan
    changed[-1][24, :, 345] = False
    mutated = build_intraday_daily_features(*changed)
    np.testing.assert_array_equal(original.fast_present[24], mutated.fast_present[24])


def test_five_minute_returns_are_adjacent_block_close_to_close() -> None:
    close = np.asarray([[[1.0, 1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0, 1.0, 110.0]]])
    returns, valid = five_minute_returns(
        close.copy(), close, np.ones_like(close, dtype=bool), cutoff=10
    )
    np.testing.assert_allclose(returns, np.log(1.1))
    assert valid.all()


def test_cotahist_close_replaces_full_session_anchor() -> None:
    result = build_intraday_daily_features(*_minutes())
    official = result.session_close.copy()
    official[-2, 0] *= 1.004
    replaced = replace_daily_close_anchors(
        result, official, np.ones_like(official, dtype=bool)
    )
    assert replaced.session_close[-2, 0] == official[-2, 0]
    expected_overnight = np.log(
        (result.entry_open[-1, 0] / np.exp(result.values[-1, 0, 1]))
        / official[-2, 0]
    )
    assert replaced.values[-1, 0, 0] == np.float32(expected_overnight)


def test_cotahist_close_in_place_mode_matches_copy_mode() -> None:
    copied_input = build_intraday_daily_features(*_minutes())
    in_place_input = build_intraday_daily_features(*_minutes())
    official = copied_input.session_close.copy()
    official[-2, 0] *= 1.004
    observed = np.ones_like(official, dtype=bool)
    copied = replace_daily_close_anchors(copied_input, official, observed)
    in_place_values = in_place_input.values
    in_place_valid = in_place_input.valid
    in_place = replace_daily_close_anchors(
        in_place_input, official, observed, copy_buffers=False
    )
    assert in_place.values is in_place_values
    assert in_place.valid is in_place_valid
    np.testing.assert_array_equal(in_place.values, copied.values)
    np.testing.assert_array_equal(in_place.valid, copied.valid)
    np.testing.assert_array_equal(in_place.session_close, copied.session_close)
    np.testing.assert_array_equal(
        in_place.session_close_valid, copied.session_close_valid
    )


def test_m1_cotahist_unit_mismatch_masks_cross_session_features() -> None:
    result = build_intraday_daily_features(*_minutes())
    official = result.session_close.copy()
    official[-2, 0] *= 1.006
    replaced = replace_daily_close_anchors(
        result, official, np.ones_like(official, dtype=bool)
    )

    assert not replaced.close_anchor_consistent[-2, 0]
    assert not replaced.session_close_valid[-2, 0]
    assert not replaced.valid[-1, 0, 0]
    assert not replaced.valid[-1, 0, 8]
    assert not replaced.valid[-1, 0, 10]
    assert replaced.valid[-2, 0, 1]


def test_decision_features_exclude_every_entry_and_later_bar_field() -> None:
    inputs = _minutes()
    original = build_intraday_daily_features(*inputs)
    changed = [value.copy() for value in inputs]
    # The entry bar is index 345.  Its open is the separate entry price; no
    # entry-bar H/L/C/volume or later value may enter a decision feature.
    for index in (1, 2, 3, 4):
        changed[index][24, :, 345:] *= 10_000.0
    mutated = build_intraday_daily_features(*changed)
    np.testing.assert_array_equal(original.values[24], mutated.values[24])
    np.testing.assert_array_equal(original.valid[24], mutated.valid[24])
    np.testing.assert_array_equal(original.entry_open[24], mutated.entry_open[24])
    assert not np.array_equal(original.session_close[24], mutated.session_close[24])


def test_exact_final_m1_close_has_an_independent_observation_mask() -> None:
    inputs = list(_minutes())
    inputs[3][24, 0, -1] = 123.45
    inputs[-1][24, 1, -1] = False
    result = build_intraday_daily_features(*inputs)
    assert result.session_close[24, 0] == 123.45
    assert result.session_close_valid[24, 0]
    assert np.isnan(result.session_close[24, 1])
    assert not result.session_close_valid[24, 1]


def test_volume_cleaning_matches_zero_for_unusable_observations() -> None:
    inputs = list(_minutes())
    changed = [value.copy() for value in inputs]
    normalized = [value.copy() for value in inputs]
    for minute, value in zip((10, 11, 12), (-1.0, np.nan, np.inf), strict=True):
        changed[4][24, 0, minute] = value
        normalized[4][24, 0, minute] = 0.0
    changed[4][24, 0, 13] = 1_000_000.0
    changed[5][24, 0, 13] = False
    normalized[4][24, 0, 13] = 0.0
    normalized[5][24, 0, 13] = False
    actual = build_intraday_daily_features(*changed)
    expected = build_intraday_daily_features(*normalized)
    np.testing.assert_array_equal(actual.values, expected.values)
    np.testing.assert_array_equal(actual.valid, expected.valid)


def test_roll_spread_uses_negative_sample_covariance_and_masks_nonnegative() -> None:
    returns = np.asarray([[[0.01, -0.01, 0.01, -0.01]]])
    valid = np.ones_like(returns, dtype=bool)
    spread, mask = _rolling_roll_spread(returns, valid, 1)
    left = returns[0, 0, :-1]
    right = returns[0, 0, 1:]
    expected = 2.0 * np.sqrt(-np.cov(left, right, ddof=1)[0, 1])
    assert mask[0, 0]
    assert spread[0, 0] == expected

    increasing = np.asarray([[[0.01, 0.02, 0.03, 0.04]]])
    spread, mask = _rolling_roll_spread(
        increasing, np.ones_like(increasing, dtype=bool), 1
    )
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
    source_path.write_bytes(b"immutable-source")
    assignments = pl.DataFrame(
        {
            "security_id": ["SEC_TEST"],
            "isin": [isin],
            "source_file": [str(source_path)],
        }
    )
    daily = pl.DataFrame({"isin": [isin] * len(dates), "trade_date": dates})
    source = _source_bars(
        schedule, frozenset(dates), marked_date=dates[-1]
    )

    import brazil_rv.preprocessing.io as io

    dense_grid = io.dense_grid
    grid_calls: list[tuple[pl.DataFrame, int, int]] = []

    def capture_grid(bars, date_count, minute_count):
        grid_calls.append((bars.clone(), date_count, minute_count))
        return dense_grid(bars, date_count, minute_count)

    monkeypatch.setattr(io, "load_source_file", lambda path: source)
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
    np.testing.assert_allclose(result.result.entry_open[24, 1], expected_mark)
    np.testing.assert_allclose(result.to_close_entry[24, 1], 777.0)
    assert result.result.entry_open[24, 1] != result.to_close_entry[24, 1]
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
    np.testing.assert_array_equal(
        native["fast_patch_mask"][:, 0].sum(axis=1), [69, 63]
    )
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
    source_path.write_bytes(b"immutable-source")
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

    monkeypatch.setattr(io, "load_source_file", lambda path: source)
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
    assert np.all(
        result.result.entry_open[27:52]
        != result.to_close_entry[27:52]
    )
    np.testing.assert_array_equal(
        result.native_arrays["fast_patch_mask"][:, 0].sum(axis=1),
        [69, 63, 69],
    )
    assert result.native_mapping.get_column("security_id").to_list() == [
        "SEC_TEST"
    ]
