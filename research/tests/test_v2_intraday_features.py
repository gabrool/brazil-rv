from datetime import date, timedelta

import numpy as np
import polars as pl

from brazil_rv.v2.build_store import stream_intraday_from_assignments
from brazil_rv.v2.intraday_features import (
    _rolling_roll_spread,
    build_intraday_daily_features,
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


def test_missing_entry_bar_disables_fast_presence() -> None:
    inputs = list(_minutes())
    inputs[-1][24, 0, 345] = False
    result = build_intraday_daily_features(*inputs)
    assert not result.entry_open_valid[24, 0]
    assert not result.fast_present[24, 0]


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
    assert not masked.valid[24, 0, 19]
    assert masked.valid[24, 1, 0]
    assert masked.valid[24, 0, 1]
    assert masked.valid[24, 0, 4]
    assert masked.valid[24, 0, 18]
    assert np.all(masked.values[~masked.valid] == 0.0)


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
    source_path = tmp_path / "source.parquet"
    source_path.write_bytes(b"immutable-source")
    assignments = pl.DataFrame(
        {"isin": [isin], "source_file": [str(source_path)]}
    )
    daily = pl.DataFrame(
        {"isin": [isin] * len(dates), "trade_date": dates}
    )
    grid = np.zeros((len(dates), 405, 5), dtype=np.float64)
    base = 100.0 + np.arange(405) * 0.01
    grid[..., 0] = base
    grid[..., 1] = base * 1.001
    grid[..., 2] = base * 0.999
    grid[..., 3] = base * 1.0001
    grid[..., 4] = 1.0
    grid[24, -1, 3] = 123.45
    observed = np.ones((len(dates), 405), dtype=bool)
    observed[23, -1] = False

    import brazil_rv.preprocessing.io as io

    monkeypatch.setattr(io, "load_source_file", lambda path: object())
    monkeypatch.setattr(io, "prepare_session_bars", lambda *args: object())
    monkeypatch.setattr(io, "dense_grid", lambda *args: (grid, observed))
    result = stream_intraday_from_assignments(
        assignments, daily, dates, [isin]
    )
    assert result.result.session_close[24, 0] == 123.45
    assert result.result.session_close_valid[24, 0]
    assert np.isnan(result.result.session_close[23, 0])
    assert not result.result.session_close_valid[23, 0]


def test_streamed_intraday_grids_only_the_assignment_date_span(
    tmp_path, monkeypatch
) -> None:
    dates = [date(2024, 1, 1) + timedelta(days=index) for index in range(80)]
    allowed_dates = dates[27:52]
    isin = "BRTESTACNOR1"
    source_path = tmp_path / "source.parquet"
    source_path.write_bytes(b"immutable-source")
    assignments = pl.DataFrame({"isin": [isin], "source_file": [str(source_path)]})
    daily = pl.DataFrame(
        {"isin": [isin] * len(allowed_dates), "trade_date": allowed_dates}
    )
    seen_calendars: list[tuple[date, ...]] = []
    local_grids: list[tuple[np.ndarray, np.ndarray]] = []

    import brazil_rv.preprocessing.io as io

    monkeypatch.setattr(io, "load_source_file", lambda path: object())

    def prepare(*args):
        seen_calendars.append(args[3])
        return object()

    def grid(_bars, date_count, minute_count):
        values = np.zeros((date_count, minute_count, 5), dtype=np.float64)
        base = 100.0 + np.arange(minute_count) * 0.01
        values[..., 0] = base
        values[..., 1] = base * 1.001
        values[..., 2] = base * 0.999
        values[..., 3] = base * 1.0001
        values[..., 4] = 1.0
        session_observed = np.asarray(
            [value in frozenset(allowed_dates) for value in seen_calendars[-1]]
        )
        values[~session_observed] = 0.0
        observed = np.broadcast_to(
            session_observed[:, None], (date_count, minute_count)
        ).copy()
        local_grids.append((values, observed))
        return values, observed

    monkeypatch.setattr(io, "prepare_session_bars", prepare)
    monkeypatch.setattr(io, "dense_grid", grid)
    result = stream_intraday_from_assignments(assignments, daily, dates, [isin])
    local_grid, local_observed = local_grids[0]
    full_grid = np.zeros((len(dates), 405, 5), dtype=np.float64)
    full_observed = np.zeros((len(dates), 405), dtype=bool)
    full_grid[7:72] = local_grid
    full_observed[7:72] = local_observed
    expected = build_intraday_daily_features(
        full_grid[:, None, :, 0],
        full_grid[:, None, :, 1],
        full_grid[:, None, :, 2],
        full_grid[:, None, :, 3],
        full_grid[:, None, :, 4],
        full_observed[:, None, :],
    )
    assert seen_calendars == [tuple(dates[7:72])]
    np.testing.assert_array_equal(result.result.values, expected.values)
    np.testing.assert_array_equal(result.result.valid, expected.valid)
    np.testing.assert_equal(result.result.entry_open, expected.entry_open)
    np.testing.assert_array_equal(
        result.result.entry_open_valid, expected.entry_open_valid
    )
    np.testing.assert_equal(result.result.session_close, expected.session_close)
    np.testing.assert_array_equal(
        result.result.session_close_valid, expected.session_close_valid
    )
    np.testing.assert_equal(
        result.result.realized_daily_vol, expected.realized_daily_vol
    )
    np.testing.assert_array_equal(result.result.fast_present, expected.fast_present)
    assert not result.result.fast_present[:27].any()
    assert result.result.fast_present[27:52].all()
    assert not result.result.fast_present[52:].any()
    assert not result.result.entry_open[:27].any()
    assert not result.result.entry_open[52:].any()
