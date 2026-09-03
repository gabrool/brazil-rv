from datetime import date, timedelta

import numpy as np
import polars as pl

from brazil_rv.v2.build_store import stream_intraday_from_assignments
from brazil_rv.v2.intraday_features import (
    _rolling_roll_spread,
    build_intraday_daily_features,
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
