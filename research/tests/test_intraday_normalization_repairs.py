from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from brazil_rv.modeling.contract import VALIDATION_END
from brazil_rv.preprocessing.contract import (
    EQUITY_SESSION_MINUTES,
    PRICE_FEATURE_CLIP,
)
from brazil_rv.preprocessing.io import SOURCE_COLUMNS
import brazil_rv.preprocessing.intraday_normalization as normalization
import brazil_rv.preprocessing.intraday_normalization_variants as variant_builder
from brazil_rv.preprocessing.intraday_normalization import (
    AFFECTED_DYNAMIC_CHANNELS,
    INVARIANT_DYNAMIC_CHANNELS,
    PROFILE_BIN_COUNT,
    ReconstructedEquity,
    _development_array_identity,
    build_seasonal_dynamic_features,
    equity_source_hashes,
)
from brazil_rv.preprocessing.intraday_normalization_variants import (
    _populate_raw_channels,
)
from brazil_rv.preprocessing.transforms import (
    add_equity_cross_sectional_dynamic,
    build_dynamic_features,
    centered_midranks,
)


def _raw_path(increments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    date_count, minute_count = increments.shape
    close = np.exp(np.cumsum(increments, axis=1))
    previous = np.concatenate((np.ones((date_count, 1)), close[:, :-1]), axis=1)
    raw = np.zeros((date_count, minute_count, 5), dtype=np.float64)
    raw[..., 0] = previous
    raw[..., 1] = np.maximum(previous, close) * np.exp(0.0002)
    raw[..., 2] = np.minimum(previous, close) * np.exp(-0.0002)
    raw[..., 3] = close
    raw[..., 4] = 100.0
    return raw, np.ones((date_count, minute_count), dtype=bool)


def test_real_populate_raw_channels_passes_and_respects_data_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    date_count = 2
    equity_count = 2
    parent = tmp_path / "parent"
    parent.mkdir()
    parent_ready = np.zeros((date_count, equity_count), dtype=bool)
    parent_ready[1] = True
    parent_dynamic = np.zeros(
        (date_count, equity_count, EQUITY_SESSION_MINUTES, 26), dtype=np.float32
    )
    equities: list[ReconstructedEquity] = []
    for slot in range(equity_count):
        raw, observed = _raw_path(
            np.full((date_count, EQUITY_SESSION_MINUTES), 0.001 * (slot + 1))
        )
        sigma = np.full(date_count, 0.002)
        dynamic, dynamic_valid = build_dynamic_features(
            raw,
            observed,
            parent_ready[:, slot],
            sigma,
            is_rate=False,
            first_observed_open=True,
        )
        parent_dynamic[:, slot] = dynamic
        equities.append(
            ReconstructedEquity(
                slot,
                f"security-{slot}",
                tmp_path / f"source-{slot}.parquet",
                raw,
                observed,
                dynamic,
                dynamic_valid,
                sigma,
                parent_ready[:, slot],
            )
        )
    np.save(parent / "equity_data_ready.npy", parent_ready, allow_pickle=False)
    np.save(parent / "equity_features.npy", parent_dynamic, allow_pickle=False)
    context = SimpleNamespace(parent=parent, allowed_date_count=date_count)
    overlays = {
        arm: np.zeros(
            (
                date_count,
                equity_count,
                variant_builder.VISIBLE_EQUITY_MINUTES,
                len(AFFECTED_DYNAMIC_CHANNELS),
            ),
            dtype=np.float32,
        )
        for arm in ("equity_tod_half", "equity_tod_full")
    }
    relative_variance = np.ones((date_count, PROFILE_BIN_COUNT), dtype=np.float64)
    relative_variance[:, 0] = 4.0
    monkeypatch.setattr(variant_builder, "EXPECTED_EQUITIES", equity_count)
    monkeypatch.setattr(
        variant_builder, "iter_reconstructed_equities", lambda _context: iter(equities)
    )

    _populate_raw_channels(context, relative_variance, overlays)

    raw_channels = tuple(
        channel for channel in AFFECTED_DYNAMIC_CHANNELS if channel < 16
    )
    for arm, gamma in (("equity_tod_half", 0.5), ("equity_tod_full", 1.0)):
        for equity in equities:
            expected, _ = build_seasonal_dynamic_features(
                equity.raw_grid,
                equity.observed,
                equity.data_ready,
                equity.sigma,
                relative_variance,
                gamma,
            )
            actual = overlays[arm][:, equity.slot]
            for source_channel in raw_channels:
                destination = AFFECTED_DYNAMIC_CHANNELS.index(source_channel)
                assert np.array_equal(
                    actual[..., destination],
                    expected[
                        :, : variant_builder.VISIBLE_EQUITY_MINUTES, source_channel
                    ],
                )
            assert not actual[0].any()
            assert actual[1].any()


def test_return_ranks_rebuild_when_seasonal_scaling_removes_clip_ties() -> None:
    equity_count = 30
    unclipped = np.linspace(
        0.75 * PRICE_FEATURE_CLIP,
        1.25 * PRICE_FEATURE_CLIP,
        equity_count,
        dtype=np.float32,
    )
    legacy = np.zeros((equity_count, 1, 26), dtype=np.float32)
    candidate = legacy.copy()
    for channel in (7, 9):
        legacy[:, 0, channel] = np.clip(
            unclipped, -PRICE_FEATURE_CLIP, PRICE_FEATURE_CLIP
        )
        candidate[:, 0, channel] = np.clip(
            unclipped / 2.0, -PRICE_FEATURE_CLIP, PRICE_FEATURE_CLIP
        )
    validity = np.ones((equity_count, 1, 4), dtype=bool)
    active = np.ones(equity_count, dtype=bool)
    gamma_zero = legacy.copy()

    add_equity_cross_sectional_dynamic(legacy, validity, active)
    add_equity_cross_sectional_dynamic(candidate, validity, active)
    add_equity_cross_sectional_dynamic(gamma_zero, validity, active)

    assert np.array_equal(gamma_zero, legacy)
    for source, rank in ((7, 22), (9, 23)):
        assert np.array_equal(
            candidate[:, 0, rank], centered_midranks(candidate[:, 0, source])
        )
        assert not np.array_equal(candidate[:, 0, rank], legacy[:, 0, rank])
    assert np.array_equal(candidate[..., 18:20], legacy[..., 18:20])
    assert tuple(sorted(AFFECTED_DYNAMIC_CHANNELS)) == AFFECTED_DYNAMIC_CHANNELS
    assert {22, 23} <= set(AFFECTED_DYNAMIC_CHANNELS)
    assert {18, 19} <= set(INVARIANT_DYNAMIC_CHANNELS)


def test_development_array_identity_ignores_held_out_tail(tmp_path: Path) -> None:
    path = tmp_path / "date_bearing.npy"
    values = np.arange(6, dtype=np.float32).reshape(3, 2)
    np.save(path, values, allow_pickle=False)
    expected = (2, 2)
    baseline = _development_array_identity(path, np.dtype(np.float32), expected)
    values[2, 0] = 900.0
    np.save(path, values, allow_pickle=False)
    assert _development_array_identity(path, np.dtype(np.float32), expected) == baseline
    values[0, 0] = -1.0
    np.save(path, values, allow_pickle=False)
    assert _development_array_identity(path, np.dtype(np.float32), expected) != baseline


def test_raw_source_identity_hashes_only_canonical_development_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "source.parquet"

    def write(dev_close: float, tail_close: float, *, extra_tail: bool = False) -> None:
        timestamps = [
            datetime(2025, 6, 29, 13, 0),
            datetime(2025, 6, 30, 13, 0),
            datetime(2025, 7, 1, 13, 0),
        ]
        closes = [10.0, dev_close, tail_close]
        if extra_tail:
            timestamps.append(datetime(2025, 7, 2, 13, 0))
            closes.append(999.0)
        pl.DataFrame(
            {
                "ts_exchange": timestamps,
                "open": closes,
                "high": [value + 1.0 for value in closes],
                "low": [value - 1.0 for value in closes],
                "close": closes,
                "real_volume": [100.0] * len(closes),
                "symbol": ["TEST3"] * len(closes),
            }
        ).select(SOURCE_COLUMNS).write_parquet(path)

    context = SimpleNamespace(
        assignments=pl.DataFrame({"source_file": [str(path)]}),
        market_dates=(date(2025, 6, 29), date(2025, 6, 30)),
    )
    monkeypatch.setattr(
        normalization,
        "sha256_file",
        lambda _path: pytest.fail("complete source-file hashing is forbidden"),
    )
    write(11.0, 12.0)
    baseline = equity_source_hashes(context)
    write(11.0, 500.0, extra_tail=True)
    assert equity_source_hashes(context) == baseline
    write(11.5, 500.0, extra_tail=True)
    assert equity_source_hashes(context) != baseline


def test_identity_scope_ends_at_validation() -> None:
    assert str(VALIDATION_END) == "2025-06-30"
