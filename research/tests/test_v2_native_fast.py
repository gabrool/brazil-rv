from __future__ import annotations

from datetime import date, time, timedelta

import numpy as np

from brazil_rv.v2.decision_clock import SessionDefinition
from brazil_rv.v2.intraday_features import (
    NATIVE_FAST_FEATURES,
    build_native_fast_features,
)


def _sessions(
    count: int,
    *,
    continuous_open: time = time(10, 0),
    decision_time: time = time(10, 10),
    continuous_close: time = time(10, 20),
) -> tuple[SessionDefinition, ...]:
    return tuple(
        SessionDefinition(
            trade_date=date(2024, 1, 1) + timedelta(days=day),
            continuous_open=continuous_open,
            decision_time=decision_time,
            continuous_close=continuous_close,
            auction_close=continuous_close,
            source="test",
        )
        for day in range(count)
    )


def test_native_fast_exact_channels_and_padding() -> None:
    days, names, minutes = 21, 1, 20
    close = np.full((days, names, minutes), 100.0)
    close[..., 5:10] = 110.0
    high = np.full_like(close, 101.0)
    low = np.full_like(close, 99.0)
    high[..., 5:10] = 112.0
    low[..., 5:10] = 100.0
    volume = np.full_like(close, 2.0)
    volume[..., 5:10] = 4.0
    volume[-1, ..., :5] = 6.0
    volume[-1, ..., 5:10] = 12.0
    observed = np.ones(close.shape, dtype=np.bool_)
    volume_valid = np.ones(close.shape, dtype=np.bool_)
    session_valid = np.ones((days, names), dtype=np.bool_)
    sigma = np.full((days, names), 0.02)

    result = build_native_fast_features(
        high,
        low,
        close,
        volume,
        observed,
        volume_valid=volume_valid,
        session_valid=session_valid,
        sigma_asof=sigma,
        sessions=_sessions(days),
        max_patches=3,
    )

    assert result.values.dtype == np.float32
    assert result.values.shape == (days, names, 3, 7)
    assert result.feature_names == NATIVE_FAST_FEATURES
    np.testing.assert_array_equal(result.patch_mask[-1, 0], [True, True, False])
    assert not result.valid[-1, 0, 0, 0]
    np.testing.assert_allclose(
        result.values[-1, 0, 1, 0], np.log(1.1) / 0.01, rtol=1e-6
    )
    np.testing.assert_allclose(
        result.values[-1, 0, 1, 1], np.log(1.12) / 0.01, rtol=1e-6
    )
    np.testing.assert_allclose(result.values[-1, 0, 1, 2], 2.0 / 3.0)
    np.testing.assert_allclose(result.values[-1, 0, :2, 3], np.log(2.0))
    np.testing.assert_allclose(result.values[-1, 0, :2, 4], 1.0)
    np.testing.assert_allclose(result.values[-1, 0, :2, 5], [0.5, 1.0])
    np.testing.assert_allclose(result.values[-1, 0, :2, 6], 0.0)
    assert result.valid[-1, 0, :2].all(axis=0)[1:].all()
    assert not result.valid[-1, 0, 2].any()
    assert not result.values[-1, 0, 2].any()


def test_native_fast_does_not_bridge_endpoints_and_retains_age() -> None:
    minutes = 30
    close = np.full((1, 1, minutes), 100.0)
    high = close.copy()
    low = close.copy()
    volume = np.ones_like(close)
    observed = np.ones(close.shape, dtype=np.bool_)
    observed[..., 9] = False
    volume_valid = np.ones(close.shape, dtype=np.bool_)
    session_valid = np.ones((1, 1), dtype=np.bool_)
    sessions = _sessions(
        1, decision_time=time(10, 20), continuous_close=time(10, 30)
    )

    result = build_native_fast_features(
        high,
        low,
        close,
        volume,
        observed,
        volume_valid=volume_valid,
        session_valid=session_valid,
        sigma_asof=np.full((1, 1), 0.02),
        sessions=sessions,
    )

    np.testing.assert_array_equal(
        result.valid[0, 0, :, 0], [False, False, False, True]
    )
    assert result.valid[0, 0, 0, 2]
    assert result.values[0, 0, 0, 2] == 0.0
    assert not result.valid[0, 0, 1, 2]
    assert result.last_price_age_valid[0, 0, 1]
    assert result.last_price_age_minutes[0, 0, 1] == 1.0
    np.testing.assert_allclose(result.values[0, 0, 1, 6], 1.0 / 30.0)

    no_risk = build_native_fast_features(
        high,
        low,
        close,
        volume,
        observed,
        volume_valid=volume_valid,
        session_valid=session_valid,
        sigma_asof=np.zeros((1, 1)),
        sessions=sessions,
    )
    assert not no_risk.valid[..., :2].any()
    assert no_risk.valid[0, 0, 0, 2]
    assert no_risk.valid[0, 0, 0, 4:].all()


def test_native_fast_activity_needs_16_of_previous_20_covered_blocks() -> None:
    days, minutes = 21, 20
    close = np.full((days, 1, minutes), 100.0)
    high = close + 1.0
    low = close - 1.0
    volume = np.ones_like(close)
    observed = np.ones(close.shape, dtype=np.bool_)
    observed[-1] = False
    volume[-1, ..., :5] = 0.0
    volume_valid = np.zeros(close.shape, dtype=np.bool_)
    volume_valid[:16, ..., :5] = True
    volume_valid[-1, ..., :5] = True
    session_valid = np.ones((days, 1), dtype=np.bool_)
    kwargs = {
        "volume_valid": volume_valid,
        "session_valid": session_valid,
        "sigma_asof": np.full((days, 1), 0.02),
        "sessions": _sessions(days, decision_time=time(10, 5)),
    }

    result = build_native_fast_features(
        high, low, close, volume, observed, **kwargs
    )

    assert not result.valid[19, 0, 0, 3]
    assert result.valid[20, 0, 0, 3]
    np.testing.assert_allclose(result.values[20, 0, 0, 3], -np.log(2.0))
    assert not result.valid[20, 0, 0, :3].any()
    assert result.valid[20, 0, 0, 4:6].all()
    assert not result.valid[20, 0, 0, 6]

    only_15 = volume_valid.copy()
    only_15[15, ..., :5] = False
    insufficient = build_native_fast_features(
        high,
        low,
        close,
        volume,
        observed,
        **(kwargs | {"volume_valid": only_15}),
    )
    assert not insufficient.valid[20, 0, 0, 3]

    zero_baseline_volume = volume.copy()
    zero_baseline_volume[:20, ..., :5] = 0.0
    zero_baseline = build_native_fast_features(
        high, low, close, zero_baseline_volume, observed, **kwargs
    )
    assert not zero_baseline.valid[20, 0, 0, 3]


def test_native_fast_activity_matches_prior_blocks_by_clock_time() -> None:
    days, minutes = 21, 45
    close = np.full((days, 1, minutes), 100.0)
    high = close + 1.0
    low = close - 1.0
    volume = np.full_like(close, 10.0)
    volume[:20, ..., 30:35] = 2.0
    volume[-1, ..., :5] = 6.0
    observed = np.ones(close.shape, dtype=np.bool_)
    volume_valid = np.ones(close.shape, dtype=np.bool_)
    sessions = list(
        _sessions(
            20,
            decision_time=time(10, 35),
            continuous_close=time(10, 45),
        )
    )
    sessions.append(
        SessionDefinition(
            date(2024, 1, 21),
            time(10, 30),
            time(10, 35),
            time(10, 45),
            time(10, 45),
            "test",
        )
    )

    result = build_native_fast_features(
        high,
        low,
        close,
        volume,
        observed,
        volume_valid=volume_valid,
        session_valid=np.ones((days, 1), dtype=np.bool_),
        sigma_asof=np.full((days, 1), 0.02),
        sessions=sessions,
    )

    assert result.valid[-1, 0, 0, 3]
    np.testing.assert_allclose(result.values[-1, 0, 0, 3], np.log(2.0))


def test_native_fast_uses_each_session_prefix_and_ignores_later_minutes() -> None:
    sessions = (
        SessionDefinition(
            date(2024, 1, 2),
            time(10, 0),
            time(15, 45),
            time(17, 0),
            time(17, 15),
            "test",
        ),
        SessionDefinition(
            date(2024, 1, 3),
            time(10, 30),
            time(15, 45),
            time(17, 0),
            time(17, 15),
            "test",
        ),
    )
    shape = (2, 1, 420)
    close = np.full(shape, 100.0)
    high = close + 1.0
    low = close - 1.0
    volume = np.ones(shape)
    observed = np.ones(shape, dtype=np.bool_)
    volume_valid = np.ones(shape, dtype=np.bool_)
    session_valid = np.ones((2, 1), dtype=np.bool_)
    sigma = np.full((2, 1), 0.02)

    def build(
        high_: np.ndarray,
        low_: np.ndarray,
        close_: np.ndarray,
        volume_: np.ndarray,
        observed_: np.ndarray,
        volume_valid_: np.ndarray,
    ):
        return build_native_fast_features(
            high_,
            low_,
            close_,
            volume_,
            observed_,
            volume_valid=volume_valid_,
            session_valid=session_valid,
            sigma_asof=sigma,
            sessions=sessions,
            max_patches=69,
        )

    baseline = build(high, low, close, volume, observed, volume_valid)
    np.testing.assert_array_equal(baseline.patch_mask.sum(axis=2)[:, 0], [69, 63])
    np.testing.assert_allclose(baseline.values[:, 0, -1, 5], [1.0, 0.0])
    np.testing.assert_allclose(baseline.values[1, 0, 62, 5], 1.0)

    changed_high = high.copy()
    changed_low = low.copy()
    changed_close = close.copy()
    changed_volume = volume.copy()
    changed_observed = observed.copy()
    changed_volume_valid = volume_valid.copy()
    for day, cutoff in enumerate((345, 315)):
        changed_high[day, :, cutoff:] = 1e12
        changed_low[day, :, cutoff:] = np.nan
        changed_close[day, :, cutoff:] = -1.0
        changed_volume[day, :, cutoff:] = np.nan
        changed_observed[day, :, cutoff:] = False
        changed_volume_valid[day, :, cutoff:] = False
    changed = build(
        changed_high,
        changed_low,
        changed_close,
        changed_volume,
        changed_observed,
        changed_volume_valid,
    )

    np.testing.assert_array_equal(changed.values, baseline.values)
    np.testing.assert_array_equal(changed.valid, baseline.valid)
    np.testing.assert_array_equal(changed.patch_mask, baseline.patch_mask)
    np.testing.assert_array_equal(
        changed.last_price_age_minutes, baseline.last_price_age_minutes
    )
