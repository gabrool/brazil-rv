from __future__ import annotations

from datetime import date, time, timedelta

import numpy as np

from brazil_rv.v2.decision_clock import SessionDefinition
from brazil_rv.v2.intraday_features import build_native_fast_features
from brazil_rv.v2.native_fast_audit import _hand_compute


def test_independent_native_fast_formulas_match_all_seven_channels() -> None:
    days, names, minutes = 40, 3, 360
    sessions = tuple(
        SessionDefinition(
            trade_date=date(2024, 1, 2) + timedelta(days=index),
            continuous_open=time(10, 0),
            decision_time=time(15, 45),
            continuous_close=time(16, 0),
            auction_close=time(16, 5),
            source="reconstructed_v1:test",
        )
        for index in range(days)
    )
    minute = np.arange(minutes, dtype=np.float64)[None, None, :]
    day = np.arange(days, dtype=np.float64)[:, None, None]
    name = np.arange(names, dtype=np.float64)[None, :, None]
    close = 10.0 + 0.002 * minute + 0.01 * day + 0.1 * name
    high = close + 0.01
    low = close - 0.01
    volume = 100.0 + minute + day + name
    market = np.stack((close, high, low, close, volume), axis=-1)
    observed = np.ones((days, names, minutes), dtype=np.bool_)
    supported = np.ones((days, names), dtype=np.bool_)
    sigma = np.full((days, names), 0.02, dtype=np.float64)

    independent = _hand_compute(
        market,
        observed,
        supported,
        sigma,
        sessions,
        max_patches=69,
    )
    canonical = build_native_fast_features(
        high,
        low,
        close,
        volume,
        observed,
        volume_valid=observed,
        session_valid=supported,
        sigma_asof=sigma,
        sessions=sessions,
        max_patches=69,
    )

    np.testing.assert_array_equal(independent[1], canonical.valid)
    np.testing.assert_array_equal(independent[2], canonical.patch_mask)
    np.testing.assert_array_equal(independent[4], canonical.last_price_age_valid)
    np.testing.assert_allclose(
        independent[0][canonical.valid], canonical.values[canonical.valid], atol=2e-6
    )
    np.testing.assert_array_equal(
        independent[3][canonical.last_price_age_valid],
        canonical.last_price_age_minutes[canonical.last_price_age_valid],
    )


def test_independent_native_fast_check_detects_a_stored_value_tamper() -> None:
    values = np.zeros((1, 1, 1, 7), dtype=np.float32)
    expected = values.copy()
    valid = np.ones_like(values, dtype=np.bool_)
    values[0, 0, 0, 4] = 0.5

    error = np.abs(values.astype(np.float64) - expected.astype(np.float64))

    assert float(error[valid].max()) > 2e-6
