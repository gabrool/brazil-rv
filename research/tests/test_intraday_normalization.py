from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from brazil_rv.preprocessing.contract import EQUITY_SESSION_MINUTES
from brazil_rv.preprocessing.intraday_normalization import (
    PROFILE_BIN_COUNT,
    build_full_tod_dynamic_features,
    daily_close_move_statistics,
    estimate_causal_profile,
    validate_profile_artifact,
    variance_from_sufficient_statistics,
    write_profile_artifact,
)


def _dates() -> tuple[date, ...]:
    training = tuple(date(2024, 6, 24) + timedelta(days=index) for index in range(5))
    return (*training, date(2024, 7, 8), date(2024, 7, 9))


def _profile_inputs() -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(7)
    variance = generator.uniform(0.2, 3.0, size=(7, PROFILE_BIN_COUNT))
    count = np.full(variance.shape, 200, dtype=np.int64)
    return variance, count


def _raw(days: int = 7, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    minute = np.arange(EQUITY_SESSION_MINUTES)
    raw = np.zeros((days, EQUITY_SESSION_MINUTES, 5), dtype=np.float64)
    for day_idx in range(days):
        close = (
            scale
            * (100 + day_idx)
            * np.exp(0.0001 * minute + 0.0005 * np.sin(minute / 13 + day_idx))
        )
        raw[day_idx, :, 0] = np.r_[close[0], close[:-1]]
        raw[day_idx, :, 1] = np.maximum(raw[day_idx, :, 0], close) * 1.0001
        raw[day_idx, :, 2] = np.minimum(raw[day_idx, :, 0], close) * 0.9999
        raw[day_idx, :, 3] = close
        raw[day_idx, :, 4] = 1000 + minute
    return raw, np.ones(raw.shape[:2], dtype=bool)


def test_profile_is_emit_then_update_and_future_causal() -> None:
    variance, count = _profile_inputs()
    baseline = estimate_causal_profile(variance, count, _dates())

    same_session = variance.copy()
    same_session[2] *= 100
    mutated = estimate_causal_profile(same_session, count, _dates())
    np.testing.assert_array_equal(
        mutated.relative_variance[:3], baseline.relative_variance[:3]
    )
    assert not np.array_equal(
        mutated.relative_variance[3], baseline.relative_variance[3]
    )

    future = variance.copy()
    future[4:] *= 100
    mutated_future = estimate_causal_profile(future, count, _dates())
    np.testing.assert_array_equal(
        mutated_future.relative_variance[:5], baseline.relative_variance[:5]
    )


def test_profile_freezes_after_training_and_obeys_bounds() -> None:
    variance, count = _profile_inputs()
    profile = estimate_causal_profile(variance, count, _dates())
    assert np.all(profile.relative_variance[5:] == profile.relative_variance[5])
    assert np.all(profile.relative_variance >= 0.25)
    assert np.all(profile.relative_variance <= 4.0)


def test_sufficient_statistics_recover_population_variance() -> None:
    values = np.asarray([[1.0, 2.0, 5.0]])
    total = values.sum(axis=1, keepdims=True)
    total_sq = (values**2).sum(axis=1, keepdims=True)
    count = np.asarray([[values.shape[1]]])
    np.testing.assert_allclose(
        variance_from_sufficient_statistics(total, total_sq, count),
        np.var(values, axis=1, keepdims=True),
    )


def test_full_tod_features_are_price_scale_invariant() -> None:
    variance, count = _profile_inputs()
    q = estimate_causal_profile(variance, count, _dates()).relative_variance
    raw, observed = _raw()
    scaled, scaled_observed = _raw(scale=37.0)
    ready = np.ones(len(_dates()), dtype=bool)
    sigma = np.linspace(0.01, 0.02, len(_dates()))

    features, valid = build_full_tod_dynamic_features(raw, observed, ready, sigma, q)
    scaled_features, scaled_valid = build_full_tod_dynamic_features(
        scaled, scaled_observed, ready, sigma, q
    )
    np.testing.assert_allclose(features, scaled_features, atol=2e-6, rtol=0)
    np.testing.assert_array_equal(valid, scaled_valid)


def test_close_move_statistics_respect_membership_and_readiness() -> None:
    raw, observed = _raw()
    membership = np.ones(7, dtype=bool)
    membership[1] = False
    ready = np.ones(7, dtype=bool)
    ready[2] = False
    total, total_sq, count = daily_close_move_statistics(
        raw, observed, membership, ready, np.full(7, 0.01)
    )
    assert total.shape == total_sq.shape == count.shape == (7, PROFILE_BIN_COUNT)
    assert count[0].sum() == EQUITY_SESSION_MINUTES
    assert count[1].sum() == count[2].sum() == 0


def test_profile_artifact_is_hash_bound_and_frozen(
    tmp_path: Path,
) -> None:
    variance, count = _profile_inputs()
    dates = _dates()
    profile = estimate_causal_profile(variance, count, dates)
    written = write_profile_artifact(tmp_path, profile, dates)
    assert validate_profile_artifact(tmp_path, dates) == written

    path = tmp_path / "equity_tod_profile.npy"
    values = np.load(path, allow_pickle=False)
    values[0, 0] += 0.01
    np.save(path, values, allow_pickle=False)
    with pytest.raises(ValueError, match="Invalid causal TOD profile artifact"):
        validate_profile_artifact(tmp_path, dates)
