import numpy as np

from brazil_rv.v2.round5_magnitude import FitClip, magnitude_panel


def _inputs():
    rng = np.random.default_rng(4)
    close = np.exp(np.cumsum(rng.normal(0, 0.02, (65, 3)), axis=0))
    return dict(
        wealth_open=close * 0.998,
        wealth_high=close * 1.012,
        wealth_low=close * 0.988,
        wealth_close=close,
        wealth_valid=np.ones(close.shape, bool),
        volume_brl=np.full(close.shape, 1e7),
        activity_valid=np.ones(close.shape, bool),
        daily_feature_valid=np.ones((*close.shape, 3), bool),
        slow_age_sessions=np.ones((*close.shape, 3), np.float32),
        slow_feature_names=("log_return_1", "yang_zhang_vol_20", "log_volume_mean_20"),
        economic_beta=np.full(close.shape, 1.2),
        economic_beta_valid=np.ones(close.shape, bool),
        economic_beta_age_sessions=np.ones(close.shape, np.float32),
    )


def test_daily_mutation_first_changes_next_decision_and_beta_has_no_extra_lag():
    inputs = _inputs()
    before, mask, _ = magnitude_panel(**inputs)
    inputs["wealth_close"][45, 0] *= 1.01
    after, after_mask, _ = magnitude_panel(**inputs)
    np.testing.assert_array_equal(before[:46], after[:46])
    assert before[46, 0, 0] != after[46, 0, 0]
    np.testing.assert_array_equal(mask, after_mask)
    inputs["economic_beta"][45, 0] = 2
    changed_beta, _, _ = magnitude_panel(**inputs)
    assert changed_beta[45, 0, 3] == 2
    np.testing.assert_array_equal(after[:45], changed_beta[:45])


def test_price_scale_invariance_and_parent_invalidity():
    inputs = _inputs()
    original, _, _ = magnitude_panel(**inputs)
    for field in ("wealth_open", "wealth_high", "wealth_low", "wealth_close"):
        inputs[field] *= 100
    scaled, _, _ = magnitude_panel(**inputs)
    np.testing.assert_allclose(original, scaled, atol=1e-7)
    inputs["daily_feature_valid"][45, 0, 1] = False
    _, valid, age = magnitude_panel(**inputs)
    assert not valid[45, 0, :2].any()
    assert (age[45, 0, :2] == -1).all()


def test_clipping_does_not_fit_on_selection_or_later_dates():
    values = np.arange(60, dtype=np.float32).reshape(10, 3, 2)
    valid = np.ones(values.shape, bool)
    active = np.ones(values.shape[:2], bool)
    fit = FitClip.fit(values, valid, active, np.arange(5))
    values[5:] *= 1e6
    repeated = FitClip.fit(values, valid, active, np.arange(5))
    np.testing.assert_array_equal(fit.lower, repeated.lower)
    np.testing.assert_array_equal(fit.upper, repeated.upper)
    assert (fit.transform(values, valid)[5:] <= fit.upper).all()


def test_magnitude_preserves_actual_parent_and_beta_pair_ages():
    inputs = _inputs()
    inputs["slow_age_sessions"][45, 0] = [1, 3, 2]
    inputs["economic_beta_age_sessions"][45, 0] = 4
    _, valid, ages = magnitude_panel(**inputs)
    assert valid[45, 0].all()
    np.testing.assert_array_equal(ages[45, 0], [3, 3, 2, 4])


def test_fit_support_excludes_invalid_inactive_and_unobserved_fields():
    values = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
    valid = np.ones_like(values, bool)
    active = np.ones(values.shape[:2], bool)
    valid[0, 0, 0] = False
    active[1, 1] = False
    valid[:2, :, 2] = False
    values[0, 0, 0] = -1e9
    values[1, 1] = 1e9
    clip = FitClip.fit(values, valid, active, np.arange(2))
    assert clip.lower[0] > 0
    assert clip.upper[0] < 10
    restored = FitClip.from_payload(clip.payload())
    assert restored.payload() == clip.payload()
    assert np.isneginf(restored.lower[2]) and np.isposinf(restored.upper[2])
    np.testing.assert_array_equal(
        restored.transform(values, valid)[2:, :, 2], values[2:, :, 2]
    )
