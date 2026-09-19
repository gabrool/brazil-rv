import numpy as np

from brazil_rv.v2.foundation_residual import (
    age_channels,
    choose_strength,
    date_weights,
    fit_ridge,
)


def test_known_zero_age_is_distinct_from_missing_and_old_data_is_retained():
    encoded, known = age_channels(np.array([-1.0, 0.0, 252.0, 10000.0]))
    np.testing.assert_array_equal(known, [0, 1, 1, 1])
    assert encoded[0] == encoded[1] == 0
    assert encoded[2] == 0.5
    assert 0.5 < encoded[3] < 1


def test_date_replication_does_not_change_ridge_fit():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(12, 3))
    dates = np.repeat([1, 2, 3], 4)
    y = x[:, 0] + 0.3 * x[:, 1] + 2.0
    beta, intercept = fit_ridge(x, y, dates)
    indices = np.r_[np.arange(12), np.tile(np.arange(4), 5)]
    repeated_beta, repeated_intercept = fit_ridge(
        x[indices], y[indices], dates[indices]
    )
    np.testing.assert_allclose(beta, repeated_beta, atol=1e-12)
    np.testing.assert_allclose(intercept, repeated_intercept, atol=1e-12)
    weights = date_weights(dates[indices])
    np.testing.assert_allclose(
        [weights[dates[indices] == d].sum() for d in (1, 2, 3)],
        np.full(3, len(indices) / 3),
    )


def test_intercept_is_unpenalized_and_ties_choose_unchanged_anchor():
    x = np.zeros((6, 2))
    beta, intercept = fit_ridge(x, np.full(6, 4.5), np.repeat([1, 2], 3))
    np.testing.assert_array_equal(beta, 0)
    np.testing.assert_allclose(intercept, 4.5)
    anchor = np.tile(np.arange(6)[:, None], (1, 3)).astype(float)
    valid = np.ones_like(anchor, dtype=bool)
    valid[0, 1] = False
    strength, scores = choose_strength(
        anchor, np.ones_like(anchor), anchor, valid, np.repeat([1, 2], 3)
    )
    assert strength == 0
    np.testing.assert_allclose(scores, 1)
