from datetime import date, timedelta

import numpy as np

from brazil_rv.v2.features import (
    build_slow_features,
    deterministic_average_linkage,
    exact_log_return,
    pairwise_masked_correlation,
    yang_zhang_volatility,
)


def test_exact_return_invalidates_lookbacks_crossing_unresolved_action() -> None:
    close = np.arange(10.0, 16.0)[:, None]
    unresolved = np.zeros_like(close, dtype=np.bool_)
    unresolved[2, 0] = True
    values, valid = exact_log_return(close, 3, unresolved)
    # The event is crossed by returns ending at 3 and 4, but is the starting
    # close (and thus already on both sides of the factor) at the return ending
    # at 5.
    assert valid[:, 0].tolist() == [False, False, False, False, False, True]
    assert np.isnan(values[:5, 0]).all()


def test_yang_zhang_matches_hand_computed_fixture() -> None:
    close = np.array([[100.0], [102.0], [101.0], [104.0]])
    open_ = np.array([[100.0], [101.0], [103.0], [102.0]])
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    result, valid = yang_zhang_volatility(open_, high, low, close, 3)
    overnight = np.log(open_[1:, 0] / close[:-1, 0])
    intraday = np.log(close[1:, 0] / open_[1:, 0])
    rs = np.log(high[1:, 0] / close[1:, 0]) * np.log(
        high[1:, 0] / open_[1:, 0]
    ) + np.log(low[1:, 0] / close[1:, 0]) * np.log(
        low[1:, 0] / open_[1:, 0]
    )
    k = 0.34 / (1.34 + 4 / 2)
    expected = np.sqrt(
        np.var(overnight, ddof=1)
        + k * np.var(intraday, ddof=1)
        + (1 - k) * np.mean(rs)
    )
    assert valid[3, 0]
    assert result[3, 0] == expected


def test_slow_features_are_unchanged_by_future_mutation() -> None:
    days, names = 90, 3
    base = 10.0 * np.exp(
        np.arange(days)[:, None] * 0.001 + np.arange(names)[None] * 0.01
    )
    seen = np.ones_like(base, dtype=bool)
    active = np.ones_like(base, dtype=bool)
    dates = [date(2023, 1, 2) + timedelta(days=index) for index in range(days)]
    labels = np.zeros_like(base, dtype=np.int16)
    original = build_slow_features(
        base,
        base * 1.01,
        base * 0.99,
        base,
        base,
        np.full_like(base, 3_000_000.0),
        np.full_like(base, 1_000.0),
        seen,
        active,
        dates,
        cluster_labels=labels,
    )
    changed = base.copy()
    changed[71:] *= 100.0
    mutated = build_slow_features(
        changed,
        changed * 1.01,
        changed * 0.99,
        changed,
        changed,
        np.full_like(base, 3_000_000.0),
        np.full_like(base, 1_000.0),
        seen,
        active,
        dates,
        cluster_labels=labels,
    )
    np.testing.assert_array_equal(original.values[70], mutated.values[70])
    np.testing.assert_array_equal(original.valid[70], mutated.valid[70])


def test_vectorized_correlation_and_linkage_are_deterministic() -> None:
    values = np.array(
        [
            [1.0, 1.0, -1.0, -1.0],
            [2.0, 2.1, -2.0, -2.1],
            [3.0, 2.9, -3.0, -2.9],
            [4.0, 4.0, -4.0, -4.0],
        ]
    )
    correlation = pairwise_masked_correlation(
        values, np.ones_like(values, dtype=bool), minimum_observed=3
    )
    first = deterministic_average_linkage(
        correlation, np.ones(4, dtype=bool), cluster_count=2
    )
    second = deterministic_average_linkage(
        correlation, np.ones(4, dtype=bool), cluster_count=2
    )
    np.testing.assert_array_equal(first, second)
    assert first[0] == first[1]
    assert first[2] == first[3]


def test_pairwise_correlation_is_exact_spearman_under_missingness() -> None:
    values = np.array(
        [[1.0, 10.0], [2.0, np.nan], [3.0, 30.0], [4.0, 20.0]]
    )
    valid = np.isfinite(values)
    correlation = pairwise_masked_correlation(
        values, valid, minimum_observed=3
    )
    assert correlation[0, 1] == 0.5
    assert correlation[1, 0] == 0.5


def test_five_session_range_is_log_window_extrema_not_mean_daily_range() -> None:
    days = 65
    close = np.full((days, 1), 10.0)
    high = np.full_like(close, 11.0)
    low = np.full_like(close, 9.0)
    high[60:65, 0] = [11.0, 12.0, 15.0, 13.0, 14.0]
    low[60:65, 0] = [9.0, 8.0, 7.0, 8.5, 9.0]
    dates = [date(2023, 1, 2) + timedelta(days=index) for index in range(days)]
    result = build_slow_features(
        close,
        high,
        low,
        close,
        close,
        np.full_like(close, 3_000_000.0),
        np.full_like(close, 1_000.0),
        np.ones_like(close, dtype=bool),
        np.ones_like(close, dtype=bool),
        dates,
        cluster_labels=np.zeros_like(close, dtype=np.int16),
    )
    assert result.valid[64, 0, 23]
    assert result.values[64, 0, 23] == np.float32(np.log(15.0 / 7.0))
