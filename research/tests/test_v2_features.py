from datetime import date, timedelta

import numpy as np

from brazil_rv.v2.features import (
    _peer_features,
    _rolling_stat,
    build_slow_features,
    deterministic_average_linkage,
    exact_log_return,
    pairwise_masked_correlation,
    yang_zhang_volatility,
)
from brazil_rv.v2.universe import build_daily_universe


def test_rolling_statistics_accept_eighty_percent_complete_window() -> None:
    values = np.arange(20.0)[:, None]
    values[[2, 5, 8, 11], 0] = np.nan
    mean, valid = _rolling_stat(values, 20, "mean")
    assert valid[-1, 0]
    assert mean[-1, 0] == np.nanmean(values[:, 0])
    values[14, 0] = np.nan
    _, invalid = _rolling_stat(values, 20, "mean")
    assert not invalid[-1, 0]


def test_exact_return_invalidates_lookbacks_crossing_ambiguous_action() -> None:
    close = np.arange(10.0, 16.0)[:, None]
    unresolved = np.zeros_like(close, dtype=np.bool_)
    unresolved[2, 0] = True
    values, valid = exact_log_return(close, 3, unresolved)
    # The event is crossed by returns ending at 3 and 4, but is the starting
    # close (and thus already on both sides of the factor) at the return ending
    # at 5.
    assert valid[:, 0].tolist() == [False, False, False, False, False, True]
    assert np.isnan(values[:5, 0]).all()


def test_exact_return_invalidates_a_wealth_chain_restart() -> None:
    close = np.arange(1.0, 9.0)[:, None]
    wealth_valid = np.ones_like(close, dtype=np.bool_)
    wealth_valid[3, 0] = False

    _, valid = exact_log_return(
        close,
        5,
        shareholder_wealth_valid=wealth_valid,
    )

    assert not valid[5, 0]
    assert not valid[6, 0]
    assert not valid[7, 0]


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
    ) + np.log(low[1:, 0] / close[1:, 0]) * np.log(low[1:, 0] / open_[1:, 0])
    k = 0.34 / (1.34 + 4 / 2)
    expected = np.sqrt(
        np.var(overnight, ddof=1) + k * np.var(intraday, ddof=1) + (1 - k) * np.mean(rs)
    )
    assert valid[3, 0]
    assert result[3, 0] == expected


def test_yang_zhang_masks_windows_crossing_ambiguous_actions() -> None:
    close = np.arange(100.0, 108.0)[:, None]
    open_ = close * 0.999
    high = close * 1.01
    low = open_ * 0.99
    unresolved = np.zeros_like(close, dtype=np.bool_)
    unresolved[3, 0] = True

    values, valid = yang_zhang_volatility(open_, high, low, close, 3, unresolved)

    assert valid[:, 0].tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ]
    assert np.isnan(values[3:6, 0]).all()


def test_yang_zhang_accepts_eighty_percent_complete_window() -> None:
    close = np.arange(100.0, 107.0)[:, None]
    open_ = close * 0.999
    high = close * 1.01
    low = open_ * 0.99
    high[2, 0] = np.nan

    values, valid = yang_zhang_volatility(open_, high, low, close, 5)

    assert valid[5, 0]
    assert np.isfinite(values[5, 0])
    high[3, 0] = np.nan
    _, invalid = yang_zhang_volatility(open_, high, low, close, 5)
    assert not invalid[5, 0]


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
        np.full_like(base, 3_000_000.0),
        np.full_like(base, 1_000.0),
        seen,
        active,
        dates,
        raw_high=base * 1.01,
        raw_low=base * 0.99,
        raw_close=base,
        price_observed=seen,
        history_observed=seen,
        activity_valid=seen,
        cluster_labels=labels,
    )
    changed = base.copy()
    changed[71:] *= 100.0
    mutated = build_slow_features(
        changed,
        changed * 1.01,
        changed * 0.99,
        changed,
        np.full_like(base, 3_000_000.0),
        np.full_like(base, 1_000.0),
        seen,
        active,
        dates,
        raw_high=changed * 1.01,
        raw_low=changed * 0.99,
        raw_close=changed,
        price_observed=seen,
        history_observed=seen,
        activity_valid=seen,
        cluster_labels=labels,
    )
    np.testing.assert_array_equal(original.values[70], mutated.values[70])
    np.testing.assert_array_equal(original.valid[70], mutated.valid[70])


def test_cluster_peer_path_is_unchanged_by_future_mutation() -> None:
    days, names = 90, 8
    time = np.arange(days, dtype=np.float64)[:, None]
    name = np.arange(names, dtype=np.float64)[None, :]
    base = 10.0 * np.exp(0.0007 * time + 0.004 * np.sin(time + name))
    seen = np.ones_like(base, dtype=np.bool_)
    active = np.ones_like(base, dtype=np.bool_)
    dates = [date(2023, 1, 2) + timedelta(days=index) for index in range(days)]
    labels = np.broadcast_to(
        np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int16), base.shape
    ).copy()

    def build(close: np.ndarray):
        return build_slow_features(
            close,
            close * 1.01,
            close * 0.99,
            close,
            np.full_like(close, 3_000_000.0),
            np.full_like(close, 1_000.0),
            seen,
            active,
            dates,
            raw_high=close * 1.01,
            raw_low=close * 0.99,
            raw_close=close,
            price_observed=seen,
            history_observed=seen,
            activity_valid=seen,
            cluster_labels=labels,
        )

    original = build(base)
    changed = base.copy()
    changed[71:, :4] *= np.linspace(2.0, 5.0, 4)
    mutated = build(changed)
    assert original.valid[70, :, 27:32].any()
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
    values = np.array([[1.0, 10.0], [2.0, np.nan], [3.0, 30.0], [4.0, 20.0]])
    valid = np.isfinite(values)
    correlation = pairwise_masked_correlation(values, valid, minimum_observed=3)
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
        np.full_like(close, 3_000_000.0),
        np.full_like(close, 1_000.0),
        np.ones_like(close, dtype=bool),
        np.ones_like(close, dtype=bool),
        dates,
        raw_high=high,
        raw_low=low,
        raw_close=close,
        price_observed=np.ones_like(close, dtype=bool),
        history_observed=np.ones_like(close, dtype=bool),
        activity_valid=np.ones_like(close, dtype=bool),
        cluster_labels=np.zeros_like(close, dtype=np.int16),
    )
    assert result.valid[64, 0, 23]
    assert result.values[64, 0, 23] == np.float32(np.log(15.0 / 7.0))


def test_ambiguous_event_masks_only_affected_price_features() -> None:
    days = 270
    close = 100.0 + np.arange(days, dtype=np.float64)[:, None]
    high = close * 1.01
    low = close * 0.99
    unresolved = np.zeros_like(close, dtype=np.bool_)
    unresolved[261, 0] = True
    dates = [date(2023, 1, 2) + timedelta(days=index) for index in range(days)]

    result = build_slow_features(
        close,
        high,
        low,
        close,
        np.full_like(close, 3_000_000.0),
        np.full_like(close, 1_000.0),
        np.ones_like(close, dtype=bool),
        np.ones_like(close, dtype=bool),
        dates,
        raw_high=high,
        raw_low=low,
        raw_close=close,
        price_observed=np.ones_like(close, dtype=bool),
        history_observed=np.ones_like(close, dtype=bool),
        activity_valid=np.ones_like(close, dtype=bool),
        cluster_labels=np.zeros_like(close, dtype=np.int16),
        ambiguous_action=unresolved,
    )

    # Same-session scale-free shape/activity fields do not cross the break.
    assert result.valid[261, 0, 22]
    assert result.valid[261, 0, 24]
    assert result.valid[261, 0, 17]
    # Observed-history age is not a price path and remains valid. Every
    # trailing cross-boundary price/return interval is excluded.
    assert result.valid[261, 0, 25]
    assert result.valid[269, 0, 25]
    assert not result.valid[264, 0, 1]
    assert not result.valid[264, 0, 7]
    assert not result.valid[264, 0, 14]
    assert not result.valid[264, 0, 23]


def test_unresolved_wealth_does_not_erase_safe_price_activity_or_history() -> None:
    days = 65
    shape = (days, 1)
    raw_close = np.full(shape, 10.0)
    raw_high = np.full(shape, 11.0)
    raw_low = np.full(shape, 9.0)
    price_observed = np.ones(shape, dtype=np.bool_)
    price_observed[:10] = False
    history_observed = np.ones(shape, dtype=np.bool_)
    wealth = np.zeros(shape, dtype=np.float64)
    wealth_valid = np.zeros(shape, dtype=np.bool_)
    activity_valid = np.ones(shape, dtype=np.bool_)
    dates = [date(2023, 1, 2) + timedelta(days=index) for index in range(days)]

    result = build_slow_features(
        wealth,
        wealth,
        wealth,
        wealth,
        np.full(shape, 3_000_000.0),
        np.full(shape, 1_000.0),
        wealth_valid,
        np.ones(shape, dtype=np.bool_),
        dates,
        raw_high=raw_high,
        raw_low=raw_low,
        raw_close=raw_close,
        price_observed=price_observed,
        history_observed=history_observed,
        activity_valid=activity_valid,
        cluster_labels=np.zeros(shape, dtype=np.int16),
        ambiguous_action=np.ones(shape, dtype=np.bool_),
    )

    # Cross-session wealth fields cannot cross unresolved action coverage.
    assert not result.valid[64, 0, 0]
    assert not result.valid[64, 0, 7]
    assert not result.valid[64, 0, 19]
    assert not result.valid[64, 0, 23]
    # Same-session raw-price shape, complete-source activity, and linked raw
    # observation history remain usable under their independent contracts.
    assert result.valid[64, 0, 22]
    assert result.values[64, 0, 22] == np.float32(np.log(11.0 / 9.0))
    assert result.valid[64, 0, 24]
    assert result.values[64, 0, 24] == np.float32(0.5)
    assert result.valid[64, 0, 17]
    assert result.valid[64, 0, 21]
    assert result.valid[64, 0, 25]
    assert result.values[64, 0, 25] == np.float32(64.0)
    assert result.values[64, 0, 26] == np.float32(1.0)


def test_activity_features_and_universe_share_exact_calendar_session_support() -> None:
    days = 21
    shape = (days, 1)
    close = np.full(shape, 10.0)
    volume = np.arange(1.0, days + 1.0)[:, None]
    trades = volume * 2.0
    observed = np.ones(shape, dtype=np.bool_)
    trade_observed = observed.copy()
    # A complete-source missing row is a valid no-trade zero.
    observed[5, 0] = False
    trade_observed[5, 0] = False
    volume[5, 0] = 0.0
    trades[5, 0] = 0.0
    activity_valid = np.ones(shape, dtype=np.bool_)
    dates = [date(2023, 1, 2) + timedelta(days=index) for index in range(days)]

    def build(activity: np.ndarray):
        return build_slow_features(
            close,
            close,
            close,
            close,
            volume,
            trades,
            np.ones(shape, dtype=np.bool_),
            np.ones(shape, dtype=np.bool_),
            dates,
            raw_high=close,
            raw_low=close,
            raw_close=close,
            price_observed=observed,
            history_observed=observed,
            activity_valid=activity,
            cluster_labels=np.zeros(shape, dtype=np.int16),
        )

    slow = build(activity_valid)
    window = volume[:20, 0]
    expected_mean = np.mean(window)
    expected_z = (volume[19, 0] - expected_mean) / np.std(window, ddof=0)
    assert slow.valid[19, 0, 17]
    assert slow.values[19, 0, 17] == np.float32(np.log(expected_mean))
    assert slow.values[19, 0, 18] == np.float32(expected_z)
    assert slow.values[19, 0, 20] == np.float32(expected_z)
    assert slow.values[19, 0, 21] == np.float32(volume[19, 0] / expected_mean)

    universe = build_daily_universe(
        close,
        volume,
        observed,
        trade_observed=trade_observed,
        activity_valid=activity_valid,
        source_session_complete=np.ones(days, dtype=np.bool_),
        prior_sessions=20,
        minimum_traded=19,
        minimum_median_volume_brl=0.0,
        minimum_prior_close_brl=1.0,
        minimum_history_sessions=20,
    )
    assert universe.active[20, 0]

    unknown_activity = activity_valid.copy()
    unknown_activity[5, 0] = False
    unknown_slow = build(unknown_activity)
    assert not unknown_slow.valid[19, 0, [17, 18, 20, 21]].any()
    unknown_source = np.ones(days, dtype=np.bool_)
    unknown_source[5] = False
    unknown_universe = build_daily_universe(
        close,
        volume,
        observed,
        trade_observed=trade_observed,
        activity_valid=unknown_activity,
        source_session_complete=unknown_source,
        prior_sessions=20,
        minimum_traded=19,
        minimum_median_volume_brl=0.0,
        minimum_prior_close_brl=1.0,
        minimum_history_sessions=20,
    )
    assert not unknown_universe.active[20, 0]


def test_history_age_is_not_listing_age_and_flat_close_location_is_valid() -> None:
    days = 65
    close = np.full((days, 2), 10.0)
    seen = np.ones_like(close, dtype=bool)
    seen[:10, 1] = False
    dates = [date(2023, 1, 2) + timedelta(days=index) for index in range(days)]
    result = build_slow_features(
        close,
        close,
        close,
        close,
        np.full_like(close, 3_000_000.0),
        np.full_like(close, 100.0),
        seen,
        seen,
        dates,
        raw_high=close,
        raw_low=close,
        raw_close=close,
        price_observed=seen,
        history_observed=seen,
        activity_valid=seen,
        cluster_labels=np.zeros_like(close, dtype=np.int16),
    )
    assert result.valid[64, :, 24].all()
    np.testing.assert_array_equal(result.values[64, :, 24], [0.5, 0.5])
    np.testing.assert_array_equal(result.values[64, :, 25], [64.0, 54.0])
    np.testing.assert_array_equal(result.values[64, :, 26], [1.0, 0.0])


def test_monthly_cluster_labels_are_end_to_end_causal() -> None:
    days, names = 180, 14
    dates = np.arange(
        np.datetime64("2023-01-02"),
        np.datetime64("2023-01-02") + np.timedelta64(days, "D"),
    )
    time = np.arange(days, dtype=np.float64)[:, None]
    name = np.arange(names, dtype=np.float64)[None, :]
    daily_return = 0.002 * np.sin(time / 7.0 + name / 3.0) + 0.0001 * name
    close = 100.0 * np.exp(np.cumsum(daily_return, axis=0))
    observed = np.ones_like(close, dtype=np.bool_)
    active = observed.copy()
    cutoff = 139

    def build(values: np.ndarray):
        return build_slow_features(
            values * 0.999,
            values * 1.01,
            values * 0.99,
            values,
            np.full_like(values, 3_000_000.0),
            np.full_like(values, 1_000.0),
            observed,
            active,
            dates,
            raw_high=values * 1.01,
            raw_low=values * 0.99,
            raw_close=values,
            price_observed=observed,
            history_observed=observed,
            activity_valid=observed,
        ).cluster_labels

    original = build(close)
    changed = close.copy()
    changed[cutoff + 1 :] *= (
        1.0
        + (np.arange(days - cutoff - 1)[:, None] + 1)
        * np.linspace(-0.003, 0.003, names)[None, :]
    )
    mutated = build(changed)

    assert (original[cutoff] >= 0).all()
    np.testing.assert_array_equal(original[: cutoff + 1], mutated[: cutoff + 1])


def test_cluster_peer_features_exclude_focal_and_require_three_valid_peers() -> None:
    return_5 = np.asarray([[100.0, 1.0, 2.0, 3.0, 4.0]])
    return_21 = return_5.copy()
    valid_5 = np.ones_like(return_5, dtype=bool)
    valid_5[0, 4] = False
    valid_21 = np.ones_like(return_21, dtype=bool)
    labels = np.zeros_like(return_5, dtype=np.int16)
    active = np.ones_like(return_5, dtype=bool)

    values, valid = _peer_features(
        return_5, valid_5, return_21, valid_21, labels, active
    )

    assert values[0, 0, 0] == 2.0
    assert values[0, 0, 2] == 98.0
    assert values[0, 0, 1] == 2.5
    assert values[0, 0, 4] == np.std([1.0, 2.0, 3.0, 4.0])
    assert valid[0, 0].all()
    assert not valid[0, 4, 0]
    assert not valid[0, 4, 2]

    valid_5[0, 3] = False
    _, too_few = _peer_features(return_5, valid_5, return_21, valid_21, labels, active)
    assert not too_few[0, 0, 0]
    assert not too_few[0, 0, 2]
