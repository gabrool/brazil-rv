import numpy as np

from brazil_rv.v2.corporate_actions import causal_price_adjustment_factor
from brazil_rv.v2.targets import (
    build_multi_day_targets,
    build_multi_day_targets_into,
    build_neutralized_log_returns,
    build_to_close_target,
)


def test_target_masks_exact_path_and_missing_return() -> None:
    returns = np.log(np.array(
        [
            [100.0, 100.0, 100.0],
            [101.0, 102.0, 103.0],
            [102.0, 104.0, 106.0],
            [103.0, 106.0, 109.0],
        ]
    )[1:] / np.array(
        [
            [100.0, 100.0, 100.0],
            [101.0, 102.0, 103.0],
            [102.0, 104.0, 106.0],
        ]
    ))
    returns = np.vstack((np.zeros((1, 3)), returns))
    active = np.ones_like(returns, dtype=bool)
    slow_sigma = np.full_like(returns, 0.03)
    valid = np.ones_like(returns, dtype=bool)
    valid[1, 1] = False
    result = build_multi_day_targets(
        returns,
        valid,
        active,
        slow_sigma,
        horizons=(1, 2),
    )
    assert not result.raw_valid[0].any()
    assert result.raw_valid[1, :, 0].all()
    assert not result.raw_valid[-1].any()
    assert result.primary_valid[1, 0, 0]
    assert np.all(result.primary[~result.primary_valid] == 0)


def test_raw_target_validity_is_independent_of_missing_sigma() -> None:
    close = np.array([[10.0, 10.0], [11.0, 12.0]])
    active = np.ones_like(close, dtype=bool)
    result = build_multi_day_targets(
        np.vstack((np.zeros((1, 2)), np.log(close[1:] / close[:-1]))),
        np.ones_like(close, dtype=bool),
        active,
        np.full_like(close, np.nan),
        horizons=(1,),
    )
    assert not result.raw_valid[0, :, 0].any()
    assert not result.primary_valid.any()
    assert result.raw_log_return.dtype == np.float32


def test_target_builder_streams_selected_rows_into_float32_destinations() -> None:
    close = np.arange(20, dtype=np.float64).reshape(5, 4) + 100.0
    shape = (2, 4, 2)
    value_arrays = [np.empty(shape, dtype=np.float32) for _ in range(4)]
    mask_arrays = [np.empty(shape, dtype=np.bool_) for _ in range(2)]
    build_multi_day_targets_into(
        np.vstack((np.zeros((1, 4)), np.log(close[1:] / close[:-1]))),
        np.ones_like(close, dtype=np.bool_),
        np.ones_like(close, dtype=np.bool_),
        np.full_like(close, 0.02),
        primary=value_arrays[0],
        primary_valid=mask_arrays[0],
        normalized_residual=value_arrays[1],
        raw_midrank=value_arrays[2],
        raw_valid=mask_arrays[1],
        raw_log_return=value_arrays[3],
        source_rows=np.asarray([1, 3]),
        horizons=(1, 2),
    )
    assert mask_arrays[0][0].all()
    assert not mask_arrays[0][1, :, 1].any()
    expected = np.log(close[2] / close[1]).astype(np.float32)
    np.testing.assert_array_equal(value_arrays[3][0, :, 0], expected)


def test_to_close_target_is_cross_sectionally_ranked() -> None:
    entry = np.array([[100.0, 100.0, 100.0]])
    close = np.array([[99.0, 100.0, 102.0]])
    result = build_to_close_target(
        entry,
        close,
        np.full_like(entry, 0.02),
        np.ones_like(entry, dtype=bool),
        np.ones_like(entry, dtype=bool),
    )
    np.testing.assert_array_equal(result.target[0], [0.0, 0.5, 1.0])


def test_neutralized_event_is_valid_and_uses_market_median() -> None:
    raw_close = np.asarray(
        [[100.0, 100.0], [100.0, 100.0], [50.0, 110.0], [51.0, 111.0]]
    )
    price_ratio = np.asarray(
        [[np.nan, np.nan], [np.nan, np.nan], [0.5, 1.1], [np.nan, np.nan]]
    )
    split = np.asarray(
        [[False, False], [False, False], [True, False], [False, False]]
    )
    adjusted_close = raw_close * causal_price_adjustment_factor(price_ratio, split)
    event = np.asarray(
        [[False, False], [False, False], [False, True], [False, False]]
    )
    neutralized = build_neutralized_log_returns(
        adjusted_close,
        np.ones_like(raw_close, dtype=bool),
        np.ones_like(raw_close, dtype=bool),
        event,
        minimum_cross_section=1,
    )
    result = build_multi_day_targets(
        neutralized.log_return,
        neutralized.valid,
        np.ones_like(raw_close, dtype=bool),
        np.full_like(raw_close, 0.02),
        horizons=(1,),
    )
    np.testing.assert_allclose(result.raw_log_return[1, 0, 0], 0.0, atol=1e-12)
    assert not result.raw_valid[0].any()
    np.testing.assert_allclose(neutralized.log_return[2], [0.0, 0.0], atol=1e-12)


def test_residual_is_median_removed_before_name_specific_scaling() -> None:
    returns = np.zeros((3, 5), dtype=np.float64)
    returns[2] = 0.02
    sigma = np.tile(np.asarray([0.01, 0.02, 0.03, 0.04, 0.05]), (3, 1))
    result = build_multi_day_targets(
        returns,
        np.ones_like(returns, dtype=np.bool_),
        np.ones_like(returns, dtype=np.bool_),
        sigma,
        horizons=(1,),
    )
    np.testing.assert_array_equal(result.primary[1, :, 0], np.full(5, 0.5))


def test_target_uses_prior_row_sigma_and_missing_path_stays_invalid() -> None:
    returns = np.zeros((4, 3), dtype=np.float64)
    returns[2] = [-0.02, 0.0, 0.02]
    sigma = np.full((4, 3), 0.02)
    first = build_multi_day_targets(
        returns,
        np.ones_like(returns, dtype=np.bool_),
        np.ones_like(returns, dtype=np.bool_),
        sigma,
        horizons=(1,),
    )
    changed = sigma.copy()
    changed[1] = [0.5, 0.6, 0.7]
    second = build_multi_day_targets(
        returns,
        np.ones_like(returns, dtype=np.bool_),
        np.ones_like(returns, dtype=np.bool_),
        changed,
        horizons=(1,),
    )
    np.testing.assert_array_equal(first.primary[1], second.primary[1])
    valid = np.ones_like(returns, dtype=np.bool_)
    valid[2, 0] = False
    missing = build_multi_day_targets(
        returns,
        valid,
        np.ones_like(returns, dtype=np.bool_),
        sigma,
        horizons=(1,),
    )
    assert not missing.raw_valid[1, 0, 0]
