import numpy as np

from brazil_rv.v2.corporate_actions import causal_price_adjustment_factor
from brazil_rv.v2.targets import (
    build_multi_day_targets,
    build_multi_day_targets_into,
    build_to_close_target,
)


def test_target_masks_exact_path_and_excluded_event_date() -> None:
    close = np.array(
        [
            [100.0, 100.0, 100.0],
            [101.0, 102.0, 103.0],
            [102.0, 104.0, 106.0],
            [103.0, 106.0, 109.0],
        ]
    )
    active = np.ones_like(close, dtype=bool)
    slow_sigma = np.full_like(close, 0.03)
    unresolved = np.zeros_like(close, dtype=bool)
    unresolved[1, 1] = True
    result = build_multi_day_targets(
        close,
        active,
        slow_sigma,
        unresolved,
        horizons=(1, 2),
    )
    assert result.raw_valid[0, :, 0].tolist() == [True, False, True]
    assert not result.raw_valid[-1].any()
    assert result.primary_valid[0, 0, 0]
    assert np.all(result.primary[~result.primary_valid] == 0)


def test_raw_target_validity_is_independent_of_missing_sigma() -> None:
    close = np.array([[10.0, 10.0], [11.0, 12.0]])
    active = np.ones_like(close, dtype=bool)
    result = build_multi_day_targets(
        close,
        active,
        np.full_like(close, np.nan),
        np.zeros_like(close, dtype=bool),
        horizons=(1,),
    )
    assert result.raw_valid[0, :, 0].all()
    assert not result.primary_valid.any()
    assert result.raw_log_return.dtype == np.float32


def test_target_builder_streams_selected_rows_into_float32_destinations() -> None:
    close = np.arange(20, dtype=np.float64).reshape(5, 4) + 100.0
    shape = (2, 4, 2)
    value_arrays = [np.empty(shape, dtype=np.float32) for _ in range(4)]
    mask_arrays = [np.empty(shape, dtype=np.bool_) for _ in range(2)]
    build_multi_day_targets_into(
        close,
        np.ones_like(close, dtype=np.bool_),
        np.full_like(close, 0.02),
        np.zeros_like(close, dtype=np.bool_),
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


def test_split_adjusted_target_is_continuous_and_cash_event_is_excluded() -> None:
    raw_close = np.asarray([[100.0, 100.0], [50.0, 110.0]])
    price_ratio = np.asarray([[np.nan, np.nan], [0.5, 1.1]])
    split = np.asarray([[False, False], [True, False]])
    adjusted_close = raw_close * causal_price_adjustment_factor(price_ratio, split)
    excluded = np.asarray([[False, False], [False, True]])
    result = build_multi_day_targets(
        adjusted_close,
        np.ones_like(raw_close, dtype=bool),
        np.full_like(raw_close, 0.02),
        excluded,
        horizons=(1,),
    )
    np.testing.assert_allclose(result.raw_log_return[0, 0, 0], 0.0, atol=1e-12)
    assert not result.raw_valid[0, 1, 0]
