import numpy as np

from brazil_rv.v2.targets import build_multi_day_targets, build_to_close_target


def test_target_masks_exact_path_and_unresolved_ex_date() -> None:
    close = np.array(
        [
            [100.0, 100.0, 100.0],
            [101.0, 102.0, 103.0],
            [102.0, 104.0, 106.0],
            [103.0, 106.0, 109.0],
        ]
    )
    active = np.ones_like(close, dtype=bool)
    fast_sigma = np.full_like(close, 0.02)
    slow_sigma = np.full_like(close, 0.03)
    fast = np.ones_like(close, dtype=bool)
    unresolved = np.zeros_like(close, dtype=bool)
    unresolved[1, 1] = True
    result = build_multi_day_targets(
        close,
        active,
        fast_sigma,
        slow_sigma,
        fast,
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
        np.full_like(close, np.nan),
        np.zeros_like(close, dtype=bool),
        np.zeros_like(close, dtype=bool),
        horizons=(1,),
    )
    assert result.raw_valid[0, :, 0].all()
    assert not result.primary_valid.any()


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
