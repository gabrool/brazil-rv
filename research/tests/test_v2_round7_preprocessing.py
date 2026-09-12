import numpy as np
import pytest

from brazil_rv.v2.round7_preprocessing import RobustScaler, common_snapshot


def test_robust_statistics_exclude_later_values_and_keep_sparse_signs():
    panel = np.asarray(
        [[[-4.0, 10.0]], [[0.0, 10.0]], [[4.0, 10.0]], [[999.0, -999.0]]]
    )
    valid = np.ones_like(panel, dtype=bool)
    fit = np.asarray([0, 1, 2])
    scaler = RobustScaler.fit(panel[fit], valid[fit], fit)
    panel[-1] *= -1e6
    other = RobustScaler.fit(panel[fit], valid[fit], fit)
    assert scaler.payload() == other.payload()
    np.testing.assert_array_equal(scaler.center, [0.0, 10.0])
    np.testing.assert_array_equal(scaler.scale, [4.0, 1.0])
    np.testing.assert_array_equal(
        scaler.transform(panel[:3], valid[:3])[..., 0], [[-1.0], [0.0], [1.0]]
    )
    assert np.abs(scaler.transform(panel[-1:], valid[-1:])).max() == 5.0
    assert not scaler.transform(panel, np.zeros_like(valid)).any()


def test_common_snapshot_is_unweighted_by_stock_count_and_rejects_exposures():
    values = np.asarray([[[2.0, 4.0], [2.0, 4.0]], [[3.0, 0.0], [3.0, 0.0]]])
    valid = np.ones_like(values, dtype=bool)
    valid[1, :, 1] = False
    ages = np.ones_like(values)
    result = common_snapshot(values, valid, ages)
    duplicated = common_snapshot(
        np.repeat(values, 3, axis=1),
        np.repeat(valid, 3, axis=1),
        np.repeat(ages, 3, axis=1),
    )
    for first, second in zip(result, duplicated, strict=True):
        np.testing.assert_array_equal(first, second)
    assert result[2][1, 1] == -1.0
    values[0, 1, 0] = 3.0
    with pytest.raises(ValueError, match="varies by security"):
        common_snapshot(values, valid, ages)
