import numpy as np
import pytest

from brazil_rv.v2.round7_preprocessing import RobustScaler, common_snapshot


def test_named_input_removal_preserves_masks_ages_and_checkpoint_coordinates():
    from brazil_rv.v2.round7_preprocessing import Round7Preprocessing, retained_fields

    names = ("observed", "empty", "age_only")
    assert retained_fields(names, ["empty"]) == ("observed", "age_only")
    with pytest.raises(ValueError, match="absent"):
        retained_fields(names, ["typo"])
    scaler = RobustScaler.fit(
        np.array([[1.0, 0.0], [3.0, 0.0]]),
        np.array([[True, False], [True, False]]),
        [0, 1],
    )
    prep = Round7Preprocessing(
        {"options": scaler},
        feature_names={"options": ("observed", "age_only")},
        source_columns={"options": (0, 2)},
    )
    raw = {
        "sidecar_options_values": np.array([[3.0, 999.0, 0.0]]),
        "sidecar_options_valid": np.array([[True, False, False]]),
        "sidecar_options_age_sessions": np.array([[0.0, -1.0, 7.0]]),
    }
    restored = Round7Preprocessing.from_payload(prep.payload())
    result = restored.transform_sample(raw)
    np.testing.assert_array_equal(result["sidecar_options_valid"], [[True, False]])
    np.testing.assert_array_equal(result["sidecar_options_age_sessions"], [[0.0, 7.0]])
    assert result["sidecar_options_values"][0, 1] == 0
    assert raw["sidecar_options_values"].shape[-1] == 3


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
    np.testing.assert_allclose(
        scaler.transform(panel[:3], valid[:3])[..., 0],
        np.arcsinh([[-1.0], [0.0], [1.0]]),
    )
    assert np.abs(scaler.transform(panel[-1:], valid[-1:])).max() > 5.0
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
    assert result[2][1, 1] == 1.0  # Known source age survives missing value.
    values[0, 1, 0] = 3.0
    with pytest.raises(ValueError, match="varies by security"):
        common_snapshot(values, valid, ages)


def test_parent_coordinates_are_retained_and_cold_fields_activate_from_fit_only():
    values = np.array([[1.0, 0.0, 2.0], [3.0, 0.0, 2.0]])
    valid = np.array([[True, False, True], [True, False, True]])
    parent = RobustScaler.fit(values, valid, [0, 1])
    future = np.array([[100.0, 10.0, 4.0], [200.0, 20.0, 8.0]])
    child = RobustScaler.fit(future, np.ones_like(valid), [10, 11], parent=parent)
    assert child.inherited == (True, False, False)
    assert child.center[0] == parent.center[0]
    assert child.scale[0] == parent.scale[0]
    assert child.center[1:].tolist() == [15.0, 6.0]
    np.testing.assert_array_equal(
        child.transform(values, valid)[:, 0], parent.transform(values, valid)[:, 0]
    )
    assert RobustScaler.from_payload(child.payload()).payload() == child.payload()


def test_rare_continuous_support_and_binary_fields_are_not_deleted_or_clipped():
    values = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1000.0, 1.0]])
    valid = np.ones_like(values, bool)
    scaler = RobustScaler.fit(values, valid, [0, 1, 2, 3], passthrough=(False, True))
    result = scaler.transform(values, valid)
    assert np.isfinite(result).all() and result[-1, 0] > 0
    np.testing.assert_array_equal(result[:, 1], values[:, 1])
    extremes = np.array([[1e10, 0], [1e20, 1]])
    assert (
        scaler.transform(extremes, np.ones_like(extremes, bool))[1, 0]
        > scaler.transform(extremes, np.ones_like(extremes, bool))[0, 0]
    )
