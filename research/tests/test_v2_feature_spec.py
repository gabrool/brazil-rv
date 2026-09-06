import numpy as np

from brazil_rv.v2.feature_spec import (
    FeatureSpec,
    feature_schema_sha256,
    transform_feature_panel_into,
)


def test_typed_transforms_keep_zero_distinct_from_missing() -> None:
    specs = (
        FeatureSpec("flag", "fixture", "binary", "flag"),
        FeatureSpec("fraction", "fixture", "bounded_fraction", "fraction_0_1"),
        FeatureSpec("age", "fixture", "age_sessions", "sessions"),
        FeatureSpec("rate", "fixture", "annual_rate", "annual_decimal"),
    )
    values = np.asarray(
        [[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 252.0, 0.01]]],
        dtype=np.float64,
    )
    valid = np.ones_like(values, dtype=bool)
    valid[0, 0, 2] = False
    output = np.empty_like(values, dtype=np.float32)
    output_valid = np.empty_like(valid)
    transform_feature_panel_into(
        values,
        valid,
        np.ones((1, 2), dtype=bool),
        specs,
        output,
        output_valid,
        minimum_rank_names=1,
    )
    np.testing.assert_allclose(output[0, :, 0], [0.0, 1.0])
    np.testing.assert_allclose(output[0, :, 1], [-1.0, 1.0])
    assert output[0, 0, 2] == 0.0 and not output_valid[0, 0, 2]
    assert output[0, 1, 2] == 1.0 and output_valid[0, 1, 2]
    assert output[0, 0, 3] == 0.0 and output_valid[0, 0, 3]
    assert output[0, 1, 3] == np.float32(np.arcsinh(1.0))


def test_rank_transform_requires_declared_cross_section_support() -> None:
    spec = (FeatureSpec("return", "slow", "rank_gauss", "decimal"),)
    values = np.asarray([[[1.0], [2.0], [3.0]]])
    valid = np.ones_like(values, dtype=bool)
    output = np.empty_like(values, dtype=np.float32)
    output_valid = np.empty_like(valid)
    transform_feature_panel_into(
        values,
        valid,
        np.ones((1, 3), dtype=bool),
        spec,
        output,
        output_valid,
        minimum_rank_names=4,
    )
    assert not output_valid.any()
    assert not output.any()


def test_feature_schema_hash_binds_transform_and_order() -> None:
    left = (
        FeatureSpec("a", "x", "binary", "flag"),
        FeatureSpec("b", "x", "rank_gauss", "decimal"),
    )
    assert feature_schema_sha256(left) == feature_schema_sha256(left)
    assert feature_schema_sha256(left) != feature_schema_sha256(left[::-1])
