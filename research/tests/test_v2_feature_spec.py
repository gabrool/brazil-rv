import numpy as np
from dataclasses import replace

from brazil_rv.v2.feature_spec import (
    FeatureSpec,
    feature_schema_sha256,
    feature_specs,
    native_fast_feature_specs,
    observation_age_sessions_into,
    transform_feature_panel_into,
)
from brazil_rv.v2.contract import INTRADAY_DAILY_FEATURES, SIDECAR_FEATURES, SLOW_FEATURES
from brazil_rv.v2.intraday_features import NATIVE_FAST_FEATURES


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
    spec = (
        FeatureSpec(
            "return", "slow", "rank_gauss", "decimal", minimum_support=4
        ),
    )
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


def test_decision_snapshot_applies_daily_lag_once_with_current_membership() -> None:
    spec = (FeatureSpec("lagged", "slow", "signed_identity", "decimal"),)
    values = np.asarray(
        [
            [[0.1], [0.2]],
            [[0.3], [0.4]],
            [[0.5], [0.6]],
        ],
        dtype=np.float64,
    )
    valid = np.ones_like(values, dtype=np.bool_)
    active = np.asarray(
        [[True, False], [False, True], [True, True]], dtype=np.bool_
    )
    output = np.empty_like(values, dtype=np.float32)
    output_valid = np.empty_like(valid)

    transform_feature_panel_into(
        values,
        valid,
        active,
        spec,
        output,
        output_valid,
        source_rows=np.asarray([-1, 0, 1]),
        membership_rows=np.asarray([0, 1, 2]),
        minimum_rank_names=1,
    )

    np.testing.assert_array_equal(output[0], 0.0)
    assert not output_valid[0].any()
    # Decision row 1 uses market source row 0 but membership known at row 1.
    np.testing.assert_allclose(output[1, :, 0], [0.0, 0.2])
    np.testing.assert_array_equal(output_valid[1, :, 0], [False, True])
    # Decision row 2 advances exactly once to source row 1, not row 0 or 2.
    np.testing.assert_allclose(output[2, :, 0], [0.3, 0.4])
    assert output_valid[2].all()


def test_observation_age_uses_exchange_rows_and_preserves_known_staleness() -> None:
    valid = np.asarray(
        [
            [[False], [True]],
            [[True], [False]],
            [[False], [False]],
            [[False], [True]],
        ],
        dtype=np.bool_,
    )
    active = np.ones((4, 2), dtype=np.bool_)
    active[2, 1] = False
    output = np.empty(valid.shape, dtype=np.float32)

    observation_age_sessions_into(valid, active, output)

    np.testing.assert_array_equal(
        output[..., 0],
        np.asarray(
            [
                [-1.0, 0.0],
                [0.0, 1.0],
                [1.0, -1.0],
                [2.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_source_age_uses_raw_observations_before_output_and_not_rank_support() -> None:
    raw_valid = np.asarray(
        [[[True]], [[False]], [[False]], [[True]]], dtype=np.bool_
    )
    active = np.asarray([[True], [False], [True], [True]], dtype=np.bool_)
    output = np.empty((2, 1, 1), dtype=np.float32)

    observation_age_sessions_into(
        raw_valid,
        active,
        output,
        source_rows=np.asarray([2, 3]),
        decision_rows=np.asarray([2, 3]),
    )

    # The row-0 source survives an inactive row and a pre-window scan.
    np.testing.assert_array_equal(output[:, 0, 0], [2.0, 0.0])


def test_every_canonical_feature_has_field_level_semantics() -> None:
    specs = [*feature_specs("slow", SLOW_FEATURES)]
    specs.extend(feature_specs("intraday", INTRADAY_DAILY_FEATURES))
    for group, names in SIDECAR_FEATURES.items():
        specs.extend(feature_specs(f"sidecar_{group}", names))
    assert all(spec.source_units != "declared_raw_economic_unit" for spec in specs)
    assert all(not spec.formula.startswith("canonical_") for spec in specs)
    assert len({(spec.family, spec.name) for spec in specs}) == len(specs)


def test_activity_feature_specs_bind_inclusive_exact_session_windows() -> None:
    specs = {
        spec.name: spec
        for spec in feature_specs(
            "slow",
            (
                "log_volume_mean_20",
                "volume_zscore_20",
                "amihud_20",
                "trade_count_zscore_20",
                "turnover_proxy_20",
            ),
        )
    }
    assert "complete-source no-trade sessions are exact zero" in specs[
        "log_volume_mean_20"
    ].formula
    for name in (
        "log_volume_mean_20",
        "volume_zscore_20",
        "trade_count_zscore_20",
        "turnover_proxy_20",
    ):
        assert "inclusive 20" in specs[name].formula
    assert "exact 20" in specs["amihud_20"].formula


def test_signed_last_30_return_share_is_not_treated_as_a_fraction() -> None:
    spec = feature_specs(
        "intraday", ("last_30_minute_return_share_lag1",), minimum_rank_names=2
    )
    assert spec[0].transform == "rank_gauss"
    values = np.asarray([[[-2.0], [3.0]]], dtype=np.float64)
    valid = np.ones_like(values, dtype=np.bool_)
    output = np.empty_like(values, dtype=np.float32)
    output_valid = np.empty_like(valid)
    transform_feature_panel_into(
        values,
        valid,
        np.ones((1, 2), dtype=np.bool_),
        spec,
        output,
        output_valid,
        minimum_rank_names=2,
    )
    assert output_valid.all()
    assert output[0, 0, 0] < 0.0 < output[0, 1, 0]


def test_feature_schema_hash_binds_transform_and_order() -> None:
    left = (
        FeatureSpec("a", "x", "binary", "flag"),
        FeatureSpec("b", "x", "rank_gauss", "decimal"),
    )
    assert feature_schema_sha256(left) == feature_schema_sha256(left)
    assert feature_schema_sha256(left) != feature_schema_sha256(left[::-1])


def test_feature_schema_hash_binds_timing_formula_and_version() -> None:
    spec = FeatureSpec(
        "return_20",
        "slow",
        "rank_gauss",
        "decimal_return",
        availability_rule="prices through t-1 only",
        formula="log(close[t-1]/close[t-21])",
        minimum_support=20,
        validity_rule="21 observed closes and active at t",
        age_staleness_policy="no stale endpoint",
        version="2",
    )
    original = feature_schema_sha256((spec,))
    assert original != feature_schema_sha256(
        (replace(spec, availability_rule="prices through t only"),)
    )
    assert original != feature_schema_sha256(
        (replace(spec, formula="log(close[t]/close[t-20])"),)
    )
    assert original != feature_schema_sha256((replace(spec, version="3"),))


def test_native_fast_registry_binds_order_formulas_and_identity_transform() -> None:
    specs = native_fast_feature_specs()
    assert tuple(spec.name for spec in specs) == NATIVE_FAST_FEATURES
    assert all(spec.transform == "precomputed_native" for spec in specs)
    assert specs[0].formula == (
        "log(C_p/C_{p-1}) / (sigma_asof * sqrt(5/continuous_minutes))"
    )
    assert "prior 20 sessions" in specs[3].age_staleness_policy
    assert feature_schema_sha256(specs) != feature_schema_sha256(specs[::-1])
    assert feature_schema_sha256(specs) != feature_schema_sha256(
        (replace(specs[0], formula="different"), *specs[1:])
    )
