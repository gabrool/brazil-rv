from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import brazil_rv.v2.gbdt as gbdt_module
from brazil_rv.v2.data import ScalarFeatureView
from brazil_rv.v2.gbdt import (
    GBDTConfig,
    LightGBMUnavailable,
    MultiHorizonGBDT,
    assemble_gbdt_features,
    assemble_gbdt_scalar_view,
    gbdt_scalar_feature_names,
    require_lightgbm,
)


def test_missing_lightgbm_has_clear_install_error(monkeypatch) -> None:
    monkeypatch.setattr(gbdt_module, "lgb", None)
    with pytest.raises(LightGBMUnavailable, match="uv add lightgbm"):
        require_lightgbm()


def test_gbdt_feature_assembly_uses_last_slow_step_and_fast_flag() -> None:
    slow = np.arange(2 * 3 * 4 * 2, dtype=np.float32).reshape(2, 3, 4, 2)
    intraday = np.ones((2, 3, 3), dtype=np.float32)
    present = np.array([[True, False, True], [False, True, False]])
    slow_age = np.full_like(slow, 4.0)
    intraday_age = np.full_like(intraday, 2.0)
    actual = assemble_gbdt_features(
        slow,
        intraday,
        present,
        slow_feature_mask=np.ones_like(slow, dtype=np.bool_),
        intraday_feature_mask=np.ones_like(intraday, dtype=np.bool_),
        slow_feature_age_sessions=slow_age,
        intraday_feature_age_sessions=intraday_age,
    )
    assert actual.shape == (2, 3, 11)
    assert np.array_equal(actual[..., :2], slow[:, :, -1])
    assert np.allclose(actual[..., 2:4], np.log1p(4.0) / np.log1p(252.0))
    assert np.array_equal(actual[..., 4:7], intraday)
    assert np.allclose(actual[..., 7:10], np.log1p(2.0) / np.log1p(252.0))
    assert np.array_equal(actual[..., -1], present)


def test_gbdt_scalar_adapter_preserves_view_order_masks_ages_and_axes() -> None:
    values = np.asarray([[[1.0, 99.0], [3.0, 4.0]]], dtype=np.float32)
    valid = np.asarray([[[True, False], [True, True]]], dtype=np.bool_)
    ages = np.asarray([[[0.0, -1.0], [4.0, 252.0]]], dtype=np.float32)
    view = ScalarFeatureView(
        date_indices=np.asarray([7], dtype=np.int64),
        dates=np.asarray(["2024-01-11"], dtype="datetime64[D]"),
        isins=("BR1", "BR2"),
        active=np.asarray([[True, False]], dtype=np.bool_),
        names=("alpha", "beta"),
        values=values,
        valid=valid,
        age_sessions=ages,
    )

    actual = assemble_gbdt_scalar_view(view)

    assert gbdt_scalar_feature_names(view.names) == (
        "alpha",
        "beta",
        "alpha__age_sessions",
        "beta__age_sessions",
    )
    assert actual.shape == (1, 2, 4)
    np.testing.assert_array_equal(actual[..., :2][valid], values[valid])
    assert np.isnan(actual[0, 0, 1])
    assert np.isnan(actual[0, 0, 3])
    assert actual[0, 1, 2] == pytest.approx(np.log1p(4.0) / np.log1p(252.0))
    assert actual[0, 1, 3] == pytest.approx(1.0)
    assert view.date_indices.tolist() == [7]
    assert view.isins == ("BR1", "BR2")
    assert view.active.tolist() == [[True, False]]


def test_gbdt_feature_masks_become_nan_and_nonfinite_valid_cells_fail() -> None:
    slow = np.ones((1, 2, 3, 2), dtype=np.float32)
    intraday = np.ones((1, 2, 2), dtype=np.float32)
    slow_mask = np.ones_like(slow, dtype=np.bool_)
    intraday_mask = np.ones_like(intraday, dtype=np.bool_)
    slow_age = np.zeros_like(slow, dtype=np.float32)
    intraday_age = np.zeros_like(intraday, dtype=np.float32)
    slow_mask[0, 0, -1, 1] = False
    slow_age[0, 0, -1, 1] = 7.0
    intraday_mask[0, 1, 0] = False
    intraday_age[0, 1, 0] = -1.0
    slow[0, 0, -1, 1] = 99.0
    intraday[0, 1, 0] = -99.0
    actual = assemble_gbdt_features(
        slow,
        intraday,
        np.ones((1, 2), dtype=np.bool_),
        slow_feature_mask=slow_mask,
        intraday_feature_mask=intraday_mask,
        slow_feature_age_sessions=slow_age,
        intraday_feature_age_sessions=intraday_age,
    )
    assert np.isnan(actual[0, 0, 1])
    assert actual[0, 0, 3] == pytest.approx(np.log1p(7.0) / np.log1p(252.0))
    assert np.isnan(actual[0, 1, 4])
    assert np.isnan(actual[0, 1, 6])

    slow[0, 1, -1, 0] = np.nan
    with pytest.raises(ValueError, match="valid slow"):
        assemble_gbdt_features(
            slow,
            intraday,
            np.ones((1, 2), dtype=np.bool_),
            slow_feature_mask=slow_mask,
            intraday_feature_mask=intraday_mask,
            slow_feature_age_sessions=slow_age,
            intraday_feature_age_sessions=intraday_age,
        )
    slow[0, 1, -1, 0] = np.inf
    slow_mask[0, 1, -1, 0] = False
    with pytest.raises(ValueError, match="infinities"):
        assemble_gbdt_features(
            slow,
            intraday,
            np.ones((1, 2), dtype=np.bool_),
            slow_feature_mask=slow_mask,
            intraday_feature_mask=intraday_mask,
            slow_feature_age_sessions=slow_age,
            intraday_feature_age_sessions=intraday_age,
        )


def test_gbdt_panel_rejects_nonfinite_valid_targets_and_infinite_features() -> None:
    features = np.ones((2, 3, 4), dtype=np.float32)
    targets = np.ones((2, 3, 5), dtype=np.float32)
    mask = np.ones_like(targets, dtype=np.bool_)
    targets[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="valid GBDT targets"):
        gbdt_module._validate_panel(features, targets, mask)
    mask[0, 0, 0] = False
    features[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="infinities"):
        gbdt_module._validate_panel(features, targets, mask)


def test_five_head_gbdt_round_trip_and_importances(tmp_path: Path) -> None:
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(29)
    features = rng.normal(size=(10, 16, 4)).astype(np.float32)
    targets = np.stack(
        [
            features[..., head % 4] + 0.01 * rng.normal(size=features.shape[:2])
            for head in range(5)
        ],
        axis=-1,
    ).astype(np.float32)
    mask = np.ones_like(targets, dtype=bool)
    config = GBDTConfig(
        min_data_in_leaf=2,
        maximum_rounds=20,
        early_stopping_rounds=3,
        seeds=(11,),
        num_threads=1,
    )
    model = MultiHorizonGBDT(config, feature_names=("a", "b", "c", "d"))
    model.fit(
        features[:7],
        targets[:7],
        mask[:7],
        features[7:],
        targets[7:],
        mask[7:],
    )
    scores = model.predict_ranks(features[7:], mask[7:])
    assert scores.shape == (3, 16, 5)
    assert scores[0, :, 0].mean() == pytest.approx(7.5)
    assert scores[0, :, 0].min() >= 0.0
    assert scores[0, :, 0].max() <= 15.0
    importance = model.feature_importance(features[7:])
    assert importance["gain"].shape == (4,)
    assert importance["mean_abs_tree_shap"].shape == (4,)

    repeated = MultiHorizonGBDT(config, feature_names=("a", "b", "c", "d"))
    repeated.fit(
        features[:7],
        targets[:7],
        mask[:7],
        features[7:],
        targets[7:],
        mask[7:],
    )
    assert np.array_equal(scores, repeated.predict_ranks(features[7:], mask[7:]))

    raw_before = model.predict_raw(features[7:])
    model_root = tmp_path / "models"
    manifest_path, manifest_sha256 = model.save(
        model_root, metadata={"status": "completed"}
    )
    loaded = MultiHorizonGBDT.load(
        model_root, expected_manifest_sha256=manifest_sha256
    )
    assert np.array_equal(raw_before, loaded.predict_raw(features[7:]))
    assert np.array_equal(scores, loaded.predict_ranks(features[7:], mask[7:]))
    assert manifest_path.is_file()

    (model_root / "head_0_seed_11.txt").write_text("tampered", encoding="ascii")
    with pytest.raises(ValueError, match="member hash or size"):
        MultiHorizonGBDT.load(
            model_root, expected_manifest_sha256=manifest_sha256
        )
