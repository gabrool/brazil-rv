from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import brazil_rv.v2.gbdt as gbdt_module
from brazil_rv.v2.gbdt import (
    GBDTConfig,
    LightGBMUnavailable,
    MultiHorizonGBDT,
    assemble_gbdt_features,
    require_lightgbm,
)


def test_missing_lightgbm_has_clear_install_error(monkeypatch) -> None:
    monkeypatch.setattr(gbdt_module, "lgb", None)
    with pytest.raises(LightGBMUnavailable, match="uv add lightgbm"):
        require_lightgbm()


def test_gbdt_feature_assembly_uses_last_slow_step_and_flags() -> None:
    slow = np.arange(2 * 3 * 4 * 2, dtype=np.float32).reshape(2, 3, 4, 2)
    intraday = np.ones((2, 3, 3), dtype=np.float32)
    present = np.array([[True, False, True], [False, True, False]])
    days = np.ones((2, 3), dtype=np.float32)
    actual = assemble_gbdt_features(slow, intraday, present, days)
    assert actual.shape == (2, 3, 7)
    assert np.array_equal(actual[..., :2], slow[:, :, -1])
    assert np.array_equal(actual[..., -2], present)
    assert np.array_equal(actual[..., -1], days)


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
