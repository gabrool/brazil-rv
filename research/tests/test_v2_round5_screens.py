import json
from dataclasses import asdict
from datetime import date, timedelta

import numpy as np

from brazil_rv.v2 import round5_screens as screen
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.gbdt import GBDTConfig, MultiHorizonGBDT


def test_fit_only_magnitude_bounds_ignore_later_rows_and_keep_global_identity():
    rng = np.random.default_rng(0)
    encoded = rng.normal(size=(20, 25, 8)).astype(np.float32)
    encoded[0, 0, 0] = np.nan
    active = np.ones((20, 25), bool)
    local, global_rows = np.arange(10), np.arange(100, 110)
    first = screen.clip_magnitudes(encoded, active, local, global_rows)
    encoded[10:, :, :4] = 1e8
    second = screen.clip_magnitudes(encoded, active, local, global_rows)
    assert first.payload() == second.payload()
    assert first.fit_date_indices == tuple(global_rows)
    joined = screen._features(encoded[..., :2], encoded, np.arange(20), first)
    assert np.isnan(joined[0, 0, 2])
    assert np.all(joined[10:, :, 2:6] <= first.upper)


def test_targets_respect_fit_purge_and_future_mutation():
    values = np.arange(40 * 25 * 5).reshape(40, 25, 5)
    valid = np.ones_like(values, bool)
    indices = np.arange(100, 110)
    first, mask = screen.window_targets(
        values, valid, indices, 100, np.arange(100, 120)
    )
    assert mask.all()  # The explicit purge interval supports the last fit labels.
    values[10:] = -100
    second, _ = screen.window_targets(values, valid, indices, 100, np.arange(100, 120))
    np.testing.assert_array_equal(first, second)
    _, short = screen.window_targets(values, valid, indices, 100, indices)
    assert not short[-1].any()
    assert not short[..., 4].any()


def test_primary_readout_uses_all_three_traded_heads_on_one_population():
    values = np.broadcast_to(np.arange(25)[None, :, None], (2, 25, 5)).astype(float)
    mask = np.ones_like(values, bool)
    mask[..., :2] = False
    active = np.ones((2, 25), bool)
    days = [date(2024, 1, 2), date(2024, 1, 3)]
    first = screen.primary_daily(values, values, mask, active, days)
    np.testing.assert_allclose(first, 1)
    mask[1, :6, 4] = False
    second = screen.primary_daily(values, values, mask, active, days)
    assert second[0] == 1 and np.isnan(second[1])


def test_shap_sampling_is_per_day_and_uses_only_eligible_names():
    eligible = np.zeros((3, 100), bool)
    eligible[0, 10:90] = True
    eligible[1, 50:52] = True
    rows = screen.shap_coordinates(eligible)
    assert len(rows) == 18
    assert eligible[rows[:, 0], rows[:, 1]].all()
    assert set(rows[:, 0]) == {0, 1}


def test_synthetic_cell_roundtrip_and_partial_resume_do_not_refit(
    tmp_path, monkeypatch
):
    rng = np.random.default_rng(41)
    count, names = 90, 25
    cache = tmp_path / "cache"
    cache.mkdir()
    active = np.ones((count, names), bool)
    values = rng.normal(size=(count, names, 4)).astype(np.float32)
    targets = np.repeat(values[..., :1], 5, axis=-1)
    arrays = {
        "active": active,
        "targets": targets,
        "target_mask": np.ones_like(targets, bool),
        "dates": np.array(
            [date(2024, 1, 1) + timedelta(days=i) for i in range(count)],
            dtype="datetime64[D]",
        ),
        "global_indices": np.arange(100, 100 + count),
    }
    bindings = {key: screen._save(cache, key, value) for key, value in arrays.items()}
    config = GBDTConfig(
        seeds=(11,),
        maximum_rounds=3,
        early_stopping_rounds=2,
        min_data_in_leaf=5,
        num_threads=1,
    )
    record = {
        "values": screen._save(cache, "a_slow", values),
        "presence": screen._save(cache, "presence", active),
        "encoded_names": ["x", "y", "x_age", "y_age"],
        "coverage": {
            "F1": {
                "fit": 1000,
                "selection": 500,
                "evaluation": 750,
                "informative": True,
            }
        },
    }
    design = {
        "config": asdict(config),
        "cache_start_global_index": 100,
        "arrays": bindings,
        "families": {"a_slow": record},
        "fit": {"F1": list(range(100, 140))},
        "selection": {"F1": list(range(150, 165))},
        "evaluation": {"F1": list(range(165, 190))},
        "fit_target_windows": {"F1": list(range(100, 150))},
        "tree_shap_names_per_day": 2,
    }
    path = tmp_path / "design.json"
    write_json_atomic(path, design)
    result = screen.run_cell(str(path), "a_slow", "F1", 11)
    assert result["status"] == "completed"
    root = tmp_path / "cells/a_slow/F1/seed_11"
    saved = json.loads((root / "result.json").read_text())
    assert saved["outputs"]["models"]["path"].endswith("model_manifest.json")
    assert len(saved["tree_shap_coordinates"]) == 50
    assert "mean_abs_tree_shap" in saved["importance"]

    def cannot_fit(*args, **kwargs):
        raise AssertionError("resume repeated the fit")

    monkeypatch.setattr(MultiHorizonGBDT, "fit", cannot_fit)
    assert screen.run_cell(str(path), "a_slow", "F1", 11)["status"] == "reused_complete"
    # Simulate interruption after atomic model save, before the completion marker.
    (root / "result.json").unlink()
    screen.run_cell(str(path), "a_slow", "F1", 11)
    assert json.loads((root / "result.json").read_text())["outputs"] == saved["outputs"]
