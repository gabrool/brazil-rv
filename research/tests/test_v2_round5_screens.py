import json
from concurrent.futures import Future
from dataclasses import asdict
from datetime import date, timedelta

import numpy as np
import pytest

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
        "model_program_sha256": screen.model_program_identity(),
        "library_versions": screen.library_versions(),
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


@pytest.fixture
def sealed_summary(tmp_path):
    """Bound synthetic cells exercise summary integrity without research fits."""
    count, names = 30, 25
    cache = tmp_path / "cache"
    cache.mkdir()
    active = np.ones((count, names), bool)
    values = np.broadcast_to(np.arange(names)[None, :, None], (count, names, 5)).astype(
        np.float32
    )
    arrays = {
        "active": active,
        "targets": values,
        "target_mask": np.ones_like(values, bool),
        "dates": np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-01-31")),
    }
    bindings = {key: screen._save(cache, key, value) for key, value in arrays.items()}
    features = screen._save(cache, "features", values[..., :2])
    parent = {
        "values": features,
        "presence": screen._save(cache, "parent_presence", active),
        "coverage": {"F1": {"fit": 1, "evaluation": 1, "informative": True}},
    }
    events = {
        "values": features,
        "presence": screen._save(cache, "events_presence", ~active),
        "coverage": {"F1": {"fit": 0, "evaluation": 0, "informative": False}},
    }
    design = {
        "model_program_sha256": screen.model_program_identity(),
        "library_versions": screen.library_versions(),
        "amendments": [],
        "config": {"seeds": [11]},
        "cache_start_global_index": 0,
        "arrays": bindings,
        "families": {"events": events, "a_slow": parent},
        "evaluation": {"F1": list(range(count))},
    }
    path = tmp_path / "design.json"
    # Put the skipped family first, so its parent binding must be checked on the
    # fallback path rather than incidentally through a prior baseline readout.
    path.write_text(json.dumps(design), encoding="utf8")
    for family in design["families"]:
        root = tmp_path / "cells" / family / "F1" / "seed_11"
        root.mkdir(parents=True)
        result = {
            "design_sha256": screen.sha256_file(path),
            "family": family,
            "fold": "F1",
            "seed": 11,
            "outputs": {},
        }
        if family == "events":
            result["status"] = "no_family_observation_in_fit_use_parent"
        else:
            result.update(
                {
                    "status": "completed",
                    "mean_primary_ic": 1.0,
                    "feature_names": ["x", "x_age"],
                    "importance": {"gain": [1.0, 0.0]},
                    "outputs": {"scores": screen._save(root, "scores", values)},
                }
            )
        write_json_atomic(root / "result.json", result)
    return path, design


def test_standalone_summary_verifies_and_reuses_the_matched_parent(sealed_summary):
    path, _ = sealed_summary
    result = screen.summarize(path)
    assert result["seeds"] == [11]
    assert (
        result["families"]["events"]["all_folds"]["paired_delta_vs_a_slow"]["estimate"]
        == 0
    )
    assert (
        result["families"]["events"]["informative_folds"]["status"]
        == "no_informative_folds"
    )


@pytest.mark.parametrize(
    "dependency",
    [
        "baselines",
        "validate_pipeline",
        "evaluate",
        "research_rounds",
        "contract",
        "splits",
        "config",
    ],
)
def test_semantic_dependency_change_blocks_summary_and_worker_resume(
    sealed_summary, monkeypatch, dependency
):
    path, _ = sealed_summary
    actual_hash = screen.sha256_file
    monkeypatch.setattr(
        screen,
        "sha256_file",
        lambda p: "changed" if p.name == dependency + ".py" else actual_hash(p),
    )
    with pytest.raises(ValueError, match="semantic implementation"):
        screen.summarize(path)
    with pytest.raises(ValueError, match="semantic implementation"):
        screen.run_cell(str(path), "a_slow", "F1", 11)
    assert not (path.parent / "readout.json").exists()


def test_library_change_blocks_coordinator_summary_and_worker_resume(
    sealed_summary, monkeypatch
):
    path, _ = sealed_summary
    changed = {**screen.library_versions(), "lightgbm": "different-build"}
    monkeypatch.setattr(screen, "library_versions", lambda: changed)
    for operation in (
        lambda: screen.run(path),
        lambda: screen.summarize(path),
        lambda: screen.run_cell(str(path), "a_slow", "F1", 11),
    ):
        with pytest.raises(ValueError, match="library versions"):
            operation()
    assert not (path.parent / "execution.json").exists()


@pytest.mark.parametrize("cache", ["targets", "family_presence"])
def test_standalone_summary_rejects_corrupted_input_cache(sealed_summary, cache):
    path, design = sealed_summary
    record = (
        design["arrays"]["targets"]
        if cache == "targets"
        else design["families"]["events"]["presence"]
    )
    with open(record["path"], "ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="artifact identity differs"):
        screen.summarize(path)


@pytest.mark.parametrize(
    "family,field,value",
    [
        ("events", "design_sha256", "another-design"),
        ("a_slow", "design_sha256", "another-design"),
        ("a_slow", "seed", 29),
    ],
)
def test_summary_rejects_mixed_cells_including_reused_parent(
    sealed_summary, family, field, value
):
    path, _ = sealed_summary
    cell = path.parent / "cells" / family / "F1/seed_11/result.json"
    result = json.loads(cell.read_text())
    result[field] = value
    write_json_atomic(cell, result)
    with pytest.raises(ValueError, match="another screen design or cell identity"):
        screen.summarize(path)
    assert not (path.parent / "readout.json").exists()


def test_failed_worker_cancels_queued_program_and_preserves_original_error(
    sealed_summary, monkeypatch
):
    path, _ = sealed_summary
    shutdowns, submitted = [], []

    class Pool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.shutdown(wait=True)

        def submit(self, *_):
            future = Future()
            if not submitted:
                future.set_exception(RuntimeError("synthetic worker failure"))
            submitted.append(future)
            return future

        def shutdown(self, *, wait=True, cancel_futures=False):
            shutdowns.append((wait, cancel_futures))
            if cancel_futures:
                for future in submitted:
                    future.cancel()

    monkeypatch.setattr(screen, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(screen.rr, "_git_identity", lambda: {"commit": "fixture"})
    with pytest.raises(RuntimeError, match="synthetic worker failure"):
        screen.run(path)
    assert shutdowns[0] == (True, True)
    assert submitted[1].cancelled()
    assert not (path.parent / "readout.json").exists()
