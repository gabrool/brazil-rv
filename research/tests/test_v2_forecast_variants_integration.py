"""A derived selection checkpoint scores through the registered path on CPU."""

import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from brazil_rv.v2 import round7_score
from brazil_rv.v2 import round7_training as training
from brazil_rv.v2.artifacts import sha256_file
from brazil_rv.v2.characteristic_model import CharacteristicModel
from brazil_rv.v2.forecast_variants import (
    completed_fit,
    compose,
    load_panel,
    member_epochs,
    scores_for_epoch,
)
from brazil_rv.v2.round7 import CELLS, configuration, pretrain_key
from brazil_rv.v2.round7_training import TrainingRecipe


def _fixture(tmp_path, monkeypatch):
    from test_v2_training import _tracked_pretrain_loaders
    from v2_store_fixtures import write_fixture_store

    factory, selection_loader = _tracked_pretrain_loaders(tmp_path)
    root = factory().dataset.store.root
    factory().dataset.store.close()
    selection_loader.dataset.store.close()
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["metadata"]["data_repair"] = {"synthetic_fixture": True}
    # The score-artifact loader requires the accepted stores' source-tier labels.
    manifest["metadata"]["action_terms_source"] = "inferred_cotahist_dismes_v1"
    manifest["metadata"]["schedule_source"] = "reconstructed_v1"
    root = write_fixture_store(
        tmp_path / "round7_fixture",
        dates=np.load(root / "date_index.npy") + np.timedelta64(2922, "D"),
        isins=np.load(root / "isin_index.npy").tolist(),
        arrays={
            name: np.load(root / record["path"])
            for name, record in manifest["arrays"].items()
        },
        feature_names=manifest["feature_names"],
        metadata=manifest["metadata"],
    )
    fit, selection, evaluation = (
        np.arange(100, 104),
        np.arange(120, 145),
        np.arange(160, 185),
    )

    def indices(*args):
        return fit, selection, evaluation, np.arange(100, 115)

    monkeypatch.setattr(training, "_cli_stage_indices", indices)
    monkeypatch.setattr(round7_score, "_cli_stage_indices", indices)
    monkeypatch.setattr(training, "_git_identity", lambda: {"commit": "f" * 40})
    monkeypatch.setattr(
        round7_score,
        "verify_reused_inference_source",
        lambda commit, **kwargs: {"training_commit": commit, "scoring_commit": commit},
    )
    cell = {**next(c for c in CELLS if c["cell"] == "B3"), "inputs": "slow"}
    config = configuration(cell, manifest["feature_names"])
    torch.manual_seed(22)
    parent = tmp_path / "parent.pt"
    torch.save(
        {
            "schema": training.CHECKPOINT_SCHEMA,
            "stage": "P",
            "seed": 11,
            "contract": {
                "pretrain_key": pretrain_key(cell),
                "unexposed_families": [],
                "preprocessing": {
                    "families": {},
                    "diagnostic": None,
                    "common_columns": [],
                    "per_name_columns": [],
                    "feature_names": {},
                },
                "config": asdict(config),
                "store_manifest_sha256": sha256_file(root / "manifest.json"),
            },
            "model_state_dict": CharacteristicModel(config).state_dict(),
        },
        parent,
    )
    return root, cell, parent, evaluation


def test_alternative_epoch_scores_match_the_trainer_export_format(
    tmp_path, monkeypatch
):
    root, cell, parent, evaluation = _fixture(tmp_path, monkeypatch)
    fit = tmp_path / "fits" / "B3" / "F1_seed_11"
    training.train(
        root,
        fit,
        cell=cell,
        recipe=TrainingRecipe(patience=9),
        stage="F",
        fold="F1",
        seed=11,
        epochs=4,
        parent=parent,
        parent_sha256=sha256_file(parent),
        device=torch.device("cpu"),
        compiled=False,
        export_scores=True,
        diagnostics=False,
    )
    manifest, history = completed_fit(fit)
    assert len(history) == 4 and (fit / "scores/score_manifest.json").exists()
    dates = np.load(root / "date_index.npy")[evaluation]
    isins = np.load(root / "isin_index.npy").tolist()
    schema = json.loads((root / "manifest.json").read_text())["metadata"][
        "feature_schema"
    ]["sha256"]
    own, record = scores_for_epoch(
        root,
        fit,
        manifest,
        manifest["selected_epoch"],
        tmp_path / "cache",
        view=member_epochs(history, "raw"),
        device=torch.device("cpu"),
    )
    assert own == fit / "scores" and record["source"] == "trainer_export"
    active = np.load(root / "active.npy")[evaluation]
    own_panel, valid = load_panel(
        own, dates=dates, isins=isins, feature_schema_sha256=schema, active=active
    )
    other = next(e for e in range(1, 5) if e != manifest["selected_epoch"])
    view = {"rule": "around3", "epochs": [other], "selection_score": 0.0}
    scores, record = scores_for_epoch(
        root,
        fit,
        manifest,
        other,
        tmp_path / "cache",
        view=view,
        device=torch.device("cpu"),
    )
    assert record["source"] == "derived_view_export"
    exported = json.loads((scores / "score_manifest.json").read_text())
    assert exported["status"] == "completed" and exported["test_accessed"] is False
    assert exported["checkpoint"]["sha256"] == record["checkpoint"]["sha256"]
    assert exported["round7_contract"] == manifest["contract"]
    assert exported["dataset"]["date_indices"] == evaluation.tolist()
    panel, other_valid = load_panel(
        scores, dates=dates, isins=isins, feature_schema_sha256=schema, active=active
    )
    np.testing.assert_array_equal(valid, other_valid)
    assert panel.shape == own_panel.shape and np.isfinite(panel[valid]).all()
    assert not np.allclose(panel[valid], own_panel[valid])
    blend = compose([panel, own_panel])
    assert np.all(np.abs(blend[valid]) <= 1.0)
    with pytest.raises(ValueError):
        scores_for_epoch(
            root,
            fit,
            manifest,
            other,
            tmp_path / "cache",
            view={"rule": "top3", "epochs": [1, 2], "selection_score": 0.0},
            device=torch.device("cpu"),
        )
