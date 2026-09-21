import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from brazil_rv.v2 import round7_training as training
from brazil_rv.v2.artifacts import sha256_file
from brazil_rv.v2.characteristic_model import CharacteristicModel
from brazil_rv.v2.round7 import CELLS, configuration, pretrain_key
from brazil_rv.v2.round7_training import TrainingRecipe, recipe_contract

# Captured once so repeated scripting wraps the real readout, not a spent wrapper.
_ORIGINAL_READOUT = training.selection_readout


def test_default_recipe_contract_is_byte_identical_to_the_historical_payload():
    recipe = TrainingRecipe()
    assert "selection_smoothing" not in recipe_contract(recipe)
    assert recipe_contract(recipe) == {
        k: v for k, v in asdict(recipe).items() if k != "selection_smoothing"
    }
    assert (
        recipe_contract(TrainingRecipe(selection_smoothing=3))["selection_smoothing"]
        == 3
    )


def _fixture(tmp_path, monkeypatch):
    from test_v2_training import _tracked_pretrain_loaders
    from v2_store_fixtures import write_fixture_store

    factory, selection_loader = _tracked_pretrain_loaders(tmp_path)
    root = factory().dataset.store.root
    factory().dataset.store.close()
    selection_loader.dataset.store.close()
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["metadata"]["data_repair"] = {"synthetic_fixture": True}
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
    monkeypatch.setattr(training, "_git_identity", lambda: {"commit": "f" * 40})
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
    return root, cell, parent, selection


def _scripted_selection(monkeypatch, sequence, selection):
    calls = {"n": 0}

    def scripted(model, loader, *args, **kwargs):
        value = _ORIGINAL_READOUT(model, loader, *args, **kwargs)
        if np.array_equal(loader.dataset.date_indices, selection):
            value["mean_ic"] = sequence[calls["n"]]
            calls["n"] += 1
        return value

    monkeypatch.setattr(training, "selection_readout", scripted)


def test_smoothed_selection_selects_the_window_centre_and_resumes_identically(
    tmp_path, monkeypatch
):
    root, cell, parent, selection = _fixture(tmp_path, monkeypatch)
    sequence = [0.01, 0.05, 0.02, 0.03, 0.10, 0.00, 0.00, 0.00]
    options = {
        "cell": cell,
        "recipe": TrainingRecipe(patience=9, selection_smoothing=3),
        "stage": "F",
        "fold": "F1",
        "seed": 11,
        "epochs": 8,
        "parent": parent,
        "parent_sha256": sha256_file(parent),
        "device": torch.device("cpu"),
        "compiled": False,
        "export_scores": False,
        "diagnostics": False,
    }
    _scripted_selection(monkeypatch, sequence, selection)
    full = tmp_path / "full"
    result = training.train(root, full, **options)
    # Trailing-3 means peak at epoch 5 (mean of epochs 3-5); its centre is epoch 4.
    assert result["selected_epoch"] == 4 and result["selection_smoothing"] == 3
    assert result["contract"]["recipe"]["selection_smoothing"] == 3
    assert result["selection_ic"] == pytest.approx(np.mean(sequence[2:5]))
    selected = torch.load(full / "selected.pt", weights_only=True)
    assert selected["epoch"] == 4
    epoch_four = torch.load(full / "epochs" / "epoch_004.pt", weights_only=True)
    for key, value in selected["model_state_dict"].items():
        torch.testing.assert_close(
            value, epoch_four["model_state_dict"][key], atol=0, rtol=0
        )
    history = json.loads((full / "history.json").read_text())
    assert [r["selected_epoch"] for r in history if r["selected"]][-1] == 4
    assert history[4]["selection_score"] == pytest.approx(np.mean(sequence[2:5]))
    assert not (full / "resume.pt").exists()

    _scripted_selection(monkeypatch, sequence, selection)
    original_write = training.write_json_atomic

    def interrupt(path, payload):
        if path.name == "history.json" and len(payload) == 6:
            raise RuntimeError("simulated interruption after durable epoch")
        return original_write(path, payload)

    resumed = tmp_path / "resumed"
    monkeypatch.setattr(training, "write_json_atomic", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        training.train(root, resumed, **options)
    monkeypatch.setattr(training, "write_json_atomic", original_write)
    training.train(root, resumed, **options)
    again = torch.load(resumed / "selected.pt", weights_only=True)
    assert again["epoch"] == selected["epoch"]
    for key, value in selected["model_state_dict"].items():
        torch.testing.assert_close(
            value, again["model_state_dict"][key], atol=0, rtol=0
        )


def test_raw_selection_is_unchanged_by_the_smoothing_option(tmp_path, monkeypatch):
    root, cell, parent, selection = _fixture(tmp_path, monkeypatch)
    sequence = [0.01, 0.05, 0.02, 0.03, 0.10, 0.00]
    options = {
        "cell": cell,
        "recipe": TrainingRecipe(patience=9),
        "stage": "F",
        "fold": "F1",
        "seed": 11,
        "epochs": 6,
        "parent": parent,
        "parent_sha256": sha256_file(parent),
        "device": torch.device("cpu"),
        "compiled": False,
        "export_scores": False,
        "diagnostics": False,
    }
    _scripted_selection(monkeypatch, sequence, selection)
    result = training.train(root, tmp_path / "raw", **options)
    assert result["selected_epoch"] == 5 and "selection_smoothing" not in result
    assert "selection_smoothing" not in result["contract"]["recipe"]
    history = json.loads((tmp_path / "raw" / "history.json").read_text())
    assert "selection_score" not in history[0] and "selected_epoch" not in history[0]
