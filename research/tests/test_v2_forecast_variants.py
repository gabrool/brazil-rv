import json

import numpy as np
import pytest
import torch

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.contract import HORIZONS, SCORE_ARTIFACT_SCHEMA
from brazil_rv.v2.forecast_variants import (
    completed_fit,
    compose,
    derived_checkpoint,
    epoch_checkpoint,
    load_panel,
    member_epochs,
    parse_variant,
    scores_for_epoch,
)
from brazil_rv.v2.portfolio_inputs import HEADS
from brazil_rv.v2.score import _array_record


def _fake_fit(tmp_path, scores, selected_epoch):
    fit = tmp_path / "fits" / "TE_full_p20" / "F2_seed_11"
    (fit / "epochs").mkdir(parents=True)
    history, best = [], -np.inf
    for epoch, value in enumerate(scores, start=1):
        improved = value > best + 1e-4
        best = value if improved else best
        history.append(
            {
                "epoch": epoch,
                "selection": {"mean_ic": value, "daily_ic": [value] * 45},
                "selected": improved,
            }
        )
    write_json_atomic(fit / "history.json", history)
    artifacts = {"history.json": sha256_file(fit / "history.json")}
    contract = {"padded_name_count": 4, "store_manifest_sha256": "0" * 64}
    for epoch in range(1, len(scores) + 1):
        payload = {
            "schema": "BRAZIL_RV_SELECTED_CHECKPOINT_V1",
            "stage": "F",
            "seed": 11,
            "fold": "F2",
            "epoch": epoch,
            "contract": contract,
            "model_state_dict": {"weight": torch.full((2,), float(epoch))},
        }
        relative = f"epochs/epoch_{epoch:03d}.pt"
        torch.save(payload, fit / relative)
        artifacts[relative] = sha256_file(fit / relative)
        if epoch == selected_epoch:
            torch.save(
                {**payload, "selection_ic": scores[epoch - 1]}, fit / "selected.pt"
            )
            artifacts["selected.pt"] = sha256_file(fit / "selected.pt")
    manifest = {
        "status": "completed",
        "seed": 11,
        "fold": "F2",
        "selected_epoch": selected_epoch,
        "epochs_completed": len(scores),
        "contract": contract,
        "artifacts": artifacts,
    }
    write_json_atomic(fit / "run_manifest.json", manifest)
    return fit


def _score_export(directory, dates, isins, values, mask, checkpoint_sha256, schema):
    directory.mkdir(parents=True)
    arrays = {
        "scores.npy": values.astype(np.float32),
        "score_mask.npy": mask,
        "date_index.npy": dates,
        "isin_index.npy": np.asarray(isins),
    }
    records = {}
    for name, value in arrays.items():
        np.save(directory / name, value, allow_pickle=False)
        records[name] = _array_record(directory / name, value)
    write_json_atomic(
        directory / "score_manifest.json",
        {
            "schema": SCORE_ARTIFACT_SCHEMA,
            "status": "completed",
            "checkpoint": {"sha256": checkpoint_sha256},
            "transfer_chronology_clean": True,
            "feature_schema_sha256": schema,
            "action_terms_source": "inferred_cotahist_dismes_v1",
            "schedule_source": "reconstructed_v1",
            "artifacts": records,
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )


def test_completed_fit_and_epoch_checkpoints_are_hash_bound(tmp_path):
    fit = _fake_fit(tmp_path, [0.01, 0.05, 0.02, 0.03, 0.10, 0.0, 0.0], 5)
    manifest, history = completed_fit(fit)
    assert len(history) == 7 and manifest["selected_epoch"] == 5
    path, digest = epoch_checkpoint(fit, manifest, 5)
    assert path.name == "selected.pt" and digest == manifest["artifacts"]["selected.pt"]
    path, _ = epoch_checkpoint(fit, manifest, 4)
    assert path.name == "epoch_004.pt"
    with pytest.raises(ValueError):
        epoch_checkpoint(fit, manifest, 9)
    (fit / "history.json").write_text("[]")
    with pytest.raises(ValueError):
        completed_fit(fit)


def test_derived_checkpoint_binds_the_saved_epoch_and_is_reused(tmp_path):
    fit = _fake_fit(tmp_path, [0.01, 0.05, 0.02, 0.03, 0.10, 0.0, 0.0], 5)
    manifest, history = completed_fit(fit)
    view = member_epochs(history, "centre3")
    assert view["epochs"] == [4]
    destination = tmp_path / "cache" / "epoch_004"
    target, digest = derived_checkpoint(fit, manifest, 4, destination, view=view)
    payload = torch.load(target, weights_only=True)
    assert payload["epoch"] == 4 and payload["selection_ic"] == pytest.approx(
        view["selection_score"]
    )
    torch.testing.assert_close(
        payload["model_state_dict"]["weight"], torch.full((2,), 4.0)
    )
    assert (
        payload["selection_view"]["source_checkpoint"]["sha256"]
        == manifest["artifacts"]["epochs/epoch_004.pt"]
    )
    again, same = derived_checkpoint(fit, manifest, 4, destination, view=view)
    assert again == target and same == digest
    other = member_epochs(history, "top3")
    with pytest.raises(ValueError):
        derived_checkpoint(fit, manifest, 4, destination, view=other)
    with pytest.raises(ValueError):
        derived_checkpoint(
            fit, manifest, 5, tmp_path / "cache" / "epoch_005", view=view
        )


def test_scores_for_epoch_reuses_the_trainer_export_and_scores_others_once(
    tmp_path, monkeypatch
):
    fit = _fake_fit(tmp_path, [0.01, 0.05, 0.02, 0.03, 0.10, 0.0, 0.0], 5)
    manifest, history = completed_fit(fit)
    dates = np.array(["2020-01-06", "2020-01-07"], dtype="datetime64[D]")
    isins = ["A", "B", "C", "D"]
    values = np.random.default_rng(1).normal(size=(2, 4, len(HORIZONS)))
    mask = np.ones((2, 4, len(HORIZONS)), bool)
    _score_export(
        fit / "scores",
        dates,
        isins,
        values,
        mask,
        manifest["artifacts"]["selected.pt"],
        "s" * 64,
    )
    scores, record = scores_for_epoch(
        tmp_path / "store",
        fit,
        manifest,
        5,
        tmp_path / "cache",
        view=None,
        device="cpu",
    )
    assert scores == fit / "scores" and record["source"] == "trainer_export"
    calls = []

    def fake_score(store_root, checkpoint, output, *, expected_sha256, **kwargs):
        calls.append((checkpoint, expected_sha256, kwargs["fixed_name_count"]))
        _score_export(
            output, dates, isins, values * 0.5, mask, expected_sha256, "s" * 64
        )

    from brazil_rv.v2 import round7_score

    monkeypatch.setattr(round7_score, "score", fake_score)
    view = member_epochs(history, "centre3")
    first, record = scores_for_epoch(
        tmp_path / "store",
        fit,
        manifest,
        4,
        tmp_path / "cache",
        view=view,
        device="cpu",
    )
    second, _ = scores_for_epoch(
        tmp_path / "store",
        fit,
        manifest,
        4,
        tmp_path / "cache",
        view=view,
        device="cpu",
    )
    assert first == second == tmp_path / "cache" / "epoch_004" / "scores"
    assert len(calls) == 1 and calls[0][2] == 4
    assert calls[0][1] == sha256_file(tmp_path / "cache" / "epoch_004" / "selected.pt")
    assert record["source"] == "derived_view_export"
    panel, valid = load_panel(
        first,
        dates=dates,
        isins=isins,
        feature_schema_sha256="s" * 64,
        active=np.ones((2, 4), bool),
    )
    assert panel.shape == (2, 4, len(HEADS)) and valid.all()
    assert np.allclose(sorted(panel[0, :, 0]), [-0.75, -0.25, 0.25, 0.75])
    # An inactive name is simply ineligible; a masked-out active name is a loss.
    _, valid = load_panel(
        first,
        dates=dates,
        isins=isins,
        feature_schema_sha256="s" * 64,
        active=np.array([[True, True, True, True], [True, True, False, True]]),
    )
    assert valid.sum() == 7
    dropped = mask.copy()
    dropped[1, 2, :] = False
    _score_export(
        tmp_path / "dropped", dates, isins, values, dropped, "d" * 64, "s" * 64
    )
    with pytest.raises(ValueError, match="lost eligible names"):
        load_panel(
            tmp_path / "dropped",
            dates=dates,
            isins=isins,
            feature_schema_sha256="s" * 64,
            active=np.ones((2, 4), bool),
        )


def test_compose_is_the_equal_weight_rank_mean_and_variants_parse():
    a = np.array([[[0.5, -0.5]]], np.float32)
    b = np.array([[[-0.5, 0.5]]], np.float32)
    np.testing.assert_allclose(compose([a, b]), [[[0.0, 0.0]]])
    with pytest.raises(ValueError):
        compose([a, np.zeros((1, 1, 3), np.float32)])
    assert parse_variant("TE_full_p20@centre3") == (
        "TE_full_p20@centre3",
        [("TE_full_p20", "centre3")],
    )
    assert parse_variant("ENS=C6+TE_full_p20@top3") == (
        "ENS",
        [("C6", "raw"), ("TE_full_p20", "top3")],
    )
    with pytest.raises(ValueError):
        parse_variant("C6+TE_full_p20")
    record = json.loads(
        json.dumps(
            member_epochs(
                [{"epoch": 1, "selection": {"mean_ic": 0.1}, "selected": True}], "raw"
            )
        )
    )
    assert record["epochs"] == [1]
