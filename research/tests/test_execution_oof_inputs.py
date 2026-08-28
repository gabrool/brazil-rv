from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from brazil_rv.execution.inputs import (
    OOF_PREDICTION_ARCHIVE_SCHEMA,
    load_discovery_prediction_archive,
)
from brazil_rv.execution.splits import purged_training_folds
from brazil_rv.modeling.contract import EQUITY_COUNT
from brazil_rv.modeling.data import (
    FEATURE_STORE_CONTRACT,
    feature_store_axis_identity,
    feature_store_identity,
)
from brazil_rv.modeling.oof_predictions import _indexed_trade_dates

SEEDS = (11, 29, 47, 61, 79, 97, 113, 131, 149, 167)


def test_oof_date_axis_preserves_absolute_store_indices() -> None:
    dates = (date(2021, 8, 16), date(2021, 8, 17))
    ordered, by_index, by_date = _indexed_trade_dates(
        pl.DataFrame({"date_idx": [20, 21], "trade_date": dates})
    )

    assert ordered == dates
    assert by_index == {20: dates[0], 21: dates[1]}
    assert by_date == {dates[0]: 20, dates[1]: 21}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _refresh_source(wrapper_path: Path) -> None:
    wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
    source_path = Path(wrapper["source_manifest"])
    wrapper["source_manifest_sha256"] = _sha256(source_path)
    _write_json(wrapper_path, wrapper)


def _write_oof_archive(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    dates = tuple(date(2021, 1, 1) + timedelta(days=index) for index in range(716))
    store = tmp_path / "store"
    store.mkdir()
    (store / "manifest.json").write_text("{}\n", encoding="utf-8")
    _write_json(
        store / "feature_schema.json", {"contract_version": FEATURE_STORE_CONTRACT}
    )
    pl.DataFrame({"date_idx": range(716), "trade_date": dates}).write_parquet(
        store / "date_index.parquet"
    )
    pl.DataFrame(
        {
            "equity_slot": range(EQUITY_COUNT),
            "security_id": [f"security_{index}" for index in range(EQUITY_COUNT)],
        }
    ).write_parquet(store / "equity_index.parquet")
    pl.DataFrame(
        {
            "sample_id": range(716),
            "date_idx": range(716),
            "decision_idx": [0] * 716,
            "trade_date": dates,
            "equity_cutoff_index": [15] * 716,
        }
    ).write_parquet(store / "sample_index.parquet")
    activity = np.zeros((716, EQUITY_COUNT), dtype=bool)
    activity[:, :4] = True
    np.save(store / "equity_membership.npy", activity)
    np.save(store / "equity_data_ready.npy", activity)

    prediction_path = tmp_path / "oof_predictions.npz"
    reference_path = tmp_path / "oof_reference.npz"
    scores = np.zeros((716, EQUITY_COUNT, 1), dtype=np.float32)
    scores[:, :4, 0] = [3.0, 1.0, 4.0, 2.0]
    folds = purged_training_folds(dates)
    source_fold = np.empty(716, dtype=np.int8)
    by_date = {value: index for index, value in enumerate(dates)}
    for fold_index, fold in enumerate(folds.folds):
        source_fold[[by_date[value] for value in fold.heldout_dates]] = fold_index
    np.savez(prediction_path, ranks=scores)
    np.savez(
        reference_path,
        sample_id=np.arange(716, dtype=np.int64),
        date_idx=np.arange(716, dtype=np.int64),
        decision_idx=np.zeros(716, dtype=np.int64),
        source_fold_index=source_fold,
    )

    run_prediction = tmp_path / "run_prediction.npz"
    run_reference = tmp_path / "run_reference.npz"
    np.savez(run_prediction, ema_0995=np.zeros((1, EQUITY_COUNT, 3), np.float32))
    np.savez(run_reference, sample_id=np.asarray([0], dtype=np.int64))
    identity = feature_store_identity(store)
    bindings = {}
    for fold in folds.folds:
        for seed in SEEDS:
            run_manifest = tmp_path / f"{fold.name}_seed_{seed}.json"
            _write_json(
                run_manifest,
                {
                    "schema": "BRAZIL_RV_MONITOR_FREE_OOF_RUN_V1",
                    "status": "completed",
                    "seed": seed,
                    "feature_store_identity": identity,
                    "source_fold_sha256": fold.payload()["sha256"],
                    "epochs_completed": 20,
                    "monitor": None,
                    "head_mode": "three_head",
                    "training": {
                        "heldout_evaluations_during_training": 0,
                        "final_states": ["epoch20_raw", "epoch20_ema_0995"],
                    },
                    "fit_exclusion_proof": {
                        "fit_date_identity_sha256": fold.payload()[
                            "fit_date_identity_sha256"
                        ],
                        "heldout_date_identity_sha256": fold.payload()[
                            "heldout_date_identity_sha256"
                        ],
                    },
                    "prediction_sha256": _sha256(run_prediction),
                    "reference_sha256": _sha256(run_reference),
                    "official_validation_accessed": False,
                    "test_accessed": False,
                },
            )
            bindings[f"{fold.name}/seed_{seed}"] = {
                "manifest": str(run_manifest),
                "manifest_sha256": _sha256(run_manifest),
                "prediction": str(run_prediction),
                "prediction_sha256": _sha256(run_prediction),
                "reference": str(run_reference),
                "reference_sha256": _sha256(run_reference),
            }
    source_path = tmp_path / "source_manifest.json"
    _write_json(
        source_path,
        {
            "schema": "BRAZIL_RV_OOF_MANUFACTURE_V1",
            "status": "completed",
            "feature_store_identity": identity,
            "purged_folds": folds.payload(),
            "run_bindings": bindings,
            "prediction_sha256": _sha256(prediction_path),
            "reference_sha256": _sha256(reference_path),
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    wrapper_path = tmp_path / "execution_manifest.json"
    _write_json(
        wrapper_path,
        {
            "schema": OOF_PREDICTION_ARCHIVE_SCHEMA,
            "split": "oof_train",
            "prediction_sha256": _sha256(prediction_path),
            "reference_sha256": _sha256(reference_path),
            "prediction_key": "ranks",
            "feature_store_identity": identity,
            "axes": feature_store_axis_identity(store),
            "refresh_minutes": [15],
            "source_manifest": str(source_path),
            "source_manifest_sha256": _sha256(source_path),
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    return prediction_path, reference_path, wrapper_path, store


def test_oof_loader_accepts_exact_fit_exclusion_chain(tmp_path: Path) -> None:
    prediction, reference, wrapper, store = _write_oof_archive(tmp_path)

    archive = load_discovery_prediction_archive(prediction, reference, wrapper, store)

    assert archive.ranks.shape == (716, 1, EQUITY_COUNT, 1)
    assert archive.date_idx.size == 716


def test_oof_loader_rejects_tampered_fit_window(tmp_path: Path) -> None:
    prediction, reference, wrapper, store = _write_oof_archive(tmp_path)
    wrapper_value = json.loads(wrapper.read_text(encoding="utf-8"))
    source_path = Path(wrapper_value["source_manifest"])
    source = json.loads(source_path.read_text(encoding="utf-8"))
    binding = source["run_bindings"]["fold_0/seed_11"]
    manifest_path = Path(binding["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fit_exclusion_proof"]["fit_date_identity_sha256"] = "0" * 64
    _write_json(manifest_path, manifest)
    binding["manifest_sha256"] = _sha256(manifest_path)
    _write_json(source_path, source)
    _refresh_source(wrapper)

    with pytest.raises(ValueError, match="source run differs"):
        load_discovery_prediction_archive(prediction, reference, wrapper, store)


def test_oof_loader_rejects_leaked_date(tmp_path: Path) -> None:
    prediction, reference, wrapper, store = _write_oof_archive(tmp_path)
    with np.load(reference, allow_pickle=False) as values:
        arrays = {name: values[name].copy() for name in values.files}
    arrays["source_fold_index"][0] = 1
    np.savez(reference, **arrays)
    wrapper_value = json.loads(wrapper.read_text(encoding="utf-8"))
    source_path = Path(wrapper_value["source_manifest"])
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["reference_sha256"] = _sha256(reference)
    _write_json(source_path, source)
    wrapper_value["reference_sha256"] = _sha256(reference)
    wrapper_value["source_manifest_sha256"] = _sha256(source_path)
    _write_json(wrapper, wrapper_value)

    with pytest.raises(ValueError, match="non-held-out date"):
        load_discovery_prediction_archive(prediction, reference, wrapper, store)


def test_oof_loader_rejects_missing_fold_binding(tmp_path: Path) -> None:
    prediction, reference, wrapper, store = _write_oof_archive(tmp_path)
    wrapper_value = json.loads(wrapper.read_text(encoding="utf-8"))
    source_path = Path(wrapper_value["source_manifest"])
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["run_bindings"].pop("fold_4/seed_167")
    _write_json(source_path, source)
    _refresh_source(wrapper)

    with pytest.raises(ValueError, match="exact run binding"):
        load_discovery_prediction_archive(prediction, reference, wrapper, store)
