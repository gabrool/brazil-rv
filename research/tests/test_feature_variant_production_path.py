from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch

import brazil_rv.modeling.analyze_stock_time_attribution as attribution
import brazil_rv.modeling.data as modeling_data
import brazil_rv.modeling.feature_variant as feature_variant
import brazil_rv.preprocessing.intraday_normalization as normalization
import brazil_rv.preprocessing.intraday_normalization_variants as variants
from brazil_rv.modeling.contract import (
    BASELINE_TCN_SETTINGS,
    EQUITY_COUNT,
    HORIZON_COUNT,
    RuntimeSettings,
    TEST_START,
    TCNArchitecture,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
    architecture_for_model,
)
from brazil_rv.modeling.data import (
    BatchRequest,
    VectorizedFeatureDataset,
    create_analysis_loader,
    create_evaluation_loader,
    create_training_loaders,
    feature_store_identity,
)
from brazil_rv.modeling.evaluate import (
    load_current_neural_run,
    resolve_checkpoint_feature_store,
)
from brazil_rv.modeling.model import build_neural_model
from brazil_rv.modeling.run_intraday_normalization_stage import (
    _expected_run_provenance,
    _stage_store_identity,
)
from brazil_rv.preprocessing.contract import output_array_specs
from brazil_rv.preprocessing.intraday_normalization import (
    AFFECTED_DYNAMIC_CHANNELS,
    AFFECTED_PEER_CHANNELS,
    DECISION_FEATURE_MINUTES,
    DEVELOPMENT_IDENTITY_SCHEMA,
    INVARIANT_DYNAMIC_CHANNELS,
    VARIANT_SCHEMA,
    VISIBLE_EQUITY_MINUTES,
    parent_artifact_hashes,
    parent_identity,
    sha256_file,
)

CONTRACT_VERSION = "synthetic-feature-store-v1"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _write_parent_array(path: Path, dtype: np.dtype, shape: tuple[int, ...]) -> None:
    values = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    values[...] = 0
    if path.name in {
        "equity_membership.npy",
        "equity_data_ready.npy",
        "context_data_ready.npy",
        "global_data_ready.npy",
        "label_mask.npy",
        "horizon_mask.npy",
    }:
        values[...] = True
    if path.name in {
        "equity_features.npy",
        "context_features.npy",
        "global_features.npy",
    }:
        values[..., 5] = 1.0
    if path.name in {"targets.npy", "raw_returns.npy"}:
        scores = np.linspace(-0.02, 0.02, EQUITY_COUNT, dtype=np.float32)
        values[...] = scores[None, :, None, None]
    values.flush()


def _sample_rows() -> pl.DataFrame:
    dates = (TRAIN_END, VALIDATION_END, TEST_START)
    rows: list[dict[str, object]] = []
    for date_idx, trade_date in enumerate(dates):
        for decision_idx in range(55):
            rows.append(
                {
                    "sample_id": date_idx * 55 + decision_idx,
                    "trade_date": trade_date,
                    "date_idx": date_idx,
                    "decision_idx": decision_idx,
                    "equity_cutoff_index": 15 + 5 * decision_idx,
                    "context_cutoff_index": 75 + 5 * decision_idx,
                }
            )
    return pl.DataFrame(rows)


def _loader_rows(date_idx: int, trade_date: date, sample_id: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "sample_id": [sample_id],
            "trade_date": [trade_date],
            "date_idx": [date_idx],
            "decision_idx": [0],
            "equity_cutoff_index": [15],
            "context_cutoff_index": [75],
        }
    )


@pytest.fixture
def candidate_store(tmp_path: Path) -> SimpleNamespace:
    parent = tmp_path / "parent"
    candidate = tmp_path / "candidate"
    profile = tmp_path / "profile"
    parent.mkdir()
    candidate.mkdir()
    profile.mkdir()

    development_date_count = 2
    parent_date_count = 3
    parent_manifest = {
        "contract_version": CONTRACT_VERSION,
        "build_git_commit": "synthetic-parent",
        "canonical_inputs": {},
        "constants": {},
        "outputs": {name: {} for name in output_array_specs(parent_date_count)},
    }
    _write_json(parent / "manifest.json", parent_manifest)
    _write_json(parent / "feature_schema.json", {"contract_version": CONTRACT_VERSION})
    pl.DataFrame(
        {
            "date_idx": range(parent_date_count),
            "trade_date": (TRAIN_END, VALIDATION_END, TEST_START),
        }
    ).write_parquet(parent / "date_index.parquet")
    _sample_rows().write_parquet(parent / "sample_index.parquet")
    pl.DataFrame(
        {
            "equity_slot": range(EQUITY_COUNT),
            "security_id": [f"security-{slot}" for slot in range(EQUITY_COUNT)],
        }
    ).write_parquet(parent / "equity_index.parquet")
    pl.DataFrame({"context_slot": range(7)}).write_parquet(
        parent / "context_index.parquet"
    )
    pl.DataFrame({"global_slot": range(8)}).write_parquet(
        parent / "global_context_index.parquet"
    )
    for name, spec in output_array_specs(parent_date_count).items():
        _write_parent_array(parent / name, spec.dtype, spec.shape)

    context = SimpleNamespace(
        parent=parent.resolve(),
        manifest=parent_manifest,
        allowed_date_count=development_date_count,
    )
    artifacts = parent_artifact_hashes(context)
    development_identity = parent_identity(context)
    assert artifacts["schema"] == DEVELOPMENT_IDENTITY_SCHEMA

    dynamic_path = candidate / variants.DYNAMIC_OVERLAY_FILE
    peer_path = candidate / variants.PEER_OVERLAY_FILE
    dynamic = np.lib.format.open_memmap(
        dynamic_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            development_date_count,
            EQUITY_COUNT,
            VISIBLE_EQUITY_MINUTES,
            len(AFFECTED_DYNAMIC_CHANNELS),
        ),
    )
    dynamic[...] = 0.0
    dynamic[..., 0] = 7.0
    dynamic.flush()
    peer = np.lib.format.open_memmap(
        peer_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            development_date_count,
            EQUITY_COUNT,
            len(DECISION_FEATURE_MINUTES),
            len(AFFECTED_PEER_CHANNELS),
        ),
    )
    peer[...] = 0.0
    peer.flush()

    profile_manifest = {
        "artifacts": {"synthetic_profile": "0" * 64},
        "configuration": {"bin_minutes": 30},
        "training_profile_freeze_date": str(TRAIN_END),
    }
    profile_manifest_path = profile / "equity_tod_profile.json"
    _write_json(profile_manifest_path, profile_manifest)
    manifest = {
        "schema": VARIANT_SCHEMA,
        "repository_commit": "synthetic-commit",
        "arm": "equity_tod_half",
        "gamma": 0.5,
        "contract_version": CONTRACT_VERSION,
        "profile_estimator_configuration": profile_manifest["configuration"],
        "split_boundaries": {
            "training": [str(TRAIN_START), str(TRAIN_END)],
            "validation": [str(VALIDATION_START), str(VALIDATION_END)],
        },
        "canonical_parent_feature_store": development_identity,
        "parent_artifact_sha256": artifacts,
        "profile": {
            "path": str(profile.resolve()),
            "manifest_sha256": sha256_file(profile_manifest_path),
            "artifact_sha256": profile_manifest["artifacts"],
        },
        "allowed_date_count": development_date_count,
        "allowed_date_end": str(VALIDATION_END),
        "test_accessed": False,
        "test_rows_present": False,
        "dynamic_overlay": {
            "file": variants.DYNAMIC_OVERLAY_FILE,
            "shape": list(dynamic.shape),
            "dtype": "float32",
            "channels": list(AFFECTED_DYNAMIC_CHANNELS),
            "sha256": sha256_file(dynamic_path),
        },
        "peer_overlay": {
            "file": variants.PEER_OVERLAY_FILE,
            "shape": list(peer.shape),
            "dtype": "float32",
            "minutes": list(DECISION_FEATURE_MINUTES),
            "channels": list(AFFECTED_PEER_CHANNELS),
            "sha256": sha256_file(peer_path),
        },
        "affected_arrays": {
            "equity_features.npy": list(AFFECTED_DYNAMIC_CHANNELS),
            "equity_peer_features.npy": list(AFFECTED_PEER_CHANNELS),
        },
        "parent_bound_arrays": sorted(
            name
            for name in parent_manifest["outputs"]
            if name not in {"equity_features.npy", "equity_peer_features.npy"}
        ),
        "parent_bound_dynamic_channels": list(INVARIANT_DYNAMIC_CHANNELS),
        "parent_bound_peer_channels": [2, 3],
        "profile_freeze_date": str(TRAIN_END),
        "validation_update_rule": "frozen_training_end_profile",
    }
    _write_json(candidate / variants.VARIANT_MANIFEST, manifest)
    return SimpleNamespace(
        parent=parent,
        candidate=candidate,
        context=context,
        manifest=manifest,
        profile_manifest=profile_manifest,
    )


def _legacy_identity(store: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    for path in (
        store / "manifest.json",
        store / "feature_schema.json",
        store / "sample_index.parquet",
    ):
        digest.update(path.read_bytes())
    return {
        "path": str(store.resolve()),
        "contract_version": CONTRACT_VERSION,
        "metadata_sha256": digest.hexdigest(),
    }


_ABSENT_IDENTITY = object()


def _write_run_checkpoint(
    run_dir: Path,
    store: Path,
    identity: object = _ABSENT_IDENTITY,
) -> tuple[dict[str, object], TCNArchitecture]:
    architecture = architecture_for_model("tcn", BASELINE_TCN_SETTINGS)
    assert isinstance(architecture, TCNArchitecture)
    model = build_neural_model("tcn", architecture)
    checkpoint: dict[str, object] = {
        "model_name": "tcn",
        "architecture": asdict(architecture),
        "tcn_settings": asdict(BASELINE_TCN_SETTINGS),
        "peer_features": {"mode": "none"},
        "optimizer_variant": "adamw",
        "objective": {"name": "soft_spearman", "temperature": 0.5},
        "sam": {"enabled": False},
        "seed": 11,
        "epoch": 1,
        "validation_score": 0.1,
        "feature_store": str(store),
        "global_context": "enabled",
        "model_state_dict": model.state_dict(),
    }
    if identity is not _ABSENT_IDENTITY:
        checkpoint["feature_store_identity"] = identity
    run_dir.mkdir()
    torch.save(checkpoint, run_dir / "best_checkpoint.pt")
    return checkpoint, architecture


def _write_validation_cache(cache_dir: Path, run_dir: Path) -> None:
    values = {
        name: np.zeros((1, EQUITY_COUNT, HORIZON_COUNT), dtype=np.float32)
        for name in ("predictions", "targets", "raw_returns")
    }
    values["label_mask"] = np.ones((1, EQUITY_COUNT, HORIZON_COUNT), dtype=bool)
    values["date_idx"] = np.array([1], dtype=np.int64)
    values["decision_idx"] = np.array([0], dtype=np.int64)
    attribution._write_cache(attribution._cache_path(cache_dir, run_dir), values)


def test_candidate_identity_dataset_and_all_production_loaders(
    candidate_store: SimpleNamespace,
) -> None:
    store = candidate_store.candidate
    train_rows = _loader_rows(0, TRAIN_END, 0)
    validation_rows = _loader_rows(1, VALIDATION_END, 55)
    expected_identity = _stage_store_identity(store, candidate_store.context)

    identity = feature_store_identity(store)
    assert identity == expected_identity
    provenance = _expected_run_provenance(
        11, store, expected_identity, train_rows, validation_rows
    )
    assert provenance["feature_store_identity"] == identity
    dataset = VectorizedFeatureDataset(store, train_rows, "temporal_only", None)
    direct_batch = dataset[BatchRequest((0,), 1)]
    assert direct_batch["patches"].shape[0] == 1
    assert 7.0 in direct_batch["patches"]

    runtime = RuntimeSettings(
        effective_batch_size=1,
        loader_batch_size=1,
        microbatch_size=1,
        evaluation_batch_size=1,
        num_workers=0,
    )
    train, validation, _ = create_training_loaders(
        store,
        train_rows,
        validation_rows,
        "temporal_only",
        None,
        runtime,
        11,
        allow_date_replacement=True,
    )
    evaluation = create_evaluation_loader(
        store, validation_rows, "temporal_only", None, runtime, 11
    )
    analysis = create_analysis_loader(
        store, validation_rows, "temporal_only", None, runtime, 11
    )
    for loader in (train, validation, evaluation, analysis):
        batch = next(iter(loader))
        assert batch["sample_valid_mask"].all()


def test_candidate_identity_never_calls_legacy_full_store_identity(
    candidate_store: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(_store: Path) -> dict[str, object]:
        pytest.fail("candidate identity called the legacy full-store identity")

    monkeypatch.setattr(modeling_data, "_legacy_feature_store_identity", forbidden)
    assert (
        feature_store_identity(candidate_store.candidate)["hash_scope"]["kind"]
        == "development_only"
    )


def test_candidate_identity_never_opens_parent_date_arrays(
    candidate_store: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_load = feature_variant.np.load

    def development_only_load(path: Path, *args: object, **kwargs: object) -> object:
        if Path(path).parent.resolve() == candidate_store.parent.resolve():
            pytest.fail(f"candidate identity opened parent array: {Path(path).name}")
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(feature_variant.np, "load", development_only_load)
    identity = feature_store_identity(candidate_store.candidate)
    assert identity["hash_scope"]["end_date"] == str(VALIDATION_END)


def test_held_out_parent_tail_is_identity_and_loader_isolated(
    candidate_store: SimpleNamespace,
) -> None:
    store = candidate_store.candidate
    baseline = feature_store_identity(store)
    path = candidate_store.parent / "equity_features.npy"
    values = np.load(path, mmap_mode="r+", allow_pickle=False)
    values[2, 0, 0, 0] = 900.0
    values.flush()

    assert feature_store_identity(store) == baseline
    rows = _loader_rows(1, VALIDATION_END, 55)
    dataset = VectorizedFeatureDataset(store, rows, "temporal_only", None)
    assert dataset[BatchRequest((0,), 1)]["sample_valid_mask"].all()


@pytest.mark.parametrize(
    "corruption",
    (
        "parent_path",
        "development_hash",
        "hash_scope",
        "boundary",
        "contract_version",
        "arm_gamma",
        "overlay_shape",
        "overlay_dtype",
    ),
)
def test_candidate_binding_rejects_corruption(
    candidate_store: SimpleNamespace, corruption: str
) -> None:
    manifest = candidate_store.manifest
    if corruption == "parent_path":
        manifest["canonical_parent_feature_store"]["path"] = str(
            candidate_store.candidate
        )
    elif corruption == "development_hash":
        manifest["canonical_parent_feature_store"]["metadata_sha256"] = "0" * 64
    elif corruption == "hash_scope":
        manifest["canonical_parent_feature_store"]["hash_scope"]["date_array_scope"] = (
            "complete"
        )
    elif corruption == "boundary":
        manifest["allowed_date_end"] = str(TRAIN_END)
    elif corruption == "contract_version":
        manifest["contract_version"] = "wrong"
    elif corruption == "arm_gamma":
        manifest["gamma"] = 1.0
    elif corruption == "overlay_shape":
        manifest["dynamic_overlay"]["shape"][0] += 1
    elif corruption == "overlay_dtype":
        manifest["dynamic_overlay"]["dtype"] = "float64"
    _write_json(candidate_store.candidate / variants.VARIANT_MANIFEST, manifest)

    with pytest.raises((ValueError, FileNotFoundError)):
        feature_store_identity(candidate_store.candidate)


def test_strict_stage_validator_detects_development_prefix_mutation(
    candidate_store: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(variants, "repository_commit", lambda: "synthetic-commit")
    monkeypatch.setattr(
        variants, "load_source_context", lambda _parent: candidate_store.context
    )
    monkeypatch.setattr(
        variants,
        "validate_equity_tod_profile",
        lambda _path, expected_context: (
            candidate_store.profile_manifest,
            np.ones((2, 1)),
        ),
    )
    variants.validate_intraday_normalization_variant(
        candidate_store.candidate, "equity_tod_half"
    )

    path = candidate_store.parent / "equity_features.npy"
    values = np.load(path, mmap_mode="r+", allow_pickle=False)
    values[0, 0, 0, 0] = 1.0
    values.flush()
    with pytest.raises(ValueError, match="parent identity|parent hashes"):
        variants.validate_intraday_normalization_variant(
            candidate_store.candidate, "equity_tod_half"
        )


def test_legacy_feature_store_identity_is_byte_for_byte_compatible(
    candidate_store: SimpleNamespace,
) -> None:
    assert feature_store_identity(candidate_store.parent) == _legacy_identity(
        candidate_store.parent
    )


def test_completed_run_evaluation_resolves_the_recorded_candidate_identity(
    candidate_store: SimpleNamespace,
) -> None:
    store = candidate_store.candidate
    identity = feature_store_identity(store)
    architecture = architecture_for_model("tcn", BASELINE_TCN_SETTINGS)
    assert isinstance(architecture, TCNArchitecture)
    model = build_neural_model("tcn", architecture)
    checkpoint = {
        "model_name": "tcn",
        "architecture": asdict(architecture),
        "tcn_settings": asdict(BASELINE_TCN_SETTINGS),
        "peer_features": {"mode": "none"},
        "optimizer_variant": "adamw",
        "objective": {"name": "soft_spearman", "temperature": 0.5},
        "sam": {"enabled": False},
        "seed": 11,
        "epoch": 1,
        "validation_score": 0.1,
        "feature_store": str(store),
        "feature_store_identity": identity,
        "global_context": "enabled",
        "model_state_dict": model.state_dict(),
    }
    run_dir = store.parent / "completed_run"
    run_dir.mkdir()
    checkpoint_path = run_dir / "best_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    _, loaded, resolved_store = load_current_neural_run(run_dir)
    assert loaded["feature_store_identity"] == identity
    assert resolved_store.resolve() == store.resolve()
    rows = _loader_rows(1, VALIDATION_END, 55)
    runtime = RuntimeSettings(
        effective_batch_size=1,
        loader_batch_size=1,
        microbatch_size=1,
        evaluation_batch_size=1,
        num_workers=0,
    )
    loader = create_evaluation_loader(
        resolved_store,
        rows,
        "tcn",
        "enabled",
        runtime,
        11,
        architecture,
    )
    assert next(iter(loader))["sample_valid_mask"].all()

    checkpoint["feature_store_identity"] = {
        **identity,
        "metadata_sha256": "0" * 64,
    }
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="differs from the resolved store"):
        load_current_neural_run(run_dir)


def test_completed_run_evaluation_accepts_normalization_legacy_control(
    candidate_store: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = candidate_store.parent
    identity = _stage_store_identity(store, candidate_store.context)
    architecture = architecture_for_model("tcn", BASELINE_TCN_SETTINGS)
    assert isinstance(architecture, TCNArchitecture)
    model = build_neural_model("tcn", architecture)
    checkpoint = {
        "model_name": "tcn",
        "architecture": asdict(architecture),
        "tcn_settings": asdict(BASELINE_TCN_SETTINGS),
        "peer_features": {"mode": "none"},
        "optimizer_variant": "adamw",
        "objective": {"name": "soft_spearman", "temperature": 0.5},
        "sam": {"enabled": False},
        "seed": 11,
        "epoch": 1,
        "validation_score": 0.1,
        "feature_store": str(store),
        "feature_store_identity": identity,
        "global_context": "enabled",
        "model_state_dict": model.state_dict(),
    }
    run_dir = store.parent / "legacy_control_run"
    run_dir.mkdir()
    torch.save(checkpoint, run_dir / "best_checkpoint.pt")

    _, loaded, resolved_store = load_current_neural_run(run_dir)
    assert loaded["feature_store_identity"] == identity
    assert loaded["feature_store_identity"] == _stage_store_identity(
        store, candidate_store.context
    )
    assert resolved_store.resolve() == store.resolve()
    rows = _loader_rows(1, VALIDATION_END, 55)
    runtime = RuntimeSettings(
        effective_batch_size=1,
        loader_batch_size=1,
        microbatch_size=1,
        evaluation_batch_size=1,
        num_workers=0,
    )
    loader = create_evaluation_loader(
        resolved_store,
        rows,
        "tcn",
        "enabled",
        runtime,
        11,
        architecture,
    )
    assert next(iter(loader))["sample_valid_mask"].all()

    monkeypatch.setattr(
        attribution,
        "learn_overnight_thresholds",
        lambda _values: (0.0, 0.0),
    )
    cache_dir = store.parent / "legacy_control_cache"
    _write_validation_cache(cache_dir, run_dir)
    inputs = attribution.load_attribution_inputs(run_dir.resolve(), cache_dir)
    assert inputs.run_name == run_dir.name
    assert inputs.predictions.shape == (1, EQUITY_COUNT, HORIZON_COUNT)


def test_completed_run_accepts_ordinary_full_legacy_identity(
    candidate_store: SimpleNamespace,
) -> None:
    identity = feature_store_identity(candidate_store.parent)
    assert identity == _legacy_identity(candidate_store.parent)
    run_dir = candidate_store.parent.parent / "full_legacy_run"
    _write_run_checkpoint(run_dir, candidate_store.parent, identity)

    _, loaded, store = load_current_neural_run(run_dir)

    assert loaded["feature_store_identity"] == identity
    assert store == candidate_store.parent.resolve()


def test_historical_checkpoint_without_identity_remains_loadable(
    candidate_store: SimpleNamespace,
) -> None:
    run_dir = candidate_store.parent.parent / "historical_run"
    _write_run_checkpoint(run_dir, candidate_store.parent)

    _, loaded, store = load_current_neural_run(run_dir)

    assert "feature_store_identity" not in loaded
    assert store == candidate_store.parent.resolve()


def test_development_identity_ignores_held_out_feature_and_target_tail(
    candidate_store: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _stage_store_identity(candidate_store.parent, candidate_store.context)
    run_dir = candidate_store.parent.parent / "heldout_tail_run"
    _write_run_checkpoint(run_dir, candidate_store.parent, identity)
    for filename in ("equity_features.npy", "targets.npy"):
        values = np.load(
            candidate_store.parent / filename,
            mmap_mode="r+",
            allow_pickle=False,
        )
        values[2].flat[0] = 123.0
        values.flush()

    original_load = normalization.np.load

    class PrefixOnlyArray:
        def __init__(self, values: np.ndarray) -> None:
            self.values = values
            self.dtype = values.dtype
            self.shape = values.shape

        def __getitem__(self, key: object) -> object:
            date_key = key[0] if isinstance(key, tuple) else key
            if not isinstance(date_key, slice) or (
                date_key.stop is None or date_key.stop > 2
            ):
                pytest.fail("development identity accessed the held-out array tail")
            return self.values[key]

    def guarded_load(path: Path, *args: object, **kwargs: object) -> object:
        values = original_load(path, *args, **kwargs)
        if Path(path).parent.resolve() == candidate_store.parent.resolve() and Path(
            path
        ).name in output_array_specs(3):
            return PrefixOnlyArray(values)
        return values

    monkeypatch.setattr(normalization.np, "load", guarded_load)
    _, loaded, store = load_current_neural_run(run_dir)
    monkeypatch.setattr(normalization.np, "load", original_load)

    assert loaded["feature_store_identity"] == identity
    assert store == candidate_store.parent.resolve()
    rows = _loader_rows(1, VALIDATION_END, 55)
    runtime = RuntimeSettings(
        effective_batch_size=1,
        loader_batch_size=1,
        microbatch_size=1,
        evaluation_batch_size=1,
        num_workers=0,
    )
    loader = create_evaluation_loader(
        store,
        rows,
        "tcn",
        "enabled",
        runtime,
        11,
        architecture_for_model("tcn", BASELINE_TCN_SETTINGS),
    )
    assert next(iter(loader))["sample_valid_mask"].all()


@pytest.mark.parametrize("filename", ("equity_features.npy", "targets.npy"))
def test_development_identity_rejects_prefix_mutation(
    candidate_store: SimpleNamespace,
    filename: str,
) -> None:
    identity = _stage_store_identity(candidate_store.parent, candidate_store.context)
    run_dir = candidate_store.parent.parent / f"{Path(filename).stem}_prefix_run"
    _write_run_checkpoint(run_dir, candidate_store.parent, identity)
    values = np.load(
        candidate_store.parent / filename,
        mmap_mode="r+",
        allow_pickle=False,
    )
    values[0].flat[0] = 123.0
    values.flush()

    with pytest.raises(ValueError, match="differs from the resolved store"):
        load_current_neural_run(run_dir)


@pytest.mark.parametrize(
    "corruption",
    (
        "digest",
        "path",
        "contract",
        "date_count",
        "end_date",
        "array_scope",
        "missing_field",
        "extra_field",
        "malformed",
        "candidate_on_parent",
        "full_on_candidate",
    ),
)
def test_checkpoint_identity_resolver_rejects_corruption_and_cross_type_binding(
    candidate_store: SimpleNamespace,
    corruption: str,
) -> None:
    store = candidate_store.parent
    identity: object = json.loads(
        json.dumps(_stage_store_identity(store, candidate_store.context))
    )
    assert isinstance(identity, dict)
    if corruption == "digest":
        identity["metadata_sha256"] = "0" * 64
    elif corruption == "path":
        identity["path"] = str(candidate_store.candidate.resolve())
    elif corruption == "contract":
        identity["contract_version"] = "wrong"
    elif corruption == "date_count":
        identity["hash_scope"]["date_count"] += 1
    elif corruption == "end_date":
        identity["hash_scope"]["end_date"] = str(TRAIN_END)
    elif corruption == "array_scope":
        identity["hash_scope"]["date_array_scope"] = "complete"
    elif corruption == "missing_field":
        identity.pop("metadata_sha256")
    elif corruption == "extra_field":
        identity["unexpected"] = True
    elif corruption == "malformed":
        identity = "not-an-identity"
    elif corruption == "candidate_on_parent":
        identity = feature_store_identity(candidate_store.candidate)
    elif corruption == "full_on_candidate":
        store = candidate_store.candidate
        identity = {
            **_legacy_identity(candidate_store.parent),
            "path": str(store.resolve()),
        }

    with pytest.raises(ValueError):
        resolve_checkpoint_feature_store(
            {
                "feature_store": str(store),
                "feature_store_identity": identity,
            }
        )


def test_invocation_cache_hashes_one_legacy_control_once(
    candidate_store: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = normalization._development_array_identity

    def counted(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    identity = _stage_store_identity(candidate_store.parent, candidate_store.context)
    monkeypatch.setattr(normalization, "_development_array_identity", counted)
    first_run = candidate_store.parent.parent / "cached_seed_11"
    second_run = candidate_store.parent.parent / "cached_seed_22"
    _write_run_checkpoint(first_run, candidate_store.parent, identity)
    _write_run_checkpoint(second_run, candidate_store.parent, identity)
    identity_cache: modeling_data.FeatureStoreIdentityCache = {}

    load_current_neural_run(first_run, identity_cache=identity_cache)
    first_count = calls
    load_current_neural_run(second_run, identity_cache=identity_cache)

    assert first_count == len(output_array_specs(2))
    assert calls == first_count
    assert len(identity_cache) == 1


def test_three_arm_cached_attribution_uses_the_shared_identity_resolver(
    candidate_store: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        attribution,
        "learn_overnight_thresholds",
        lambda _values: (0.0, 0.0),
    )
    full = candidate_store.parent.parent / "candidate_full"
    shutil.copytree(candidate_store.candidate, full)
    full_manifest = json.loads(
        (full / variants.VARIANT_MANIFEST).read_text(encoding="utf-8")
    )
    full_manifest["arm"] = "equity_tod_full"
    full_manifest["gamma"] = 1.0
    _write_json(full / variants.VARIANT_MANIFEST, full_manifest)

    arms = (
        (
            "legacy_control",
            candidate_store.parent,
            _stage_store_identity(candidate_store.parent, candidate_store.context),
        ),
        (
            "equity_tod_half",
            candidate_store.candidate,
            feature_store_identity(candidate_store.candidate),
        ),
        ("equity_tod_full", full, feature_store_identity(full)),
    )
    cache_dir = candidate_store.parent.parent / "attribution_cache"
    identity_cache: modeling_data.FeatureStoreIdentityCache = {}
    loaded = []
    for name, store, identity in arms:
        run_dir = candidate_store.parent.parent / f"attribution_{name}"
        _write_run_checkpoint(run_dir, store, identity)
        _write_validation_cache(cache_dir, run_dir)
        load_current_neural_run(run_dir, identity_cache=identity_cache)
        loaded.append(
            attribution.load_attribution_inputs(
                run_dir.resolve(),
                cache_dir,
                identity_cache=identity_cache,
            )
        )

    assert [value.run_name for value in loaded] == [
        f"attribution_{name}" for name, _, _ in arms
    ]
    assert all(len(value.security_ids) == EQUITY_COUNT for value in loaded)
    assert all(
        value.predictions.shape == (1, EQUITY_COUNT, HORIZON_COUNT) for value in loaded
    )
    assert len(identity_cache) == 3
