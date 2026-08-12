from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from brazil_rv.modeling import evaluate
from brazil_rv.modeling.context_ablation import get_context_ablation
from brazil_rv.modeling.contract import (
    BASELINE_TCN_SETTINGS,
    HardwareInfo,
    TCNSettings,
    architecture_for_model,
    model_consumes_context,
    peer_feature_metadata,
)
from brazil_rv.modeling.data import CacheWarmupReport
from brazil_rv.modeling.engine import objective_metadata, sam_metadata
from brazil_rv.modeling.train import _model_metadata


def _run_manifest(
    run_dir: Path, feature_store: Path, model_name: str, peer_mode: str
) -> dict[str, object]:
    if model_name == "xgboost":
        model_metadata: dict[str, object] = {
            "model_name": "xgboost",
            "model_family": "xgboost",
            "tcn_settings": None,
            "architecture_constants": None,
            "parameter_count": None,
            "peer_features": peer_feature_metadata("xgboost", None, "none"),
        }
        optimizer_variant = None
        objective = None
        sam = None
        global_context = "enabled"
    else:
        settings: TCNSettings | None = (
            BASELINE_TCN_SETTINGS if model_name == "tcn" else None
        )
        architecture = architecture_for_model(model_name, settings)
        model_metadata = _model_metadata(model_name, architecture, settings, peer_mode)
        optimizer_variant = "adamw"
        objective = objective_metadata("soft_spearman", 0.1)
        sam = sam_metadata("adamw", None)
        global_context = (
            "enabled" if model_consumes_context(model_name, settings) else None
        )
    manifest = {
        **model_metadata,
        "status": "completed",
        "optimizer_variant": optimizer_variant,
        "objective": objective,
        "sam": sam,
        "seed": 11,
        "resolved_feature_store_path": str(feature_store),
        "git_commit_sha": "test-sha",
        "feature_manifest_contract_version": "test-feature-contract",
        "global_context": global_context,
        "global_context_source_hashes": {"source": "source-sha256"},
        "global_context_normalized_store_hashes": {"store": "store-sha256"},
        "context_ablation": get_context_ablation("none").metadata(),
    }
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _patch_evaluation_runtime(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    *,
    neural_evaluator: object | None = None,
) -> None:
    monkeypatch.setattr(
        evaluate,
        "parse_args",
        lambda: SimpleNamespace(run_dir=run_dir, split="validation"),
    )
    monkeypatch.setattr(
        evaluate,
        "validate_runtime",
        lambda: HardwareInfo(
            device_name="cpu-test",
            compute_capability=(0, 0),
            total_vram_bytes=0,
            cpu_architecture="test",
            platform="test",
            pytorch_version="test",
            cuda_version=None,
            cudnn_version=None,
        ),
    )
    monkeypatch.setattr(evaluate, "validate_feature_store", lambda _: pl.DataFrame())
    monkeypatch.setattr(
        evaluate, "select_sample_split", lambda sample_index, split: sample_index
    )
    monkeypatch.setattr(
        evaluate,
        "warm_feature_store_cache",
        lambda store, mode: CacheWarmupReport(0, 0, 0.0),
    )

    def evaluate_neural(
        manifest: dict[str, object], feature_store: Path, rows: pl.DataFrame
    ) -> tuple[object, ...]:
        del feature_store, rows
        return (
            {},
            [],
            {},
            evaluate._normalize_context_ablation_identity(manifest, run_dir=run_dir),
            evaluate._normalize_feature_ablation_identity(manifest, run_dir=run_dir),
        )

    monkeypatch.setattr(
        evaluate,
        "_evaluate_neural",
        evaluate_neural if neural_evaluator is None else neural_evaluator,
    )
    monkeypatch.setattr(
        evaluate,
        "_validate_xgboost_identity",
        lambda manifest, store, path: (
            {},
            evaluate._normalize_context_ablation_identity(manifest, run_dir=run_dir),
        ),
    )
    monkeypatch.setattr(
        evaluate, "resolve_context_ablation_for_store", lambda store, key: object()
    )
    monkeypatch.setattr(evaluate, "validate_xgboost_runtime", lambda: {})
    monkeypatch.setattr(
        evaluate,
        "evaluate_saved_xgboost",
        lambda *args: (None, {}, [], pl.DataFrame()),
    )
    monkeypatch.setattr(evaluate, "_atomic_write_parquet", lambda path, frame: None)
    monkeypatch.setattr(evaluate, "_daily_frame", lambda rows, store: pl.DataFrame())


@pytest.mark.parametrize(
    ("model_name", "peer_mode"),
    (
        ("tcn", "none"),
        ("tcn", "masked_control"),
        ("tcn", "selected"),
        ("tcn", "selected_plus_issuer"),
        ("temporal_only", "none"),
        ("mlp", "none"),
        ("xgboost", "none"),
    ),
)
def test_evaluation_manifest_records_complete_validated_peer_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    peer_mode: str,
) -> None:
    run_dir = tmp_path / f"{model_name}-{peer_mode}"
    feature_store = tmp_path / "feature-store"
    feature_store.mkdir(exist_ok=True)
    manifest = _run_manifest(run_dir, feature_store, model_name, peer_mode)
    expected = copy.deepcopy(manifest["peer_features"])
    _patch_evaluation_runtime(monkeypatch, run_dir)

    evaluate.main()

    output = next((run_dir / "evaluations").glob("*/evaluation_manifest.json"))
    published = json.loads(output.read_text(encoding="utf-8"))
    assert published["peer_features"] == expected
    if model_name != "tcn":
        assert published["peer_features"] == peer_feature_metadata(
            model_name, None, "none"
        )


@pytest.mark.parametrize("corruption", ("missing", "malformed", "forbidden"))
def test_invalid_run_peer_identity_cannot_publish_evaluation_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    model_name = "temporal_only" if corruption == "forbidden" else "tcn"
    peer_mode = "none" if corruption == "forbidden" else "selected"
    run_dir = tmp_path / corruption
    feature_store = tmp_path / "feature-store"
    feature_store.mkdir(exist_ok=True)
    manifest = _run_manifest(run_dir, feature_store, model_name, peer_mode)
    if corruption == "missing":
        del manifest["peer_features"]
    elif corruption == "malformed":
        manifest["peer_features"]["adapter"]["zero_initialized"] = False
    else:
        manifest["peer_features"] = peer_feature_metadata(
            "tcn",
            architecture_for_model("tcn", BASELINE_TCN_SETTINGS),
            "selected",
        )
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _patch_evaluation_runtime(monkeypatch, run_dir)

    with pytest.raises(ValueError, match="[Pp]eer-feature identity metadata"):
        evaluate.main()

    assert not list(run_dir.glob("evaluations/*/evaluation_manifest.json"))


def test_checkpoint_peer_mismatch_cannot_publish_evaluation_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "checkpoint-mismatch"
    feature_store = tmp_path / "feature-store"
    feature_store.mkdir()
    manifest = _run_manifest(run_dir, feature_store, "tcn", "selected")
    checkpoint = copy.deepcopy(manifest)
    checkpoint["peer_features"] = peer_feature_metadata(
        "tcn",
        architecture_for_model("tcn", BASELINE_TCN_SETTINGS),
        "masked_control",
    )

    def reject_mismatch(
        loaded_manifest: dict[str, object],
        store: Path,
        rows: pl.DataFrame,
    ) -> tuple[object, ...]:
        del rows
        evaluate._validate_run_checkpoint_identity(
            loaded_manifest, checkpoint, store, run_dir
        )
        raise AssertionError("Mismatched checkpoint identity was accepted")

    _patch_evaluation_runtime(monkeypatch, run_dir, neural_evaluator=reject_mismatch)

    with pytest.raises(ValueError, match="Run/checkpoint identity mismatch"):
        evaluate.main()

    assert not list(run_dir.glob("evaluations/*/evaluation_manifest.json"))
