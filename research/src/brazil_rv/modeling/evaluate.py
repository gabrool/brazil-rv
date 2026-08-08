from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import torch

from .context_ablation import (
    get_context_ablation,
    resolve_context_ablation_for_store,
)
from .contract import (
    ALLOWED_SEEDS,
    GLOBAL_CONTEXT_SETTINGS,
    GH200_RUNTIME,
    NeuralArchitecture,
    NEURAL_MODELS,
    TCNArchitecture,
    TCNSettings,
    XGBOOST_DEVICE,
    XGBOOST_FIXED_PARAMETERS,
    XGBOOST_OBJECTIVE,
    XGBOOST_VERSION,
    architecture_for_model,
    expected_trainable_parameter_count,
    model_consumes_context,
)
from .data import (
    create_evaluation_loader,
    select_sample_split,
    validate_feature_store,
    warm_feature_store_cache,
)
from .engine import (
    build_compile_metadata,
    clone_eager_reference_model,
    compile_model,
    evaluate_model,
    objective_metadata,
    qualify_eager_compiled_model,
    require_compile_parity,
    sam_metadata,
    validate_runtime,
    warmup_compiled_evaluation,
)
from .feature_ablation import (
    get_feature_ablation,
    resolve_feature_ablation,
    resolve_feature_ablation_for_store,
)
from .model import build_neural_model
from brazil_rv.preprocessing.contract import SLOW_CHANNELS
from .xgboost_model import (
    evaluate_saved_xgboost,
    validate_booster_hashes,
    validate_xgboost_runtime,
)

_CHECKPOINT_IDENTITY_FIELDS = (
    "model_name",
    "optimizer_variant",
    "objective",
    "sam",
    "seed",
    "resolved_feature_store_path",
    "git_commit_sha",
    "tcn_settings",
    "architecture_constants",
    "parameter_count",
    "feature_manifest_contract_version",
    "global_context",
    "global_context_source_hashes",
    "global_context_normalized_store_hashes",
)

_EXPLICIT_ABLATION_IDENTITY = "explicit_registry_metadata"
_LEGACY_ABLATION_IDENTITY = "legacy_implicit_none"


@dataclass(frozen=True)
class ContextAblationIdentity:
    metadata: dict[str, object]
    source: str


@dataclass(frozen=True)
class FeatureAblationIdentity:
    metadata: dict[str, object]
    source: str


def _validate_context_ablation_metadata(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("context_ablation metadata must be an explicit object")
    key = value.get("key")
    if not isinstance(key, str):
        raise ValueError("context_ablation metadata has no valid key")
    expected = get_context_ablation(key).metadata()
    if value != expected:
        raise ValueError("context_ablation specification identity is invalid")
    return expected


def _ablation_named_run(manifest: dict[str, object], run_dir: Path | None) -> bool:
    candidate = run_dir
    if candidate is None and isinstance(manifest.get("run_dir"), str):
        candidate = Path(str(manifest["run_dir"]))
    return candidate is not None and "_ablation-" in candidate.name


def _normalize_context_ablation_identity(
    manifest: dict[str, object],
    checkpoint: dict[str, object] | None = None,
    *,
    run_dir: Path | None = None,
) -> ContextAblationIdentity:
    manifest_has_metadata = "context_ablation" in manifest
    if checkpoint is not None:
        checkpoint_has_metadata = "context_ablation" in checkpoint
        if manifest_has_metadata != checkpoint_has_metadata:
            raise ValueError(
                "Run/checkpoint identity mismatch: context_ablation presence"
            )

    if not manifest_has_metadata:
        if _ablation_named_run(manifest, run_dir):
            raise ValueError(
                "Ablation-named run cannot use legacy implicit-none identity"
            )
        return ContextAblationIdentity(
            get_context_ablation("none").metadata(),
            _LEGACY_ABLATION_IDENTITY,
        )

    manifest_metadata = _validate_context_ablation_metadata(
        manifest["context_ablation"]
    )
    if checkpoint is not None:
        checkpoint_metadata = _validate_context_ablation_metadata(
            checkpoint["context_ablation"]
        )
        if manifest_metadata != checkpoint_metadata:
            raise ValueError("Run/checkpoint identity mismatch: context_ablation")
    return ContextAblationIdentity(
        manifest_metadata,
        _EXPLICIT_ABLATION_IDENTITY,
    )


def _evaluation_context_ablation_fields(
    identity: ContextAblationIdentity,
) -> dict[str, object]:
    return {
        "context_ablation": identity.metadata,
        "context_ablation_identity_source": identity.source,
    }


def _validate_feature_ablation_metadata(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("feature_ablation metadata must be an explicit object")
    key = value.get("key")
    if not isinstance(key, str):
        raise ValueError("feature_ablation metadata has no valid key")
    expected = resolve_feature_ablation(
        get_feature_ablation(key), slow_features=SLOW_CHANNELS
    ).metadata()
    if value != expected:
        raise ValueError("feature_ablation specification identity is invalid")
    return expected


def _normalize_feature_ablation_identity(
    manifest: dict[str, object],
    checkpoint: dict[str, object] | None = None,
    *,
    run_dir: Path | None = None,
) -> FeatureAblationIdentity:
    manifest_has_metadata = "feature_ablation" in manifest
    if checkpoint is not None:
        checkpoint_has_metadata = "feature_ablation" in checkpoint
        if manifest_has_metadata != checkpoint_has_metadata:
            raise ValueError(
                "Run/checkpoint identity mismatch: feature_ablation presence"
            )
    if not manifest_has_metadata:
        candidate = run_dir
        if candidate is None and isinstance(manifest.get("run_dir"), str):
            candidate = Path(str(manifest["run_dir"]))
        if candidate is not None and "_feature-" in candidate.name:
            raise ValueError(
                "Feature-ablation-named run cannot use legacy implicit-none identity"
            )
        return FeatureAblationIdentity(
            resolve_feature_ablation(
                get_feature_ablation("none"), slow_features=SLOW_CHANNELS
            ).metadata(),
            _LEGACY_ABLATION_IDENTITY,
        )
    manifest_metadata = _validate_feature_ablation_metadata(
        manifest["feature_ablation"]
    )
    if checkpoint is not None:
        checkpoint_metadata = _validate_feature_ablation_metadata(
            checkpoint["feature_ablation"]
        )
        if manifest_metadata != checkpoint_metadata:
            raise ValueError("Run/checkpoint identity mismatch: feature_ablation")
    return FeatureAblationIdentity(
        manifest_metadata,
        _EXPLICIT_ABLATION_IDENTITY,
    )


def _evaluation_feature_ablation_fields(
    identity: FeatureAblationIdentity,
) -> dict[str, object]:
    return {
        "feature_ablation": identity.metadata,
        "feature_ablation_identity_source": identity.source,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    return parser.parse_args()


def _validate_objective_and_optimizer(identity: dict[str, object]) -> None:
    objective = identity["objective"]
    if (
        not isinstance(objective, dict)
        or not {"name", "temperature"} <= objective.keys()
    ):
        raise ValueError("Invalid neural objective metadata")
    raw_temperature = objective["temperature"]
    temperature = None if raw_temperature is None else float(raw_temperature)
    if objective != objective_metadata(str(objective["name"]), temperature):
        raise ValueError("Invalid neural objective metadata")
    optimizer_variant = identity["optimizer_variant"]
    sam = identity["sam"]
    rho = None if sam is None else float(sam["rho"])
    if sam != sam_metadata(str(optimizer_variant), rho):
        raise ValueError("Invalid neural optimizer metadata")


def _architecture_from_identity(
    identity: dict[str, object],
) -> NeuralArchitecture:
    model_name = str(identity["model_name"])
    if model_name not in NEURAL_MODELS:
        raise ValueError(f"Invalid neural model identity: {model_name}")
    raw_settings = identity.get("tcn_settings")
    if model_name == "tcn":
        if not isinstance(raw_settings, dict):
            raise ValueError("TCN settings metadata is missing")
        try:
            settings = TCNSettings(**raw_settings)
        except TypeError as error:
            raise ValueError("Invalid TCN settings metadata") from error
    else:
        if raw_settings is not None:
            raise ValueError(f"TCN settings are forbidden for model: {model_name}")
        settings = None
    return architecture_for_model(model_name, settings)


def _validate_architecture_identity(identity: dict[str, object]) -> None:
    model_name = str(identity["model_name"])
    architecture = _architecture_from_identity(identity)
    expected = asdict(architecture)
    if identity["architecture_constants"] != expected:
        raise ValueError(f"Invalid architecture metadata for model: {model_name}")
    expected_parameter_count = expected_trainable_parameter_count(
        model_name, architecture
    )
    if identity.get("parameter_count") != expected_parameter_count:
        raise ValueError(f"Invalid parameter count for model: {model_name}")


def _validate_global_identity(
    identity: dict[str, object], feature_store: Path
) -> str | None:
    model_name = str(identity["model_name"])
    raw_tcn = identity.get("tcn_settings")
    tcn_settings = (
        TCNSettings(**raw_tcn)
        if model_name == "tcn" and isinstance(raw_tcn, dict)
        else None
    )
    consumes_context = model_consumes_context(model_name, tcn_settings)
    setting = identity.get("global_context")
    if consumes_context and setting not in GLOBAL_CONTEXT_SETTINGS:
        raise ValueError(
            "Context-consuming identity has invalid global context setting"
        )
    if not consumes_context and setting is not None:
        raise ValueError("Context-free identity has a global context setting")

    ablation_metadata = _validate_context_ablation_metadata(
        identity.get("context_ablation")
    )
    ablation_key = str(ablation_metadata["key"])
    if ablation_key != "none" and not consumes_context:
        raise ValueError("Context-free identity has a context ablation")
    if ablation_key != "none" and setting != "enabled":
        raise ValueError("Context ablations require enabled global context")
    feature_ablation = _validate_feature_ablation_metadata(
        identity.get("feature_ablation")
    )
    if model_name == "xgboost" and feature_ablation["key"] != "none":
        raise ValueError("XGBoost identity has a feature ablation")

    feature_manifest = json.loads(
        (feature_store / "manifest.json").read_text(encoding="utf-8")
    )
    global_metadata = feature_manifest["global_context"]
    expected = {
        "feature_manifest_contract_version": feature_manifest["contract_version"],
        "global_context_source_hashes": global_metadata["source_hashes"],
        "global_context_normalized_store_hashes": global_metadata[
            "normalized_store_hashes"
        ],
    }
    for field, value in expected.items():
        if identity.get(field) != value:
            raise ValueError(f"Run identity does not match feature store: {field}")
    return None if setting is None else str(setting)


def _validate_run_checkpoint_identity(
    manifest: dict[str, object],
    checkpoint: dict[str, object],
    feature_store: Path,
    run_dir: Path | None = None,
) -> ContextAblationIdentity:
    context_ablation_identity = _normalize_context_ablation_identity(
        manifest, checkpoint, run_dir=run_dir
    )
    feature_ablation_identity = _normalize_feature_ablation_identity(
        manifest, checkpoint, run_dir=run_dir
    )
    for field in _CHECKPOINT_IDENTITY_FIELDS:
        if field not in manifest or field not in checkpoint:
            raise ValueError(f"Missing run/checkpoint identity field: {field}")
        if manifest[field] != checkpoint[field]:
            raise ValueError(f"Run/checkpoint identity mismatch: {field}")
    _validate_architecture_identity(manifest)
    _validate_objective_and_optimizer(manifest)
    normalized_manifest = {
        **manifest,
        "context_ablation": context_ablation_identity.metadata,
        "feature_ablation": feature_ablation_identity.metadata,
    }
    _validate_global_identity(normalized_manifest, feature_store)
    manifest_store = Path(str(manifest["resolved_feature_store_path"])).expanduser()
    if manifest_store.resolve() != feature_store:
        raise ValueError("Validated feature store does not match the run identity")
    return context_ablation_identity


def _validate_xgboost_identity(
    manifest: dict[str, object], feature_store: Path, run_dir: Path
) -> tuple[dict[str, str], ContextAblationIdentity]:
    if manifest.get("status") != "completed":
        raise ValueError("Standalone XGBoost evaluation requires a completed run")
    if (
        manifest.get("model_name") != "xgboost"
        or manifest.get("model_family") != "xgboost"
    ):
        raise ValueError("Invalid XGBoost run identity")
    if manifest.get("seed") not in ALLOWED_SEEDS:
        raise ValueError("Invalid XGBoost run seed identity")
    commit_sha = manifest.get("git_commit_sha")
    if (
        not isinstance(commit_sha, str)
        or len(commit_sha) != 40
        or any(character not in "0123456789abcdef" for character in commit_sha)
    ):
        raise ValueError("Invalid XGBoost Git SHA identity")
    for field in (
        "optimizer_variant",
        "tcn_settings",
        "architecture_constants",
        "parameter_count",
        "compile",
        "bf16",
    ):
        if manifest.get(field) is not None:
            raise ValueError(f"XGBoost field must be nonapplicable: {field}")
    manifest_store = Path(str(manifest["resolved_feature_store_path"])).expanduser()
    if manifest_store.resolve() != feature_store:
        raise ValueError("Validated feature store does not match the run identity")
    context_ablation_identity = _normalize_context_ablation_identity(
        manifest, run_dir=run_dir
    )
    feature_ablation_identity = _normalize_feature_ablation_identity(
        manifest, run_dir=run_dir
    )
    normalized_manifest = {
        **manifest,
        "context_ablation": context_ablation_identity.metadata,
        "feature_ablation": feature_ablation_identity.metadata,
    }
    _validate_global_identity(normalized_manifest, feature_store)

    metadata = manifest.get("xgboost")
    if not isinstance(metadata, dict):
        raise ValueError("Completed XGBoost metadata is missing")
    if metadata.get("version") != XGBOOST_VERSION:
        raise ValueError("Invalid XGBoost version identity")
    if metadata.get("device") != XGBOOST_DEVICE:
        raise ValueError("Invalid XGBoost device identity")
    if metadata.get("objective") != XGBOOST_OBJECTIVE:
        raise ValueError("Invalid XGBoost objective identity")
    if metadata.get("fixed_parameters") != dict(XGBOOST_FIXED_PARAMETERS):
        raise ValueError("Invalid XGBoost fixed-parameter identity")
    selected = metadata.get("selected_settings")
    if not isinstance(selected, dict):
        raise ValueError("Completed XGBoost selected settings are missing")
    selected_store = Path(str(selected.get("feature_store"))).expanduser()
    if selected_store.resolve() != feature_store:
        raise ValueError("XGBoost selected settings use a different feature store")
    if selected.get("fixed_parameters") != dict(XGBOOST_FIXED_PARAMETERS):
        raise ValueError("Invalid selected XGBoost fixed parameters")
    qualification = metadata.get("native_cuda_qualification")
    if (
        not isinstance(qualification, dict)
        or qualification.get("passed") is not True
        or qualification.get("exact_reload_prediction_equality") is not True
        or not str(qualification.get("device", "")).startswith("cuda")
    ):
        raise ValueError("Completed XGBoost native CUDA qualification is invalid")
    if "booster_sha256" not in metadata:
        raise ValueError("Completed XGBoost booster SHA256 metadata is missing")
    return (
        validate_booster_hashes(run_dir, metadata["booster_sha256"]),
        context_ablation_identity,
    )


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_write_parquet(path: Path, frame: pl.DataFrame) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    frame.write_parquet(temporary)
    os.replace(temporary, path)


def _daily_frame(rows: list[dict[str, object]], feature_store: Path) -> pl.DataFrame:
    dates = dict(
        pl.read_parquet(feature_store / "date_index.parquet")
        .select("date_idx", "trade_date")
        .iter_rows()
    )
    return pl.DataFrame(
        [{"trade_date": dates[int(row["date_idx"])], **row} for row in rows]
    )


def _evaluate_neural(
    manifest: dict[str, object],
    feature_store: Path,
    rows: pl.DataFrame,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    dict[str, object],
    ContextAblationIdentity,
    FeatureAblationIdentity,
]:
    training_compile = manifest.get("compile")
    if not isinstance(training_compile, dict):
        raise ValueError("Run manifest compile metadata is missing")
    training_parity = training_compile.get("parity")
    if (
        not isinstance(training_parity, dict)
        or training_parity.get("passed") is not True
    ):
        raise ValueError("Training run did not pass eager/compiled qualification")
    checkpoint = torch.load(
        Path(str(manifest["run_dir"])) / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    context_ablation_identity = _validate_run_checkpoint_identity(
        manifest,
        checkpoint,
        feature_store,
        Path(str(manifest["run_dir"])),
    )
    feature_ablation_identity = _normalize_feature_ablation_identity(
        manifest,
        checkpoint,
        run_dir=Path(str(manifest["run_dir"])),
    )
    architecture = _architecture_from_identity(checkpoint)
    tcn_architecture = (
        architecture if isinstance(architecture, TCNArchitecture) else None
    )
    objective = manifest["objective"]
    objective_name = str(objective["name"])
    raw_temperature = objective["temperature"]
    temperature = None if raw_temperature is None else float(raw_temperature)
    model_name = str(checkpoint["model_name"])
    context_ablation = resolve_context_ablation_for_store(
        feature_store, str(context_ablation_identity.metadata["key"])
    )
    feature_ablation = resolve_feature_ablation_for_store(
        feature_store, str(feature_ablation_identity.metadata["key"])
    )
    loader = create_evaluation_loader(
        feature_store,
        rows,
        model_name,
        manifest["global_context"],
        GH200_RUNTIME,
        int(manifest["seed"]),
        tcn_architecture,
        context_ablation,
        feature_ablation,
    )
    model = build_neural_model(model_name, tcn_architecture)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to("cuda")
    eager_reference = clone_eager_reference_model(model)
    evaluation_batch = next(iter(loader))
    compile_setup = compile_model(model, GH200_RUNTIME)
    compile_parity = qualify_eager_compiled_model(
        eager_reference,
        model,
        evaluation_batch,
        include_backward=False,
        objective=objective_name,
        temperature=temperature,
    )
    require_compile_parity(compile_parity)
    del eager_reference
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    compile_report = warmup_compiled_evaluation(model, evaluation_batch)
    compile_metadata = build_compile_metadata(
        compile_setup, compile_parity, compile_report
    )
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    summary, daily_rows = evaluate_model(model, loader, objective_name, temperature)
    torch.cuda.synchronize()
    metadata = {
        "compile": compile_metadata,
        "evaluation_seconds": time.perf_counter() - started,
        "peak_allocated_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_cuda_memory_bytes": torch.cuda.max_memory_reserved(),
    }
    return (
        summary,
        daily_rows,
        metadata,
        context_ablation_identity,
        feature_ablation_identity,
    )


def main() -> None:
    args = parse_args()
    hardware = validate_runtime()
    torch.set_float32_matmul_precision("high")

    manifest_path = args.run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Standalone evaluation requires a completed run")
    manifest["run_dir"] = str(args.run_dir.resolve())
    feature_store = (
        Path(str(manifest["resolved_feature_store_path"])).expanduser().resolve()
    )
    sample_index = validate_feature_store(feature_store)
    training_rows = select_sample_split(sample_index, "train")
    rows = select_sample_split(sample_index, args.split)
    cache_report = warm_feature_store_cache(feature_store)

    created_at = datetime.now(timezone.utc)
    evaluation_dir = (
        args.run_dir / "evaluations" / f"{args.split}_{created_at:%Y%m%dT%H%M%S%fZ}"
    )
    if evaluation_dir.exists():
        raise FileExistsError(f"Evaluation output already exists: {evaluation_dir}")
    evaluation_dir.mkdir(parents=True)

    model_family = str(manifest.get("model_family"))
    if model_family == "xgboost":
        (
            booster_sha256,
            context_ablation_identity,
        ) = _validate_xgboost_identity(manifest, feature_store, args.run_dir)
        feature_ablation_identity = _normalize_feature_ablation_identity(
            manifest, run_dir=args.run_dir
        )
        context_ablation = resolve_context_ablation_for_store(
            feature_store, str(context_ablation_identity.metadata["key"])
        )
        xgboost_runtime = validate_xgboost_runtime()
        started = time.perf_counter()
        _, summary, daily_rows, predictions = evaluate_saved_xgboost(
            feature_store,
            training_rows,
            rows,
            str(manifest["global_context"]),
            context_ablation,
            args.run_dir,
            evaluation_dir,
            booster_sha256,
        )
        family_metadata: dict[str, object] = {
            "xgboost_runtime": xgboost_runtime,
            "compile": None,
            "evaluation_seconds": time.perf_counter() - started,
            "peak_allocated_cuda_memory_bytes": None,
            "peak_reserved_cuda_memory_bytes": None,
        }
        _atomic_write_parquet(evaluation_dir / "predictions.parquet", predictions)
    elif model_family in {"transformer", "tcn", "mlp"}:
        (
            summary,
            daily_rows,
            family_metadata,
            context_ablation_identity,
            feature_ablation_identity,
        ) = _evaluate_neural(manifest, feature_store, rows)
    else:
        raise ValueError(f"Unknown model family in run manifest: {model_family}")

    _atomic_write_json(evaluation_dir / "metrics.json", summary)
    _atomic_write_parquet(
        evaluation_dir / "daily_metrics.parquet",
        _daily_frame(daily_rows, feature_store),
    )
    evaluation_manifest = {
        "created_at_utc": created_at.isoformat(),
        "split": args.split,
        "hardware": asdict(hardware),
        "model_name": manifest["model_name"],
        "model_family": manifest["model_family"],
        "tcn_settings": manifest["tcn_settings"],
        "architecture_constants": manifest["architecture_constants"],
        "parameter_count": manifest["parameter_count"],
        "optimizer_variant": manifest["optimizer_variant"],
        "global_context": manifest["global_context"],
        **_evaluation_context_ablation_fields(context_ablation_identity),
        **_evaluation_feature_ablation_fields(feature_ablation_identity),
        "feature_manifest_contract_version": manifest[
            "feature_manifest_contract_version"
        ],
        "global_context_source_hashes": manifest["global_context_source_hashes"],
        "global_context_normalized_store_hashes": manifest[
            "global_context_normalized_store_hashes"
        ],
        "objective": manifest.get("objective"),
        "sam": manifest.get("sam"),
        "feature_cache_warmup": asdict(cache_report),
        **family_metadata,
    }
    _atomic_write_json(evaluation_dir / "evaluation_manifest.json", evaluation_manifest)
    print(f"Evaluated {args.split}: {evaluation_dir}")


if __name__ == "__main__":
    main()
