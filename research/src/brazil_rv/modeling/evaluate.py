from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import torch

from .contract import (
    ALLOWED_SEEDS,
    EXPECTED_TRAINABLE_PARAMETER_COUNTS,
    GH200_RUNTIME,
    NEURAL_MODELS,
    XGBOOST_DEVICE,
    XGBOOST_FIXED_PARAMETERS,
    XGBOOST_OBJECTIVE,
    XGBOOST_VERSION,
    architecture_for_model,
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
from .model import build_neural_model
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
    "architecture_constants",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    return parser.parse_args()


def _validate_objective_and_optimizer(identity: dict[str, object]) -> None:
    objective = identity["objective"]
    if not isinstance(objective, dict) or "temperature" not in objective:
        raise ValueError("Invalid neural objective metadata")
    if objective != objective_metadata(float(objective["temperature"])):
        raise ValueError("Invalid neural objective metadata")
    optimizer_variant = identity["optimizer_variant"]
    sam = identity["sam"]
    rho = None if sam is None else float(sam["rho"])
    if sam != sam_metadata(str(optimizer_variant), rho):
        raise ValueError("Invalid neural optimizer metadata")


def _validate_architecture_identity(identity: dict[str, object]) -> None:
    model_name = str(identity["model_name"])
    if model_name not in NEURAL_MODELS:
        raise ValueError(f"Invalid neural model identity: {model_name}")
    expected = asdict(architecture_for_model(model_name))
    if identity["architecture_constants"] != expected:
        raise ValueError(f"Invalid architecture metadata for model: {model_name}")
    expected_parameter_count = EXPECTED_TRAINABLE_PARAMETER_COUNTS[model_name]
    if identity.get("parameter_count") != expected_parameter_count:
        raise ValueError(f"Invalid parameter count for model: {model_name}")


def _validate_run_checkpoint_identity(
    manifest: dict[str, object],
    checkpoint: dict[str, object],
    feature_store: Path,
) -> None:
    for field in _CHECKPOINT_IDENTITY_FIELDS:
        if field not in manifest or field not in checkpoint:
            raise ValueError(f"Missing run/checkpoint identity field: {field}")
        if manifest[field] != checkpoint[field]:
            raise ValueError(f"Run/checkpoint identity mismatch: {field}")
    _validate_architecture_identity(manifest)
    _validate_objective_and_optimizer(manifest)
    manifest_store = Path(str(manifest["resolved_feature_store_path"])).expanduser()
    if manifest_store.resolve() != feature_store:
        raise ValueError("Validated feature store does not match the run identity")


def _validate_xgboost_identity(
    manifest: dict[str, object], feature_store: Path, run_dir: Path
) -> dict[str, str]:
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
    return validate_booster_hashes(run_dir, metadata["booster_sha256"])


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
    _validate_run_checkpoint_identity(manifest, checkpoint, feature_store)
    objective = manifest["objective"]
    temperature = float(objective["temperature"])
    model_name = str(checkpoint["model_name"])
    loader = create_evaluation_loader(
        feature_store,
        rows,
        model_name,
        GH200_RUNTIME,
        int(manifest["seed"]),
    )
    model = build_neural_model(model_name)
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
    summary, daily_rows = evaluate_model(model, loader, temperature)
    torch.cuda.synchronize()
    metadata = {
        "compile": compile_metadata,
        "evaluation_seconds": time.perf_counter() - started,
        "peak_allocated_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_cuda_memory_bytes": torch.cuda.max_memory_reserved(),
    }
    return summary, daily_rows, metadata


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
        booster_sha256 = _validate_xgboost_identity(
            manifest, feature_store, args.run_dir
        )
        xgboost_runtime = validate_xgboost_runtime()
        started = time.perf_counter()
        _, summary, daily_rows, predictions = evaluate_saved_xgboost(
            feature_store,
            training_rows,
            rows,
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
    elif model_family in {
        architecture_for_model(name).family for name in NEURAL_MODELS
    }:
        summary, daily_rows, family_metadata = _evaluate_neural(
            manifest, feature_store, rows
        )
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
        "architecture_constants": manifest["architecture_constants"],
        "parameter_count": manifest["parameter_count"],
        "optimizer_variant": manifest["optimizer_variant"],
        "objective": manifest.get("objective"),
        "sam": manifest.get("sam"),
        "feature_cache_warmup": asdict(cache_report),
        **family_metadata,
    }
    _atomic_write_json(evaluation_dir / "evaluation_manifest.json", evaluation_manifest)
    print(f"Evaluated {args.split}: {evaluation_dir}")


if __name__ == "__main__":
    main()
