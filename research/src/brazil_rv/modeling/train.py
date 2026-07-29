from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch

from .contract import (
    ALLOWED_SEEDS,
    EARLY_STOP_PATIENCE,
    EFFECTIVE_BATCH_SIZE,
    EXPECTED_TRAINABLE_PARAMETER_COUNTS,
    FEATURE_STORE_POINTER,
    GH200_RUNTIME,
    MAX_EPOCHS,
    MIN_IC_IMPROVEMENT,
    NEURAL_MODELS,
    OPTIMIZER_VARIANTS,
    PROJECT_ROOT,
    RUN_OUTPUT_BASE,
    SAM_RHOS,
    SOFT_RANK_TEMPERATURES,
    SUPPORTED_MODELS,
    AdamWConstants,
    SchedulerConstants,
    SplitBoundaries,
    TrainingConstants,
    XGBOOST_DEVICE,
    XGBOOST_FIXED_PARAMETERS,
    XGBOOST_OBJECTIVE,
    architecture_for_model,
)
from .data import (
    create_training_loaders,
    resolve_feature_store,
    select_sample_split,
    validate_feature_store,
    warm_feature_store_cache,
)
from .engine import (
    build_compile_metadata,
    checkpoint_payload,
    clone_eager_reference_model,
    compile_model,
    experiment_decimal,
    evaluate_model,
    objective_metadata,
    qualify_eager_compiled_model,
    require_compile_parity,
    sam_metadata,
    train_one_epoch,
    validate_runtime,
    warmup_compiled_model,
)
from .model import build_neural_model, count_trainable_parameters
from .optim import build_optimizer, build_scheduler
from .xgboost_model import train_xgboost_run, validate_xgboost_runtime


_HISTORY_COLUMNS = (
    "epoch",
    "optimizer_steps",
    "backward_passes",
    "train_loss",
    "validation_soft_spearman_loss",
    "validation_primary_ic",
    "validation_ic_30",
    "validation_ic_60",
    "validation_ic_120",
    "mean_gradient_norm",
    "maximum_gradient_norm",
    "mean_first_pass_sam_gradient_norm",
    "mean_sam_perturbation_norm",
    "mean_second_pass_sam_gradient_norm",
    "all_finite",
    "adamw_lr",
    "epoch_seconds",
    "peak_allocated_cuda_memory_bytes",
    "peak_reserved_cuda_memory_bytes",
)


def validate_cli_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> argparse.Namespace:
    if args.model == "xgboost" and args.optimizer is not None:
        parser.error("--optimizer is not allowed when --model xgboost")
    if args.model == "xgboost" and args.temperature is not None:
        parser.error("--soft-rank-temperature is not allowed when --model xgboost")
    if args.model == "xgboost" and args.sam_rho is not None:
        parser.error("--sam-rho is not allowed when --model xgboost")
    if args.model in NEURAL_MODELS and args.optimizer is None:
        parser.error("--optimizer is required for neural models")
    if args.model in NEURAL_MODELS and args.temperature is None:
        parser.error("--soft-rank-temperature is required for neural models")
    if args.optimizer == "sam_adamw" and args.sam_rho is None:
        parser.error("--sam-rho is required for --optimizer sam_adamw")
    if args.optimizer == "adamw" and args.sam_rho is not None:
        parser.error("--sam-rho is forbidden for --optimizer adamw")
    return args


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--optimizer", choices=OPTIMIZER_VARIANTS)
    parser.add_argument(
        "--soft-rank-temperature",
        dest="temperature",
        type=float,
        choices=SOFT_RANK_TEMPERATURES,
    )
    parser.add_argument("--sam-rho", type=float, choices=SAM_RHOS)
    parser.add_argument("--seed", required=True, type=int, choices=ALLOWED_SEEDS)
    return validate_cli_args(parser, parser.parse_args(arguments))


def set_seeds(seed: int, *, neural: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if neural:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _model_metadata(model_name: str) -> dict[str, object]:
    architecture = architecture_for_model(model_name)
    return {
        "model_name": model_name,
        "model_family": architecture.family,
        "architecture_constants": asdict(architecture),
        "parameter_count": EXPECTED_TRAINABLE_PARAMETER_COUNTS[model_name],
    }


def _run_directory_name(
    model_name: str,
    optimizer_variant: str | None,
    temperature: float | None,
    sam_rho: float | None,
    seed: int,
    created_at: datetime,
) -> str:
    if optimizer_variant is None:
        return f"{model_name}_seed{seed}_{created_at:%Y%m%dT%H%M%S%fZ}"
    if temperature is None:
        raise ValueError("Neural run names require a soft-rank temperature")
    optimizer_part = f"_{optimizer_variant}"
    rho_part = "" if sam_rho is None else f"_rho{experiment_decimal(sam_rho, 3)}"
    tau_part = f"_tau{experiment_decimal(temperature, 2)}"
    return (
        f"{model_name}{optimizer_part}{rho_part}{tau_part}_seed{seed}_"
        f"{created_at:%Y%m%dT%H%M%S%fZ}"
    )


def clean_git_commit_sha() -> str:
    repository = PROJECT_ROOT / "quant" / "b3-quant"
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("Production training requires a clean Git worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_write_history(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=_HISTORY_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_parquet(path: Path, frame: pl.DataFrame) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    frame.write_parquet(temporary)
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _write_daily_metrics(
    path: Path,
    rows: list[dict[str, object]],
    feature_store: Path,
) -> None:
    dates = dict(
        pl.read_parquet(feature_store / "date_index.parquet")
        .select("date_idx", "trade_date")
        .iter_rows()
    )
    with_dates = [{"trade_date": dates[int(row["date_idx"])], **row} for row in rows]
    _atomic_write_parquet(path, pl.DataFrame(with_dates))


def _common_manifest(
    *,
    model_name: str,
    model_family: str,
    optimizer_variant: str | None,
    seed: int,
    commit_sha: str,
    feature_store: Path,
    feature_manifest: dict[str, object],
    hardware: object,
    created_at: datetime,
) -> dict[str, object]:
    return {
        "status": "running",
        "model_name": model_name,
        "model_family": model_family,
        "optimizer_variant": optimizer_variant,
        "seed": seed,
        "git_commit_sha": commit_sha,
        "feature_store_pointer": str(FEATURE_STORE_POINTER),
        "resolved_feature_store_path": str(feature_store),
        "feature_manifest_contract_version": feature_manifest["contract_version"],
        "resolved_source_paths": feature_manifest.get("canonical_inputs"),
        "split_boundaries": {
            key: str(value) for key, value in asdict(SplitBoundaries()).items()
        },
        "hardware": asdict(hardware),
        "created_at_utc": created_at.isoformat(),
        "training_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_at_utc": None,
        "training_duration_seconds": None,
    }


def _run_xgboost(
    *,
    args: argparse.Namespace,
    hardware: object,
    commit_sha: str,
    feature_store: Path,
    feature_manifest: dict[str, object],
    train_rows: pl.DataFrame,
    validation_rows: pl.DataFrame,
    cache_report: object,
    created_at: datetime,
    run_dir: Path,
) -> None:
    xgboost_runtime = validate_xgboost_runtime()
    manifest = {
        **_common_manifest(
            model_name="xgboost",
            model_family="xgboost",
            optimizer_variant=None,
            seed=args.seed,
            commit_sha=commit_sha,
            feature_store=feature_store,
            feature_manifest=feature_manifest,
            hardware=hardware,
            created_at=created_at,
        ),
        "architecture_constants": None,
        "parameter_count": None,
        "compile": None,
        "precision": None,
        "bf16": None,
        "grad_scaler_used": None,
        "optimizer_state": None,
        "checkpoint_identity": None,
        "feature_cache_warmup": asdict(cache_report),
        "xgboost": {
            "version": xgboost_runtime["version"],
            "build_info": xgboost_runtime["build_info"],
            "device": XGBOOST_DEVICE,
            "objective": XGBOOST_OBJECTIVE,
            "fixed_parameters": dict(XGBOOST_FIXED_PARAMETERS),
        },
    }
    _atomic_write_json(run_dir / "run_manifest.json", manifest)
    started = time.perf_counter()
    result = train_xgboost_run(
        feature_store,
        train_rows,
        validation_rows,
        run_dir,
        args.seed,
    )
    completed = {
        **manifest,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_duration_seconds": time.perf_counter() - started,
        "xgboost": {
            **manifest["xgboost"],
            "selected_settings": result.selected_settings,
            "boosting_rounds": result.boosting_rounds,
            "matrix_dimensions": result.matrix_dimensions,
            "validation_primary_score": result.validation_summary["primary_score"],
            "booster_sha256": result.booster_sha256,
            "native_cuda_qualification": result.native_cuda_qualification,
        },
    }
    _atomic_write_json(run_dir / "run_manifest.json", completed)


def _run_neural(
    *,
    args: argparse.Namespace,
    hardware: object,
    commit_sha: str,
    feature_store: Path,
    feature_manifest: dict[str, object],
    train_rows: pl.DataFrame,
    validation_rows: pl.DataFrame,
    cache_report: object,
    created_at: datetime,
    run_dir: Path,
) -> None:
    if args.optimizer is None:
        raise AssertionError("Validated neural CLI is missing its optimizer")
    if args.temperature is None:
        raise AssertionError("Validated neural CLI is missing its temperature")
    runtime = GH200_RUNTIME
    train_loader, validation_loader, sampler = create_training_loaders(
        feature_store,
        train_rows,
        validation_rows,
        args.model,
        runtime,
        args.seed,
    )
    training_batch = next(iter(train_loader))
    evaluation_batch = next(iter(validation_loader))

    model = build_neural_model(args.model).to("cuda")
    model_metadata = _model_metadata(args.model)
    parameter_count = count_trainable_parameters(model)
    expected_parameter_count = int(model_metadata["parameter_count"])
    if parameter_count != expected_parameter_count:
        raise ValueError(
            f"{args.model} parameter count must be {expected_parameter_count}: "
            f"got {parameter_count}"
        )
    optimizer, _ = build_optimizer(model)
    scheduler, steps_per_epoch, warmup_steps = build_scheduler(
        optimizer, train_rows.height
    )
    eager_reference = clone_eager_reference_model(model)
    compile_setup = compile_model(model, runtime)
    compile_parity = qualify_eager_compiled_model(
        eager_reference,
        model,
        training_batch,
        include_backward=True,
        temperature=args.temperature,
    )
    require_compile_parity(compile_parity)
    del eager_reference
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    compile_report = warmup_compiled_model(
        model, training_batch, evaluation_batch, args.temperature
    )
    compile_metadata = build_compile_metadata(
        compile_setup, compile_parity, compile_report
    )

    manifest: dict[str, object] = {
        **_common_manifest(
            model_name=args.model,
            model_family=str(model_metadata["model_family"]),
            optimizer_variant=args.optimizer,
            seed=args.seed,
            commit_sha=commit_sha,
            feature_store=feature_store,
            feature_manifest=feature_manifest,
            hardware=hardware,
            created_at=created_at,
        ),
        **model_metadata,
        "objective": objective_metadata(args.temperature),
        "sam": sam_metadata(args.optimizer, args.sam_rho),
        "physical_microbatch_size": runtime.microbatch_size,
        "accumulation_steps": runtime.accumulation_steps,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "evaluation_batch_size": runtime.evaluation_batch_size,
        "num_workers": runtime.num_workers,
        "prefetch_factor": runtime.prefetch_factor,
        "precision": "bf16",
        "bf16": True,
        "grad_scaler_used": False,
        "training_constants": asdict(TrainingConstants()),
        "optimizer_constants": {"adamw": asdict(AdamWConstants())},
        "scheduler_constants": asdict(SchedulerConstants()),
        "scheduler_steps": {
            "steps_per_epoch": steps_per_epoch,
            "total_steps": steps_per_epoch * MAX_EPOCHS,
            "warmup_steps": warmup_steps,
        },
        "pytorch_version": hardware.pytorch_version,
        "cuda_version": hardware.cuda_version,
        "gpu_name": hardware.device_name,
        "gpu_total_memory": hardware.total_vram_bytes,
        "compile": compile_metadata,
        "feature_cache_warmup": asdict(cache_report),
        "loader": {
            "vectorized": True,
            "batching": "BatchRequest",
            "multiprocessing_context": "spawn",
            "host_arrays": "read_only_memmap",
            "evaluation_padding": "repeat_last_and_mask",
            "sam_replay": "retain_eight_cpu_pinned_batches",
            "sam_rng_replay": "restore_cpu_and_cuda_state_before_second_pass",
        },
        "peak_memory": {
            "compile_warmup_allocated": compile_report.peak_allocated_cuda_memory_bytes,
            "compile_warmup_reserved": compile_report.peak_reserved_cuda_memory_bytes,
            "run_allocated": None,
            "run_reserved": None,
        },
        "sdpa_configuration": {
            "is_causal": False,
            "dropout_p": 0.0,
            "enable_gqa": False,
            "backend_selection": "automatic",
        },
        "best_epoch": None,
        "best_validation_primary_score": None,
        "stopped_epoch": None,
        "bitwise_gpu_reproducibility_guaranteed": False,
    }
    _atomic_write_json(run_dir / "run_manifest.json", manifest)

    started = time.perf_counter()
    history: list[dict[str, object]] = []
    best_score = float("-inf")
    best_epoch = 0
    best_metrics: dict[str, object] | None = None
    best_daily_rows: list[dict[str, object]] | None = None
    evaluations_without_improvement = 0
    stopped_epoch = 0
    run_peak_allocated = 0
    run_peak_reserved = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        sampler.set_epoch(epoch - 1)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        epoch_started = time.perf_counter()
        training = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            runtime,
            args.optimizer,
            args.temperature,
            args.sam_rho,
        )
        validation, daily_rows = evaluate_model(
            model, validation_loader, args.temperature
        )
        torch.cuda.synchronize()
        epoch_seconds = time.perf_counter() - epoch_started
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        run_peak_allocated = max(run_peak_allocated, peak_allocated)
        run_peak_reserved = max(run_peak_reserved, peak_reserved)
        primary_score = float(validation["primary_score"])
        horizon_ic = [
            float(row["mean_daily_spearman_ic"]) for row in validation["horizons"]
        ]
        history.append(
            {
                "epoch": epoch,
                "optimizer_steps": training["optimizer_steps"],
                "backward_passes": training["backward_passes"],
                "train_loss": training["train_loss"],
                "validation_soft_spearman_loss": validation["soft_spearman_loss"],
                "validation_primary_ic": primary_score,
                "validation_ic_30": horizon_ic[0],
                "validation_ic_60": horizon_ic[1],
                "validation_ic_120": horizon_ic[2],
                "mean_gradient_norm": training["mean_gradient_norm"],
                "maximum_gradient_norm": training["maximum_gradient_norm"],
                "mean_first_pass_sam_gradient_norm": training[
                    "mean_first_pass_sam_gradient_norm"
                ],
                "mean_sam_perturbation_norm": training["mean_sam_perturbation_norm"],
                "mean_second_pass_sam_gradient_norm": training[
                    "mean_second_pass_sam_gradient_norm"
                ],
                "all_finite": training["all_finite"],
                "adamw_lr": training["adamw_lr"],
                "epoch_seconds": epoch_seconds,
                "peak_allocated_cuda_memory_bytes": peak_allocated,
                "peak_reserved_cuda_memory_bytes": peak_reserved,
            }
        )
        if primary_score > best_score + MIN_IC_IMPROVEMENT:
            best_score = primary_score
            best_epoch = epoch
            best_metrics = validation
            best_daily_rows = daily_rows
            evaluations_without_improvement = 0
            _atomic_torch_save(
                run_dir / "best.pt",
                checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    args.model,
                    args.optimizer,
                    args.temperature,
                    args.sam_rho,
                    args.seed,
                    epoch,
                    primary_score,
                    feature_store,
                    commit_sha,
                ),
            )
        else:
            evaluations_without_improvement += 1
        stopped_epoch = epoch
        _atomic_write_history(run_dir / "history.csv", history)
        if evaluations_without_improvement >= EARLY_STOP_PATIENCE:
            break

    _atomic_torch_save(
        run_dir / "final.pt",
        checkpoint_payload(
            model,
            optimizer,
            scheduler,
            args.model,
            args.optimizer,
            args.temperature,
            args.sam_rho,
            args.seed,
            stopped_epoch,
            float(history[-1]["validation_primary_ic"]),
            feature_store,
            commit_sha,
        ),
    )
    if best_metrics is None or best_daily_rows is None:
        raise RuntimeError("Training did not produce a best checkpoint")
    _atomic_write_json(run_dir / "validation_metrics.json", best_metrics)
    _write_daily_metrics(
        run_dir / "validation_daily_metrics.parquet", best_daily_rows, feature_store
    )
    completed_manifest = {
        **manifest,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_epoch": best_epoch,
        "best_validation_primary_score": best_score,
        "stopped_epoch": stopped_epoch,
        "successful_optimizer_updates": sum(
            int(row["optimizer_steps"]) for row in history
        ),
        "training_duration_seconds": time.perf_counter() - started,
        "peak_memory": {
            **manifest["peak_memory"],
            "run_allocated": run_peak_allocated,
            "run_reserved": run_peak_reserved,
        },
    }
    _atomic_write_json(run_dir / "run_manifest.json", completed_manifest)


def main() -> None:
    args = parse_args()
    neural = args.model in NEURAL_MODELS
    hardware = validate_runtime()
    commit_sha = clean_git_commit_sha()
    set_seeds(args.seed, neural=neural)
    if neural:
        torch.set_float32_matmul_precision("high")

    feature_store = resolve_feature_store()
    sample_index = validate_feature_store(feature_store)
    train_rows = select_sample_split(sample_index, "train")
    validation_rows = select_sample_split(sample_index, "validation")
    feature_manifest = json.loads(
        (feature_store / "manifest.json").read_text(encoding="utf-8")
    )
    cache_report = warm_feature_store_cache(feature_store)

    created_at = datetime.now(timezone.utc)
    run_dir = RUN_OUTPUT_BASE / _run_directory_name(
        args.model,
        args.optimizer,
        args.temperature,
        args.sam_rho,
        args.seed,
        created_at,
    )
    if run_dir.exists():
        raise FileExistsError(f"Run output already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    if neural:
        _run_neural(
            args=args,
            hardware=hardware,
            commit_sha=commit_sha,
            feature_store=feature_store,
            feature_manifest=feature_manifest,
            train_rows=train_rows,
            validation_rows=validation_rows,
            cache_report=cache_report,
            created_at=created_at,
            run_dir=run_dir,
        )
    else:
        _run_xgboost(
            args=args,
            hardware=hardware,
            commit_sha=commit_sha,
            feature_store=feature_store,
            feature_manifest=feature_manifest,
            train_rows=train_rows,
            validation_rows=validation_rows,
            cache_report=cache_report,
            created_at=created_at,
            run_dir=run_dir,
        )
    print(f"Completed run: {run_dir}")


if __name__ == "__main__":
    main()
