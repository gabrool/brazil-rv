from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch

from .contract import (
    ALLOWED_SEEDS,
    CLOUD_RUNTIME_CONTRACT_VERSION,
    AdamWConstants,
    ArchitectureConstants,
    CONTRACT_VERSION,
    CROSS_ASSET_DEPTH,
    EARLY_STOP_PATIENCE,
    EFFECTIVE_BATCH_SIZE,
    FEATURE_STORE_POINTER,
    MAX_EPOCHS,
    MIN_IC_IMPROVEMENT,
    MODEL_VARIANTS,
    MUON_COMPATIBILITY_CONTRACT_VERSION,
    MuonConstants,
    OPTIMIZER_VARIANTS,
    PROJECT_ROOT,
    RUNTIME_PROFILES,
    RUNTIME_PROFILE_NAMES,
    RUN_OUTPUT_BASE,
    SchedulerConstants,
    SplitBoundaries,
    TEMPORAL_DEPTH,
    TORCH_COMPILE_COMPATIBILITY_CONTRACT_VERSION,
    TrainingConstants,
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
    evaluate_model,
    qualify_eager_compiled_model,
    require_compile_parity,
    train_one_epoch,
    validate_runtime_profile,
    warmup_compiled_model,
)
from .model import CrossAssetPatchITransformerV1, count_trainable_parameters
from .muon import PYTORCH_MUON_REFERENCE
from .optim import build_optimizers, build_schedulers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=RUNTIME_PROFILE_NAMES)
    parser.add_argument("--model", required=True, choices=MODEL_VARIANTS)
    parser.add_argument("--optimizer", required=True, choices=OPTIMIZER_VARIANTS)
    parser.add_argument("--seed", required=True, type=int, choices=ALLOWED_SEEDS)
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    columns = (
        "epoch",
        "optimizer_steps",
        "train_loss",
        "validation_loss",
        "validation_primary_ic",
        "validation_ic_30",
        "validation_ic_60",
        "validation_ic_120",
        "mean_gradient_norm",
        "maximum_gradient_norm",
        "muon_lr",
        "adamw_lr",
        "epoch_seconds",
        "peak_allocated_cuda_memory_bytes",
        "peak_reserved_cuda_memory_bytes",
    )
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


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


def main() -> None:
    args = parse_args()
    profile = RUNTIME_PROFILES[args.profile]
    hardware = validate_runtime_profile(profile)
    commit_sha = clean_git_commit_sha()
    set_seeds(args.seed)
    torch.set_float32_matmul_precision("high")

    feature_store = resolve_feature_store()
    sample_index = validate_feature_store(feature_store)
    train_rows = select_sample_split(sample_index, "train")
    validation_rows = select_sample_split(sample_index, "validation")
    feature_manifest = json.loads(
        (feature_store / "manifest.json").read_text(encoding="utf-8")
    )
    cache_report = warm_feature_store_cache(feature_store)
    train_loader, validation_loader, sampler = create_training_loaders(
        feature_store, train_rows, validation_rows, profile, args.seed
    )
    training_batch = next(iter(train_loader))
    evaluation_batch = next(iter(validation_loader))

    model = CrossAssetPatchITransformerV1(args.model).to("cuda")
    parameter_count = count_trainable_parameters(model)
    if args.model == "full" and not 6_300_000 <= parameter_count <= 6_600_000:
        raise ValueError(
            f"Full-model parameter count is out of range: {parameter_count}"
        )
    optimizers, _, muon_backend = build_optimizers(model, args.optimizer)
    schedulers, steps_per_epoch, warmup_steps = build_schedulers(
        optimizers, train_rows.height
    )
    eager_reference = clone_eager_reference_model(model)
    compile_setup = compile_model(model, profile)
    compile_parity = qualify_eager_compiled_model(
        eager_reference,
        model,
        training_batch,
        include_backward=True,
    )
    require_compile_parity(compile_parity)
    del eager_reference
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    compile_report = warmup_compiled_model(
        model,
        training_batch,
        evaluation_batch,
    )
    compile_metadata = build_compile_metadata(
        compile_setup, compile_parity, compile_report
    )

    created_at = datetime.now(timezone.utc)
    run_dir = RUN_OUTPUT_BASE / (
        "cross_asset_patch_itransformer_v1_"
        f"{args.model}_{args.optimizer}_{profile.name}_seed{args.seed}_"
        f"{created_at:%Y%m%dT%H%M%S%fZ}"
    )
    if run_dir.exists():
        raise FileExistsError(f"Run output already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    training_started_at = datetime.now(timezone.utc)
    manifest: dict[str, object] = {
        "status": "running",
        "contract_version": CONTRACT_VERSION,
        "cloud_runtime_contract_version": CLOUD_RUNTIME_CONTRACT_VERSION,
        "muon_compatibility_contract_version": (MUON_COMPATIBILITY_CONTRACT_VERSION),
        "torch_compile_compatibility_contract_version": (
            TORCH_COMPILE_COMPATIBILITY_CONTRACT_VERSION
        ),
        "muon_backend": muon_backend,
        "muon_reference": dict(PYTORCH_MUON_REFERENCE),
        "feature_store_pointer": str(FEATURE_STORE_POINTER),
        "resolved_feature_store_path": str(feature_store),
        "feature_manifest_contract_version": feature_manifest["contract_version"],
        "model_variant": args.model,
        "optimizer_variant": args.optimizer,
        "runtime_profile": profile.name,
        "runtime_profile_constants": asdict(profile),
        "physical_microbatch_size": profile.microbatch_size,
        "accumulation_steps": profile.accumulation_steps,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "evaluation_batch_size": profile.evaluation_batch_size,
        "num_workers": profile.num_workers,
        "prefetch_factor": profile.prefetch_factor,
        "precision": "bf16",
        "grad_scaler_used": False,
        "seed": args.seed,
        "split_boundaries": {
            key: str(value) for key, value in asdict(SplitBoundaries()).items()
        },
        "architecture_constants": asdict(ArchitectureConstants()),
        "training_constants": asdict(TrainingConstants()),
        "optimizer_constants": {
            "muon": asdict(MuonConstants()),
            "adamw": asdict(AdamWConstants()),
        },
        "scheduler_constants": asdict(SchedulerConstants()),
        "scheduler_steps": {
            "steps_per_epoch": steps_per_epoch,
            "total_steps": steps_per_epoch * MAX_EPOCHS,
            "warmup_steps": warmup_steps,
        },
        "parameter_count": parameter_count,
        "pytorch_version": hardware.pytorch_version,
        "cuda_version": hardware.cuda_version,
        "gpu_name": hardware.device_name,
        "gpu_total_memory": hardware.total_vram_bytes,
        "compile": compile_metadata,
        "hardware": asdict(hardware),
        "feature_cache_warmup": asdict(cache_report),
        "loader": {
            "vectorized": True,
            "batching": "BatchRequest",
            "multiprocessing_context": "spawn",
            "host_arrays": "read_only_memmap",
            "evaluation_padding": "repeat_last_and_mask",
        },
        "peak_memory": {
            "compile_warmup_allocated": (
                compile_report.peak_allocated_cuda_memory_bytes
            ),
            "compile_warmup_reserved": (compile_report.peak_reserved_cuda_memory_bytes),
            "run_allocated": None,
            "run_reserved": None,
        },
        "sdpa_configuration": {
            "is_causal": False,
            "dropout_p": 0.0,
            "enable_gqa": False,
            "backend_selection": "automatic",
        },
        "created_at_utc": created_at.isoformat(),
        "training_started_at_utc": training_started_at.isoformat(),
        "git_commit_sha": commit_sha,
        "best_epoch": None,
        "best_validation_primary_score": None,
        "stopped_epoch": None,
        "training_duration_seconds": None,
        "temporal_depth": TEMPORAL_DEPTH,
        "cross_asset_depth": (CROSS_ASSET_DEPTH if args.model == "full" else 0),
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
        training = train_one_epoch(model, train_loader, optimizers, schedulers, profile)
        validation, daily_rows = evaluate_model(model, validation_loader)
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
                "train_loss": training["train_loss"],
                "validation_loss": validation["masked_huber_loss"],
                "validation_primary_ic": primary_score,
                "validation_ic_30": horizon_ic[0],
                "validation_ic_60": horizon_ic[1],
                "validation_ic_120": horizon_ic[2],
                "mean_gradient_norm": training["mean_gradient_norm"],
                "maximum_gradient_norm": training["maximum_gradient_norm"],
                "muon_lr": training["muon_lr"],
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
                    args.model,
                    args.optimizer,
                    muon_backend,
                    profile.name,
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
            args.model,
            args.optimizer,
            muon_backend,
            profile.name,
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
        run_dir / "validation_daily_metrics.parquet",
        best_daily_rows,
        feature_store,
    )
    completed_manifest = {
        **manifest,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_epoch": best_epoch,
        "best_validation_primary_score": best_score,
        "stopped_epoch": stopped_epoch,
        "training_duration_seconds": time.perf_counter() - started,
        "peak_memory": {
            **manifest["peak_memory"],
            "run_allocated": run_peak_allocated,
            "run_reserved": run_peak_reserved,
        },
    }
    _atomic_write_json(run_dir / "run_manifest.json", completed_manifest)
    print(f"Completed run: {run_dir}")


if __name__ == "__main__":
    main()
