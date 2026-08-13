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

from .context_ablation import (
    CONTEXT_ABLATION_KEYS,
    ResolvedContextAblation,
    resolve_context_ablation_for_store,
)
from .contract import (
    ALLOWED_SEEDS,
    CONTEXT_ROUTING_EXPERIMENTS,
    CONTEXT_ROUTING_MODES,
    DEFAULT_NEURAL_OBJECTIVE,
    EARLY_STOP_PATIENCE,
    EFFECTIVE_BATCH_SIZE,
    FEATURE_STORE_POINTER,
    GLOBAL_CONTEXT_SETTINGS,
    GH200_RUNTIME,
    MIN_IC_IMPROVEMENT,
    NEURAL_MODELS,
    NEURAL_OBJECTIVES,
    OPTIMIZER_VARIANTS,
    PROJECT_ROOT,
    PEER_FEATURE_MODES,
    RUN_OUTPUT_BASE,
    SAM_RHOS,
    SOFT_RANK_TEMPERATURES,
    SUPPORTED_MODELS,
    AdamWConstants,
    NeuralArchitecture,
    SchedulerConstants,
    SplitBoundaries,
    TCNArchitecture,
    TCNSettings,
    TCN_BLOCK_VARIANTS,
    TCN_FUSIONS,
    TCN_RECEPTIVE_FIELDS,
    TCN_WIDTHS,
    TrainingConstants,
    XGBOOST_DEVICE,
    XGBOOST_FIXED_PARAMETERS,
    model_consumes_context,
    XGBOOST_OBJECTIVE,
    architecture_for_model,
    expected_trainable_parameter_count,
    context_routing_metadata,
    peer_feature_metadata,
)
from .data import (
    create_training_loaders,
    resolve_feature_store,
    select_sample_split,
    validate_feature_store,
    warm_feature_store_cache,
)
from .feature_ablation import (
    FEATURE_ABLATION_KEYS,
    ResolvedFeatureAblation,
    resolve_feature_ablation_for_store,
)
from .engine import (
    PERFORMANCE_PROFILE_VERSION,
    PROFILER_TRACE_FILENAME,
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
from .process_lock import PRODUCTION_TRAINING_LOCK, exclusive_process_lock
from .run_profiles import (
    RUN_PROFILE_NAMES,
    RunProfile,
    filter_profile_rows,
    resolve_run_profile,
    write_run_profile,
)
from .session_preparation import validate_session_preparation
from .xgboost_model import train_xgboost_run, validate_xgboost_runtime


_HISTORY_COLUMNS = (
    "epoch",
    "objective",
    "soft_rank_temperature",
    "optimizer",
    "rho",
    "seed",
    "optimizer_steps",
    "backward_passes",
    "train_objective_loss",
    "validation_objective_loss",
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
    if not hasattr(args, "run_profile"):
        args.run_profile = "production"
    if not hasattr(args, "session_preparation_artifact"):
        args.session_preparation_artifact = None
    if not hasattr(args, "feature_ablation"):
        args.feature_ablation = "none"
    if not hasattr(args, "peer_features"):
        args.peer_features = "none"
    for field in (
        "slow_routing",
        "macro_temporal_routing",
        "context_routing_experiment",
    ):
        if not hasattr(args, field):
            setattr(args, field, None)
    if args.run_profile == "experiment" and args.model != "tcn":
        parser.error("--run-profile experiment is supported only for --model tcn")
    if args.peer_features != "none" and args.model != "tcn":
        parser.error("--peer-features is supported only for --model tcn")
    if args.model == "xgboost" and args.optimizer is not None:
        parser.error("--optimizer is not allowed when --model xgboost")
    if args.model == "xgboost" and args.objective is not None:
        parser.error("--objective is not allowed when --model xgboost")
    if args.model == "xgboost" and args.temperature is not None:
        parser.error("--soft-rank-temperature is not allowed when --model xgboost")
    if args.model == "xgboost" and args.sam_rho is not None:
        parser.error("--sam-rho is not allowed when --model xgboost")
    if args.model == "xgboost" and args.feature_ablation != "none":
        parser.error("--feature-ablation is allowed only for neural models")
    if args.model in NEURAL_MODELS and args.optimizer is None:
        parser.error("--optimizer is required for neural models")
    if args.model in NEURAL_MODELS and args.objective is None:
        args.objective = DEFAULT_NEURAL_OBJECTIVE
    if args.objective == "soft_spearman" and args.temperature is None:
        parser.error("--soft-rank-temperature is required for soft_spearman")
    if args.objective == "rank_huber" and args.temperature is not None:
        parser.error("--soft-rank-temperature is forbidden for rank_huber")
    if args.optimizer == "sam_adamw" and args.sam_rho is None:
        parser.error("--sam-rho is required for --optimizer sam_adamw")
    if args.optimizer == "adamw" and args.sam_rho is not None:
        parser.error("--sam-rho is forbidden for --optimizer adamw")
    routing_values = (
        args.slow_routing,
        args.macro_temporal_routing,
        args.context_routing_experiment,
    )
    if args.model == "tcn":
        if args.slow_routing is None:
            args.slow_routing = "late_only"
        if args.macro_temporal_routing is None:
            args.macro_temporal_routing = "late_only"
        if args.context_routing_experiment is None:
            args.context_routing_experiment = "legacy"
    elif any(value is not None for value in routing_values):
        parser.error("Context-routing arguments are supported only for --model tcn")
    tcn_values = (
        args.tcn_fusion,
        args.tcn_width,
        args.tcn_receptive_field,
        args.tcn_block,
    )
    if args.model == "tcn" and any(value is None for value in tcn_values):
        parser.error(
            "--tcn-fusion, --tcn-width, --tcn-receptive-field, and --tcn-block "
            "are required when --model tcn"
        )
    if args.model != "tcn" and any(value is not None for value in tcn_values):
        parser.error("TCN architecture arguments are forbidden unless --model tcn")
    consumes_context = model_consumes_context(args.model, _tcn_settings_from_args(args))
    if consumes_context and args.global_context is None:
        args.global_context = "enabled"
    if not consumes_context and args.global_context is not None:
        parser.error("--global-context is forbidden for context-free models")
    if not consumes_context and args.context_ablation != "none":
        parser.error("--context-ablation is forbidden for context-free models")
    if args.global_context == "masked" and args.context_ablation != "none":
        parser.error("--global-context masked cannot be combined with an ablation")
    settings = _tcn_settings_from_args(args)
    if settings is not None and settings.context_routing_experiment == "factorial_v1":
        frozen = (
            settings.fusion == "context_pooled"
            and settings.width == 64
            and settings.receptive_field == "full"
            and settings.block == "swiglu"
            and args.peer_features == "selected"
            and args.global_context == "enabled"
            and args.context_ablation == "drop_win_and_global_non_rates"
            and args.feature_ablation == "none"
            and args.objective == "soft_spearman"
            and args.temperature == 0.50
            and args.optimizer == "sam_adamw"
            and args.sam_rho == 0.125
        )
        if not frozen:
            parser.error(
                "factorial_v1 routing requires the frozen width-64 context_pooled "
                "SwiGLU/full selected-peer incumbent configuration"
            )
    return args


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument(
        "--run-profile", choices=RUN_PROFILE_NAMES, default="production"
    )
    parser.add_argument("--session-preparation-artifact", type=Path)
    parser.add_argument("--optimizer", choices=OPTIMIZER_VARIANTS)
    parser.add_argument("--objective", choices=NEURAL_OBJECTIVES)
    parser.add_argument(
        "--soft-rank-temperature",
        dest="temperature",
        type=float,
        choices=SOFT_RANK_TEMPERATURES,
    )
    parser.add_argument("--sam-rho", type=float, choices=SAM_RHOS)
    parser.add_argument("--tcn-fusion", choices=TCN_FUSIONS)
    parser.add_argument("--tcn-width", type=int, choices=TCN_WIDTHS)
    parser.add_argument("--tcn-receptive-field", choices=TCN_RECEPTIVE_FIELDS)
    parser.add_argument("--tcn-block", choices=TCN_BLOCK_VARIANTS)
    parser.add_argument("--slow-routing", choices=CONTEXT_ROUTING_MODES)
    parser.add_argument("--macro-temporal-routing", choices=CONTEXT_ROUTING_MODES)
    parser.add_argument(
        "--context-routing-experiment", choices=CONTEXT_ROUTING_EXPERIMENTS
    )
    parser.add_argument("--global-context", choices=GLOBAL_CONTEXT_SETTINGS)
    parser.add_argument(
        "--context-ablation", choices=CONTEXT_ABLATION_KEYS, default="none"
    )
    parser.add_argument(
        "--feature-ablation", choices=FEATURE_ABLATION_KEYS, default="none"
    )
    parser.add_argument("--peer-features", choices=PEER_FEATURE_MODES, default="none")
    parser.add_argument("--seed", required=True, type=int, choices=ALLOWED_SEEDS)
    return validate_cli_args(parser, parser.parse_args(arguments))


def set_seeds(seed: int, *, neural: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if neural:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _tcn_settings_from_args(args: argparse.Namespace) -> TCNSettings | None:
    if args.model != "tcn":
        return None
    return TCNSettings(
        fusion=args.tcn_fusion,
        width=args.tcn_width,
        receptive_field=args.tcn_receptive_field,
        block=args.tcn_block,
        slow_routing=args.slow_routing,
        macro_temporal_routing=args.macro_temporal_routing,
        context_routing_experiment=args.context_routing_experiment,
    )


def _model_metadata(
    model_name: str,
    architecture: NeuralArchitecture,
    tcn_settings: TCNSettings | None,
    peer_features: str = "none",
) -> dict[str, object]:
    return {
        "model_name": model_name,
        "model_family": architecture.family,
        "tcn_settings": None if tcn_settings is None else asdict(tcn_settings),
        "context_routing": (
            context_routing_metadata(architecture)
            if isinstance(architecture, TCNArchitecture)
            else None
        ),
        "architecture_constants": asdict(architecture),
        "parameter_count": expected_trainable_parameter_count(
            model_name, architecture, peer_features
        ),
        "peer_features": peer_feature_metadata(model_name, architecture, peer_features),
    }


def _run_directory_name(
    model_name: str,
    tcn_settings: TCNSettings | None,
    optimizer_variant: str | None,
    objective: str | None,
    temperature: float | None,
    sam_rho: float | None,
    global_context: str | None,
    seed: int,
    created_at: datetime,
    context_ablation: str = "none",
    feature_ablation: str = "none",
    peer_features: str = "none",
    run_profile: str = "production",
) -> str:
    context_part = "" if global_context is None else f"_global-{global_context}"
    ablation_part = (
        "" if context_ablation == "none" else f"_ablation-{context_ablation}"
    )
    feature_part = "" if feature_ablation == "none" else f"_feature-{feature_ablation}"
    peer_part = "" if peer_features == "none" else f"_peer-{peer_features}"
    profile_part = "" if run_profile == "production" else "_profile-experiment"
    if optimizer_variant is None:
        return f"{model_name}{context_part}{ablation_part}{feature_part}{profile_part}_seed{seed}_{created_at:%Y%m%dT%H%M%S%fZ}"
    if objective is None:
        raise ValueError("Neural run names require an objective")
    objective_metadata(objective, temperature)
    model_part = model_name
    if tcn_settings is not None:
        model_part = (
            f"tcn_{tcn_settings.fusion}_w{tcn_settings.width}_"
            f"rf{tcn_settings.receptive_field}_b{tcn_settings.block}"
        )
        if tcn_settings.context_routing_experiment != "legacy":
            model_part += (
                f"_routing-{tcn_settings.context_routing_experiment}"
                f"_slow-{tcn_settings.slow_routing}"
                f"_macro-{tcn_settings.macro_temporal_routing}"
            )
    objective_part = f"_{objective}"
    optimizer_part = f"_{optimizer_variant}"
    rho_part = "" if sam_rho is None else f"_rho{experiment_decimal(sam_rho, 3)}"
    tau_part = (
        "" if temperature is None else f"_tau{experiment_decimal(temperature, 2)}"
    )
    return (
        f"{model_part}{objective_part}{optimizer_part}{rho_part}{tau_part}{context_part}{ablation_part}{feature_part}{peer_part}{profile_part}_seed{seed}_"
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


def _atomic_write_history(
    path: Path,
    rows: list[dict[str, object]],
    run_profile: RunProfile | None = None,
) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as output:
            experiment = run_profile is not None and run_profile.name == "experiment"
            fields = (
                (*_HISTORY_COLUMNS, "run_profile", "run_profile_identity_sha256")
                if experiment
                else _HISTORY_COLUMNS
            )
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                (
                    {
                        **row,
                        "run_profile": run_profile.name,
                        "run_profile_identity_sha256": run_profile.identity_sha256,
                    }
                    if experiment
                    else row
                )
                for row in rows
            )
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
    run_profile: RunProfile | None = None,
) -> None:
    dates = dict(
        pl.read_parquet(feature_store / "date_index.parquet")
        .select("date_idx", "trade_date")
        .iter_rows()
    )
    with_dates = [{"trade_date": dates[int(row["date_idx"])], **row} for row in rows]
    if run_profile is not None and run_profile.name == "experiment":
        with_dates = [
            {
                **row,
                "run_profile": run_profile.name,
                "run_profile_identity_sha256": run_profile.identity_sha256,
            }
            for row in with_dates
        ]
    _atomic_write_parquet(path, pl.DataFrame(with_dates))


def _common_manifest(
    *,
    model_name: str,
    model_family: str,
    optimizer_variant: str | None,
    global_context: str | None,
    context_ablation: ResolvedContextAblation,
    feature_ablation: ResolvedFeatureAblation,
    seed: int,
    commit_sha: str,
    feature_store: Path,
    feature_manifest: dict[str, object],
    hardware: object,
    created_at: datetime,
    run_profile: RunProfile,
) -> dict[str, object]:
    return {
        "status": "running",
        "model_name": model_name,
        "model_family": model_family,
        "optimizer_variant": optimizer_variant,
        "global_context": global_context,
        "context_ablation": context_ablation.metadata(),
        "feature_ablation": feature_ablation.metadata(),
        "global_context_source_hashes": feature_manifest["global_context"][
            "source_hashes"
        ],
        "global_context_normalized_store_hashes": feature_manifest["global_context"][
            "normalized_store_hashes"
        ],
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
        "run_profile": run_profile.metadata(),
        "run_profile_identity_sha256": run_profile.identity_sha256,
    }


def _run_xgboost(
    *,
    args: argparse.Namespace,
    context_ablation: ResolvedContextAblation,
    feature_ablation: ResolvedFeatureAblation,
    hardware: object,
    commit_sha: str,
    feature_store: Path,
    feature_manifest: dict[str, object],
    train_rows: pl.DataFrame,
    validation_rows: pl.DataFrame,
    cache_report: object,
    created_at: datetime,
    run_dir: Path,
    run_profile: RunProfile,
) -> None:
    xgboost_runtime = validate_xgboost_runtime()
    manifest = {
        **_common_manifest(
            model_name="xgboost",
            model_family="xgboost",
            optimizer_variant=None,
            global_context=args.global_context,
            context_ablation=context_ablation,
            feature_ablation=feature_ablation,
            seed=args.seed,
            commit_sha=commit_sha,
            feature_store=feature_store,
            feature_manifest=feature_manifest,
            hardware=hardware,
            created_at=created_at,
            run_profile=run_profile,
        ),
        "tcn_settings": None,
        "architecture_constants": None,
        "parameter_count": None,
        "peer_features": peer_feature_metadata("xgboost", None, args.peer_features),
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
        args.global_context,
        context_ablation,
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
    context_ablation: ResolvedContextAblation,
    feature_ablation: ResolvedFeatureAblation,
    tcn_settings: TCNSettings | None,
    hardware: object,
    commit_sha: str,
    feature_store: Path,
    feature_manifest: dict[str, object],
    train_rows: pl.DataFrame,
    validation_rows: pl.DataFrame,
    cache_report: object,
    created_at: datetime,
    run_dir: Path,
    run_profile: RunProfile,
) -> None:
    if args.optimizer is None:
        raise AssertionError("Validated neural CLI is missing its optimizer")
    if args.objective is None:
        raise AssertionError("Validated neural CLI is missing its objective")
    architecture = architecture_for_model(args.model, tcn_settings)
    tcn_architecture = (
        architecture if isinstance(architecture, TCNArchitecture) else None
    )
    runtime = GH200_RUNTIME
    train_loader, validation_loader, sampler = create_training_loaders(
        feature_store,
        train_rows,
        validation_rows,
        args.model,
        args.global_context,
        runtime,
        args.seed,
        tcn_architecture,
        context_ablation,
        feature_ablation,
        args.peer_features,
        run_profile,
    )
    training_batch = next(iter(train_loader))
    evaluation_batch = next(iter(validation_loader))

    model = build_neural_model(
        args.model, tcn_architecture, args.peer_features, run_profile.equity_count
    ).to("cuda")
    model_metadata = _model_metadata(
        args.model, architecture, tcn_settings, args.peer_features
    )
    parameter_count = count_trainable_parameters(model)
    expected_parameter_count = int(model_metadata["parameter_count"])
    if parameter_count != expected_parameter_count:
        raise ValueError(
            f"{args.model} parameter count must be {expected_parameter_count}: "
            f"got {parameter_count}"
        )
    optimizer, _ = build_optimizer(model)
    scheduler, steps_per_epoch, warmup_steps = build_scheduler(
        optimizer, train_rows.height, run_profile.maximum_epochs
    )
    eager_reference = clone_eager_reference_model(model)
    compile_setup = compile_model(model, runtime)
    compile_parity = qualify_eager_compiled_model(
        eager_reference,
        model,
        training_batch,
        include_backward=True,
        objective=args.objective,
        temperature=args.temperature,
    )
    require_compile_parity(compile_parity)
    del eager_reference
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    compile_report = warmup_compiled_model(
        model,
        training_batch,
        evaluation_batch,
        args.objective,
        args.temperature,
    )
    compile_metadata = build_compile_metadata(
        compile_setup, compile_parity, compile_report
    )

    manifest: dict[str, object] = {
        **_common_manifest(
            model_name=args.model,
            model_family=str(model_metadata["model_family"]),
            optimizer_variant=args.optimizer,
            global_context=args.global_context,
            context_ablation=context_ablation,
            feature_ablation=feature_ablation,
            seed=args.seed,
            commit_sha=commit_sha,
            feature_store=feature_store,
            feature_manifest=feature_manifest,
            hardware=hardware,
            created_at=created_at,
            run_profile=run_profile,
        ),
        **model_metadata,
        "objective": objective_metadata(args.objective, args.temperature),
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
        "training_constants": asdict(
            TrainingConstants(maximum_epochs=run_profile.maximum_epochs)
        ),
        "optimizer_constants": {"adamw": asdict(AdamWConstants())},
        "scheduler_constants": asdict(
            SchedulerConstants(maximum_epochs=run_profile.maximum_epochs)
        ),
        "scheduler_steps": {
            "steps_per_epoch": steps_per_epoch,
            "total_steps": steps_per_epoch * run_profile.maximum_epochs,
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
            "sam_replay": "one_device_transfer_per_effective_batch_reused_by_both_passes",
            "decision_grouped_batches": run_profile.decision_grouped_batches,
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
    performance_epochs: list[dict[str, object]] = []
    best_score = float("-inf")
    best_epoch = 0
    best_metrics: dict[str, object] | None = None
    best_daily_rows: list[dict[str, object]] | None = None
    evaluations_without_improvement = 0
    stopped_epoch = 0
    run_peak_allocated = 0
    run_peak_reserved = 0
    bounded_training_update: dict[str, object] | None = None
    bounded_validation_batch: dict[str, object] | None = None
    profiler_trace: dict[str, object] | None = None
    profiler_trace_path = run_dir / PROFILER_TRACE_FILENAME
    profile_bounded_phases = run_profile.name == "experiment"

    for epoch in range(1, run_profile.maximum_epochs + 1):
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
            args.objective,
            args.temperature,
            args.sam_rho,
            epoch=epoch,
            profiler_trace_path=(
                profiler_trace_path
                if profile_bounded_phases and epoch == 1
                else None
            ),
        )
        validation_performance: dict[str, object] = {}
        validation, daily_rows = evaluate_model(
            model,
            validation_loader,
            args.objective,
            args.temperature,
            performance=validation_performance,
            profile_first_batch=profile_bounded_phases and epoch == 1,
        )
        torch.cuda.synchronize()
        if epoch == 1 and profile_bounded_phases:
            bounded_training_update = training["bounded_training_update"]
            bounded_validation_batch = validation_performance.pop(
                "bounded_validation_batch"
            )
            profiler_trace = training["profiler_trace"]
        else:
            validation_performance.pop("bounded_validation_batch", None)
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
                "objective": args.objective,
                "soft_rank_temperature": args.temperature,
                "optimizer": args.optimizer,
                "rho": args.sam_rho,
                "seed": args.seed,
                "optimizer_steps": training["optimizer_steps"],
                "backward_passes": training["backward_passes"],
                "train_objective_loss": training["train_objective_loss"],
                "validation_objective_loss": validation["objective_loss"],
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
        artifact_started = time.perf_counter()
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
                    architecture,
                    tcn_settings,
                    args.optimizer,
                    args.objective,
                    args.temperature,
                    args.sam_rho,
                    args.seed,
                    epoch,
                    primary_score,
                    feature_store,
                    args.global_context,
                    feature_manifest,
                    commit_sha,
                    context_ablation,
                    feature_ablation,
                    args.peer_features,
                    run_profile.metadata(),
                ),
            )
        else:
            evaluations_without_improvement += 1
        stopped_epoch = epoch
        _atomic_write_history(run_dir / "history.csv", history, run_profile)
        epoch_artifact_io_seconds = (
            time.perf_counter()
            - artifact_started
            + float(
                training["performance"][
                    "profiler_trace_artifact_io_wall_seconds"
                ]
            )
        )
        performance_epochs.append(
            {
                "epoch": epoch,
                "aggregate_wall_clock": {
                    "training_seconds": training["performance"][
                        "total_epoch_training_wall_seconds"
                    ],
                    "validation_seconds": validation_performance[
                        "total_validation_wall_seconds"
                    ],
                    "artifact_io_seconds": epoch_artifact_io_seconds,
                    "epoch_seconds": time.perf_counter() - epoch_started,
                },
                "training": training["performance"],
                "validation": validation_performance,
                "training_decision_grouping": training["decision_grouping"],
                "peak_cuda_memory": {
                    "allocated_bytes": peak_allocated,
                    "reserved_bytes": peak_reserved,
                },
            }
        )
        if evaluations_without_improvement >= EARLY_STOP_PATIENCE:
            break

    final_artifact_started = time.perf_counter()
    _atomic_torch_save(
        run_dir / "final.pt",
        checkpoint_payload(
            model,
            optimizer,
            scheduler,
            args.model,
            architecture,
            tcn_settings,
            args.optimizer,
            args.objective,
            args.temperature,
            args.sam_rho,
            args.seed,
            stopped_epoch,
            float(history[-1]["validation_primary_ic"]),
            feature_store,
            args.global_context,
            feature_manifest,
            commit_sha,
            context_ablation,
            feature_ablation,
            args.peer_features,
            run_profile.metadata(),
        ),
    )
    if best_metrics is None or best_daily_rows is None:
        raise RuntimeError("Training did not produce a best checkpoint")
    best_metrics["run_profile"] = run_profile.metadata()
    best_metrics["run_profile_identity_sha256"] = run_profile.identity_sha256
    _atomic_write_json(run_dir / "validation_metrics.json", best_metrics)
    _write_daily_metrics(
        run_dir / "validation_daily_metrics.parquet",
        best_daily_rows,
        feature_store,
        run_profile,
    )
    final_artifact_io_seconds = time.perf_counter() - final_artifact_started
    if profile_bounded_phases and (
        bounded_training_update is None
        or bounded_validation_batch is None
        or profiler_trace is None
    ):
        raise RuntimeError("Bounded performance profiling did not complete")
    whole_run_wall_seconds = time.perf_counter() - started
    aggregate_training_seconds = sum(
        float(epoch["aggregate_wall_clock"]["training_seconds"])
        for epoch in performance_epochs
    )
    aggregate_validation_seconds = sum(
        float(epoch["aggregate_wall_clock"]["validation_seconds"])
        for epoch in performance_epochs
    )
    aggregate_artifact_io_seconds = final_artifact_io_seconds + sum(
        float(epoch["aggregate_wall_clock"]["artifact_io_seconds"])
        for epoch in performance_epochs
    )
    performance_profile = {
        "version": PERFORMANCE_PROFILE_VERSION,
        "run_profile": run_profile.name,
        "run_profile_identity_sha256": run_profile.identity_sha256,
        "measurement_contract": {
            "aggregate_wall_clock_clock": "time.perf_counter",
            "bounded_sampling_enabled": profile_bounded_phases,
            "bounded_cuda_clock": (
                "torch.cuda.Event" if profile_bounded_phases else None
            ),
            "bounded_training_scope": (
                "first_completed_effective_training_update_of_epoch_1"
                if profile_bounded_phases
                else None
            ),
            "bounded_validation_scope": (
                "first_validation_batch_of_epoch_1"
                if profile_bounded_phases
                else None
            ),
            "cuda_synchronization_policy": (
                "one synchronization at each bounded CUDA profiling boundary only"
                if profile_bounded_phases
                else "bounded_cuda_profiling_not_collected_for_production"
            ),
            "sampled_cuda_timings_are_not_extrapolated": True,
            "worker_construction_is_sum_across_workers_and_may_overlap": True,
        },
        "epochs": performance_epochs,
        "bounded_training_update": bounded_training_update,
        "bounded_validation_batch": bounded_validation_batch,
        "profiler_trace": profiler_trace,
        "final_artifact_io_wall_seconds": final_artifact_io_seconds,
        "whole_run": {
            "training_wall_seconds": aggregate_training_seconds,
            "validation_wall_seconds": aggregate_validation_seconds,
            "artifact_io_wall_seconds": aggregate_artifact_io_seconds,
            "run_wall_seconds": whole_run_wall_seconds,
            "h2d_bytes": sum(
                int(epoch["training"]["h2d_bytes"])
                + int(epoch["validation"]["h2d_bytes"])
                for epoch in performance_epochs
            ),
        },
        "peak_cuda_memory": {
            "allocated_bytes": run_peak_allocated,
            "reserved_bytes": run_peak_reserved,
        },
    }
    _atomic_write_json(run_dir / "performance_profile.json", performance_profile)
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


def _run(args: argparse.Namespace) -> None:
    tcn_settings = _tcn_settings_from_args(args)
    neural = args.model in NEURAL_MODELS
    hardware = validate_runtime()
    commit_sha = clean_git_commit_sha()
    set_seeds(args.seed, neural=neural)
    if neural:
        torch.set_float32_matmul_precision("high")

    feature_store = resolve_feature_store()
    run_profile = resolve_run_profile(args.run_profile, feature_store)
    preparation_value = args.session_preparation_artifact or os.environ.get(
        "BRAZIL_RV_SESSION_PREPARATION_ARTIFACT"
    )
    if preparation_value is None:
        sample_index = validate_feature_store(feature_store)
        cache_report = warm_feature_store_cache(feature_store, args.peer_features)
    else:
        sample_index, cache_report = validate_session_preparation(
            Path(preparation_value).expanduser().resolve(),
            feature_store,
            commit_sha,
            run_profile,
        )
    context_ablation = resolve_context_ablation_for_store(
        feature_store, args.context_ablation
    )
    feature_ablation = resolve_feature_ablation_for_store(
        feature_store, args.feature_ablation
    )
    train_rows = filter_profile_rows(
        select_sample_split(sample_index, "train"),
        feature_store,
        run_profile,
    )
    validation_rows = filter_profile_rows(
        select_sample_split(sample_index, "validation"),
        feature_store,
        run_profile,
        require_training_dates=False,
    )
    feature_manifest = json.loads(
        (feature_store / "manifest.json").read_text(encoding="utf-8")
    )
    created_at = datetime.now(timezone.utc)
    run_dir = RUN_OUTPUT_BASE / _run_directory_name(
        args.model,
        tcn_settings,
        args.optimizer,
        args.objective,
        args.temperature,
        args.sam_rho,
        args.global_context,
        args.seed,
        created_at,
        args.context_ablation,
        args.feature_ablation,
        args.peer_features,
        args.run_profile,
    )
    if run_dir.exists():
        raise FileExistsError(f"Run output already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    write_run_profile(run_dir / "run_profile.json", run_profile)

    if neural:
        _run_neural(
            args=args,
            context_ablation=context_ablation,
            feature_ablation=feature_ablation,
            tcn_settings=tcn_settings,
            hardware=hardware,
            commit_sha=commit_sha,
            feature_store=feature_store,
            feature_manifest=feature_manifest,
            train_rows=train_rows,
            validation_rows=validation_rows,
            cache_report=cache_report,
            created_at=created_at,
            run_dir=run_dir,
            run_profile=run_profile,
        )
    else:
        _run_xgboost(
            args=args,
            context_ablation=context_ablation,
            feature_ablation=feature_ablation,
            hardware=hardware,
            commit_sha=commit_sha,
            feature_store=feature_store,
            feature_manifest=feature_manifest,
            train_rows=train_rows,
            validation_rows=validation_rows,
            cache_report=cache_report,
            created_at=created_at,
            run_dir=run_dir,
            run_profile=run_profile,
        )
    print(f"Completed run: {run_dir}")


def main() -> None:
    args = parse_args()
    with exclusive_process_lock(PRODUCTION_TRAINING_LOCK, "production training"):
        _run(args)


if __name__ == "__main__":
    main()
