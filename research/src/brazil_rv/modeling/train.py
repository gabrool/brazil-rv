from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch

from .contract import (
    ALLOWED_SEEDS,
    BASELINE_TCN_SETTINGS,
    CONTEXT_FAMILY_ABLATIONS,
    CONTEXT_ROUTING_MODES,
    EARLY_STOP_PATIENCE,
    GLOBAL_CONTEXT_SETTINGS,
    GH200_RUNTIME,
    MAX_EPOCHS,
    MIN_IC_IMPROVEMENT,
    NEURAL_MODELS,
    NEURAL_OBJECTIVES,
    OPTIMIZER_VARIANTS,
    PEER_FEATURE_MODES,
    RUN_OUTPUT_BASE,
    SAM_RHOS,
    SOFT_RANK_TEMPERATURES,
    TCN_BLOCK_VARIANTS,
    TCN_FUSIONS,
    TCN_READOUTS,
    TCN_RECEPTIVE_FIELDS,
    TCN_WIDTHS,
    TRAINING_HORIZONS,
    TCNArchitecture,
    TCNSettings,
    architecture_for_model,
    model_consumes_context,
)
from .data import (
    create_training_loaders,
    feature_store_identity,
    load_sample_index,
    resolve_feature_store,
    sample_window_metadata,
    select_sample_split,
)
from .engine import (
    EvaluationObservations,
    checkpoint_payload,
    collect_validation_observations,
    compile_model,
    compile_training_objective,
    experiment_decimal,
    objective_metadata,
    sam_metadata,
    summarize_evaluation_observations,
    train_one_epoch,
    validation_primary_metric,
)
from .model import build_neural_model, count_trainable_parameters
from .optim import build_optimizer, build_scheduler
from .provenance import build_run_provenance, repository_commit


def validate_cli_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> argparse.Namespace:
    args.optimizer = args.optimizer or "sam_adamw"
    args.objective = args.objective or "soft_spearman"
    if args.objective == "soft_spearman":
        args.temperature = 0.50 if args.temperature is None else args.temperature
    elif args.temperature is not None:
        parser.error("rank_huber does not accept --soft-rank-temperature")
    if args.optimizer == "sam_adamw":
        args.sam_rho = 0.125 if args.sam_rho is None else args.sam_rho
    elif args.sam_rho is not None:
        parser.error("AdamW does not accept --sam-rho")
    if args.model != "tcn" and args.tcn_readout != "final":
        parser.error("TCN readouts are supported only for TCN")
    args.peer_features = (
        ("selected" if args.model == "tcn" else "none")
        if args.peer_features is None
        else args.peer_features
    )
    if args.model != "tcn" and args.peer_features != "none":
        parser.error("Peer features are supported only for TCN")
    settings = _tcn_settings_from_args(args)
    consumes_context = model_consumes_context(args.model, settings)
    if consumes_context:
        args.global_context = args.global_context or "enabled"
    elif args.global_context is not None:
        parser.error("Context-free models do not accept --global-context")
    if not consumes_context and args.context_family_ablation != "none":
        parser.error("Context ablation requires context inputs")
    return args


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one current full-universe model"
    )
    parser.add_argument("--model", choices=NEURAL_MODELS, default="tcn")
    parser.add_argument("--optimizer", choices=OPTIMIZER_VARIANTS)
    parser.add_argument("--objective", choices=NEURAL_OBJECTIVES)
    parser.add_argument(
        "--soft-rank-temperature",
        dest="temperature",
        type=float,
        choices=SOFT_RANK_TEMPERATURES,
    )
    parser.add_argument("--sam-rho", type=float, choices=SAM_RHOS)
    parser.add_argument(
        "--tcn-fusion", choices=TCN_FUSIONS, default=BASELINE_TCN_SETTINGS.fusion
    )
    parser.add_argument(
        "--tcn-width", type=int, choices=TCN_WIDTHS, default=BASELINE_TCN_SETTINGS.width
    )
    parser.add_argument(
        "--tcn-receptive-field",
        choices=TCN_RECEPTIVE_FIELDS,
        default=BASELINE_TCN_SETTINGS.receptive_field,
    )
    parser.add_argument(
        "--tcn-block", choices=TCN_BLOCK_VARIANTS, default=BASELINE_TCN_SETTINGS.block
    )
    parser.add_argument(
        "--slow-routing", choices=CONTEXT_ROUTING_MODES, default="late_only"
    )
    parser.add_argument(
        "--macro-temporal-routing", choices=CONTEXT_ROUTING_MODES, default="late_only"
    )
    parser.add_argument(
        "--tcn-readout", choices=TCN_READOUTS, default=BASELINE_TCN_SETTINGS.readout
    )
    parser.add_argument("--training-horizon", choices=TRAINING_HORIZONS, default="all")
    parser.add_argument(
        "--context-family-ablation",
        choices=CONTEXT_FAMILY_ABLATIONS,
        default="none",
    )
    parser.add_argument("--global-context", choices=GLOBAL_CONTEXT_SETTINGS)
    parser.add_argument("--peer-features", choices=PEER_FEATURE_MODES)
    parser.add_argument("--seed", type=int, choices=ALLOWED_SEEDS, default=29)
    parser.add_argument("--output-base", type=Path, default=RUN_OUTPUT_BASE)
    return validate_cli_args(parser, parser.parse_args(arguments))


def _tcn_settings_from_args(args: argparse.Namespace) -> TCNSettings | None:
    if args.model != "tcn":
        return None
    return TCNSettings(
        args.tcn_fusion,
        args.tcn_width,
        args.tcn_receptive_field,
        args.tcn_block,
        args.slow_routing,
        args.macro_temporal_routing,
        args.tcn_readout,
    )


def _run_directory_name(args: argparse.Namespace, created_at: datetime) -> str:
    if args.model == "tcn":
        model = (
            f"tcn_{args.tcn_fusion}_w{args.tcn_width}"
            f"_rf{args.tcn_receptive_field}_b{args.tcn_block}"
            f"_readout-{args.tcn_readout}"
        )
        if (
            args.slow_routing != "late_only"
            or args.macro_temporal_routing != "late_only"
        ):
            model += f"_slow-{args.slow_routing}_macro-{args.macro_temporal_routing}"
    else:
        model = args.model
    model += f"_{args.objective}_{args.optimizer}"
    if args.sam_rho is not None:
        model += f"_rho{experiment_decimal(args.sam_rho, 3)}"
    if args.temperature is not None:
        model += f"_tau{experiment_decimal(args.temperature, 2)}"
    if args.global_context is not None:
        model += f"_global-{args.global_context}"
    if args.peer_features != "none":
        model += f"_peer-{args.peer_features}"
    if args.training_horizon != "all":
        model += f"_horizon-{args.training_horizon}"
    if args.context_family_ablation != "none":
        model += f"_without-{args.context_family_ablation}"
    return f"{model}_seed{args.seed}_{created_at:%Y%m%dT%H%M%S%fZ}"


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _write_history(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _run_neural(
    args: argparse.Namespace,
    store: Path,
    train_rows: pl.DataFrame,
    validation_rows: pl.DataFrame,
    run_dir: Path,
    *,
    fit_name: str = "train",
    selection_name: str = "validation",
    allow_date_replacement: bool = False,
    feature_store_metadata: dict[str, object] | None = None,
) -> None:
    torch.set_float32_matmul_precision("high")
    settings = _tcn_settings_from_args(args)
    architecture = architecture_for_model(args.model, settings)
    store_identity = (
        feature_store_identity(store)
        if feature_store_metadata is None
        else feature_store_metadata
    )
    fit_window = sample_window_metadata(train_rows, fit_name)
    selection_window = sample_window_metadata(validation_rows, selection_name)
    loaders = create_training_loaders(
        store,
        train_rows,
        validation_rows,
        args.model,
        args.global_context,
        GH200_RUNTIME,
        args.seed,
        architecture if isinstance(architecture, TCNArchitecture) else None,
        args.peer_features,
        args.context_family_ablation,
        allow_date_replacement,
    )
    train_loader, validation_loader, sampler = loaders
    model = build_neural_model(
        args.model,
        architecture if isinstance(architecture, TCNArchitecture) else None,
        args.peer_features,
    ).cuda()
    parameter_count = count_trainable_parameters(model)
    optimizer, _ = build_optimizer(model)
    scheduler, steps_per_epoch, warmup_steps = build_scheduler(
        optimizer, train_rows.height, MAX_EPOCHS
    )
    objective = objective_metadata(args.objective, args.temperature)
    sam = sam_metadata(args.optimizer, args.sam_rho)
    run_provenance = build_run_provenance(
        repository_commit_value=repository_commit(),
        feature_store=store,
        feature_store_metadata=store_identity,
        model_name=args.model,
        architecture=architecture,
        settings=settings,
        peer_features=args.peer_features,
        global_context=args.global_context,
        objective=objective,
        optimizer=args.optimizer,
        sam=sam,
        seed=args.seed,
        training_horizon=args.training_horizon,
        selection_horizon=args.training_horizon,
        context_family_ablation=args.context_family_ablation,
        fit_window=fit_window,
        selection_window=selection_window,
        allow_date_replacement=allow_date_replacement,
        parameter_count=parameter_count,
        training_sample_count=train_rows.height,
        maximum_epochs=MAX_EPOCHS,
        early_stop_patience=EARLY_STOP_PATIENCE,
        runtime=GH200_RUNTIME,
    )
    recorded_training = run_provenance["training"]
    if not isinstance(recorded_training, dict) or (
        recorded_training["steps_per_epoch"],
        recorded_training["warmup_steps"],
    ) != (steps_per_epoch, warmup_steps):
        raise RuntimeError("Scheduler and recorded training contract differ")
    compiled_model = compile_model(model)
    compiled_objective = compile_training_objective(args.objective, args.temperature)
    manifest = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_commit": run_provenance["repository_commit"],
        "run_provenance": run_provenance,
        "feature_store": run_provenance["feature_store"],
        "feature_store_identity": store_identity,
        "split": {
            "training": fit_name,
            "selection": selection_name,
            "fit_window": fit_window,
            "selection_window": selection_window,
            "test_accessed": False,
        },
        "seed": args.seed,
        "global_context": args.global_context,
        "training_horizon": args.training_horizon,
        "selection_horizon": args.training_horizon,
        "context_family_ablation": args.context_family_ablation,
        "model": run_provenance["model"],
        "parameter_count": parameter_count,
        "objective": objective,
        "optimizer": args.optimizer,
        "sam": sam,
        "training": recorded_training,
    }
    _atomic_json(run_dir / "run_manifest.json", manifest)
    history: list[dict[str, object]] = []
    best_score = -float("inf")
    best_epoch = 0
    best_observations: EvaluationObservations | None = None
    best_objective_loss = float("nan")
    stale_epochs = 0
    run_started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_started = time.perf_counter()
        sampler.set_epoch(epoch)
        training_started = time.perf_counter()
        training = train_one_epoch(
            compiled_model,
            train_loader,
            optimizer,
            scheduler,
            GH200_RUNTIME,
            args.optimizer,
            args.objective,
            args.temperature,
            args.sam_rho,
            compiled_objective,
            args.training_horizon,
        )
        training_seconds = time.perf_counter() - training_started
        collection_started = time.perf_counter()
        observations, validation_objective_loss = collect_validation_observations(
            model,
            validation_loader,
            args.objective,
            args.temperature,
            args.training_horizon,
        )
        validation_collection_seconds = time.perf_counter() - collection_started
        metric_started = time.perf_counter()
        score = (
            validation_primary_metric(observations)
            if args.training_horizon == "all"
            else validation_primary_metric(observations, args.training_horizon)
        )
        validation_primary_metric_seconds = time.perf_counter() - metric_started
        row = {
            "epoch": epoch,
            "train_objective_loss": training["objective_loss"],
            "validation_objective_loss": validation_objective_loss,
            "validation_primary_ic": score,
            "optimizer_steps": training["optimizer_steps"],
            "training_seconds": training_seconds,
            "validation_collection_seconds": validation_collection_seconds,
            "validation_primary_metric_seconds": validation_primary_metric_seconds,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        _write_history(run_dir / "history.csv", history)
        if not np.isfinite(score):
            raise FloatingPointError("Validation primary IC is non-finite")
        if score > best_score + MIN_IC_IMPROVEMENT:
            best_score, best_epoch, stale_epochs = score, epoch, 0
            best_observations = observations
            best_objective_loss = validation_objective_loss
            _atomic_torch_save(
                run_dir / "best_checkpoint.pt",
                checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    args.model,
                    architecture,
                    settings,
                    args.optimizer,
                    args.objective,
                    args.temperature,
                    args.sam_rho,
                    args.seed,
                    epoch,
                    score,
                    store,
                    args.global_context,
                    args.peer_features,
                    args.training_horizon,
                    args.context_family_ablation,
                    run_provenance=run_provenance,
                ),
            )
        else:
            stale_epochs += 1
            if stale_epochs >= EARLY_STOP_PATIENCE:
                break
    if best_observations is None:
        raise RuntimeError("Training completed without a best validation epoch")
    reporting_started = time.perf_counter()
    validation, daily = summarize_evaluation_observations(
        best_observations,
        args.objective,
        args.temperature,
        best_objective_loss,
    )
    _atomic_json(run_dir / "validation_metrics.json", validation)
    pl.DataFrame(daily).write_parquet(run_dir / "validation_daily_metrics.parquet")
    final_validation_reporting_seconds = time.perf_counter() - reporting_started
    completed = {
        **manifest,
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "epochs_completed": len(history),
        "total_run_seconds": time.perf_counter() - run_started,
        "final_validation_reporting_seconds": final_validation_reporting_seconds,
    }
    _atomic_json(run_dir / "run_manifest.json", completed)


def _run(args: argparse.Namespace) -> Path:
    store = resolve_feature_store()
    sample_index = load_sample_index(store)
    train_rows = select_sample_split(sample_index, "train")
    validation_rows = select_sample_split(sample_index, "validation")
    set_seeds(args.seed)
    run_dir = args.output_base / _run_directory_name(args, datetime.now(timezone.utc))
    run_dir.mkdir(parents=True, exist_ok=False)
    _run_neural(args, store, train_rows, validation_rows, run_dir)
    return run_dir


def main() -> None:
    print(_run(parse_args()))


if __name__ == "__main__":
    main()
