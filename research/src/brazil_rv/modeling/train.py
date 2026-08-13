from __future__ import annotations

import argparse
import csv
import json
import os
import random
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
    BASELINE_TCN_SETTINGS,
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
    SUPPORTED_MODELS,
    TCN_BLOCK_VARIANTS,
    TCN_FUSIONS,
    TCN_RECEPTIVE_FIELDS,
    TCN_WIDTHS,
    NeuralArchitecture,
    TCNArchitecture,
    TCNSettings,
    architecture_for_model,
    context_routing_metadata,
    model_consumes_context,
    peer_feature_metadata,
)
from .data import (
    create_training_loaders,
    load_sample_index,
    resolve_feature_store,
    select_sample_split,
)
from .engine import (
    checkpoint_payload,
    compile_model,
    evaluate_model,
    experiment_decimal,
    objective_metadata,
    sam_metadata,
    train_one_epoch,
)
from .model import build_neural_model, count_trainable_parameters
from .optim import build_optimizer, build_scheduler
from .xgboost_model import train_xgboost_run


def validate_cli_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> argparse.Namespace:
    if args.model == "xgboost":
        if (
            args.optimizer is not None
            or args.objective is not None
            or args.temperature is not None
            or args.sam_rho is not None
        ):
            parser.error(
                "Neural objective and optimizer options are not valid for XGBoost"
            )
        args.peer_features = "none"
    else:
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
    return args


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one current full-universe model"
    )
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="tcn")
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
    )


def _model_metadata(
    model_name: str,
    architecture: NeuralArchitecture,
    settings: TCNSettings | None,
    peer_features: str = "none",
) -> dict[str, object]:
    return {
        "model_name": model_name,
        "architecture": asdict(architecture),
        "tcn_settings": None if settings is None else asdict(settings),
        "context_routing": context_routing_metadata(architecture)
        if isinstance(architecture, TCNArchitecture)
        else None,
        "peer_features": peer_feature_metadata(model_name, architecture, peer_features),
    }


def _run_directory_name(args: argparse.Namespace, created_at: datetime) -> str:
    if args.model == "xgboost":
        model = "xgboost"
    elif args.model == "tcn":
        model = f"tcn_{args.tcn_fusion}_w{args.tcn_width}_rf{args.tcn_receptive_field}_b{args.tcn_block}"
        if (
            args.slow_routing != "late_only"
            or args.macro_temporal_routing != "late_only"
        ):
            model += f"_slow-{args.slow_routing}_macro-{args.macro_temporal_routing}"
    else:
        model = args.model
    if args.model in NEURAL_MODELS:
        model += f"_{args.objective}_{args.optimizer}"
        if args.sam_rho is not None:
            model += f"_rho{experiment_decimal(args.sam_rho, 3)}"
        if args.temperature is not None:
            model += f"_tau{experiment_decimal(args.temperature, 2)}"
    if args.global_context is not None:
        model += f"_global-{args.global_context}"
    if args.peer_features != "none":
        model += f"_peer-{args.peer_features}"
    return f"{model}_seed{args.seed}_{created_at:%Y%m%dT%H%M%S%fZ}"


def set_seeds(seed: int, *, neural: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if neural:
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
) -> None:
    settings = _tcn_settings_from_args(args)
    architecture = architecture_for_model(args.model, settings)
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
    )
    train_loader, validation_loader, sampler = loaders
    model = build_neural_model(
        args.model,
        architecture if isinstance(architecture, TCNArchitecture) else None,
        args.peer_features,
    ).cuda()
    optimizer, _ = build_optimizer(model)
    scheduler, steps_per_epoch, warmup_steps = build_scheduler(
        optimizer, train_rows.height, MAX_EPOCHS
    )
    compiled = compile_model(model)
    manifest = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_store": str(store),
        "split": {
            "training": "train",
            "selection": "validation",
            "test_accessed": False,
        },
        "seed": args.seed,
        "global_context": args.global_context,
        "model": _model_metadata(
            args.model, architecture, settings, args.peer_features
        ),
        "parameter_count": count_trainable_parameters(model),
        "objective": objective_metadata(args.objective, args.temperature),
        "optimizer": args.optimizer,
        "sam": sam_metadata(args.optimizer, args.sam_rho),
        "training": {
            "maximum_epochs": MAX_EPOCHS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "steps_per_epoch": steps_per_epoch,
            "warmup_steps": warmup_steps,
        },
    }
    _atomic_json(run_dir / "run_manifest.json", manifest)
    history: list[dict[str, object]] = []
    best_score = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    run_started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        sampler.set_epoch(epoch)
        training = train_one_epoch(
            compiled,
            train_loader,
            optimizer,
            scheduler,
            GH200_RUNTIME,
            args.optimizer,
            args.objective,
            args.temperature,
            args.sam_rho,
        )
        validation, daily = evaluate_model(
            compiled, validation_loader, args.objective, args.temperature
        )
        row = {
            "epoch": epoch,
            "train_objective_loss": training["objective_loss"],
            "validation_objective_loss": validation["objective_loss"],
            "validation_primary_ic": validation["primary_score"],
            "optimizer_steps": training["optimizer_steps"],
            "epoch_seconds": training["epoch_seconds"],
        }
        history.append(row)
        _write_history(run_dir / "history.csv", history)
        score = float(validation["primary_score"])
        if not np.isfinite(score):
            raise FloatingPointError("Validation primary IC is non-finite")
        if score > best_score + MIN_IC_IMPROVEMENT:
            best_score, best_epoch, stale_epochs = score, epoch, 0
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
                ),
            )
            _atomic_json(run_dir / "validation_metrics.json", validation)
            pl.DataFrame(daily).write_parquet(
                run_dir / "validation_daily_metrics.parquet"
            )
        else:
            stale_epochs += 1
            if stale_epochs >= EARLY_STOP_PATIENCE:
                break
    completed = {
        **manifest,
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "epochs_completed": len(history),
        "total_run_seconds": time.perf_counter() - run_started,
    }
    _atomic_json(run_dir / "run_manifest.json", completed)


def _run(args: argparse.Namespace) -> Path:
    store = resolve_feature_store()
    sample_index = load_sample_index(store)
    train_rows = select_sample_split(sample_index, "train")
    validation_rows = select_sample_split(sample_index, "validation")
    set_seeds(args.seed, neural=args.model in NEURAL_MODELS)
    run_dir = args.output_base / _run_directory_name(args, datetime.now(timezone.utc))
    run_dir.mkdir(parents=True, exist_ok=False)
    if args.model == "xgboost":
        train_xgboost_run(
            store, train_rows, validation_rows, args.global_context, run_dir, args.seed
        )
    else:
        _run_neural(args, store, train_rows, validation_rows, run_dir)
    return run_dir


def main() -> None:
    print(_run(parse_args()))


if __name__ == "__main__":
    main()
