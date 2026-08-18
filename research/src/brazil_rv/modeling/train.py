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
    EARLY_STOP_PATIENCE,
    GH200_RUNTIME,
    MAX_EPOCHS,
    MIN_IC_IMPROVEMENT,
    RECENCY_POLICIES,
    RUN_OUTPUT_BASE,
    VALIDATION_END,
)
from .data import (
    create_training_loaders,
    feature_store_identity,
    load_sample_index,
    prepare_training_rows,
    resolve_feature_store,
    sample_window_metadata,
    select_sample_split,
    target_scale_identity,
)
from .engine import (
    EvaluationObservations,
    checkpoint_payload,
    collect_validation_observations,
    compile_model,
    compile_training_objective,
    objective_metadata,
    sam_metadata,
    summarize_evaluation_observations,
    train_one_epoch,
    validation_primary_metric,
)
from .model import build_model, count_trainable_parameters
from .optim import build_optimizer, build_scheduler
from .provenance import build_run_provenance, repository_commit


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the current PIT-clean TCN")
    parser.add_argument("--seed", type=int, choices=ALLOWED_SEEDS, default=29)
    parser.add_argument("--recency-policy", choices=RECENCY_POLICIES, default="uniform")
    parser.add_argument("--cross-equity-attention", action="store_true")
    parser.add_argument("--target-scale-dir", required=True, type=Path)
    parser.add_argument("--output-base", type=Path, default=RUN_OUTPUT_BASE)
    return parser.parse_args(arguments)


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


def _write_observations(path: Path, observations: EvaluationObservations) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as output:
        np.savez(
            output,
            **{
                name: getattr(observations, name)
                for name in EvaluationObservations.__dataclass_fields__
            },
        )
    os.replace(temporary, path)


def run_training(
    *,
    store: Path,
    target_scale_dir: Path,
    seed: int,
    recency_policy: str,
    cross_equity_attention: bool,
    run_dir: Path,
) -> Path:
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    torch.set_float32_matmul_precision("high")
    set_seeds(seed)
    sample_index = load_sample_index(store, through=VALIDATION_END)
    raw_train_rows = select_sample_split(sample_index, "train")
    validation_rows = select_sample_split(sample_index, "validation")
    train_rows, date_weights, recency = prepare_training_rows(
        raw_train_rows, recency_policy
    )
    store_identity = feature_store_identity(store)
    scale_identity = target_scale_identity(target_scale_dir, store_identity)
    train_loader, validation_loader, sampler = create_training_loaders(
        store,
        target_scale_dir,
        train_rows,
        validation_rows,
        date_weights,
        GH200_RUNTIME,
        seed,
    )
    model = build_model(cross_equity_attention=cross_equity_attention).cuda()
    parameter_count = count_trainable_parameters(model)
    optimizer, _ = build_optimizer(model)
    scheduler, steps_per_epoch, warmup_steps = build_scheduler(
        optimizer, train_rows.height, MAX_EPOCHS
    )
    run_provenance = build_run_provenance(
        repository_commit_value=repository_commit(),
        feature_store=store,
        feature_store_metadata=store_identity,
        target_scale_dir=target_scale_dir,
        target_scale_metadata=scale_identity,
        cross_equity_attention=cross_equity_attention,
        seed=seed,
        recency=recency,
        fit_window=sample_window_metadata(train_rows, "train"),
        selection_window=sample_window_metadata(validation_rows, "validation"),
        parameter_count=parameter_count,
        training_sample_count=train_rows.height,
        date_replacement=sampler.replace_dates,
    )
    recorded_training = run_provenance["training"]
    if (
        recorded_training["steps_per_epoch"],
        recorded_training["warmup_steps"],
    ) != (steps_per_epoch, warmup_steps):
        raise RuntimeError("Scheduler and recorded training contract differ")
    manifest = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_commit": run_provenance["repository_commit"],
        "run_provenance": run_provenance,
        "feature_store": str(store.resolve()),
        "feature_store_identity": store_identity,
        "target_scale": str(target_scale_dir.resolve()),
        "target_scale_identity": scale_identity,
        "split": {
            "training": "train",
            "selection": "validation",
            "fit_window": run_provenance["fit_window"],
            "selection_window": run_provenance["selection_window"],
            "test_accessed": False,
        },
        "seed": seed,
        "recency_policy": recency_policy,
        "cross_equity_attention": cross_equity_attention,
        "model": run_provenance["model"],
        "parameter_count": parameter_count,
        "objective": objective_metadata(),
        "optimizer": "sam_adamw",
        "sam": sam_metadata(),
        "training": recorded_training,
    }
    _atomic_json(run_dir / "run_manifest.json", manifest)
    compiled_model = compile_model(model)
    compiled_objective = compile_training_objective()
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
            compiled_objective,
        )
        training_seconds = time.perf_counter() - training_started
        collection_started = time.perf_counter()
        observations, validation_loss = collect_validation_observations(
            model, validation_loader
        )
        collection_seconds = time.perf_counter() - collection_started
        score = validation_primary_metric(observations)
        row = {
            "epoch": epoch,
            "train_objective_loss": training["objective_loss"],
            "validation_objective_loss": validation_loss,
            "validation_primary_ic": score,
            "optimizer_steps": training["optimizer_steps"],
            "training_seconds": training_seconds,
            "validation_collection_seconds": collection_seconds,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        _write_history(run_dir / "history.csv", history)
        if not np.isfinite(score):
            raise FloatingPointError("Validation primary IC is non-finite")
        if score > best_score + MIN_IC_IMPROVEMENT:
            best_score, best_epoch, stale_epochs = score, epoch, 0
            best_observations = observations
            best_objective_loss = validation_loss
            _atomic_torch_save(
                run_dir / "best_checkpoint.pt",
                checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    cross_equity_attention=cross_equity_attention,
                    recency_policy=recency_policy,
                    seed=seed,
                    epoch=epoch,
                    validation_score=score,
                    feature_store=store,
                    target_scale_dir=target_scale_dir,
                    target_scale_identity=scale_identity,
                    run_provenance=run_provenance,
                ),
            )
        else:
            stale_epochs += 1
            if stale_epochs >= EARLY_STOP_PATIENCE:
                break
    if best_observations is None:
        raise RuntimeError("Training completed without a best validation epoch")
    validation, daily = summarize_evaluation_observations(
        best_observations, best_objective_loss
    )
    _atomic_json(run_dir / "validation_metrics.json", validation)
    pl.DataFrame(daily).write_parquet(run_dir / "validation_daily_metrics.parquet")
    _write_observations(run_dir / "validation_observations.npz", best_observations)
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
    return run_dir


def _run(args: argparse.Namespace) -> Path:
    created_at = datetime.now(timezone.utc)
    attention = "_attention" if args.cross_equity_attention else ""
    name = (
        f"tcn_{args.recency_policy}{attention}_seed{args.seed}_"
        f"{created_at:%Y%m%dT%H%M%S%fZ}"
    )
    return run_training(
        store=resolve_feature_store(),
        target_scale_dir=args.target_scale_dir,
        seed=args.seed,
        recency_policy=args.recency_policy,
        cross_equity_attention=args.cross_equity_attention,
        run_dir=args.output_base / name,
    )


def main() -> None:
    print(_run(parse_args()))


if __name__ == "__main__":
    main()
