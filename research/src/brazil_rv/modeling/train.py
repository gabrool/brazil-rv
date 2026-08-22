from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch

from .contract import (
    ALLOWED_SEEDS,
    EMA_DECAYS,
    GH200_RUNTIME,
    MAX_EPOCHS,
    RUN_OUTPUT_BASE,
    VALIDATION_END,
)
from .data import (
    create_training_loaders,
    feature_store_identity,
    load_external_sidecar,
    load_sample_index,
    resolve_feature_store,
    sample_window_metadata,
    select_training_window,
)
from .engine import (
    EvaluationObservations,
    assert_observations_aligned,
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
from .trajectory import (
    EMA_KEYS,
    ModelEMA,
    average_state_dicts,
    load_checkpoint,
    load_frozen_selection,
    predictions_for_rule,
    retrospective_best_epoch,
    simulate_patience3,
    temporarily_load_state,
)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one fixed PIT-clean trajectory")
    parser.add_argument("--seed", type=int, choices=ALLOWED_SEEDS, default=29)
    parser.add_argument(
        "--selection-window",
        choices=("fold_c", "fold_a", "fold_b", "official"),
        default="fold_a",
    )
    parser.add_argument("--selection-rule-file", type=Path)
    parser.add_argument("--sidecar-dir", type=Path)
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


def _atomic_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.savez(output, **values)
    os.replace(temporary, path)


def _write_history(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_reference(path: Path, observations: EvaluationObservations) -> None:
    _atomic_npz(
        path,
        {
            name: getattr(observations, name)
            for name in EvaluationObservations.__dataclass_fields__
            if name != "predictions"
        },
    )


def _selection_metadata(path: Path | None, window: str) -> dict[str, object] | None:
    if path is None:
        return None
    if window != "official":
        raise ValueError(
            "A frozen selection rule may be applied only to an official run"
        )
    selection = load_frozen_selection(path)
    return {
        "selected_rule": selection["selected_rule"],
        "selection_file": str(path.resolve()),
        "selection_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "official_validation_reselection_allowed": False,
    }


def _collect_variants(
    model: torch.nn.Module,
    validation_loader,
    emas: Sequence[ModelEMA],
) -> tuple[
    EvaluationObservations,
    dict[str, np.ndarray],
    dict[str, float],
    dict[str, float],
]:
    raw, raw_loss = collect_validation_observations(model, validation_loader)
    predictions = {"raw": raw.predictions}
    losses = {"raw": raw_loss}
    scores = {"raw": validation_primary_metric(raw)}
    for ema in emas:
        with temporarily_load_state(model, ema.shadow):
            observations, loss = collect_validation_observations(
                model, validation_loader
            )
        assert_observations_aligned(raw, observations)
        predictions[ema.key] = observations.predictions
        losses[ema.key] = loss
        scores[ema.key] = validation_primary_metric(observations)
    return raw, predictions, losses, scores


def _evaluate_weight_average(
    model: torch.nn.Module,
    validation_loader,
    reference: EvaluationObservations,
    run_dir: Path,
    length: int,
) -> tuple[np.ndarray, float, float]:
    states = [
        load_checkpoint(run_dir, epoch)["model_state_dict"]
        for epoch in range(MAX_EPOCHS - length + 1, MAX_EPOCHS + 1)
    ]
    with temporarily_load_state(model, average_state_dicts(states)):
        observations, loss = collect_validation_observations(model, validation_loader)
    assert_observations_aligned(reference, observations)
    return observations.predictions, loss, validation_primary_metric(observations)


def run_training(
    *,
    store: Path,
    seed: int,
    selection_window: str,
    run_dir: Path,
    selection_rule_file: Path | None = None,
    sidecar_dir: Path | None = None,
    zero_dynamic_channels: tuple[int, ...] = (),
    zero_slow_fields: tuple[int, ...] = (),
) -> Path:
    sidecar = None if sidecar_dir is None else load_external_sidecar(sidecar_dir, store)
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "validation_predictions").mkdir()
    torch.set_float32_matmul_precision("high")
    set_seeds(seed)
    sample_index = load_sample_index(store, through=VALIDATION_END)
    train_rows, validation_rows, selection_note = select_training_window(
        sample_index, selection_window
    )
    frozen_selection = _selection_metadata(selection_rule_file, selection_window)
    store_identity = feature_store_identity(store)
    train_loader, validation_loader, sampler = create_training_loaders(
        store,
        train_rows,
        validation_rows,
        GH200_RUNTIME,
        seed,
        sidecar,
        zero_dynamic_channels,
        zero_slow_fields,
    )
    model = build_model(None if sidecar is None else sidecar.feature_count).cuda()
    emas = tuple(ModelEMA(model, decay) for decay in EMA_DECAYS)
    parameter_count = count_trainable_parameters(model)
    optimizer, _ = build_optimizer(model)
    scheduler, steps_per_epoch, warmup_steps = build_scheduler(
        optimizer, train_rows.height, MAX_EPOCHS
    )
    run_provenance = build_run_provenance(
        repository_commit_value=repository_commit(),
        feature_store=store,
        feature_store_metadata=store_identity,
        seed=seed,
        fit_window=sample_window_metadata(train_rows, f"{selection_window}_fit"),
        selection_window=sample_window_metadata(
            validation_rows, f"{selection_window}_selection"
        ),
        selection_note=selection_note,
        parameter_count=parameter_count,
        training_sample_count=train_rows.height,
        date_replacement=sampler.replace_dates,
        external_sidecar=None if sidecar is None else sidecar.identity,
        base_feature_ablation={
            "scope": "equity_inputs_only",
            "dynamic_channel_indices": list(zero_dynamic_channels),
            "slow_field_indices": list(zero_slow_fields),
        },
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
        "external_sidecar": None if sidecar is None else sidecar.identity,
        "base_feature_ablation": run_provenance["base_feature_ablation"],
        "split": {
            "training": selection_window,
            "selection": selection_window,
            "fit_window": run_provenance["fit_window"],
            "selection_window": run_provenance["selection_window"],
            "selection_note": selection_note,
            "test_accessed": False,
        },
        "seed": seed,
        "model": run_provenance["model"],
        "parameter_count": parameter_count,
        "objective": objective_metadata(),
        "optimizer": "sam_adamw",
        "sam": sam_metadata(),
        "training": recorded_training,
        "frozen_selection": frozen_selection,
    }
    _atomic_json(run_dir / "run_manifest.json", manifest)
    compiled_model = compile_model(model)
    compiled_objective = compile_training_objective()
    history: list[dict[str, object]] = []
    raw_scores: list[float] = []
    raw_prediction_tail: list[np.ndarray] = []
    reference: EvaluationObservations | None = None
    run_started = time.perf_counter()

    def update_emas() -> None:
        for ema in emas:
            ema.update(model)

    try:
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
                after_update=update_emas,
            )
            training_seconds = time.perf_counter() - training_started
            validation_started = time.perf_counter()
            raw, predictions, losses, scores = _collect_variants(
                model, validation_loader, emas
            )
            if reference is None:
                reference = raw
                _write_reference(run_dir / "validation_reference.npz", raw)
            else:
                assert_observations_aligned(reference, raw)
            raw_prediction_tail.append(raw.predictions)
            raw_prediction_tail = raw_prediction_tail[-5:]
            raw_scores.append(scores["raw"])
            _atomic_npz(
                run_dir / "validation_predictions" / f"epoch_{epoch:02d}.npz",
                predictions,
            )
            _atomic_torch_save(
                run_dir / "checkpoints" / f"epoch_{epoch:02d}.pt",
                checkpoint_payload(
                    model,
                    {ema.key: ema.cpu_state_dict() for ema in emas},
                    seed=seed,
                    epoch=epoch,
                    validation_scores=scores,
                    feature_store=store,
                    run_provenance=run_provenance,
                ),
            )
            row = {
                "epoch": epoch,
                "train_objective_loss": training["objective_loss"],
                "raw_validation_objective_loss": losses["raw"],
                "raw_validation_primary_ic": scores["raw"],
                **{
                    f"{key}_validation_objective_loss": losses[key]
                    for key in EMA_KEYS.values()
                },
                **{
                    f"{key}_validation_primary_ic": scores[key]
                    for key in EMA_KEYS.values()
                },
                "optimizer_steps": training["optimizer_steps"],
                "training_seconds": training_seconds,
                "validation_collection_seconds": (
                    time.perf_counter() - validation_started
                ),
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
            history.append(row)
            _write_history(run_dir / "history.csv", history)
            if not all(np.isfinite(value) for value in scores.values()):
                raise FloatingPointError("Validation primary IC is non-finite")

        assert reference is not None
        tail_predictions: dict[str, np.ndarray] = {}
        tail_scores: dict[str, float] = {}
        tail_losses: dict[str, float | None] = {}
        for length in (3, 5):
            key = f"last{length}_weight_average"
            prediction, loss, score = _evaluate_weight_average(
                model,
                validation_loader,
                reference,
                run_dir,
                length,
            )
            tail_predictions[key] = prediction
            tail_losses[key] = loss
            tail_scores[key] = score
            key = f"tail{length}_prediction_average"
            prediction = np.mean(
                np.stack(raw_prediction_tail[-length:]), axis=0, dtype=np.float64
            ).astype(np.float32)
            observations = replace(reference, predictions=prediction)
            tail_predictions[key] = prediction
            tail_losses[key] = None
            tail_scores[key] = validation_primary_metric(observations)
        _atomic_npz(
            run_dir / "validation_predictions" / "tail_candidates.npz",
            tail_predictions,
        )
        best_epoch = retrospective_best_epoch(raw_scores)
        diagnostics = {
            "patience3": simulate_patience3(raw_scores),
            "retrospective_best_epoch": {
                "selected_epoch": best_epoch,
                "selected_score": raw_scores[best_epoch - 1],
                "selection_eligible": False,
            },
            "final_epoch_scores": {
                key: history[-1][f"{key}_validation_primary_ic"]
                for key in ("raw", *EMA_KEYS.values())
            },
            "tail_candidate_scores": tail_scores,
            "tail_candidate_objective_losses": tail_losses,
        }
        _atomic_json(run_dir / "trajectory_diagnostics.json", diagnostics)
        if frozen_selection is not None:
            selected = replace(
                reference,
                predictions=predictions_for_rule(
                    run_dir, str(frozen_selection["selected_rule"])
                ),
            )
            validation, daily = summarize_evaluation_observations(selected, None)
            _atomic_json(run_dir / "validation_metrics.json", validation)
            pl.DataFrame(daily).write_parquet(
                run_dir / "validation_daily_metrics.parquet"
            )
        completed = {
            **manifest,
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "epochs_completed": MAX_EPOCHS,
            "trajectory_diagnostics": diagnostics,
            "total_run_seconds": time.perf_counter() - run_started,
        }
        _atomic_json(run_dir / "run_manifest.json", completed)
    except BaseException:
        failed = {
            **manifest,
            "status": "failed",
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "epochs_completed": len(history),
            "total_run_seconds": time.perf_counter() - run_started,
        }
        _atomic_json(run_dir / "run_manifest.json", failed)
        raise
    return run_dir


def _run(args: argparse.Namespace) -> Path:
    created_at = datetime.now(timezone.utc)
    name = f"tcn_{args.selection_window}_seed{args.seed}_{created_at:%Y%m%dT%H%M%S%fZ}"
    return run_training(
        store=resolve_feature_store(),
        seed=args.seed,
        selection_window=args.selection_window,
        selection_rule_file=args.selection_rule_file,
        sidecar_dir=args.sidecar_dir,
        run_dir=args.output_base / name,
    )


def main() -> None:
    print(_run(parse_args()))


if __name__ == "__main__":
    main()
