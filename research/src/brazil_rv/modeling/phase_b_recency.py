from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .analyze import compare_observation_ensembles
from .contract import (
    ADAMW_LR,
    ALLOWED_SEEDS,
    GH200_RUNTIME,
    VALIDATION_END,
)
from .data import (
    auxiliary_target_identity,
    create_training_loaders,
    feature_store_identity,
    int64_identity_sha256,
    load_sample_index,
    resolve_feature_store,
    sample_window_metadata,
    select_training_window,
)
from .engine import (
    EvaluationObservations,
    assert_observations_aligned,
    collect_validation_observations,
    compile_model,
    compile_training_objective,
    objective_metadata,
    sam_metadata,
    state_dict_to_cpu,
    train_one_epoch,
    validation_primary_metric,
)
from .metrics import rank_average_predictions
from .model import build_auxiliary_model, build_model, count_trainable_parameters
from .optim import build_optimizer
from .phase_b import (
    DISCOVERY_FOLDS,
    _candidate_run,
    _date_parities,
    _parent_run,
    crossfit_patience_observations,
)
from .provenance import repository_commit
from .train import set_seeds
from .trajectory import load_checkpoint

RECENT_DATE_COUNT = 120
FINE_TUNE_EPOCHS = 3
FINE_TUNE_LR = ADAMW_LR / 10.0
DIRECTIONS = (("odd", "even"), ("even", "odd"))


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.savez(output, **values)
    os.replace(temporary, path)


def _source_run(
    source: str,
    phase_b_campaign: Path,
    parent_campaign: Path,
    fold: str,
    seed: int,
) -> Path:
    if source == "parent":
        return _parent_run(parent_campaign, fold, seed)
    return _candidate_run(phase_b_campaign, source, fold, seed)


def _recency_run(
    output_dir: Path,
    fold: str,
    seed: int,
    selection_parity: str,
) -> Path:
    return output_dir / "recency" / fold / f"seed_{seed}" / f"select_{selection_parity}"


def _selected_epoch(run_dir: Path, selection_parity: str) -> int:
    _, directions = crossfit_patience_observations(run_dir)
    matches = [row for row in directions if row["selection_parity"] == selection_parity]
    if len(matches) != 1:
        raise ValueError("Source trajectory has no unique parity-selected checkpoint")
    return int(matches[0]["selected_epoch"])


def _run_direction(
    *,
    store: Path,
    sidecar_dir: Path,
    source_run: Path,
    source_variant: str,
    fold: str,
    seed: int,
    selection_parity: str,
    evaluation_parity: str,
    sidecar_identity: Mapping[str, object],
    output_dir: Path,
) -> Path:
    run_dir = _recency_run(output_dir, fold, seed, selection_parity)
    commit = repository_commit()
    selected_epoch = _selected_epoch(source_run, selection_parity)
    source_checkpoint = source_run / "checkpoints" / f"epoch_{selected_epoch:02d}.pt"
    expected = {
        "repository_commit": commit,
        "source_run": str(source_run.resolve()),
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_selected_epoch": selected_epoch,
        "source_variant": source_variant,
        "fold": fold,
        "seed": seed,
        "selection_parity": selection_parity,
        "evaluation_parity": evaluation_parity,
    }
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "completed" and all(
            manifest.get(key) == value for key, value in expected.items()
        ):
            return run_dir
        raise ValueError(f"Existing recency run differs: {run_dir}")

    run_dir.mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "validation_predictions").mkdir()
    torch.set_float32_matmul_precision("high")
    set_seeds(seed)
    sample_index = load_sample_index(store, through=VALIDATION_END)
    fit_rows, validation_rows, screening_note = select_training_window(
        sample_index, fold
    )
    fit_dates = (
        fit_rows.select("date_idx").unique().sort("date_idx").tail(RECENT_DATE_COUNT)
    )
    recent_rows = fit_rows.join(fit_dates, on="date_idx", how="semi")
    if recent_rows.get_column("date_idx").n_unique() != RECENT_DATE_COUNT:
        raise ValueError("Recency fine-tune requires exactly 120 fit dates")
    auxiliary_variant = None if source_variant == "parent" else source_variant
    auxiliary_dir = None if auxiliary_variant is None else sidecar_dir
    train_loader, validation_loader, sampler = create_training_loaders(
        store,
        recent_rows,
        validation_rows,
        GH200_RUNTIME,
        seed,
        auxiliary_target_dir=auxiliary_dir,
    )
    model = (
        build_model()
        if auxiliary_variant is None
        else build_auxiliary_model(auxiliary_variant)
    ).cuda()
    checkpoint = load_checkpoint(source_run, selected_epoch)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer, _ = build_optimizer(model)
    for group in optimizer.param_groups:
        group["lr"] = FINE_TUNE_LR
    compiled_model = compile_model(model)
    compiled_objective = compile_training_objective(auxiliary_variant=auxiliary_variant)
    store_identity = feature_store_identity(store)
    run_sidecar_identity = None if auxiliary_variant is None else dict(sidecar_identity)
    parities = _date_parities(validation_rows.get_column("date_idx").to_numpy())
    manifest = {
        "schema": "PHASE_B_RECENCY_FINE_TUNE",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        **expected,
        "feature_store": str(store.resolve()),
        "feature_store_identity": store_identity,
        "auxiliary_target_identity": run_sidecar_identity,
        "objective": objective_metadata(auxiliary_variant),
        "model": checkpoint["model"],
        "parameter_count": count_trainable_parameters(model),
        "optimizer": "fresh_sam_adamw",
        "sam": sam_metadata(),
        "learning_rate": FINE_TUNE_LR,
        "scheduler": None,
        "recent_window": sample_window_metadata(
            recent_rows, f"{fold}_most_recent_120_fit_dates"
        ),
        "selection_window": sample_window_metadata(
            validation_rows, f"{fold}_selection"
        ),
        "screening_note": screening_note,
        "selection_date_identity_sha256": int64_identity_sha256(
            parities[selection_parity]
        ),
        "evaluation_date_identity_sha256": int64_identity_sha256(
            parities[evaluation_parity]
        ),
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(manifest_path, manifest)
    history = []
    reference: EvaluationObservations | None = None
    started = time.perf_counter()
    try:
        for epoch in range(1, FINE_TUNE_EPOCHS + 1):
            sampler.set_epoch(epoch)
            epoch_started = time.perf_counter()
            training = train_one_epoch(
                compiled_model,
                train_loader,
                optimizer,
                None,
                GH200_RUNTIME,
                compiled_objective,
                auxiliary_variant=auxiliary_variant,
            )
            observations, main_loss = collect_validation_observations(
                model, validation_loader
            )
            if reference is None:
                reference = observations
            else:
                assert_observations_aligned(reference, observations)
            score = validation_primary_metric(observations)
            _atomic_npz(
                run_dir / "validation_predictions" / f"epoch_{epoch:02d}.npz",
                raw=observations.predictions,
            )
            _atomic_torch_save(
                run_dir / "checkpoints" / f"epoch_{epoch:02d}.pt",
                {
                    "schema": "PHASE_B_RECENCY_CHECKPOINT",
                    **expected,
                    "fine_tune_epoch": epoch,
                    "feature_store_identity": store_identity,
                    "auxiliary_target_identity": run_sidecar_identity,
                    "objective": objective_metadata(auxiliary_variant),
                    "model": checkpoint["model"],
                    "model_state_dict": state_dict_to_cpu(model.state_dict()),
                    "validation_main_objective_loss": main_loss,
                    "validation_primary_ic": score,
                },
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_objective_loss": training["objective_loss"],
                    "validation_main_objective_loss": main_loss,
                    "validation_primary_ic": score,
                    "optimizer_steps": training["optimizer_steps"],
                    "epoch_seconds": time.perf_counter() - epoch_started,
                }
            )
            _atomic_json(run_dir / "history.json", {"epochs": history})
        _atomic_json(
            manifest_path,
            {
                **manifest,
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "epochs_completed": FINE_TUNE_EPOCHS,
                "total_run_seconds": time.perf_counter() - started,
            },
        )
    except BaseException:
        _atomic_json(
            manifest_path,
            {
                **manifest,
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "epochs_completed": len(history),
                "total_run_seconds": time.perf_counter() - started,
            },
        )
        raise
    return run_dir


def _crossfit_fine_tune_observations(
    source_run: Path,
    recency_root: Path,
    *,
    fold: str,
    seed: int,
    epoch: int,
) -> tuple[EvaluationObservations, EvaluationObservations]:
    full, _ = crossfit_patience_observations(source_run)
    fine_predictions = np.empty_like(full.predictions)
    parities = _date_parities(full.date_idx)
    for selection_parity, evaluation_parity in DIRECTIONS:
        run_dir = _recency_run(recency_root, fold, seed, selection_parity)
        with np.load(
            run_dir / "validation_predictions" / f"epoch_{epoch:02d}.npz",
            allow_pickle=False,
        ) as values:
            predictions = values["raw"]
        evaluation_mask = np.isin(full.date_idx, parities[evaluation_parity])
        fine_predictions[evaluation_mask] = predictions[evaluation_mask]
    fine = replace(full, predictions=fine_predictions)
    blended = replace(
        full,
        predictions=rank_average_predictions(
            [full.predictions, fine.predictions], full.label_mask
        ),
    )
    return fine, blended


def _analyze_recency(
    *,
    source: str,
    phase_b_campaign: Path,
    parent_campaign: Path,
    output_dir: Path,
) -> Path:
    summary_path = output_dir / "recency_analysis.json"
    epoch_summaries = {}
    for epoch in range(1, FINE_TUNE_EPOCHS + 1):
        folds = {}
        for fold in DISCOVERY_FOLDS:
            full_members = {}
            fine_members = {}
            blended_members = {}
            for seed in ALLOWED_SEEDS:
                run_dir = _source_run(
                    source, phase_b_campaign, parent_campaign, fold, seed
                )
                full, _ = crossfit_patience_observations(run_dir)
                fine, blended = _crossfit_fine_tune_observations(
                    run_dir,
                    output_dir,
                    fold=fold,
                    seed=seed,
                    epoch=epoch,
                )
                full_members[f"seed_{seed}"] = full
                fine_members[f"seed_{seed}"] = fine
                blended_members[f"seed_{seed}"] = blended
            base = output_dir / "analysis" / f"epoch_{epoch}" / fold
            metadata = {
                "source": source,
                "fine_tune_epoch": epoch,
                "fold": fold,
                "recent_date_count": RECENT_DATE_COUNT,
                "learning_rate": FINE_TUNE_LR,
                "adaptive_checkpoint_crossfit": True,
                "ensemble_weights_learned": False,
                "official_validation_accessed": False,
                "test_accessed": False,
            }
            compare_observation_ensembles(
                fine_members,
                full_members,
                candidate_rule=f"recency_epoch_{epoch}",
                parent_rule="full_history_patience3_raw",
                output_dir=base / "fine_only",
                comparison_metadata={**metadata, "composition": "fine_tuned_only"},
            )
            compare_observation_ensembles(
                blended_members,
                full_members,
                candidate_rule=f"full_plus_recency_epoch_{epoch}_equal_rank",
                parent_rule="full_history_patience3_raw",
                output_dir=base / "full_plus_fine",
                comparison_metadata={
                    **metadata,
                    "composition": "50_50_uniform_rank_full_plus_fine",
                },
            )
            folds[fold] = {
                "fine_only": json.loads(
                    (base / "fine_only" / "analysis.json").read_text(encoding="utf-8")
                ),
                "full_plus_fine": json.loads(
                    (base / "full_plus_fine" / "analysis.json").read_text(
                        encoding="utf-8"
                    )
                ),
            }
        epoch_summaries[str(epoch)] = {
            "folds": folds,
            "fine_only_mean_delta": float(
                np.mean(
                    [
                        folds[fold]["fine_only"]["candidate_minus_parent_primary_ic"]
                        for fold in DISCOVERY_FOLDS
                    ]
                )
            ),
            "full_plus_fine_mean_delta": float(
                np.mean(
                    [
                        folds[fold]["full_plus_fine"][
                            "candidate_minus_parent_primary_ic"
                        ]
                        for fold in DISCOVERY_FOLDS
                    ]
                )
            ),
        }
    candidates = {
        f"fine_epoch_{epoch}": (
            epoch_summaries[str(epoch)]["fine_only_mean_delta"],
            "fine_only",
            epoch,
        )
        for epoch in range(1, FINE_TUNE_EPOCHS + 1)
    }
    candidates.update(
        {
            f"full_plus_fine_epoch_{epoch}": (
                epoch_summaries[str(epoch)]["full_plus_fine_mean_delta"],
                "full_plus_fine",
                epoch,
            )
            for epoch in range(1, FINE_TUNE_EPOCHS + 1)
        }
    )
    winner = min(candidates, key=lambda key: (-candidates[key][0], key))
    delta, composition, epoch = candidates[winner]
    positive_both = all(
        epoch_summaries[str(epoch)]["folds"][fold][composition][
            "candidate_minus_parent_primary_ic"
        ]
        > 0.0
        for fold in DISCOVERY_FOLDS
    )
    selected = winner if delta > 0.0 and positive_both else "full_history"
    summary = {
        "schema": "PHASE_B_RECENCY_ANALYSIS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "epochs": epoch_summaries,
        "selected_rule": selected,
        "best_candidate_before_guardrail": winner,
        "best_candidate_mean_delta": delta,
        "best_candidate_positive_on_both_folds": positive_both,
        "ensemble_weights_learned": False,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(summary_path, summary)
    return summary_path


def run_recency_campaign(
    store: Path,
    auxiliary_target_dir: Path,
    parent_campaign: Path,
    phase_b_campaign: Path,
) -> Path:
    phase_b_manifest = json.loads(
        (phase_b_campaign / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    if (
        phase_b_manifest.get("status") != "completed"
        or phase_b_manifest.get("official_validation_accessed") is not False
        or phase_b_manifest.get("test_accessed") is not False
    ):
        raise ValueError(
            "Phase B campaign must be completed before recency fine-tuning"
        )
    selection = json.loads(
        (phase_b_campaign / "discovery_selection.json").read_text(encoding="utf-8")
    )
    source = str(selection["recency_full_history_source"])
    sidecar_identity = auxiliary_target_identity(
        auxiliary_target_dir, feature_store_identity(store)
    )
    for fold in DISCOVERY_FOLDS:
        for seed in ALLOWED_SEEDS:
            source_run = _source_run(
                source, phase_b_campaign, parent_campaign, fold, seed
            )
            for selection_parity, evaluation_parity in DIRECTIONS:
                _run_direction(
                    store=store,
                    sidecar_dir=auxiliary_target_dir,
                    source_run=source_run,
                    source_variant=source,
                    fold=fold,
                    seed=seed,
                    selection_parity=selection_parity,
                    evaluation_parity=evaluation_parity,
                    sidecar_identity=sidecar_identity,
                    output_dir=phase_b_campaign,
                )
    analysis = _analyze_recency(
        source=source,
        phase_b_campaign=phase_b_campaign,
        parent_campaign=parent_campaign,
        output_dir=phase_b_campaign,
    )
    _atomic_json(
        phase_b_campaign / "recency_manifest.json",
        {
            "schema": "PHASE_B_RECENCY_CAMPAIGN",
            "status": "completed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repository_commit": repository_commit(),
            "source": source,
            "feature_store_identity": feature_store_identity(store),
            "auxiliary_target_identity": sidecar_identity,
            "recent_date_count": RECENT_DATE_COUNT,
            "epochs": FINE_TUNE_EPOCHS,
            "learning_rate": FINE_TUNE_LR,
            "analysis": str(analysis.resolve()),
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    return phase_b_campaign


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-fit the three-epoch Phase B recency trajectory"
    )
    parser.add_argument("--auxiliary-target-dir", required=True, type=Path)
    parser.add_argument("--parent-campaign", required=True, type=Path)
    parser.add_argument("--phase-b-campaign", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        run_recency_campaign(
            resolve_feature_store(),
            args.auxiliary_target_dir,
            args.parent_campaign,
            args.phase_b_campaign,
        )
    )


if __name__ == "__main__":
    main()
