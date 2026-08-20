from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from .analyze import (
    BOOTSTRAP_BLOCK_LENGTHS,
    BOOTSTRAP_REPLICATIONS,
    compare_observation_ensembles,
    load_run_observations,
)
from .contract import ALLOWED_SEEDS, MAX_EPOCHS, TRAIN_END
from .data import resolve_feature_store
from .engine import EvaluationObservations, assert_observations_aligned
from .metrics import (
    per_date_primary_ic,
    primary_validation_score,
    rank_average_predictions,
    sample_level_spearman_ic,
)
from .trajectory import predictions_for_rule, simulate_patience3

DISCOVERY_FOLDS = ("fold_a", "fold_b")
R1_CANDIDATES = (
    "patience_raw_plus_final_ema_0995",
    "patience_raw_plus_selected_epoch_ema_0995",
)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _load_legacy_observations(path: Path) -> EvaluationObservations:
    with np.load(path, allow_pickle=False) as values:
        return EvaluationObservations(
            **{
                name: values[name].copy()
                for name in EvaluationObservations.__dataclass_fields__
            }
        )


def _parent_validation_members(
    parent_reproduction: Path,
) -> dict[str, EvaluationObservations]:
    members = {}
    for path in parent_reproduction.rglob("validation_observations.npz"):
        manifest_path = path.parent / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            continue
        seed = int(manifest["seed"])
        if seed not in ALLOWED_SEEDS:
            continue
        key = f"seed_{seed}"
        if key in members:
            raise ValueError(f"Duplicate parent validation observations for {key}")
        members[key] = _load_legacy_observations(path)
    expected = {f"seed_{seed}" for seed in ALLOWED_SEEDS}
    if set(members) != expected:
        raise ValueError(f"Parent reproduction members differ: {sorted(members)}")
    reference = members["seed_11"]
    for member in members.values():
        assert_observations_aligned(reference, member)
    if np.unique(reference.date_idx).size != 244:
        raise ValueError("D1 requires the 244-date consumed official validation")
    return members


def _block_sample_mean(
    values: np.ndarray, block_length: int, rng: np.random.Generator
) -> float:
    count = values.size
    blocks = int(np.ceil(count / block_length))
    starts = rng.integers(0, count, size=blocks)
    indices = (
        starts[:, None] + np.arange(block_length, dtype=np.int64)[None, :]
    ) % count
    return float(np.mean(values[indices.ravel()[:count]]))


def _two_period_bootstrap(
    early: np.ndarray, late: np.ndarray, block_length: int
) -> dict[str, object]:
    rng = np.random.default_rng(20260820 + block_length)
    draws = np.asarray(
        [
            _block_sample_mean(late, block_length, rng)
            - _block_sample_mean(early, block_length, rng)
            for _ in range(BOOTSTRAP_REPLICATIONS)
        ],
        dtype=np.float64,
    )
    return {
        "observed": float(np.mean(late) - np.mean(early)),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "replications": BOOTSTRAP_REPLICATIONS,
        "block_length": block_length,
    }


def run_staleness_diagnostic(
    store: Path, parent_reproduction: Path, output_dir: Path
) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    members = _parent_validation_members(parent_reproduction)
    reference = members["seed_11"]
    predictions = rank_average_predictions(
        [members[f"seed_{seed}"].predictions for seed in ALLOWED_SEEDS],
        reference.label_mask,
    )
    sample_ic = sample_level_spearman_ic(
        predictions, reference.targets, reference.label_mask
    )
    date_idx, daily_ic = per_date_primary_ic(sample_ic, reference.date_idx)
    date_frame = (
        pl.read_parquet(store / "date_index.parquet")
        .select("date_idx", "trade_date")
        .filter(pl.col("date_idx").is_in(date_idx))
        .sort("date_idx")
    )
    if not np.array_equal(date_frame["date_idx"].to_numpy(), date_idx):
        raise ValueError("D1 date identities do not match the feature store")
    trade_dates = tuple(date_frame["trade_date"].to_list())
    quarters: dict[str, list[float]] = {}
    for value, ic in zip(trade_dates, daily_ic, strict=True):
        key = f"{value.year}-Q{(value.month - 1) // 3 + 1}"
        quarters.setdefault(key, []).append(float(ic))
    h2_2024 = np.asarray(
        [
            ic
            for value, ic in zip(trade_dates, daily_ic, strict=True)
            if date(2024, 7, 1) <= value <= date(2024, 12, 31)
        ],
        dtype=np.float64,
    )
    h1_2025 = np.asarray(
        [
            ic
            for value, ic in zip(trade_dates, daily_ic, strict=True)
            if date(2025, 1, 1) <= value <= date(2025, 6, 30)
        ],
        dtype=np.float64,
    )
    if not h2_2024.size or not h1_2025.size:
        raise ValueError("D1 staleness periods are incomplete")
    days_since_train = np.asarray(
        [(value - TRAIN_END).days for value in trade_dates], dtype=np.float64
    )
    slope, intercept = np.polyfit(days_since_train, daily_ic, 1)
    bootstrap = {
        str(block): _two_period_bootstrap(h2_2024, h1_2025, block)
        for block in BOOTSTRAP_BLOCK_LENGTHS
    }
    daily = pl.DataFrame(
        {
            "date_idx": date_idx,
            "trade_date": trade_dates,
            "ensemble_daily_ic": daily_ic,
            "days_since_training_end": days_since_train.astype(np.int64),
        }
    )
    daily.write_parquet(output_dir / "daily_ic.parquet")
    report = {
        "schema": "PARENT_VALIDATION_STALENESS_DIAGNOSTIC",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_parent_reproduction": str(parent_reproduction.resolve()),
        "date_count": len(trade_dates),
        "quarterly_mean_ic": {
            key: float(np.mean(values)) for key, values in quarters.items()
        },
        "h1_2025_minus_h2_2024": {
            "h2_2024_mean": float(np.mean(h2_2024)),
            "h1_2025_mean": float(np.mean(h1_2025)),
            "moving_block_bootstrap": bootstrap,
        },
        "linear_staleness_slope": {
            "ic_per_day": float(slope),
            "ic_per_100_days": float(100.0 * slope),
            "intercept": float(intercept),
            "x": "calendar days since 2024-06-28 training end",
        },
        "interpretation_contract": (
            "diagnostic for retraining cadence only; consumed official validation "
            "is not used to select Phase C architecture candidates"
        ),
        "official_validation_accessed": True,
        "test_accessed": False,
    }
    _atomic_json(output_dir / "staleness_report.json", report)
    return output_dir


def _date_parities(date_idx: np.ndarray) -> dict[str, np.ndarray]:
    dates = np.unique(date_idx)
    if dates.size != 102:
        raise ValueError(f"R1 expected 102 selection dates, found {dates.size}")
    return {"odd": dates[0::2], "even": dates[1::2]}


def crossfit_patience_observations(
    run_dir: Path,
    *,
    blend: str | None = None,
) -> tuple[EvaluationObservations, list[dict[str, object]]]:
    reference = load_run_observations(run_dir, "final_raw")
    raw_epochs = []
    ema_epochs = []
    for epoch in range(1, MAX_EPOCHS + 1):
        with np.load(
            run_dir / "validation_predictions" / f"epoch_{epoch:02d}.npz",
            allow_pickle=False,
        ) as values:
            raw_epochs.append(values["raw"].copy())
            ema_epochs.append(values["ema_0995"].copy())
    final_ema = predictions_for_rule(run_dir, "final_ema_0995")
    predictions = np.empty_like(reference.predictions)
    directions = []
    parities = _date_parities(reference.date_idx)
    for selection_parity, evaluation_parity in (("odd", "even"), ("even", "odd")):
        selection_mask = np.isin(reference.date_idx, parities[selection_parity])
        evaluation_mask = np.isin(reference.date_idx, parities[evaluation_parity])
        scores = []
        for values in raw_epochs:
            scores.append(
                primary_validation_score(
                    values[selection_mask],
                    reference.targets[selection_mask],
                    reference.label_mask[selection_mask],
                    reference.date_idx[selection_mask],
                )
            )
        replay = simulate_patience3(scores)
        selected_epoch = int(replay["selected_epoch"])
        selected = raw_epochs[selected_epoch - 1]
        if blend == "final":
            selected = rank_average_predictions(
                [selected, final_ema], reference.label_mask
            )
        elif blend == "selected_epoch":
            selected = rank_average_predictions(
                [selected, ema_epochs[selected_epoch - 1]],
                reference.label_mask,
            )
        elif blend is not None:
            raise ValueError(f"Unknown R1 blend: {blend}")
        predictions[evaluation_mask] = selected[evaluation_mask]
        directions.append(
            {
                "selection_parity": selection_parity,
                "evaluation_parity": evaluation_parity,
                "selected_epoch": selected_epoch,
                "stopped_epoch": int(replay["stopped_epoch"]),
                "selection_half_ic": float(replay["selected_score"]),
            }
        )
    return replace(reference, predictions=predictions), directions


def _parent_run(parent_campaign: Path, fold: str, seed: int) -> Path:
    return parent_campaign / fold / f"seed_{seed}"


def run_r1_analysis(parent_campaign: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    candidate_blends = {
        "patience_raw_plus_final_ema_0995": "final",
        "patience_raw_plus_selected_epoch_ema_0995": "selected_epoch",
    }
    summaries = {}
    for candidate_name, blend in candidate_blends.items():
        folds = {}
        for fold in DISCOVERY_FOLDS:
            parent_members = {}
            candidate_members = {}
            candidate_replays = {}
            parent_replays = {}
            for seed in ALLOWED_SEEDS:
                key = f"seed_{seed}"
                run = _parent_run(parent_campaign, fold, seed)
                parent, parent_replay = crossfit_patience_observations(run)
                candidate, candidate_replay = crossfit_patience_observations(
                    run, blend=blend
                )
                parent_members[key] = parent
                candidate_members[key] = candidate
                parent_replays[key] = parent_replay
                candidate_replays[key] = candidate_replay
            comparison = output_dir / candidate_name / fold
            compare_observation_ensembles(
                candidate_members,
                parent_members,
                candidate_rule=candidate_name,
                parent_rule="patience3_raw",
                output_dir=comparison,
                comparison_metadata={
                    "zero_training": True,
                    "same_trajectory_rank_blend": True,
                    "candidate_replays": candidate_replays,
                    "parent_replays": parent_replays,
                    "official_validation_accessed": False,
                    "test_accessed": False,
                },
            )
            report = json.loads(
                (comparison / "analysis.json").read_text(encoding="utf-8")
            )
            folds[fold] = {
                "candidate_ensemble_ic": report["candidate"]["ensemble_ic"],
                "parent_ensemble_ic": report["parent"]["ensemble_ic"],
                "candidate_minus_parent_primary_ic": report[
                    "candidate_minus_parent_primary_ic"
                ],
                "per_date_delta_bootstrap": report["per_date_delta_bootstrap"],
                "analysis": str((comparison / "analysis.json").resolve()),
            }
        retained = all(
            folds[fold]["candidate_minus_parent_primary_ic"] > 0.0
            for fold in DISCOVERY_FOLDS
        )
        summaries[candidate_name] = {
            "folds": folds,
            "mean_fold_candidate_minus_parent_ic": float(
                np.mean(
                    [
                        folds[fold]["candidate_minus_parent_primary_ic"]
                        for fold in DISCOVERY_FOLDS
                    ]
                )
            ),
            "retained_positive_both_folds": retained,
        }
    eligible = [
        name
        for name, summary in summaries.items()
        if summary["retained_positive_both_folds"]
    ]
    selected = (
        max(
            eligible,
            key=lambda name: summaries[name]["mean_fold_candidate_minus_parent_ic"],
        )
        if eligible
        else None
    )
    _atomic_json(
        output_dir / "r1_summary.json",
        {
            "schema": "R1_SAME_TRAJECTORY_RANK_BLEND",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_parent_campaign": str(parent_campaign.resolve()),
            "candidates": summaries,
            "selected_candidate": selected,
            "retention_rule": "strictly positive paired IC delta on both folds",
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    return output_dir


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run next-stage zero-training diagnostics"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    d1 = subparsers.add_parser("d1")
    d1.add_argument("--parent-reproduction", required=True, type=Path)
    d1.add_argument("--output-dir", required=True, type=Path)
    r1 = subparsers.add_parser("r1")
    r1.add_argument("--parent-campaign", required=True, type=Path)
    r1.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(arguments)


def main() -> None:
    args = parse_args()
    if args.command == "d1":
        print(
            run_staleness_diagnostic(
                resolve_feature_store(),
                args.parent_reproduction,
                args.output_dir,
            )
        )
    else:
        print(run_r1_analysis(args.parent_campaign, args.output_dir))


if __name__ == "__main__":
    main()
