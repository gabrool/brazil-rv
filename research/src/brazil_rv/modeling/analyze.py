from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl

from .contract import HORIZONS
from .engine import EvaluationObservations, assert_observations_aligned
from .metrics import (
    daily_horizon_ic,
    finite_mean,
    moving_block_bootstrap,
    per_date_primary_ic,
    primary_validation_score,
    rank_average_predictions,
    sample_level_spearman_ic,
)
from .trajectory import (
    DIAGNOSTIC_RULES,
    ELIGIBLE_RULES,
    predictions_for_rule,
)

BOOTSTRAP_BLOCK_LENGTHS = (5, 10)
BOOTSTRAP_REPLICATIONS = 10_000


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def load_run_observations(run_dir: Path, rule: str) -> EvaluationObservations:
    with np.load(run_dir / "validation_reference.npz", allow_pickle=False) as values:
        fields = {
            name: values[name].copy()
            for name in EvaluationObservations.__dataclass_fields__
            if name != "predictions"
        }
    return EvaluationObservations(
        predictions=predictions_for_rule(run_dir, rule),
        **fields,
    )


def _ordered(observations: EvaluationObservations) -> EvaluationObservations:
    order = np.argsort(observations.sample_id, kind="stable")
    sample_id = observations.sample_id[order]
    if sample_id.size == 0 or np.unique(sample_id).size != sample_id.size:
        raise ValueError("Observation sample_id values must be non-empty and unique")
    return EvaluationObservations(
        **{
            name: getattr(observations, name)[order]
            for name in EvaluationObservations.__dataclass_fields__
        }
    )


def align_observations(
    observations: Mapping[str, EvaluationObservations],
) -> dict[str, EvaluationObservations]:
    if not observations:
        raise ValueError("At least one observation set is required")
    ordered = {name: _ordered(value) for name, value in observations.items()}
    reference = next(iter(ordered.values()))
    for candidate in ordered.values():
        assert_observations_aligned(reference, candidate)
    return ordered


def _member_name(run_dir: Path) -> str:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    return f"seed_{manifest['seed']}"


def _ensemble_summary(
    members: Mapping[str, EvaluationObservations],
) -> tuple[dict[str, object], EvaluationObservations]:
    aligned = align_observations(members)
    reference = next(iter(aligned.values()))
    member_scores = {
        name: primary_validation_score(
            value.predictions,
            value.targets,
            value.label_mask,
            value.date_idx,
        )
        for name, value in aligned.items()
    }
    ensemble_predictions = rank_average_predictions(
        [value.predictions for value in aligned.values()],
        reference.label_mask,
    )
    ensemble = replace(reference, predictions=ensemble_predictions)
    ensemble_score = primary_validation_score(
        ensemble.predictions,
        ensemble.targets,
        ensemble.label_mask,
        ensemble.date_idx,
    )
    diversity = []
    for (left_name, left), (right_name, right) in combinations(aligned.items(), 2):
        diversity.append(
            {
                "left": left_name,
                "right": right_name,
                "prediction_spearman": primary_validation_score(
                    left.predictions,
                    right.predictions,
                    reference.label_mask,
                    reference.date_idx,
                ),
            }
        )
    mean_member = float(np.mean(tuple(member_scores.values())))
    best_member = max(member_scores.values())
    return (
        {
            "member_ic": member_scores,
            "ensemble_ic": ensemble_score,
            "pairwise_seed_diversity": diversity,
            "gain_vs_mean_member": ensemble_score - mean_member,
            "gain_vs_best_member": ensemble_score - best_member,
            "ensemble_method": (
                "uniform within-sample/horizon average of tie-aware member ranks"
            ),
            "learned_weights": False,
        },
        ensemble,
    )


def _bootstrap(values: np.ndarray, *, seed_offset: int = 0) -> dict[str, object]:
    result: dict[str, object] = {}
    for block_length in BOOTSTRAP_BLOCK_LENGTHS:
        interval = moving_block_bootstrap(
            values,
            replications=BOOTSTRAP_REPLICATIONS,
            block_length=block_length,
            seed=20260819 + seed_offset + block_length,
        )
        result[str(block_length)] = {
            name: np.asarray(value).tolist() for name, value in interval.items()
        }
    return result


def compare_ensembles(
    candidate_runs: Sequence[Path],
    parent_runs: Sequence[Path],
    *,
    candidate_rule: str,
    parent_rule: str,
    output_dir: Path,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    candidate_members = {
        _member_name(run): load_run_observations(run, candidate_rule)
        for run in candidate_runs
    }
    parent_members = {
        _member_name(run): load_run_observations(run, parent_rule)
        for run in parent_runs
    }
    all_members = align_observations(
        {
            **{f"candidate_{name}": value for name, value in candidate_members.items()},
            **{f"parent_{name}": value for name, value in parent_members.items()},
        }
    )
    candidate_members = {
        name.removeprefix("candidate_"): value
        for name, value in all_members.items()
        if name.startswith("candidate_")
    }
    parent_members = {
        name.removeprefix("parent_"): value
        for name, value in all_members.items()
        if name.startswith("parent_")
    }
    candidate_summary, candidate = _ensemble_summary(candidate_members)
    parent_summary, parent = _ensemble_summary(parent_members)
    candidate_ic = sample_level_spearman_ic(
        candidate.predictions, candidate.targets, candidate.label_mask
    )
    parent_ic = sample_level_spearman_ic(
        parent.predictions, parent.targets, parent.label_mask
    )
    delta_ic = candidate_ic - parent_ic
    dates, candidate_daily = per_date_primary_ic(candidate_ic, candidate.date_idx)
    parent_dates, parent_daily = per_date_primary_ic(parent_ic, parent.date_idx)
    if not np.array_equal(dates, parent_dates):
        raise ValueError("Candidate and parent date sets differ")
    daily_delta = candidate_daily - parent_daily
    pl.DataFrame(
        {
            "date_idx": dates,
            "candidate_ic": candidate_daily,
            "parent_ic": parent_daily,
            "candidate_minus_parent_ic": daily_delta,
        }
    ).write_parquet(output_dir / "daily_delta.parquet")

    _, candidate_horizon = daily_horizon_ic(candidate_ic, candidate.date_idx)
    _, parent_horizon = daily_horizon_ic(parent_ic, parent.date_idx)
    horizon_delta = candidate_horizon - parent_horizon
    horizon_rows = []
    for index, minutes in enumerate(HORIZONS):
        horizon_rows.append(
            {
                "horizon_minutes": minutes,
                "candidate_minus_parent_ic": finite_mean(horizon_delta[:, index]),
                "bootstrap": _bootstrap(horizon_delta[:, index], seed_offset=index * 100),
            }
        )
    pl.DataFrame(
        [
            {
                "horizon_minutes": row["horizon_minutes"],
                "candidate_minus_parent_ic": row["candidate_minus_parent_ic"],
            }
            for row in horizon_rows
        ]
    ).write_parquet(output_dir / "horizon_guardrails.parquet")

    tod_rows = []
    for decision in np.unique(candidate.decision_idx):
        by_date = np.asarray(
            [
                finite_mean(
                    delta_ic[
                        (candidate.date_idx == date_value)
                        & (candidate.decision_idx == decision)
                    ].ravel()
                )
                for date_value in dates
            ],
            dtype=np.float64,
        )
        minute = 10 * 60 + 15 + 5 * int(decision)
        tod_rows.append(
            {
                "decision_idx": int(decision),
                "decision_time": f"{minute // 60:02d}:{minute % 60:02d}",
                "candidate_minus_parent_ic": finite_mean(by_date),
                "bootstrap": _bootstrap(by_date, seed_offset=1000 + int(decision) * 100),
            }
        )
    pl.DataFrame(
        [
            {
                "decision_idx": row["decision_idx"],
                "decision_time": row["decision_time"],
                "candidate_minus_parent_ic": row["candidate_minus_parent_ic"],
            }
            for row in tod_rows
        ]
    ).write_parquet(output_dir / "time_of_day_guardrails.parquet")

    report = {
        "schema": "ENSEMBLE_COMPARISON",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_rule": candidate_rule,
        "parent_rule": parent_rule,
        "candidate": candidate_summary,
        "parent": parent_summary,
        "candidate_minus_parent_primary_ic": (
            candidate_summary["ensemble_ic"] - parent_summary["ensemble_ic"]
        ),
        "per_date_delta_bootstrap": _bootstrap(daily_delta),
        "horizon_guardrails": horizon_rows,
        "time_of_day_guardrails": tod_rows,
        "alignment_fields": [
            "sample_id",
            "date_idx",
            "decision_idx",
            "targets",
            "label_mask",
            "raw_returns",
        ],
    }
    _atomic_json(output_dir / "analysis.json", report)
    return output_dir


def select_trajectory_rule(
    fold_runs: Mapping[str, Sequence[Path]],
    output_path: Path,
) -> Path:
    if set(fold_runs) != {"fold_a", "fold_b"}:
        raise ValueError("Trajectory selection requires fold_a and fold_b")
    reports: dict[str, dict[str, object]] = {}
    for rule in (*ELIGIBLE_RULES, *DIAGNOSTIC_RULES):
        fold_scores = {}
        fold_members = {}
        for fold, runs in fold_runs.items():
            members = {
                _member_name(run): load_run_observations(run, rule) for run in runs
            }
            summary, _ = _ensemble_summary(members)
            fold_scores[fold] = summary["ensemble_ic"]
            fold_members[fold] = summary["member_ic"]
        reports[rule] = {
            "fold_ensemble_ic": fold_scores,
            "fold_member_ic": fold_members,
            "mean_fold_ensemble_ic": float(np.mean(tuple(fold_scores.values()))),
            "selection_eligible": rule in ELIGIBLE_RULES,
        }
    selected = min(
        ELIGIBLE_RULES,
        key=lambda rule: (-float(reports[rule]["mean_fold_ensemble_ic"]), rule),
    )
    result = {
        "schema": "TRAJECTORY_SELECTION",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_rule": selected,
        "selection_criterion": (
            "maximum mean of the two internal-fold three-seed rank-ensemble ICs; "
            "lexical tie-break"
        ),
        "official_validation_reselection_allowed": False,
        "test_accessed": False,
        "rules": reports,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_path, result)
    return output_path


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze aligned trajectory predictions")
    subparsers = parser.add_subparsers(dest="command", required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--candidate-run", action="append", type=Path, required=True)
    compare.add_argument("--parent-run", action="append", type=Path, required=True)
    compare.add_argument("--candidate-rule", choices=ELIGIBLE_RULES, required=True)
    compare.add_argument("--parent-rule", choices=ELIGIBLE_RULES, required=True)
    compare.add_argument("--output-dir", type=Path, required=True)
    select = subparsers.add_parser("select-trajectory")
    select.add_argument("--fold-a-run", action="append", type=Path, required=True)
    select.add_argument("--fold-b-run", action="append", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    return parser.parse_args(arguments)


def main() -> None:
    args = parse_args()
    if args.command == "compare":
        result = compare_ensembles(
            args.candidate_run,
            args.parent_run,
            candidate_rule=args.candidate_rule,
            parent_rule=args.parent_rule,
            output_dir=args.output_dir,
        )
    else:
        result = select_trajectory_rule(
            {"fold_a": args.fold_a_run, "fold_b": args.fold_b_run},
            args.output,
        )
    print(result)


if __name__ == "__main__":
    main()
