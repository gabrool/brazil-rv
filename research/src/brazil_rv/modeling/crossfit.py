from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .analyze import BOOTSTRAP_BLOCK_LENGTHS, BOOTSTRAP_REPLICATIONS
from .contract import HORIZONS, MAX_EPOCHS, VALIDATION_END
from .data import (
    create_evaluation_loader,
    load_sample_index,
    load_recorded_external_sidecar,
    select_training_window,
    validate_feature_store_identity,
)
from .engine import (
    EvaluationObservations,
    assert_observations_aligned,
    collect_validation_observations,
)
from .metrics import (
    daily_horizon_ic,
    finite_mean,
    moving_block_bootstrap,
    per_date_primary_ic,
    primary_validation_score,
    rank_average_predictions,
    sample_level_spearman_ic,
)
from .model import build_model
from .provenance import repository_commit
from .trajectory import (
    ELIGIBLE_RULES,
    average_state_dicts,
    load_checkpoint,
    predictions_for_rule,
    simulate_patience3,
)

EXTENDED_WEIGHT_RULES = (
    "last7_weight_average",
    "last10_weight_average",
)
FIXED_RULES = (*ELIGIBLE_RULES, *EXTENDED_WEIGHT_RULES)
ADAPTIVE_RULE_KEYS = {
    "patience3_raw": "raw",
    "patience3_ema_0995": "ema_0995",
}
CANDIDATE_RULES = (*FIXED_RULES, *ADAPTIVE_RULE_KEYS)


@dataclass(frozen=True)
class TrajectoryMember:
    name: str
    reference: EvaluationObservations
    fixed_predictions: Mapping[str, np.ndarray]
    epoch_predictions: Mapping[str, Sequence[np.ndarray]]


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.savez(output, **values)
    os.replace(temporary, path)


def _load_reference(run_dir: Path) -> tuple[EvaluationObservations, np.ndarray]:
    with np.load(run_dir / "validation_reference.npz", allow_pickle=False) as values:
        fields = {
            name: values[name].copy()
            for name in EvaluationObservations.__dataclass_fields__
            if name != "predictions"
        }
    order = np.argsort(fields["sample_id"], kind="stable")
    if order.size == 0 or np.unique(fields["sample_id"]).size != order.size:
        raise ValueError("Observation sample_id values must be non-empty and unique")
    ordered = {name: value[order] for name, value in fields.items()}
    return (
        EvaluationObservations(
            predictions=np.empty_like(ordered["targets"], dtype=np.float32),
            **ordered,
        ),
        order,
    )


def _member_name_and_fold(run_dir: Path) -> tuple[str, str]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(f"Trajectory is not completed: {run_dir}")
    fold = str(manifest.get("split", {}).get("training"))
    if fold not in ("fold_a", "fold_b"):
        raise ValueError(f"Trajectory is not an internal discovery fold: {run_dir}")
    return f"seed_{int(manifest['seed'])}", fold


def _extension_path(extension_dir: Path, run_dir: Path) -> Path:
    return extension_dir / f"{run_dir.parent.name}_{run_dir.name}.npz"


def load_trajectory_member(run_dir: Path, extension_dir: Path) -> TrajectoryMember:
    name, _ = _member_name_and_fold(run_dir)
    reference, order = _load_reference(run_dir)
    fixed = {
        rule: predictions_for_rule(run_dir, rule)[order] for rule in ELIGIBLE_RULES
    }
    extension_path = _extension_path(extension_dir, run_dir)
    with np.load(extension_path, allow_pickle=False) as values:
        for rule in EXTENDED_WEIGHT_RULES:
            fixed[rule] = values[rule][order].copy()
    epochs = {key: [] for key in ADAPTIVE_RULE_KEYS.values()}
    for epoch in range(1, MAX_EPOCHS + 1):
        with np.load(
            run_dir / "validation_predictions" / f"epoch_{epoch:02d}.npz",
            allow_pickle=False,
        ) as values:
            for key in epochs:
                epochs[key].append(values[key][order].copy())
    return TrajectoryMember(
        name=name,
        reference=reference,
        fixed_predictions=fixed,
        epoch_predictions=epochs,
    )


def _score(
    reference: EvaluationObservations,
    predictions: np.ndarray,
    sample_mask: np.ndarray,
) -> float:
    return primary_validation_score(
        predictions[sample_mask],
        reference.targets[sample_mask],
        reference.label_mask[sample_mask],
        reference.date_idx[sample_mask],
    )


def _ensemble(
    reference: EvaluationObservations,
    member_predictions: Mapping[str, np.ndarray],
    sample_mask: np.ndarray,
) -> tuple[float, np.ndarray]:
    predictions = rank_average_predictions(
        [value[sample_mask] for value in member_predictions.values()],
        reference.label_mask[sample_mask],
    )
    score = primary_validation_score(
        predictions,
        reference.targets[sample_mask],
        reference.label_mask[sample_mask],
        reference.date_idx[sample_mask],
    )
    return score, predictions


def _bootstrap(values: np.ndarray, *, seed_offset: int = 0) -> dict[str, object]:
    result = {}
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


def _paired_summary(
    reference: EvaluationObservations,
    candidate: np.ndarray,
    parent: np.ndarray,
) -> dict[str, object]:
    candidate_ic = sample_level_spearman_ic(
        candidate, reference.targets, reference.label_mask
    )
    parent_ic = sample_level_spearman_ic(
        parent, reference.targets, reference.label_mask
    )
    delta_ic = candidate_ic - parent_ic
    dates, candidate_daily = per_date_primary_ic(candidate_ic, reference.date_idx)
    parent_dates, parent_daily = per_date_primary_ic(parent_ic, reference.date_idx)
    if not np.array_equal(dates, parent_dates):
        raise ValueError("Candidate and parent date sets differ")
    daily_delta = candidate_daily - parent_daily
    _, candidate_horizon = daily_horizon_ic(candidate_ic, reference.date_idx)
    _, parent_horizon = daily_horizon_ic(parent_ic, reference.date_idx)
    horizon_delta = candidate_horizon - parent_horizon
    tod = {}
    for decision in np.unique(reference.decision_idx):
        by_date = np.asarray(
            [
                finite_mean(
                    delta_ic[
                        (reference.date_idx == date_value)
                        & (reference.decision_idx == decision)
                    ].ravel()
                )
                for date_value in dates
            ],
            dtype=np.float64,
        )
        tod[str(int(decision))] = finite_mean(by_date)
    return {
        "candidate_minus_parent_primary_ic": finite_mean(daily_delta),
        "per_date_delta_bootstrap": _bootstrap(daily_delta),
        "horizon_delta": {
            str(minutes): finite_mean(horizon_delta[:, index])
            for index, minutes in enumerate(HORIZONS)
        },
        "time_of_day_delta": tod,
    }


def _date_parities(date_idx: np.ndarray) -> dict[str, np.ndarray]:
    dates = np.unique(date_idx)
    if dates.size < 2 or dates.size % 2:
        raise ValueError("Odd/even cross-fitting requires an even number of dates")
    return {"odd": dates[0::2], "even": dates[1::2]}


def crossfit_fold(members: Sequence[TrajectoryMember]) -> dict[str, object]:
    if not members:
        raise ValueError("At least one trajectory member is required")
    names = [member.name for member in members]
    if len(set(names)) != len(names):
        raise ValueError("Trajectory member names must be unique")
    reference = members[0].reference
    for member in members[1:]:
        assert_observations_aligned(reference, member.reference)
    parities = _date_parities(reference.date_idx)
    crossfit_member_predictions = {
        rule: {
            member.name: np.empty_like(reference.targets, dtype=np.float32)
            for member in members
        }
        for rule in CANDIDATE_RULES
    }
    procedure_predictions = np.empty_like(reference.targets, dtype=np.float32)
    directions = []
    for selection_parity, evaluation_parity in (("odd", "even"), ("even", "odd")):
        selection_dates = parities[selection_parity]
        evaluation_dates = parities[evaluation_parity]
        selection_mask = np.isin(reference.date_idx, selection_dates)
        evaluation_mask = np.isin(reference.date_idx, evaluation_dates)
        rule_rows = {}
        evaluation_ensemble_predictions = {}
        for rule in CANDIDATE_RULES:
            selected_predictions = {}
            member_replays = {}
            for member in members:
                if rule in FIXED_RULES:
                    predictions = member.fixed_predictions[rule]
                else:
                    key = ADAPTIVE_RULE_KEYS[rule]
                    scores = [
                        _score(member.reference, predictions, selection_mask)
                        for predictions in member.epoch_predictions[key]
                    ]
                    replay = simulate_patience3(scores)
                    predictions = member.epoch_predictions[key][
                        int(replay["selected_epoch"]) - 1
                    ]
                    member_replays[member.name] = {
                        "selected_epoch": int(replay["selected_epoch"]),
                        "stopped_epoch": int(replay["stopped_epoch"]),
                        "selection_half_ic": float(replay["selected_score"]),
                    }
                selected_predictions[member.name] = predictions
                crossfit_member_predictions[rule][member.name][evaluation_mask] = (
                    predictions[evaluation_mask]
                )
            selection_ic, _ = _ensemble(reference, selected_predictions, selection_mask)
            evaluation_ic, evaluation_predictions = _ensemble(
                reference, selected_predictions, evaluation_mask
            )
            evaluation_ensemble_predictions[rule] = evaluation_predictions
            rule_rows[rule] = {
                "selection_half_ensemble_ic": selection_ic,
                "evaluation_half_ensemble_ic": evaluation_ic,
                "member_patience_replay": member_replays,
            }
        selected_rule = min(
            CANDIDATE_RULES,
            key=lambda rule: (-rule_rows[rule]["selection_half_ensemble_ic"], rule),
        )
        procedure_predictions[evaluation_mask] = evaluation_ensemble_predictions[
            selected_rule
        ]
        directions.append(
            {
                "selection_parity": selection_parity,
                "evaluation_parity": evaluation_parity,
                "selection_date_idx": selection_dates.tolist(),
                "evaluation_date_idx": evaluation_dates.tolist(),
                "rules": rule_rows,
                "rule_selected_on_selection_half": selected_rule,
                "selected_rule_evaluation_half_ic": rule_rows[selected_rule][
                    "evaluation_half_ensemble_ic"
                ],
            }
        )
    rule_reports = {}
    ensemble_predictions = {}
    full_mask = np.ones(reference.date_idx.shape, dtype=bool)
    for rule, member_predictions in crossfit_member_predictions.items():
        member_ic = {
            name: _score(reference, predictions, full_mask)
            for name, predictions in member_predictions.items()
        }
        ensemble_ic, predictions = _ensemble(reference, member_predictions, full_mask)
        ensemble_predictions[rule] = predictions
        rule_reports[rule] = {
            "member_crossfit_ic": member_ic,
            "ensemble_crossfit_ic": ensemble_ic,
        }
    parent = ensemble_predictions["final_ema_0995"]
    for rule in CANDIDATE_RULES:
        rule_reports[rule]["paired_vs_final_ema_0995"] = _paired_summary(
            reference, ensemble_predictions[rule], parent
        )
    procedure_ic = _score(reference, procedure_predictions, full_mask)
    return {
        "date_split": (
            "chronologically sorted selection dates alternate odd/even; each direction "
            "selects only on one parity and reports only on the opposite parity"
        ),
        "date_count": int(np.unique(reference.date_idx).size),
        "rules": rule_reports,
        "rule_selection_crossfit": {
            "directions": directions,
            "combined_out_of_half_ensemble_ic": procedure_ic,
        },
    }


def _materialize_extended_weight_predictions(
    run_dirs: Sequence[Path], extension_dir: Path
) -> None:
    extension_dir.mkdir(parents=True, exist_ok=False)
    torch.set_float32_matmul_precision("high")
    manifest_rows = []
    identity_cache = {}
    for run_dir in run_dirs:
        member_name, fold = _member_name_and_fold(run_dir)
        manifest = json.loads(
            (run_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        store = Path(str(manifest["feature_store"])).resolve()
        validate_feature_store_identity(
            store,
            manifest["feature_store_identity"],
            identity_cache=identity_cache,
        )
        sample_index = load_sample_index(store, through=VALIDATION_END)
        _, selection_rows, _ = select_training_window(sample_index, fold)
        sidecar = load_recorded_external_sidecar(
            manifest.get("external_sidecar"), store
        )
        loader = create_evaluation_loader(
            store,
            selection_rows,
            seed=int(manifest["seed"]),
            sidecar=sidecar,
        )
        reference, _ = _load_reference(run_dir)
        model = build_model(None if sidecar is None else sidecar.feature_count).cuda()
        predictions = {}
        scores = {}
        for length in (7, 10):
            states = [
                load_checkpoint(run_dir, epoch)["model_state_dict"]
                for epoch in range(MAX_EPOCHS - length + 1, MAX_EPOCHS + 1)
            ]
            model.load_state_dict(average_state_dicts(states), strict=True)
            observations, _ = collect_validation_observations(model, loader)
            assert_observations_aligned(reference, observations)
            rule = f"last{length}_weight_average"
            predictions[rule] = observations.predictions
            scores[rule] = primary_validation_score(
                observations.predictions,
                observations.targets,
                observations.label_mask,
                observations.date_idx,
            )
        output = _extension_path(extension_dir, run_dir)
        _atomic_npz(output, predictions)
        manifest_rows.append(
            {
                "source_run": str(run_dir.resolve()),
                "member": member_name,
                "fold": fold,
                "output": str(output.resolve()),
                "scores": scores,
            }
        )
        del model
        torch.cuda.empty_cache()
    _atomic_json(
        extension_dir / "manifest.json",
        {
            "schema": "TRAJECTORY_WEIGHT_AVERAGE_EXTENSION",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repository_commit": repository_commit(),
            "source_artifacts_mutated": False,
            "rules": list(EXTENDED_WEIGHT_RULES),
            "runs": manifest_rows,
        },
    )


def run_crossfit_analysis(
    fold_runs: Mapping[str, Sequence[Path]], output_dir: Path
) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if set(fold_runs) != {"fold_a", "fold_b"}:
        raise ValueError("Cross-fit analysis requires fold_a and fold_b")
    all_runs = [run for fold in ("fold_a", "fold_b") for run in fold_runs[fold]]
    output_dir.mkdir(parents=True)
    extension_dir = output_dir / "extended_predictions"
    _materialize_extended_weight_predictions(all_runs, extension_dir)
    folds = {}
    for fold, run_dirs in fold_runs.items():
        members = []
        for run_dir in run_dirs:
            _, actual_fold = _member_name_and_fold(run_dir)
            if actual_fold != fold:
                raise ValueError(f"Run assigned to {fold} belongs to {actual_fold}")
            members.append(load_trajectory_member(run_dir, extension_dir))
        folds[fold] = crossfit_fold(members)
    rules = {}
    for rule in CANDIDATE_RULES:
        fold_scores = {
            fold: report["rules"][rule]["ensemble_crossfit_ic"]
            for fold, report in folds.items()
        }
        rules[rule] = {
            "fold_crossfit_ensemble_ic": fold_scores,
            "mean_fold_crossfit_ensemble_ic": float(
                np.mean(tuple(fold_scores.values()))
            ),
        }
    selected_rule = min(
        CANDIDATE_RULES,
        key=lambda rule: (-rules[rule]["mean_fold_crossfit_ensemble_ic"], rule),
    )
    report = {
        "schema": "TRAJECTORY_ODD_EVEN_CROSSFIT",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_commit": repository_commit(),
        "selected_rule": selected_rule,
        "selection_criterion": (
            "maximum mean of fold-A/fold-B bidirectional odd/even cross-fitted "
            "three-seed ensemble IC; lexical tie-break"
        ),
        "official_validation_accessed": False,
        "test_accessed": False,
        "rules": rules,
        "folds": folds,
        "rule_selection_crossfit_mean_oos_ic": float(
            np.mean(
                [
                    fold["rule_selection_crossfit"]["combined_out_of_half_ensemble_ic"]
                    for fold in folds.values()
                ]
            )
        ),
    }
    _atomic_json(output_dir / "crossfit_analysis.json", report)
    _atomic_json(
        output_dir / "trajectory_selection.json",
        {
            "schema": "TRAJECTORY_SELECTION",
            "created_at": report["created_at"],
            "selected_rule": selected_rule,
            "selection_criterion": report["selection_criterion"],
            "crossfit_analysis": str((output_dir / "crossfit_analysis.json").resolve()),
            "official_validation_reselection_allowed": False,
            "test_accessed": False,
            "rules": rules,
        },
    )
    return output_dir


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-fit trajectory checkpoint and rule selection"
    )
    parser.add_argument("--fold-a-run", action="append", type=Path, required=True)
    parser.add_argument("--fold-b-run", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(arguments)


def main() -> None:
    args = parse_args()
    print(
        run_crossfit_analysis(
            {"fold_a": args.fold_a_run, "fold_b": args.fold_b_run},
            args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
