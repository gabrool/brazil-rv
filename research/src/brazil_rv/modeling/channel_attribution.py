from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from brazil_rv.preprocessing.contract import DYNAMIC_CHANNELS, SLOW_CHANNELS

from .contract import ALLOWED_SEEDS, GH200_RUNTIME, HORIZONS
from .data import (
    create_evaluation_loader,
    load_sample_index,
    select_training_window,
)
from .designated_challenger import crossfit_patience3_observations
from .engine import (
    EvaluationObservations,
    assert_observations_aligned,
    collect_equity_input_ablation_predictions,
    compile_model,
)
from .metrics import (
    daily_horizon_ic,
    finite_mean,
    moving_block_bootstrap,
    per_date_primary_ic,
    rank_average_predictions,
    sample_level_spearman_ic,
)
from .model import build_model
from .trajectory import load_checkpoint

FOLDS = ("fold_a", "fold_b")
KEEP_MIN_MEAN_DROP = 0.00025


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _bootstrap(values: np.ndarray) -> dict[str, object]:
    return {
        str(block): {
            key: np.asarray(item).tolist()
            for key, item in moving_block_bootstrap(
                values,
                replications=10_000,
                block_length=block,
                seed=20260822 + block,
            ).items()
        }
        for block in (5, 10)
    }


def _ensemble(members: Mapping[str, EvaluationObservations]) -> EvaluationObservations:
    reference = next(iter(members.values()))
    for member in members.values():
        assert_observations_aligned(reference, member)
    predictions = rank_average_predictions(
        [member.predictions for member in members.values()], reference.label_mask
    )
    return replace(reference, predictions=predictions)


def _comparison(
    parent: EvaluationObservations, candidate: EvaluationObservations
) -> dict[str, object]:
    assert_observations_aligned(parent, candidate)
    parent_ic = sample_level_spearman_ic(
        parent.predictions, parent.targets, parent.label_mask
    )
    candidate_ic = sample_level_spearman_ic(
        candidate.predictions, candidate.targets, candidate.label_mask
    )
    dates, parent_daily = per_date_primary_ic(parent_ic, parent.date_idx)
    _, candidate_daily = per_date_primary_ic(candidate_ic, candidate.date_idx)
    _, parent_horizon = daily_horizon_ic(parent_ic, parent.date_idx)
    _, candidate_horizon = daily_horizon_ic(candidate_ic, candidate.date_idx)
    daily_drop = parent_daily - candidate_daily
    return {
        "parent_minus_zeroed_ic": finite_mean(daily_drop),
        "per_horizon_parent_minus_zeroed_ic": {
            str(minutes): finite_mean(
                parent_horizon[:, index] - candidate_horizon[:, index]
            )
            for index, minutes in enumerate(HORIZONS)
        },
        "date_count": int(dates.size),
        "moving_block_bootstrap": _bootstrap(daily_drop),
    }


def _date_parities(date_idx: np.ndarray) -> dict[str, np.ndarray]:
    dates = np.unique(date_idx)
    return {"odd": dates[0::2], "even": dates[1::2]}


def run_channel_attribution(
    *, store: Path, parent_campaign: Path, output_dir: Path
) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    sample_index = load_sample_index(store)
    parent_by_fold: dict[str, dict[str, EvaluationObservations]] = {}
    ablated_by_fold: dict[str, dict[str, dict[str, EvaluationObservations]]] = {}
    model = build_model().cuda()
    compiled = compile_model(model)

    for fold in FOLDS:
        _, selection_rows, _ = select_training_window(sample_index, fold)
        parent_by_fold[fold] = {}
        ablated_by_fold[fold] = {
            **{f"dynamic_{index}": {} for index in range(len(DYNAMIC_CHANNELS))},
            **{f"slow_{index}": {} for index in range(len(SLOW_CHANNELS))},
        }
        for seed in ALLOWED_SEEDS:
            run = parent_campaign / fold / f"seed_{seed}"
            parent, directions = crossfit_patience3_observations(run)
            parent_by_fold[fold][f"seed_{seed}"] = parent
            per_ablation = {
                key: np.zeros_like(parent.predictions) for key in ablated_by_fold[fold]
            }
            parities = _date_parities(parent.date_idx)
            for direction in directions:
                evaluation_parity = str(direction["evaluation_parity"])
                selected_epoch = int(direction["selected_epoch"])
                evaluation_dates = parities[evaluation_parity]
                rows = selection_rows.filter(
                    selection_rows.get_column("date_idx").is_in(evaluation_dates)
                )
                loader = create_evaluation_loader(store, rows, GH200_RUNTIME, seed)
                state = load_checkpoint(run, selected_epoch)["model_state_dict"]
                model.load_state_dict(state)
                reference, predictions = collect_equity_input_ablation_predictions(
                    compiled,
                    loader,
                    dynamic_channel_count=len(DYNAMIC_CHANNELS),
                    slow_field_count=len(SLOW_CHANNELS),
                )
                expected = np.isin(parent.date_idx, evaluation_dates)
                sliced = replace(
                    parent,
                    predictions=parent.predictions[expected],
                    targets=parent.targets[expected],
                    raw_returns=parent.raw_returns[expected],
                    label_mask=parent.label_mask[expected],
                    sample_id=parent.sample_id[expected],
                    date_idx=parent.date_idx[expected],
                    decision_idx=parent.decision_idx[expected],
                )
                assert_observations_aligned(sliced, reference)
                for key, values in predictions.items():
                    per_ablation[key][expected] = values
            for key, predictions in per_ablation.items():
                ablated_by_fold[fold][key][f"seed_{seed}"] = replace(
                    parent, predictions=predictions
                )

    results: dict[str, dict[str, object]] = {}
    for kind, names in (("dynamic", DYNAMIC_CHANNELS), ("slow", SLOW_CHANNELS)):
        for index, name in enumerate(names):
            key = f"{kind}_{index}"
            folds = {
                fold: _comparison(
                    _ensemble(parent_by_fold[fold]),
                    _ensemble(ablated_by_fold[fold][key]),
                )
                for fold in FOLDS
            }
            drops = [float(folds[fold]["parent_minus_zeroed_ic"]) for fold in FOLDS]
            dead = all(value <= 0 for value in drops) and all(
                sum(
                    float(value) <= 0
                    for value in folds[fold][
                        "per_horizon_parent_minus_zeroed_ic"
                    ].values()
                )
                >= 2
                for fold in FOLDS
            )
            keep = np.mean(drops) >= KEEP_MIN_MEAN_DROP and all(
                value >= 0 for value in drops
            )
            results[key] = {
                "kind": kind,
                "index": index,
                "name": name,
                "folds": folds,
                "mean_parent_minus_zeroed_ic": float(np.mean(drops)),
                "classification": "dead" if dead else "keep" if keep else "suspect",
            }
    ranked = sorted(
        results.values(),
        key=lambda row: (
            -float(row["mean_parent_minus_zeroed_ic"]),
            str(row["kind"]),
            int(row["index"]),
        ),
    )
    report = {
        "schema": "P0_CHANNEL_ATTRIBUTION_V1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": (
            "Cross-fitted Patience epoch selected on one date parity; one equity "
            "input field zeroed and evaluated only on the opposite parity; both "
            "directions and all three seeds rank-ensembled. Context inputs unchanged."
        ),
        "classification_rule": {
            "dead": "overall drop <= 0 on both folds and >=2/3 horizon drops <=0 on each fold",
            "keep": f"mean drop >= {KEEP_MIN_MEAN_DROP} and neither fold drop < 0",
            "suspect": "otherwise",
        },
        "ranked_features": ranked,
        "dead_dynamic_indices": [
            int(row["index"])
            for row in ranked
            if row["kind"] == "dynamic" and row["classification"] == "dead"
        ],
        "dead_slow_indices": [
            int(row["index"])
            for row in ranked
            if row["kind"] == "slow" and row["classification"] == "dead"
        ],
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(output_dir / "channel_attribution.json", report)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P0.3 parent channel attribution")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--parent-campaign", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        run_channel_attribution(
            store=args.store,
            parent_campaign=args.parent_campaign,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
