from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from brazil_rv.preprocessing.p1_features import slice_feature_sidecar

from .analyze import compare_observation_ensembles
from .channel_attribution import _comparison, _ensemble
from .contract import ALLOWED_SEEDS, GH200_RUNTIME
from .data import (
    create_evaluation_loader,
    load_external_sidecar,
    load_sample_index,
    select_training_window,
)
from .engine import (
    assert_observations_aligned,
    collect_sidecar_feature_ablation_predictions,
    compile_model,
)
from .model import build_model
from .p1_campaign import (
    FOLDS,
    _extract,
    _gate,
    _members,
    _run_job,
    crossfit_patience_observations,
)
from .trajectory import load_checkpoint

MEAN_FULL_TOLERANCE = -0.0005
FOLD_FULL_TOLERANCE = -0.001


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _parent_path(
    p1_campaign: Path, parent_campaign: Path, fold: str, seed: int
) -> Path:
    return (
        p1_campaign / "fold_c_parent" / fold / f"seed_{seed}"
        if fold == "fold_c"
        else parent_campaign / fold / f"seed_{seed}"
    )


def _candidate_path(root: Path, fold: str, seed: int) -> Path:
    return root / fold / f"seed_{seed}"


def _inference_attribution(
    *,
    store: Path,
    p1_campaign: Path,
    sidecar: Path,
    dynamic: tuple[int, ...],
    slow: tuple[int, ...],
) -> dict[str, object]:
    names = json.loads((sidecar / "manifest.json").read_text(encoding="utf-8"))[
        "feature_names"
    ]
    model = build_model(len(names)).cuda()
    compiled = compile_model(model)
    sample_index = load_sample_index(store)
    external_sidecar = load_external_sidecar(sidecar, store)
    full_by_fold = {}
    ablated_by_fold = {}
    for fold in FOLDS:
        _, selection_rows, _ = select_training_window(sample_index, fold)
        full_by_fold[fold] = {}
        ablated_by_fold[fold] = {f"feature_{index}": {} for index in range(len(names))}
        for seed in ALLOWED_SEEDS:
            run = p1_campaign / "f3_candidate" / fold / f"seed_{seed}"
            full, directions = crossfit_patience_observations(run)
            full_by_fold[fold][f"seed_{seed}"] = full
            per_feature = {
                key: np.zeros_like(full.predictions) for key in ablated_by_fold[fold]
            }
            dates = np.unique(full.date_idx)
            parities = {"odd": dates[0::2], "even": dates[1::2]}
            for direction in directions:
                evaluation_dates = parities[str(direction["evaluation_parity"])]
                evaluation = np.isin(full.date_idx, evaluation_dates)
                rows = selection_rows.filter(
                    selection_rows.get_column("date_idx").is_in(evaluation_dates)
                )
                loader = create_evaluation_loader(
                    store,
                    rows,
                    GH200_RUNTIME,
                    seed,
                    sidecar=external_sidecar,
                    zero_dynamic_channels=dynamic,
                    zero_slow_fields=slow,
                )
                state = load_checkpoint(run, int(direction["selected_epoch"]))[
                    "model_state_dict"
                ]
                model.load_state_dict(state)
                reference, predictions = collect_sidecar_feature_ablation_predictions(
                    compiled, loader, feature_count=len(names)
                )
                sliced = replace(
                    full,
                    predictions=full.predictions[evaluation],
                    targets=full.targets[evaluation],
                    raw_returns=full.raw_returns[evaluation],
                    label_mask=full.label_mask[evaluation],
                    sample_id=full.sample_id[evaluation],
                    date_idx=full.date_idx[evaluation],
                    decision_idx=full.decision_idx[evaluation],
                )
                assert_observations_aligned(sliced, reference)
                for key, values in predictions.items():
                    per_feature[key][evaluation] = values
            for key, predictions in per_feature.items():
                ablated_by_fold[fold][key][f"seed_{seed}"] = replace(
                    full, predictions=predictions
                )

    rows = []
    for index, name in enumerate(names):
        key = f"feature_{index}"
        folds = {
            fold: _comparison(
                _ensemble(full_by_fold[fold]),
                _ensemble(ablated_by_fold[fold][key]),
            )
            for fold in FOLDS
        }
        drops = [float(folds[fold]["parent_minus_zeroed_ic"]) for fold in FOLDS]
        rows.append(
            {
                "feature": name,
                "folds": folds,
                "mean_full_minus_zeroed_ic": float(np.mean(drops)),
                "survives": np.mean(drops) > 0
                and sum(value >= 0 for value in drops) >= 2,
            }
        )
    return {
        "method": "zero value and observedness mask on opposite-parity cross-fit states",
        "features": rows,
        "surviving_features": [row["feature"] for row in rows if row["survives"]],
    }


def run_f4(
    *,
    store: Path,
    parent_campaign: Path,
    p1_campaign: Path,
    selected_sidecar: Path,
    output_dir: Path,
    parallel_processes: int = 2,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    f3 = json.loads((p1_campaign / "f3_summary.json").read_text(encoding="utf-8"))
    if not f3.get("f3_passed"):
        _atomic_json(
            output_dir / "f4_summary.json",
            {
                "schema": "P1_F4_SUMMARY_V1",
                "status": "not_run",
                "reason": "F3 primary gate did not pass",
                "official_validation_accessed": False,
                "test_accessed": False,
            },
        )
        return output_dir
    manifest = json.loads(
        (p1_campaign / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    dynamic = tuple(int(value) for value in manifest["dead_dynamic_indices"])
    slow = tuple(int(value) for value in manifest["dead_slow_indices"])
    attribution = _inference_attribution(
        store=store,
        p1_campaign=p1_campaign,
        sidecar=selected_sidecar,
        dynamic=dynamic,
        slow=slow,
    )
    selected = json.loads(
        (selected_sidecar / "manifest.json").read_text(encoding="utf-8")
    )["feature_names"]
    survivors = attribution["surviving_features"]
    if not survivors:
        _atomic_json(
            output_dir / "f4_summary.json",
            {
                "schema": "P1_F4_SUMMARY_V1",
                "status": "completed",
                "inference_attribution": attribution,
                "promotion_eligible": False,
                "reason": "No F2 feature survived the preregistered F4 rule",
                "official_validation_accessed": False,
                "test_accessed": False,
            },
        )
        return output_dir
    if survivors == selected:
        _atomic_json(
            output_dir / "f4_summary.json",
            {
                "schema": "P1_F4_SUMMARY_V1",
                "status": "completed",
                "inference_attribution": attribution,
                "minimal_recipe": "unchanged_full_f3",
                "promotion_eligible": True,
                "official_validation_accessed": False,
                "test_accessed": False,
            },
        )
        return output_dir

    reduced_sidecar = slice_feature_sidecar(
        selected_sidecar, survivors, output_dir / "reduced_sidecar"
    )
    reduced_root = output_dir / "reduced_candidate"
    jobs = [
        (
            store,
            _candidate_path(reduced_root, fold, seed),
            seed,
            fold,
            reduced_sidecar,
            dynamic,
            slow,
        )
        for fold in FOLDS
        for seed in ALLOWED_SEEDS
    ]
    with ProcessPoolExecutor(
        max_workers=parallel_processes, mp_context=mp.get_context("spawn")
    ) as executor:
        futures = [executor.submit(_run_job, *job) for job in jobs]
        for future in as_completed(futures):
            print(future.result(), flush=True)

    path_results: dict[str, dict[str, object]] = {
        "standalone": {},
        "parent_plus_candidate": {},
    }
    full_deltas: dict[str, list[float]] = {key: [] for key in path_results}
    for fold in FOLDS:
        reduced, _ = _members(
            {
                f"seed_{seed}": _candidate_path(reduced_root, fold, seed)
                for seed in ALLOWED_SEEDS
            },
            "patience3_raw",
        )
        full, _ = _members(
            {
                f"seed_{seed}": p1_campaign / "f3_candidate" / fold / f"seed_{seed}"
                for seed in ALLOWED_SEEDS
            },
            "patience3_raw",
        )
        parent, _ = _members(
            {
                f"seed_{seed}": _parent_path(p1_campaign, parent_campaign, fold, seed)
                for seed in ALLOWED_SEEDS
            },
            "patience3_raw",
        )
        recipes = {
            "standalone": (reduced, full),
            "parent_plus_candidate": (
                {
                    **{f"parent_{key}": value for key, value in parent.items()},
                    **{f"candidate_{key}": value for key, value in reduced.items()},
                },
                {
                    **{f"parent_{key}": value for key, value in parent.items()},
                    **{f"candidate_{key}": value for key, value in full.items()},
                },
            ),
        }
        for path_name, (recipe, full_recipe) in recipes.items():
            vs_parent = compare_observation_ensembles(
                recipe,
                parent,
                candidate_rule="f4_crossfit_patience3_raw",
                parent_rule="crossfit_patience3_raw",
                output_dir=output_dir / "analysis" / path_name / fold / "vs_parent",
                comparison_metadata={"retention_comparator": True},
            )
            vs_full = compare_observation_ensembles(
                recipe,
                full_recipe,
                candidate_rule="f4_crossfit_patience3_raw",
                parent_rule="full_f3_crossfit_patience3_raw",
                output_dir=output_dir / "analysis" / path_name / fold / "vs_full_f3",
                comparison_metadata={"minimality_guardrail": True},
            )
            parent_result = _extract(vs_parent)
            full_result = _extract(vs_full)
            path_results[path_name][fold] = {
                "vs_parent": parent_result,
                "vs_full_f3": full_result,
            }
            full_deltas[path_name].append(
                float(full_result["candidate_minus_parent_ic"])
            )

    promotion_paths = []
    for path_name in path_results:
        folds = {fold: path_results[path_name][fold]["vs_parent"] for fold in FOLDS}
        if (
            _gate(folds)
            and np.mean(full_deltas[path_name]) >= MEAN_FULL_TOLERANCE
            and all(value >= FOLD_FULL_TOLERANCE for value in full_deltas[path_name])
        ):
            promotion_paths.append(path_name)
    _atomic_json(
        output_dir / "f4_summary.json",
        {
            "schema": "P1_F4_SUMMARY_V1",
            "status": "completed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inference_attribution": attribution,
            "reduced_sidecar": str(reduced_sidecar),
            "path_results": path_results,
            "promotion_eligible_paths": promotion_paths,
            "promotion_eligible": bool(promotion_paths),
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run conditional P1 F4 minimality")
    for name in (
        "store",
        "parent_campaign",
        "p1_campaign",
        "selected_sidecar",
        "output_dir",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--parallel-processes", type=int, default=2)
    print(run_f4(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
