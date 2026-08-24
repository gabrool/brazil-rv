from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from brazil_rv.preprocessing.external_sidecar import materialize_external_sidecar
from brazil_rv.preprocessing.external_sidecar_subset import subset_external_sidecar
from brazil_rv.preprocessing.options_full import (
    FEATURES as OPTIONS_FEATURES,
    IV_FEATURES,
    build_full_options_source,
)

from .analyze import compare_observation_ensembles, load_run_observations
from .contract import ALLOWED_SEEDS, GH200_RUNTIME, HORIZONS
from .data import (
    create_evaluation_loader,
    feature_store_identity,
    load_sample_index,
    select_training_window,
)
from .engine import (
    EvaluationObservations,
    assert_observations_aligned,
    collect_equity_input_ablation_predictions,
    compile_model,
)
from .feature_removal import (
    FEATURE_BY_KEY,
    FEATURES,
    FOLDS,
    PREVIEW_FOLD_LIMIT,
    PREVIEW_MEAN_LIMIT,
    REPRESENTATIVE_TEST_THRESHOLD,
    _comparison,
    _definition,
    _ensemble,
    _mean_drop,
    _preview_passes,
    _read_json,
    _sha256,
    _slice_observations,
    _write_artifact_inventory,
    build_stage_a,
)
from .model import build_model
from .provenance import repository_commit
from .three_fold_sidecar_screen import crossfit_patience_observations
from .train import run_training
from .trajectory import load_checkpoint, predictions_for_rule

R2_SPEC_SHA256 = "08c04de3396fdc31d67b6baeabab1fea80cfd137d55bf2a1aef4ee69d1a34b72"
R2_SUMMARY_SHA256 = "1070ecfadb99eef42d224b8eacc0ef31fc8e0e08ecc6a7e39aa5153e57fb18b8"
R3_NONINFERIOR_FLOOR = -0.0005
R3_IMPROVEMENT_MEAN = 0.0005
OPTIONS_GATE_MEAN = 0.0005
OPTIONS_GATE_FLOOR = -0.0005
MAX_PARALLEL = 2
PRIMARY_READOUT = "patience3_raw"
SECONDARY_READOUT = "final_ema_0995"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _r2_paths(root: Path, fold: str, seed: int) -> Path:
    return root / "stage_c" / "candidates" / "prune_r2" / fold / f"seed_{seed}"


def load_r2_contract(root: Path) -> dict[str, object]:
    specification_path = root / "stage_c" / "store_v2_feature_specification.json"
    summary_path = root / "stage_c" / "stage_c_summary.json"
    if _sha256(specification_path) != R2_SPEC_SHA256:
        raise ValueError("Experiment-41 prune-R2 specification hash changed")
    if _sha256(summary_path) != R2_SUMMARY_SHA256:
        raise ValueError("Experiment-41 Stage-C summary hash changed")
    specification = _read_json(specification_path)
    if (
        specification.get("status") != "selected"
        or specification.get("winner") != "prune_r2"
    ):
        raise ValueError("Experiment-41 prune-R2 is not the selected specification")
    removed = tuple(str(value) for value in specification["removed_fields"])
    if len(removed) != 24 or any(key not in FEATURE_BY_KEY for key in removed):
        raise ValueError("Experiment-41 prune-R2 must remove exactly 24 known fields")
    definition = _definition(removed)
    if len(definition[0]) != 6 or len(definition[1]) != 18:
        raise ValueError("Experiment-41 prune-R2 dynamic/slow counts changed")

    replays: dict[str, dict[str, list[dict[str, object]]]] = {}
    for fold in FOLDS:
        analysis_path = (
            root
            / "stage_c"
            / "analysis"
            / "prune_r2"
            / PRIMARY_READOUT
            / fold
            / "standalone"
            / "analysis.json"
        )
        analysis = _read_json(analysis_path)
        fold_replays = analysis.get("comparison_metadata", {}).get(
            "candidate_patience_replays"
        )
        expected = {f"seed_{seed}" for seed in ALLOWED_SEEDS}
        if not isinstance(fold_replays, dict) or set(fold_replays) != expected:
            raise ValueError(f"R2 analysis lacks exact candidate replays for {fold}")
        replays[fold] = {}
        for seed in ALLOWED_SEEDS:
            key = f"seed_{seed}"
            rows = fold_replays[key]
            if not isinstance(rows, list) or len(rows) != 2:
                raise ValueError(f"Malformed R2 replays for {fold} {key}")
            for row in rows:
                epoch = int(row["selected_epoch"])
                run = _r2_paths(root, fold, seed)
                for artifact in (
                    run / "checkpoints" / f"epoch_{epoch:02d}.pt",
                    run / "validation_predictions" / f"epoch_{epoch:02d}.npz",
                ):
                    if not artifact.is_file():
                        raise FileNotFoundError(artifact)
            replays[fold][key] = [dict(row) for row in rows]
            final_predictions = (
                _r2_paths(root, fold, seed)
                / "validation_predictions"
                / "epoch_20.npz"
            )
            if not final_predictions.is_file():
                raise FileNotFoundError(final_predictions)
    return {
        "root": root.resolve(),
        "removed": removed,
        "definition": definition,
        "survivors": tuple(key for key in FEATURE_BY_KEY if key not in removed),
        "replays": replays,
        "specification_path": specification_path.resolve(),
        "summary_path": summary_path.resolve(),
    }


def evaluate_r2_ablation_sets(
    *,
    store: Path,
    definitions: Mapping[str, tuple[tuple[int, ...], tuple[int, ...]]],
    r2: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    if not definitions:
        return {}
    sample_index = load_sample_index(store)
    base_dynamic, base_slow = r2["definition"]
    model = build_model().cuda()
    compiled = compile_model(model)
    results: dict[str, dict[str, object]] = {name: {} for name in definitions}
    for fold in FOLDS:
        _, selection_rows, _ = select_training_window(sample_index, fold)
        parent_members: dict[str, EvaluationObservations] = {}
        ablated_members: dict[str, dict[str, EvaluationObservations]] = {
            name: {} for name in definitions
        }
        for seed in ALLOWED_SEEDS:
            run = _r2_paths(Path(r2["root"]), fold, seed)
            parent, directions = crossfit_patience_observations(
                run, r2["replays"][fold][f"seed_{seed}"]
            )
            parent_members[f"seed_{seed}"] = parent
            predictions_by_definition = {
                name: np.zeros_like(parent.predictions) for name in definitions
            }
            dates = np.unique(parent.date_idx)
            parities = {"odd": dates[0::2], "even": dates[1::2]}
            for direction in directions:
                evaluation_dates = parities[str(direction["evaluation_parity"])]
                rows = selection_rows.filter(
                    selection_rows.get_column("date_idx").is_in(evaluation_dates)
                )
                loader = create_evaluation_loader(
                    store,
                    rows,
                    GH200_RUNTIME,
                    seed,
                    zero_dynamic_channels=base_dynamic,
                    zero_slow_fields=base_slow,
                )
                checkpoint = load_checkpoint(run, int(direction["selected_epoch"]))
                model.load_state_dict(checkpoint["model_state_dict"])
                reference, predictions = collect_equity_input_ablation_predictions(
                    compiled, loader, definitions
                )
                expected = np.isin(parent.date_idx, evaluation_dates)
                assert_observations_aligned(_slice_observations(parent, expected), reference)
                for name, values in predictions.items():
                    predictions_by_definition[name][expected] = values
            for name, predictions in predictions_by_definition.items():
                ablated_members[name][f"seed_{seed}"] = replace(
                    parent, predictions=predictions
                )
        parent_ensemble = _ensemble(parent_members)
        for name in definitions:
            results[name][fold] = _comparison(
                parent_ensemble, _ensemble(ablated_members[name])
            )
    return results


def _single_dead(folds: Mapping[str, Mapping[str, object]]) -> bool:
    return all(
        float(folds[fold]["parent_minus_ablated_ic"]) <= 0
        and sum(
            float(value) <= 0
            for value in folds[fold]["per_horizon_parent_minus_ablated_ic"].values()
        )
        >= 2
        for fold in FOLDS
    )


def run_r3_stage_b(
    *, store: Path, r2: Mapping[str, object], output_dir: Path
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    cluster_table = build_stage_a(
        store=store,
        output_dir=output_dir / "stage_a",
        feature_keys=r2["survivors"],
    )
    stage_a = _read_json(cluster_table)
    singles_raw = evaluate_r2_ablation_sets(
        store=store,
        definitions={key: _definition([key]) for key in r2["survivors"]},
        r2=r2,
    )
    singles = {
        key: {
            "field": FEATURE_BY_KEY[key],
            "folds": singles_raw[key],
            "three_fold_mean_drop": _mean_drop(singles_raw[key]),
            "classification": "dead" if _single_dead(singles_raw[key]) else "keep",
        }
        for key in r2["survivors"]
    }
    sets = [
        {**row, "set_kind": "correlation_cluster"}
        for row in stage_a["clusters"]
        if len(row["members"]) > 1
    ]
    sets.extend(
        {**row, "set_kind": "semantic_group"}
        for row in stage_a["semantic_groups"]
    )
    signature_to_id: dict[tuple[str, ...], str] = {}
    group_definitions = {}
    for row in sets:
        signature = tuple(sorted(row["members"]))
        evaluation_id = signature_to_id.setdefault(
            signature, f"group_signature_{len(signature_to_id):02d}"
        )
        row["evaluation_id"] = evaluation_id
        group_definitions[evaluation_id] = _definition(signature)
    group_results = evaluate_r2_ablation_sets(
        store=store, definitions=group_definitions, r2=r2
    )
    groups = {}
    representative_definitions = {}
    for row in sets:
        folds = group_results[row["evaluation_id"]]
        drops = [float(folds[fold]["parent_minus_ablated_ic"]) for fold in FOLDS]
        group_dead = float(np.mean(drops)) <= 0 and sum(value > 0 for value in drops) <= 1
        representative = min(
            row["members"],
            key=lambda key: (
                -float(singles[key]["three_fold_mean_drop"]),
                float(FEATURE_BY_KEY[key]["model_input_missing_fraction"]),
                len(str(FEATURE_BY_KEY[key]["name"])),
                int(FEATURE_BY_KEY[key]["global_index"]),
            ),
        )
        representative_test = (
            float(np.mean(drops)) >= REPRESENTATIVE_TEST_THRESHOLD and not group_dead
        )
        if representative_test:
            representative_definitions[f"representative_{row['set_id']}"] = _definition(
                [key for key in row["members"] if key != representative]
            )
        groups[row["set_id"]] = {
            **row,
            "joint_folds": folds,
            "joint_mean_drop": float(np.mean(drops)),
            "group_dead": group_dead,
            "representative": representative,
            "representative_test_required": representative_test,
        }
    representative_results = evaluate_r2_ablation_sets(
        store=store, definitions=representative_definitions, r2=r2
    )
    proposals: set[str] = set()
    reasons: dict[str, list[str]] = {key: [] for key in r2["survivors"]}
    for set_id, row in groups.items():
        if row["group_dead"]:
            row["classification"] = "group_dead"
            for key in row["members"]:
                proposals.add(key)
                reasons[key].append(f"group_dead:{set_id}")
        elif row["representative_test_required"]:
            folds = representative_results[f"representative_{set_id}"]
            residual = _mean_drop(folds)
            row["minus_representative_folds"] = folds
            row["minus_representative_mean_drop"] = residual
            row["classification"] = (
                "representative_sufficient"
                if residual <= REPRESENTATIVE_TEST_THRESHOLD
                else "keep_all"
            )
            if row["classification"] == "representative_sufficient":
                for key in row["members"]:
                    if key != row["representative"]:
                        proposals.add(key)
                        reasons[key].append(f"non_representative:{set_id}")
        else:
            row["classification"] = "keep_all"
    singleton_keys = {
        row["members"][0] for row in stage_a["clusters"] if len(row["members"]) == 1
    }
    for key in singleton_keys:
        if singles[key]["classification"] == "dead":
            proposals.add(key)
            reasons[key].append("dead_singleton_all_three_folds")

    preview_history = []

    def preview(name: str, members: set[str]) -> dict[str, object]:
        if not members:
            folds = {
                fold: {
                    "parent_minus_ablated_ic": 0.0,
                    "per_horizon_parent_minus_ablated_ic": {
                        str(horizon): 0.0 for horizon in HORIZONS
                    },
                    "date_count": 0,
                    "block10_interval": {},
                }
                for fold in FOLDS
            }
        else:
            folds = evaluate_r2_ablation_sets(
                store=store,
                definitions={name: _definition(sorted(members))},
                r2=r2,
            )[name]
        result = {
            "folds": folds,
            "mean_drop": _mean_drop(folds),
            "gate_passed": _preview_passes(folds),
        }
        preview_history.append({"name": name, "members": sorted(members), **result})
        return result

    current = set(proposals)
    final_preview = preview("r3_initial", current)
    if not final_preview["gate_passed"]:
        current = set()
        for key in sorted(
            proposals, key=lambda item: float(singles[item]["three_fold_mean_drop"])
        ):
            trial = current | {key}
            result = preview(f"r3_walkback_add_{key}", trial)
            if not result["gate_passed"]:
                break
            current = trial
            final_preview = result
    removed = tuple(sorted(set(r2["removed"]) | current))
    frozen = {
        "schema": "R3_FROZEN_REMOVAL_SET_V1",
        "created_at": _now(),
        "r2_specification_sha256": R2_SPEC_SHA256,
        "cluster_table": str(cluster_table.resolve()),
        "cluster_table_sha256": _sha256(cluster_table),
        "r2_removed_fields": list(r2["removed"]),
        "initial_new_removals": sorted(proposals),
        "accepted_new_removals": sorted(current),
        "r3_removed_fields": list(removed),
        "r3_definition": _definition(removed),
        "proposal_reasons": {key: value for key, value in reasons.items() if value},
        "final_preview": final_preview,
        "preview_history": preview_history,
        "preview_gate": {
            "mean_cost_max": PREVIEW_MEAN_LIMIT,
            "single_fold_cost_max": PREVIEW_FOLD_LIMIT,
        },
        "no_r3_retrain": not current,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    frozen_path = output_dir / "frozen_r3.json"
    _atomic_json(frozen_path, frozen)
    _atomic_json(
        output_dir / "stage_b_prime_report.json",
        {
            "schema": "R3_STAGE_B_PRIME_V1",
            "created_at": _now(),
            "survivor_count": len(r2["survivors"]),
            "singles": singles,
            "groups": groups,
            "frozen_r3": str(frozen_path.resolve()),
            "frozen_r3_sha256": _sha256(frozen_path),
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    return frozen_path


def _run_stage_b_isolated(**kwargs: object) -> Path:
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
        return pool.submit(run_r3_stage_b, **kwargs).result()


def _training_job(
    store: Path,
    run_dir: Path,
    seed: int,
    fold: str,
    dynamic: tuple[int, ...],
    slow: tuple[int, ...],
    sidecar: Path | None,
) -> str:
    run_training(
        store=store,
        seed=seed,
        selection_window=fold,
        run_dir=run_dir,
        zero_dynamic_channels=dynamic,
        zero_slow_fields=slow,
        sidecar_dir=sidecar,
    )
    return str(run_dir)


def _candidate_members(
    paths: Mapping[str, Path], readout: str
) -> tuple[dict[str, EvaluationObservations], dict[str, object]]:
    members = {}
    replays = {}
    for name, path in paths.items():
        if readout == PRIMARY_READOUT:
            observations, replay = crossfit_patience_observations(path)
        elif readout == SECONDARY_READOUT:
            reference = load_run_observations(path, "final_raw")
            observations = replace(
                reference, predictions=predictions_for_rule(path, SECONDARY_READOUT)
            )
            replay = []
        else:
            raise ValueError(readout)
        members[name] = observations
        replays[name] = replay
    return members, replays


def _parent_members(
    *, fold: str, r2: Mapping[str, object], readout: str
) -> tuple[dict[str, EvaluationObservations], dict[str, object]]:
    members = {}
    replays = {}
    for seed in ALLOWED_SEEDS:
        key = f"seed_{seed}"
        run = _r2_paths(Path(r2["root"]), fold, seed)
        if readout == PRIMARY_READOUT:
            observations, replay = crossfit_patience_observations(
                run, r2["replays"][fold][key]
            )
        elif readout == SECONDARY_READOUT:
            reference = load_run_observations(run, "final_raw")
            observations = replace(
                reference, predictions=predictions_for_rule(run, SECONDARY_READOUT)
            )
            replay = []
        else:
            raise ValueError(readout)
        members[key] = observations
        replays[key] = replay
    return members, replays


def _analysis_summary(path: Path) -> dict[str, object]:
    value = _read_json(path / "analysis.json")
    return {
        "candidate_ensemble_ic": value["candidate"]["ensemble_ic"],
        "parent_ensemble_ic": value["parent"]["ensemble_ic"],
        "candidate_minus_parent_ic": value["candidate_minus_parent_primary_ic"],
        "per_date_delta_bootstrap": value["per_date_delta_bootstrap"],
        "horizon_guardrails": value["horizon_guardrails"],
        "time_of_day_guardrails": value["time_of_day_guardrails"],
        "analysis": str((path / "analysis.json").resolve()),
        "daily_delta": str((path / "daily_delta.parquet").resolve()),
    }


def _compare_candidate(
    *,
    candidate_root: Path,
    candidate_name: str,
    r2: Mapping[str, object],
    output_dir: Path,
) -> dict[str, object]:
    folds = {}
    for fold in FOLDS:
        paths = {
            f"seed_{seed}": candidate_root / fold / f"seed_{seed}"
            for seed in ALLOWED_SEEDS
        }
        candidate_primary, candidate_replays = _candidate_members(
            paths, PRIMARY_READOUT
        )
        parent_primary, parent_replays = _parent_members(
            fold=fold, r2=r2, readout=PRIMARY_READOUT
        )
        standalone_path = compare_observation_ensembles(
            candidate_primary,
            parent_primary,
            candidate_rule=PRIMARY_READOUT,
            parent_rule="prune_r2_patience3_raw",
            output_dir=output_dir / fold / "primary_standalone",
            comparison_metadata={
                "fold": fold,
                "candidate_patience_replays": candidate_replays,
                "parent_patience_replays": parent_replays,
                "official_validation_accessed": False,
                "test_accessed": False,
            },
        )
        candidate_ema, _ = _candidate_members(paths, SECONDARY_READOUT)
        parent_ema, _ = _parent_members(
            fold=fold, r2=r2, readout=SECONDARY_READOUT
        )
        ema_path = compare_observation_ensembles(
            candidate_ema,
            parent_ema,
            candidate_rule=SECONDARY_READOUT,
            parent_rule="prune_r2_final_ema_0995",
            output_dir=output_dir / fold / "secondary_ema_standalone",
            comparison_metadata={
                "fold": fold,
                "informational_only": True,
                "official_validation_accessed": False,
                "test_accessed": False,
            },
        )
        mixed_path = compare_observation_ensembles(
            {
                **{
                    f"parent_patience_{key}": value
                    for key, value in parent_primary.items()
                },
                **{
                    f"candidate_ema_{key}": value
                    for key, value in candidate_ema.items()
                },
            },
            parent_primary,
            candidate_rule=f"prune_r2_patience_plus_{candidate_name}_ema_uniform_6",
            parent_rule="prune_r2_patience3_raw",
            output_dir=output_dir / fold / "secondary_mixed_state",
            comparison_metadata={
                "fold": fold,
                "predeclared_options_fallback": candidate_name.startswith("opt_"),
                "official_validation_accessed": False,
                "test_accessed": False,
            },
        )
        folds[fold] = {
            "primary_standalone": _analysis_summary(standalone_path),
            "secondary_ema_standalone": _analysis_summary(ema_path),
            "secondary_mixed_state": _analysis_summary(mixed_path),
            "candidate_patience_replays": candidate_replays,
            "parent_patience_replays": parent_replays,
        }
    return folds


def _pooled_interval(
    folds: Mapping[str, Mapping[str, object]], path: str
) -> dict[str, object]:
    daily = []
    date_indices = []
    for fold in FOLDS:
        analysis = folds[fold][path]
        frame = pl.read_parquet(analysis["daily_delta"]).sort("date_idx")
        daily.append(frame.get_column("candidate_minus_parent_ic").to_numpy())
        date_indices.extend(frame.get_column("date_idx").to_list())
    if len(date_indices) != len(set(date_indices)):
        raise ValueError("Pooled fold dates overlap")
    values = np.concatenate(daily).astype(np.float64, copy=False)
    block_length = 10
    replications = 10_000
    block_count = math.ceil(values.size / block_length)
    generator = np.random.default_rng(20260842)
    starts = generator.integers(
        0,
        values.size - block_length + 1,
        size=(replications, block_count),
    )
    indices = (
        starts[..., None] + np.arange(block_length, dtype=np.int64)
    ).reshape(replications, -1)[:, : values.size]
    means = np.nanmean(values[indices], axis=1)
    return {
        "estimate": float(np.nanmean(values)),
        "lower_90": float(np.nanquantile(means, 0.05)),
        "upper_90": float(np.nanquantile(means, 0.95)),
        "block_length": block_length,
        "replications": replications,
        "date_count": int(values.size),
        "seed": 20260842,
    }


def options_gate(
    folds: Mapping[str, Mapping[str, object]], path: str
) -> dict[str, object]:
    deltas = {
        fold: float(folds[fold][path]["candidate_minus_parent_ic"])
        for fold in FOLDS
    }
    pooled = _pooled_interval(folds, path)
    mean_delta = float(np.mean(tuple(deltas.values())))
    checks = {
        "mean_at_least_0_0005": mean_delta >= OPTIONS_GATE_MEAN,
        "at_least_two_folds_positive": sum(value > 0 for value in deltas.values()) >= 2,
        "no_fold_below_minus_0_0005": min(deltas.values()) >= OPTIONS_GATE_FLOOR,
        "pooled_block10_90_interval_excludes_zero": (
            pooled["lower_90"] > 0 or pooled["upper_90"] < 0
        ),
    }
    return {
        "fold_deltas": deltas,
        "mean_delta": mean_delta,
        "pooled_daily_delta": pooled,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _r3_summary(
    *,
    frozen_r3: Path,
    r2: Mapping[str, object],
    runs: Path | None,
    output_dir: Path,
) -> Path:
    frozen = _read_json(frozen_r3)
    if frozen["no_r3_retrain"]:
        selected = "prune_r2"
        folds = None
        primary = None
        noninferior = None
        improvement = None
    else:
        folds = _compare_candidate(
            candidate_root=runs,
            candidate_name="prune_r3",
            r2=r2,
            output_dir=output_dir / "analysis",
        )
        deltas = [
            float(folds[fold]["primary_standalone"]["candidate_minus_parent_ic"])
            for fold in FOLDS
        ]
        primary = {
            "fold_deltas": dict(zip(FOLDS, deltas, strict=True)),
            "mean_delta": float(np.mean(deltas)),
        }
        noninferior = primary["mean_delta"] >= 0 and all(
            value >= R3_NONINFERIOR_FLOOR for value in deltas
        )
        improvement = primary["mean_delta"] >= R3_IMPROVEMENT_MEAN and all(
            value >= 0 for value in deltas
        )
        selected = "prune_r3" if noninferior else "prune_r2"
    removed = (
        frozen["r3_removed_fields"]
        if selected == "prune_r3"
        else list(r2["removed"])
    )
    removed_set = set(removed)
    proposed = set(frozen["accepted_new_removals"])
    verdicts = []
    for row in FEATURES:
        key = row["key"]
        if key in r2["removed"]:
            rule = "retained_experiment41_prune_r2_removal"
        elif key in removed_set:
            rule = "selected_noninferior_r3_removal"
        elif key in proposed:
            rule = "r3_proposed_but_retraining_failed_noninferiority"
        else:
            rule = "survivor_not_selected_for_r3_removal"
        verdicts.append(
            {**row, "verdict": "remove" if key in removed_set else "keep", "rule": rule}
        )
    summary = {
        "schema": "R3_SUMMARY_V1",
        "created_at": _now(),
        "frozen_r3": str(frozen_r3.resolve()),
        "frozen_r3_sha256": _sha256(frozen_r3),
        "retraining_performed": not frozen["no_r3_retrain"],
        "folds": folds,
        "primary": primary,
        "noninferior": noninferior,
        "improvement": improvement,
        "selected_specification": selected,
        "outcome": (
            "r2_is_correlation_conditioned_frontier"
            if frozen["no_r3_retrain"]
            else "r3_selected" if selected == "prune_r3" else "r3_rejected_r2_stands"
        ),
        "hard_stop_no_r4": True,
        "field_verdicts": verdicts,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    path = output_dir / "r3_summary.json"
    _atomic_json(path, summary)
    _atomic_json(
        output_dir / "final_store_v2_specification.json",
        {
            "schema": "FINAL_STORE_V2_SPECIFICATION_V1",
            "created_at": _now(),
            "selected_specification": selected,
            "removed_fields": sorted(removed_set),
            "retained_fields": sorted(set(FEATURE_BY_KEY) - removed_set),
            "source_summary_sha256": _sha256(path),
            "store_rebuild_performed": False,
            "canonical_recipe_changed": False,
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    return path


def _options_summary(
    *, candidate_roots: Mapping[str, Path], r2: Mapping[str, object], output_dir: Path
) -> Path:
    candidates = {}
    for candidate, root in candidate_roots.items():
        folds = _compare_candidate(
            candidate_root=root,
            candidate_name=candidate,
            r2=r2,
            output_dir=output_dir / "analysis" / candidate,
        )
        candidates[candidate] = {
            "folds": folds,
            "standalone_gate": options_gate(folds, "primary_standalone"),
            "mixed_state_gate": options_gate(folds, "secondary_mixed_state"),
        }
    standalone = [
        name for name, value in candidates.items() if value["standalone_gate"]["passed"]
    ]
    if standalone:
        selected = max(
            standalone,
            key=lambda name: float(candidates[name]["standalone_gate"]["mean_delta"]),
        )
        composition = "standalone"
    else:
        mixed = [
            name
            for name, value in candidates.items()
            if value["mixed_state_gate"]["passed"]
        ]
        selected = (
            max(
                mixed,
                key=lambda name: float(
                    candidates[name]["mixed_state_gate"]["mean_delta"]
                ),
            )
            if mixed
            else None
        )
        composition = "parent_patience_plus_candidate_ema" if selected else None
    summary = {
        "schema": "FULL_OPTIONS_PROGRAM_SUMMARY_V1",
        "created_at": _now(),
        "baseline": "experiment41_prune_r2",
        "optional_f2_trim_used": False,
        "candidates": candidates,
        "advancing_candidate": selected,
        "advancing_composition": composition,
        "outcome": "read_arm_earned" if selected else "options_family_parked",
        "multiplicity": "two candidates; higher standalone mean wins if both pass",
        "no_third_subset": True,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    path = output_dir / "options_summary.json"
    _atomic_json(path, summary)
    _atomic_json(
        output_dir / "read_cycle_preparation.json",
        {
            "schema": "OPTIONS_READ_CYCLE_PREPARATION_V1",
            "created_at": _now(),
            "status": "arm_registered_not_executed" if selected else "not_applicable",
            "candidate": selected,
            "composition": composition,
            "required_before_future_read": (
                [
                    "extend identical BVBG/COTAHIST sidecar contract through 2025-06-30",
                    "train full-716-date members on the final store-v2 specification",
                    "register alongside canonical, challenger, and store-v2 parent",
                ]
                if selected
                else []
            ),
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    return path


def run_program(
    *,
    store: Path,
    r2_root: Path,
    cotahist_archives: Sequence[Path],
    bvbg_raw_dir: Path,
    calendar_dir: Path,
    assignments_path: Path,
    context_dir: Path,
    catalogue_path: Path,
    oi_source: Path,
    output_dir: Path,
    options_start: date,
    options_end: date,
    options_build_workers: int = 6,
    parallel_processes: int = 2,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if parallel_processes != MAX_PARALLEL:
        raise ValueError("Experiment 42 is frozen to exactly two training workers")
    output_dir.mkdir(parents=True)
    r2 = load_r2_contract(r2_root)
    manifest_path = output_dir / "program_manifest.json"
    manifest: dict[str, object] = {
        "schema": "R3_FULL_OPTIONS_PROGRAM_V1",
        "status": "running",
        "created_at": _now(),
        "repository_commit": repository_commit(),
        "feature_store": str(store.resolve()),
        "feature_store_identity": feature_store_identity(store),
        "r2_root": str(r2_root.resolve()),
        "r2_specification_sha256": R2_SPEC_SHA256,
        "stages": {
            name: {"status": "pending"}
            for name in ("options_source", "r3_stage_b_prime", "training", "analysis")
        },
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(manifest_path, manifest)
    try:
        manifest["stages"]["options_source"] = {"status": "running"}
        _atomic_json(manifest_path, manifest)
        source_dir = build_full_options_source(
            archives=cotahist_archives,
            bvbg_raw_dir=bvbg_raw_dir,
            calendar_dir=calendar_dir,
            assignments_path=assignments_path,
            context_dir=context_dir,
            catalogue_path=catalogue_path,
            oi_source=oi_source,
            output_dir=output_dir / "sources" / "full_options",
            start=options_start,
            end=options_end,
            workers=options_build_workers,
        )
        full_source = source_dir / "full_options.parquet"
        full_sidecar = materialize_external_sidecar(
            store=store,
            source=full_source,
            output_dir=output_dir / "sidecars" / "opt_full",
            cadence="daily",
            features=OPTIONS_FEATURES,
            source_date_column="source_trade_date",
        )
        iv_sidecar = subset_external_sidecar(
            store=store,
            source_dir=full_sidecar,
            output_dir=output_dir / "sidecars" / "opt_iv",
            features=IV_FEATURES,
        )
        manifest["stages"]["options_source"] = {
            "status": "completed",
            "source": str(source_dir.resolve()),
            "full_sidecar": str(full_sidecar.resolve()),
            "iv_sidecar": str(iv_sidecar.resolve()),
        }
        manifest["stages"]["r3_stage_b_prime"] = {"status": "running"}
        _atomic_json(manifest_path, manifest)
        frozen_r3 = _run_stage_b_isolated(
            store=store, r2=r2, output_dir=output_dir / "r3" / "stage_b_prime"
        )
        manifest["stages"]["r3_stage_b_prime"] = {
            "status": "completed",
            "frozen_r3": str(frozen_r3.resolve()),
            "sha256": _sha256(frozen_r3),
        }
        frozen = _read_json(frozen_r3)
        manifest["stages"]["training"] = {"status": "running"}
        _atomic_json(manifest_path, manifest)
        jobs = []
        r3_runs = None
        if not frozen["no_r3_retrain"]:
            r3_runs = output_dir / "r3" / "runs"
            dynamic, slow = (
                tuple(int(value) for value in part)
                for part in frozen["r3_definition"]
            )
            jobs.extend(
                (
                    store,
                    r3_runs / fold / f"seed_{seed}",
                    seed,
                    fold,
                    dynamic,
                    slow,
                    None,
                )
                for fold in FOLDS
                for seed in ALLOWED_SEEDS
            )
        base_dynamic, base_slow = r2["definition"]
        option_roots = {
            "opt_full": output_dir / "options" / "runs" / "opt_full",
            "opt_iv": output_dir / "options" / "runs" / "opt_iv",
        }
        for candidate, sidecar in (
            ("opt_full", full_sidecar),
            ("opt_iv", iv_sidecar),
        ):
            jobs.extend(
                (
                    store,
                    option_roots[candidate] / fold / f"seed_{seed}",
                    seed,
                    fold,
                    base_dynamic,
                    base_slow,
                    sidecar,
                )
                for fold in FOLDS
                for seed in ALLOWED_SEEDS
            )
        with ProcessPoolExecutor(
            max_workers=parallel_processes, mp_context=mp.get_context("spawn")
        ) as pool:
            futures = [pool.submit(_training_job, *job) for job in jobs]
            for future in as_completed(futures):
                print(future.result(), flush=True)
        manifest["stages"]["training"] = {
            "status": "completed",
            "trajectory_count": len(jobs),
            "maximum_parallel_processes": parallel_processes,
        }
        manifest["stages"]["analysis"] = {"status": "running"}
        _atomic_json(manifest_path, manifest)
        r3_summary = _r3_summary(
            frozen_r3=frozen_r3,
            r2=r2,
            runs=r3_runs,
            output_dir=output_dir / "r3",
        )
        options_summary = _options_summary(
            candidate_roots=option_roots,
            r2=r2,
            output_dir=output_dir / "options",
        )
        manifest["stages"]["analysis"] = {
            "status": "completed",
            "r3_summary": str(r3_summary.resolve()),
            "r3_summary_sha256": _sha256(r3_summary),
            "options_summary": str(options_summary.resolve()),
            "options_summary_sha256": _sha256(options_summary),
        }
        manifest.update({"status": "completed", "completed_at": _now()})
        _atomic_json(manifest_path, manifest)
        inventory = _write_artifact_inventory(output_dir)
        print(f"artifact_inventory={inventory} sha256={_sha256(inventory)}", flush=True)
    except BaseException as error:
        manifest.update(
            {
                "status": "failed",
                "failed_at": _now(),
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            }
        )
        _atomic_json(manifest_path, manifest)
        raise
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run preregistered R3 elimination and full options screens"
    )
    for name in (
        "store", "r2_root", "bvbg_raw_dir", "calendar_dir", "assignments_path",
        "context_dir", "catalogue_path", "oi_source", "output_dir",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--cotahist-archive", type=Path, action="append", required=True)
    parser.add_argument("--options-start", type=date.fromisoformat, required=True)
    parser.add_argument("--options-end", type=date.fromisoformat, required=True)
    parser.add_argument("--options-build-workers", type=int, default=6)
    parser.add_argument("--parallel-processes", type=int, default=2)
    args = parser.parse_args()
    print(
        run_program(
            cotahist_archives=args.cotahist_archive,
            **{
                key: value
                for key, value in vars(args).items()
                if key != "cotahist_archive"
            },
        )
    )


if __name__ == "__main__":
    main()
