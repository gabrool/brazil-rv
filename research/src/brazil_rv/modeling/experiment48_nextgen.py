from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from ..preprocessing.nextgen_targets import (
    build_nextgen_targets,
    nextgen_target_identity,
    target_scale_source_identity,
)
from .analyze import compare_observation_ensembles, load_run_observations
from .contract import ALLOWED_SEEDS, HORIZONS, TrainingSpecification
from .data import feature_store_identity
from .engine import EvaluationObservations, assert_observations_aligned
from .hpo_sweep import (
    EXPECTED_STORE_IDENTITY_SHA256,
    PRIMARY_READOUT,
    SECONDARY_READOUT,
    STAGE2_FOLDS,
    STORE_V2_DYNAMIC_ZERO,
    STORE_V2_SLOW_ZERO,
    _validate_sources,
    sweep_configurations,
)
from .metrics import (
    finite_mean,
    moving_block_bootstrap,
    per_date_primary_ic,
    rank_average_predictions,
    sample_level_spearman_ic,
)
from .provenance import repository_commit
from .three_fold_sidecar_screen import crossfit_patience_observations
from .train import run_training

EXPECTED_EXPERIMENT47_RESULT_SHA256 = (
    "464c2a213a0953f1ec68eb2869825a9418a3458aaa7dd3262bf197b18a844b1a"
)
EXPECTED_EXPERIMENT47_CONFIRMATION_SHA256 = (
    "5d36b859577f58cd7f4b49edbd3996fe5c0c2e59e4bf6e3f048462babb76fafe"
)
EXPECTED_EXPERIMENT47_AUDIT_POINTER_SHA256 = (
    "3842e5e2b2b8e419b056c087b3962dcba595d306a68a76b8a4f71f51cc3d738e"
)
PART_A_GATE_RULE = "leg-1 IC >= leg-2 IC on at least 2 of 3 folds"
NONINFERIOR_GATE_RULE = "three-fold mean >= 0 AND no fold < -0.0005"
SUPERIOR_GATE_RULE = (
    "three-fold mean >= +0.001, every fold >= 0, and pooled block-5 "
    "and block-10 lower 95% bounds > 0"
)
MAX_PARALLEL = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _jsonable_bootstrap(values: np.ndarray, block: int, seed: int) -> dict[str, object]:
    return {
        key: np.asarray(value).tolist()
        for key, value in moving_block_bootstrap(
            values,
            replications=10_000,
            block_length=block,
            seed=seed,
        ).items()
    }


def _source_run_inventory(
    run_dir: Path,
    *,
    epochs: Sequence[int] = tuple(range(1, 21)),
    include_tail: bool = True,
) -> dict[str, object]:
    files = [
        run_dir / "run_manifest.json",
        run_dir / "history.csv",
        run_dir / "validation_reference.npz",
        run_dir / "trajectory_diagnostics.json",
        *[
            run_dir / "validation_predictions" / f"epoch_{epoch:02d}.npz"
            for epoch in epochs
        ],
    ]
    if include_tail:
        files.append(run_dir / "validation_predictions" / "tail_candidates.npz")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen trajectory archive is incomplete: {missing}")
    return {
        "path": str(run_dir),
        "files": {str(path.relative_to(run_dir)): _sha256(path) for path in files},
    }


def _experiment47_sources(root: Path) -> dict[str, object]:
    required = {
        "experiment47_result.json": EXPECTED_EXPERIMENT47_RESULT_SHA256,
        "stage2/stage2_confirmation.json": EXPECTED_EXPERIMENT47_CONFIRMATION_SHA256,
        "final_audit_pointer.json": EXPECTED_EXPERIMENT47_AUDIT_POINTER_SHA256,
    }
    for relative, expected in required.items():
        if _sha256(root / relative) != expected:
            raise ValueError(f"Experiment-47 source differs: {relative}")
    result = _read_json(root / "experiment47_result.json")
    confirmation = _read_json(root / "stage2" / "stage2_confirmation.json")
    if (
        result.get("official_validation_accessed") is not False
        or result.get("test_accessed") is not False
        or confirmation.get("future_training_recipe_specification") != "R1"
        or confirmation.get("official_validation_accessed") is not False
        or confirmation.get("test_accessed") is not False
    ):
        raise ValueError("Experiment-47 R1 adoption source differs")
    runs: dict[str, object] = {}
    analyses: dict[str, object] = {}
    expected_architecture = json.loads(
        json.dumps(
            asdict(
                {item.config_id: item for item in sweep_configurations()}[
                    "R1"
                ].specification.architecture
            )
        )
    )
    for fold in STAGE2_FOLDS:
        analysis = (
            root / "stage2" / "analysis" / "R1" / fold / "primary" / "analysis.json"
        )
        value = _read_json(analysis)
        replays = value.get("comparison_metadata", {}).get("candidate_patience_replays")
        if not isinstance(replays, dict) or set(replays) != {
            f"seed_{seed}" for seed in ALLOWED_SEEDS
        }:
            raise ValueError(f"Experiment-47 R1 replay source differs: {fold}")
        analyses[fold] = {"path": str(analysis), "sha256": _sha256(analysis)}
        for seed in ALLOWED_SEEDS:
            run = root / "stage2" / "runs" / "R1" / fold / f"seed_{seed}"
            manifest = _read_json(run / "run_manifest.json")
            if (
                manifest.get("status") != "completed"
                or manifest.get("seed") != seed
                or manifest.get("split", {}).get("training") != fold
                or manifest.get("split", {}).get("test_accessed") is not False
                or manifest.get("model", {}).get("architecture")
                != expected_architecture
                or manifest.get("training", {}).get("objective", {}).get("temperature")
                != 0.5
                or manifest.get("equity_input_zeroing", {}).get("dynamic_channels")
                != list(STORE_V2_DYNAMIC_ZERO)
                or manifest.get("equity_input_zeroing", {}).get("slow_fields")
                != list(STORE_V2_SLOW_ZERO)
            ):
                raise ValueError(f"Experiment-47 R1 run differs: {fold}/seed_{seed}")
            runs[f"{fold}/seed_{seed}"] = _source_run_inventory(run)
    return {
        "root": str(root),
        "required_file_sha256": required,
        "analyses": analyses,
        "runs": runs,
    }


def _parent_prediction_sources(
    parent_root: Path, parent_replay_report: Path
) -> dict[str, object]:
    replays = _read_json(parent_replay_report)["comparison_metadata"][
        "parent_patience_replays_by_fold"
    ]
    return {
        f"{fold}/seed_{seed}": _source_run_inventory(
            parent_root / fold / f"seed_{seed}",
            epochs=sorted(
                {int(row["selected_epoch"]) for row in replays[fold][f"seed_{seed}"]}
            ),
            include_tail=False,
        )
        for fold in STAGE2_FOLDS
        for seed in ALLOWED_SEEDS
    }


def _r1_specification(
    *, temperature: float, output_horizons: int = 3
) -> TrainingSpecification:
    r1 = {item.config_id: item for item in sweep_configurations()}["R1"].specification
    return replace(
        r1,
        architecture=replace(r1.architecture, output_horizons=output_horizons),
        soft_rank_temperature=temperature,
    )


def _run_path(root: Path, part: str, fold: str, seed: int) -> Path:
    return root / part / "runs" / fold / f"seed_{seed}"


def _run_job(
    store: Path,
    run_dir: Path,
    fold: str,
    seed: int,
    temperature: float,
    target_sidecar: Path | None,
) -> str:
    run_training(
        store=store,
        seed=seed,
        selection_window=fold,
        run_dir=run_dir,
        zero_dynamic_channels=STORE_V2_DYNAMIC_ZERO,
        zero_slow_fields=STORE_V2_SLOW_ZERO,
        training_specification=_r1_specification(
            temperature=temperature,
            output_horizons=4 if target_sidecar is not None else 3,
        ),
        target_sidecar_dir=target_sidecar,
    )
    return str(run_dir)


def _completed_run(
    run: Path,
    fold: str,
    seed: int,
    temperature: float,
    target_sidecar: Path | None,
) -> bool:
    manifest_path = run / "run_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = _read_json(manifest_path)
    expected = _r1_specification(
        temperature=temperature,
        output_horizons=4 if target_sidecar is not None else 3,
    )
    return (
        manifest.get("status") == "completed"
        and manifest.get("seed") == seed
        and manifest.get("split", {}).get("training") == fold
        and manifest.get("split", {}).get("test_accessed") is False
        and manifest.get("model", {}).get("architecture")
        == json.loads(json.dumps(asdict(expected.architecture)))
        and manifest.get("training", {}).get("objective", {}).get("temperature")
        == temperature
        and (manifest.get("target_sidecar") is not None) == (target_sidecar is not None)
        and len(list((run / "validation_predictions").glob("epoch_*.npz"))) == 20
    )


def _execute_jobs(
    jobs: Sequence[tuple[Path, Path, str, int, float, Path | None]],
    parallel_processes: int,
) -> None:
    pending = []
    for job in jobs:
        _, run, fold, seed, temperature, sidecar = job
        if _completed_run(run, fold, seed, temperature, sidecar):
            continue
        if run.exists():
            raise RuntimeError(f"Incomplete run requires reviewed repair: {run}")
        pending.append(job)
    if parallel_processes == 1:
        for job in pending:
            print(_run_job(*job), flush=True)
        return
    with ProcessPoolExecutor(
        max_workers=parallel_processes, mp_context=mp.get_context("spawn")
    ) as executor:
        futures = [executor.submit(_run_job, *job) for job in pending]
        for future in as_completed(futures):
            print(future.result(), flush=True)


def _slice_primary(observations: EvaluationObservations) -> EvaluationObservations:
    return replace(
        observations,
        predictions=observations.predictions[..., : len(HORIZONS)],
        targets=observations.targets[..., : len(HORIZONS)],
        label_mask=observations.label_mask[..., : len(HORIZONS)],
        raw_returns=observations.raw_returns[..., : len(HORIZONS)],
    )


def _part_a(
    root: Path,
    parent_root: Path,
    parent_replay_report: Path,
    target_sidecar: Path,
) -> dict[str, object]:
    result_path = root / "part_a" / "leg_decomposition.json"
    if result_path.is_file():
        return _read_json(result_path)
    frozen = _read_json(parent_replay_report)["comparison_metadata"][
        "parent_patience_replays_by_fold"
    ]
    leg_targets = np.load(target_sidecar / "leg_targets.npy", mmap_mode="r")
    leg_masks = np.load(target_sidecar / "leg_label_mask.npy", mmap_mode="r")
    fold_rows = []
    daily_rows = []
    for fold_index, fold in enumerate(STAGE2_FOLDS):
        members = []
        replay_rows = {}
        for seed in ALLOWED_SEEDS:
            key = f"seed_{seed}"
            observations, replay = crossfit_patience_observations(
                parent_root / fold / key,
                frozen[fold][key],
            )
            if members:
                assert_observations_aligned(members[0], observations)
            members.append(observations)
            replay_rows[key] = replay
        reference = members[0]
        targets = np.stack(
            [
                leg_targets[int(date), :, int(decision)]
                for date, decision in zip(
                    reference.date_idx, reference.decision_idx, strict=True
                )
            ]
        ).astype(np.float32)
        masks = np.stack(
            [
                leg_masks[int(date), :, int(decision)]
                for date, decision in zip(
                    reference.date_idx, reference.decision_idx, strict=True
                )
            ]
        ).astype(bool)
        daily_by_leg = []
        means = []
        for leg in range(2):
            mask = masks[..., leg : leg + 1]
            predictions = rank_average_predictions(
                [value.predictions[..., 0:1] for value in members], mask
            )
            sample_ic = sample_level_spearman_ic(
                predictions, targets[..., leg : leg + 1], mask
            )
            dates, daily = per_date_primary_ic(sample_ic, reference.date_idx)
            daily_by_leg.append(daily)
            means.append(finite_mean(daily))
        predictions_60 = rank_average_predictions(
            [value.predictions[..., 1:2] for value in members], masks[..., 0:1]
        )
        ic_60 = sample_level_spearman_ic(
            predictions_60, targets[..., 0:1], masks[..., 0:1]
        )
        _, daily_60 = per_date_primary_ic(ic_60, reference.date_idx)
        paired = daily_by_leg[0] - daily_by_leg[1]
        denominator = means[0] + means[1]
        if not np.isfinite(denominator) or denominator == 0.0:
            raise ValueError(f"Early-realization share is undefined: {fold}")
        interval = _jsonable_bootstrap(paired, block=10, seed=20260848 + fold_index)
        fold_rows.append(
            {
                "fold": fold,
                "leg1_ic": means[0],
                "leg2_ic": means[1],
                "leg1_minus_leg2": finite_mean(paired),
                "paired_daily_difference_block10": interval,
                "early_realization_share": means[0] / denominator,
                "head60_against_leg1_ic": finite_mean(daily_60),
                "leg1_at_least_leg2": means[0] >= means[1],
                "parent_patience_replays": replay_rows,
            }
        )
        daily_rows.extend(
            {
                "fold": fold,
                "date_idx": int(date),
                "leg1_ic": float(daily_by_leg[0][index]),
                "leg2_ic": float(daily_by_leg[1][index]),
                "leg1_minus_leg2": float(paired[index]),
                "head60_against_leg1_ic": float(daily_60[index]),
            }
            for index, date in enumerate(dates)
        )
    passed_count = sum(bool(row["leg1_at_least_leg2"]) for row in fold_rows)
    result = {
        "schema": "EXPERIMENT48_PART_A_LEG_DECOMPOSITION_V1",
        "folds": fold_rows,
        "daily_table": str(root / "part_a" / "daily_leg_ic.parquet"),
        "gate_rule": PART_A_GATE_RULE,
        "passing_fold_count": passed_count,
        "gate_passed": passed_count >= 2,
        "claim_replicated_in_house": passed_count >= 2,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(daily_rows).write_parquet(root / "part_a" / "daily_leg_ic.parquet")
    _atomic_json(result_path, result)
    return result


def _comparison(
    candidate: Mapping[str, EvaluationObservations],
    parent: Mapping[str, EvaluationObservations],
    output_dir: Path,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    analysis_path = output_dir / "analysis.json"
    if not analysis_path.is_file():
        compare_observation_ensembles(
            candidate,
            parent,
            candidate_rule=PRIMARY_READOUT,
            parent_rule="frozen_part_b_parent",
            output_dir=output_dir,
            comparison_metadata=metadata,
        )
    value = _read_json(analysis_path)
    return {
        "delta": float(value["candidate_minus_parent_primary_ic"]),
        "analysis": str(analysis_path),
        "daily_delta": str(output_dir / "daily_delta.parquet"),
    }


def _pooled_intervals(paths: Sequence[Path]) -> dict[str, object]:
    values = np.concatenate(
        [
            np.asarray(pl.read_parquet(path)["candidate_minus_parent_ic"])
            for path in paths
        ]
    )
    return {
        str(block): _jsonable_bootstrap(values, block, 48 + block) for block in (5, 10)
    }


def _archived_r1_members(
    experiment47_root: Path, fold: str
) -> tuple[dict[str, EvaluationObservations], dict[str, object]]:
    analysis = _read_json(
        experiment47_root
        / "stage2"
        / "analysis"
        / "R1"
        / fold
        / "primary"
        / "analysis.json"
    )
    frozen = analysis["comparison_metadata"]["candidate_patience_replays"]
    members = {}
    replays = {}
    for seed in ALLOWED_SEEDS:
        key = f"seed_{seed}"
        run = experiment47_root / "stage2" / "runs" / "R1" / fold / key
        members[key], replays[key] = crossfit_patience_observations(run, frozen[key])
    return members, replays


def _fresh_members(
    root: Path,
    part: str,
    fold: str,
    *,
    primary_horizon_count: int | None = None,
) -> tuple[dict[str, EvaluationObservations], dict[str, object]]:
    members = {}
    replays = {}
    for seed in ALLOWED_SEEDS:
        key = f"seed_{seed}"
        members[key], replays[key] = crossfit_patience_observations(
            _run_path(root, part, fold, seed),
            primary_horizon_count=primary_horizon_count,
        )
    return members, replays


def _part_b_analysis(root: Path, experiment47_root: Path) -> dict[str, object]:
    result_path = root / "part_b" / "decision.json"
    if result_path.is_file():
        return _read_json(result_path)
    folds = {}
    daily_paths = []
    for fold in STAGE2_FOLDS:
        candidate, candidate_replays = _fresh_members(root, "part_b", fold)
        parent, parent_replays = _archived_r1_members(experiment47_root, fold)
        base = root / "part_b" / "analysis" / fold
        primary = _comparison(
            candidate,
            parent,
            base / "primary",
            {
                "part": "B",
                "fold": fold,
                "candidate_patience_replays": candidate_replays,
                "parent_patience_replays": parent_replays,
                "official_validation_accessed": False,
                "test_accessed": False,
            },
        )
        secondary_dir = base / "secondary_ema_0995"
        if not (secondary_dir / "analysis.json").is_file():
            compare_observation_ensembles(
                {
                    f"seed_{seed}": load_run_observations(
                        _run_path(root, "part_b", fold, seed), SECONDARY_READOUT
                    )
                    for seed in ALLOWED_SEEDS
                },
                {
                    f"seed_{seed}": load_run_observations(
                        experiment47_root
                        / "stage2"
                        / "runs"
                        / "R1"
                        / fold
                        / f"seed_{seed}",
                        SECONDARY_READOUT,
                    )
                    for seed in ALLOWED_SEEDS
                },
                candidate_rule=SECONDARY_READOUT,
                parent_rule=f"R1_T0.5_{SECONDARY_READOUT}",
                output_dir=secondary_dir,
                comparison_metadata={
                    "decision_eligible": False,
                    "official_validation_accessed": False,
                    "test_accessed": False,
                },
            )
        folds[fold] = {
            "primary": primary,
            "secondary": str(secondary_dir / "analysis.json"),
        }
        daily_paths.append(Path(primary["daily_delta"]))
    deltas = {fold: float(folds[fold]["primary"]["delta"]) for fold in STAGE2_FOLDS}
    mean = float(np.mean(tuple(deltas.values())))
    intervals = _pooled_intervals(daily_paths)
    passed = mean >= 0.0 and all(value >= -0.0005 for value in deltas.values())
    result = {
        "schema": "EXPERIMENT48_PART_B_DECISION_V1",
        "folds": folds,
        "fold_deltas": deltas,
        "three_fold_mean": mean,
        "pooled_daily_delta_bootstrap": intervals,
        "gate_rule": NONINFERIOR_GATE_RULE,
        "gate_passed": passed,
        "winner": "R1_T1.0" if passed else "R1_T0.5",
        "preference_at_parity": (
            "Experiment-47 independently confirmed the positive T=1 effect"
            if passed
            else "temperature effect recorded as base-dependent"
        ),
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(result_path, result)
    return result


def _part_c_measurement(
    root: Path,
    full_members: Mapping[str, Mapping[str, EvaluationObservations]],
) -> dict[str, object]:
    result_path = root / "part_c" / "head15_measurement.json"
    if result_path.is_file():
        return _read_json(result_path)
    folds = {}
    rows = []
    for fold_index, fold in enumerate(STAGE2_FOLDS):
        members = full_members[fold]
        reference = next(iter(members.values()))
        prediction_15 = rank_average_predictions(
            [value.predictions[..., 3:4] for value in members.values()],
            reference.label_mask[..., 3:4],
        )
        prediction_30 = rank_average_predictions(
            [value.predictions[..., 0:1] for value in members.values()],
            reference.label_mask[..., 0:1],
        )
        ic_15 = sample_level_spearman_ic(
            prediction_15,
            reference.targets[..., 3:4],
            reference.label_mask[..., 3:4],
        )
        ic_30 = sample_level_spearman_ic(
            prediction_30,
            reference.targets[..., 0:1],
            reference.label_mask[..., 0:1],
        )
        dates, daily_15 = per_date_primary_ic(ic_15, reference.date_idx)
        _, daily_30 = per_date_primary_ic(ic_30, reference.date_idx)
        tod = {}
        for decision in np.unique(reference.decision_idx):
            selected = reference.decision_idx == decision
            tod[str(int(decision))] = finite_mean(ic_15[selected].ravel())
        folds[fold] = {
            "head15_ic": finite_mean(daily_15),
            "head30_ic": finite_mean(daily_30),
            "head15_block5": _jsonable_bootstrap(
                daily_15, 5, 20260848 + fold_index * 10 + 5
            ),
            "head15_block10": _jsonable_bootstrap(
                daily_15, 10, 20260848 + fold_index * 10 + 10
            ),
            "head15_time_of_day_ic": tod,
        }
        rows.extend(
            {
                "fold": fold,
                "date_idx": int(date),
                "head15_ic": float(daily_15[index]),
                "head30_ic": float(daily_30[index]),
            }
            for index, date in enumerate(dates)
        )
    table = root / "part_c" / "head15_daily_ic.parquet"
    pl.DataFrame(rows).write_parquet(table)
    result = {
        "schema": "EXPERIMENT48_PART_C_HEAD15_MEASUREMENT_V1",
        "folds": folds,
        "daily_table": str(table),
        "calibration_prior": "expected at or above the 30m level; not a gate",
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(result_path, result)
    return result


def _part_c_analysis(
    root: Path,
    experiment47_root: Path,
    part_b: Mapping[str, object],
) -> dict[str, object]:
    result_path = root / "part_c" / "decision.json"
    if result_path.is_file():
        return _read_json(result_path)
    use_t1 = part_b["winner"] == "R1_T1.0"
    folds = {}
    daily_paths = []
    full_by_fold = {}
    for fold in STAGE2_FOLDS:
        full_candidate, candidate_replays = _fresh_members(
            root, "part_c", fold, primary_horizon_count=3
        )
        full_by_fold[fold] = full_candidate
        candidate = {
            key: _slice_primary(value) for key, value in full_candidate.items()
        }
        if use_t1:
            parent, parent_replays = _fresh_members(root, "part_b", fold)
        else:
            parent, parent_replays = _archived_r1_members(experiment47_root, fold)
        primary = _comparison(
            candidate,
            parent,
            root / "part_c" / "analysis" / fold / "primary",
            {
                "part": "C",
                "fold": fold,
                "metric_horizons": list(HORIZONS),
                "candidate_patience_replays": candidate_replays,
                "parent_patience_replays": parent_replays,
                "official_validation_accessed": False,
                "test_accessed": False,
            },
        )
        secondary_dir = root / "part_c" / "analysis" / fold / "secondary_ema_0995"
        if not (secondary_dir / "analysis.json").is_file():
            candidate_secondary = {
                f"seed_{seed}": _slice_primary(
                    load_run_observations(
                        _run_path(root, "part_c", fold, seed), SECONDARY_READOUT
                    )
                )
                for seed in ALLOWED_SEEDS
            }
            if use_t1:
                parent_secondary = {
                    f"seed_{seed}": load_run_observations(
                        _run_path(root, "part_b", fold, seed), SECONDARY_READOUT
                    )
                    for seed in ALLOWED_SEEDS
                }
            else:
                parent_secondary = {
                    f"seed_{seed}": load_run_observations(
                        experiment47_root
                        / "stage2"
                        / "runs"
                        / "R1"
                        / fold
                        / f"seed_{seed}",
                        SECONDARY_READOUT,
                    )
                    for seed in ALLOWED_SEEDS
                }
            compare_observation_ensembles(
                candidate_secondary,
                parent_secondary,
                candidate_rule=SECONDARY_READOUT,
                parent_rule=f"part_b_winner_{SECONDARY_READOUT}",
                output_dir=secondary_dir,
                comparison_metadata={
                    "decision_eligible": False,
                    "official_validation_accessed": False,
                    "test_accessed": False,
                },
            )
        folds[fold] = {
            "primary": primary,
            "secondary": str(secondary_dir / "analysis.json"),
        }
        daily_paths.append(Path(primary["daily_delta"]))
    deltas = {fold: float(folds[fold]["primary"]["delta"]) for fold in STAGE2_FOLDS}
    mean = float(np.mean(tuple(deltas.values())))
    intervals = _pooled_intervals(daily_paths)
    noninferior = mean >= 0.0 and all(value >= -0.0005 for value in deltas.values())
    supported = all(
        float(intervals[str(block)]["lower_95"][0]) > 0.0 for block in (5, 10)
    )
    superior = (
        mean >= 0.001 and all(value >= 0.0 for value in deltas.values()) and supported
    )
    measurement = _part_c_measurement(root, full_by_fold)
    result = {
        "schema": "EXPERIMENT48_PART_C_DECISION_V1",
        "folds": folds,
        "fold_deltas": deltas,
        "three_fold_mean": mean,
        "pooled_daily_delta_bootstrap": intervals,
        "primary_gate_rule": NONINFERIOR_GATE_RULE,
        "primary_gate_passed": noninferior,
        "superiority_rule": SUPERIOR_GATE_RULE,
        "superiority_passed": superior,
        "future_validation_read_worthy": superior,
        "head15_measurement": measurement,
        "dedicated_head15_future_option_not_built": not noninferior,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(result_path, result)
    return result


def run_experiment48(
    *,
    store: Path,
    parent_root: Path,
    parent_replay_report: Path,
    experiment47_root: Path,
    target_scale_source: Path,
    output_dir: Path,
    parallel_processes: int = 2,
    resume: bool = False,
) -> Path:
    if not 1 <= parallel_processes <= MAX_PARALLEL:
        raise ValueError("Experiment 48 allows one or two training processes")
    paths = [
        store,
        parent_root,
        parent_replay_report,
        experiment47_root,
        target_scale_source,
        output_dir,
    ]
    (
        store,
        parent_root,
        parent_replay_report,
        experiment47_root,
        target_scale_source,
        output_dir,
    ) = (path.resolve() for path in paths)
    if output_dir.exists() and not resume:
        raise FileExistsError(output_dir)
    if not output_dir.exists():
        parent_sources = _validate_sources(store, parent_root, parent_replay_report)
        store_identity = feature_store_identity(store)
        if store_identity["metadata_sha256"] != EXPECTED_STORE_IDENTITY_SHA256:
            raise ValueError("Canonical feature store is not store-v2")
        sources = {
            "store_v2_and_experiment41": parent_sources,
            "experiment41_prediction_archives": _parent_prediction_sources(
                parent_root, parent_replay_report
            ),
            "experiment47_r1": _experiment47_sources(experiment47_root),
            "target_scale": target_scale_source_identity(
                target_scale_source, store_identity
            ),
        }
        output_dir.mkdir(parents=True)
        design = {
            "schema": "EXPERIMENT48_FROZEN_DESIGN_V1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repository_commit": repository_commit(),
            "sources": sources,
            "part_a": {
                "folds": list(STAGE2_FOLDS),
                "readout": PRIMARY_READOUT,
                "gate_rule": PART_A_GATE_RULE,
            },
            "part_b": {
                "specification": asdict(
                    _r1_specification(temperature=1.0, output_horizons=3)
                ),
                "seeds": list(ALLOWED_SEEDS),
                "folds": list(STAGE2_FOLDS),
                "trajectory_count": 9,
                "gate_rule": NONINFERIOR_GATE_RULE,
            },
            "part_c": {
                "conditional_on_part_a": True,
                "fourth_horizon_minutes": 15,
                "equal_weight_four_head_loss": True,
                "seeds": list(ALLOWED_SEEDS),
                "folds": list(STAGE2_FOLDS),
                "maximum_trajectory_count": 9,
                "gate_rule": NONINFERIOR_GATE_RULE,
                "superiority_rule": SUPERIOR_GATE_RULE,
            },
            "deployment_changed": False,
            "pool_scored": False,
            "official_validation_accessed": False,
            "test_accessed": False,
        }
        design_path = output_dir / "frozen_design.json"
        _atomic_json(design_path, design)
        manifest = {
            "schema": "EXPERIMENT48_PROGRAM_MANIFEST_V1",
            "status": "frozen",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repository_commit": repository_commit(),
            "frozen_design": str(design_path),
            "frozen_design_sha256": _sha256(design_path),
            "official_validation_accessed": False,
            "test_accessed": False,
        }
        _atomic_json(output_dir / "program_manifest.json", manifest)
    else:
        manifest = _read_json(output_dir / "program_manifest.json")
        if (
            manifest.get("status") == "completed"
            or manifest.get("repository_commit") != repository_commit()
            or manifest.get("official_validation_accessed") is not False
            or manifest.get("test_accessed") is not False
        ):
            raise ValueError("Experiment-48 program root is not eligible for resume")

    target_sidecar = build_nextgen_targets(
        store, target_scale_source, output_dir / "target_sidecar"
    )
    target_identity = nextgen_target_identity(
        target_sidecar, feature_store_identity(store)
    )
    _atomic_json(
        output_dir / "program_manifest.json",
        {**manifest, "status": "part_a_running", "target_sidecar": target_identity},
    )
    part_a = _part_a(output_dir, parent_root, parent_replay_report, target_sidecar)
    part_b_jobs = [
        (
            store,
            _run_path(output_dir, "part_b", fold, seed),
            fold,
            seed,
            1.0,
            None,
        )
        for fold in STAGE2_FOLDS
        for seed in ALLOWED_SEEDS
    ]
    _atomic_json(
        output_dir / "program_manifest.json",
        {
            **manifest,
            "status": "part_b_running",
            "target_sidecar": target_identity,
            "part_a": str(output_dir / "part_a" / "leg_decomposition.json"),
            "part_b_job_count": 9,
        },
    )
    _execute_jobs(part_b_jobs, parallel_processes)
    part_b = _part_b_analysis(output_dir, experiment47_root)
    part_c = None
    if bool(part_a["gate_passed"]):
        temperature = 1.0 if part_b["winner"] == "R1_T1.0" else 0.5
        part_c_jobs = [
            (
                store,
                _run_path(output_dir, "part_c", fold, seed),
                fold,
                seed,
                temperature,
                target_sidecar,
            )
            for fold in STAGE2_FOLDS
            for seed in ALLOWED_SEEDS
        ]
        _atomic_json(
            output_dir / "program_manifest.json",
            {
                **manifest,
                "status": "part_c_running",
                "target_sidecar": target_identity,
                "part_a": str(output_dir / "part_a" / "leg_decomposition.json"),
                "part_b": str(output_dir / "part_b" / "decision.json"),
                "part_c_job_count": 9,
            },
        )
        _execute_jobs(part_c_jobs, parallel_processes)
        part_c = _part_c_analysis(output_dir, experiment47_root, part_b)
    nextgen_spec = (
        f"{part_b['winner']}_four_head_30_60_120_plus_15"
        if part_c is not None and part_c["primary_gate_passed"]
        else f"{part_b['winner']}_three_head_30_60_120"
    )
    all_runs = sorted(
        str(path.parent) for path in output_dir.rglob("run_manifest.json")
    )
    result = {
        "schema": "EXPERIMENT48_RESULT_V1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "part_a": part_a,
        "part_b": part_b,
        "part_c": part_c,
        "part_c_skipped": not bool(part_a["gate_passed"]),
        "trajectory_count": len(all_runs),
        "final_next_generation_spec": nextgen_spec,
        "future_validation_read_worthy": bool(
            part_c is not None and part_c["future_validation_read_worthy"]
        ),
        "deployment_changed": False,
        "pool_scored": False,
        "read_registered": False,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    result_path = output_dir / "experiment48_result.json"
    _atomic_json(result_path, result)
    _atomic_json(
        output_dir / "program_manifest.json",
        {
            **manifest,
            "status": "completed",
            "completed_at": result["completed_at"],
            "target_sidecar": target_identity,
            "part_a": str(output_dir / "part_a" / "leg_decomposition.json"),
            "part_b": str(output_dir / "part_b" / "decision.json"),
            "part_c": (
                None if part_c is None else str(output_dir / "part_c" / "decision.json")
            ),
            "result": str(result_path),
            "trajectory_count": len(all_runs),
        },
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen Experiment-48 next-gen program"
    )
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--parent-replay-report", type=Path, required=True)
    parser.add_argument("--experiment47-root", type=Path, required=True)
    parser.add_argument("--target-scale-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parallel-processes", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    print(run_experiment48(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
