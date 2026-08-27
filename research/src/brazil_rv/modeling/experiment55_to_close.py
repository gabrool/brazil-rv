from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from ..execution.experiment54 import (
    _STATE_COLUMNS,
    _allocate_frontier,
    _load_cache_array,
    build_state_events,
)
from ..execution.inputs import causal_rank_scores, load_discovery_prediction_archive
from ..preprocessing.contract import EQUITY_SESSION_MINUTES
from ..preprocessing.to_close_targets import (
    build_to_close_targets,
    to_close_target_identity,
)
from .analyze import load_run_observations
from .contract import ALLOWED_SEEDS, TrainingSpecification
from .data import feature_store_identity
from .engine import EvaluationObservations, assert_observations_aligned
from .hpo_sweep import STORE_V2_DYNAMIC_ZERO, STORE_V2_SLOW_ZERO
from .metrics import primary_validation_score, rank_average_predictions
from .oof_predictions import run_to_close_oof_extension
from .provenance import repository_commit
from .three_fold_sidecar_screen import crossfit_patience_observations
from .train import run_training

FOLDS = ("fold_c", "fold_a", "fold_b")
SEEDS = ALLOWED_SEEDS
MAX_PARALLEL = 2
ADOPTION_GAIN_BPS = 2.0
GUARDRAIL_FLOOR = -0.0005


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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


def _four_head_specification() -> TrainingSpecification:
    base = TrainingSpecification()
    return replace(
        base,
        architecture=replace(base.architecture, output_horizons=4),
        soft_rank_temperature=0.5,
    )


def _verify_experiment54(root: Path) -> dict[str, object]:
    design_path = root / "frozen_design.json"
    result_path = root / "experiment54_result.json"
    audit_path = root / "final_audit.json"
    design, result, audit = map(_read_json, (design_path, result_path, audit_path))
    if (
        audit.get("status") != "passed"
        or result.get("schema") != "EXPERIMENT54_RESULT_V1"
        or result.get("taker_decision") != "VIABLE"
        or any(
            value.get("official_validation_accessed") is not False
            or value.get("test_accessed") is not False
            for value in (design, result, audit)
        )
    ):
        raise ValueError("Experiment 54 source differs from the accepted result")
    return {
        "root": str(root.resolve()),
        "frozen_design_sha256": _sha256(design_path),
        "result_sha256": _sha256(result_path),
        "final_audit_sha256": _sha256(audit_path),
        "taker_frontier_sha256": _sha256(root / "taker_frontier.parquet"),
        "bucket_definitions_sha256": _sha256(root / "bucket_definitions.json"),
        "raw_ohlc_manifest_sha256": _sha256(root / "raw_ohlc" / "manifest.json"),
        "source_experiment52_root": design["source_experiment52_root"],
        "fold_sources": design["fold_sources"],
    }


def freeze_program(
    *,
    experiment54_root: Path,
    base_oof_root: Path,
    target_scale_source: Path,
    preregistration: Path,
    output_dir: Path,
) -> Path:
    experiment54_root = experiment54_root.resolve()
    base_oof_root = base_oof_root.resolve()
    target_scale_source = target_scale_source.resolve()
    preregistration = preregistration.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    source54 = _verify_experiment54(experiment54_root)
    base_oof_audit = _read_json(base_oof_root / "final_audit.json")
    base_oof_result = _read_json(base_oof_root / "result.json")
    if (
        base_oof_audit.get("status") != "passed"
        or base_oof_result.get("status") != "completed"
        or base_oof_result.get("official_validation_accessed") is not False
        or base_oof_result.get("test_accessed") is not False
    ):
        raise ValueError("Base OOF manufacture is not complete and sealed")
    source_design = _read_json(experiment54_root / "frozen_design.json")
    store = Path(str(source_design["store"]["path"])).resolve()
    identity = feature_store_identity(store)
    if identity != source_design["store"]["identity"]:
        raise ValueError("Experiment 55 store differs from Experiment 54")
    output_dir.mkdir(parents=True)
    try:
        target_sidecar = build_to_close_targets(
            store, target_scale_source, output_dir / "target_sidecar"
        )
        target_identity = to_close_target_identity(target_sidecar, identity)
        buckets = _read_json(experiment54_root / "bucket_definitions.json")
        design: dict[str, object] = {
            "schema": "EXPERIMENT55_FROZEN_DESIGN_V1",
            "created_at": _now(),
            "repository_commit": repository_commit(),
            "preregistration": {
                "path": str(preregistration),
                "sha256": _sha256(preregistration),
            },
            "store": {"path": str(store), "identity": identity},
            "experiment54": source54,
            "base_oof": {
                "root": str(base_oof_root),
                "result_sha256": _sha256(base_oof_root / "result.json"),
                "final_audit_sha256": _sha256(base_oof_root / "final_audit.json"),
            },
            "target_scale_source": {
                "path": str(target_scale_source),
                "manifest_sha256": _sha256(target_scale_source / "manifest.json"),
                "target_scale_sha256": _sha256(
                    target_scale_source / "target_scale.npy"
                ),
            },
            "target_sidecar": target_identity,
            "folds": list(FOLDS),
            "seeds": list(SEEDS),
            "trajectory_count": len(FOLDS) * len(SEEDS),
            "maximum_parallel_training_processes": MAX_PARALLEL,
            "model": {
                "incumbent_heads": 3,
                "to_close_readouts": 3,
                "to_close_basis": ["1", "H/405", "sqrt(H/405)"],
                "to_close_zero_initialized": True,
                "objective": "equal_weight_four_head_soft_spearman",
                "temperature": 0.5,
            },
            "gates": {
                "guardrail": "mean delta >= 0 and no fold < -0.0005",
                "economics": (
                    "to-close frontier improves on best 3-head frontier by at "
                    "least 2 NAV bps/day on at least two folds"
                ),
                "adoption": "guardrail AND economics",
                "conditional_oof_extension_trajectories": 50,
            },
            "bucket_definitions": buckets,
            "official_validation_accessed": False,
            "test_accessed": False,
        }
        design["sha256"] = _canonical_sha256(design)
        _atomic_json(output_dir / "frozen_design.json", design)
    except BaseException:
        import shutil

        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return output_dir / "frozen_design.json"


def _validate_design(root: Path) -> dict[str, object]:
    design = _read_json(root / "frozen_design.json")
    digest = design.get("sha256")
    payload = dict(design)
    payload.pop("sha256", None)
    if digest != _canonical_sha256(payload):
        raise ValueError("Experiment 55 frozen design hash differs")
    if design.get("repository_commit") != repository_commit():
        raise ValueError("Repository commit differs from Experiment 55 freeze")
    store = Path(str(design["store"]["path"]))
    if feature_store_identity(store) != design["store"]["identity"]:
        raise ValueError("Experiment 55 store identity differs")
    source54 = Path(str(design["experiment54"]["root"]))
    if _verify_experiment54(source54) != design["experiment54"]:
        raise ValueError("Experiment 54 changed after Experiment 55 freeze")
    to_close_target_identity(Path(str(design["target_sidecar"]["path"])), design["store"]["identity"])
    return design


def _run_path(root: Path, fold: str, seed: int) -> Path:
    return root / "runs" / fold / f"seed_{seed}"


def _completed_run(path: Path, fold: str, seed: int) -> bool:
    manifest_path = path / "run_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = _read_json(manifest_path)
    return (
        manifest.get("status") == "completed"
        and manifest.get("seed") == seed
        and manifest.get("split", {}).get("training") == fold
        and manifest.get("split", {}).get("test_accessed") is False
        and manifest.get("model", {}).get("to_close_head", {}).get("readouts") == 3
        and manifest.get("target_sidecar", {}).get("schema")
        == "EXPERIMENT55_TO_CLOSE_TARGETS_V1"
        and len(list((path / "validation_predictions").glob("epoch_*.npz"))) == 20
    )


def _run_job(store: Path, sidecar: Path, run: Path, fold: str, seed: int) -> str:
    run_training(
        store=store,
        seed=seed,
        selection_window=fold,
        run_dir=run,
        zero_dynamic_channels=STORE_V2_DYNAMIC_ZERO,
        zero_slow_fields=STORE_V2_SLOW_ZERO,
        training_specification=_four_head_specification(),
        target_sidecar_dir=sidecar,
        to_close_head=True,
    )
    return str(run)


def _execute_jobs(root: Path, design: Mapping[str, object], parallel: int) -> None:
    store = Path(str(design["store"]["path"]))
    sidecar = Path(str(design["target_sidecar"]["path"]))
    jobs = []
    for fold in FOLDS:
        for seed in SEEDS:
            run = _run_path(root, fold, seed)
            if _completed_run(run, fold, seed):
                continue
            if run.exists():
                raise RuntimeError(f"Incomplete Experiment 55 run: {run}")
            jobs.append((store, sidecar, run, fold, seed))
    if parallel == 1:
        for job in jobs:
            print(_run_job(*job), flush=True)
        return
    with ProcessPoolExecutor(
        max_workers=parallel, mp_context=mp.get_context("spawn")
    ) as executor:
        futures = [executor.submit(_run_job, *job) for job in jobs]
        for future in as_completed(futures):
            print(future.result(), flush=True)


def _causal_member_ensemble(
    store: Path, members: Sequence[EvaluationObservations]
) -> tuple[EvaluationObservations, np.ndarray]:
    reference = members[0]
    for member in members[1:]:
        assert_observations_aligned(reference, member)
    activity = np.asarray(
        np.load(store / "equity_membership.npy", mmap_mode="r")[reference.date_idx]
        & np.load(store / "equity_data_ready.npy", mmap_mode="r")[reference.date_idx],
        dtype=bool,
    )
    valid = np.broadcast_to(activity[..., None], reference.predictions.shape)
    ranks = rank_average_predictions(
        [member.predictions for member in members], valid
    )
    return replace(reference, predictions=ranks), valid


def _reshape_grid(
    observations: EvaluationObservations, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dates = np.unique(observations.date_idx)
    decisions = np.unique(observations.decision_idx)
    expected = dates.size * decisions.size
    if observations.sample_id.size != expected:
        raise ValueError("Experiment 55 observations are not a complete grid")
    date_position = np.searchsorted(dates, observations.date_idx)
    decision_position = np.searchsorted(decisions, observations.decision_idx)
    shape = (dates.size, decisions.size, *observations.predictions.shape[1:])
    ranks = np.empty(shape, dtype=observations.predictions.dtype)
    mask = np.empty(shape, dtype=bool)
    ranks[date_position, decision_position] = observations.predictions
    mask[date_position, decision_position] = valid
    return causal_rank_scores(ranks, mask), mask, dates, decisions


def _to_close_edge_bps(
    events: Mapping[str, np.ndarray],
    *,
    open_price: np.ndarray,
    close_price: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    day = events["day"].astype(np.int64)
    name = events["name"].astype(np.int64)
    start = events["minute"].astype(np.int64) + 1
    direction = events["direction"].astype(np.float64)
    result = np.full(day.size, np.nan)
    inside = start < EQUITY_SESSION_MINUTES
    selected = np.nonzero(inside)[0]
    d, n, s = day[selected], name[selected], start[selected]
    valid = (
        observed[d, s, n]
        & observed[d, EQUITY_SESSION_MINUTES - 1, n]
        & np.isfinite(open_price[d, s, n])
        & (open_price[d, s, n] > 0.0)
        & np.isfinite(close_price[d, EQUITY_SESSION_MINUTES - 1, n])
        & (close_price[d, EQUITY_SESSION_MINUTES - 1, n] > 0.0)
    )
    chosen = selected[valid]
    result[chosen] = (
        direction[chosen]
        * (
            close_price[d[valid], EQUITY_SESSION_MINUTES - 1, n[valid]]
            / open_price[d[valid], s[valid], n[valid]]
            - 1.0
        )
        * 10_000.0
    )
    return result


def _conditional_table(
    fold: str, dates: Sequence[object], events: Mapping[str, np.ndarray], edge: np.ndarray
) -> pl.DataFrame:
    valid = np.isfinite(edge)
    payload = {name: values[valid] for name, values in events.items() if name in _STATE_COLUMNS}
    payload["day"] = events["day"][valid]
    payload["gross_edge_bps"] = edge[valid]
    payload["measured_cost_bps"] = events["taker_cost_measured_bps"][valid]
    return (
        pl.DataFrame(payload)
        .group_by(list(_STATE_COLUMNS), maintain_order=True)
        .agg(
            pl.len().alias("event_count"),
            pl.col("day").n_unique().alias("date_count"),
            pl.col("gross_edge_bps").mean().alias("mean_gross_edge_bps"),
            pl.col("gross_edge_bps").median().alias("median_gross_edge_bps"),
            (pl.col("gross_edge_bps") > 7.0).mean().alias("fraction_gt_7_bps"),
            (pl.col("gross_edge_bps") > pl.col("measured_cost_bps"))
            .mean()
            .alias("fraction_clears_measured_cost"),
            pl.col("measured_cost_bps").mean().alias("mean_measured_cost_bps"),
        )
        .with_columns(
            pl.lit(fold).alias("fold"),
            pl.lit("to_close").alias("horizon"),
            (pl.col("event_count") / len(dates)).alias("events_per_day"),
        )
    )


def _frontier(
    fold: str, dates: Sequence[object], events: Mapping[str, np.ndarray], edge: np.ndarray
) -> tuple[pl.DataFrame, dict[str, object]]:
    state = events["state_cell_id"].astype(np.int64)
    valid = np.isfinite(edge)
    means = (
        pl.DataFrame({"state": state[valid], "gross": edge[valid]})
        .group_by("state")
        .agg(pl.col("gross").mean().alias("mean"))
    )
    by_state = dict(means.iter_rows())
    expected_gross = np.asarray([by_state.get(int(value), np.nan) for value in state])
    expected_net = expected_gross - events["taker_cost_measured_bps"]
    daily = pl.DataFrame(
        _allocate_frontier(
            events=events,
            expected_net_bps=expected_net,
            eligible=valid & (expected_gross > 7.0),
            dates=dates,
        )
    ).with_columns(pl.lit(fold).alias("fold"), pl.lit("to_close").alias("horizon"))
    return daily, {
        "fold": fold,
        "horizon": "to_close",
        "threshold_bps": 7.0,
        "date_count": daily.height,
        "mean_net_nav_bps_per_day": float(daily["expected_net_nav_bps"].mean()),
        "median_net_nav_bps_per_day": float(daily["expected_net_nav_bps"].median()),
        "mean_allocated_notional_brl_per_day": float(
            daily["allocated_notional_brl"].mean()
        ),
    }


def _analyze(root: Path, design: Mapping[str, object]) -> dict[str, object]:
    store = Path(str(design["store"]["path"]))
    source54_root = Path(str(design["experiment54"]["root"]))
    source52_root = Path(str(design["experiment54"]["source_experiment52_root"]))
    source52_dates = pl.read_parquet(
        source52_root / "market_inputs" / "dates.parquet"
    ).sort("date_idx")
    cache_idx = np.asarray(_load_cache_array(source52_root, "date_idx.npy"))
    cache_position = {int(value): index for index, value in enumerate(cache_idx)}
    close_all = np.load(
        source54_root / "raw_ohlc" / "close_price.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    buckets = design["bucket_definitions"]
    baseline = pl.read_parquet(source54_root / "taker_frontier.parquet").filter(
        pl.col("threshold_bps") == 7.0
    )
    guardrail_rows = []
    capability_rows = []
    secondary_rows = []
    conditional_tables = []
    frontier_daily_tables = []
    frontier_rows = []
    replay_bindings: dict[str, object] = {}
    for fold in FOLDS:
        members = []
        secondary = []
        for seed in SEEDS:
            run = _run_path(root, fold, seed)
            observations, replays = crossfit_patience_observations(
                run, primary_horizon_count=4
            )
            members.append(observations)
            secondary.append(load_run_observations(run, "final_ema_0995"))
            replay_bindings[f"{fold}/seed_{seed}"] = replays
        candidate, candidate_valid = _causal_member_ensemble(store, members)
        secondary_candidate, _ = _causal_member_ensemble(store, secondary)
        source = design["experiment54"]["fold_sources"][fold]
        comparator = load_discovery_prediction_archive(
            Path(source["ensemble_prediction"]["path"]),
            Path(source["prediction_reference"]["path"]),
            Path(source["execution_manifest"]["path"]),
            store,
        )
        ranks, valid_grid, date_idx, _ = _reshape_grid(candidate, candidate_valid)
        if not np.array_equal(date_idx, comparator.date_idx):
            raise ValueError(f"Experiment 55 comparator dates differ: {fold}")
        candidate_primary = primary_validation_score(
            candidate.predictions[..., :3],
            candidate.targets[..., :3],
            candidate.label_mask[..., :3],
            candidate.date_idx,
        )
        comparator_primary = primary_validation_score(
            comparator.ranks.reshape(candidate.predictions[..., :3].shape),
            candidate.targets[..., :3],
            candidate.label_mask[..., :3],
            candidate.date_idx,
        )
        guardrail_rows.append(
            {
                "fold": fold,
                "candidate_three_head_ic": candidate_primary,
                "comparator_three_head_ic": comparator_primary,
                "delta": candidate_primary - comparator_primary,
            }
        )
        to_close_mask = candidate.label_mask[..., 3:4]
        capability_rows.append(
            {
                "fold": fold,
                "session_third": "overall",
                "to_close_ic": primary_validation_score(
                    candidate.predictions[..., 3:4],
                    candidate.targets[..., 3:4],
                    to_close_mask,
                    candidate.date_idx,
                ),
            }
        )
        thirds = ((0, 18, "morning"), (18, 37, "middle"), (37, 55, "late"))
        for start, stop, name in thirds:
            take = (candidate.decision_idx >= start) & (candidate.decision_idx < stop)
            capability_rows.append(
                {
                    "fold": fold,
                    "session_third": name,
                    "to_close_ic": primary_validation_score(
                        candidate.predictions[take, ..., 3:4],
                        candidate.targets[take, ..., 3:4],
                        candidate.label_mask[take, ..., 3:4],
                        candidate.date_idx[take],
                    ),
                }
            )
        secondary_rows.append(
            {
                "fold": fold,
                "final_ema_four_head_ic": primary_validation_score(
                    secondary_candidate.predictions,
                    secondary_candidate.targets,
                    secondary_candidate.label_mask,
                    secondary_candidate.date_idx,
                ),
            }
        )

        positions = np.asarray([cache_position[int(value)] for value in date_idx])
        date_table = source52_dates.filter(pl.col("date_idx").is_in(date_idx)).sort(
            "date_idx"
        )
        dates = tuple(date_table["trade_date"])
        open_price = np.asarray(_load_cache_array(source52_root, "open_price.npy")[positions])
        observed = np.asarray(_load_cache_array(source52_root, "open_observed.npy")[positions])
        close = np.asarray(close_all[positions])
        adv = np.asarray(_load_cache_array(source52_root, "adv20_brl.npy")[positions])
        spread = np.asarray(_load_cache_array(source52_root, "full_spread.npy")[positions])
        sigma = np.asarray(_load_cache_array(source52_root, "sigma_daily.npy")[positions])
        capacity = np.asarray(
            _load_cache_array(source52_root, "minute_notional20_brl.npy")[positions]
        )
        events, _ = build_state_events(
            ranks=ranks,
            valid=valid_grid,
            refresh_minutes=comparator.refresh_minutes,
            adv20_brl=adv,
            full_spread=spread,
            sigma_daily=sigma,
            minute_notional20_brl=capacity,
            buckets=buckets,
        )
        edge = _to_close_edge_bps(
            events, open_price=open_price, close_price=close, observed=observed
        )
        conditional_tables.append(_conditional_table(fold, dates, events, edge))
        daily, summary = _frontier(fold, dates, events, edge)
        frontier_daily_tables.append(daily)
        baseline_fold = baseline.filter(pl.col("fold") == fold)
        best_baseline = float(baseline_fold["mean_net_nav_bps_per_day"].max())
        summary["best_three_head_frontier_nav_bps_per_day"] = best_baseline
        summary["incremental_nav_bps_per_day"] = (
            summary["mean_net_nav_bps_per_day"] - best_baseline
        )
        frontier_rows.append(summary)

    guardrail_deltas = [float(row["delta"]) for row in guardrail_rows]
    guardrail = np.mean(guardrail_deltas) >= 0.0 and min(guardrail_deltas) >= GUARDRAIL_FLOOR
    economics_folds = sum(
        float(row["incremental_nav_bps_per_day"]) >= ADOPTION_GAIN_BPS
        for row in frontier_rows
    )
    economics = economics_folds >= 2
    adopted = bool(guardrail and economics)
    pl.DataFrame(guardrail_rows).write_parquet(root / "three_head_guardrail.parquet")
    pl.DataFrame(capability_rows).write_parquet(root / "to_close_capability.parquet")
    pl.DataFrame(secondary_rows).write_parquet(root / "secondary_readout.parquet")
    pl.concat(conditional_tables, how="diagonal_relaxed").write_parquet(
        root / "to_close_conditional_edges.parquet"
    )
    pl.concat(frontier_daily_tables, how="diagonal_relaxed").write_parquet(
        root / "to_close_frontier_daily.parquet"
    )
    pl.DataFrame(frontier_rows).write_parquet(root / "to_close_frontier.parquet")
    _atomic_json(root / "crossfit_replays.json", replay_bindings)
    return {
        "guardrail_rows": guardrail_rows,
        "capability_rows": capability_rows,
        "frontier_rows": frontier_rows,
        "guardrail_passed": bool(guardrail),
        "economics_folds_passing": int(economics_folds),
        "economics_passed": bool(economics),
        "adopted_for_execution_layer": adopted,
        "deployed_official_recipe_changed": False,
        "morning_nonpositive_late_concentration_fallback_registered": bool(
            all(
                row["to_close_ic"] <= 0
                for row in capability_rows
                if row["session_third"] == "morning"
            )
            and all(
                row["to_close_ic"] > 0
                for row in capability_rows
                if row["session_third"] == "late"
            )
        ),
    }


def _cleanup_checkpoints(root: Path) -> dict[str, object]:
    inventory = []
    for checkpoint in sorted((root / "runs").glob("*/*/checkpoints/epoch_*.pt")):
        inventory.append(
            {"path": str(checkpoint), "sha256": _sha256(checkpoint), "bytes": checkpoint.stat().st_size}
        )
        checkpoint.unlink()
    value = {
        "schema": "EXPERIMENT55_CHECKPOINT_CLEANUP_V1",
        "removed_count": len(inventory),
        "removed": inventory,
        "all_predictions_and_manifests_retained": True,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(root / "checkpoint_cleanup.json", value)
    return value


def run_program(root: Path, parallel: int = MAX_PARALLEL) -> Path:
    root = root.resolve()
    if parallel not in (1, 2):
        raise ValueError("Experiment 55 allows one or two training processes")
    design = _validate_design(root)
    _execute_jobs(root, design, parallel)
    analysis = _analyze(root, design)
    cleanup = _cleanup_checkpoints(root)
    result = {
        "schema": "EXPERIMENT55_RESULT_V1",
        "status": "completed",
        "created_at": _now(),
        **analysis,
        "conditional_oof_extension_required": analysis[
            "adopted_for_execution_layer"
        ],
        "conditional_oof_extension_completed": False,
        "checkpoint_cleanup_removed_count": cleanup["removed_count"],
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(root / "experiment55_result.json", result)
    _atomic_json(
        root / "final_audit.json",
        {
            "schema": "EXPERIMENT55_FINAL_AUDIT_V1",
            "status": "passed_pending_conditional_extension"
            if result["conditional_oof_extension_required"]
            else "passed",
            "result_sha256": _sha256(root / "experiment55_result.json"),
            "all_nine_runs_completed": True,
            "all_180_epochs_completed": True,
            "target_sidecar_verified": True,
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    if result["conditional_oof_extension_required"]:
        extension_result = run_to_close_oof_extension(
            base_oof_root=Path(str(design["base_oof"]["root"])),
            target_sidecar=Path(str(design["target_sidecar"]["path"])),
            output_dir=root / "oof_four_head_extension",
            parallel=parallel,
        )
        result["conditional_oof_extension_completed"] = True
        result["conditional_oof_extension"] = {
            "result": str(extension_result),
            "result_sha256": _sha256(extension_result),
            "final_audit_sha256": _sha256(
                extension_result.parent / "final_audit.json"
            ),
        }
        _atomic_json(root / "experiment55_result.json", result)
        _atomic_json(
            root / "final_audit.json",
            {
                "schema": "EXPERIMENT55_FINAL_AUDIT_V1",
                "status": "passed",
                "result_sha256": _sha256(root / "experiment55_result.json"),
                "all_nine_runs_completed": True,
                "all_180_epochs_completed": True,
                "target_sidecar_verified": True,
                "conditional_oof_extension_completed": True,
                "official_validation_accessed": False,
                "test_accessed": False,
            },
        )
    return root / "experiment55_result.json"


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 55 to-close screen")
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--experiment54-root", type=Path, required=True)
    freeze.add_argument("--base-oof-root", type=Path, required=True)
    freeze.add_argument("--target-scale-source", type=Path, required=True)
    freeze.add_argument("--preregistration", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--parallel", type=int, default=MAX_PARALLEL)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    if args.command == "freeze":
        print(
            freeze_program(
                experiment54_root=args.experiment54_root,
                base_oof_root=args.base_oof_root,
                target_scale_source=args.target_scale_source,
                preregistration=args.preregistration,
                output_dir=args.output_dir,
            )
        )
    else:
        print(run_program(args.root, args.parallel))


if __name__ == "__main__":
    main()
