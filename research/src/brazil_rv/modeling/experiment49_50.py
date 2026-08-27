from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from ..preprocessing.contract import MIN_ACTIVE_EQUITIES
from ..preprocessing.economics_targets import (
    ECONOMICS_HORIZONS,
    build_economics_inputs,
    economics_input_identity,
)
from ..preprocessing.nextgen_targets import nextgen_target_identity
from ..preprocessing.transforms import centered_midranks
from .analyze import compare_observation_ensembles
from .contract import HORIZONS, RUN_OUTPUT_BASE, TrainingSpecification
from .data import feature_store_identity
from .engine import EvaluationObservations, assert_observations_aligned
from .experiment48_nextgen import _r1_specification
from .hpo_sweep import STAGE2_FOLDS, STORE_V2_DYNAMIC_ZERO, STORE_V2_SLOW_ZERO
from .metrics import (
    finite_mean,
    per_date_primary_ic,
    primary_validation_score,
    rank_average_predictions,
    sample_level_spearman_ic,
)
from .official_read import ALL_SEEDS, PATIENCE_RULE, SELECTION_ARTIFACT, _load_reference
from .provenance import repository_commit
from .three_fold_sidecar_screen import crossfit_patience_observations
from .train import run_training
from .trajectory import predictions_for_rule

MAX_PARALLEL = 2
FEE_PER_SIDE = 2.0e-4
DEPLOYMENT_MARGIN = -0.0005
HEAD15_RETENTION_FLOOR = 0.60
PRIMARY_HEADS = len(HORIZONS)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def keep_head15(
    *,
    mid_proxy_retention: Sequence[float],
    net_daily_return: Sequence[float],
    combined_sharpe: Sequence[float],
    comparator_sharpe: Sequence[float],
) -> bool:
    if not all(
        len(values) == 3
        for values in (
            mid_proxy_retention,
            net_daily_return,
            combined_sharpe,
            comparator_sharpe,
        )
    ):
        raise ValueError("Experiment-49 verdict requires exactly three folds")
    return bool(
        sum(value >= HEAD15_RETENTION_FLOOR for value in mid_proxy_retention) >= 2
        and all(value > 0.0 for value in net_daily_return)
        and sum(
            combined >= comparator
            for combined, comparator in zip(
                combined_sharpe, comparator_sharpe, strict=True
            )
        )
        >= 2
    )


def deploy_next_generation(block10_lower_95: float) -> bool:
    return bool(block10_lower_95 >= DEPLOYMENT_MARGIN)


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _selected_epochs(value: object) -> set[int]:
    epochs: set[int] = set()
    if isinstance(value, dict):
        if "selected_epoch" in value:
            epochs.add(int(value["selected_epoch"]))
        for item in value.values():
            epochs.update(_selected_epochs(item))
    elif isinstance(value, list):
        for item in value:
            epochs.update(_selected_epochs(item))
    return epochs


def _run_source_inventory(run: Path, replays: object) -> dict[str, object]:
    epochs = sorted(_selected_epochs(replays))
    files = [
        run / "run_manifest.json",
        run / "history.csv",
        run / "validation_reference.npz",
        run / "trajectory_diagnostics.json",
        *[
            run / "validation_predictions" / f"epoch_{epoch:02d}.npz"
            for epoch in epochs
        ],
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen prediction source incomplete: {missing}")
    return {
        "path": str(run.resolve()),
        "selected_epochs": epochs,
        "files": {str(path.relative_to(run)): _sha256(path) for path in files},
    }


def _validate_experiment48_sources(
    root: Path, store_identity: Mapping[str, object]
) -> dict[str, object]:
    program = _read_json(root / "program_manifest.json")
    result = _read_json(root / "experiment48_result.json")
    design = _read_json(root / "frozen_design.json")
    if (
        program.get("status") != "completed"
        or result.get("official_validation_accessed") is not False
        or result.get("test_accessed") is not False
        or design.get("official_validation_accessed") is not False
        or design.get("test_accessed") is not False
        or result.get("final_next_generation_spec") != "R1_T1.0_three_head_30_60_120"
    ):
        raise ValueError("Experiment-48 source contract differs")
    store_source = design["sources"]["store_v2_and_experiment41"]
    if store_source["feature_store"] != store_identity:
        raise ValueError("Experiment-48 source store differs")
    parent_root = Path(store_source["parent_root"])
    parent_report = Path(store_source["parent_replay_report"])
    parent_replays = _read_json(parent_report)["comparison_metadata"][
        "parent_patience_replays_by_fold"
    ]
    candidate_runs = {}
    parent_runs = {}
    analyses = {}
    for fold in STAGE2_FOLDS:
        analysis_path = (
            root / "part_c" / "analysis" / fold / "primary" / "analysis.json"
        )
        analysis = _read_json(analysis_path)
        replays = analysis["comparison_metadata"]["candidate_patience_replays"]
        analyses[fold] = _artifact(analysis_path)
        for seed in (11, 29, 47):
            key = f"seed_{seed}"
            candidate_runs[f"{fold}/{key}"] = _run_source_inventory(
                root / "part_c" / "runs" / fold / key, replays[key]
            )
            parent_runs[f"{fold}/{key}"] = _run_source_inventory(
                parent_root / fold / key, parent_replays[fold][key]
            )
    target_sidecar = root / "target_sidecar"
    return {
        "root": str(root.resolve()),
        "program_manifest": _artifact(root / "program_manifest.json"),
        "result": _artifact(root / "experiment48_result.json"),
        "frozen_design": _artifact(root / "frozen_design.json"),
        "part_a": _artifact(root / "part_a" / "leg_decomposition.json"),
        "part_c_decision": _artifact(root / "part_c" / "decision.json"),
        "part_c_analyses": analyses,
        "candidate_runs": candidate_runs,
        "parent_root": str(parent_root.resolve()),
        "parent_replay_report": _artifact(parent_report),
        "parent_runs": parent_runs,
        "target_scale_source": design["sources"]["target_scale"],
        "target_sidecar": nextgen_target_identity(target_sidecar, store_identity),
    }


def _validate_experiment45_manifests(root: Path) -> dict[str, object]:
    deployed = _read_json(root / "deployed_recipe.json")
    manifest = _read_json(root / "consolidation_read_manifest.json")
    expected_jobs = {f"arm1_store_v2__seed_{seed}" for seed in ALL_SEEDS}
    if (
        manifest.get("status") != "completed"
        or manifest.get("test_accessed") is not False
        or deployed.get("test_accessed") is not False
        or set(deployed.get("measured_member_job_names", ())) != expected_jobs
    ):
        raise ValueError("Experiment-45 deployed ten-seed source differs")
    runs = {}
    for seed in ALL_SEEDS:
        run = root / "runs" / f"arm1_store_v2__seed_{seed}"
        run_manifest = _read_json(run / "run_manifest.json")
        if (
            run_manifest.get("status") != "completed"
            or run_manifest.get("seed") != seed
            or run_manifest.get("split", {}).get("training") != "official"
            or run_manifest.get("split", {}).get("selection") != "official"
            or run_manifest.get("split", {}).get("test_accessed") is not False
        ):
            raise ValueError(f"Experiment-45 member differs: seed {seed}")
        runs[f"seed_{seed}"] = {
            "path": str(run.resolve()),
            "run_manifest": _artifact(run / "run_manifest.json"),
            "history": _artifact(run / "history.csv"),
            "trajectory_diagnostics": _artifact(run / "trajectory_diagnostics.json"),
        }
    return {
        "root": str(root.resolve()),
        "program_manifest": _artifact(root / "consolidation_read_manifest.json"),
        "frozen_design": _artifact(root / "freeze" / "frozen_design.json"),
        "deployed_recipe": _artifact(root / "deployed_recipe.json"),
        "promotion_decision": _artifact(root / "promotion_decision.json"),
        "runs_pre_event_manifest_binding": runs,
    }


def freeze_programs(
    *,
    store: Path,
    experiment48_root: Path,
    experiment45_root: Path,
    output49: Path,
    output50: Path,
    preregistration49: Path,
    preregistration50: Path,
) -> tuple[Path, Path]:
    paths = (
        store,
        experiment48_root,
        experiment45_root,
        output49,
        output50,
        preregistration49,
        preregistration50,
    )
    (
        store,
        experiment48_root,
        experiment45_root,
        output49,
        output50,
        preregistration49,
        preregistration50,
    ) = (path.resolve() for path in paths)
    if output49.exists() or output50.exists():
        raise FileExistsError("Experiment-49/50 roots must both be new")
    store_identity = feature_store_identity(store)
    source48 = _validate_experiment48_sources(experiment48_root, store_identity)
    source45 = _validate_experiment45_manifests(experiment45_root)
    commit = repository_commit()
    output49.mkdir(parents=True)
    design49 = {
        "schema": "EXPERIMENT49_FROZEN_DESIGN_V1",
        "created_at": _now(),
        "repository_commit": commit,
        "preregistration": _artifact(preregistration49),
        "feature_store": store_identity,
        "experiment48_sources": source48,
        "variants": ["open_to_open", "mid_proxy"],
        "horizons_minutes": list(ECONOMICS_HORIZONS),
        "portfolio": {
            "universe": "causal trailing-20-observed-session ADV top 80",
            "gross": 2.0,
            "fee_per_side_fraction": FEE_PER_SIDE,
            "impact": 0.0,
            "risk_weight_prior_dates": 20,
        },
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(output49 / "frozen_design.json", design49)
    _atomic_json(
        output49 / "program_manifest.json",
        {
            "schema": "EXPERIMENT49_PROGRAM_MANIFEST_V1",
            "status": "frozen",
            "created_at": _now(),
            "repository_commit": commit,
            "frozen_design": str((output49 / "frozen_design.json").resolve()),
            "frozen_design_sha256": _sha256(output49 / "frozen_design.json"),
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    output50.mkdir(parents=True)
    design50 = {
        "schema": "EXPERIMENT50_CONDITIONAL_FROZEN_DESIGN_V1",
        "created_at": _now(),
        "repository_commit": commit,
        "preregistration": _artifact(preregistration50),
        "feature_store": store_identity,
        "experiment49_root": str(output49.resolve()),
        "experiment48_root": str(experiment48_root.resolve()),
        "experiment48_target_sidecar": source48["target_sidecar"],
        "experiment45_sources": source45,
        "conditional_head_count": {"KEEP": 4, "DROP": 3},
        "seeds": list(ALL_SEEDS),
        "specification_three_head": asdict(
            _r1_specification(temperature=1.0, output_horizons=3)
        ),
        "specification_four_head": asdict(
            _r1_specification(temperature=1.0, output_horizons=4)
        ),
        "selection_rule": PATIENCE_RULE,
        "maximum_parallel_training_processes": MAX_PARALLEL,
        "deployment_rule": (
            "paired block-10 95% lower bound candidate minus Experiment-45 "
            "comparator >= -0.0005 inclusive"
        ),
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(output50 / "conditional_frozen_design.json", design50)
    _atomic_json(
        output50 / "program_manifest.json",
        {
            "schema": "EXPERIMENT50_PROGRAM_MANIFEST_V1",
            "status": "conditionally_frozen",
            "created_at": _now(),
            "repository_commit": commit,
            "conditional_design": str(
                (output50 / "conditional_frozen_design.json").resolve()
            ),
            "conditional_design_sha256": _sha256(
                output50 / "conditional_frozen_design.json"
            ),
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    return output49, output50


def _alternative_observations(
    *,
    reference: EvaluationObservations,
    economics_dir: Path,
    variant: str,
    economics_horizon_indices: Sequence[int],
    target_scale: np.ndarray,
    membership: np.ndarray,
    data_ready: np.ndarray,
) -> EvaluationObservations:
    returns_source = np.load(
        economics_dir / f"{variant}_returns.npy", mmap_mode="r", allow_pickle=False
    )
    mask_source = np.load(
        economics_dir / f"{variant}_mask.npy", mmap_mode="r", allow_pickle=False
    )
    shape = (
        reference.predictions.shape[0],
        reference.predictions.shape[1],
        len(economics_horizon_indices),
    )
    raw_returns = np.zeros(shape, dtype=np.float32)
    targets = np.zeros(shape, dtype=np.float32)
    label_mask = np.zeros(shape, dtype=bool)
    for sample, (date_idx, decision_idx) in enumerate(
        zip(reference.date_idx, reference.decision_idx, strict=True)
    ):
        date_value = int(date_idx)
        decision = int(decision_idx)
        sigma = np.asarray(target_scale[date_value], dtype=np.float64)
        active = np.asarray(membership[date_value] & data_ready[date_value], dtype=bool)
        for output_horizon, source_horizon in enumerate(economics_horizon_indices):
            raw = np.asarray(
                returns_source[date_value, :, decision, source_horizon],
                dtype=np.float32,
            )
            valid = (
                active
                & np.asarray(
                    mask_source[date_value, :, decision, source_horizon], dtype=bool
                )
                & np.isfinite(sigma)
                & (sigma > 0.0)
            )
            if int(valid.sum()) < MIN_ACTIVE_EQUITIES:
                continue
            values = raw[valid].astype(np.float64)
            median = float(np.median(values))
            horizon = ECONOMICS_HORIZONS[source_horizon]
            standardized = (values - median) / (sigma[valid] * np.sqrt(horizon))
            raw_returns[sample, valid, output_horizon] = values
            targets[sample, valid, output_horizon] = centered_midranks(standardized)
            label_mask[sample, valid, output_horizon] = True
    return replace(
        reference,
        predictions=np.zeros(shape, dtype=np.float32),
        targets=targets,
        label_mask=label_mask,
        raw_returns=raw_returns,
    )


def _ensemble_head(
    members: Mapping[str, EvaluationObservations],
    head: int,
    target: EvaluationObservations,
) -> np.ndarray:
    reference = next(iter(members.values()))
    for member in members.values():
        assert_observations_aligned(reference, member)
    return rank_average_predictions(
        [member.predictions[..., head : head + 1] for member in members.values()],
        target.label_mask,
    )


def _daily_ic(
    prediction: np.ndarray, target: EvaluationObservations
) -> tuple[float, np.ndarray, np.ndarray]:
    sample_ic = sample_level_spearman_ic(prediction, target.targets, target.label_mask)
    dates, daily = per_date_primary_ic(sample_ic, target.date_idx)
    return finite_mean(daily), dates, daily


def _quarter(value: object) -> str:
    text = str(value)
    year, month = int(text[:4]), int(text[5:7])
    return f"{year}Q{(month - 1) // 3 + 1}"


def _spread_matrix(
    schedule: pl.DataFrame,
    dates: pl.DataFrame,
    equity_index: pl.DataFrame,
) -> np.ndarray:
    by_key = {
        (row["security_id"], row["quarter"]): row["schedule_full_spread_fraction"]
        for row in schedule.iter_rows(named=True)
    }
    security_ids = equity_index.sort("equity_slot")["security_id"].to_list()
    result = np.zeros((dates.height, len(security_ids)), dtype=np.float64)
    for date_idx, trade_date in dates.select("date_idx", "trade_date").iter_rows():
        quarter = _quarter(trade_date)
        result[int(date_idx)] = [
            by_key[(security_id, quarter)] for security_id in security_ids
        ]
    return result


def _terciles(reference: EvaluationObservations, spread: np.ndarray) -> np.ndarray:
    values = np.median(spread[np.unique(reference.date_idx)], axis=0)
    order = np.argsort(values, kind="stable")
    groups = np.empty(values.size, dtype=np.int8)
    for group, slots in enumerate(np.array_split(order, 3), start=1):
        groups[slots] = group
    return groups


def _rank_linear_weights(prediction: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    weights = np.zeros(prediction.shape, dtype=np.float64)
    slots = np.flatnonzero(eligible)
    if slots.size < MIN_ACTIVE_EQUITIES:
        return weights
    values = centered_midranks(np.asarray(prediction[slots], dtype=np.float64))
    gross = float(np.abs(values).sum())
    if gross > 0.0:
        weights[slots] = 2.0 * values / gross
    return weights


def _book(
    *,
    reference: EvaluationObservations,
    prediction: np.ndarray,
    horizon_minutes: int,
    trailing_adv: np.ndarray,
    spread: np.ndarray,
    membership: np.ndarray,
) -> dict[str, object]:
    step = horizon_minutes // 5
    daily: defaultdict[int, dict[str, float]] = defaultdict(
        lambda: {"gross": 0.0, "net": 0.0, "turnover": 0.0, "trades": 0.0}
    )
    for sample, (date_idx, decision_idx) in enumerate(
        zip(reference.date_idx, reference.decision_idx, strict=True)
    ):
        if int(decision_idx) % step:
            continue
        date_value = int(date_idx)
        adv = np.asarray(trailing_adv[date_value], dtype=np.float64)
        liquid = membership[date_value] & np.isfinite(adv) & (adv > 0.0)
        ranked = np.flatnonzero(liquid)
        if ranked.size > 80:
            ranked = ranked[np.argsort(-adv[ranked], kind="stable")[:80]]
        eligible = np.zeros(prediction.shape[1], dtype=bool)
        eligible[ranked] = True
        eligible &= reference.label_mask[sample, :, 0]
        weights = _rank_linear_weights(prediction[sample, :, 0], eligible)
        if not np.any(weights):
            continue
        gross = float(np.dot(weights, reference.raw_returns[sample, :, 0]))
        # Independent horizon holdings enter and exit once. Each side pays the
        # measured half-spread plus the frozen blended fee.
        cost = float(
            np.sum(np.abs(weights) * 2.0 * (spread[date_value] / 2.0 + FEE_PER_SIDE))
        )
        day = daily[date_value]
        day["gross"] += gross
        day["net"] += gross - cost
        day["turnover"] += 2.0 * float(np.abs(weights).sum())
        day["trades"] += 1.0
    dates = np.asarray(sorted(daily), dtype=np.int64)
    gross = np.asarray([daily[value]["gross"] for value in dates])
    net = np.asarray([daily[value]["net"] for value in dates])
    turnover = np.asarray([daily[value]["turnover"] for value in dates])
    trades = np.asarray([daily[value]["trades"] for value in dates])
    sharpe = (
        float(np.sqrt(252.0) * np.mean(net) / np.std(net, ddof=1))
        if net.size > 1 and np.std(net, ddof=1) > 0.0
        else float("nan")
    )
    return {
        "dates": dates,
        "gross": gross,
        "net": net,
        "turnover": turnover,
        "trades": trades,
        "summary": {
            "gross_bps_per_day": float(np.mean(gross) * 10_000.0),
            "net_bps_per_day": float(np.mean(net) * 10_000.0),
            "daily_net_sharpe": sharpe,
            "turnover_per_day": float(np.mean(turnover)),
            "per_rebalance_net_alpha_bps": float(
                np.sum(net) / np.sum(trades) * 10_000.0
            ),
            "date_count": int(dates.size),
            "rebalance_count": int(np.sum(trades)),
        },
    }


def _combined_book(
    left: Mapping[str, object], right: Mapping[str, object]
) -> dict[str, object]:
    if not np.array_equal(left["dates"], right["dates"]):
        raise ValueError("15m and 30m book dates differ")
    left_gross = np.asarray(left["gross"], dtype=np.float64)
    right_gross = np.asarray(right["gross"], dtype=np.float64)
    left_net = np.asarray(left["net"], dtype=np.float64)
    right_net = np.asarray(right["net"], dtype=np.float64)
    weights = np.full((left_net.size, 2), 0.5, dtype=np.float64)
    for index in range(20, left_net.size):
        vols = np.asarray(
            [
                np.std(left_gross[:index], ddof=1),
                np.std(right_gross[:index], ddof=1),
            ]
        )
        if np.all(np.isfinite(vols)) and np.all(vols > 0.0):
            inverse = 1.0 / vols
            weights[index] = inverse / inverse.sum()
    net = weights[:, 0] * left_net + weights[:, 1] * right_net
    gross = weights[:, 0] * left_gross + weights[:, 1] * right_gross
    sharpe = (
        float(np.sqrt(252.0) * np.mean(net) / np.std(net, ddof=1))
        if net.size > 1 and np.std(net, ddof=1) > 0.0
        else float("nan")
    )
    return {
        "weights": weights,
        "gross": gross,
        "net": net,
        "summary": {
            "gross_bps_per_day": float(np.mean(gross) * 10_000.0),
            "net_bps_per_day": float(np.mean(net) * 10_000.0),
            "daily_net_sharpe": sharpe,
        },
    }


def _spread_summary(schedule: pl.DataFrame) -> pl.DataFrame:
    return (
        schedule.group_by("liquidity_group")
        .agg(
            pl.len().alias("security_quarter_count"),
            pl.col("exact_full_spread_fraction")
            .is_not_null()
            .sum()
            .alias("exact_count"),
            pl.col("schedule_full_spread_bps").median().alias("median_bps"),
            pl.col("schedule_full_spread_bps").quantile(0.25).alias("p25_bps"),
            pl.col("schedule_full_spread_bps").quantile(0.75).alias("p75_bps"),
            pl.col("schedule_full_spread_bps").max().alias("max_bps"),
        )
        .sort("liquidity_group")
    )


def run_experiment49(*, store: Path, output49: Path, output50: Path) -> Path:
    store, output49, output50 = (path.resolve() for path in (store, output49, output50))
    manifest_path = output49 / "program_manifest.json"
    manifest = _read_json(manifest_path)
    design = _read_json(output49 / "frozen_design.json")
    if (
        manifest.get("status") != "frozen"
        or manifest.get("repository_commit") != repository_commit()
        or _read_json(output50 / "program_manifest.json").get("status")
        != "conditionally_frozen"
    ):
        raise ValueError("Experiment-49/50 preregistration state differs")
    economics_dir = build_economics_inputs(store, output49 / "economics_inputs")
    economics_identity = economics_input_identity(
        economics_dir, feature_store_identity(store)
    )
    _atomic_json(
        manifest_path,
        {
            **manifest,
            "status": "analysis_running",
            "economics_inputs": economics_identity,
        },
    )
    source48 = design["experiment48_sources"]
    root48 = Path(source48["root"])
    parent_root = Path(source48["parent_root"])
    parent_report = Path(source48["parent_replay_report"]["path"])
    frozen_parent = _read_json(parent_report)["comparison_metadata"][
        "parent_patience_replays_by_fold"
    ]
    target_scale_dir = Path(source48["target_scale_source"]["path"])
    target_scale = np.load(
        target_scale_dir / "target_scale.npy", mmap_mode="r", allow_pickle=False
    )
    membership = np.load(
        store / "equity_membership.npy", mmap_mode="r", allow_pickle=False
    )
    data_ready = np.load(
        store / "equity_data_ready.npy", mmap_mode="r", allow_pickle=False
    )
    trailing_adv = np.load(
        economics_dir / "trailing_adv.npy", mmap_mode="r", allow_pickle=False
    )
    schedule = pl.read_parquet(economics_dir / "roll_schedule.parquet")
    dates_table = pl.read_parquet(store / "date_index.parquet").sort("date_idx")
    equity_index = pl.read_parquet(store / "equity_index.parquet").sort("equity_slot")
    spread = _spread_matrix(schedule, dates_table, equity_index)
    spread_summary = _spread_summary(schedule)
    spread_summary.write_parquet(output49 / "spread_distribution.parquet")
    _atomic_json(
        output49 / "spread_methodology.json",
        {
            "schema": "EXPERIMENT49_ROLL_METHODOLOGY_V1",
            "economics_inputs": economics_identity,
            "distribution_table": _artifact(output49 / "spread_distribution.parquet"),
            "modeled_tick_spread_comparison": (
                "not available as a canonical repository artifact; no value synthesized"
            ),
            "sanity_anchor_top40_median_single_digit_bps": bool(
                spread_summary.filter(pl.col("liquidity_group") == "top_40").item(
                    0, "median_bps"
                )
                < 10.0
            ),
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )

    retention_rows: list[dict[str, object]] = []
    tercile_rows: list[dict[str, object]] = []
    portfolio_rows: list[dict[str, object]] = []
    portfolio_daily_rows: list[dict[str, object]] = []
    books_by_fold: dict[str, dict[str, object]] = {}
    for fold in STAGE2_FOLDS:
        analysis = _read_json(
            root48 / "part_c" / "analysis" / fold / "primary" / "analysis.json"
        )
        candidate_replays = analysis["comparison_metadata"][
            "candidate_patience_replays"
        ]
        candidate = {}
        comparator = {}
        for seed in (11, 29, 47):
            key = f"seed_{seed}"
            candidate[key], _ = crossfit_patience_observations(
                root48 / "part_c" / "runs" / fold / key,
                candidate_replays[key],
                primary_horizon_count=3,
            )
            comparator[key], _ = crossfit_patience_observations(
                parent_root / fold / key, frozen_parent[fold][key]
            )
        candidate_reference = next(iter(candidate.values()))
        comparator_reference = next(iter(comparator.values()))
        if not np.array_equal(
            candidate_reference.sample_id, comparator_reference.sample_id
        ):
            raise ValueError(f"Candidate/comparator samples differ: {fold}")
        alternatives = {
            "candidate": {
                variant: _alternative_observations(
                    reference=candidate_reference,
                    economics_dir=economics_dir,
                    variant=variant,
                    economics_horizon_indices=(1, 2, 3, 0),
                    target_scale=target_scale,
                    membership=membership,
                    data_ready=data_ready,
                )
                for variant in ("open_to_open", "mid_proxy")
            },
            "comparator": {
                variant: _alternative_observations(
                    reference=comparator_reference,
                    economics_dir=economics_dir,
                    variant=variant,
                    economics_horizon_indices=(1, 2, 3),
                    target_scale=target_scale,
                    membership=membership,
                    data_ready=data_ready,
                )
                for variant in ("open_to_open", "mid_proxy")
            },
        }
        model_contracts = (
            ("candidate", candidate, candidate_reference, (30, 60, 120, 15)),
            ("comparator", comparator, comparator_reference, (30, 60, 120)),
        )
        fold_terciles = _terciles(candidate_reference, spread)
        for model_name, members, raw_reference, horizon_names in model_contracts:
            raw_scores = {}
            for head, horizon in enumerate(horizon_names):
                raw_target = replace(
                    raw_reference,
                    predictions=np.zeros_like(
                        raw_reference.predictions[..., head : head + 1]
                    ),
                    targets=raw_reference.targets[..., head : head + 1],
                    label_mask=raw_reference.label_mask[..., head : head + 1],
                    raw_returns=raw_reference.raw_returns[..., head : head + 1],
                )
                raw_prediction = _ensemble_head(members, head, raw_target)
                raw_ic, _, _ = _daily_ic(raw_prediction, raw_target)
                raw_scores[horizon] = raw_ic
                retention_rows.append(
                    {
                        "fold": fold,
                        "model": model_name,
                        "horizon_minutes": horizon,
                        "variant": "raw",
                        "raw_ic": raw_ic,
                        "alternative_ic": raw_ic,
                        "retention_ratio": 1.0,
                    }
                )
                for variant, alternative in alternatives[model_name].items():
                    alt_target = replace(
                        alternative,
                        predictions=np.zeros_like(
                            alternative.predictions[..., head : head + 1]
                        ),
                        targets=alternative.targets[..., head : head + 1],
                        label_mask=alternative.label_mask[..., head : head + 1],
                        raw_returns=alternative.raw_returns[..., head : head + 1],
                    )
                    prediction = _ensemble_head(members, head, alt_target)
                    alt_ic, _, _ = _daily_ic(prediction, alt_target)
                    retention_rows.append(
                        {
                            "fold": fold,
                            "model": model_name,
                            "horizon_minutes": horizon,
                            "variant": variant,
                            "raw_ic": raw_ic,
                            "alternative_ic": alt_ic,
                            "retention_ratio": alt_ic / raw_ic,
                        }
                    )
                targets_by_variant = {"raw": raw_target}
                targets_by_variant.update(
                    {
                        variant: replace(
                            value,
                            predictions=np.zeros_like(
                                value.predictions[..., head : head + 1]
                            ),
                            targets=value.targets[..., head : head + 1],
                            label_mask=value.label_mask[..., head : head + 1],
                            raw_returns=value.raw_returns[..., head : head + 1],
                        )
                        for variant, value in alternatives[model_name].items()
                    }
                )
                for variant, target in targets_by_variant.items():
                    for tercile in (1, 2, 3):
                        mask = target.label_mask.copy()
                        mask[:, fold_terciles != tercile, :] = False
                        localized = replace(target, label_mask=mask)
                        prediction = _ensemble_head(members, head, localized)
                        ic, _, _ = _daily_ic(prediction, localized)
                        tercile_rows.append(
                            {
                                "fold": fold,
                                "model": model_name,
                                "horizon_minutes": horizon,
                                "variant": variant,
                                "spread_tercile": tercile,
                                "ic": ic,
                            }
                        )

        open_candidate = alternatives["candidate"]["open_to_open"]
        open_comparator = alternatives["comparator"]["open_to_open"]
        target15 = replace(
            open_candidate,
            predictions=np.zeros_like(open_candidate.predictions[..., 3:4]),
            targets=open_candidate.targets[..., 3:4],
            label_mask=open_candidate.label_mask[..., 3:4],
            raw_returns=open_candidate.raw_returns[..., 3:4],
        )
        target30 = replace(
            open_comparator,
            predictions=np.zeros_like(open_comparator.predictions[..., 0:1]),
            targets=open_comparator.targets[..., 0:1],
            label_mask=open_comparator.label_mask[..., 0:1],
            raw_returns=open_comparator.raw_returns[..., 0:1],
        )
        book15 = _book(
            reference=target15,
            prediction=_ensemble_head(candidate, 3, target15),
            horizon_minutes=15,
            trailing_adv=trailing_adv,
            spread=spread,
            membership=membership,
        )
        book30 = _book(
            reference=target30,
            prediction=_ensemble_head(comparator, 0, target30),
            horizon_minutes=30,
            trailing_adv=trailing_adv,
            spread=spread,
            membership=membership,
        )
        combined = _combined_book(book15, book30)
        correlation = float(np.corrcoef(book15["net"], book30["net"])[0, 1])
        books_by_fold[fold] = {
            "head15": book15["summary"],
            "comparator30": book30["summary"],
            "combined": combined["summary"],
            "daily_net_pnl_correlation": correlation,
        }
        for name, book in (("head15", book15), ("comparator30", book30)):
            portfolio_rows.append({"fold": fold, "book": name, **book["summary"]})
        portfolio_rows.append({"fold": fold, "book": "combined", **combined["summary"]})
        for index, date_idx in enumerate(book15["dates"]):
            portfolio_daily_rows.extend(
                (
                    {
                        "fold": fold,
                        "date_idx": int(date_idx),
                        "book": "head15",
                        "gross_return": float(book15["gross"][index]),
                        "net_return": float(book15["net"][index]),
                        "turnover": float(book15["turnover"][index]),
                        "risk_weight": float(combined["weights"][index, 0]),
                    },
                    {
                        "fold": fold,
                        "date_idx": int(date_idx),
                        "book": "comparator30",
                        "gross_return": float(book30["gross"][index]),
                        "net_return": float(book30["net"][index]),
                        "turnover": float(book30["turnover"][index]),
                        "risk_weight": float(combined["weights"][index, 1]),
                    },
                    {
                        "fold": fold,
                        "date_idx": int(date_idx),
                        "book": "combined",
                        "gross_return": float(combined["gross"][index]),
                        "net_return": float(combined["net"][index]),
                        "turnover": None,
                        "risk_weight": 1.0,
                    },
                )
            )

    retention = pl.DataFrame(retention_rows)
    retention.write_parquet(output49 / "retention_table.parquet")
    pl.DataFrame(tercile_rows).write_parquet(output49 / "spread_tercile_ic.parquet")
    pl.DataFrame(portfolio_rows).write_parquet(output49 / "portfolio_summary.parquet")
    pl.DataFrame(portfolio_daily_rows).write_parquet(
        output49 / "portfolio_daily.parquet"
    )
    head15_mid = retention.filter(
        (pl.col("model") == "candidate")
        & (pl.col("horizon_minutes") == 15)
        & (pl.col("variant") == "mid_proxy")
    )
    robustness_count = int(
        head15_mid.filter(pl.col("retention_ratio") >= HEAD15_RETENTION_FLOOR).height
    )
    standalone = all(
        float(books_by_fold[fold]["head15"]["net_bps_per_day"]) > 0.0
        for fold in STAGE2_FOLDS
    )
    marginal_count = sum(
        float(books_by_fold[fold]["combined"]["daily_net_sharpe"])
        >= float(books_by_fold[fold]["comparator30"]["daily_net_sharpe"])
        for fold in STAGE2_FOLDS
    )
    keep = keep_head15(
        mid_proxy_retention=head15_mid.sort("fold")["retention_ratio"].to_list(),
        net_daily_return=[
            float(books_by_fold[fold]["head15"]["net_bps_per_day"])
            for fold in STAGE2_FOLDS
        ],
        combined_sharpe=[
            float(books_by_fold[fold]["combined"]["daily_net_sharpe"])
            for fold in STAGE2_FOLDS
        ],
        comparator_sharpe=[
            float(books_by_fold[fold]["comparator30"]["daily_net_sharpe"])
            for fold in STAGE2_FOLDS
        ],
    )
    calibration = retention.filter(
        (pl.col("horizon_minutes").is_in(list(HORIZONS))) & (pl.col("variant") != "raw")
    )
    calibration.write_parquet(output49 / "pretest_retention_calibration.parquet")
    verdict = {
        "schema": "EXPERIMENT49_KEEP_DROP_VERDICT_V1",
        "created_at": _now(),
        "verdict": "KEEP" if keep else "DROP",
        "head_count_for_experiment50": 4 if keep else 3,
        "rule": {
            "bounce_robustness": "15m mid-proxy retention >= 0.60 on >=2 folds",
            "standalone_viability": "15m net expected return > 0 on all folds",
            "marginal_value": "combined Sharpe >= 30m Sharpe on >=2 folds",
        },
        "measurements": {
            "bounce_passing_fold_count": robustness_count,
            "standalone_viability_passed": standalone,
            "marginal_value_passing_fold_count": marginal_count,
            "portfolio_by_fold": books_by_fold,
        },
        "dedicated_15m_option_registered_but_unbuilt": not keep,
        "retention_table": _artifact(output49 / "retention_table.parquet"),
        "pretest_calibration": _artifact(
            output49 / "pretest_retention_calibration.parquet"
        ),
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(output49 / "verdict.json", verdict)
    result = {
        "schema": "EXPERIMENT49_RESULT_V1",
        "completed_at": _now(),
        "verdict": verdict,
        "artifacts": {
            name: _artifact(output49 / name)
            for name in (
                "spread_methodology.json",
                "spread_distribution.parquet",
                "retention_table.parquet",
                "spread_tercile_ic.parquet",
                "portfolio_summary.parquet",
                "portfolio_daily.parquet",
                "pretest_retention_calibration.parquet",
                "verdict.json",
            )
        },
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(output49 / "experiment49_result.json", result)
    _atomic_json(
        manifest_path,
        {
            **manifest,
            "status": "completed",
            "completed_at": result["completed_at"],
            "economics_inputs": economics_identity,
            "result": _artifact(output49 / "experiment49_result.json"),
            "verdict": verdict["verdict"],
        },
    )
    inventory = _root_inventory(output49)
    _atomic_json(output49 / "final_inventory.json", inventory)
    _atomic_json(
        output49 / "final_audit.json",
        {
            "schema": "EXPERIMENT49_FINAL_AUDIT_V1",
            "completed_at": _now(),
            "source_prediction_run_count": 18,
            "retention_row_count": retention.height,
            "spread_schedule_row_count": schedule.height,
            "tercile_row_count": len(tercile_rows),
            "portfolio_summary_row_count": len(portfolio_rows),
            "verdict": verdict["verdict"],
            "inventory": {
                "file_count": inventory["file_count"],
                "total_bytes": inventory["total_bytes"],
                "sha256": _sha256(output49 / "final_inventory.json"),
            },
            "official_validation_accessed": False,
            "test_accessed": False,
            "passed": True,
        },
    )
    return output49


def _run50_job(
    store: Path,
    run_dir: Path,
    seed: int,
    head_count: int,
    target_sidecar: Path | None,
) -> str:
    specification: TrainingSpecification = _r1_specification(
        temperature=1.0, output_horizons=head_count
    )
    run_training(
        store=store,
        seed=seed,
        selection_window="official",
        run_dir=run_dir,
        selection_rule_file=(
            RUN_OUTPUT_BASE / SELECTION_ARTIFACT / "trajectory_selection.json"
        ),
        zero_dynamic_channels=STORE_V2_DYNAMIC_ZERO,
        zero_slow_fields=STORE_V2_SLOW_ZERO,
        training_specification=specification,
        target_sidecar_dir=target_sidecar,
    )
    return str(run_dir)


def _execute50_jobs(
    jobs: Sequence[tuple[Path, Path, int, int, Path | None]], parallel: int
) -> None:
    pending = []
    for job in jobs:
        run = job[1]
        if (run / "run_manifest.json").is_file() and _read_json(
            run / "run_manifest.json"
        ).get("status") == "completed":
            continue
        if run.exists():
            raise RuntimeError(
                f"Incomplete official run requires reviewed repair: {run}"
            )
        pending.append(job)
    if parallel == 1:
        for job in pending:
            print(_run50_job(*job), flush=True)
        return
    with ProcessPoolExecutor(
        max_workers=parallel, mp_context=mp.get_context("spawn")
    ) as executor:
        futures = [executor.submit(_run50_job, *job) for job in pending]
        for future in as_completed(futures):
            print(future.result(), flush=True)


def _official_observation(run: Path, head_count: int) -> EvaluationObservations:
    reference = _load_reference(run)
    predictions = predictions_for_rule(run, PATIENCE_RULE)
    observation = replace(reference, predictions=predictions)
    if head_count == PRIMARY_HEADS:
        return observation
    return replace(
        observation,
        predictions=observation.predictions[..., :PRIMARY_HEADS],
        targets=observation.targets[..., :PRIMARY_HEADS],
        label_mask=observation.label_mask[..., :PRIMARY_HEADS],
        raw_returns=observation.raw_returns[..., :PRIMARY_HEADS],
    )


def _verify_experiment45_predictions(source: Mapping[str, object]) -> dict[str, object]:
    inventory = {}
    for seed in ALL_SEEDS:
        run = Path(source["root"]) / "runs" / f"arm1_store_v2__seed_{seed}"
        diagnostics = _read_json(run / "trajectory_diagnostics.json")
        selected = int(diagnostics["patience3"]["selected_epoch"])
        files = (
            run / "validation_reference.npz",
            run / "validation_predictions" / f"epoch_{selected:02d}.npz",
            run / "checkpoints" / f"epoch_{selected:02d}.pt",
            run / "checkpoints" / "epoch_20.pt",
        )
        inventory[f"seed_{seed}"] = [_artifact(path) for path in files]
    return inventory


def _checkpoint_inventory(root: Path) -> list[dict[str, object]]:
    inventory = []
    for seed in ALL_SEEDS:
        run = root / "runs" / f"seed_{seed}"
        diagnostics = _read_json(run / "trajectory_diagnostics.json")
        selected = int(diagnostics["patience3"]["selected_epoch"])
        for epoch in sorted({selected, 20}):
            inventory.append(_artifact(run / "checkpoints" / f"epoch_{epoch:02d}.pt"))
    return inventory


def _reviewed_cleanup(root: Path) -> dict[str, object]:
    keep = {str(Path(item["path"]).resolve()) for item in _checkpoint_inventory(root)}
    candidates = sorted((root / "runs").rglob("checkpoints/epoch_*.pt"))
    remove = [path for path in candidates if str(path.resolve()) not in keep]
    plan = {
        "schema": "EXPERIMENT50_REVIEWED_CHECKPOINT_CLEANUP_V1",
        "created_at": _now(),
        "root": str(root.resolve()),
        "keep": sorted(keep),
        "remove": [_artifact(path) for path in remove],
        "all_predictions_manifests_histories_analyses_retained": True,
    }
    _atomic_json(root / "cleanup_plan.json", plan)
    for path in remove:
        resolved = path.resolve()
        if not resolved.is_relative_to((root / "runs").resolve()):
            raise ValueError(f"Cleanup path escapes run root: {resolved}")
        path.unlink()
    result = {
        **plan,
        "removed_count": len(remove),
        "removed_bytes": sum(int(item["size_bytes"]) for item in plan["remove"]),
        "retained_checkpoint_inventory": _checkpoint_inventory(root),
    }
    _atomic_json(root / "cleanup_result.json", result)
    return result


def _root_inventory(root: Path) -> dict[str, object]:
    excluded = {root / "final_inventory.json", root / "final_audit.json"}
    files = [
        path for path in root.rglob("*") if path.is_file() and path not in excluded
    ]
    return {
        "schema": "EXPERIMENT50_FINAL_ARTIFACT_INVENTORY_V1",
        "created_at": _now(),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "files": {
            str(path.relative_to(root)): {
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(files)
        },
    }


def run_experiment50(
    *, store: Path, output49: Path, output50: Path, parallel_processes: int = 2
) -> Path:
    if not 1 <= parallel_processes <= MAX_PARALLEL:
        raise ValueError("Experiment 50 allows one or two training processes")
    store, output49, output50 = (path.resolve() for path in (store, output49, output50))
    result49 = _read_json(output49 / "experiment49_result.json")
    verdict = result49["verdict"]["verdict"]
    head_count = 4 if verdict == "KEEP" else 3
    manifest_path = output50 / "program_manifest.json"
    manifest = _read_json(manifest_path)
    design = _read_json(output50 / "conditional_frozen_design.json")
    if (
        manifest.get("status") != "conditionally_frozen"
        or manifest.get("repository_commit") != repository_commit()
        or result49.get("official_validation_accessed") is not False
        or result49.get("test_accessed") is not False
        or int(result49["verdict"]["head_count_for_experiment50"]) != head_count
    ):
        raise ValueError("Experiment-50 conditional realization differs")
    target_sidecar = (
        Path(design["experiment48_root"]) / "target_sidecar"
        if head_count == 4
        else None
    )
    realization = {
        "schema": "EXPERIMENT50_FROZEN_REALIZATION_V1",
        "created_at": _now(),
        "repository_commit": repository_commit(),
        "experiment49_result": _artifact(output49 / "experiment49_result.json"),
        "experiment49_verdict": verdict,
        "head_count": head_count,
        "horizons_minutes": [30, 60, 120, 15] if head_count == 4 else [30, 60, 120],
        "target_sidecar": (
            None
            if target_sidecar is None
            else nextgen_target_identity(target_sidecar, feature_store_identity(store))
        ),
        "seeds": list(ALL_SEEDS),
        "selection_rule": PATIENCE_RULE,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(output50 / "frozen_realization.json", realization)
    comparator_inventory = _verify_experiment45_predictions(
        design["experiment45_sources"]
    )
    jobs = [
        (store, output50 / "runs" / f"seed_{seed}", seed, head_count, target_sidecar)
        for seed in ALL_SEEDS
    ]
    _atomic_json(
        output50 / "validation_access_ledger.json",
        {
            "schema": "VALIDATION_ACCESS_LEDGER_EVENT_V1",
            "status": "running",
            "created_at": _now(),
            "event": 5,
            "experiment": 50,
            "arm": "next_generation_ten_seed",
            "official_monitor_runs": [str(job[1]) for job in jobs],
            "comparator_prediction_inventory": comparator_inventory,
            "official_validation_accessed": True,
            "test_accessed": False,
        },
    )
    _atomic_json(
        manifest_path,
        {
            **manifest,
            "status": "official_training_running",
            "frozen_realization": _artifact(output50 / "frozen_realization.json"),
            "official_validation_accessed": True,
            "test_accessed": False,
        },
    )
    _execute50_jobs(jobs, parallel_processes)
    candidate_full = {
        f"seed_{seed}": replace(
            _load_reference(output50 / "runs" / f"seed_{seed}"),
            predictions=predictions_for_rule(
                output50 / "runs" / f"seed_{seed}", PATIENCE_RULE
            ),
        )
        for seed in ALL_SEEDS
    }
    candidate = {
        key: _official_observation(output50 / "runs" / key, head_count)
        for key in candidate_full
    }
    comparator_root = Path(design["experiment45_sources"]["root"])
    comparator = {
        f"seed_{seed}": _official_observation(
            comparator_root / "runs" / f"arm1_store_v2__seed_{seed}", 3
        )
        for seed in ALL_SEEDS
    }
    analysis_dir = output50 / "analysis" / "nextgen_vs_experiment45"
    compare_observation_ensembles(
        candidate,
        comparator,
        candidate_rule=f"nextgen_{head_count}_head_ten_seed_patience3_raw",
        parent_rule="experiment45_deployed_ten_seed_patience3_raw",
        output_dir=analysis_dir,
        comparison_metadata={
            "event": 5,
            "decision_horizons_minutes": list(HORIZONS),
            "fourth_head_decision_neutral": head_count == 4,
            "official_validation_accessed": True,
            "test_accessed": False,
        },
    )
    analysis = _read_json(analysis_dir / "analysis.json")
    lower = float(analysis["per_date_delta_bootstrap"]["10"]["lower_95"][0])
    deploy = deploy_next_generation(lower)
    official15 = None
    if head_count == 4:
        reference = next(iter(candidate_full.values()))
        prediction15 = rank_average_predictions(
            [value.predictions[..., 3:4] for value in candidate_full.values()],
            reference.label_mask[..., 3:4],
        )
        official15 = primary_validation_score(
            prediction15,
            reference.targets[..., 3:4],
            reference.label_mask[..., 3:4],
            reference.date_idx,
        )
    checkpoint_inventory = _checkpoint_inventory(output50)
    decision = {
        "schema": "EXPERIMENT50_DEPLOYMENT_DECISION_V1",
        "created_at": _now(),
        "decision": "DEPLOY_NEXT_GENERATION" if deploy else "RETAIN_EXPERIMENT45",
        "deployment_rule": ("paired block-10 lower 95% >= -0.0005 inclusive"),
        "paired_block10_lower_95": lower,
        "margin": DEPLOYMENT_MARGIN,
        "noninferiority_passed": deploy,
        "superiority_recorded": lower > 0.0,
        "candidate_primary_ic": analysis["candidate"]["ensemble_ic"],
        "comparator_primary_ic": analysis["parent"]["ensemble_ic"],
        "official_head15_ic_decision_neutral": official15,
        "official_validation_accessed": True,
        "test_accessed": False,
    }
    _atomic_json(output50 / "deployment_decision.json", decision)
    deployed = {
        "schema": "EXPERIMENT50_DEPLOYED_RECIPE_V1",
        "created_at": _now(),
        "recipe": (
            f"next_generation_{head_count}_head_ten_seed"
            if deploy
            else "experiment45_store_v2_ten_seed"
        ),
        "members": (
            [f"seed_{seed}" for seed in ALL_SEEDS]
            if deploy
            else [f"experiment45_seed_{seed}" for seed in ALL_SEEDS]
        ),
        "ensemble": "uniform tie-aware rank average",
        "candidate_selected_and_final_checkpoint_inventory": checkpoint_inventory,
        "deployed_checkpoint_retention": (
            checkpoint_inventory
            if deploy
            else design["experiment45_sources"]["deployed_recipe"]
        ),
        "official_validation_accessed": True,
        "test_accessed": False,
    }
    _atomic_json(output50 / "deployed_recipe.json", deployed)
    cleanup = _reviewed_cleanup(output50)
    result = {
        "schema": "EXPERIMENT50_RESULT_V1",
        "completed_at": _now(),
        "head_count": head_count,
        "trajectory_count": len(ALL_SEEDS),
        "analysis": _artifact(analysis_dir / "analysis.json"),
        "decision": decision,
        "deployed_recipe": deployed["recipe"],
        "cleanup": {
            "removed_count": cleanup["removed_count"],
            "removed_bytes": cleanup["removed_bytes"],
        },
        "official_validation_accessed": True,
        "test_accessed": False,
    }
    _atomic_json(output50 / "experiment50_result.json", result)
    _atomic_json(
        output50 / "validation_access_ledger.json",
        {
            **_read_json(output50 / "validation_access_ledger.json"),
            "status": "completed",
            "completed_at": result["completed_at"],
            "official_validation_accessed": True,
            "test_accessed": False,
        },
    )
    _atomic_json(
        manifest_path,
        {
            **manifest,
            "status": "completed",
            "completed_at": result["completed_at"],
            "frozen_realization": _artifact(output50 / "frozen_realization.json"),
            "result": _artifact(output50 / "experiment50_result.json"),
            "official_validation_accessed": True,
            "test_accessed": False,
        },
    )
    inventory = _root_inventory(output50)
    _atomic_json(output50 / "final_inventory.json", inventory)
    manifests = list((output50 / "runs").rglob("run_manifest.json"))
    if len(manifests) != len(ALL_SEEDS) or any(
        _read_json(path).get("test_accessed") is not False for path in manifests
    ):
        raise ValueError("Experiment-50 run-manifest audit failed")
    final_audit = {
        "schema": "EXPERIMENT50_FINAL_AUDIT_V1",
        "completed_at": _now(),
        "run_manifest_count": len(manifests),
        "history_epoch_count": sum(
            pl.read_csv(path.parent / "history.csv").height for path in manifests
        ),
        "prediction_archive_count": len(
            list((output50 / "runs").rglob("validation_predictions/*.npz"))
        ),
        "retained_checkpoint_count": len(
            list((output50 / "runs").rglob("checkpoints/epoch_*.pt"))
        ),
        "inventory": {
            "file_count": inventory["file_count"],
            "total_bytes": inventory["total_bytes"],
            "sha256": _sha256(output50 / "final_inventory.json"),
        },
        "deployment_decision_sha256": _sha256(output50 / "deployment_decision.json"),
        "deployed_recipe_sha256": _sha256(output50 / "deployed_recipe.json"),
        "all_required_prediction_manifest_analysis_artifacts_retained": True,
        "official_validation_accessed": True,
        "test_accessed": False,
        "passed": True,
    }
    _atomic_json(output50 / "final_audit.json", final_audit)
    return output50


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Experiments 49 and 50")
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--store", type=Path, required=True)
    freeze.add_argument("--experiment48-root", type=Path, required=True)
    freeze.add_argument("--experiment45-root", type=Path, required=True)
    freeze.add_argument("--output49", type=Path, required=True)
    freeze.add_argument("--output50", type=Path, required=True)
    freeze.add_argument("--preregistration49", type=Path, required=True)
    freeze.add_argument("--preregistration50", type=Path, required=True)
    run49 = subparsers.add_parser("run49")
    run49.add_argument("--store", type=Path, required=True)
    run49.add_argument("--output49", type=Path, required=True)
    run49.add_argument("--output50", type=Path, required=True)
    run50 = subparsers.add_parser("run50")
    run50.add_argument("--store", type=Path, required=True)
    run50.add_argument("--output49", type=Path, required=True)
    run50.add_argument("--output50", type=Path, required=True)
    run50.add_argument("--parallel-processes", type=int, default=2)
    arguments = vars(parser.parse_args())
    command = arguments.pop("command")
    if command == "freeze":
        print(freeze_programs(**arguments))
    elif command == "run49":
        print(run_experiment49(**arguments))
    else:
        print(run_experiment50(**arguments))


if __name__ == "__main__":
    main()
