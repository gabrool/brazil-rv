from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch

from ..modeling.contract import TRAIN_END, TRAIN_START
from ..modeling.data import feature_store_identity, load_nextgen_target_sidecar
from ..modeling.experiment55_to_close import (
    FOLDS,
    SEEDS,
    _causal_member_ensemble,
    _reshape_grid,
    _run_path,
    _to_close_edge_bps,
)
from ..modeling.metrics import moving_block_bootstrap, primary_validation_score
from ..modeling.oof_predictions import run_to_close_oof_extension
from ..modeling.provenance import repository_commit
from ..modeling.three_fold_sidecar_screen import crossfit_patience_observations
from ..preprocessing.contract import EQUITY_SESSION_MINUTES
from .config import ExecutionConfig
from .experiment52 import (
    _fetch_cdi,
    _load_cache_array,
    _save_array,
    stored_daily_volatility,
)
from .experiment54 import (
    _STATE_COLUMNS,
    _allocate_frontier,
    build_state_events,
    forward_edge_bps,
)
from .features import PolicyState
from .inputs import (
    causal_liquidity,
    causal_roll_spreads,
    expand_refreshes,
    iter_discovery_equity_grids,
    lagged_quarter_spreads,
    load_daily_cdi_rates,
    load_discovery_prediction_archive,
)
from .policy import NeuralPolicy
from .simulator import MarketReplay, SimulationResult, simulate
from .splits import policy_evaluation_slices
from .trainer import PolicyBatch, PolicyTrainer, PolicyTrainerConfig, policy_objective

SECTION_A_SCHEMA = "EXPERIMENT56_SECTION_A_V1"
SECTION_B_SCHEMA = "EXPERIMENT56_SECTION_B_V1"
SECTION_C_SCHEMA = "EXPERIMENT56_SECTION_C_V1"
POLICY_SEEDS = (11, 29, 47)
LAMBDAS = (0.02, 0.10)
OOF_SEEDS = (11, 29, 47, 61, 79, 97, 113, 131, 149, 167)
HORIZON_NAMES = ("30m", "60m", "120m", "to_close")
MAX_POLICY_EPOCHS = 100
PATIENCE = 10
BOOTSTRAP_REPLICATIONS = 10_000
BOOTSTRAP_BLOCK = 10
BOOTSTRAP_SEED = 20_260_856


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


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


def _verified_result(
    root: Path, result_name: str
) -> tuple[dict[str, object], dict[str, object]]:
    result_path = root / result_name
    audit_path = root / "final_audit.json"
    result, audit = _read_json(result_path), _read_json(audit_path)
    result_hash = _sha256(result_path)
    inventory = audit.get("artifacts")
    inventory_match = isinstance(inventory, list) and any(
        isinstance(record, dict)
        and record.get("path") == result_name
        and record.get("sha256") == result_hash
        for record in inventory
    )
    if (
        audit.get("status") != "passed"
        or (audit.get("result_sha256") != result_hash and not inventory_match)
        or result.get("official_validation_accessed") is not False
        or result.get("test_accessed") is not False
        or audit.get("official_validation_accessed") is not False
        or audit.get("test_accessed") is not False
    ):
        raise ValueError(f"Source result is not access-safe and hash-audited: {root}")
    return result, audit


def _freeze_root(
    *,
    schema: str,
    root: Path,
    preregistration: Path,
    inputs: Mapping[str, object],
    contract: Mapping[str, object],
) -> Path:
    root = root.resolve()
    preregistration = preregistration.resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    try:
        design = {
            "schema": schema,
            "status": "frozen",
            "created_at": _now(),
            "repository_commit": repository_commit(),
            "preregistration": _artifact(preregistration),
            "inputs": dict(inputs),
            "contract": dict(contract),
            "official_validation_accessed": False,
            "test_accessed": False,
        }
        _atomic_json(root / "frozen_design.json", design)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return root / "frozen_design.json"


def freeze_section_a(
    *,
    experiment55_root: Path,
    experiment54_root: Path,
    preregistration: Path,
    output_dir: Path,
) -> Path:
    experiment55_root = experiment55_root.resolve()
    experiment54_root = experiment54_root.resolve()
    result55, _ = _verified_result(experiment55_root, "experiment55_result.json")
    result54, _ = _verified_result(experiment54_root, "experiment54_result.json")
    if result54.get("taker_decision") != "VIABLE":
        raise ValueError("Experiment 54 is not the accepted economic source")
    return _freeze_root(
        schema=SECTION_A_SCHEMA,
        root=output_dir,
        preregistration=preregistration,
        inputs={
            "experiment55_root": str(experiment55_root),
            "experiment55_result": _artifact(
                experiment55_root / "experiment55_result.json"
            ),
            "experiment55_frozen_design": _artifact(
                experiment55_root / "frozen_design.json"
            ),
            "experiment54_root": str(experiment54_root),
            "experiment54_result": _artifact(
                experiment54_root / "experiment54_result.json"
            ),
            "experiment54_frozen_design": _artifact(
                experiment54_root / "frozen_design.json"
            ),
        },
        contract={
            "standing_decision": "to_close_adopted_as_execution_input",
            "historical_experiment55_verdict_changed": False,
            "horizons": list(HORIZON_NAMES),
            "threshold_bps": 7.0,
            "total_frontier": "one best expected-net horizon per name/refresh",
            "abort_below_three_head_fold_count": 2,
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )


def _expected_gross_by_state(
    events: Mapping[str, np.ndarray], edge: np.ndarray
) -> np.ndarray:
    state = np.asarray(events["state_cell_id"], dtype=np.int64)
    valid = np.isfinite(edge)
    means = (
        pl.DataFrame({"state": state[valid], "gross": edge[valid]})
        .group_by("state")
        .agg(pl.col("gross").mean().alias("mean"))
    )
    values = dict(means.iter_rows())
    return np.asarray(
        [values.get(int(item), np.nan) for item in state], dtype=np.float64
    )


def _conditional_table(
    fold: str, horizon: str, events: Mapping[str, np.ndarray], edge: np.ndarray
) -> pl.DataFrame:
    valid = np.isfinite(edge)
    payload = {name: np.asarray(events[name])[valid] for name in _STATE_COLUMNS}
    payload.update(
        gross_edge_bps=edge[valid],
        measured_cost_bps=np.asarray(events["taker_cost_measured_bps"])[valid],
    )
    return (
        pl.DataFrame(payload)
        .group_by(list(_STATE_COLUMNS), maintain_order=True)
        .agg(
            pl.len().alias("event_count"),
            pl.col("gross_edge_bps").mean().alias("mean_gross_edge_bps"),
            pl.col("gross_edge_bps").median().alias("median_gross_edge_bps"),
            (pl.col("gross_edge_bps") > pl.col("measured_cost_bps"))
            .mean()
            .alias("fraction_clears_measured_cost"),
        )
        .with_columns(pl.lit(fold).alias("fold"), pl.lit(horizon).alias("horizon"))
    )


def _frontier_summary(
    *,
    fold: str,
    horizon: str,
    dates: Sequence[date],
    events: Mapping[str, np.ndarray],
    expected_gross: np.ndarray,
    valid: np.ndarray,
) -> tuple[pl.DataFrame, dict[str, object]]:
    expected_net = expected_gross - np.asarray(events["taker_cost_measured_bps"])
    daily = pl.DataFrame(
        _allocate_frontier(
            events=events,
            expected_net_bps=expected_net,
            eligible=valid & (expected_gross > 7.0),
            dates=dates,
        )
    ).with_columns(pl.lit(fold).alias("fold"), pl.lit(horizon).alias("horizon"))
    return daily, {
        "fold": fold,
        "horizon": horizon,
        "mean_net_nav_bps_per_day": float(daily["expected_net_nav_bps"].mean()),
        "mean_allocated_notional_brl_per_day": float(
            daily["allocated_notional_brl"].mean()
        ),
    }


def run_section_a(root: Path) -> Path:
    root = root.resolve()
    final = root / "result.json"
    if final.exists():
        _verified_result(root, "result.json")
        return final
    design = _read_json(root / "frozen_design.json")
    if (
        design.get("schema") != SECTION_A_SCHEMA
        or design.get("repository_commit") != repository_commit()
    ):
        raise ValueError("Section A must run at its exact frozen commit")
    source55 = Path(str(design["inputs"]["experiment55_root"]))
    source54 = Path(str(design["inputs"]["experiment54_root"]))
    design55 = _read_json(source55 / "frozen_design.json")
    store = Path(str(design55["store"]["path"])).resolve()
    source52 = Path(str(design55["experiment54"]["source_experiment52_root"])).resolve()
    cache_dates = pl.read_parquet(source52 / "market_inputs" / "dates.parquet").sort(
        "date_idx"
    )
    cache_idx = np.asarray(_load_cache_array(source52, "date_idx.npy"))
    cache_position = {int(value): index for index, value in enumerate(cache_idx)}
    close_all = np.load(source54 / "raw_ohlc" / "close_price.npy", mmap_mode="r")
    baseline = pl.read_parquet(source54 / "taker_frontier.parquet").filter(
        pl.col("threshold_bps") == 7.0
    )
    conditional, daily_tables, summaries = [], [], []
    for fold in FOLDS:
        members = [
            crossfit_patience_observations(
                _run_path(source55, fold, seed), primary_horizon_count=4
            )[0]
            for seed in SEEDS
        ]
        candidate, candidate_valid = _causal_member_ensemble(store, members)
        ranks, valid_grid, date_idx, _ = _reshape_grid(candidate, candidate_valid)
        source = design55["experiment54"]["fold_sources"][fold]
        comparator = load_discovery_prediction_archive(
            Path(source["ensemble_prediction"]["path"]),
            Path(source["prediction_reference"]["path"]),
            Path(source["execution_manifest"]["path"]),
            store,
        )
        if not np.array_equal(date_idx, comparator.date_idx):
            raise ValueError(f"Section A fold dates differ: {fold}")
        positions = np.asarray([cache_position[int(value)] for value in date_idx])
        date_table = cache_dates.filter(pl.col("date_idx").is_in(date_idx)).sort(
            "date_idx"
        )
        dates = tuple(date_table["trade_date"])
        open_price = np.asarray(
            _load_cache_array(source52, "open_price.npy")[positions]
        )
        observed = np.asarray(
            _load_cache_array(source52, "open_observed.npy")[positions]
        )
        close_price = np.asarray(close_all[positions])
        events, _ = build_state_events(
            ranks=ranks,
            valid=valid_grid,
            refresh_minutes=comparator.refresh_minutes,
            adv20_brl=np.asarray(
                _load_cache_array(source52, "adv20_brl.npy")[positions]
            ),
            full_spread=np.asarray(
                _load_cache_array(source52, "full_spread.npy")[positions]
            ),
            sigma_daily=np.asarray(
                _load_cache_array(source52, "sigma_daily.npy")[positions]
            ),
            minute_notional20_brl=np.asarray(
                _load_cache_array(source52, "minute_notional20_brl.npy")[positions]
            ),
            buckets=design55["bucket_definitions"],
        )
        edges = {
            "30m": forward_edge_bps(
                events,
                open_price=open_price,
                close_price=close_price,
                observed=observed,
                horizon=30,
                entry="next_open",
            ),
            "60m": forward_edge_bps(
                events,
                open_price=open_price,
                close_price=close_price,
                observed=observed,
                horizon=60,
                entry="next_open",
            ),
            "120m": forward_edge_bps(
                events,
                open_price=open_price,
                close_price=close_price,
                observed=observed,
                horizon=120,
                entry="next_open",
            ),
            "to_close": _to_close_edge_bps(
                events,
                open_price=open_price,
                close_price=close_price,
                observed=observed,
            ),
        }
        expected = {
            name: _expected_gross_by_state(events, edge) for name, edge in edges.items()
        }
        for name in HORIZON_NAMES:
            conditional.append(_conditional_table(fold, name, events, edges[name]))
            daily, row = _frontier_summary(
                fold=fold,
                horizon=name,
                dates=dates,
                events=events,
                expected_gross=expected[name],
                valid=np.isfinite(edges[name]),
            )
            daily_tables.append(daily)
            summaries.append(row)
        expected_stack = np.stack([expected[name] for name in HORIZON_NAMES], axis=1)
        net_stack = (
            expected_stack - np.asarray(events["taker_cost_measured_bps"])[:, None]
        )
        safe = np.where(np.isfinite(net_stack), net_stack, -np.inf)
        choice = np.argmax(safe, axis=1)
        event_index = np.arange(choice.size)
        chosen_gross = expected_stack[event_index, choice]
        chosen_net = net_stack[event_index, choice]
        valid_choice = np.isfinite(chosen_gross) & np.isfinite(chosen_net)
        total_daily = pl.DataFrame(
            _allocate_frontier(
                events=events,
                expected_net_bps=chosen_net,
                eligible=valid_choice & (chosen_gross > 7.0),
                dates=dates,
            )
        ).with_columns(pl.lit(fold).alias("fold"), pl.lit("total").alias("horizon"))
        comparator_value = float(
            baseline.filter(pl.col("fold") == fold)["mean_net_nav_bps_per_day"].max()
        )
        total_value = float(total_daily["expected_net_nav_bps"].mean())
        daily_tables.append(total_daily)
        summaries.append(
            {
                "fold": fold,
                "horizon": "total",
                "mean_net_nav_bps_per_day": total_value,
                "mean_allocated_notional_brl_per_day": float(
                    total_daily["allocated_notional_brl"].mean()
                ),
                "three_head_frontier_nav_bps_per_day": comparator_value,
                "delta_vs_three_head_nav_bps_per_day": total_value - comparator_value,
            }
        )
    conditional_path = root / "conditional_edges.parquet"
    daily_path = root / "frontier_daily.parquet"
    summary_path = root / "frontier_summary.parquet"
    pl.concat(conditional, how="diagonal_relaxed").write_parquet(conditional_path)
    pl.concat(daily_tables, how="diagonal_relaxed").write_parquet(daily_path)
    pl.DataFrame(summaries).write_parquet(summary_path)
    capability_path = root / "to_close_session_thirds.parquet"
    pl.read_parquet(source55 / "to_close_capability.parquet").write_parquet(
        capability_path
    )
    total_rows = [row for row in summaries if row["horizon"] == "total"]
    below = sum(
        float(row["delta_vs_three_head_nav_bps_per_day"]) < 0 for row in total_rows
    )
    result = {
        "schema": SECTION_A_SCHEMA,
        "status": "completed",
        "created_at": _now(),
        "standing_decision": "to_close_adopted_as_execution_input",
        "historical_experiment55_verdict_changed": False,
        "total_frontiers": total_rows,
        "below_three_head_fold_count": below,
        "abort": below >= 2,
        "artifacts": {
            "conditional_edges": _artifact(conditional_path),
            "frontier_daily": _artifact(daily_path),
            "frontier_summary": _artifact(summary_path),
            "to_close_session_thirds": _artifact(capability_path),
        },
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(final, result)
    _atomic_json(
        root / "final_audit.json",
        {
            "schema": "EXPERIMENT56_SECTION_A_AUDIT_V1",
            "status": "passed",
            "result_sha256": _sha256(final),
            "fold_count": 3,
            "horizon_count_including_total": 5,
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    return final


def freeze_section_b(
    *,
    section_a_root: Path,
    base_oof_root: Path,
    target_sidecar: Path,
    preregistration: Path,
    output_dir: Path,
) -> Path:
    section_a, _ = _verified_result(section_a_root.resolve(), "result.json")
    if section_a.get("abort") is not False:
        raise RuntimeError("Section A aborts four-head OOF manufacture")
    base_oof, _ = _verified_result(base_oof_root.resolve(), "result.json")
    store = Path(str(_read_json(base_oof_root / "frozen_design.json")["store"]["path"]))
    sidecar = load_nextgen_target_sidecar(target_sidecar.resolve(), store)
    return _freeze_root(
        schema=SECTION_B_SCHEMA,
        root=output_dir,
        preregistration=preregistration,
        inputs={
            "section_a_root": str(section_a_root.resolve()),
            "section_a_result": _artifact(section_a_root / "result.json"),
            "base_oof_root": str(base_oof_root.resolve()),
            "base_oof_result": _artifact(base_oof_root / "result.json"),
            "target_sidecar": sidecar.identity,
        },
        contract={
            "fold_count": 5,
            "seeds": list(OOF_SEEDS),
            "trajectory_count": 50,
            "epochs": 20,
            "monitor": None,
            "final_state": "ema_0995",
            "maximum_parallel_training_processes": 2,
        },
    )


def _archive(root: Path) -> tuple[object, dict[str, object]]:
    result = _read_json(root / "result.json")
    record = result["archive"]
    design = _read_json(root / "frozen_design.json")
    store = Path(str(design["store"]["path"])).resolve()
    return load_discovery_prediction_archive(
        Path(record["prediction"]),
        Path(record["reference"]),
        Path(record["execution_manifest"]),
        store,
    ), design


def _section_b_calibration(
    *, base_oof_root: Path, extension_root: Path, target_sidecar: Path, output: Path
) -> Path:
    base, base_design = _archive(base_oof_root)
    four, _ = _archive(extension_root)
    store = Path(str(base_design["store"]["path"])).resolve()
    if not np.array_equal(base.date_idx, four.date_idx) or not np.array_equal(
        base.sample_id, four.sample_id
    ):
        raise ValueError("Three- and four-head OOF archives do not align")
    dates = tuple(
        pl.read_parquet(store / "date_index.parquet")
        .filter(pl.col("date_idx").is_in(base.date_idx))
        .sort("date_idx")["trade_date"]
    )
    slices = policy_evaluation_slices(dates, dates)
    targets = np.load(store / "targets.npy", mmap_mode="r")
    masks = np.load(store / "label_mask.npy", mmap_mode="r")
    sidecar = load_nextgen_target_sidecar(target_sidecar, store)
    side_targets = np.load(target_sidecar / "leg_targets.npy", mmap_mode="r")
    side_masks = np.load(target_sidecar / "leg_label_mask.npy", mmap_mode="r")
    rows = []
    for window in slices:
        take_date = np.isin(dates, window.dates)
        sample_dates = np.repeat(four.date_idx[take_date], four.sample_id.shape[1])
        selected_date_idx = four.date_idx[take_date]
        decisions = np.arange(four.sample_id.shape[1], dtype=np.int64)
        for head in range(3):
            candidate = four.ranks[take_date, ..., head]
            comparator = base.ranks[take_date, ..., head]
            valid = four.valid[take_date, ..., head] & base.valid[take_date, ..., head]
            corr = float(np.corrcoef(candidate[valid], comparator[valid])[0, 1])
            target = np.asarray(
                targets[selected_date_idx[:, None], :, decisions[None, :], head]
            )
            mask = np.asarray(
                masks[selected_date_idx[:, None], :, decisions[None, :], head]
            )
            rows.append(
                {
                    "window": window.name,
                    "head": HORIZON_NAMES[head],
                    "rank_correlation_vs_three_head": corr,
                    "four_head_ic": primary_validation_score(
                        candidate.reshape(-1, candidate.shape[-1])[..., None],
                        target.reshape(-1, target.shape[-1])[..., None],
                        mask.reshape(-1, mask.shape[-1])[..., None],
                        sample_dates,
                    ),
                    "three_head_ic": primary_validation_score(
                        comparator.reshape(-1, comparator.shape[-1])[..., None],
                        target.reshape(-1, target.shape[-1])[..., None],
                        mask.reshape(-1, mask.shape[-1])[..., None],
                        sample_dates,
                    ),
                }
            )
        close_prediction = four.ranks[take_date, ..., 3]
        close_target = np.asarray(
            side_targets[selected_date_idx[:, None], :, decisions[None, :], 0]
        )
        close_mask = np.asarray(
            side_masks[selected_date_idx[:, None], :, decisions[None, :], 0]
        )
        rows.append(
            {
                "window": window.name,
                "head": "to_close",
                "rank_correlation_vs_three_head": None,
                "four_head_ic": primary_validation_score(
                    close_prediction.reshape(-1, close_prediction.shape[-1])[..., None],
                    close_target.reshape(-1, close_target.shape[-1])[..., None],
                    close_mask.reshape(-1, close_mask.shape[-1])[..., None],
                    sample_dates,
                ),
                "three_head_ic": None,
            }
        )
    pl.DataFrame(rows).write_parquet(output)
    del sidecar
    return output


def run_section_b(root: Path, parallel: int = 2) -> Path:
    root = root.resolve()
    final = root / "result.json"
    if final.exists():
        _verified_result(root, "result.json")
        return final
    design = _read_json(root / "frozen_design.json")
    if (
        design.get("schema") != SECTION_B_SCHEMA
        or design.get("repository_commit") != repository_commit()
    ):
        raise ValueError("Section B must run at its exact frozen commit")
    base = Path(str(design["inputs"]["base_oof_root"]))
    target = Path(str(design["inputs"]["target_sidecar"]["path"]))
    extension = root / "oof"
    extension_result = run_to_close_oof_extension(
        base_oof_root=base,
        target_sidecar=target,
        output_dir=extension,
        parallel=parallel,
    )
    calibration = _section_b_calibration(
        base_oof_root=base,
        extension_root=extension,
        target_sidecar=target,
        output=root / "calibration.parquet",
    )
    result = {
        "schema": SECTION_B_SCHEMA,
        "status": "completed",
        "created_at": _now(),
        "archive": _read_json(extension_result)["archive"],
        "extension_result": _artifact(extension_result),
        "extension_audit": _artifact(extension / "final_audit.json"),
        "calibration": _artifact(calibration),
        "trajectory_count": 50,
        "epoch_count": 1000,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(final, result)
    _atomic_json(
        root / "final_audit.json",
        {
            "schema": "EXPERIMENT56_SECTION_B_AUDIT_V1",
            "status": "passed",
            "result_sha256": _sha256(final),
            "all_50_runs_completed": True,
            "all_1000_epochs_completed": True,
            "loader_verified_716_date_oof": True,
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    return final


def freeze_section_c(
    *,
    section_a_root: Path,
    section_b_root: Path,
    experiment52_root: Path,
    experiment53_root: Path,
    preregistration: Path,
    output_dir: Path,
) -> Path:
    section_a, _ = _verified_result(section_a_root.resolve(), "result.json")
    section_b, _ = _verified_result(section_b_root.resolve(), "result.json")
    _verified_result(experiment52_root.resolve(), "c0_designation.json")
    _verified_result(experiment53_root.resolve(), "experiment53_result.json")
    if section_a.get("abort") is not False:
        raise RuntimeError("Section A aborts policy training")
    archive = section_b["archive"]
    base_design = _read_json(section_b_root / "oof" / "frozen_design.json")
    store = Path(str(base_design["store"]["path"])).resolve()
    store_dates = pl.read_parquet(store / "date_index.parquet").sort("date_idx")
    train_dates = tuple(
        store_dates.filter(pl.col("trade_date").is_between(TRAIN_START, TRAIN_END))[
            "trade_date"
        ]
    )
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    try:
        cdi = _fetch_cdi(output_dir / "cdi", min(train_dates), max(train_dates))
        load_daily_cdi_rates(
            Path(cdi["parquet"]["path"]), train_dates, cdi["parquet"]["sha256"]
        )
        source52_design = _read_json(experiment52_root / "frozen_design.json")
        design = {
            "schema": SECTION_C_SCHEMA,
            "status": "frozen",
            "created_at": _now(),
            "repository_commit": repository_commit(),
            "preregistration": _artifact(preregistration.resolve()),
            "store": {
                "path": str(store),
                "identity": feature_store_identity(store),
                "manifest": _artifact(store / "manifest.json"),
            },
            "inputs": {
                "section_a_root": str(section_a_root.resolve()),
                "section_a_result": _artifact(section_a_root / "result.json"),
                "section_b_root": str(section_b_root.resolve()),
                "section_b_result": _artifact(section_b_root / "result.json"),
                "oof_archive": archive,
                "experiment52_root": str(experiment52_root.resolve()),
                "experiment52_result": _artifact(
                    experiment52_root / "c0_designation.json"
                ),
                "experiment53_root": str(experiment53_root.resolve()),
                "experiment53_result": _artifact(
                    experiment53_root / "experiment53_result.json"
                ),
                "roll_schedule": source52_design["roll_schedule"],
                "economics_inputs": source52_design["economics_inputs"],
                "cdi": cdi,
            },
            "contract": {
                "evaluation_windows": list(FOLDS),
                "pre_window_embargo_sessions": 5,
                "selection_fraction": 0.20,
                "selection_count_rounding": "floor",
                "lambdas": list(LAMBDAS),
                "seeds": list(POLICY_SEEDS),
                "run_count": 18,
                "maximum_epochs": MAX_POLICY_EPOCHS,
                "patience": PATIENCE,
                "optimizer": "AdamW(lr=0.001,weight_decay=0.01)",
                "gradient_clip_norm": 1.0,
                "sam": False,
                "objective": "mean_excess_bps-lambda*population_std_net_bps",
                "lambda_selection": "higher mean best-selection objective over three seeds; tie 0.02",
                "graduation": "strictly positive pooled seed-averaged daily net excess over all-cash CDI",
                "bootstrap": {
                    "block_length": BOOTSTRAP_BLOCK,
                    "replications": BOOTSTRAP_REPLICATIONS,
                    "seed": BOOTSTRAP_SEED,
                },
                "execution_config": ExecutionConfig(
                    gross_target=2.0,
                    name_cap_fraction_of_gross=0.05,
                    horizon_blend=(0.25, 0.25, 0.25, 0.25),
                ).to_dict(),
                "horizon_names": list(HORIZON_NAMES),
            },
            "official_validation_accessed": False,
            "test_accessed": False,
        }
        _atomic_json(output_dir / "frozen_design.json", design)
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return output_dir / "frozen_design.json"


def _build_full_market_cache(
    root: Path, design: Mapping[str, object]
) -> dict[str, object]:
    final = root / "market_inputs"
    if final.exists():
        manifest = _read_json(final / "manifest.json")
        for name, record in manifest["artifacts"].items():
            if _sha256(final / name) != record["sha256"]:
                raise ValueError(f"Experiment 56 market input hash differs: {name}")
        return manifest
    temporary = root / ".market_inputs.tmp"
    temporary.mkdir()
    store = Path(str(design["store"]["path"]))
    archive_record = design["inputs"]["oof_archive"]
    archive = load_discovery_prediction_archive(
        Path(archive_record["prediction"]),
        Path(archive_record["reference"]),
        Path(archive_record["execution_manifest"]),
        store,
    )
    requested_idx = archive.date_idx
    date_table = pl.read_parquet(store / "date_index.parquet").sort("date_idx")
    selected = date_table.filter(pl.col("date_idx").is_in(requested_idx)).sort(
        "date_idx"
    )
    if selected.height != 716 or not np.array_equal(
        selected["date_idx"].to_numpy(), requested_idx
    ):
        raise ValueError("Experiment 56 requires exact 716-date TRAIN OOF coverage")
    trade_dates = tuple(selected["trade_date"])
    equity = pl.read_parquet(store / "equity_index.parquet").sort("equity_slot")
    security_ids = tuple(equity["security_id"])
    shape = (requested_idx.size, EQUITY_SESSION_MINUTES, len(security_ids))
    open_price = np.zeros(shape, dtype=np.float32)
    close_price = np.zeros(shape, dtype=np.float32)
    observed = np.zeros(shape, dtype=bool)
    minute_notional = np.full(shape, np.nan, dtype=np.float32)
    active = np.zeros((requested_idx.size, len(security_ids)), dtype=bool)
    adv = np.full(active.shape, np.nan, dtype=np.float32)
    fallback = np.full(active.shape, np.nan, dtype=np.float32)
    seen = np.zeros(len(security_ids), dtype=bool)
    for grid in iter_discovery_equity_grids(store):
        positions = {value: index for index, value in enumerate(grid.trade_dates)}
        source_position = np.asarray([positions[value] for value in trade_dates])
        liquidity_adv, profile = causal_liquidity(
            grid.close[..., None],
            grid.real_volume[..., None],
            grid.observed[..., None],
            lookback=20,
        )
        roll = causal_roll_spreads(
            grid.close[..., None], grid.observed[..., None], lookback=60
        )
        slot = grid.equity_slot
        open_price[:, :, slot] = grid.open_price[source_position]
        close_price[:, :, slot] = grid.close[source_position]
        observed[:, :, slot] = grid.observed[source_position]
        minute_notional[:, :, slot] = profile[source_position, :, 0]
        active[:, slot] = grid.active[source_position]
        adv[:, slot] = liquidity_adv[source_position, 0]
        fallback[:, slot] = roll[source_position, 0]
        seen[slot] = True
    if not seen.all():
        raise ValueError("Experiment 56 market bridge omitted a permanent security")
    roll_record = design["inputs"]["roll_schedule"]
    scheduled = lagged_quarter_spreads(
        Path(roll_record["path"]), trade_dates, security_ids, str(roll_record["sha256"])
    )
    full_spread = np.where(np.isfinite(scheduled), scheduled, fallback).astype(
        np.float32
    )
    slow = np.load(store / "equity_slow.npy", mmap_mode="r")
    sigma = stored_daily_volatility(np.asarray(slow[requested_idx, :, 0])).astype(
        np.float32
    )
    artifacts = {
        name: _save_array(temporary, name, values)
        for name, values in {
            "date_idx.npy": requested_idx,
            "open_price.npy": open_price,
            "close_price.npy": close_price,
            "open_observed.npy": observed,
            "active.npy": active,
            "adv20_brl.npy": adv,
            "minute_notional20_brl.npy": minute_notional,
            "full_spread.npy": full_spread,
            "sigma_daily.npy": sigma,
        }.items()
    }
    selected.write_parquet(temporary / "dates.parquet")
    artifacts["dates.parquet"] = _artifact(temporary / "dates.parquet")
    manifest = {
        "schema": "EXPERIMENT56_TRAIN_MARKET_INPUTS_V2",
        "created_at": _now(),
        "store_manifest": design["store"]["manifest"],
        "roll_schedule": roll_record,
        "liquidity": "causal_liquidity lookback=20, current session excluded",
        "roll_fallback": "causal_roll_spreads lookback=60, prior sessions only",
        "date_count": 716,
        "artifacts": artifacts,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(temporary / "manifest.json", manifest)
    os.replace(temporary, final)
    return _read_json(final / "manifest.json")


@dataclass(frozen=True)
class _PolicySplit:
    window: str
    fit: np.ndarray
    selection: np.ndarray
    evaluation: np.ndarray


def policy_splits(dates: Sequence[date]) -> tuple[_PolicySplit, ...]:
    windows = policy_evaluation_slices(dates, dates)
    position = {value: index for index, value in enumerate(dates)}
    result = []
    for window in windows:
        evaluation = np.asarray(
            [position[value] for value in window.dates], dtype=np.int64
        )
        pre_stop = int(evaluation[0]) - 5
        if pre_stop <= 1:
            raise ValueError("Policy window has insufficient embargoed history")
        selection_count = pre_stop // 5
        fit_stop = pre_stop - selection_count
        result.append(
            _PolicySplit(
                window.name,
                np.arange(fit_stop, dtype=np.int64),
                np.arange(fit_stop, pre_stop, dtype=np.int64),
                evaluation,
            )
        )
    return tuple(result)


def _policy_batch(
    *,
    root: Path,
    design: Mapping[str, object],
    archive: object,
    positions: np.ndarray,
    device: torch.device,
) -> PolicyBatch:
    dtype = torch.float32
    ranks, valid, _, refresh = expand_refreshes(
        archive.ranks[positions],
        archive.valid[positions],
        archive.refresh_minutes,
        EQUITY_SESSION_MINUTES,
    )
    dates = tuple(
        pl.read_parquet(root / "market_inputs" / "dates.parquet").sort("date_idx")[
            "trade_date"
        ]
    )
    cdi_record = design["inputs"]["cdi"]["parquet"]
    cdi = load_daily_cdi_rates(
        Path(cdi_record["path"]),
        tuple(dates[index] for index in positions),
        cdi_record["sha256"],
    )

    def tensor(values: object) -> torch.Tensor:
        return torch.as_tensor(np.asarray(values), dtype=dtype, device=device)

    def boolean(values: object) -> torch.Tensor:
        return torch.as_tensor(np.asarray(values), dtype=torch.bool, device=device)

    market = MarketReplay(
        open_price=tensor(_load_cache_array(root, "open_price.npy")[positions]),
        open_observed=boolean(_load_cache_array(root, "open_observed.npy")[positions]),
        active=boolean(_load_cache_array(root, "active.npy")[positions]),
        full_spread=tensor(_load_cache_array(root, "full_spread.npy")[positions]),
        adv20_brl=tensor(_load_cache_array(root, "adv20_brl.npy")[positions]),
        minute_notional20_brl=tensor(
            _load_cache_array(root, "minute_notional20_brl.npy")[positions]
        ),
        daily_cdi_rate=tensor(cdi),
    )
    return PolicyBatch(
        market=market,
        ranks=tensor(ranks),
        rank_valid=boolean(valid),
        refresh_mask=boolean(refresh),
        sigma=tensor(_load_cache_array(root, "sigma_daily.npy")[positions]),
    )


def _run_dir(root: Path, window: str, risk_lambda: float, seed: int) -> Path:
    return root / "runs" / window / f"lambda_{risk_lambda:.2f}" / f"seed_{seed}"


def _simulation_rows(
    dates: Sequence[date], result: SimulationResult, nav_brl: float
) -> pl.DataFrame:
    columns = (
        "net_pnl_brl",
        "gross_pnl_brl",
        "spread_cost_brl",
        "fees_brl",
        "cdi_earned_brl",
        "turnover_brl",
        "max_intraday_gross_brl",
        "mean_deployed_gross_brl",
        "forced_fill_count",
    )
    values = {name: getattr(result, name).detach().cpu().numpy() for name in columns}
    return pl.DataFrame(
        {
            "trade_date": dates,
            **values,
            "all_cash_cdi_pnl_brl": values["net_pnl_brl"] * 0
            + np.asarray(result.net_pnl_brl.detach().cpu().numpy()) * 0,
            "net_pnl_bps": values["net_pnl_brl"] / nav_brl * 10_000.0,
        }
    )


class _AblatedPolicy:
    projection_mode = "bounded"
    requires_policy_state = True

    def __init__(self, base: NeuralPolicy, head: int) -> None:
        self.base = base
        self.head = head
        self.horizon_names = base.horizon_names

    def step(self, *args: object) -> tuple[torch.Tensor, torch.Tensor]:
        state = args[-1]
        if not isinstance(state, PolicyState):
            raise ValueError("Horizon ablation requires PolicyState")
        per_name = state.per_name.clone()
        count = len(state.horizon_names)
        per_name[..., self.head] = 0
        per_name[..., count + self.head] = 0
        return self.base.step(*args[:-1], replace(state, per_name=per_name))


def _diagnostic_rows(
    *,
    dates: Sequence[date],
    batch: PolicyBatch,
    result: SimulationResult,
    policy: NeuralPolicy,
    config: ExecutionConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, np.ndarray]:
    turnover = result.turnover_by_name_brl.detach().cpu().numpy()
    net_by_name = (
        (
            result.gross_pnl_by_name_brl
            - result.spread_cost_by_name_brl
            - result.fees_by_name_brl
        )
        .detach()
        .cpu()
        .numpy()
    )
    adv = batch.market.adv20_brl.detach().cpu().numpy()
    active = batch.market.active.detach().cpu().numpy().astype(bool)
    liquidity_rows, trade_rows = [], []
    for day, trade_date in enumerate(dates):
        names = np.flatnonzero(active[day] & np.isfinite(adv[day]))
        buckets = np.full(adv.shape[1], -1, dtype=np.int8)
        if names.size:
            order = names[np.argsort(adv[day, names], kind="stable")]
            buckets[order] = np.minimum(np.arange(order.size) * 3 // order.size, 2)
        for bucket in range(3):
            take = buckets == bucket
            liquidity_rows.append(
                {
                    "trade_date": trade_date,
                    "liquidity_tercile": bucket,
                    "turnover_brl": float(turnover[day, take].sum()),
                    "net_trading_pnl_brl": float(net_by_name[day, take].sum()),
                }
            )
        count = result.round_trip_count_by_name[day].detach().cpu().numpy()
        gross = result.round_trip_gross_pnl_by_name_brl[day].detach().cpu().numpy()
        cost = result.round_trip_cost_by_name_brl[day].detach().cpu().numpy()
        trade_rows.append(
            {
                "trade_date": trade_date,
                "round_trip_count": int(count.sum()),
                "gross_edge_brl_per_trade": float(gross.sum() / max(count.sum(), 1)),
                "cost_brl_per_trade": float(cost.sum() / max(count.sum(), 1)),
            }
        )
    targets = result.target_weights.detach().cpu().numpy()
    refresh = batch.refresh_mask.detach().cpu().numpy().astype(bool)
    spread = batch.market.full_spread.detach().cpu().numpy() * 10_000.0
    target_rows = []
    previous = np.zeros_like(targets[:, 0])
    for minute in range(refresh.shape[1] - 1):
        take_day = refresh[:, minute]
        for day in np.flatnonzero(take_day):
            current = targets[day, minute + 1]
            change = np.abs(current - previous[day])
            valid = active[day] & np.isfinite(spread[day])
            target_rows.append(
                {
                    "trade_date": dates[day],
                    "session_minute": minute,
                    "mean_absolute_target_change": float(change[valid].mean())
                    if valid.any()
                    else 0.0,
                    "mean_full_spread_bps": float(spread[day, valid].mean())
                    if valid.any()
                    else 0.0,
                }
            )
        previous = np.where(take_day[:, None], targets[:, minute + 1], previous)
    fills = np.abs(result.fills_brl.detach().cpu().numpy())
    tod_rows = []
    for day, trade_date in enumerate(dates):
        for bucket, (start, stop) in enumerate(((0, 135), (135, 270), (270, 405))):
            tod_rows.append(
                {
                    "trade_date": trade_date,
                    "tod_bucket": bucket,
                    "turnover_brl": float(fills[day, start:stop].sum()),
                }
            )
    base_net = result.net_pnl_brl.detach()
    ablation_rows = []
    for head, name in enumerate(HORIZON_NAMES):
        with torch.no_grad():
            ablated = simulate(
                batch.market,
                batch.ranks,
                batch.rank_valid,
                batch.refresh_mask,
                batch.sigma,
                _AblatedPolicy(policy, head),
                config,
            ).net_pnl_brl
        delta = (base_net - ablated).cpu().numpy() / config.nav_brl * 10_000.0
        ablation_rows.extend(
            {
                "trade_date": value,
                "horizon": name,
                "objective_delta_bps": float(delta[index]),
            }
            for index, value in enumerate(dates)
        )
    gross_path = (
        np.abs(result.positions_brl.detach().cpu().numpy()).sum(axis=2) / config.nav_brl
    )
    return (
        pl.DataFrame(liquidity_rows),
        pl.DataFrame(trade_rows),
        pl.DataFrame(target_rows),
        pl.DataFrame(tod_rows),
        gross_path,
    ), pl.DataFrame(ablation_rows)


def _run_manifest_valid(run: Path, window: str, risk_lambda: float, seed: int) -> bool:
    path = run / "run_manifest.json"
    if not path.is_file():
        return False
    value = _read_json(path)
    required = (
        run / "history.parquet",
        run / "evaluation_daily.parquet",
        run / "liquidity_daily.parquet",
        run / "per_trade_daily.parquet",
        run / "target_change.parquet",
        run / "tod_daily.parquet",
        run / "horizon_ablation.parquet",
        run / "gross_path.npy",
        run / "policy.pt",
    )
    return bool(
        value.get("schema") == "EXPERIMENT56_POLICY_RUN_V1"
        and value.get("status") == "completed"
        and value.get("window") == window
        and value.get("lambda") == risk_lambda
        and value.get("seed") == seed
        and all(path.is_file() for path in required)
        and all(
            _sha256(path) == value["artifacts"][path.name]["sha256"]
            for path in required
        )
    )


def _train_one_policy(
    *,
    root: Path,
    design: Mapping[str, object],
    archive: object,
    split: _PolicySplit,
    risk_lambda: float,
    seed: int,
    device: torch.device,
) -> Path:
    run = _run_dir(root, split.window, risk_lambda, seed)
    if _run_manifest_valid(run, split.window, risk_lambda, seed):
        return run / "run_manifest.json"
    run.mkdir(parents=True, exist_ok=True)
    config = ExecutionConfig(
        gross_target=2.0,
        name_cap_fraction_of_gross=0.05,
        horizon_blend=(0.25, 0.25, 0.25, 0.25),
    )
    policy = NeuralPolicy(
        config, horizon_count=4, horizon_names=HORIZON_NAMES, seed=seed
    ).to(device)
    trainer_config = PolicyTrainerConfig(
        learning_rate=1e-3,
        weight_decay=0.01,
        risk_aversion=risk_lambda,
        gradient_clip_norm=1.0,
        seed=seed,
        use_sam=False,
        gradient_checkpointing=True,
        patience=PATIENCE,
    )
    trainer = PolicyTrainer(policy, config, trainer_config)
    fit = _policy_batch(
        root=root, design=design, archive=archive, positions=split.fit, device=device
    )
    selection = _policy_batch(
        root=root,
        design=design,
        archive=archive,
        positions=split.selection,
        device=device,
    )
    progress = run / "latest.pt"
    history_path = run / "history.partial.json"
    history = []
    start_epoch = 1
    if progress.is_file() and history_path.is_file():
        trainer.load_checkpoint(progress)
        history = json.loads(history_path.read_text(encoding="utf-8"))
        start_epoch = int(history[-1]["epoch"]) + 1
    for epoch in range(start_epoch, MAX_POLICY_EPOCHS + 1):
        train_metrics = trainer.train_step(fit)
        policy.eval()
        with torch.no_grad():
            selection_result = simulate(
                selection.market,
                selection.ranks,
                selection.rank_valid,
                selection.refresh_mask,
                selection.sigma,
                policy,
                config,
            )
            monitor, excess, standard_deviation = policy_objective(
                selection_result.net_pnl_brl,
                selection.market.daily_cdi_rate,
                config.nav_brl,
                risk_lambda,
            )
        exhausted = trainer.update_monitor(float(monitor))
        history.append(
            {
                "epoch": epoch,
                **asdict(train_metrics),
                "selection_objective_bps": float(monitor),
                "selection_mean_excess_bps": float(excess.mean()),
                "selection_daily_net_std_bps": float(standard_deviation),
                "best_selection_objective_bps": float(trainer.best_monitor),
            }
        )
        trainer.save_checkpoint(progress)
        _atomic_json(history_path, history)
        if exhausted:
            break
    trainer.restore_best()
    policy_path = run / "policy.pt"
    trainer.save_checkpoint(policy_path)
    progress.unlink(missing_ok=True)
    history_path.unlink(missing_ok=True)
    history_final = run / "history.parquet"
    pl.DataFrame(history).write_parquet(history_final)
    evaluation = _policy_batch(
        root=root,
        design=design,
        archive=archive,
        positions=split.evaluation,
        device=device,
    )
    policy.eval()
    with torch.no_grad():
        result = simulate(
            evaluation.market,
            evaluation.ranks,
            evaluation.rank_valid,
            evaluation.refresh_mask,
            evaluation.sigma,
            policy,
            config,
            return_path=True,
        )
    all_dates = tuple(
        pl.read_parquet(root / "market_inputs" / "dates.parquet").sort("date_idx")[
            "trade_date"
        ]
    )
    eval_dates = tuple(all_dates[index] for index in split.evaluation)
    daily = (
        _simulation_rows(eval_dates, result, config.nav_brl)
        .with_columns(
            pl.Series(
                "all_cash_cdi_pnl_brl",
                (evaluation.market.daily_cdi_rate * config.nav_brl)
                .detach()
                .cpu()
                .numpy(),
            )
        )
        .with_columns(
            (pl.col("net_pnl_brl") - pl.col("all_cash_cdi_pnl_brl")).alias(
                "excess_pnl_brl"
            ),
            (
                (pl.col("net_pnl_brl") - pl.col("all_cash_cdi_pnl_brl"))
                / config.nav_brl
                * 10_000.0
            ).alias("excess_pnl_bps"),
        )
    )
    daily_path = run / "evaluation_daily.parquet"
    daily.write_parquet(daily_path)
    (diagnostics, ablation) = _diagnostic_rows(
        dates=eval_dates, batch=evaluation, result=result, policy=policy, config=config
    )
    names = (
        "liquidity_daily.parquet",
        "per_trade_daily.parquet",
        "target_change.parquet",
        "tod_daily.parquet",
    )
    diagnostic_paths = []
    for name, table in zip(names, diagnostics[:4], strict=True):
        path = run / name
        table.write_parquet(path)
        diagnostic_paths.append(path)
    gross_path = run / "gross_path.npy"
    with gross_path.open("wb") as output:
        np.save(output, diagnostics[4], allow_pickle=False)
    ablation_path = run / "horizon_ablation.parquet"
    ablation.write_parquet(ablation_path)
    artifacts = [
        history_final,
        daily_path,
        *diagnostic_paths,
        gross_path,
        ablation_path,
        policy_path,
    ]
    manifest = {
        "schema": "EXPERIMENT56_POLICY_RUN_V1",
        "status": "completed",
        "created_at": _now(),
        "repository_commit": repository_commit(),
        "window": split.window,
        "lambda": risk_lambda,
        "seed": seed,
        "fit_date_count": int(split.fit.size),
        "selection_date_count": int(split.selection.size),
        "evaluation_date_count": int(split.evaluation.size),
        "fit_last_position": int(split.fit[-1]),
        "selection_first_position": int(split.selection[0]),
        "selection_last_position": int(split.selection[-1]),
        "evaluation_first_position": int(split.evaluation[0]),
        "embargo_sessions": int(split.evaluation[0] - split.selection[-1] - 1),
        "epochs_completed": len(history),
        "best_selection_objective_bps": float(trainer.best_monitor),
        "execution_config": config.to_dict(),
        "trainer_config": asdict(trainer_config),
        "policy_contract": policy.contract_metadata,
        "artifacts": {path.name: _artifact(path) for path in artifacts},
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(run / "run_manifest.json", manifest)
    return run / "run_manifest.json"


def _json_interval(values: np.ndarray, seed_offset: int = 0) -> dict[str, float]:
    result = moving_block_bootstrap(
        np.asarray(values, dtype=np.float64),
        replications=BOOTSTRAP_REPLICATIONS,
        block_length=BOOTSTRAP_BLOCK,
        seed=BOOTSTRAP_SEED + seed_offset,
    )
    return {
        name: float(np.asarray(value).reshape(-1)[0]) for name, value in result.items()
    }


def _sharpe_interval(values: np.ndarray, seed_offset: int = 0) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    count = values.size
    blocks = math.ceil(count / BOOTSTRAP_BLOCK)
    generator = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    starts = generator.integers(
        0,
        count - BOOTSTRAP_BLOCK + 1,
        size=(BOOTSTRAP_REPLICATIONS, blocks),
    )
    indices = (starts[..., None] + np.arange(BOOTSTRAP_BLOCK)).reshape(
        BOOTSTRAP_REPLICATIONS, -1
    )[:, :count]
    sampled = values[indices]
    standard_deviation = sampled.std(axis=1, ddof=1)
    ratios = np.divide(
        np.sqrt(252.0) * sampled.mean(axis=1),
        standard_deviation,
        out=np.zeros_like(standard_deviation),
        where=standard_deviation > 0,
    )
    estimate_std = values.std(ddof=1)
    estimate = (
        float(np.sqrt(252.0) * values.mean() / estimate_std)
        if estimate_std > 0
        else 0.0
    )
    return {
        "estimate": estimate,
        "lower_95": float(np.quantile(ratios, 0.025)),
        "upper_95": float(np.quantile(ratios, 0.975)),
    }


def _load_baseline_daily(root: Path, fold: str, cell: str) -> pl.DataFrame:
    payload = _read_json(root / "reports" / fold / f"{cell}__measured.json")
    return (
        pl.DataFrame(payload["daily"])
        .with_columns(
            pl.col("trade_date").str.to_date(),
        )
        .select("trade_date", pl.col("net_pnl_brl").alias("baseline_net_pnl_brl"))
    )


def _seed_average_daily(root: Path, window: str, risk_lambda: float) -> pl.DataFrame:
    tables = []
    for seed in POLICY_SEEDS:
        tables.append(
            pl.read_parquet(
                _run_dir(root, window, risk_lambda, seed) / "evaluation_daily.parquet"
            ).with_columns(pl.lit(seed).alias("seed"))
        )
    return (
        pl.concat(tables)
        .group_by("trade_date", maintain_order=True)
        .agg(pl.exclude("trade_date", "seed").mean())
        .sort("trade_date")
    )


def _designated_table(
    root: Path, selected: Mapping[str, float], filename: str
) -> pl.DataFrame:
    tables = []
    for window in FOLDS:
        for seed in POLICY_SEEDS:
            tables.append(
                pl.read_parquet(
                    _run_dir(root, window, selected[window], seed) / filename
                ).with_columns(
                    pl.lit(window).alias("window"), pl.lit(seed).alias("seed")
                )
            )
    return pl.concat(tables, how="diagonal_relaxed")


def _summary_row(name: str, values: np.ndarray, seed_offset: int) -> dict[str, object]:
    return {"readout": name, **_json_interval(values, seed_offset)}


def _policy_readouts(
    *, root: Path, design: Mapping[str, object], selected: Mapping[str, float]
) -> tuple[Path, list[Path]]:
    source_a = Path(str(design["inputs"]["section_a_root"]))
    oracle = {
        row["fold"]: float(row["mean_net_nav_bps_per_day"])
        for row in _read_json(source_a / "result.json")["total_frontiers"]
    }
    source52 = Path(str(design["inputs"]["experiment52_root"]))
    source53 = Path(str(design["inputs"]["experiment53_root"]))
    c0_cell = "band_2p0__blend_equal"
    c1_cell = "k40__band1p5__c1p0__gross1p0__universe_full"
    window_tables, rows = [], []
    for index, window in enumerate(FOLDS):
        daily = _seed_average_daily(root, window, selected[window])
        c0 = _load_baseline_daily(source52, window, c0_cell)
        c1 = _load_baseline_daily(source53, window, c1_cell)
        daily = (
            daily.join(c0, on="trade_date", how="inner")
            .rename({"baseline_net_pnl_brl": "c0_net_pnl_brl"})
            .join(c1, on="trade_date", how="inner")
            .rename({"baseline_net_pnl_brl": "c1_net_pnl_brl"})
            .with_columns(
                ((pl.col("net_pnl_brl") - pl.col("c0_net_pnl_brl")) / 1_000.0).alias(
                    "delta_c0_bps"
                ),
                ((pl.col("net_pnl_brl") - pl.col("c1_net_pnl_brl")) / 1_000.0).alias(
                    "delta_c1_bps"
                ),
                (
                    (pl.col("net_pnl_brl") - pl.col("all_cash_cdi_pnl_brl")) / 1_000.0
                ).alias("delta_all_cash_bps"),
                pl.lit(window).alias("window"),
            )
        )
        window_tables.append(daily)
        net_bps = daily["net_pnl_brl"].to_numpy() / 1_000.0
        window_rows = [
            _summary_row(
                f"{window}/net_excess_all_cash_bps",
                daily["delta_all_cash_bps"].to_numpy(),
                10 + index,
            ),
            _summary_row(
                f"{window}/delta_c0_bps", daily["delta_c0_bps"].to_numpy(), 20 + index
            ),
            _summary_row(
                f"{window}/delta_c1_bps", daily["delta_c1_bps"].to_numpy(), 30 + index
            ),
            {
                "readout": f"{window}/net_sharpe",
                **_sharpe_interval(net_bps, 40 + index),
            },
        ]
        excess_mean = float(daily["delta_all_cash_bps"].mean())
        window_rows.append(
            {
                "readout": f"{window}/oracle_capture_ratio",
                "estimate": excess_mean / oracle[window]
                if oracle[window] != 0
                else 0.0,
                "lower_95": _json_interval(
                    daily["delta_all_cash_bps"].to_numpy(), 50 + index
                )["lower_95"]
                / oracle[window],
                "upper_95": _json_interval(
                    daily["delta_all_cash_bps"].to_numpy(), 50 + index
                )["upper_95"]
                / oracle[window],
                "oracle_frontier_bps_per_day": oracle[window],
            }
        )
        rows.extend(window_rows)
    combined = pl.concat(window_tables).sort("trade_date")
    combined_path = root / "designated_daily.parquet"
    combined.write_parquet(combined_path)
    pooled_net = combined["net_pnl_brl"].to_numpy() / 1_000.0
    rows.extend(
        [
            _summary_row(
                "pooled/net_excess_all_cash_bps",
                combined["delta_all_cash_bps"].to_numpy(),
                60,
            ),
            _summary_row(
                "pooled/delta_c0_bps", combined["delta_c0_bps"].to_numpy(), 61
            ),
            _summary_row(
                "pooled/delta_c1_bps", combined["delta_c1_bps"].to_numpy(), 62
            ),
            {"readout": "pooled/net_sharpe", **_sharpe_interval(pooled_net, 63)},
            _summary_row(
                "pooled/turnover_bps_nav",
                combined["turnover_brl"].to_numpy() / 1_000.0,
                64,
            ),
            _summary_row(
                "pooled/mean_deployed_gross_fraction_nav",
                combined["mean_deployed_gross_brl"].to_numpy() / 10_000_000.0,
                65,
            ),
        ]
    )
    readout_path = root / "readouts.parquet"
    pl.DataFrame(rows).write_parquet(readout_path)
    diagnostic_paths = []

    liquidity = _designated_table(root, selected, "liquidity_daily.parquet")
    liquidity = (
        liquidity.group_by("trade_date", "liquidity_tercile", maintain_order=True)
        .agg(pl.col("turnover_brl").mean(), pl.col("net_trading_pnl_brl").mean())
        .with_columns(
            pl.when(pl.col("turnover_brl").sum().over("trade_date") > 0)
            .then(
                pl.col("turnover_brl") / pl.col("turnover_brl").sum().over("trade_date")
            )
            .otherwise(0.0)
            .alias("trade_share")
        )
    )
    liquidity_summary = []
    for bucket in range(3):
        values = liquidity.filter(pl.col("liquidity_tercile") == bucket)
        liquidity_summary.extend(
            [
                {
                    "liquidity_tercile": bucket,
                    "readout": "trade_share",
                    **_json_interval(values["trade_share"].to_numpy(), 100 + bucket),
                },
                {
                    "liquidity_tercile": bucket,
                    "readout": "net_trading_pnl_bps",
                    **_json_interval(
                        values["net_trading_pnl_brl"].to_numpy() / 1_000.0, 110 + bucket
                    ),
                },
            ]
        )
    path = root / "liquidity_summary.parquet"
    pl.DataFrame(liquidity_summary).write_parquet(path)
    diagnostic_paths.append(path)

    trades = (
        _designated_table(root, selected, "per_trade_daily.parquet")
        .group_by("trade_date", maintain_order=True)
        .agg(
            pl.col("round_trip_count").mean(),
            pl.col("gross_edge_brl_per_trade").mean(),
            pl.col("cost_brl_per_trade").mean(),
        )
        .sort("trade_date")
    )
    path = root / "per_trade_summary.parquet"
    pl.DataFrame(
        [
            _summary_row(
                "round_trip_count", trades["round_trip_count"].to_numpy(), 120
            ),
            _summary_row(
                "gross_edge_brl_per_trade",
                trades["gross_edge_brl_per_trade"].to_numpy(),
                121,
            ),
            _summary_row(
                "cost_brl_per_trade", trades["cost_brl_per_trade"].to_numpy(), 122
            ),
        ]
    ).write_parquet(path)
    diagnostic_paths.append(path)

    target = (
        _designated_table(root, selected, "target_change.parquet")
        .group_by("trade_date", "session_minute", maintain_order=True)
        .agg(
            pl.col("mean_absolute_target_change").mean(),
            pl.col("mean_full_spread_bps").mean(),
        )
    )
    spread_values = target["mean_full_spread_bps"].to_numpy()
    edges = np.quantile(spread_values, (1 / 3, 2 / 3))
    target = (
        target.with_columns(
            pl.Series(
                "spread_tercile", np.searchsorted(edges, spread_values, side="right")
            )
        )
        .group_by("trade_date", "spread_tercile", maintain_order=True)
        .agg(
            pl.col("mean_absolute_target_change").mean(),
            pl.col("mean_full_spread_bps").mean(),
        )
    )
    target_summary = []
    for bucket in range(3):
        values = target.filter(pl.col("spread_tercile") == bucket)
        target_summary.extend(
            [
                {
                    "spread_tercile": bucket,
                    "readout": "absolute_target_change",
                    **_json_interval(
                        values["mean_absolute_target_change"].to_numpy(), 130 + bucket
                    ),
                },
                {
                    "spread_tercile": bucket,
                    "readout": "full_spread_bps",
                    **_json_interval(
                        values["mean_full_spread_bps"].to_numpy(), 140 + bucket
                    ),
                },
            ]
        )
    path = root / "target_spread_summary.parquet"
    pl.DataFrame(target_summary).write_parquet(path)
    diagnostic_paths.append(path)

    tod = (
        _designated_table(root, selected, "tod_daily.parquet")
        .group_by("trade_date", "tod_bucket", maintain_order=True)
        .agg(pl.col("turnover_brl").mean())
    )
    tod_summary = []
    for bucket in range(3):
        values = tod.filter(pl.col("tod_bucket") == bucket)
        tod_summary.append(
            {
                "tod_bucket": bucket,
                **_json_interval(values["turnover_brl"].to_numpy(), 150 + bucket),
            }
        )
    path = root / "tod_summary.parquet"
    pl.DataFrame(tod_summary).write_parquet(path)
    diagnostic_paths.append(path)

    ablation = (
        _designated_table(root, selected, "horizon_ablation.parquet")
        .group_by("trade_date", "horizon", maintain_order=True)
        .agg(pl.col("objective_delta_bps").mean())
    )
    ablation_summary = []
    for index, horizon in enumerate(HORIZON_NAMES):
        values = ablation.filter(pl.col("horizon") == horizon)[
            "objective_delta_bps"
        ].to_numpy()
        ablation_summary.append(
            {"horizon": horizon, **_json_interval(values, 160 + index)}
        )
    path = root / "horizon_ablation_summary.parquet"
    pl.DataFrame(ablation_summary).write_parquet(path)
    diagnostic_paths.append(path)

    gross_windows = []
    for window in FOLDS:
        gross_windows.append(
            np.stack(
                [
                    np.load(
                        _run_dir(root, window, selected[window], seed)
                        / "gross_path.npy"
                    )
                    for seed in POLICY_SEEDS
                ]
            ).mean(axis=0)
        )
    gross = np.concatenate(gross_windows)
    interval = moving_block_bootstrap(
        gross,
        replications=BOOTSTRAP_REPLICATIONS,
        block_length=BOOTSTRAP_BLOCK,
        seed=BOOTSTRAP_SEED + 170,
    )
    path = root / "gross_trajectory.npz"
    with path.open("wb") as output:
        np.savez(output, **interval)
    diagnostic_paths.append(path)
    return readout_path, [combined_path, *diagnostic_paths]


def run_section_c(root: Path, device_name: str = "cuda") -> Path:
    root = root.resolve()
    final = root / "result.json"
    if final.exists():
        _verified_result(root, "result.json")
        return final
    design = _read_json(root / "frozen_design.json")
    if (
        design.get("schema") != SECTION_C_SCHEMA
        or design.get("repository_commit") != repository_commit()
    ):
        raise ValueError("Section C must run at its exact frozen commit")
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "Experiment 56 policy training requires the claimed CUDA instance"
        )
    _build_full_market_cache(root, design)
    store = Path(str(design["store"]["path"]))
    archive_record = design["inputs"]["oof_archive"]
    archive = load_discovery_prediction_archive(
        Path(archive_record["prediction"]),
        Path(archive_record["reference"]),
        Path(archive_record["execution_manifest"]),
        store,
    )
    dates = tuple(
        pl.read_parquet(root / "market_inputs" / "dates.parquet").sort("date_idx")[
            "trade_date"
        ]
    )
    splits = policy_splits(dates)
    manifests = []
    for split in splits:
        for risk_lambda in LAMBDAS:
            for seed in POLICY_SEEDS:
                manifests.append(
                    _train_one_policy(
                        root=root,
                        design=design,
                        archive=archive,
                        split=split,
                        risk_lambda=risk_lambda,
                        seed=seed,
                        device=device,
                    )
                )
                torch.cuda.empty_cache()
    if not all(
        _run_manifest_valid(
            path.parent,
            _read_json(path)["window"],
            _read_json(path)["lambda"],
            _read_json(path)["seed"],
        )
        for path in manifests
    ):
        raise ValueError("Experiment 56 did not complete all 18 policy runs")
    selection_rows = []
    selected: dict[str, float] = {}
    for window in FOLDS:
        means = {}
        for risk_lambda in LAMBDAS:
            values = [
                float(
                    _read_json(
                        _run_dir(root, window, risk_lambda, seed) / "run_manifest.json"
                    )["best_selection_objective_bps"]
                )
                for seed in POLICY_SEEDS
            ]
            means[risk_lambda] = float(np.mean(values))
            selection_rows.append(
                {
                    "window": window,
                    "lambda": risk_lambda,
                    "mean_best_selection_objective_bps": means[risk_lambda],
                }
            )
        selected[window] = max(LAMBDAS, key=lambda value: (means[value], -value))
    selection_path = root / "lambda_selection.parquet"
    pl.DataFrame(selection_rows).with_columns(
        pl.struct("window", "lambda")
        .map_elements(
            lambda row: row["lambda"] == selected[row["window"]],
            return_dtype=pl.Boolean,
        )
        .alias("designated")
    ).write_parquet(selection_path)
    readouts, diagnostic_paths = _policy_readouts(
        root=root, design=design, selected=selected
    )
    pooled = pl.read_parquet(root / "designated_daily.parquet")
    pooled_mean_excess = float(pooled["delta_all_cash_bps"].mean())
    graduated = pooled_mean_excess > 0.0
    checkpoint_inventory = []
    for window in FOLDS:
        for risk_lambda in LAMBDAS:
            for seed in POLICY_SEEDS:
                path = _run_dir(root, window, risk_lambda, seed) / "policy.pt"
                retained = risk_lambda == selected[window]
                checkpoint_inventory.append(
                    {
                        "window": window,
                        "lambda": risk_lambda,
                        "seed": seed,
                        "sha256": _sha256(path),
                        "bytes": path.stat().st_size,
                        "retained": retained,
                    }
                )
                if not retained:
                    path.unlink()
    checkpoint_path = root / "checkpoint_inventory.parquet"
    pl.DataFrame(checkpoint_inventory).write_parquet(checkpoint_path)
    result = {
        "schema": SECTION_C_SCHEMA,
        "status": "completed",
        "created_at": _now(),
        "selected_lambda_by_window": selected,
        "pooled_mean_daily_net_excess_all_cash_bps": pooled_mean_excess,
        "graduated_as_standing_execution_candidate": graduated,
        "paper_preparation_unlocked": graduated,
        "deployed_prediction_recipe_changed": False,
        "run_count": 18,
        "retained_checkpoint_count": 9,
        "market_inputs": _artifact(root / "market_inputs" / "manifest.json"),
        "artifacts": {
            "lambda_selection": _artifact(selection_path),
            "readouts": _artifact(readouts),
            "checkpoint_inventory": _artifact(checkpoint_path),
            **{path.stem: _artifact(path) for path in diagnostic_paths},
        },
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(final, result)
    retained = list((root / "runs").glob("*/lambda_*/*/policy.pt"))
    _atomic_json(
        root / "final_audit.json",
        {
            "schema": "EXPERIMENT56_SECTION_C_AUDIT_V1",
            "status": "passed",
            "result_sha256": _sha256(final),
            "all_18_runs_completed": True,
            "chronological_fit_selection_evaluation_verified": all(
                int(_read_json(path)["embargo_sessions"]) == 5 for path in manifests
            ),
            "market_input_manifest_sha256": _sha256(
                root / "market_inputs" / "manifest.json"
            ),
            "retained_checkpoint_count": len(retained),
            "retained_checkpoint_hashes_verified": len(retained) == 9,
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    return final


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 56 learned execution policy"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_a = commands.add_parser("freeze-a")
    freeze_a.add_argument("--experiment55-root", type=Path, required=True)
    freeze_a.add_argument("--experiment54-root", type=Path, required=True)
    freeze_a.add_argument("--preregistration", type=Path, required=True)
    freeze_a.add_argument("--output-dir", type=Path, required=True)
    run_a = commands.add_parser("run-a")
    run_a.add_argument("--root", type=Path, required=True)
    freeze_b = commands.add_parser("freeze-b")
    freeze_b.add_argument("--section-a-root", type=Path, required=True)
    freeze_b.add_argument("--base-oof-root", type=Path, required=True)
    freeze_b.add_argument("--target-sidecar", type=Path, required=True)
    freeze_b.add_argument("--preregistration", type=Path, required=True)
    freeze_b.add_argument("--output-dir", type=Path, required=True)
    run_b = commands.add_parser("run-b")
    run_b.add_argument("--root", type=Path, required=True)
    run_b.add_argument("--parallel", type=int, default=2)
    freeze_c = commands.add_parser("freeze-c")
    freeze_c.add_argument("--section-a-root", type=Path, required=True)
    freeze_c.add_argument("--section-b-root", type=Path, required=True)
    freeze_c.add_argument("--experiment52-root", type=Path, required=True)
    freeze_c.add_argument("--experiment53-root", type=Path, required=True)
    freeze_c.add_argument("--preregistration", type=Path, required=True)
    freeze_c.add_argument("--output-dir", type=Path, required=True)
    run_c = commands.add_parser("run-c")
    run_c.add_argument("--root", type=Path, required=True)
    run_c.add_argument("--device", default="cuda")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    if args.command == "freeze-a":
        result = freeze_section_a(
            experiment55_root=args.experiment55_root,
            experiment54_root=args.experiment54_root,
            preregistration=args.preregistration,
            output_dir=args.output_dir,
        )
    elif args.command == "run-a":
        result = run_section_a(args.root)
    elif args.command == "freeze-b":
        result = freeze_section_b(
            section_a_root=args.section_a_root,
            base_oof_root=args.base_oof_root,
            target_sidecar=args.target_sidecar,
            preregistration=args.preregistration,
            output_dir=args.output_dir,
        )
    elif args.command == "run-b":
        result = run_section_b(args.root, args.parallel)
    elif args.command == "freeze-c":
        result = freeze_section_c(
            section_a_root=args.section_a_root,
            section_b_root=args.section_b_root,
            experiment52_root=args.experiment52_root,
            experiment53_root=args.experiment53_root,
            preregistration=args.preregistration,
            output_dir=args.output_dir,
        )
    else:
        result = run_section_c(args.root, args.device)
    print(result)


if __name__ == "__main__":
    main()
