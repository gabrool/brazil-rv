from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from ..modeling.contract import TRAIN_END, TRAIN_START
from ..modeling.data import feature_store_identity
from ..modeling.experiment55_to_close import FOLDS, _to_close_edge_bps
from ..modeling.provenance import repository_commit
from ..preprocessing.contract import EQUITY_SESSION_MINUTES
from .config import ExecutionConfig
from .constraints import project_weights_bounded
from .experiment52 import _fetch_cdi, _load_cache_array
from .experiment54 import build_state_events, forward_edge_bps
from .experiment56 import (
    BOOTSTRAP_BLOCK,
    BOOTSTRAP_REPLICATIONS,
    HORIZON_NAMES,
    MAX_POLICY_EPOCHS,
    PATIENCE,
    POLICY_SEEDS,
    _artifact,
    _atomic_json,
    _build_full_market_cache,
    _json_interval,
    _load_baseline_daily,
    _policy_batch,
    _read_json,
    _sha256,
    _sharpe_interval,
    _simulation_rows,
    _verified_result,
    policy_splits,
)
from .inputs import load_discovery_prediction_archive
from .policy import NeuralPolicy
from .simulator import SimulationResult, simulate, tradeable_universe
from .splits import policy_evaluation_slices
from .trainer import PolicyTrainer, PolicyTrainerConfig, policy_objective


STAGE0_SCHEMA = "EXPERIMENT57_STAGE0_V1"
STAGE1_SCHEMA = "EXPERIMENT57_STAGE1_V1"
STAGE2_SCHEMA = "EXPERIMENT57_STAGE2_V1"
THRESHOLDS_BPS = (0.0, 4.5, 7.0)
CLONE_EPOCHS = 20
RISK_LAMBDA = 0.02
GRADUATION_GROSS_FRACTION_NAV = 0.10
ACTION_MINUTES = EQUITY_SESSION_MINUTES - 1


def _execution_config() -> ExecutionConfig:
    return ExecutionConfig(
        gross_target=2.0,
        name_cap_fraction_of_gross=0.05,
        horizon_blend=(0.25, 0.25, 0.25, 0.25),
    )


def _assert_frozen(design: Mapping[str, object], schema: str) -> None:
    if (
        design.get("schema") != schema
        or design.get("repository_commit") != repository_commit()
    ):
        raise ValueError(f"{schema} must run at its exact frozen commit")
    if (
        design.get("official_validation_accessed") is not False
        or design.get("test_accessed") is not False
    ):
        raise ValueError("Experiment 57 access flags differ")


def _create_root(path: Path) -> Path:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(path)
    path.mkdir(parents=True)
    return path


def _source_artifact(root: Path, name: str) -> dict[str, object]:
    return _artifact(root.resolve() / name)


def freeze_stage0(
    *,
    section_a_root: Path,
    section_b_root: Path,
    experiment55_root: Path,
    experiment54_root: Path,
    experiment52_root: Path,
    experiment53_root: Path,
    preregistration: Path,
    output_dir: Path,
) -> Path:
    section_a, _ = _verified_result(section_a_root.resolve(), "result.json")
    section_b, _ = _verified_result(section_b_root.resolve(), "result.json")
    _verified_result(experiment55_root.resolve(), "experiment55_result.json")
    _verified_result(experiment54_root.resolve(), "experiment54_result.json")
    _verified_result(experiment52_root.resolve(), "c0_designation.json")
    _verified_result(experiment53_root.resolve(), "experiment53_result.json")
    if section_a.get("abort") is not False:
        raise ValueError("Experiment 56 Section A did not pass")
    base_design = _read_json(section_b_root / "oof" / "frozen_design.json")
    store = Path(str(base_design["store"]["path"])).resolve()
    dates = tuple(
        pl.read_parquet(store / "date_index.parquet")
        .sort("date_idx")
        .filter(pl.col("trade_date").is_between(TRAIN_START, TRAIN_END))["trade_date"]
    )
    source52_design = _read_json(experiment52_root / "frozen_design.json")
    source55_design = _read_json(experiment55_root / "frozen_design.json")
    root = _create_root(output_dir)
    try:
        cdi = _fetch_cdi(root / "cdi", min(dates), max(dates))
        design = {
            "schema": STAGE0_SCHEMA,
            "status": "frozen",
            "repository_commit": repository_commit(),
            "preregistration": _artifact(preregistration.resolve()),
            "store": {
                "path": str(store),
                "identity": feature_store_identity(store),
                "manifest": _artifact(store / "manifest.json"),
            },
            "inputs": {
                "section_a_root": str(section_a_root.resolve()),
                "section_a_result": _source_artifact(section_a_root, "result.json"),
                "section_b_root": str(section_b_root.resolve()),
                "section_b_result": _source_artifact(section_b_root, "result.json"),
                "oof_archive": section_b["archive"],
                "experiment55_root": str(experiment55_root.resolve()),
                "experiment55_result": _source_artifact(
                    experiment55_root, "experiment55_result.json"
                ),
                "experiment54_root": str(experiment54_root.resolve()),
                "experiment54_result": _source_artifact(
                    experiment54_root, "experiment54_result.json"
                ),
                "experiment52_root": str(experiment52_root.resolve()),
                "experiment52_result": _source_artifact(
                    experiment52_root, "c0_designation.json"
                ),
                "experiment53_root": str(experiment53_root.resolve()),
                "experiment53_result": _source_artifact(
                    experiment53_root, "experiment53_result.json"
                ),
                "roll_schedule": source52_design["roll_schedule"],
                "economics_inputs": source52_design["economics_inputs"],
                "cdi": cdi,
                "bucket_definitions": source55_design["bucket_definitions"],
            },
            "contract": {
                "evaluation_windows": list(FOLDS),
                "estimation_folds": "the two C/A/B folds other than evaluation",
                "thresholds_bps": list(THRESHOLDS_BPS),
                "threshold_applies_to": "expected_net_edge_bps_strictly_greater",
                "horizon_names": list(HORIZON_NAMES),
                "horizon_tie_order": list(HORIZON_NAMES),
                "active_name_ignores_new_events": True,
                "books": ["dollar_neutral", "neutrality_free"],
                "execution_config": _execution_config().to_dict(),
                "teacher_threshold_selection": (
                    "highest mean primary-book net excess on estimation folds; "
                    "tie lower threshold"
                ),
                "bootstrap_block": BOOTSTRAP_BLOCK,
                "bootstrap_replications": BOOTSTRAP_REPLICATIONS,
                "graduation": {
                    "mean_excess_strictly_positive": True,
                    "minimum_mean_deployed_gross_fraction_nav": (
                        GRADUATION_GROSS_FRACTION_NAV
                    ),
                },
            },
            "official_validation_accessed": False,
            "test_accessed": False,
        }
        _atomic_json(root / "frozen_design.json", design)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return root / "frozen_design.json"


def _load_archive(root: Path, design: Mapping[str, object]):
    record = design["inputs"]["oof_archive"]
    return load_discovery_prediction_archive(
        Path(record["prediction"]),
        Path(record["reference"]),
        Path(record["execution_manifest"]),
        Path(str(design["store"]["path"])),
    )


def _event_bundle(
    root: Path, design: Mapping[str, object], archive: object
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], tuple[date, ...]]:
    dates = tuple(
        pl.read_parquet(root / "market_inputs" / "dates.parquet")
        .sort("date_idx")["trade_date"]
    )
    open_price = np.asarray(_load_cache_array(root, "open_price.npy"))
    close_price = np.asarray(_load_cache_array(root, "close_price.npy"))
    observed = np.asarray(_load_cache_array(root, "open_observed.npy"))
    if close_price.shape != open_price.shape:
        raise ValueError("Experiment 57 close prices differ from the TRAIN market axis")
    events, _ = build_state_events(
        ranks=archive.ranks,
        valid=archive.valid,
        refresh_minutes=archive.refresh_minutes,
        adv20_brl=np.asarray(_load_cache_array(root, "adv20_brl.npy")),
        full_spread=np.asarray(_load_cache_array(root, "full_spread.npy")),
        sigma_daily=np.asarray(_load_cache_array(root, "sigma_daily.npy")),
        minute_notional20_brl=np.asarray(
            _load_cache_array(root, "minute_notional20_brl.npy")
        ),
        buckets=design["inputs"]["bucket_definitions"],
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
    return events, edges, dates


def cross_fold_conditional_means(
    *,
    events: Mapping[str, np.ndarray],
    edges: Mapping[str, np.ndarray],
    estimation_days: np.ndarray,
) -> tuple[dict[str, dict[int, float]], pl.DataFrame]:
    day = np.asarray(events["day"], dtype=np.int64)
    state = np.asarray(events["state_cell_id"], dtype=np.int64)
    in_estimation = np.isin(day, np.asarray(estimation_days, dtype=np.int64))
    mappings: dict[str, dict[int, float]] = {}
    rows: list[dict[str, object]] = []
    for horizon in HORIZON_NAMES:
        edge = np.asarray(edges[horizon], dtype=np.float64)
        valid = in_estimation & np.isfinite(edge)
        grouped = (
            pl.DataFrame({"state_cell_id": state[valid], "gross_edge_bps": edge[valid]})
            .group_by("state_cell_id", maintain_order=True)
            .agg(
                pl.len().alias("event_count"),
                pl.col("gross_edge_bps").mean().alias("mean_gross_edge_bps"),
            )
            .sort("state_cell_id")
        )
        mappings[horizon] = {
            int(cell): float(mean)
            for cell, mean in grouped.select(
                "state_cell_id", "mean_gross_edge_bps"
            ).iter_rows()
        }
        rows.extend({"horizon": horizon, **row} for row in grouped.to_dicts())
    return mappings, pl.DataFrame(rows)


def _expected_stack(
    events: Mapping[str, np.ndarray], mappings: Mapping[str, Mapping[int, float]]
) -> np.ndarray:
    state = np.asarray(events["state_cell_id"], dtype=np.int64)
    measured = np.asarray(events["taker_cost_measured_bps"], dtype=np.float64)
    expected = np.stack(
        [
            np.asarray(
                [mappings[horizon].get(int(value), np.nan) for value in state],
                dtype=np.float64,
            )
            for horizon in HORIZON_NAMES
        ],
        axis=1,
    )
    return expected - measured[:, None]


def build_rule_schedule(
    *,
    events: Mapping[str, np.ndarray],
    expected_net_bps: np.ndarray,
    selected_days: np.ndarray,
    threshold_bps: float,
    name_count: int,
) -> tuple[np.ndarray, pl.DataFrame]:
    """Build one signed action path while enforcing the frozen horizon lock."""
    selected_days = np.asarray(selected_days, dtype=np.int64)
    local = {int(value): index for index, value in enumerate(selected_days)}
    schedule = np.zeros(
        (selected_days.size, ACTION_MINUTES, name_count), dtype=np.int8
    )
    safe = np.where(np.isfinite(expected_net_bps), expected_net_bps, -np.inf)
    choice = np.argmax(safe, axis=1)
    chosen = safe[np.arange(safe.shape[0]), choice]
    rows: list[dict[str, object]] = []
    event_day = np.asarray(events["day"], dtype=np.int64)
    event_minute = np.asarray(events["minute"], dtype=np.int64)
    event_name = np.asarray(events["name"], dtype=np.int64)
    direction = np.asarray(events["direction"], dtype=np.int8)
    fixed_horizons = (30, 60, 120, ACTION_MINUTES)
    by_day: dict[int, list[int]] = {}
    for index in np.flatnonzero(np.isin(event_day, selected_days)):
        by_day.setdefault(int(event_day[index]), []).append(int(index))
    for global_day in selected_days:
        local_day = local[int(global_day)]
        active_until = np.full(name_count, -1, dtype=np.int16)
        active_direction = np.zeros(name_count, dtype=np.int8)
        minute_events: dict[int, list[int]] = {}
        for index in by_day.get(int(global_day), []):
            minute_events.setdefault(int(event_minute[index]), []).append(index)
        for minute in range(ACTION_MINUTES):
            expired = active_until <= minute
            active_direction[expired] = 0
            active_until[expired] = -1
            for index in minute_events.get(minute, []):
                name = int(event_name[index])
                if active_until[name] > minute or not (chosen[index] > threshold_bps):
                    continue
                horizon_index = int(choice[index])
                horizon = int(fixed_horizons[horizon_index])
                active_direction[name] = int(direction[index])
                active_until[name] = (
                    ACTION_MINUTES
                    if horizon_index == len(HORIZON_NAMES) - 1
                    else min(ACTION_MINUTES, minute + horizon)
                )
                rows.append(
                    {
                        "day": int(global_day),
                        "action_minute": minute,
                        "name": name,
                        "horizon": HORIZON_NAMES[horizon_index],
                        "expected_net_bps": float(chosen[index]),
                    }
                )
            schedule[local_day, minute] = active_direction
    usage = pl.DataFrame(
        rows,
        schema={
            "day": pl.Int64,
            "action_minute": pl.Int64,
            "name": pl.Int64,
            "horizon": pl.String,
            "expected_net_bps": pl.Float64,
        },
    )
    return schedule, usage


def neutrality_free_projection(
    raw: Tensor, mask: Tensor, caps: Tensor, gross_target: float
) -> Tensor:
    candidate = torch.where(
        mask,
        torch.maximum(torch.minimum(raw, caps), -caps),
        torch.zeros_like(raw),
    )
    gross = candidate.abs().sum(dim=-1)
    needs_scale = gross > gross_target
    denominator = torch.where(needs_scale, gross, torch.ones_like(gross))
    scale = torch.where(needs_scale, gross_target / denominator, 1.0)
    return candidate * scale.unsqueeze(-1)


class RulePathPolicy:
    requires_policy_state = False

    def __init__(self, schedule: Tensor, *, neutral: bool) -> None:
        self.schedule = schedule
        self.neutral = neutral
        self.action = 0
        self.projection_mode = "bounded" if neutral else "bounded_non_neutral"

    def step(
        self,
        ranks: Tensor,
        refresh: Tensor,
        current_weights: Tensor,
        sigma: Tensor,
        previous_target: Tensor,
        initialized: Tensor,
        tradeable_mask: Tensor | None = None,
        cap_weights: Tensor | None = None,
        full_spread: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        del ranks, current_weights, sigma, previous_target, full_spread
        if tradeable_mask is None or cap_weights is None:
            raise ValueError("Rule path requires tradeability and cap weights")
        if self.action >= self.schedule.shape[1]:
            raise RuntimeError("Rule path received too many action steps")
        raw = self.schedule[:, self.action].to(cap_weights) * cap_weights
        self.action += 1
        target = (
            project_weights_bounded(raw, tradeable_mask, cap_weights, 2.0)
            if self.neutral
            else neutrality_free_projection(raw, tradeable_mask, cap_weights, 2.0)
        )
        active = initialized.bool() | refresh.bool()
        return target, active


def _simulate_rule(
    *,
    root: Path,
    design: Mapping[str, object],
    archive: object,
    positions: np.ndarray,
    schedule: np.ndarray,
    neutral: bool,
    return_path: bool = True,
) -> tuple[object, SimulationResult]:
    device = torch.device("cpu")
    batch = _policy_batch(
        root=root,
        design=design,
        archive=archive,
        positions=np.asarray(positions, dtype=np.int64),
        device=device,
    )
    policy = RulePathPolicy(torch.as_tensor(schedule, device=device), neutral=neutral)
    with torch.no_grad():
        result = simulate(
            batch.market,
            batch.ranks,
            batch.rank_valid,
            batch.refresh_mask,
            batch.sigma,
            policy,
            _execution_config(),
            return_path=return_path,
        )
    return batch, result


def _daily_from_result(
    dates: Sequence[date], batch: object, result: SimulationResult
) -> pl.DataFrame:
    config = _execution_config()
    cdi = (batch.market.daily_cdi_rate * config.nav_brl).cpu().numpy()
    return (
        _simulation_rows(dates, result, config.nav_brl)
        .with_columns(pl.Series("all_cash_cdi_pnl_brl", cdi))
        .with_columns(
            (pl.col("net_pnl_brl") - pl.col("all_cash_cdi_pnl_brl")).alias(
                "excess_pnl_brl"
            ),
            (
                (pl.col("net_pnl_brl") - pl.col("all_cash_cdi_pnl_brl"))
                / config.nav_brl
                * 10_000.0
            ).alias("excess_pnl_bps"),
            (pl.col("mean_deployed_gross_brl") / config.nav_brl).alias(
                "mean_deployed_gross_fraction_nav"
            ),
        )
    )


def _rule_diagnostics(
    *,
    dates: Sequence[date],
    batch: object,
    result: SimulationResult,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if result.target_weights is None or result.fills_brl is None:
        raise ValueError("Rule diagnostics require retained replay paths")
    turnover = result.turnover_by_name_brl.cpu().numpy()
    net_by_name = (
        result.gross_pnl_by_name_brl
        - result.spread_cost_by_name_brl
        - result.fees_by_name_brl
    ).cpu().numpy()
    adv = batch.market.adv20_brl.cpu().numpy()
    active = batch.market.active.cpu().numpy().astype(bool)
    liquidity_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    for day_index, trade_date in enumerate(dates):
        names = np.flatnonzero(active[day_index] & np.isfinite(adv[day_index]))
        buckets = np.full(adv.shape[1], -1, dtype=np.int8)
        if names.size:
            order = names[np.argsort(adv[day_index, names], kind="stable")]
            buckets[order] = np.minimum(np.arange(order.size) * 3 // order.size, 2)
        for bucket in range(3):
            take = buckets == bucket
            liquidity_rows.append(
                {
                    "trade_date": trade_date,
                    "liquidity_tercile": bucket,
                    "turnover_brl": float(turnover[day_index, take].sum()),
                    "net_trading_pnl_brl": float(net_by_name[day_index, take].sum()),
                }
            )
        count = result.round_trip_count_by_name[day_index].cpu().numpy()
        gross = result.round_trip_gross_pnl_by_name_brl[day_index].cpu().numpy()
        cost = result.round_trip_cost_by_name_brl[day_index].cpu().numpy()
        trade_rows.append(
            {
                "trade_date": trade_date,
                "round_trip_count": int(count.sum()),
                "gross_edge_brl_per_trade": float(gross.sum() / max(count.sum(), 1)),
                "cost_brl_per_trade": float(cost.sum() / max(count.sum(), 1)),
            }
        )
    targets = result.target_weights.cpu().numpy()
    refresh = batch.refresh_mask.cpu().numpy().astype(bool)
    spread = batch.market.full_spread.cpu().numpy() * 10_000.0
    target_rows: list[dict[str, object]] = []
    for minute in range(ACTION_MINUTES):
        for day_index in np.flatnonzero(refresh[:, minute]):
            change = np.abs(targets[day_index, minute + 1] - targets[day_index, minute])
            valid = active[day_index] & np.isfinite(spread[day_index])
            target_rows.append(
                {
                    "trade_date": dates[day_index],
                    "session_minute": minute,
                    "mean_absolute_target_change": float(change[valid].mean())
                    if valid.any()
                    else 0.0,
                    "mean_full_spread_bps": float(spread[day_index, valid].mean())
                    if valid.any()
                    else 0.0,
                }
            )
    fills = np.abs(result.fills_brl.cpu().numpy())
    tod_rows = []
    for day_index, trade_date in enumerate(dates):
        for third, (start, stop) in enumerate(((0, 135), (135, 270), (270, 405))):
            tod_rows.append(
                {
                    "trade_date": trade_date,
                    "session_third": third,
                    "turnover_brl": float(fills[day_index, start:stop].sum()),
                }
            )
    return (
        pl.DataFrame(liquidity_rows),
        pl.DataFrame(trade_rows),
        pl.DataFrame(target_rows),
        pl.DataFrame(tod_rows),
    )


def _fold_positions(dates: Sequence[date]) -> dict[str, np.ndarray]:
    slices = policy_evaluation_slices(dates, dates)
    by_date = {value: index for index, value in enumerate(dates)}
    return {
        item.name: np.asarray([by_date[value] for value in item.dates], dtype=np.int64)
        for item in slices
    }


def _teacher_summary(
    daily: pl.DataFrame, source_a: Path
) -> tuple[pl.DataFrame, float, float, bool]:
    oracle = {
        row["fold"]: float(row["mean_net_nav_bps_per_day"])
        for row in _read_json(source_a / "result.json")["total_frontiers"]
    }
    rows: list[dict[str, object]] = []
    for index, window in enumerate(FOLDS):
        values = daily.filter(
            (pl.col("window") == window)
            & (pl.col("book") == "dollar_neutral")
            & pl.col("designated")
        )
        excess = values["excess_pnl_bps"].to_numpy()
        row = {
            "readout": f"{window}/net_excess_all_cash_bps",
            **_json_interval(excess, 570 + index),
        }
        rows.append(row)
        rows.append(
            {
                "readout": f"{window}/oracle_capture_ratio",
                "estimate": float(np.mean(excess) / oracle[window]),
                "lower_95": float(row["lower_95"] / oracle[window]),
                "upper_95": float(row["upper_95"] / oracle[window]),
                "oracle_frontier_bps_per_day": oracle[window],
            }
        )
    pooled = daily.filter(
        (pl.col("book") == "dollar_neutral") & pl.col("designated")
    )
    mean_excess = float(pooled["excess_pnl_bps"].mean())
    gross = float(pooled["mean_deployed_gross_fraction_nav"].mean())
    rows.extend(
        [
            {
                "readout": "pooled/net_excess_all_cash_bps",
                **_json_interval(pooled["excess_pnl_bps"].to_numpy(), 580),
            },
            {
                "readout": "pooled/net_sharpe",
                **_sharpe_interval(pooled["net_pnl_bps"].to_numpy(), 581),
            },
            {
                "readout": "pooled/turnover_bps_nav",
                **_json_interval(pooled["turnover_brl"].to_numpy() / 1_000.0, 582),
            },
            {
                "readout": "pooled/mean_deployed_gross_fraction_nav",
                **_json_interval(
                    pooled["mean_deployed_gross_fraction_nav"].to_numpy(), 583
                ),
            },
        ]
    )
    graduated = mean_excess > 0.0 and gross >= GRADUATION_GROSS_FRACTION_NAV
    return pl.DataFrame(rows), mean_excess, gross, graduated


def _stage0_cell_readouts(
    *, daily: pl.DataFrame, design: Mapping[str, object], selected: Mapping[str, float]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    source52 = Path(str(design["inputs"]["experiment52_root"]))
    source53 = Path(str(design["inputs"]["experiment53_root"]))
    joined = []
    for window in FOLDS:
        values = daily.filter(pl.col("window") == window)
        c0 = _load_baseline_daily(source52, window, "band_2p0__blend_equal")
        c1 = _load_baseline_daily(
            source53, window, "k40__band1p5__c1p0__gross1p0__universe_full"
        )
        joined.append(
            values.join(c0, on="trade_date")
            .rename({"baseline_net_pnl_brl": "c0_net_pnl_brl"})
            .join(c1, on="trade_date")
            .rename({"baseline_net_pnl_brl": "c1_net_pnl_brl"})
            .with_columns(
                ((pl.col("net_pnl_brl") - pl.col("c0_net_pnl_brl")) / 1_000.0).alias(
                    "delta_c0_bps"
                ),
                ((pl.col("net_pnl_brl") - pl.col("c1_net_pnl_brl")) / 1_000.0).alias(
                    "delta_c1_bps"
                ),
                (
                    pl.col("mean_deployed_gross_fraction_nav")
                    / _execution_config().gross_target
                ).alias("deployment_fraction_of_gross_cap"),
            )
        )
    comparison = pl.concat(joined).sort(
        "trade_date", "window", "book", "threshold_bps"
    )
    rows = []
    for window_index, window in enumerate(FOLDS):
        for book_index, book in enumerate(("dollar_neutral", "neutrality_free")):
            for threshold_index, threshold in enumerate(THRESHOLDS_BPS):
                values = comparison.filter(
                    (pl.col("window") == window)
                    & (pl.col("book") == book)
                    & (pl.col("threshold_bps") == threshold)
                )
                offset = 720 + window_index * 100 + book_index * 30 + threshold_index
                for metric_index, column in enumerate(
                    (
                        "excess_pnl_bps",
                        "delta_c0_bps",
                        "delta_c1_bps",
                        "turnover_brl",
                        "mean_deployed_gross_fraction_nav",
                        "deployment_fraction_of_gross_cap",
                    )
                ):
                    array = values[column].to_numpy()
                    if column == "turnover_brl":
                        array = array / 1_000.0
                    rows.append(
                        {
                            "window": window,
                            "book": book,
                            "threshold_bps": threshold,
                            "designated": threshold == selected[window],
                            "readout": column,
                            **_json_interval(array, offset + 3 * metric_index),
                        }
                    )
    return comparison, pl.DataFrame(rows)


def run_stage0(root: Path) -> Path:
    root = root.resolve()
    final = root / "result.json"
    if final.exists():
        _verified_result(root, "result.json")
        return final
    design = _read_json(root / "frozen_design.json")
    _assert_frozen(design, STAGE0_SCHEMA)
    _build_full_market_cache(root, design)
    archive = _load_archive(root, design)
    events, edges, dates = _event_bundle(root, design, archive)
    folds = _fold_positions(dates)
    conditional_tables = []
    selection_rows = []
    daily_tables = []
    usage_tables = []
    diagnostic_tables: dict[str, list[pl.DataFrame]] = {
        "liquidity": [],
        "per_trade": [],
        "target_change": [],
        "session_third": [],
    }
    selected_thresholds: dict[str, float] = {}
    target_records: dict[str, dict[str, object]] = {}
    name_count = archive.ranks.shape[2]
    for window in FOLDS:
        estimation_windows = [value for value in FOLDS if value != window]
        estimation_positions = np.sort(
            np.concatenate([folds[value] for value in estimation_windows])
        )
        mappings, table = cross_fold_conditional_means(
            events=events, edges=edges, estimation_days=estimation_positions
        )
        table = table.with_columns(
            pl.lit(window).alias("evaluation_window"),
            pl.lit("+".join(estimation_windows)).alias("estimation_windows"),
        )
        conditional_tables.append(table)
        expected = _expected_stack(events, mappings)
        threshold_values: dict[float, float] = {}
        for threshold in THRESHOLDS_BPS:
            schedule, _ = build_rule_schedule(
                events=events,
                expected_net_bps=expected,
                selected_days=estimation_positions,
                threshold_bps=threshold,
                name_count=name_count,
            )
            batch, replay = _simulate_rule(
                root=root,
                design=design,
                archive=archive,
                positions=estimation_positions,
                schedule=schedule,
                neutral=True,
                return_path=False,
            )
            estimation_dates = tuple(dates[index] for index in estimation_positions)
            mean = float(
                _daily_from_result(estimation_dates, batch, replay)[
                    "excess_pnl_bps"
                ].mean()
            )
            threshold_values[threshold] = mean
            selection_rows.append(
                {
                    "window": window,
                    "threshold_bps": threshold,
                    "mean_estimation_net_excess_bps": mean,
                }
            )
        selected = max(
            THRESHOLDS_BPS,
            key=lambda value: (threshold_values[value], -value),
        )
        selected_thresholds[window] = selected
        evaluation_positions = folds[window]
        evaluation_dates = tuple(dates[index] for index in evaluation_positions)
        for threshold in THRESHOLDS_BPS:
            schedule, usage = build_rule_schedule(
                events=events,
                expected_net_bps=expected,
                selected_days=evaluation_positions,
                threshold_bps=threshold,
                name_count=name_count,
            )
            if usage.height:
                usage_tables.append(
                    usage.with_columns(
                        pl.lit(window).alias("window"),
                        pl.lit(threshold).alias("threshold_bps"),
                        pl.lit(threshold == selected).alias("designated"),
                    )
                )
            for book, neutral in (("dollar_neutral", True), ("neutrality_free", False)):
                batch, replay = _simulate_rule(
                    root=root,
                    design=design,
                    archive=archive,
                    positions=evaluation_positions,
                    schedule=schedule,
                    neutral=neutral,
                )
                daily_tables.append(
                    _daily_from_result(evaluation_dates, batch, replay).with_columns(
                        pl.lit(window).alias("window"),
                        pl.lit(threshold).alias("threshold_bps"),
                        pl.lit(book).alias("book"),
                        pl.lit(threshold == selected).alias("designated"),
                    )
                )
                diagnostics = _rule_diagnostics(
                    dates=evaluation_dates, batch=batch, result=replay
                )
                for key, table_value in zip(diagnostic_tables, diagnostics, strict=True):
                    diagnostic_tables[key].append(
                        table_value.with_columns(
                            pl.lit(window).alias("window"),
                            pl.lit(threshold).alias("threshold_bps"),
                            pl.lit(book).alias("book"),
                            pl.lit(threshold == selected).alias("designated"),
                        )
                    )
                if neutral and threshold == selected:
                    target_path = root / f"teacher_evaluation_targets_{window}.npy"
                    with target_path.open("wb") as output:
                        np.save(
                            output,
                            replay.target_weights.cpu().numpy().astype(np.float32),
                            allow_pickle=False,
                        )
                    target_records[window] = _artifact(target_path)
    conditional_path = root / "conditional_means.parquet"
    pl.concat(conditional_tables).write_parquet(conditional_path)
    selection_path = root / "teacher_threshold_selection.parquet"
    pl.DataFrame(selection_rows).with_columns(
        pl.struct("window", "threshold_bps")
        .map_elements(
            lambda row: row["threshold_bps"] == selected_thresholds[row["window"]],
            return_dtype=pl.Boolean,
        )
        .alias("designated")
    ).write_parquet(selection_path)
    daily = pl.concat(daily_tables).sort("trade_date", "window", "book", "threshold_bps")
    daily_path = root / "replay_daily.parquet"
    daily.write_parquet(daily_path)
    usage_path = root / "horizon_usage.parquet"
    usage = pl.concat(usage_tables, how="diagonal_relaxed")
    usage.write_parquet(usage_path)
    usage_summary_path = root / "horizon_usage_summary.parquet"
    (
        usage.group_by("window", "threshold_bps", "designated", "horizon")
        .agg(pl.len().alias("assignment_count"))
        .with_columns(
            (
                pl.col("assignment_count")
                / pl.col("assignment_count").sum().over("window", "threshold_bps")
            ).alias("assignment_share")
        )
        .sort("window", "threshold_bps", "horizon")
        .write_parquet(usage_summary_path)
    )
    diagnostic_records = {}
    for key, tables in diagnostic_tables.items():
        path = root / f"{key}.parquet"
        pl.concat(tables, how="diagonal_relaxed").write_parquet(path)
        diagnostic_records[key] = _artifact(path)
    comparison, cell_readouts = _stage0_cell_readouts(
        daily=daily, design=design, selected=selected_thresholds
    )
    comparison_path = root / "comparison_daily.parquet"
    comparison.write_parquet(comparison_path)
    cell_readouts_path = root / "cell_readouts.parquet"
    cell_readouts.write_parquet(cell_readouts_path)
    readouts, mean_excess, mean_gross, graduated = _teacher_summary(
        daily, Path(str(design["inputs"]["section_a_root"]))
    )
    readouts_path = root / "readouts.parquet"
    readouts.write_parquet(readouts_path)
    result = {
        "schema": STAGE0_SCHEMA,
        "status": "completed",
        "repository_commit": repository_commit(),
        "selected_threshold_by_window": selected_thresholds,
        "pooled_mean_daily_net_excess_all_cash_bps": mean_excess,
        "pooled_mean_deployed_gross_fraction_nav": mean_gross,
        "teacher_graduated": graduated,
        "artifacts": {
            "conditional_means": _artifact(conditional_path),
            "teacher_threshold_selection": _artifact(selection_path),
            "replay_daily": _artifact(daily_path),
            "horizon_usage": _artifact(usage_path),
            "horizon_usage_summary": _artifact(usage_summary_path),
            "readouts": _artifact(readouts_path),
            "comparison_daily": _artifact(comparison_path),
            "cell_readouts": _artifact(cell_readouts_path),
            "teacher_evaluation_targets": target_records,
            **diagnostic_records,
        },
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(final, result)
    _atomic_json(
        root / "final_audit.json",
        {
            "schema": "EXPERIMENT57_STAGE0_AUDIT_V1",
            "status": "passed",
            "result_sha256": _sha256(final),
            "all_three_rotations_cross_fold": True,
            "all_three_thresholds_and_two_books_replayed": True,
            "teacher_targets_retained": len(target_records) == 3,
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    return final


def freeze_stage1(
    *, stage0_root: Path, preregistration: Path, output_dir: Path
) -> Path:
    stage0, _ = _verified_result(stage0_root.resolve(), "result.json")
    stage0_design = _read_json(stage0_root / "frozen_design.json")
    root = _create_root(output_dir)
    try:
        design = {
            "schema": STAGE1_SCHEMA,
            "status": "frozen",
            "repository_commit": repository_commit(),
            "preregistration": _artifact(preregistration.resolve()),
            "inputs": {
                "stage0_root": str(stage0_root.resolve()),
                "stage0_result": _artifact(stage0_root / "result.json"),
                "stage0_design": _artifact(stage0_root / "frozen_design.json"),
                "oof_archive": stage0_design["inputs"]["oof_archive"],
            },
            "contract": {
                "windows": list(FOLDS),
                "seeds": list(POLICY_SEEDS),
                "run_count": 9,
                "epochs": CLONE_EPOCHS,
                "fit": "chronological pre-window dates excluding selection and embargo",
                "selection": "last floor-20% of pre-embargo dates; report only",
                "loss": "MSE post-projection weights over tradeable fit scan",
                "teacher_state_trajectory": True,
                "optimizer": "AdamW(lr=0.001,weight_decay=0.01)",
                "gradient_clip_norm": 1.0,
                "sam": False,
                "execution_config": _execution_config().to_dict(),
            },
            "stage0_teacher_graduated": stage0["teacher_graduated"],
            "official_validation_accessed": False,
            "test_accessed": False,
        }
        _atomic_json(root / "frozen_design.json", design)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return root / "frozen_design.json"


def _torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _mapping_for_window(stage0_root: Path, window: str) -> dict[str, dict[int, float]]:
    table = pl.read_parquet(stage0_root / "conditional_means.parquet").filter(
        pl.col("evaluation_window") == window
    )
    return {
        horizon: {
            int(cell): float(mean)
            for cell, mean in table.filter(pl.col("horizon") == horizon).select(
                "state_cell_id", "mean_gross_edge_bps"
            ).iter_rows()
        }
        for horizon in HORIZON_NAMES
    }


class FixedTargetPolicy:
    projection_mode = "bounded"
    requires_policy_state = False

    def __init__(self, target_path: Tensor) -> None:
        self.target_path = target_path
        self.action = 0

    def step(
        self,
        ranks: Tensor,
        refresh: Tensor,
        current_weights: Tensor,
        sigma: Tensor,
        previous_target: Tensor,
        initialized: Tensor,
        tradeable_mask: Tensor | None = None,
        cap_weights: Tensor | None = None,
        full_spread: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        del ranks, current_weights, sigma, previous_target, tradeable_mask
        del cap_weights, full_spread
        if self.action + 1 >= self.target_path.shape[1]:
            raise RuntimeError("Fixed teacher path received too many action steps")
        target = self.target_path[:, self.action + 1]
        self.action += 1
        return target, initialized.bool() | refresh.bool()


class ImitationPolicy:
    projection_mode = "bounded"
    requires_policy_state = True

    def __init__(
        self, student: NeuralPolicy, teacher_path: Tensor, denominator: int
    ) -> None:
        self.student = student
        self.teacher_path = teacher_path
        self.denominator = denominator
        self.action = 0
        self.loss_sum = 0.0
        self.horizon_names = student.horizon_names

    def step(self, *args: object) -> tuple[Tensor, Tensor]:
        teacher = self.teacher_path[:, self.action + 1]
        student_target, active = self.student.step(*args)
        state = args[-1]
        mask = state.tradeable_mask
        squared = torch.where(mask, (student_target - teacher).square(), 0).sum()
        (squared / self.denominator).backward()
        self.loss_sum += float(squared.detach())
        self.action += 1
        return teacher, active


def _valid_imitation_count(batch: object, config: ExecutionConfig) -> int:
    base = tradeable_universe(batch.market, config)
    valid = batch.rank_valid[:, :ACTION_MINUTES].all(dim=-1)
    count = int((base[:, None, :] & valid).sum().item())
    if count <= 0:
        raise ValueError("Imitation fit scan has no tradeable observations")
    return count


def _run_clone_epoch(
    *,
    policy: NeuralPolicy,
    optimizer: torch.optim.Optimizer,
    batch: object,
    teacher_path: Tensor,
    denominator: int,
    config: ExecutionConfig,
) -> tuple[float, float]:
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    wrapper = ImitationPolicy(policy, teacher_path, denominator)
    simulate(
        batch.market,
        batch.ranks,
        batch.rank_valid,
        batch.refresh_mask,
        batch.sigma,
        wrapper,
        config,
    )
    norm = clip_grad_norm_(policy.parameters(), 1.0)
    if not math.isfinite(float(norm)):
        raise FloatingPointError("Clone gradient norm is not finite")
    optimizer.step()
    return wrapper.loss_sum / denominator, float(norm)


def _replay_policy(batch: object, policy: object, *, return_path: bool = True):
    with torch.no_grad():
        return simulate(
            batch.market,
            batch.ranks,
            batch.rank_valid,
            batch.refresh_mask,
            batch.sigma,
            policy,
            _execution_config(),
            return_path=return_path,
        )


def _weight_correlations(
    clone: np.ndarray, teacher: np.ndarray
) -> tuple[float, float]:
    left = clone.reshape(-1).astype(np.float64)
    right = teacher.reshape(-1).astype(np.float64)

    def correlation(mask: np.ndarray) -> float:
        if mask.sum() < 2:
            return 0.0
        x, y = left[mask], right[mask]
        if x.std() == 0 or y.std() == 0:
            return 1.0 if np.array_equal(x, y) else 0.0
        return float(np.corrcoef(x, y)[0, 1])

    return correlation(np.ones(left.size, dtype=bool)), correlation(right != 0.0)


def _clone_run_dir(root: Path, window: str, seed: int) -> Path:
    return root / "runs" / window / f"seed_{seed}"


def _save_clone_checkpoint(
    path: Path, policy: NeuralPolicy, seed: int, optimizer: object | None = None
) -> None:
    _torch_save(
        path,
        {
            "schema": "EXPERIMENT57_CLONE_CHECKPOINT_V1",
            "repository_commit": repository_commit(),
            "seed": seed,
            "policy_contract": policy.contract_metadata,
            "policy_state": policy.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        },
    )


def _load_clone(path: Path, device: torch.device, seed: int) -> NeuralPolicy:
    policy = NeuralPolicy(
        _execution_config(),
        horizon_count=4,
        horizon_names=HORIZON_NAMES,
        seed=seed,
    ).to(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    if (
        payload.get("schema") != "EXPERIMENT57_CLONE_CHECKPOINT_V1"
        or payload.get("seed") != seed
        or payload.get("policy_contract") != policy.contract_metadata
    ):
        raise ValueError("Clone checkpoint contract differs")
    policy.load_state_dict(payload["policy_state"])
    return policy


def run_stage1(root: Path, device_name: str = "cuda") -> Path:
    root = root.resolve()
    final = root / "result.json"
    if final.exists():
        _verified_result(root, "result.json")
        return final
    design = _read_json(root / "frozen_design.json")
    _assert_frozen(design, STAGE1_SCHEMA)
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Experiment 57 cloning requires the claimed CUDA instance")
    stage0_root = Path(str(design["inputs"]["stage0_root"]))
    stage0_design = _read_json(stage0_root / "frozen_design.json")
    _verified_result(stage0_root, "result.json")
    archive = _load_archive(stage0_root, stage0_design)
    events, edges, dates = _event_bundle(stage0_root, stage0_design, archive)
    splits = {value.window: value for value in policy_splits(dates)}
    selected_thresholds = _read_json(stage0_root / "result.json")[
        "selected_threshold_by_window"
    ]
    path_records = {}
    run_manifests = []
    for window in FOLDS:
        split = splits[window]
        pre_positions = np.concatenate((split.fit, split.selection))
        mappings = _mapping_for_window(stage0_root, window)
        expected = _expected_stack(events, mappings)
        schedule, usage = build_rule_schedule(
            events=events,
            expected_net_bps=expected,
            selected_days=pre_positions,
            threshold_bps=float(selected_thresholds[window]),
            name_count=archive.ranks.shape[2],
        )
        _, teacher_full = _simulate_rule(
            root=stage0_root,
            design=stage0_design,
            archive=archive,
            positions=pre_positions,
            schedule=schedule,
            neutral=True,
        )
        path = root / "teacher_paths" / f"{window}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        teacher_numpy = teacher_full.target_weights.cpu().numpy().astype(np.float32)
        with path.open("wb") as output:
            np.save(output, teacher_numpy, allow_pickle=False)
        usage_path = root / "teacher_paths" / f"{window}_usage.parquet"
        usage.write_parquet(usage_path)
        path_records[window] = {
            "target_weights": _artifact(path),
            "usage": _artifact(usage_path),
            "date_count": int(pre_positions.size),
            "fit_date_count": int(split.fit.size),
            "selection_date_count": int(split.selection.size),
            "embargo_sessions": int(split.evaluation[0] - split.selection[-1] - 1),
        }
        fit_count = split.fit.size
        fit_teacher = torch.as_tensor(
            teacher_numpy[:fit_count], dtype=torch.float32, device=device
        )
        selection_teacher = torch.as_tensor(
            teacher_numpy[fit_count:], dtype=torch.float32, device=device
        )
        fit_batch = _policy_batch(
            root=stage0_root,
            design=stage0_design,
            archive=archive,
            positions=split.fit,
            device=device,
        )
        selection_batch = _policy_batch(
            root=stage0_root,
            design=stage0_design,
            archive=archive,
            positions=split.selection,
            device=device,
        )
        denominator = _valid_imitation_count(fit_batch, _execution_config())
        selection_dates = tuple(dates[index] for index in split.selection)
        teacher_selection = _replay_policy(
            selection_batch, FixedTargetPolicy(selection_teacher), return_path=True
        )
        teacher_daily = _daily_from_result(
            selection_dates, selection_batch, teacher_selection
        ).select(
            "trade_date",
            pl.col("net_pnl_brl").alias("teacher_net_pnl_brl"),
            pl.col("excess_pnl_bps").alias("teacher_excess_pnl_bps"),
            pl.col("mean_deployed_gross_fraction_nav").alias(
                "teacher_mean_deployed_gross_fraction_nav"
            ),
        )
        for seed in POLICY_SEEDS:
            run = _clone_run_dir(root, window, seed)
            manifest_path = run / "run_manifest.json"
            if manifest_path.is_file():
                run_manifests.append(manifest_path)
                continue
            run.mkdir(parents=True, exist_ok=True)
            policy = NeuralPolicy(
                _execution_config(),
                horizon_count=4,
                horizon_names=HORIZON_NAMES,
                seed=seed,
            ).to(device)
            optimizer = torch.optim.AdamW(
                policy.parameters(), lr=1e-3, weight_decay=0.01
            )
            progress = run / "progress.pt"
            history: list[dict[str, object]] = []
            start_epoch = 1
            if progress.is_file():
                payload = torch.load(progress, map_location=device, weights_only=False)
                policy.load_state_dict(payload["policy_state"])
                optimizer.load_state_dict(payload["optimizer_state"])
                history = payload["history"]
                start_epoch = int(history[-1]["epoch"]) + 1
            for epoch in range(start_epoch, CLONE_EPOCHS + 1):
                mse, gradient_norm = _run_clone_epoch(
                    policy=policy,
                    optimizer=optimizer,
                    batch=fit_batch,
                    teacher_path=fit_teacher,
                    denominator=denominator,
                    config=_execution_config(),
                )
                history.append(
                    {
                        "epoch": epoch,
                        "fit_weight_mse": mse,
                        "gradient_norm": gradient_norm,
                    }
                )
                _torch_save(
                    progress,
                    {
                        "policy_state": policy.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "history": history,
                    },
                )
            checkpoint = run / "clone.pt"
            _save_clone_checkpoint(checkpoint, policy, seed)
            progress.unlink(missing_ok=True)
            history_path = run / "history.parquet"
            pl.DataFrame(history).write_parquet(history_path)
            policy.eval()
            clone_result = _replay_policy(selection_batch, policy, return_path=True)
            clone_daily = _daily_from_result(
                selection_dates, selection_batch, clone_result
            ).select(
                "trade_date",
                pl.col("net_pnl_brl").alias("clone_net_pnl_brl"),
                pl.col("excess_pnl_bps").alias("clone_excess_pnl_bps"),
                pl.col("mean_deployed_gross_fraction_nav").alias(
                    "clone_mean_deployed_gross_fraction_nav"
                ),
            )
            daily_path = run / "selection_daily.parquet"
            clone_daily.join(teacher_daily, on="trade_date").with_columns(
                (pl.col("clone_net_pnl_brl") - pl.col("teacher_net_pnl_brl")).alias(
                    "clone_minus_teacher_net_pnl_brl"
                )
            ).write_parquet(daily_path)
            all_corr, active_corr = _weight_correlations(
                clone_result.target_weights.cpu().numpy(),
                selection_teacher.cpu().numpy(),
            )
            quality_path = run / "clone_quality.json"
            _atomic_json(
                quality_path,
                {
                    "weight_correlation_all": all_corr,
                    "weight_correlation_teacher_nonzero": active_corr,
                    "clone_mean_selection_excess_bps": float(
                        clone_daily["clone_excess_pnl_bps"].mean()
                    ),
                    "teacher_mean_selection_excess_bps": float(
                        teacher_daily["teacher_excess_pnl_bps"].mean()
                    ),
                    "clone_mean_deployed_gross_fraction_nav": float(
                        clone_daily["clone_mean_deployed_gross_fraction_nav"].mean()
                    ),
                    "teacher_mean_deployed_gross_fraction_nav": float(
                        teacher_daily[
                            "teacher_mean_deployed_gross_fraction_nav"
                        ].mean()
                    ),
                    "gate": None,
                },
            )
            artifacts = (checkpoint, history_path, daily_path, quality_path)
            manifest = {
                "schema": "EXPERIMENT57_CLONE_RUN_V1",
                "status": "completed",
                "repository_commit": repository_commit(),
                "window": window,
                "seed": seed,
                "epochs_completed": len(history),
                "fit_date_count": int(split.fit.size),
                "selection_date_count": int(split.selection.size),
                "embargo_sessions": int(
                    split.evaluation[0] - split.selection[-1] - 1
                ),
                "imitation_observation_count": denominator,
                "artifacts": {value.name: _artifact(value) for value in artifacts},
                "official_validation_accessed": False,
                "test_accessed": False,
            }
            _atomic_json(manifest_path, manifest)
            run_manifests.append(manifest_path)
            torch.cuda.empty_cache()
    if len(run_manifests) != 9:
        raise ValueError("Experiment 57 Stage 1 did not complete exactly nine clones")
    result = {
        "schema": STAGE1_SCHEMA,
        "status": "completed",
        "repository_commit": repository_commit(),
        "run_count": 9,
        "epoch_count": 9 * CLONE_EPOCHS,
        "teacher_paths": path_records,
        "runs": [_artifact(path) for path in run_manifests],
        "clone_quality_is_a_gate": False,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(final, result)
    _atomic_json(
        root / "final_audit.json",
        {
            "schema": "EXPERIMENT57_STAGE1_AUDIT_V1",
            "status": "passed",
            "result_sha256": _sha256(final),
            "all_nine_clones_completed": True,
            "all_180_clone_epochs_completed": True,
            "teacher_paths_hash_bound": len(path_records) == 3,
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    return final


def freeze_stage2(
    *,
    stage0_root: Path,
    stage1_root: Path,
    preregistration: Path,
    output_dir: Path,
) -> Path:
    stage0, _ = _verified_result(stage0_root.resolve(), "result.json")
    stage1, _ = _verified_result(stage1_root.resolve(), "result.json")
    if stage1.get("run_count") != 9:
        raise ValueError("Stage 1 lacks the frozen nine clones")
    root = _create_root(output_dir)
    try:
        design = {
            "schema": STAGE2_SCHEMA,
            "status": "frozen",
            "repository_commit": repository_commit(),
            "preregistration": _artifact(preregistration.resolve()),
            "inputs": {
                "stage0_root": str(stage0_root.resolve()),
                "stage0_result": _artifact(stage0_root / "result.json"),
                "stage1_root": str(stage1_root.resolve()),
                "stage1_result": _artifact(stage1_root / "result.json"),
            },
            "contract": {
                "windows": list(FOLDS),
                "seeds": list(POLICY_SEEDS),
                "run_count": 9,
                "risk_lambda": RISK_LAMBDA,
                "maximum_epochs": MAX_POLICY_EPOCHS,
                "patience": PATIENCE,
                "optimizer": "AdamW(lr=0.001,weight_decay=0.01)",
                "gradient_clip_norm": 1.0,
                "sam": False,
                "objective": "mean_excess_bps-0.02*population_std_net_bps",
                "designation": (
                    "fine-tuned unless best fine-tuned selection objective is "
                    "strictly below own clone; tie fine-tuned"
                ),
                "seed_aggregation": "uniform daily mean",
                "execution_config": _execution_config().to_dict(),
                "graduation": {
                    "mean_excess_strictly_positive": True,
                    "minimum_mean_deployed_gross_fraction_nav": (
                        GRADUATION_GROSS_FRACTION_NAV
                    ),
                    "teacher_outranks_nonbeating_neural": True,
                },
            },
            "stage0_teacher_graduated": stage0["teacher_graduated"],
            "official_validation_accessed": False,
            "test_accessed": False,
        }
        _atomic_json(root / "frozen_design.json", design)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return root / "frozen_design.json"


def _selection_metrics(
    batch: object, policy: object, risk_lambda: float
) -> tuple[float, float, float, SimulationResult]:
    result = _replay_policy(batch, policy, return_path=True)
    objective, excess, standard_deviation = policy_objective(
        result.net_pnl_brl,
        batch.market.daily_cdi_rate,
        _execution_config().nav_brl,
        risk_lambda,
    )
    return (
        float(objective),
        float(excess.mean()),
        float(standard_deviation),
        result,
    )


def _stage2_run_dir(root: Path, window: str, seed: int) -> Path:
    return root / "runs" / window / f"seed_{seed}"


def _save_designated(
    path: Path,
    policy: NeuralPolicy,
    *,
    window: str,
    seed: int,
    kind: str,
    selection_objective_bps: float,
) -> None:
    _torch_save(
        path,
        {
            "schema": "EXPERIMENT57_DESIGNATED_POLICY_V1",
            "repository_commit": repository_commit(),
            "window": window,
            "seed": seed,
            "kind": kind,
            "selection_objective_bps": selection_objective_bps,
            "policy_contract": policy.contract_metadata,
            "policy_state": policy.state_dict(),
        },
    )


def _evaluate_ladder(
    *,
    run: Path,
    window: str,
    seed: int,
    dates: Sequence[date],
    batch: object,
    teacher_path: Tensor,
    clone: NeuralPolicy,
    fine_tuned: NeuralPolicy,
    designated: NeuralPolicy,
) -> tuple[Path, list[Path]]:
    rows = []
    results: dict[str, SimulationResult] = {}
    policies: tuple[tuple[str, object], ...] = (
        ("teacher", FixedTargetPolicy(teacher_path)),
        ("clone", clone),
        ("fine_tuned", fine_tuned),
        ("designated", designated),
    )
    for kind, policy in policies:
        result = _replay_policy(batch, policy, return_path=True)
        results[kind] = result
        rows.append(
            _daily_from_result(dates, batch, result).with_columns(
                pl.lit(window).alias("window"),
                pl.lit(seed).alias("seed"),
                pl.lit(kind).alias("policy_kind"),
            )
        )
    daily_path = run / "evaluation_ladder_daily.parquet"
    pl.concat(rows).write_parquet(daily_path)
    diagnostics = _rule_diagnostics(
        dates=dates, batch=batch, result=results["designated"]
    )
    paths = []
    for name, table in zip(
        ("liquidity", "per_trade", "target_change", "session_third"),
        diagnostics,
        strict=True,
    ):
        path = run / f"designated_{name}.parquet"
        table.with_columns(
            pl.lit(window).alias("window"), pl.lit(seed).alias("seed")
        ).write_parquet(path)
        paths.append(path)
    return daily_path, paths


def _run_fine_tune(
    *,
    root: Path,
    stage0_root: Path,
    stage1_root: Path,
    stage0_design: Mapping[str, object],
    archive: object,
    dates: Sequence[date],
    split: object,
    seed: int,
    teacher_path: Tensor,
    device: torch.device,
) -> Path:
    run = _stage2_run_dir(root, split.window, seed)
    manifest_path = run / "run_manifest.json"
    if manifest_path.is_file():
        return manifest_path
    run.mkdir(parents=True, exist_ok=True)
    clone_path = _clone_run_dir(stage1_root, split.window, seed) / "clone.pt"
    clone = _load_clone(clone_path, device, seed)
    clone_state = copy.deepcopy(clone.state_dict())
    fit = _policy_batch(
        root=stage0_root,
        design=stage0_design,
        archive=archive,
        positions=split.fit,
        device=device,
    )
    selection = _policy_batch(
        root=stage0_root,
        design=stage0_design,
        archive=archive,
        positions=split.selection,
        device=device,
    )
    evaluation = _policy_batch(
        root=stage0_root,
        design=stage0_design,
        archive=archive,
        positions=split.evaluation,
        device=device,
    )
    clone_objective, clone_excess, clone_std, clone_selection = _selection_metrics(
        selection, clone, RISK_LAMBDA
    )
    clone_selection_gross = float(
        clone_selection.mean_deployed_gross_brl.mean()
        / _execution_config().nav_brl
    )
    trainer_config = PolicyTrainerConfig(
        learning_rate=1e-3,
        weight_decay=0.01,
        risk_aversion=RISK_LAMBDA,
        gradient_clip_norm=1.0,
        seed=seed,
        use_sam=False,
        gradient_checkpointing=True,
        patience=PATIENCE,
    )
    trainer = PolicyTrainer(clone, _execution_config(), trainer_config)
    history: list[dict[str, object]] = [
        {
            "epoch": 0,
            "checkpoint_kind": "clone",
            "selection_objective_bps": clone_objective,
            "selection_mean_excess_bps": clone_excess,
            "selection_daily_net_std_bps": clone_std,
            "selection_mean_deployed_gross_fraction_nav": clone_selection_gross,
            "gradient_norm": 0.0,
        }
    ]
    progress = run / "progress.pt"
    start_epoch = 1
    if progress.is_file():
        trainer.load_checkpoint(progress)
        history = json.loads((run / "history.partial.json").read_text(encoding="utf-8"))
        start_epoch = int(history[-1]["epoch"]) + 1
    for epoch in range(start_epoch, MAX_POLICY_EPOCHS + 1):
        metrics = trainer.train_step(fit)
        clone.eval()
        objective, excess, standard_deviation, selection_result = _selection_metrics(
            selection, clone, RISK_LAMBDA
        )
        exhausted = trainer.update_monitor(objective)
        history.append(
            {
                "epoch": epoch,
                "checkpoint_kind": "fine_tuned",
                **asdict(metrics),
                "selection_objective_bps": objective,
                "selection_mean_excess_bps": excess,
                "selection_daily_net_std_bps": standard_deviation,
                "selection_mean_deployed_gross_fraction_nav": float(
                    selection_result.mean_deployed_gross_brl.mean()
                    / _execution_config().nav_brl
                ),
                "best_fine_tuned_selection_objective_bps": float(
                    trainer.best_monitor
                ),
            }
        )
        trainer.save_checkpoint(progress)
        _atomic_json(run / "history.partial.json", history)
        if exhausted:
            break
    trainer.restore_best()
    fine_tuned_path = run / "fine_tuned.pt"
    trainer.save_checkpoint(fine_tuned_path)
    progress.unlink(missing_ok=True)
    (run / "history.partial.json").unlink(missing_ok=True)
    history_path = run / "history.parquet"
    pl.DataFrame(history).write_parquet(history_path)
    fine_state = copy.deepcopy(clone.state_dict())
    best_fine = float(trainer.best_monitor)
    designate_fine = best_fine >= clone_objective
    designated_kind = "fine_tuned" if designate_fine else "clone"
    designated_objective = best_fine if designate_fine else clone_objective
    if not designate_fine:
        clone.load_state_dict(clone_state)
    designated_state = copy.deepcopy(clone.state_dict())
    designated_path = run / "designated.pt"
    _save_designated(
        designated_path,
        clone,
        window=split.window,
        seed=seed,
        kind=designated_kind,
        selection_objective_bps=designated_objective,
    )
    clone_policy = NeuralPolicy(
        _execution_config(),
        horizon_count=4,
        horizon_names=HORIZON_NAMES,
        seed=seed,
    ).to(device)
    clone_policy.load_state_dict(clone_state)
    fine_policy = NeuralPolicy(
        _execution_config(),
        horizon_count=4,
        horizon_names=HORIZON_NAMES,
        seed=seed,
    ).to(device)
    fine_policy.load_state_dict(fine_state)
    designated_policy = NeuralPolicy(
        _execution_config(),
        horizon_count=4,
        horizon_names=HORIZON_NAMES,
        seed=seed,
    ).to(device)
    designated_policy.load_state_dict(designated_state)
    evaluation_dates = tuple(dates[index] for index in split.evaluation)
    daily_path, diagnostic_paths = _evaluate_ladder(
        run=run,
        window=split.window,
        seed=seed,
        dates=evaluation_dates,
        batch=evaluation,
        teacher_path=teacher_path,
        clone=clone_policy,
        fine_tuned=fine_policy,
        designated=designated_policy,
    )
    artifacts = [
        history_path,
        fine_tuned_path,
        designated_path,
        daily_path,
        *diagnostic_paths,
    ]
    manifest = {
        "schema": "EXPERIMENT57_FINE_TUNE_RUN_V1",
        "status": "completed",
        "repository_commit": repository_commit(),
        "window": split.window,
        "seed": seed,
        "risk_lambda": RISK_LAMBDA,
        "clone_selection_objective_bps": clone_objective,
        "best_fine_tuned_selection_objective_bps": best_fine,
        "designated_kind": designated_kind,
        "designated_selection_objective_bps": designated_objective,
        "epochs_completed": len(history) - 1,
        "fit_date_count": int(split.fit.size),
        "selection_date_count": int(split.selection.size),
        "evaluation_date_count": int(split.evaluation.size),
        "embargo_sessions": int(split.evaluation[0] - split.selection[-1] - 1),
        "trainer_config": asdict(trainer_config),
        "clone_checkpoint": _artifact(clone_path),
        "artifacts": {value.name: _artifact(value) for value in artifacts},
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(manifest_path, manifest)
    return manifest_path


def _stage2_readouts(
    *, root: Path, stage0_root: Path, stage0_design: Mapping[str, object]
) -> tuple[Path, Path, dict[str, object]]:
    tables = [
        pl.read_parquet(path)
        for path in sorted((root / "runs").glob("*/seed_*/evaluation_ladder_daily.parquet"))
    ]
    ladder = (
        pl.concat(tables)
        .group_by("window", "trade_date", "policy_kind", maintain_order=True)
        .agg(pl.exclude("window", "trade_date", "policy_kind", "seed").mean())
        .sort("trade_date", "window", "policy_kind")
    )
    ladder_path = root / "ladder_daily.parquet"
    ladder.write_parquet(ladder_path)
    oracle = {
        row["fold"]: float(row["mean_net_nav_bps_per_day"])
        for row in _read_json(
            Path(str(stage0_design["inputs"]["section_a_root"])) / "result.json"
        )["total_frontiers"]
    }
    rows = []
    for kind_index, kind in enumerate(("teacher", "clone", "fine_tuned", "designated")):
        for window_index, window in enumerate(FOLDS):
            values = ladder.filter(
                (pl.col("policy_kind") == kind) & (pl.col("window") == window)
            )
            excess = values["excess_pnl_bps"].to_numpy()
            interval = _json_interval(excess, 600 + 10 * kind_index + window_index)
            rows.append(
                {
                    "policy_kind": kind,
                    "window": window,
                    "readout": "net_excess_all_cash_bps",
                    **interval,
                }
            )
            rows.append(
                {
                    "policy_kind": kind,
                    "window": window,
                    "readout": "oracle_capture_ratio",
                    "estimate": float(excess.mean() / oracle[window]),
                    "lower_95": float(interval["lower_95"] / oracle[window]),
                    "upper_95": float(interval["upper_95"] / oracle[window]),
                }
            )
        pooled = ladder.filter(pl.col("policy_kind") == kind)
        rows.extend(
            [
                {
                    "policy_kind": kind,
                    "window": "pooled",
                    "readout": "net_excess_all_cash_bps",
                    **_json_interval(
                        pooled["excess_pnl_bps"].to_numpy(), 680 + kind_index
                    ),
                },
                {
                    "policy_kind": kind,
                    "window": "pooled",
                    "readout": "net_sharpe",
                    **_sharpe_interval(
                        pooled["net_pnl_bps"].to_numpy(), 690 + kind_index
                    ),
                },
                {
                    "policy_kind": kind,
                    "window": "pooled",
                    "readout": "mean_deployed_gross_fraction_nav",
                    **_json_interval(
                        pooled["mean_deployed_gross_fraction_nav"].to_numpy(),
                        700 + kind_index,
                    ),
                },
                {
                    "policy_kind": kind,
                    "window": "pooled",
                    "readout": "deployment_fraction_of_gross_cap",
                    **_json_interval(
                        pooled["mean_deployed_gross_fraction_nav"].to_numpy()
                        / _execution_config().gross_target,
                        705 + kind_index,
                    ),
                },
                {
                    "policy_kind": kind,
                    "window": "pooled",
                    "readout": "turnover_bps_nav",
                    **_json_interval(
                        pooled["turnover_brl"].to_numpy() / 1_000.0,
                        710 + kind_index,
                    ),
                },
            ]
        )
    designated_daily = ladder.filter(pl.col("policy_kind") == "designated")
    source52 = Path(str(stage0_design["inputs"]["experiment52_root"]))
    source53 = Path(str(stage0_design["inputs"]["experiment53_root"]))
    comparisons = []
    for window in FOLDS:
        values = designated_daily.filter(pl.col("window") == window)
        c0 = _load_baseline_daily(source52, window, "band_2p0__blend_equal")
        c1 = _load_baseline_daily(
            source53, window, "k40__band1p5__c1p0__gross1p0__universe_full"
        )
        comparisons.append(
            values.join(c0, on="trade_date")
            .rename({"baseline_net_pnl_brl": "c0_net_pnl_brl"})
            .join(c1, on="trade_date")
            .rename({"baseline_net_pnl_brl": "c1_net_pnl_brl"})
            .with_columns(
                ((pl.col("net_pnl_brl") - pl.col("c0_net_pnl_brl")) / 1_000.0).alias(
                    "delta_c0_bps"
                ),
                ((pl.col("net_pnl_brl") - pl.col("c1_net_pnl_brl")) / 1_000.0).alias(
                    "delta_c1_bps"
                ),
            )
        )
    comparison = pl.concat(comparisons)
    comparison_path = root / "designated_comparisons.parquet"
    comparison.write_parquet(comparison_path)
    for window_index, window in enumerate(FOLDS):
        values = comparison.filter(pl.col("window") == window)
        for metric_index, column in enumerate(("delta_c0_bps", "delta_c1_bps")):
            rows.append(
                {
                    "policy_kind": "designated",
                    "window": window,
                    "readout": column,
                    **_json_interval(
                        values[column].to_numpy(),
                        730 + window_index * 4 + metric_index,
                    ),
                }
            )
    for metric_index, column in enumerate(("delta_c0_bps", "delta_c1_bps")):
        rows.append(
            {
                "policy_kind": "designated",
                "window": "pooled",
                "readout": column,
                **_json_interval(comparison[column].to_numpy(), 750 + metric_index),
            }
        )
    readouts_path = root / "readouts.parquet"
    pl.DataFrame(rows).write_parquet(readouts_path)
    neural_excess = float(designated_daily["excess_pnl_bps"].mean())
    neural_gross = float(
        designated_daily["mean_deployed_gross_fraction_nav"].mean()
    )
    neural_graduated = (
        neural_excess > 0.0 and neural_gross >= GRADUATION_GROSS_FRACTION_NAV
    )
    stage0 = _read_json(stage0_root / "result.json")
    teacher_excess = float(stage0["pooled_mean_daily_net_excess_all_cash_bps"])
    teacher_graduated = bool(stage0["teacher_graduated"])
    if teacher_graduated and (
        not neural_graduated or neural_excess <= teacher_excess
    ):
        standing = "teacher_rule"
    elif neural_graduated:
        standing = "neural_policy"
    else:
        standing = "none"
    decision = {
        "teacher_graduated": teacher_graduated,
        "teacher_pooled_mean_daily_net_excess_all_cash_bps": teacher_excess,
        "neural_graduated": neural_graduated,
        "neural_pooled_mean_daily_net_excess_all_cash_bps": neural_excess,
        "neural_pooled_mean_deployed_gross_fraction_nav": neural_gross,
        "standing_execution_candidate": standing,
        "deployed_recipe_changed": False,
    }
    return ladder_path, readouts_path, {"comparisons": comparison_path, **decision}


def run_stage2(root: Path, device_name: str = "cuda") -> Path:
    root = root.resolve()
    final = root / "result.json"
    if final.exists():
        _verified_result(root, "result.json")
        return final
    design = _read_json(root / "frozen_design.json")
    _assert_frozen(design, STAGE2_SCHEMA)
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Experiment 57 fine-tuning requires the claimed CUDA instance")
    stage0_root = Path(str(design["inputs"]["stage0_root"]))
    stage1_root = Path(str(design["inputs"]["stage1_root"]))
    _verified_result(stage0_root, "result.json")
    _verified_result(stage1_root, "result.json")
    stage0_design = _read_json(stage0_root / "frozen_design.json")
    archive = _load_archive(stage0_root, stage0_design)
    dates = tuple(
        pl.read_parquet(stage0_root / "market_inputs" / "dates.parquet")
        .sort("date_idx")["trade_date"]
    )
    splits = policy_splits(dates)
    manifests = []
    for split in splits:
        target_record = _read_json(stage0_root / "result.json")["artifacts"][
            "teacher_evaluation_targets"
        ][split.window]
        target_path = Path(str(target_record["path"]))
        if _sha256(target_path) != target_record["sha256"]:
            raise ValueError("Stage-0 teacher evaluation target hash differs")
        teacher_path = torch.as_tensor(
            np.load(target_path, mmap_mode="r"), dtype=torch.float32, device=device
        )
        for seed in POLICY_SEEDS:
            manifests.append(
                _run_fine_tune(
                    root=root,
                    stage0_root=stage0_root,
                    stage1_root=stage1_root,
                    stage0_design=stage0_design,
                    archive=archive,
                    dates=dates,
                    split=split,
                    seed=seed,
                    teacher_path=teacher_path,
                    device=device,
                )
            )
            torch.cuda.empty_cache()
    if len(manifests) != 9:
        raise ValueError("Experiment 57 Stage 2 did not complete exactly nine runs")
    ladder_path, readouts_path, decision = _stage2_readouts(
        root=root, stage0_root=stage0_root, stage0_design=stage0_design
    )
    comparisons = Path(str(decision.pop("comparisons")))
    inventory_rows = []
    for manifest_path in manifests:
        manifest = _read_json(manifest_path)
        checkpoint = manifest_path.parent / "designated.pt"
        inventory_rows.append(
            {
                "window": manifest["window"],
                "seed": manifest["seed"],
                "designated_kind": manifest["designated_kind"],
                "path": str(checkpoint.resolve()),
                "bytes": checkpoint.stat().st_size,
                "sha256": _sha256(checkpoint),
                "retained": True,
            }
        )
        fine = manifest_path.parent / "fine_tuned.pt"
        inventory_rows.append(
            {
                "window": manifest["window"],
                "seed": manifest["seed"],
                "designated_kind": "fine_tuned_source",
                "path": str(fine.resolve()),
                "bytes": fine.stat().st_size,
                "sha256": _sha256(fine),
                "retained": True,
            }
        )
    inventory_path = root / "checkpoint_inventory.parquet"
    pl.DataFrame(inventory_rows).write_parquet(inventory_path)
    result = {
        "schema": STAGE2_SCHEMA,
        "status": "completed",
        "repository_commit": repository_commit(),
        **decision,
        "run_count": 9,
        "retained_designated_checkpoint_count": 9,
        "artifacts": {
            "ladder_daily": _artifact(ladder_path),
            "readouts": _artifact(readouts_path),
            "designated_comparisons": _artifact(comparisons),
            "checkpoint_inventory": _artifact(inventory_path),
            "run_manifests": [_artifact(path) for path in manifests],
        },
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(final, result)
    _atomic_json(
        root / "final_audit.json",
        {
            "schema": "EXPERIMENT57_STAGE2_AUDIT_V1",
            "status": "passed",
            "result_sha256": _sha256(final),
            "all_nine_fine_tunes_completed": True,
            "clone_fallback_checked_per_seed": True,
            "all_nine_designated_checkpoints_retained": True,
            "graduation_requires_actual_deployment": True,
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    return final


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 57 frontier imitation")
    commands = parser.add_subparsers(dest="command", required=True)
    freeze0 = commands.add_parser("freeze-stage0")
    freeze0.add_argument("--section-a-root", type=Path, required=True)
    freeze0.add_argument("--section-b-root", type=Path, required=True)
    freeze0.add_argument("--experiment55-root", type=Path, required=True)
    freeze0.add_argument("--experiment54-root", type=Path, required=True)
    freeze0.add_argument("--experiment52-root", type=Path, required=True)
    freeze0.add_argument("--experiment53-root", type=Path, required=True)
    freeze0.add_argument("--preregistration", type=Path, required=True)
    freeze0.add_argument("--output-dir", type=Path, required=True)
    run0 = commands.add_parser("run-stage0")
    run0.add_argument("--root", type=Path, required=True)
    freeze1 = commands.add_parser("freeze-stage1")
    freeze1.add_argument("--stage0-root", type=Path, required=True)
    freeze1.add_argument("--preregistration", type=Path, required=True)
    freeze1.add_argument("--output-dir", type=Path, required=True)
    run1 = commands.add_parser("run-stage1")
    run1.add_argument("--root", type=Path, required=True)
    run1.add_argument("--device", default="cuda")
    freeze2 = commands.add_parser("freeze-stage2")
    freeze2.add_argument("--stage0-root", type=Path, required=True)
    freeze2.add_argument("--stage1-root", type=Path, required=True)
    freeze2.add_argument("--preregistration", type=Path, required=True)
    freeze2.add_argument("--output-dir", type=Path, required=True)
    run2 = commands.add_parser("run-stage2")
    run2.add_argument("--root", type=Path, required=True)
    run2.add_argument("--device", default="cuda")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    if args.command == "freeze-stage0":
        path = freeze_stage0(
            section_a_root=args.section_a_root,
            section_b_root=args.section_b_root,
            experiment55_root=args.experiment55_root,
            experiment54_root=args.experiment54_root,
            experiment52_root=args.experiment52_root,
            experiment53_root=args.experiment53_root,
            preregistration=args.preregistration,
            output_dir=args.output_dir,
        )
    elif args.command == "run-stage0":
        path = run_stage0(args.root)
    elif args.command == "freeze-stage1":
        path = freeze_stage1(
            stage0_root=args.stage0_root,
            preregistration=args.preregistration,
            output_dir=args.output_dir,
        )
    elif args.command == "run-stage1":
        path = run_stage1(args.root, args.device)
    elif args.command == "freeze-stage2":
        path = freeze_stage2(
            stage0_root=args.stage0_root,
            stage1_root=args.stage1_root,
            preregistration=args.preregistration,
            output_dir=args.output_dir,
        )
    else:
        path = run_stage2(args.root, args.device)
    print(path)


if __name__ == "__main__":
    main()
