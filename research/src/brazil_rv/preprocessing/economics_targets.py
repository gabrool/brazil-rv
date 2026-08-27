from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import polars as pl
from numpy.lib.format import open_memmap
from numpy.typing import NDArray

from ..modeling.contract import HORIZONS, VALIDATION_END, workspace_path
from ..modeling.data import feature_store_identity
from .contract import DECISION_EQUITY_INDICES, EQUITY_SESSION_MINUTES
from .io import (
    cotahist_files,
    dense_grid,
    load_assignments,
    load_market_dates_and_security_dates,
    load_source_file,
    prepare_session_bars,
    read_research_interval,
    validate_physical_source_identity,
    validate_source_date_isolation,
)

ECONOMICS_SCHEMA = "EXPERIMENT49_ECONOMICS_INPUTS_V1"
ECONOMICS_HORIZONS = (15, *HORIZONS)
ECONOMICS_ARRAYS = (
    "open_to_open_returns.npy",
    "open_to_open_mask.npy",
    "mid_proxy_returns.npy",
    "mid_proxy_mask.npy",
    "daily_dollar_volume.npy",
    "trailing_adv.npy",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _recorded_input(store_manifest: Mapping[str, object], name: str) -> Path:
    inputs = store_manifest.get("canonical_inputs")
    if not isinstance(inputs, dict) or not isinstance(inputs.get(name), dict):
        raise ValueError(f"Feature store does not record canonical input {name}")
    value = inputs[name].get("resolved_path")
    if not isinstance(value, str):
        raise ValueError(f"Feature store canonical input {name} has no resolved path")
    return workspace_path(value)


def exact_alternative_returns(
    raw_grid: NDArray[np.float64],
    observed: NDArray[np.bool_],
    horizons: Sequence[int] = ECONOMICS_HORIZONS,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.bool_],
    NDArray[np.float32],
    NDArray[np.bool_],
]:
    """Return exact open/open and adjacent-close-midpoint horizon returns."""
    if raw_grid.ndim != 3 or raw_grid.shape[2] != 5:
        raise ValueError("Raw equity grid must be [date, minute, OHLCV]")
    if observed.shape != raw_grid.shape[:2]:
        raise ValueError("Observation mask does not align to the raw grid")
    shape = (raw_grid.shape[0], len(DECISION_EQUITY_INDICES), len(horizons))
    open_returns = np.zeros(shape, dtype=np.float32)
    open_mask = np.zeros(shape, dtype=bool)
    mid_returns = np.zeros(shape, dtype=np.float32)
    mid_mask = np.zeros(shape, dtype=bool)
    for decision, entry in enumerate(DECISION_EQUITY_INDICES):
        for horizon_index, horizon in enumerate(horizons):
            exit_index = entry + int(horizon)
            if exit_index >= raw_grid.shape[1]:
                continue
            valid_open = observed[:, entry] & observed[:, exit_index]
            open_mask[:, decision, horizon_index] = valid_open
            open_returns[valid_open, decision, horizon_index] = np.log(
                raw_grid[valid_open, exit_index, 0] / raw_grid[valid_open, entry, 0]
            ).astype(np.float32)

            valid_mid = (
                observed[:, entry - 1]
                & observed[:, entry]
                & observed[:, exit_index - 1]
                & observed[:, exit_index]
            )
            mid_mask[:, decision, horizon_index] = valid_mid
            entry_mid = 0.5 * (
                raw_grid[valid_mid, entry - 1, 3] + raw_grid[valid_mid, entry, 3]
            )
            exit_mid = 0.5 * (
                raw_grid[valid_mid, exit_index - 1, 3]
                + raw_grid[valid_mid, exit_index, 3]
            )
            mid_returns[valid_mid, decision, horizon_index] = np.log(
                exit_mid / entry_mid
            ).astype(np.float32)
    return open_returns, open_mask, mid_returns, mid_mask


def roll_covariance_inputs(
    raw_grid: NDArray[np.float64], observed: NDArray[np.bool_]
) -> tuple[int, float, float, float]:
    """Sufficient statistics for sample cov(r_t, r_t-1), within sessions."""
    if raw_grid.ndim != 3 or observed.shape != raw_grid.shape[:2]:
        raise ValueError("Roll inputs do not align")
    triple = observed[:, 2:] & observed[:, 1:-1] & observed[:, :-2]
    current = np.log(
        raw_grid[:, 2:, 3], where=observed[:, 2:], out=np.zeros_like(raw_grid[:, 2:, 3])
    ) - np.log(
        raw_grid[:, 1:-1, 3],
        where=observed[:, 1:-1],
        out=np.zeros_like(raw_grid[:, 1:-1, 3]),
    )
    lagged = np.log(
        raw_grid[:, 1:-1, 3],
        where=observed[:, 1:-1],
        out=np.zeros_like(raw_grid[:, 1:-1, 3]),
    ) - np.log(
        raw_grid[:, :-2, 3],
        where=observed[:, :-2],
        out=np.zeros_like(raw_grid[:, :-2, 3]),
    )
    x = current[triple]
    y = lagged[triple]
    return int(x.size), float(x.sum()), float(y.sum()), float(np.dot(x, y))


def roll_effective_spread(
    count: int, sum_current: float, sum_lagged: float, sum_product: float
) -> tuple[float, float]:
    if count < 2:
        return float("nan"), float("nan")
    covariance = (sum_product - sum_current * sum_lagged / count) / (count - 1)
    spread = 2.0 * np.sqrt(-covariance) if covariance < 0.0 else float("nan")
    return float(covariance), float(spread)


def causal_trailing_adv(
    daily_dollar_volume: NDArray[np.float64], window: int = 20
) -> NDArray[np.float64]:
    if daily_dollar_volume.ndim != 2 or window <= 0:
        raise ValueError("Daily dollar volume must be [date, equity]")
    output = np.full(daily_dollar_volume.shape, np.nan, dtype=np.float64)
    for equity in range(daily_dollar_volume.shape[1]):
        history: deque[float] = deque(maxlen=window)
        for date_idx in range(daily_dollar_volume.shape[0]):
            if history:
                output[date_idx, equity] = float(np.mean(history))
            value = float(daily_dollar_volume[date_idx, equity])
            if np.isfinite(value) and value > 0.0:
                history.append(value)
    return output


def _quarter(value: date) -> str:
    return f"{value.year}Q{(value.month - 1) // 3 + 1}"


def _liquidity_group(rank: int) -> str:
    if rank <= 40:
        return "top_40"
    if rank <= 100:
        return "rank_41_100"
    return "rank_101_158"


def _roll_schedule(
    *,
    dates: Sequence[date],
    security_ids: Sequence[str],
    membership: NDArray[np.bool_],
    trailing_adv: NDArray[np.float64],
    sufficient: Mapping[tuple[int, str], tuple[int, float, float, float]],
) -> pl.DataFrame:
    quarters = tuple(dict.fromkeys(_quarter(value) for value in dates))
    date_quarters = np.asarray([_quarter(value) for value in dates])
    records: list[dict[str, object]] = []
    prior: dict[int, float] = {}
    for quarter in quarters:
        on_quarter = date_quarters == quarter
        median_adv = np.nanmedian(
            np.where(membership[on_quarter], trailing_adv[on_quarter], np.nan), axis=0
        )
        order = np.argsort(-np.nan_to_num(median_adv, nan=-np.inf), kind="stable")
        ranks = np.empty(len(security_ids), dtype=np.int16)
        ranks[order] = np.arange(1, len(security_ids) + 1, dtype=np.int16)
        exact: dict[int, float] = {}
        covariance: dict[int, float] = {}
        counts: dict[int, int] = {}
        for slot in range(len(security_ids)):
            values = sufficient.get((slot, quarter), (0, 0.0, 0.0, 0.0))
            cov, spread = roll_effective_spread(*values)
            covariance[slot] = cov
            counts[slot] = values[0]
            if np.isfinite(spread):
                exact[slot] = spread
        group_medians = {
            group: float(np.median(values))
            for group in ("top_40", "rank_41_100", "rank_101_158")
            if (
                values := [
                    spread
                    for slot, spread in exact.items()
                    if _liquidity_group(int(ranks[slot])) == group
                ]
            )
        }
        market_median = (
            float(np.median(list(exact.values()))) if exact else float("nan")
        )
        if not np.isfinite(market_median) and not prior:
            raise ValueError(f"No valid Roll estimate or prior in {quarter}")
        for slot, security_id in enumerate(security_ids):
            group = _liquidity_group(int(ranks[slot]))
            if slot in exact:
                spread = exact[slot]
                method = "exact_security_quarter"
                prior[slot] = spread
            elif slot in prior:
                spread = prior[slot]
                method = "prior_security_quarter"
            elif group in group_medians:
                spread = group_medians[group]
                method = "quarter_liquidity_group_median"
            else:
                spread = market_median
                method = "quarter_market_median"
            records.append(
                {
                    "security_id": security_id,
                    "quarter": quarter,
                    "liquidity_rank": int(ranks[slot]),
                    "liquidity_group": group,
                    "median_trailing_adv": (
                        float(median_adv[slot])
                        if np.isfinite(median_adv[slot])
                        else None
                    ),
                    "lag_pair_count": counts[slot],
                    "serial_covariance": (
                        covariance[slot] if np.isfinite(covariance[slot]) else None
                    ),
                    "exact_full_spread_fraction": exact.get(slot),
                    "schedule_full_spread_fraction": spread,
                    "schedule_half_spread_fraction": spread / 2.0,
                    "schedule_full_spread_bps": spread * 10_000.0,
                    "schedule_half_spread_bps": spread * 5_000.0,
                    "schedule_method": method,
                }
            )
    return pl.DataFrame(records)


def economics_input_identity(
    directory: Path, store_identity: Mapping[str, object]
) -> dict[str, object]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = {
        name: _sha256(directory / name)
        for name in (*ECONOMICS_ARRAYS, "roll_schedule.parquet")
    }
    if (
        manifest.get("schema") != ECONOMICS_SCHEMA
        or manifest.get("source_feature_store") != store_identity
        or manifest.get("through") != VALIDATION_END.isoformat()
        or manifest.get("artifact_sha256") != hashes
        or manifest.get("official_validation_accessed") is not False
        or manifest.get("test_accessed") is not False
    ):
        raise ValueError("Experiment-49 economics input identity differs")
    return {
        "path": str(directory.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "artifact_sha256": hashes,
    }


def _close_memmaps(arrays: Mapping[str, np.memmap]) -> None:
    for array in arrays.values():
        array.flush()
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            mapping.close()


def build_economics_inputs(store: Path, output_dir: Path) -> Path:
    store = store.resolve()
    store_identity = feature_store_identity(store)
    if output_dir.exists():
        economics_input_identity(output_dir, store_identity)
        return output_dir
    store_manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    assignments_dir = _recorded_input(store_manifest, "accepted_xp_assignments")
    cotahist_dir = _recorded_input(store_manifest, "parsed_cotahist")
    universe_dir = _recorded_input(store_manifest, "point_in_time_universe")
    research_start, research_end = read_research_interval(universe_dir)
    through = min(research_end, VALIDATION_END)
    assignments = load_assignments(assignments_dir)
    security_ids = tuple(assignments.get_column("security_id").to_list())
    market_dates, assignment_dates = load_market_dates_and_security_dates(
        cotahist_files(cotahist_dir),
        security_ids,
        research_start,
        through,
        allow_empty_security_dates=True,
    )
    validate_source_date_isolation(assignments, assignment_dates)
    date_index = (
        pl.read_parquet(store / "date_index.parquet")
        .filter(pl.col("trade_date") <= through)
        .sort("date_idx")
    )
    equity_index = pl.read_parquet(store / "equity_index.parquet").sort("equity_slot")
    if (
        tuple(date_index["trade_date"]) != market_dates
        or tuple(equity_index["security_id"]) != security_ids
    ):
        raise ValueError("Experiment-49 axes differ from the canonical store")
    date_count = len(market_dates)
    equity_count = len(security_ids)
    shape = (
        date_count,
        equity_count,
        len(DECISION_EQUITY_INDICES),
        len(ECONOMICS_HORIZONS),
    )
    partial = output_dir.with_name(f".{output_dir.name}.tmp-{uuid4().hex}")
    partial.mkdir(parents=True)
    specs = {
        "open_to_open_returns.npy": (np.float32, shape),
        "open_to_open_mask.npy": (bool, shape),
        "mid_proxy_returns.npy": (np.float32, shape),
        "mid_proxy_mask.npy": (bool, shape),
        "daily_dollar_volume.npy": (np.float64, (date_count, equity_count)),
        "trailing_adv.npy": (np.float64, (date_count, equity_count)),
    }
    arrays = {
        name: open_memmap(partial / name, mode="w+", dtype=dtype, shape=array_shape)
        for name, (dtype, array_shape) in specs.items()
    }
    for array in arrays.values():
        array[...] = 0
    roll_stats: defaultdict[tuple[int, str], list[float]] = defaultdict(
        lambda: [0, 0.0, 0.0, 0.0]
    )
    slot_by_security = {
        security_id: slot for slot, security_id in enumerate(security_ids)
    }
    date_quarters = np.asarray([_quarter(value) for value in market_dates])
    try:
        groups = assignments.partition_by("source_file", maintain_order=True)
        for source_number, group in enumerate(groups, start=1):
            source_path = Path(group.item(0, "source_file"))
            source = load_source_file(source_path)
            validate_physical_source_identity(group, source, source_path)
            allowed_dates = frozenset().union(
                *(assignment_dates[value] for value in group["security_id"])
            )
            session_bars = prepare_session_bars(
                source,
                source_path,
                allowed_dates,
                market_dates,
                10 * 60,
                EQUITY_SESSION_MINUTES,
            )
            for assignment in group.iter_rows(named=True):
                security_id = assignment["security_id"]
                bars = session_bars.filter(
                    pl.col("trade_date").is_in(tuple(assignment_dates[security_id]))
                )
                raw_grid, observed = dense_grid(
                    bars, date_count, EQUITY_SESSION_MINUTES
                )
                open_return, open_mask, mid_return, mid_mask = (
                    exact_alternative_returns(raw_grid, observed)
                )
                slot = slot_by_security[security_id]
                arrays["open_to_open_returns.npy"][:, slot] = open_return
                arrays["open_to_open_mask.npy"][:, slot] = open_mask
                arrays["mid_proxy_returns.npy"][:, slot] = mid_return
                arrays["mid_proxy_mask.npy"][:, slot] = mid_mask
                arrays["daily_dollar_volume.npy"][:, slot] = np.sum(
                    raw_grid[:, :, 3] * raw_grid[:, :, 4] * observed, axis=1
                )
                for quarter in np.unique(date_quarters):
                    on_quarter = date_quarters == quarter
                    values = roll_covariance_inputs(
                        raw_grid[on_quarter], observed[on_quarter]
                    )
                    aggregate = roll_stats[(slot, str(quarter))]
                    for index, value in enumerate(values):
                        aggregate[index] += value
            if source_number % 20 == 0 or source_number == len(groups):
                print(f"Built Experiment-49 raw inputs {source_number}/{len(groups)}")

        arrays["trailing_adv.npy"][:] = causal_trailing_adv(
            np.asarray(arrays["daily_dollar_volume.npy"])
        )
        membership = np.asarray(
            np.load(store / "equity_membership.npy", mmap_mode="r", allow_pickle=False)[
                :date_count
            ],
            dtype=bool,
        )
        schedule = _roll_schedule(
            dates=market_dates,
            security_ids=security_ids,
            membership=membership,
            trailing_adv=np.asarray(arrays["trailing_adv.npy"]),
            sufficient={key: tuple(value) for key, value in roll_stats.items()},
        )
        schedule.write_parquet(partial / "roll_schedule.parquet")
        _close_memmaps(arrays)
        arrays.clear()
        hashes = {
            name: _sha256(partial / name)
            for name in (*ECONOMICS_ARRAYS, "roll_schedule.parquet")
        }
        manifest = {
            "schema": ECONOMICS_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_feature_store": store_identity,
            "through": through.isoformat(),
            "date_count": date_count,
            "equity_count": equity_count,
            "decision_count": len(DECISION_EQUITY_INDICES),
            "horizons_minutes": list(ECONOMICS_HORIZONS),
            "alternative_labels": {
                "open_to_open": "log(open[T+h]/open[T])",
                "mid_proxy": (
                    "log(mean(close[T+h-1],close[T+h]) / mean(close[T-1],close[T]))"
                ),
            },
            "roll": (
                "2*sqrt(-sample_cov(r_t,r_t-1)) for negative covariance; "
                "three consecutive observed within-session closes required"
            ),
            "adv": (
                "mean daily close*real_volume over up to 20 prior observed "
                "sessions, current date excluded"
            ),
            "artifact_sha256": hashes,
            "official_validation_accessed": False,
            "test_accessed": False,
        }
        (partial / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, output_dir)
    except BaseException:
        if arrays:
            _close_memmaps(arrays)
        shutil.rmtree(partial, ignore_errors=True)
        raise
    economics_input_identity(output_dir, store_identity)
    return output_dir
