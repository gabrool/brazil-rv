from __future__ import annotations

import argparse
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from ..modeling.contract import MIN_IC_EQUITIES
from ..modeling.data import feature_store_identity
from ..modeling.metrics import average_ranks, moving_block_bootstrap
from ..modeling.provenance import repository_commit
from .experiment56 import (
    BOOTSTRAP_BLOCK,
    BOOTSTRAP_REPLICATIONS,
    HORIZON_NAMES,
    _artifact,
    _atomic_json,
    _read_json,
    _sha256,
    _verified_result,
)
from .inputs import iter_discovery_equity_grids, load_daily_cdi_rates
from .inputs import load_discovery_prediction_archive
from .splits import policy_evaluation_slices


SCHEMA = "EXPERIMENT58_SWING_SCREEN_V1"
TARGET_HORIZONS = (1, 2, 3, 5, 10)
PART_B_SIGNALS = ("head_to_close_eod", "four_head_mean_eod")
CONCENTRATIONS = (15, 30)
RANK_BANDS = (0.0, 0.3)
COSTS_BPS = (2.0, 4.0, 7.0)
BORROW_RATES = (0.02, 0.04)
LIMIT_LEVELS = ("close", "inside_half_half_spread")
WAIT_WINDOWS = ("morning_60m", "full_next_session")
MARGIN_FRACTION = 0.5
NAV_BRL = 10_000_000.0
BOOTSTRAP_SEED = 20260858


def _create_root(path: Path) -> Path:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(path)
    path.mkdir(parents=True)
    return path


def _assert_frozen(design: Mapping[str, object]) -> None:
    if (
        design.get("schema") != SCHEMA
        or design.get("repository_commit") != repository_commit()
    ):
        raise ValueError("Experiment 58 must run at its exact frozen commit")
    if (
        design.get("official_validation_accessed") is not False
        or design.get("test_accessed") is not False
    ):
        raise ValueError("Experiment 58 access flags differ")


def freeze(
    *,
    section_b_root: Path,
    experiment57_stage0_root: Path,
    preregistration: Path,
    output_dir: Path,
) -> Path:
    section_b_root = section_b_root.resolve()
    experiment57_stage0_root = experiment57_stage0_root.resolve()
    section_b, _ = _verified_result(section_b_root, "result.json")
    stage0, _ = _verified_result(experiment57_stage0_root, "result.json")
    if stage0.get("teacher_graduated") is not False:
        raise ValueError("Experiment 57 Stage 0 result differs from the retained loss")
    stage0_design = _read_json(experiment57_stage0_root / "frozen_design.json")
    store = Path(str(stage0_design["store"]["path"])).resolve()
    oof = section_b["archive"]
    root = _create_root(output_dir)
    try:
        design = {
            "schema": SCHEMA,
            "status": "frozen",
            "repository_commit": repository_commit(),
            "preregistration": _artifact(preregistration.resolve()),
            "store": {
                "path": str(store),
                "identity": feature_store_identity(store),
                "manifest": _artifact(store / "manifest.json"),
            },
            "inputs": {
                "section_b_root": str(section_b_root),
                "section_b_result": _artifact(section_b_root / "result.json"),
                "oof_archive": oof,
                "experiment57_stage0_root": str(experiment57_stage0_root),
                "experiment57_stage0_result": _artifact(
                    experiment57_stage0_root / "result.json"
                ),
                "experiment57_stage0_design": _artifact(
                    experiment57_stage0_root / "frozen_design.json"
                ),
                "experiment57_replay_daily": _artifact(
                    experiment57_stage0_root / "replay_daily.parquet"
                ),
                "cdi": stage0_design["inputs"]["cdi"],
                "market_inputs": {
                    name: _artifact(experiment57_stage0_root / "market_inputs" / name)
                    for name in (
                        "manifest.json",
                        "dates.parquet",
                        "date_idx.npy",
                        "active.npy",
                        "adv20_brl.npy",
                        "full_spread.npy",
                        "sigma_daily.npy",
                    )
                },
            },
            "contract": {
                "cpu_only": True,
                "head_names": list(HORIZON_NAMES),
                "target_horizons_sessions": list(TARGET_HORIZONS),
                "part_b_signals": list(PART_B_SIGNALS),
                "concentrations_per_side": list(CONCENTRATIONS),
                "rank_movement_bands": list(RANK_BANDS),
                "per_side_costs_bps": list(COSTS_BPS),
                "annual_short_borrow_rates": list(BORROW_RATES),
                "margin_fraction_of_gross": MARGIN_FRACTION,
                "gross_target": 2.0,
                "bootstrap_block": BOOTSTRAP_BLOCK,
                "bootstrap_replications": BOOTSTRAP_REPLICATIONS,
                "limit_levels": list(LIMIT_LEVELS),
                "wait_windows": list(WAIT_WINDOWS),
                "terminal_liquidation": True,
            },
            "official_validation_accessed": False,
            "test_accessed": False,
        }
        _atomic_json(root / "frozen_design.json", design)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return root / "frozen_design.json"


def _load_archive(design: Mapping[str, object]):
    record = design["inputs"]["oof_archive"]
    return load_discovery_prediction_archive(
        Path(record["prediction"]),
        Path(record["reference"]),
        Path(record["execution_manifest"]),
        Path(str(design["store"]["path"])),
    )


def _save_array(directory: Path, name: str, values: np.ndarray) -> dict[str, object]:
    path = directory / name
    with path.open("wb") as output:
        np.save(output, values, allow_pickle=False)
    return _artifact(path)


def _observed_extreme(
    values: np.ndarray, observed: np.ndarray, *, minimum: bool
) -> np.ndarray:
    fill = np.inf if minimum else -np.inf
    result = (np.min if minimum else np.max)(np.where(observed, values, fill), axis=1)
    result[~np.isfinite(result)] = np.nan
    return result


def _build_daily_market(
    root: Path, design: Mapping[str, object]
) -> Mapping[str, object]:
    final = root / "daily_market"
    if final.exists():
        manifest = _read_json(final / "manifest.json")
        for name, record in manifest["artifacts"].items():
            if _sha256(final / name) != record["sha256"]:
                raise ValueError(f"Daily market hash mismatch: {name}")
        return manifest
    temporary = root / ".daily_market.tmp"
    temporary.mkdir()
    source_root = Path(str(design["inputs"]["experiment57_stage0_root"]))
    dates_table = pl.read_parquet(source_root / "market_inputs" / "dates.parquet").sort(
        "date_idx"
    )
    requested_idx = np.load(
        source_root / "market_inputs" / "date_idx.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    trade_dates = tuple(dates_table["trade_date"])
    if dates_table.height != 716 or requested_idx.size != 716:
        raise ValueError("Experiment 58 requires exactly 716 TRAIN sessions")
    store = Path(str(design["store"]["path"]))
    names = pl.read_parquet(store / "equity_index.parquet").sort("equity_slot")
    shape = (len(trade_dates), names.height)
    arrays = {
        name: np.full(shape, np.nan, dtype=np.float64)
        for name in (
            "daily_close",
            "morning_low",
            "morning_high",
            "session_low",
            "session_high",
            "morning_60m_close",
        )
    }
    exact_close_observed = np.zeros(shape, dtype=bool)
    seen = np.zeros(names.height, dtype=bool)
    for grid in iter_discovery_equity_grids(store):
        source_by_date = {value: index for index, value in enumerate(grid.trade_dates)}
        positions = np.asarray([source_by_date[value] for value in trade_dates])
        observed = grid.observed[positions]
        slot = grid.equity_slot
        arrays["daily_close"][:, slot] = np.where(
            observed[:, -1], grid.close[positions, -1], np.nan
        )
        exact_close_observed[:, slot] = observed[:, -1]
        arrays["morning_low"][:, slot] = _observed_extreme(
            grid.low[positions, :60], observed[:, :60], minimum=True
        )
        arrays["morning_high"][:, slot] = _observed_extreme(
            grid.high[positions, :60], observed[:, :60], minimum=False
        )
        arrays["session_low"][:, slot] = _observed_extreme(
            grid.low[positions], observed, minimum=True
        )
        arrays["session_high"][:, slot] = _observed_extreme(
            grid.high[positions], observed, minimum=False
        )
        arrays["morning_60m_close"][:, slot] = np.where(
            observed[:, 59], grid.close[positions, 59], np.nan
        )
        seen[slot] = True
    if not seen.all():
        raise ValueError("Daily market bridge omitted a permanent security")
    artifacts = {
        name + ".npy": _save_array(temporary, name + ".npy", values)
        for name, values in arrays.items()
    }
    artifacts["exact_close_observed.npy"] = _save_array(
        temporary, "exact_close_observed.npy", exact_close_observed
    )
    dates_table.write_parquet(temporary / "dates.parquet")
    names.write_parquet(temporary / "equities.parquet")
    artifacts["dates.parquet"] = _artifact(temporary / "dates.parquet")
    artifacts["equities.parquet"] = _artifact(temporary / "equities.parquet")
    manifest = {
        "schema": "EXPERIMENT58_DAILY_MARKET_V1",
        "source_store_identity": design["store"]["identity"],
        "date_count": len(trade_dates),
        "equity_count": names.height,
        "daily_close": "exact session minute 404 only; no stale substitution",
        "morning_window": "observed raw high/low minutes 0 through 59",
        "full_window": "observed raw high/low minutes 0 through 404",
        "artifacts": artifacts,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(temporary / "manifest.json", manifest)
    os.replace(temporary, final)
    return _read_json(final / "manifest.json")


def _market_array(root: Path, name: str) -> np.ndarray:
    return np.load(root / "daily_market" / name, mmap_mode="r", allow_pickle=False)


def _midrank_row(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=np.float64)
    take = np.asarray(valid, dtype=bool) & np.isfinite(values)
    count = int(take.sum())
    if count:
        ranks = average_ranks(np.asarray(values[take], dtype=np.float64))
        result[take] = 2.0 * ((ranks + 0.5) / count) - 1.0
    return result


def _midrank_matrix(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if values.shape != valid.shape:
        raise ValueError("Midrank values and masks differ")
    return np.stack(
        [_midrank_row(row, mask) for row, mask in zip(values, valid, strict=True)]
    )


def _build_signals(
    root: Path, design: Mapping[str, object], archive: object
) -> tuple[tuple[str, ...], np.ndarray]:
    path = root / "signals.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as values:
            names = tuple(str(value) for value in values["names"].tolist())
            return names, values["signals"].copy()
    ranks = np.asarray(archive.ranks, dtype=np.float64)
    valid = np.asarray(archive.valid, dtype=bool)
    if ranks.shape[0] != 716 or ranks.shape[-1] != 4:
        raise ValueError("Experiment 58 requires a 716-date four-head OOF archive")
    final = ranks[:, -1]
    final_valid = valid[:, -1]
    last_minute = int(archive.refresh_minutes[-1])
    last_hour = np.flatnonzero(archive.refresh_minutes >= last_minute - 60)
    if not last_hour.size:
        raise ValueError("OOF archive has no final-hour refreshes")
    hour_values = np.where(valid[:, last_hour], ranks[:, last_hour], np.nan)
    hour_mean = np.nanmean(hour_values, axis=1)
    hour_valid = valid[:, last_hour].all(axis=1)
    active = np.load(
        Path(str(design["inputs"]["experiment57_stage0_root"]))
        / "market_inputs"
        / "active.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    close = np.asarray(_market_array(root, "daily_close.npy"))
    signals: list[np.ndarray] = []
    names: list[str] = []
    for head, head_name in enumerate(HORIZON_NAMES):
        names.append(f"head_{head_name}_eod")
        signals.append(_midrank_matrix(final[..., head], final_valid[..., head]))
    for head, head_name in enumerate(HORIZON_NAMES):
        names.append(f"head_{head_name}_last_hour")
        signals.append(_midrank_matrix(hour_mean[..., head], hour_valid[..., head]))
    names.extend(("four_head_mean_eod", "four_head_mean_last_hour"))
    signals.append(
        _midrank_matrix(
            np.nanmean(np.where(final_valid, final, np.nan), axis=-1),
            final_valid.all(axis=-1),
        )
    )
    signals.append(
        _midrank_matrix(
            np.nanmean(np.where(hour_valid, hour_mean, np.nan), axis=-1),
            hour_valid.all(axis=-1),
        )
    )
    reversal = np.full(close.shape, np.nan)
    momentum = np.full(close.shape, np.nan)
    reversal[5:] = -(close[5:] / close[:-5] - 1.0)
    momentum[20:] = close[20:] / close[:-20] - 1.0
    reversal_valid = np.isfinite(reversal) & np.asarray(active, dtype=bool)
    momentum_valid = np.isfinite(momentum) & np.asarray(active, dtype=bool)
    names.extend(("reversal_5d", "momentum_20d"))
    signals.extend(
        (
            _midrank_matrix(reversal, reversal_valid),
            _midrank_matrix(momentum, momentum_valid),
        )
    )
    stacked = np.stack(signals, axis=1).astype(np.float32)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as output:
        np.savez(
            output,
            names=np.asarray(names),
            signals=stacked,
            last_hour_refresh_minutes=np.asarray(archive.refresh_minutes[last_hour]),
        )
    os.replace(temporary, path)
    return tuple(names), stacked


def _build_targets(
    root: Path, design: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    path = root / "targets.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as values:
            return values["target_ranks"].copy(), values["raw_returns"].copy()
    close = np.asarray(_market_array(root, "daily_close.npy"), dtype=np.float64)
    source_root = Path(str(design["inputs"]["experiment57_stage0_root"]))
    sigma = np.asarray(
        np.load(
            source_root / "market_inputs" / "sigma_daily.npy",
            mmap_mode="r",
            allow_pickle=False,
        ),
        dtype=np.float64,
    )
    active = np.asarray(
        np.load(
            source_root / "market_inputs" / "active.npy",
            mmap_mode="r",
            allow_pickle=False,
        ),
        dtype=bool,
    )
    targets, raw_returns = _daily_targets(close, sigma, active)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as output:
        np.savez(
            output,
            horizons=np.asarray(TARGET_HORIZONS),
            target_ranks=targets,
            raw_returns=raw_returns,
        )
    os.replace(temporary, path)
    return targets, raw_returns


def _daily_targets(
    close: np.ndarray, sigma: np.ndarray, active: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    close = np.asarray(close, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    if close.shape != sigma.shape or close.shape != active.shape:
        raise ValueError("Daily target inputs differ")
    targets = np.full((len(TARGET_HORIZONS), *close.shape), np.nan, dtype=np.float32)
    raw_returns = np.full_like(targets, np.nan)
    for horizon_index, horizon in enumerate(TARGET_HORIZONS):
        raw = close[horizon:] / close[:-horizon] - 1.0
        normalized = raw / (sigma[:-horizon] * math.sqrt(horizon))
        valid = (
            active[:-horizon]
            & np.isfinite(raw)
            & np.isfinite(normalized)
            & (sigma[:-horizon] > 0.0)
        )
        centered = normalized.copy()
        for day in range(centered.shape[0]):
            take = valid[day]
            if take.any():
                centered[day, take] -= np.median(centered[day, take])
        targets[horizon_index, :-horizon] = _midrank_matrix(centered, valid).astype(
            np.float32
        )
        raw_returns[horizon_index, :-horizon] = np.where(valid, raw, np.nan).astype(
            np.float32
        )
    return targets, raw_returns


def _cross_sectional_correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < MIN_IC_EQUITIES:
        return math.nan
    x = left[valid] - np.mean(left[valid])
    y = right[valid] - np.mean(right[valid])
    denominator = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    return float(np.sum(x * y) / denominator) if denominator > 0.0 else math.nan


def _decile_spread(signal: np.ndarray, returns: np.ndarray, horizon: int) -> float:
    valid = np.isfinite(signal) & np.isfinite(returns)
    count = int(valid.sum())
    if count < 20:
        return math.nan
    names = np.flatnonzero(valid)
    order = names[np.argsort(signal[names], kind="stable")]
    tail = max(1, count // 10)
    return float(
        (np.mean(returns[order[-tail:]]) - np.mean(returns[order[:tail]]))
        * 10_000.0
        / horizon
    )


def _interval(values: np.ndarray, seed_offset: int) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < BOOTSTRAP_BLOCK:
        return {"estimate": math.nan, "lower_95": math.nan, "upper_95": math.nan}
    result = moving_block_bootstrap(
        finite,
        replications=BOOTSTRAP_REPLICATIONS,
        block_length=BOOTSTRAP_BLOCK,
        seed=BOOTSTRAP_SEED + seed_offset,
    )
    return {
        name: float(np.asarray(value).reshape(-1)[0]) for name, value in result.items()
    }


def _sharpe(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2 or np.std(finite, ddof=1) == 0.0:
        return math.nan
    return float(np.sqrt(252.0) * np.mean(finite) / np.std(finite, ddof=1))


def _windows(dates: Sequence[date]) -> dict[str, np.ndarray]:
    by_date = {value: index for index, value in enumerate(dates)}
    result = {"all_train": np.arange(len(dates), dtype=np.int64)}
    for item in policy_evaluation_slices(dates, dates):
        result[item.name] = np.asarray([by_date[value] for value in item.dates])
    return result


def _part0(root: Path, design: Mapping[str, object]) -> Path:
    source = Path(str(design["inputs"]["experiment57_stage0_root"]))
    daily = pl.read_parquet(source / "replay_daily.parquet").filter(
        (pl.col("book") == "dollar_neutral") & pl.col("designated")
    )
    required = {
        "window",
        "gross_pnl_brl",
        "spread_cost_brl",
        "fees_brl",
        "cdi_earned_brl",
        "net_pnl_brl",
        "all_cash_cdi_pnl_brl",
        "turnover_brl",
        "excess_pnl_brl",
    }
    if not required.issubset(daily.columns):
        raise ValueError(
            "Experiment 57 retained daily report lacks attribution columns"
        )
    rows = []
    for window in ("fold_c", "fold_a", "fold_b", "pooled"):
        values = (
            daily if window == "pooled" else daily.filter(pl.col("window") == window)
        )
        scale = 10_000.0 / NAV_BRL
        rows.append(
            {
                "window": window,
                "date_count": values.height,
                "gross_pnl_bps_per_day": float(values["gross_pnl_brl"].mean() * scale),
                "spread_cost_bps_per_day": float(
                    values["spread_cost_brl"].mean() * scale
                ),
                "fee_cost_bps_per_day": float(values["fees_brl"].mean() * scale),
                "cdi_earned_bps_per_day": float(
                    values["cdi_earned_brl"].mean() * scale
                ),
                "net_pnl_bps_per_day": float(values["net_pnl_brl"].mean() * scale),
                "all_cash_cdi_bps_per_day": float(
                    values["all_cash_cdi_pnl_brl"].mean() * scale
                ),
                "net_excess_bps_per_day": float(
                    values["excess_pnl_brl"].mean() * scale
                ),
                "turnover_bps_nav_per_day": float(
                    values["turnover_brl"].mean() * scale
                ),
            }
        )
    path = root / "part0_attribution.parquet"
    table = pl.DataFrame(rows)
    expected = float(
        _read_json(source / "result.json")["pooled_mean_daily_net_excess_all_cash_bps"]
    )
    actual = float(
        table.filter(pl.col("window") == "pooled").item(0, "net_excess_bps_per_day")
    )
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("Part 0 attribution does not reproduce the retained result")
    table.write_parquet(path)
    return path


def _part_a(
    root: Path,
    dates: Sequence[date],
    signal_names: Sequence[str],
    signals: np.ndarray,
    target_ranks: np.ndarray,
    raw_returns: np.ndarray,
) -> dict[str, Path]:
    windows = _windows(dates)
    daily_rows: list[dict[str, object]] = []
    persistence_rows: list[dict[str, object]] = []
    for signal_index, signal_name in enumerate(signal_names):
        signal_kind = (
            "context" if signal_name in ("reversal_5d", "momentum_20d") else "oof"
        )
        for horizon_index, horizon in enumerate(TARGET_HORIZONS):
            for day in range(len(dates) - horizon):
                daily_rows.append(
                    {
                        "signal_date": dates[day],
                        "signal": signal_name,
                        "signal_kind": signal_kind,
                        "horizon_sessions": horizon,
                        "ic": _cross_sectional_correlation(
                            signals[day, signal_index], target_ranks[horizon_index, day]
                        ),
                        "decile_spread_bps_per_holding_day": _decile_spread(
                            signals[day, signal_index],
                            raw_returns[horizon_index, day],
                            horizon,
                        ),
                    }
                )
        for day in range(1, len(dates)):
            persistence_rows.append(
                {
                    "trade_date": dates[day],
                    "prior_trade_date": dates[day - 1],
                    "signal": signal_name,
                    "signal_kind": signal_kind,
                    "rank_autocorrelation": _cross_sectional_correlation(
                        signals[day - 1, signal_index], signals[day, signal_index]
                    ),
                }
            )
    daily = pl.DataFrame(daily_rows)
    persistence_daily = pl.DataFrame(persistence_rows)
    readouts = []
    decay = []
    spread = []
    persistence = []
    for signal_index, signal_name in enumerate(signal_names):
        signal_kind = (
            "context" if signal_name in ("reversal_5d", "momentum_20d") else "oof"
        )
        for window_index, (window, positions) in enumerate(windows.items()):
            position_set = set(positions.tolist())
            allowed_dates = [dates[index] for index in positions]
            for horizon_index, horizon in enumerate(TARGET_HORIZONS):
                values = daily.filter(
                    (pl.col("signal") == signal_name)
                    & (pl.col("horizon_sessions") == horizon)
                    & pl.col("signal_date").is_in(allowed_dates)
                )
                ic = _interval(
                    values["ic"].to_numpy(),
                    1000 * signal_index + 100 * window_index + horizon_index,
                )
                decile = _interval(
                    values["decile_spread_bps_per_holding_day"].to_numpy(),
                    20000 + 1000 * signal_index + 100 * window_index + horizon_index,
                )
                base = {
                    "signal": signal_name,
                    "signal_kind": signal_kind,
                    "window": window,
                    "horizon_sessions": horizon,
                    "date_count": values.height,
                }
                readouts.append({**base, **ic})
                decay.append({**base, "mean_ic": ic["estimate"]})
                spread.append({**base, **decile})
            pair_dates = [
                dates[index]
                for index in positions
                if index > 0 and index - 1 in position_set
            ]
            values = persistence_daily.filter(
                (pl.col("signal") == signal_name)
                & pl.col("trade_date").is_in(pair_dates)
            )
            persistence.append(
                {
                    "signal": signal_name,
                    "signal_kind": signal_kind,
                    "window": window,
                    "pair_count": values.height,
                    **_interval(
                        values["rank_autocorrelation"].to_numpy(),
                        40000 + 1000 * signal_index + 100 * window_index,
                    ),
                }
            )
    paths = {
        "daily_ic_and_spread": root / "part_a_daily.parquet",
        "ic_readouts": root / "part_a_ic_readouts.parquet",
        "ic_decay": root / "part_a_ic_decay.parquet",
        "decile_spreads": root / "part_a_decile_spreads.parquet",
        "persistence_daily": root / "part_a_persistence_daily.parquet",
        "persistence_readouts": root / "part_a_persistence_readouts.parquet",
    }
    daily.write_parquet(paths["daily_ic_and_spread"])
    pl.DataFrame(readouts).write_parquet(paths["ic_readouts"])
    pl.DataFrame(decay).write_parquet(paths["ic_decay"])
    pl.DataFrame(spread).write_parquet(paths["decile_spreads"])
    persistence_daily.write_parquet(paths["persistence_daily"])
    pl.DataFrame(persistence).write_parquet(paths["persistence_readouts"])
    return paths


def _tail_weights(scores: np.ndarray, eligible: np.ndarray, k: int) -> np.ndarray:
    names = np.flatnonzero(eligible & np.isfinite(scores))
    result = np.zeros(scores.shape, dtype=np.float64)
    if names.size < 2 * k:
        return result
    order = names[np.argsort(scores[names], kind="stable")]
    result[order[:k]] = -1.0 / k
    result[order[-k:]] = 1.0 / k
    return result


def _cell_weights(
    signal: np.ndarray,
    close: np.ndarray,
    active: np.ndarray,
    k: int,
    band: float,
) -> np.ndarray:
    days, names = signal.shape
    result = np.zeros((days - 1, names), dtype=np.float64)
    for day in range(days - 1):
        eligible = (
            active[day]
            & np.isfinite(signal[day])
            & np.isfinite(close[day])
            & np.isfinite(close[day + 1])
        )
        effective = signal[day].copy()
        if band > 0.0 and day > 0:
            both = np.isfinite(signal[day - 1]) & np.isfinite(signal[day])
            unchanged = both & (np.abs(signal[day] - signal[day - 1]) <= band)
            effective[unchanged] = signal[day - 1, unchanged]
        result[day] = _tail_weights(effective, eligible, k)
    return result


def _turnover_with_terminal_liquidation(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or not weights.shape[0]:
        raise ValueError("Swing weights must be a nonempty day-name matrix")
    prior = np.vstack((np.zeros((1, weights.shape[1])), weights[:-1]))
    turnover = np.abs(weights - prior)
    turnover[-1] += np.abs(weights[-1])
    return turnover


def _liquidity_buckets(adv: np.ndarray, active: np.ndarray) -> np.ndarray:
    buckets = np.full(adv.shape, -1, dtype=np.int8)
    for day in range(adv.shape[0]):
        names = np.flatnonzero(active[day] & np.isfinite(adv[day]))
        if names.size:
            order = names[np.argsort(adv[day, names], kind="stable")]
            buckets[day, order] = np.minimum(np.arange(order.size) * 3 // names.size, 2)
    return buckets


def _part_b(
    root: Path,
    design: Mapping[str, object],
    dates: Sequence[date],
    signal_names: Sequence[str],
    signals: np.ndarray,
) -> tuple[dict[str, Path], list[dict[str, object]]]:
    source_root = Path(str(design["inputs"]["experiment57_stage0_root"]))
    close = np.asarray(_market_array(root, "daily_close.npy"), dtype=np.float64)
    active = np.asarray(
        np.load(
            source_root / "market_inputs" / "active.npy",
            mmap_mode="r",
            allow_pickle=False,
        ),
        dtype=bool,
    )
    adv = np.asarray(
        np.load(
            source_root / "market_inputs" / "adv20_brl.npy",
            mmap_mode="r",
            allow_pickle=False,
        ),
        dtype=np.float64,
    )
    cdi_record = design["inputs"]["cdi"]["parquet"]
    cdi = load_daily_cdi_rates(
        Path(cdi_record["path"]), dates, str(cdi_record["sha256"])
    )
    buckets = _liquidity_buckets(adv, active)
    windows = _windows(dates)
    daily_rows: list[dict[str, object]] = []
    liquidity_rows: list[dict[str, object]] = []
    for signal_name in PART_B_SIGNALS:
        signal = np.asarray(
            signals[:, signal_names.index(signal_name)], dtype=np.float64
        )
        for k in CONCENTRATIONS:
            for band in RANK_BANDS:
                weights = _cell_weights(signal, close, active, k, band)
                turnover_by_name = _turnover_with_terminal_liquidation(weights)
                returns = close[1:] / close[:-1] - 1.0
                gross_by_name = np.where(np.isfinite(returns), weights * returns, 0.0)
                gross = gross_by_name.sum(axis=1) * 10_000.0
                turnover = turnover_by_name.sum(axis=1)
                short = np.maximum(-weights, 0.0)
                deployed_gross = np.abs(weights).sum(axis=1)
                for cost_bps in COSTS_BPS:
                    cost_by_name = turnover_by_name * cost_bps
                    for borrow_rate in BORROW_RATES:
                        borrow_by_name = short * (borrow_rate / 252.0) * 10_000.0
                        cdi_earned = (
                            cdi[1:]
                            * np.maximum(1.0 - MARGIN_FRACTION * deployed_gross, 0.0)
                            * 10_000.0
                        )
                        all_cash_cdi = cdi[1:] * 10_000.0
                        net = (
                            gross
                            - cost_by_name.sum(axis=1)
                            - borrow_by_name.sum(axis=1)
                            + cdi_earned
                        )
                        excess = net - all_cash_cdi
                        cell_id = f"{signal_name}__k{k}__band{band:.1f}"
                        for day in range(weights.shape[0]):
                            daily_rows.append(
                                {
                                    "signal_date": dates[day],
                                    "exit_date": dates[day + 1],
                                    "cell_id": cell_id,
                                    "signal": signal_name,
                                    "k_per_side": k,
                                    "rank_band": band,
                                    "cost_bps_per_side": cost_bps,
                                    "annual_borrow_rate": borrow_rate,
                                    "gross_pnl_bps": gross[day],
                                    "turnover_fraction_nav": turnover[day],
                                    "turnover_cost_bps": cost_by_name[day].sum(),
                                    "borrow_cost_bps": borrow_by_name[day].sum(),
                                    "cdi_earned_bps": cdi_earned[day],
                                    "all_cash_cdi_bps": all_cash_cdi[day],
                                    "net_pnl_bps": net[day],
                                    "net_excess_all_cash_bps": excess[day],
                                    "deployed_gross_fraction_nav": deployed_gross[day],
                                }
                            )
                        net_by_name = (
                            gross_by_name * 10_000.0 - cost_by_name - borrow_by_name
                        )
                        for tercile in range(3):
                            take = buckets[:-1] == tercile
                            liquidity_rows.append(
                                {
                                    "cell_id": cell_id,
                                    "signal": signal_name,
                                    "k_per_side": k,
                                    "rank_band": band,
                                    "cost_bps_per_side": cost_bps,
                                    "annual_borrow_rate": borrow_rate,
                                    "liquidity_tercile": tercile,
                                    "mean_net_trading_attribution_bps_per_day": float(
                                        np.where(take, net_by_name, 0.0)
                                        .sum(axis=1)
                                        .mean()
                                    ),
                                    "mean_turnover_fraction_nav_per_day": float(
                                        np.where(take, turnover_by_name, 0.0)
                                        .sum(axis=1)
                                        .mean()
                                    ),
                                }
                            )
    daily = pl.DataFrame(daily_rows)
    grid_rows = []
    headline = []
    for cell_index, cell_id in enumerate(daily["cell_id"].unique().sort()):
        cell = daily.filter(pl.col("cell_id") == cell_id)
        for cost_index, cost in enumerate(COSTS_BPS):
            for borrow_index, borrow in enumerate(BORROW_RATES):
                sensitivity = cell.filter(
                    (pl.col("cost_bps_per_side") == cost)
                    & (pl.col("annual_borrow_rate") == borrow)
                )
                for window_index, (window, positions) in enumerate(windows.items()):
                    allowed_dates = [
                        dates[index] for index in positions if index < len(dates) - 1
                    ]
                    values = sensitivity.filter(
                        pl.col("signal_date").is_in(allowed_dates)
                    )
                    interval = _interval(
                        values["net_excess_all_cash_bps"].to_numpy(),
                        60000
                        + 1000 * cell_index
                        + 100 * cost_index
                        + 10 * borrow_index
                        + window_index,
                    )
                    gross_sum = float(values["deployed_gross_fraction_nav"].sum())
                    turnover_sum = float(values["turnover_fraction_nav"].sum())
                    row = {
                        "cell_id": cell_id,
                        "signal": values.item(0, "signal") if values.height else "",
                        "k_per_side": int(values.item(0, "k_per_side"))
                        if values.height
                        else 0,
                        "rank_band": float(values.item(0, "rank_band"))
                        if values.height
                        else math.nan,
                        "cost_bps_per_side": cost,
                        "annual_borrow_rate": borrow,
                        "window": window,
                        "date_count": values.height,
                        **interval,
                        "annualized_net_sharpe": _sharpe(
                            values["net_pnl_bps"].to_numpy()
                        ),
                        "mean_turnover_fraction_nav_per_day": float(
                            values["turnover_fraction_nav"].mean()
                        ),
                        "average_holding_days": 2.0 * gross_sum / turnover_sum
                        if turnover_sum > 0.0
                        else math.nan,
                    }
                    grid_rows.append(row)
                    if (
                        window == "all_train"
                        and cost == 4.0
                        and interval["lower_95"] > 0.0
                    ):
                        headline.append(row)
    paths = {
        "daily": root / "part_b_daily.parquet",
        "grid": root / "part_b_grid.parquet",
        "liquidity": root / "part_b_liquidity_attribution.parquet",
    }
    daily.write_parquet(paths["daily"])
    pl.DataFrame(grid_rows).write_parquet(paths["grid"])
    pl.DataFrame(liquidity_rows).write_parquet(paths["liquidity"])
    return paths, headline


def _part_c(
    root: Path,
    design: Mapping[str, object],
    dates: Sequence[date],
    signal_names: Sequence[str],
    signals: np.ndarray,
) -> Path:
    source_root = Path(str(design["inputs"]["experiment57_stage0_root"]))
    close = np.asarray(_market_array(root, "daily_close.npy"), dtype=np.float64)
    morning_low = np.asarray(_market_array(root, "morning_low.npy"))
    morning_high = np.asarray(_market_array(root, "morning_high.npy"))
    session_low = np.asarray(_market_array(root, "session_low.npy"))
    session_high = np.asarray(_market_array(root, "session_high.npy"))
    morning_fallback = np.asarray(_market_array(root, "morning_60m_close.npy"))
    active = np.asarray(
        np.load(
            source_root / "market_inputs" / "active.npy",
            mmap_mode="r",
            allow_pickle=False,
        ),
        dtype=bool,
    )
    spread = np.asarray(
        np.load(
            source_root / "market_inputs" / "full_spread.npy",
            mmap_mode="r",
            allow_pickle=False,
        ),
        dtype=np.float64,
    )
    rows = []
    for signal_name in PART_B_SIGNALS:
        signal = np.asarray(
            signals[:, signal_names.index(signal_name)], dtype=np.float64
        )
        for k in CONCENTRATIONS:
            selected: list[tuple[int, np.ndarray, np.ndarray]] = []
            for day in range(len(dates) - 1):
                eligible = (
                    active[day] & np.isfinite(signal[day]) & np.isfinite(close[day])
                )
                names = np.flatnonzero(eligible)
                if names.size < 2 * k:
                    selected.append(
                        (day, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int8))
                    )
                    continue
                order = names[np.argsort(signal[day, names], kind="stable")]
                chosen = np.concatenate((order[-k:], order[:k]))
                direction = np.concatenate(
                    (np.ones(k, dtype=np.int8), -np.ones(k, dtype=np.int8))
                )
                selected.append((day, chosen, direction))
            for level in LIMIT_LEVELS:
                for wait in WAIT_WINDOWS:
                    low = morning_low if wait == "morning_60m" else session_low
                    high = morning_high if wait == "morning_60m" else session_high
                    fallback = morning_fallback if wait == "morning_60m" else close
                    for horizon in TARGET_HORIZONS:
                        orders = fills = 0
                        unconditional: list[float] = []
                        conditional: list[float] = []
                        composite: list[float] = []
                        for day, names, direction in selected:
                            if day + horizon >= len(dates) or not names.size:
                                continue
                            base = close[day, names]
                            width = np.where(
                                np.isfinite(spread[day, names]),
                                spread[day, names],
                                np.nan,
                            )
                            limits = base.copy()
                            if level == "inside_half_half_spread":
                                limits = base * (1.0 - direction * 0.25 * width)
                            future_low = low[day + 1, names]
                            future_high = high[day + 1, names]
                            path_known = np.where(
                                direction > 0,
                                np.isfinite(future_low),
                                np.isfinite(future_high),
                            )
                            exit_price = close[day + horizon, names]
                            valid = (
                                path_known
                                & np.isfinite(limits)
                                & np.isfinite(exit_price)
                            )
                            if not valid.any():
                                continue
                            through = _strict_through(
                                direction, future_low, future_high, limits
                            )
                            filled = valid & through
                            uncond = (
                                direction
                                * (exit_price / base - 1.0)
                                * 10_000.0
                                / horizon
                            )
                            unconditional.extend(uncond[valid].tolist())
                            fill_alpha = (
                                direction
                                * (exit_price / limits - 1.0)
                                * 10_000.0
                                / horizon
                            )
                            conditional.extend(fill_alpha[filled].tolist())
                            fallback_price = fallback[day + 1, names]
                            entry = np.where(filled, limits, fallback_price)
                            composite_valid = valid & np.isfinite(entry)
                            comp = (
                                direction
                                * (exit_price / entry - 1.0)
                                * 10_000.0
                                / horizon
                            )
                            composite.extend(comp[composite_valid].tolist())
                            orders += int(valid.sum())
                            fills += int(filled.sum())
                        unconditional_mean = (
                            float(np.mean(unconditional)) if unconditional else math.nan
                        )
                        conditional_mean = (
                            float(np.mean(conditional)) if conditional else math.nan
                        )
                        rows.append(
                            {
                                "signal": signal_name,
                                "k_per_side": k,
                                "limit_level": level,
                                "wait_window": wait,
                                "horizon_sessions": horizon,
                                "order_count": orders,
                                "fill_count": fills,
                                "fill_rate": fills / orders if orders else math.nan,
                                "unconditional_alpha_bps_per_holding_day": unconditional_mean,
                                "conditional_fill_alpha_bps_per_holding_day": conditional_mean,
                                "adverse_selection_gap_bps_per_holding_day": conditional_mean
                                - unconditional_mean,
                                "limit_then_taker_alpha_bps_per_holding_day": float(
                                    np.mean(composite)
                                )
                                if composite
                                else math.nan,
                                "composite_count": len(composite),
                            }
                        )
    path = root / "part_c_patient_entry.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


def _strict_through(
    direction: np.ndarray,
    observed_low: np.ndarray,
    observed_high: np.ndarray,
    limit: np.ndarray,
) -> np.ndarray:
    direction = np.asarray(direction)
    return np.where(direction > 0, observed_low < limit, observed_high > limit)


def run(root: Path) -> Path:
    root = root.resolve()
    final = root / "result.json"
    if final.exists():
        _verified_result(root, "result.json")
        return final
    design = _read_json(root / "frozen_design.json")
    _assert_frozen(design)
    market_manifest = _build_daily_market(root, design)
    archive = _load_archive(design)
    source_root = Path(str(design["inputs"]["experiment57_stage0_root"]))
    source_date_idx = np.load(
        source_root / "market_inputs" / "date_idx.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    if not np.array_equal(archive.date_idx, source_date_idx):
        raise ValueError("OOF and retained TRAIN market date axes differ")
    dates = tuple(
        pl.read_parquet(source_root / "market_inputs" / "dates.parquet").sort(
            "date_idx"
        )["trade_date"]
    )
    signal_names, signals = _build_signals(root, design, archive)
    target_ranks, raw_returns = _build_targets(root, design)
    part0 = _part0(root, design)
    part_a = _part_a(root, dates, signal_names, signals, target_ranks, raw_returns)
    part_b, headline = _part_b(root, design, dates, signal_names, signals)
    part_c = _part_c(root, design, dates, signal_names, signals)
    artifacts = {
        "daily_market_manifest": _artifact(root / "daily_market" / "manifest.json"),
        "signals": _artifact(root / "signals.npz"),
        "targets": _artifact(root / "targets.npz"),
        "part0_attribution": _artifact(part0),
        **{f"part_a_{name}": _artifact(path) for name, path in part_a.items()},
        **{f"part_b_{name}": _artifact(path) for name, path in part_b.items()},
        "part_c_patient_entry": _artifact(part_c),
    }
    result = {
        "schema": SCHEMA,
        "status": "completed",
        "repository_commit": repository_commit(),
        "train_date_count": len(dates),
        "signal_count_including_context": len(signal_names),
        "part_b_cell_count": 8,
        "part_b_sensitivity_count": 48,
        "middle_cost_supported_cell_count": len(headline),
        "middle_cost_supported_cells": [
            {
                "cell_id": row["cell_id"],
                "annual_borrow_rate": row["annual_borrow_rate"],
                "estimate": row["estimate"],
                "lower_95": row["lower_95"],
                "upper_95": row["upper_95"],
            }
            for row in headline
        ],
        "daily_market": market_manifest,
        "artifacts": artifacts,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(final, result)
    _atomic_json(
        root / "final_audit.json",
        {
            "schema": "EXPERIMENT58_BASE_AUDIT_V1",
            "status": "passed",
            "result_sha256": _sha256(final),
            "exact_716_train_dates": len(dates) == 716,
            "all_target_horizons": list(TARGET_HORIZONS),
            "all_eight_part_b_cells": True,
            "all_48_part_b_sensitivities": True,
            "part_c_informational_only": True,
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    return final


def audit(root: Path) -> Path:
    root = root.resolve()
    result, _ = _verified_result(root, "result.json")
    if result.get("schema") != SCHEMA:
        raise ValueError("Experiment 58 result schema differs")
    required = (
        "operational.stdout.log",
        "operational.stderr.log",
        "part0_attribution.parquet",
        "part_a_ic_readouts.parquet",
        "part_b_grid.parquet",
        "part_c_patient_entry.parquet",
        "final_audit.json",
    )
    if any(not (root / name).is_file() for name in required):
        raise ValueError("Experiment 58 root lacks a required final artifact or log")
    rows = []
    access_files = 0
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        if path.name in ("full_inventory.parquet", "full_inventory_audit.json"):
            continue
        if path.suffix == ".json":
            payload = _read_json(path)
            for key in ("official_validation_accessed", "test_accessed"):
                if key in payload and payload[key] is not False:
                    raise ValueError(f"Access flag differs in {path}")
            if any(
                key in payload
                for key in ("official_validation_accessed", "test_accessed")
            ):
                access_files += 1
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    inventory_path = root / "full_inventory.parquet"
    pl.DataFrame(rows).write_parquet(inventory_path)
    audit_path = root / "full_inventory_audit.json"
    _atomic_json(
        audit_path,
        {
            "schema": "EXPERIMENT58_FULL_INVENTORY_AUDIT_V1",
            "status": "passed",
            "inventory": _artifact(inventory_path),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "json_access_flag_file_count": access_files,
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    return audit_path


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 58 swing screen")
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--section-b-root", type=Path, required=True)
    freeze_parser.add_argument("--experiment57-stage0-root", type=Path, required=True)
    freeze_parser.add_argument("--preregistration", type=Path, required=True)
    freeze_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--root", type=Path, required=True)
    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    if args.command == "freeze":
        path = freeze(
            section_b_root=args.section_b_root,
            experiment57_stage0_root=args.experiment57_stage0_root,
            preregistration=args.preregistration,
            output_dir=args.output_dir,
        )
    elif args.command == "run":
        path = run(args.root)
    else:
        path = audit(args.root)
    print(path)


if __name__ == "__main__":
    main()
