from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from ..modeling.data import feature_store_identity
from ..modeling.provenance import repository_commit
from ..preprocessing.contract import EQUITY_SESSION_MINUTES
from .experiment52 import FOLDS, _load_cache_array
from .experiment53 import _verify_experiment52
from .inputs import (
    iter_discovery_equity_grids,
    load_daily_cdi_rates,
    load_discovery_prediction_archive,
)

SCHEMA = "EXPERIMENT54_EDGE_MAKER_V1"
HORIZONS = (15, 30, 60, 120)
ENTRIES = ("decision_open", "next_open", "mean_open_10", "mean_open_30")
THRESHOLDS_BPS = (4.5, 7.0, 10.0)
WAITS = (5, 15, 30)
LIMIT_VARIANTS = ("last_close", "improved_half_half_spread")
NAV_BRL = 10_000_000.0
GROSS_LIMIT = 2.0
PARTICIPATION_RATE = 0.10
NAME_CAP_FRACTION_OF_GROSS = 0.05
FEE_BPS = 2.0
MARGIN_FRACTION_OF_GROSS = 0.5
TAKER_DECISION_THRESHOLD_BPS = 7.0
FRONTIER_HURDLE_BPS_PER_DAY = 8.0


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


def _save_array(directory: Path, name: str, values: np.ndarray) -> dict[str, object]:
    path = directory / name
    with path.open("wb") as output:
        np.save(output, values, allow_pickle=False)
    return {"path": name, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _verify_artifact_record(root: Path, record: Mapping[str, object]) -> None:
    path = root / str(record["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(record["bytes"])
        or _sha256(path) != record["sha256"]
    ):
        raise ValueError(f"Source artifact differs: {path}")


def _verify_experiment53(root: Path) -> dict[str, object]:
    manifest = _read_json(root / "program_manifest.json")
    audit_path = root / "final_audit.json"
    audit = _read_json(audit_path)
    result = _read_json(root / "experiment53_result.json")
    designation = _read_json(root / "c1_designation.json")
    if (
        manifest.get("status") != "completed"
        or manifest.get("report_count") != 432
        or audit.get("status") != "passed"
        or audit.get("standard_report_count") != 432
        or audit.get("summary_rows") != 432
        or designation.get("c1_cell_id")
        != "k40__band1p5__c1p0__gross1p0__universe_full"
        or result.get("c0_cell_id") != "band_2p0__blend_equal"
        or any(
            payload.get("official_validation_accessed") is not False
            or payload.get("test_accessed") is not False
            for payload in (manifest, audit, result, designation)
        )
    ):
        raise ValueError("Experiment 53 source contract differs")
    for record in audit.get("artifacts", []):
        _verify_artifact_record(root, record)
    design = _read_json(root / "frozen_design.json")
    source52 = Path(str(design["source_experiment52"]["root"])).resolve()
    _verify_experiment52(source52)
    return {
        "root": str(root),
        "program_manifest": _artifact(root / "program_manifest.json"),
        "frozen_design": _artifact(root / "frozen_design.json"),
        "result": _artifact(root / "experiment53_result.json"),
        "c1_designation": _artifact(root / "c1_designation.json"),
        "final_audit": _artifact(audit_path),
        "source_experiment52_root": str(source52),
        "official_validation_accessed": False,
        "test_accessed": False,
    }


def _verify_cache(directory: Path) -> dict[str, object]:
    manifest = _read_json(directory / "manifest.json")
    for name, record in manifest["artifacts"].items():
        path = directory / name
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or _sha256(path) != record["sha256"]
        ):
            raise ValueError(f"Experiment 54 OHLC cache differs: {path}")
    return manifest


def _build_ohlc_cache(
    *, store: Path, source52_root: Path, output_dir: Path
) -> dict[str, object]:
    final = output_dir / "raw_ohlc"
    if final.exists():
        return _verify_cache(final)
    temporary = output_dir / ".raw_ohlc.tmp"
    temporary.mkdir()
    requested_idx = np.asarray(_load_cache_array(source52_root, "date_idx.npy"))
    dates = pl.read_parquet(source52_root / "market_inputs" / "dates.parquet").sort(
        "date_idx"
    )
    if not np.array_equal(dates["date_idx"].to_numpy(), requested_idx):
        raise ValueError("Experiment 52 market dates are not in canonical order")
    trade_dates = tuple(dates["trade_date"])
    source_open = _load_cache_array(source52_root, "open_price.npy")
    source_observed = _load_cache_array(source52_root, "open_observed.npy")
    shape = source_open.shape
    high = np.full(shape, np.nan, dtype=np.float64)
    low = np.full(shape, np.nan, dtype=np.float64)
    close = np.full(shape, np.nan, dtype=np.float64)
    seen = np.zeros(shape[2], dtype=bool)
    for grid in iter_discovery_equity_grids(store):
        position = {value: index for index, value in enumerate(grid.trade_dates)}
        source_positions = np.asarray([position[value] for value in trade_dates])
        slot = grid.equity_slot
        observed = grid.observed[source_positions]
        expected_observed = np.asarray(source_observed[:, :, slot])
        if not np.array_equal(observed, expected_observed):
            raise ValueError(f"Raw observed mask differs for {grid.security_id}")
        expected_open = np.asarray(source_open[:, :, slot])
        if not np.array_equal(
            np.where(observed, grid.open_price[source_positions], 0.0),
            np.where(expected_observed, expected_open, 0.0),
        ):
            raise ValueError(f"Raw open prices differ for {grid.security_id}")
        high[:, :, slot] = grid.high[source_positions]
        low[:, :, slot] = grid.low[source_positions]
        close[:, :, slot] = grid.close[source_positions]
        seen[slot] = True
    if not seen.all():
        raise ValueError("Raw OHLC bridge did not emit every permanent security")
    observed = np.asarray(source_observed, dtype=bool)
    open_price = np.asarray(source_open)
    valid = (
        np.isfinite(open_price)
        & np.isfinite(high)
        & np.isfinite(low)
        & np.isfinite(close)
        & (open_price > 0.0)
        & (high > 0.0)
        & (low > 0.0)
        & (close > 0.0)
        & (high >= np.maximum(open_price, close))
        & (low <= np.minimum(open_price, close))
    )
    if not valid[observed].all():
        raise ValueError("Observed raw bars fail the frozen OHLC contract")
    artifacts = {
        name: _save_array(temporary, name, values)
        for name, values in {
            "high_price.npy": high,
            "low_price.npy": low,
            "close_price.npy": close,
        }.items()
    }
    manifest = {
        "schema": "EXPERIMENT54_RAW_OHLC_V1",
        "created_at": _now(),
        "date_count": shape[0],
        "minute_count": shape[1],
        "security_count": shape[2],
        "observed_bar_count": int(observed.sum()),
        "ohlc_valid_observed_bar_count": int(valid[observed].sum()),
        "source_open": _artifact(source52_root / "market_inputs" / "open_price.npy"),
        "source_observed": _artifact(
            source52_root / "market_inputs" / "open_observed.npy"
        ),
        "artifacts": artifacts,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(temporary / "manifest.json", manifest)
    os.replace(temporary, final)
    return _verify_cache(final)


def quantile_edges(values: np.ndarray, probabilities: Sequence[float]) -> list[float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Frozen bucket source is empty")
    result = np.quantile(finite, probabilities, method="linear")
    if not np.isfinite(result).all() or np.any(np.diff(result) < 0.0):
        raise ValueError("Frozen bucket edges are invalid")
    return [float(value) for value in result]


def _fold_archive(design: Mapping[str, object], fold: str, store: Path):
    source = design["fold_sources"][fold]
    return load_discovery_prediction_archive(
        Path(source["ensemble_prediction"]["path"]),
        Path(source["prediction_reference"]["path"]),
        Path(source["execution_manifest"]["path"]),
        store,
    )


def _freeze_buckets(
    *, source_design: Mapping[str, object], store: Path, source52_root: Path
) -> dict[str, object]:
    archive = _fold_archive(source_design, "fold_c", store)
    cache_idx = np.asarray(_load_cache_array(source52_root, "date_idx.npy"))
    position = {int(value): index for index, value in enumerate(cache_idx)}
    rows = np.asarray([position[int(value)] for value in archive.date_idx])
    ranks = np.asarray(archive.ranks, dtype=np.float64)
    current_valid = np.asarray(archive.valid, dtype=bool).all(axis=-1)
    blended = ranks.mean(axis=-1)
    previous_valid = np.zeros_like(current_valid)
    previous_valid[:, 1:] = current_valid[:, :-1]
    delta = np.full_like(blended, np.nan)
    delta[:, 1:] = np.abs(blended[:, 1:] - blended[:, :-1])
    adv = np.asarray(_load_cache_array(source52_root, "adv20_brl.npy")[rows])
    spread = np.asarray(_load_cache_array(source52_root, "full_spread.npy")[rows])
    sigma = np.asarray(_load_cache_array(source52_root, "sigma_daily.npy")[rows])
    state_mask = (
        current_valid
        & previous_valid
        & np.isfinite(blended)
        & (blended != 0.0)
        & np.isfinite(delta)
        & np.isfinite(adv[:, None, :])
        & (adv[:, None, :] > 0.0)
        & np.isfinite(spread[:, None, :])
        & (spread[:, None, :] >= 0.0)
        & np.isfinite(sigma[:, None, :])
        & (sigma[:, None, :] > 0.0)
    )
    current_mask = current_valid & np.isfinite(blended)
    return {
        "schema": "EXPERIMENT54_BUCKET_DEFINITIONS_V1",
        "source_fold": "fold_c",
        "assignment": "numpy.searchsorted(edges, value, side='right')",
        "rank_decile_edges": quantile_edges(
            blended[current_mask], tuple(index / 10 for index in range(1, 10))
        ),
        "absolute_delta_quintile_edges": quantile_edges(
            delta[state_mask], (0.2, 0.4, 0.6, 0.8)
        ),
        "adv20_tercile_edges": quantile_edges(
            np.broadcast_to(adv[:, None, :], blended.shape)[state_mask],
            (1 / 3, 2 / 3),
        ),
        "full_spread_tercile_edges": quantile_edges(
            np.broadcast_to(spread[:, None, :], blended.shape)[state_mask],
            (1 / 3, 2 / 3),
        ),
        "daily_sigma_tercile_edges": quantile_edges(
            np.broadcast_to(sigma[:, None, :], blended.shape)[state_mask],
            (1 / 3, 2 / 3),
        ),
        "eligible_fold_c_state_event_count": int(state_mask.sum()),
        "first_refresh_excluded": True,
        "tail_rank_deciles": [0, 1, 8, 9],
        "tod_definition": (
            "first/final 60 minutes of the prediction refresh schedule; middle "
            "otherwise"
        ),
        "official_validation_accessed": False,
        "test_accessed": False,
    }


def freeze_program(
    *, experiment53_root: Path, preregistration: Path, output_dir: Path
) -> Path:
    experiment53_root, preregistration, output_dir = (
        path.resolve() for path in (experiment53_root, preregistration, output_dir)
    )
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    try:
        source53 = _verify_experiment53(experiment53_root)
        source52_root = Path(str(source53["source_experiment52_root"]))
        source52_design = _read_json(source52_root / "frozen_design.json")
        store = Path(str(source52_design["store"]["path"])).resolve()
        if feature_store_identity(store) != source52_design["store"]["identity"]:
            raise ValueError("Canonical store identity differs")
        ohlc = _build_ohlc_cache(
            store=store, source52_root=source52_root, output_dir=output_dir
        )
        buckets = _freeze_buckets(
            source_design=source52_design, store=store, source52_root=source52_root
        )
        bucket_path = output_dir / "bucket_definitions.json"
        _atomic_json(bucket_path, buckets)
        design = {
            "schema": SCHEMA,
            "status": "frozen",
            "created_at": _now(),
            "repository_commit": repository_commit(),
            "preregistration": _artifact(preregistration),
            "source_experiment53": source53,
            "source_experiment52": _verify_experiment52(source52_root),
            "source_experiment52_root": str(source52_root),
            "store": {"path": str(store), "identity": feature_store_identity(store)},
            "fold_sources": source52_design["fold_sources"],
            "cdi": source52_design["cdi"],
            "raw_ohlc": {
                **ohlc,
                "manifest": _artifact(output_dir / "raw_ohlc" / "manifest.json"),
            },
            "bucket_definitions": _artifact(bucket_path),
            "horizons": list(HORIZONS),
            "entries": list(ENTRIES),
            "thresholds_bps": list(THRESHOLDS_BPS),
            "waits": list(WAITS),
            "limit_variants": list(LIMIT_VARIANTS),
            "sizing": {
                "nav_brl": NAV_BRL,
                "gross_limit": GROSS_LIMIT,
                "participation_rate": PARTICIPATION_RATE,
                "name_cap_fraction_of_gross": NAME_CAP_FRACTION_OF_GROSS,
                "fee_bps": FEE_BPS,
                "margin_fraction_of_gross": MARGIN_FRACTION_OF_GROSS,
                "side_neutrality": False,
            },
            "taker_decision": {
                "threshold_bps": TAKER_DECISION_THRESHOLD_BPS,
                "hurdle_nav_bps_per_day": FRONTIER_HURDLE_BPS_PER_DAY,
                "closed_rule": "below hurdle on every fold",
                "viable_rule": "clears hurdle on at least two folds",
                "otherwise": "inconclusive",
            },
            "cpu_only": True,
            "simulator_run": False,
            "official_validation_accessed": False,
            "test_accessed": False,
        }
        design_path = output_dir / "frozen_design.json"
        _atomic_json(design_path, design)
        _atomic_json(
            output_dir / "program_manifest.json",
            {
                "schema": SCHEMA,
                "status": "frozen",
                "created_at": _now(),
                "repository_commit": design["repository_commit"],
                "frozen_design": _artifact(design_path),
                "official_validation_accessed": False,
                "test_accessed": False,
            },
        )
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return output_dir / "frozen_design.json"


_STATE_COLUMNS = (
    "state_cell_id",
    "rank_decile",
    "delta_quintile",
    "tail_entry",
    "liquidity_tercile",
    "tod_bucket",
    "head_agreement",
    "spread_tercile",
    "sigma_tercile",
)


def _bucket(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    return np.searchsorted(np.asarray(edges), values, side="right").astype(np.int8)


def _tod_buckets(refresh_minutes: np.ndarray) -> np.ndarray:
    values = np.asarray(refresh_minutes, dtype=np.int64)
    first_end = int(values.min()) + 60
    last_start = int(values.max()) - 55
    return np.where(values < first_end, 0, np.where(values >= last_start, 2, 1))


def build_state_events(
    *,
    ranks: np.ndarray,
    valid: np.ndarray,
    refresh_minutes: np.ndarray,
    adv20_brl: np.ndarray,
    full_spread: np.ndarray,
    sigma_daily: np.ndarray,
    minute_notional20_brl: np.ndarray,
    buckets: Mapping[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    ranks = np.asarray(ranks, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if ranks.ndim != 4 or ranks.shape != valid.shape or ranks.shape[-1] != 3:
        raise ValueError("Experiment 54 ranks must be [day,refresh,name,3]")
    days, refreshes, names, _ = ranks.shape
    expected_daily = (days, names)
    expected_minute = (days, EQUITY_SESSION_MINUTES, names)
    if (
        any(
            np.asarray(values).shape != expected_daily
            for values in (adv20_brl, full_spread, sigma_daily)
        )
        or np.asarray(minute_notional20_brl).shape != expected_minute
    ):
        raise ValueError("Experiment 54 causal state inputs do not align")
    current_valid = valid.all(axis=-1)
    blended = ranks.mean(axis=-1)
    previous_valid = np.zeros_like(current_valid)
    previous_valid[:, 1:] = current_valid[:, :-1]
    previous_blended = np.zeros_like(blended)
    previous_blended[:, 1:] = blended[:, :-1]
    delta = np.abs(blended - previous_blended)
    state_mask = (
        current_valid
        & previous_valid
        & np.isfinite(blended)
        & (blended != 0.0)
        & np.isfinite(adv20_brl[:, None, :])
        & (adv20_brl[:, None, :] > 0.0)
        & np.isfinite(full_spread[:, None, :])
        & (full_spread[:, None, :] >= 0.0)
        & np.isfinite(sigma_daily[:, None, :])
        & (sigma_daily[:, None, :] > 0.0)
    )
    day, refresh, name = np.nonzero(state_mask)
    score = blended[day, refresh, name]
    prior_score = previous_blended[day, refresh, name]
    rank_decile = _bucket(score, buckets["rank_decile_edges"])
    prior_decile = _bucket(prior_score, buckets["rank_decile_edges"])
    delta_quintile = _bucket(
        delta[day, refresh, name], buckets["absolute_delta_quintile_edges"]
    )
    liquidity = _bucket(adv20_brl[day, name], buckets["adv20_tercile_edges"])
    spread_bucket = _bucket(
        full_spread[day, name], buckets["full_spread_tercile_edges"]
    )
    sigma_bucket = _bucket(sigma_daily[day, name], buckets["daily_sigma_tercile_edges"])
    tail = np.isin(rank_decile, (0, 1, 8, 9))
    prior_tail = np.isin(prior_decile, (0, 1, 8, 9))
    tail_entry = tail & ~prior_tail
    head_sign = np.sign(ranks[day, refresh, name])
    agreement = np.all(head_sign == head_sign[:, :1], axis=1)
    tod = _tod_buckets(refresh_minutes)[refresh].astype(np.int8)
    state_id = rank_decile.astype(np.int64)
    for values, width in (
        (delta_quintile, 5),
        (tail_entry.astype(np.int8), 2),
        (liquidity, 3),
        (tod, 3),
        (agreement.astype(np.int8), 2),
        (spread_bucket, 3),
        (sigma_bucket, 3),
    ):
        state_id = state_id * width + values
    fill_minute = np.asarray(refresh_minutes, dtype=np.int64)[refresh] + 1
    in_session = fill_minute < EQUITY_SESSION_MINUTES
    capacity = np.zeros(day.size, dtype=np.float64)
    capacity[in_session] = np.minimum(
        PARTICIPATION_RATE
        * minute_notional20_brl[
            day[in_session], fill_minute[in_session], name[in_session]
        ],
        NAME_CAP_FRACTION_OF_GROSS * GROSS_LIMIT * NAV_BRL,
    )
    capacity = np.where(np.isfinite(capacity) & (capacity > 0.0), capacity, 0.0)
    spread_bps = full_spread[day, name] * 10_000.0
    events = {
        "day": day.astype(np.int32),
        "refresh": refresh.astype(np.int16),
        "name": name.astype(np.int16),
        "minute": np.asarray(refresh_minutes, dtype=np.int64)[refresh].astype(np.int16),
        "direction": np.sign(score).astype(np.int8),
        "blended_rank": score,
        "state_cell_id": state_id,
        "rank_decile": rank_decile,
        "delta_quintile": delta_quintile,
        "tail_entry": tail_entry,
        "liquidity_tercile": liquidity,
        "tod_bucket": tod,
        "head_agreement": agreement,
        "spread_tercile": spread_bucket,
        "sigma_tercile": sigma_bucket,
        "tail": tail,
        "full_spread_bps": spread_bps,
        "taker_cost_half_spread_bps": FEE_BPS + 0.25 * spread_bps,
        "taker_cost_measured_bps": FEE_BPS + 0.5 * spread_bps,
        "taker_cost_conservative_bps": FEE_BPS + spread_bps,
        "capacity_brl": capacity,
    }
    exclusions = {
        "current_rank_valid_events": int(current_valid.sum()),
        "first_refresh_current_valid_events": int(current_valid[:, 0].sum()),
        "eligible_state_events": int(day.size),
        "excluded_state_events": int(current_valid.sum() - day.size),
    }
    return events, exclusions


def _window_mean_open(
    open_price: np.ndarray, observed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    safe = np.where(observed, open_price, 0.0)
    sums = np.concatenate(
        (np.zeros((safe.shape[0], 1, safe.shape[2])), np.cumsum(safe, axis=1)),
        axis=1,
    )
    counts = np.concatenate(
        (
            np.zeros((safe.shape[0], 1, safe.shape[2]), dtype=np.int16),
            np.cumsum(observed, axis=1, dtype=np.int16),
        ),
        axis=1,
    )
    return sums, counts


def forward_edge_bps(
    events: Mapping[str, np.ndarray],
    *,
    open_price: np.ndarray,
    close_price: np.ndarray,
    observed: np.ndarray,
    horizon: int,
    entry: str,
    open_sums: np.ndarray | None = None,
    open_counts: np.ndarray | None = None,
) -> np.ndarray:
    day = events["day"].astype(np.int64)
    name = events["name"].astype(np.int64)
    minute = events["minute"].astype(np.int64)
    direction = events["direction"].astype(np.float64)
    if entry == "decision_open":
        start = minute
        end = minute + horizon - 1
        width = 0
    elif entry == "next_open":
        start = minute + 1
        end = start + horizon - 1
        width = 0
    elif entry in {"mean_open_10", "mean_open_30"}:
        width = int(entry.rsplit("_", maxsplit=1)[1])
        start = minute + 1
        end = minute + width + horizon - 1
        if open_sums is None or open_counts is None:
            raise ValueError("Mean-open entries require cumulative open arrays")
    else:
        raise ValueError(f"Unknown Experiment 54 entry: {entry}")
    result = np.full(day.size, np.nan, dtype=np.float64)
    inside = (start >= 0) & (end < open_price.shape[1])
    if not inside.any():
        return result
    selected = np.nonzero(inside)[0]
    d, n, s, e = day[selected], name[selected], start[selected], end[selected]
    if width:
        window_end = s + width
        count = open_counts[d, window_end, n] - open_counts[d, s, n]
        entry_price = (open_sums[d, window_end, n] - open_sums[d, s, n]) / width
        valid_entry = count == width
    else:
        entry_price = open_price[d, s, n]
        valid_entry = observed[d, s, n]
    valid = (
        valid_entry
        & observed[d, e, n]
        & np.isfinite(entry_price)
        & (entry_price > 0.0)
        & np.isfinite(close_price[d, e, n])
        & (close_price[d, e, n] > 0.0)
    )
    chosen = selected[valid]
    result[chosen] = (
        direction[chosen]
        * (close_price[d[valid], e[valid], n[valid]] / entry_price[valid] - 1.0)
        * 10_000.0
    )
    return result


def _event_frame(
    events: Mapping[str, np.ndarray], values: Mapping[str, np.ndarray]
) -> pl.DataFrame:
    payload = {name: events[name] for name in _STATE_COLUMNS}
    payload.update(values)
    return pl.DataFrame(payload)


def conditional_edge_table(
    *,
    fold: str,
    dates: Sequence[date],
    events: Mapping[str, np.ndarray],
    edges: Mapping[tuple[int, str], np.ndarray],
) -> pl.DataFrame:
    tables = []
    day = events["day"].astype(np.int64)
    for horizon in HORIZONS:
        for entry in ENTRIES:
            edge = edges[(horizon, entry)]
            valid = np.isfinite(edge)
            frame = _event_frame(
                {name: values[valid] for name, values in events.items()},
                {
                    "day": day[valid],
                    "gross_edge_bps": edge[valid],
                    "cost_half": events["taker_cost_half_spread_bps"][valid],
                    "cost_measured": events["taker_cost_measured_bps"][valid],
                    "cost_conservative": events["taker_cost_conservative_bps"][valid],
                },
            )
            grouped = frame.group_by(list(_STATE_COLUMNS), maintain_order=True).agg(
                pl.len().alias("event_count"),
                pl.col("day").n_unique().alias("date_count"),
                pl.col("gross_edge_bps").mean().alias("mean_gross_edge_bps"),
                pl.col("gross_edge_bps").median().alias("median_gross_edge_bps"),
                (pl.col("gross_edge_bps") > 4.5).mean().alias("fraction_gt_4p5_bps"),
                (pl.col("gross_edge_bps") > 7.0).mean().alias("fraction_gt_7_bps"),
                (pl.col("gross_edge_bps") > 10.0).mean().alias("fraction_gt_10_bps"),
                (pl.col("gross_edge_bps") > pl.col("cost_half"))
                .mean()
                .alias("fraction_clears_half_spread_cost"),
                (pl.col("gross_edge_bps") > pl.col("cost_measured"))
                .mean()
                .alias("fraction_clears_measured_cost"),
                (pl.col("gross_edge_bps") > pl.col("cost_conservative"))
                .mean()
                .alias("fraction_clears_conservative_cost"),
                pl.col("cost_half").mean().alias("mean_half_spread_cost_bps"),
                pl.col("cost_measured").mean().alias("mean_measured_cost_bps"),
                pl.col("cost_conservative").mean().alias("mean_conservative_cost_bps"),
            )
            tables.append(
                grouped.with_columns(
                    pl.lit(fold).alias("fold"),
                    pl.lit(horizon).alias("horizon_minutes"),
                    pl.lit(entry).alias("entry"),
                    (pl.col("event_count") / len(dates)).alias("events_per_day"),
                )
            )
    return pl.concat(tables, how="diagonal_relaxed")


def latency_decay_table(
    *, fold: str, dates: Sequence[date], edges: Mapping[tuple[int, str], np.ndarray]
) -> pl.DataFrame:
    rows = []
    for horizon in HORIZONS:
        reference = edges[(horizon, "decision_open")]
        for entry in ENTRIES:
            values = edges[(horizon, entry)]
            common = np.isfinite(reference) & np.isfinite(values)
            rows.append(
                {
                    "fold": fold,
                    "horizon_minutes": horizon,
                    "entry": entry,
                    "event_count": int(np.isfinite(values).sum()),
                    "date_count": len(dates),
                    "mean_gross_edge_bps": float(np.nanmean(values)),
                    "median_gross_edge_bps": float(np.nanmedian(values)),
                    "paired_event_count": int(common.sum()),
                    "mean_decay_from_decision_open_bps": float(
                        np.mean(reference[common] - values[common])
                    ),
                    "median_decay_from_decision_open_bps": float(
                        np.median(reference[common] - values[common])
                    ),
                }
            )
    return pl.DataFrame(rows)


def _allocate_frontier(
    *,
    events: Mapping[str, np.ndarray],
    expected_net_bps: np.ndarray,
    eligible: np.ndarray,
    dates: Sequence[date],
    gross_brl: float = GROSS_LIMIT * NAV_BRL,
) -> list[dict[str, object]]:
    day = events["day"].astype(np.int64)
    refresh = events["refresh"].astype(np.int64)
    name = events["name"].astype(np.int64)
    capacity = np.minimum(
        events["capacity_brl"].astype(np.float64),
        NAME_CAP_FRACTION_OF_GROSS * GROSS_LIMIT * NAV_BRL,
    )
    pnl = np.zeros(len(dates), dtype=np.float64)
    used = np.zeros(len(dates), dtype=np.float64)
    traded = np.zeros(len(dates), dtype=np.int64)
    keys = day * (int(refresh.max()) + 1) + refresh
    order = np.argsort(keys, kind="stable")
    ordered_keys = keys[order]
    starts = np.flatnonzero(
        np.concatenate(([True], ordered_keys[1:] != ordered_keys[:-1]))
    )
    ends = np.concatenate((starts[1:], [order.size]))
    for start, end in zip(starts, ends, strict=True):
        group = order[start:end]
        take = (
            eligible[group]
            & np.isfinite(expected_net_bps[group])
            & (expected_net_bps[group] > 0.0)
            & (capacity[group] > 0.0)
        )
        indices = group[take]
        if not indices.size:
            continue
        ranked = np.lexsort((name[indices], -expected_net_bps[indices]))
        indices = indices[ranked]
        prior = np.concatenate(([0.0], np.cumsum(capacity[indices])[:-1]))
        allocated = np.minimum(capacity[indices], np.maximum(gross_brl - prior, 0.0))
        positive = allocated > 0.0
        day_index = int(day[indices[0]])
        pnl[day_index] += float(
            np.sum(allocated[positive] * expected_net_bps[indices[positive]]) / 10_000.0
        )
        used[day_index] += float(allocated[positive].sum())
        traded[day_index] += int(positive.sum())
    return [
        {
            "trade_date": trade_date,
            "expected_net_pnl_brl": float(pnl[index]),
            "expected_net_nav_bps": float(pnl[index] / NAV_BRL * 10_000.0),
            "allocated_notional_brl": float(used[index]),
            "allocated_event_count": int(traded[index]),
        }
        for index, trade_date in enumerate(dates)
    ]


def taker_frontier_tables(
    *,
    fold: str,
    dates: Sequence[date],
    events: Mapping[str, np.ndarray],
    edges: Mapping[tuple[int, str], np.ndarray],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    daily_rows = []
    state = events["state_cell_id"].astype(np.int64)
    measured_cost = events["taker_cost_measured_bps"]
    for horizon in HORIZONS:
        gross = edges[(horizon, "next_open")]
        valid = np.isfinite(gross)
        means = (
            pl.DataFrame({"state": state[valid], "gross": gross[valid]})
            .group_by("state")
            .agg(pl.col("gross").mean().alias("mean"))
        )
        by_state = dict(means.iter_rows())
        expected_gross = np.asarray(
            [by_state.get(int(value), np.nan) for value in state], dtype=np.float64
        )
        expected_net = expected_gross - measured_cost
        for threshold in THRESHOLDS_BPS:
            rows = _allocate_frontier(
                events=events,
                expected_net_bps=expected_net,
                eligible=valid & (expected_gross > threshold),
                dates=dates,
            )
            for row in rows:
                daily_rows.append(
                    {
                        **row,
                        "fold": fold,
                        "horizon_minutes": horizon,
                        "threshold_bps": threshold,
                    }
                )
    daily = pl.DataFrame(daily_rows)
    summary = daily.group_by(
        "fold", "horizon_minutes", "threshold_bps", maintain_order=True
    ).agg(
        pl.len().alias("date_count"),
        pl.col("expected_net_nav_bps").mean().alias("mean_net_nav_bps_per_day"),
        pl.col("expected_net_nav_bps").median().alias("median_net_nav_bps_per_day"),
        pl.col("expected_net_nav_bps").min().alias("minimum_net_nav_bps_per_day"),
        pl.col("expected_net_nav_bps").max().alias("maximum_net_nav_bps_per_day"),
        pl.col("allocated_notional_brl")
        .mean()
        .alias("mean_allocated_notional_brl_per_day"),
        pl.col("allocated_event_count")
        .mean()
        .alias("mean_allocated_event_count_per_day"),
    )
    return daily, summary


def last_observed_close_before(
    close_price: np.ndarray, observed: np.ndarray
) -> np.ndarray:
    close_price = np.asarray(close_price, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    if close_price.shape != observed.shape or close_price.ndim != 3:
        raise ValueError("Last-close inputs must align as [day,minute,name]")
    result = np.full(close_price.shape, np.nan, dtype=np.float64)
    prior = np.full((close_price.shape[0], close_price.shape[2]), np.nan)
    for minute in range(close_price.shape[1]):
        result[:, minute] = prior
        prior = np.where(observed[:, minute], close_price[:, minute], prior)
    return result


def strict_through_fill_minutes(
    *,
    direction: np.ndarray,
    limit_price: np.ndarray,
    day: np.ndarray,
    decision_minute: np.ndarray,
    name: np.ndarray,
    high_price: np.ndarray,
    low_price: np.ndarray,
    observed: np.ndarray,
    wait: int,
) -> np.ndarray:
    direction = np.asarray(direction)
    limit_price = np.asarray(limit_price, dtype=np.float64)
    result = np.full(direction.size, -1, dtype=np.int16)
    unresolved = np.isfinite(limit_price) & (limit_price > 0.0)
    for offset in range(1, wait + 1):
        minute = decision_minute + offset
        inside = unresolved & (minute < high_price.shape[1])
        index = np.nonzero(inside)[0]
        if not index.size:
            continue
        d, m, n = day[index], minute[index], name[index]
        through = observed[d, m, n] & np.where(
            direction[index] > 0,
            low_price[d, m, n] < limit_price[index],
            high_price[d, m, n] > limit_price[index],
        )
        filled = index[through]
        result[filled] = minute[filled].astype(np.int16)
        unresolved[filled] = False
    return result


def _maker_metrics(
    *,
    fold: str,
    dates: Sequence[date],
    events: Mapping[str, np.ndarray],
    unconditional_edge: np.ndarray,
    open_price: np.ndarray,
    close_price: np.ndarray,
    high_price: np.ndarray,
    low_price: np.ndarray,
    observed: np.ndarray,
    last_close: np.ndarray,
    horizon: int,
    wait: int,
    variant: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    tail = events["tail"]
    selected = np.nonzero(tail)[0]
    subset = {name: values[selected] for name, values in events.items()}
    day = subset["day"].astype(np.int64)
    name = subset["name"].astype(np.int64)
    minute = subset["minute"].astype(np.int64)
    direction = subset["direction"].astype(np.float64)
    reference = last_close[day, minute, name]
    if variant == "last_close":
        limit = reference
    elif variant == "improved_half_half_spread":
        limit = reference * (
            1.0 - direction * 0.25 * subset["full_spread_bps"] / 10_000.0
        )
    else:
        raise ValueError(f"Unknown maker limit variant: {variant}")
    quote_valid = (
        np.isfinite(limit) & (limit > 0.0) & (minute + wait < EQUITY_SESSION_MINUTES)
    )
    fill_minute = strict_through_fill_minutes(
        direction=direction,
        limit_price=limit,
        day=day,
        decision_minute=minute,
        name=name,
        high_price=high_price,
        low_price=low_price,
        observed=observed,
        wait=wait,
    )
    filled = quote_valid & (fill_minute >= 0)
    fill_exit = fill_minute.astype(np.int64) + horizon - 1
    fill_inside = filled & (fill_exit < EQUITY_SESSION_MINUTES)
    fill_path = np.zeros(selected.size, dtype=bool)
    index = np.nonzero(fill_inside)[0]
    if index.size:
        fill_path[index] = (
            observed[day[index], fill_minute[index], name[index]]
            & observed[day[index], fill_exit[index], name[index]]
        )
    market_post = np.full(selected.size, np.nan)
    exact_limit = np.full(selected.size, np.nan)
    index = np.nonzero(fill_path)[0]
    if index.size:
        exit_price = close_price[day[index], fill_exit[index], name[index]]
        fill_open = open_price[day[index], fill_minute[index], name[index]]
        market_post[index] = (
            direction[index] * (exit_price / fill_open - 1.0) * 10_000.0
        )
        exact_limit[index] = (
            direction[index] * (exit_price / limit[index] - 1.0) * 10_000.0
        )
    improvement = exact_limit - market_post
    filled_net = exact_limit - FEE_BPS

    fallback_minute = minute + wait + 1
    fallback_exit = fallback_minute + horizon - 1
    fallback_inside = (
        quote_valid
        & ~filled
        & (fallback_minute < EQUITY_SESSION_MINUTES)
        & (fallback_exit < EQUITY_SESSION_MINUTES)
    )
    fallback_path = np.zeros(selected.size, dtype=bool)
    index = np.nonzero(fallback_inside)[0]
    if index.size:
        fallback_path[index] = (
            observed[day[index], fallback_minute[index], name[index]]
            & observed[day[index], fallback_exit[index], name[index]]
        )
    fallback_net = np.full(selected.size, np.nan)
    index = np.nonzero(fallback_path)[0]
    if index.size:
        gross = (
            direction[index]
            * (
                close_price[day[index], fallback_exit[index], name[index]]
                / open_price[day[index], fallback_minute[index], name[index]]
                - 1.0
            )
            * 10_000.0
        )
        fallback_net[index] = gross - subset["taker_cost_measured_bps"][index]
    composite = np.where(fill_path, filled_net, fallback_net)
    composite_valid = np.isfinite(composite)
    unconditional = unconditional_edge[selected]
    frame = _event_frame(
        subset,
        {
            "day": day,
            "quote_valid": quote_valid,
            "filled": filled,
            "fill_offset": fill_minute.astype(np.float64) - minute,
            "fill_path": fill_path,
            "market_post": market_post,
            "unconditional": unconditional,
            "improvement": improvement,
            "filled_net": filled_net,
            "fallback_path": fallback_path,
            "fallback_net": fallback_net,
            "composite_valid": composite_valid,
            "composite_net": composite,
        },
    ).filter(pl.col("quote_valid"))
    grouped = (
        frame.group_by(list(_STATE_COLUMNS), maintain_order=True)
        .agg(
            pl.len().alias("quote_event_count"),
            pl.col("day").n_unique().alias("date_count"),
            pl.col("filled").sum().alias("filled_event_count"),
            pl.col("filled").mean().alias("fill_rate"),
            pl.when(pl.col("filled"))
            .then(pl.col("fill_offset"))
            .otherwise(None)
            .mean()
            .alias("mean_time_to_fill_minutes"),
            pl.col("composite_valid").sum().alias("complete_composite_count"),
            pl.when(pl.col("fill_path"))
            .then(pl.col("market_post"))
            .otherwise(None)
            .mean()
            .alias("mean_market_post_fill_alpha_bps"),
            pl.when(pl.col("unconditional").is_finite())
            .then(pl.col("unconditional"))
            .otherwise(None)
            .mean()
            .alias("mean_unconditional_matched_alpha_bps"),
            pl.when(pl.col("fill_path"))
            .then(pl.col("improvement"))
            .otherwise(None)
            .mean()
            .alias("mean_price_improvement_bps"),
            pl.when(pl.col("fill_path"))
            .then(pl.col("filled_net"))
            .otherwise(None)
            .mean()
            .alias("mean_filled_net_edge_bps"),
            pl.when(pl.col("fallback_path"))
            .then(pl.col("fallback_net"))
            .otherwise(None)
            .mean()
            .alias("mean_fallback_net_edge_bps"),
            pl.when(pl.col("composite_valid"))
            .then(pl.col("composite_net"))
            .otherwise(None)
            .mean()
            .alias("mean_composite_net_edge_bps"),
            pl.when(pl.col("composite_valid"))
            .then(pl.col("composite_net"))
            .otherwise(None)
            .median()
            .alias("median_composite_net_edge_bps"),
        )
        .with_columns(
            (
                pl.col("mean_market_post_fill_alpha_bps")
                - pl.col("mean_unconditional_matched_alpha_bps")
            ).alias("adverse_selection_gap_bps"),
            (pl.col("quote_event_count") / len(dates)).alias("quote_events_per_day"),
            pl.lit(fold).alias("fold"),
            pl.lit(horizon).alias("horizon_minutes"),
            pl.lit(wait).alias("wait_minutes"),
            pl.lit(variant).alias("limit_variant"),
        )
    )
    valid = composite_valid
    means = (
        pl.DataFrame(
            {
                "state": subset["state_cell_id"][valid],
                "composite": composite[valid],
            }
        )
        .group_by("state")
        .agg(pl.col("composite").mean().alias("mean"))
    )
    by_state = dict(means.iter_rows())
    expected = np.asarray(
        [by_state.get(int(value), np.nan) for value in subset["state_cell_id"]]
    )
    daily_rows = _allocate_frontier(
        events=subset,
        expected_net_bps=expected,
        eligible=valid & (expected > 0.0),
        dates=dates,
    )
    daily = pl.DataFrame(
        [
            {
                **row,
                "fold": fold,
                "horizon_minutes": horizon,
                "wait_minutes": wait,
                "limit_variant": variant,
            }
            for row in daily_rows
        ]
    )
    summary = daily.group_by(
        "fold", "horizon_minutes", "wait_minutes", "limit_variant"
    ).agg(
        pl.len().alias("date_count"),
        pl.col("expected_net_nav_bps").mean().alias("mean_net_nav_bps_per_day"),
        pl.col("expected_net_nav_bps").median().alias("median_net_nav_bps_per_day"),
        pl.col("expected_net_nav_bps").min().alias("minimum_net_nav_bps_per_day"),
        pl.col("expected_net_nav_bps").max().alias("maximum_net_nav_bps_per_day"),
        pl.col("allocated_notional_brl")
        .mean()
        .alias("mean_allocated_notional_brl_per_day"),
        pl.col("allocated_event_count")
        .mean()
        .alias("mean_allocated_event_count_per_day"),
    )
    return grouped, daily, summary


def _allocate_priority(
    capacity: np.ndarray,
    priority: np.ndarray,
    name: np.ndarray,
    gross_limit_brl: float,
) -> np.ndarray:
    order = np.lexsort((name, -priority))
    prior = np.concatenate(([0.0], np.cumsum(capacity[order])[:-1]))
    allocated = np.minimum(capacity[order], np.maximum(gross_limit_brl - prior, 0.0))
    result = np.zeros_like(capacity)
    result[order] = allocated
    return result


def positioning_comparison(
    *,
    fold: str,
    dates: Sequence[date],
    events: Mapping[str, np.ndarray],
    edges: Mapping[tuple[int, str], np.ndarray],
    daily_cdi_rate: np.ndarray,
) -> pl.DataFrame:
    rows = []
    day = events["day"].astype(np.int64)
    refresh = events["refresh"].astype(np.int64)
    name = events["name"].astype(np.int64)
    state = events["state_cell_id"].astype(np.int64)
    tail = events["tail"]
    direction = events["direction"]
    for horizon in HORIZONS:
        gross = edges[(horizon, "next_open")]
        valid = np.isfinite(gross)
        means = (
            pl.DataFrame({"state": state[valid], "gross": gross[valid]})
            .group_by("state")
            .agg(pl.col("gross").mean().alias("mean"))
        )
        by_state = dict(means.iter_rows())
        expected_gross = np.asarray(
            [by_state.get(int(value), np.nan) for value in state]
        )
        expected_net = expected_gross - events["taker_cost_measured_bps"]
        market_edge = gross * direction
        daily_values = {
            construction: {
                "trading": [],
                "cdi": [],
                "total": [],
                "drift": [],
                "deployed": [],
            }
            for construction in ("long_short_tails", "long_only_tails_plus_cash")
        }
        refresh_width = int(refresh.max()) + 1
        keys = day * refresh_width + refresh
        order = np.argsort(keys, kind="stable")
        ordered_keys = keys[order]
        starts = np.flatnonzero(
            np.concatenate(([True], ordered_keys[1:] != ordered_keys[:-1]))
        )
        ends = np.concatenate((starts[1:], [order.size]))
        groups = [order[start:end] for start, end in zip(starts, ends, strict=True)]
        groups_by_day: list[list[np.ndarray]] = [[] for _ in dates]
        for group in groups:
            groups_by_day[int(day[group[0]])].append(group)
        for day_index, _ in enumerate(dates):
            accum = {
                key: {"pnl": 0.0, "drift": 0.0, "gross": []} for key in daily_values
            }
            for group in groups_by_day[day_index]:
                index = group[
                    tail[group]
                    & valid[group]
                    & np.isfinite(expected_net[group])
                    & (events["capacity_brl"][group] > 0.0)
                ]
                market = float(np.mean(market_edge[index])) if index.size else 0.0
                long_short_deployed = 0.0
                for side in (-1, 1):
                    side_index = index[direction[index] == side]
                    if not side_index.size:
                        continue
                    allocation = _allocate_priority(
                        events["capacity_brl"][side_index],
                        np.abs(events["blended_rank"][side_index]),
                        name[side_index],
                        NAV_BRL,
                    )
                    accum["long_short_tails"]["pnl"] += float(
                        np.sum(allocation * expected_net[side_index]) / 10_000.0
                    )
                    long_short_deployed += float(allocation.sum())
                accum["long_short_tails"]["gross"].append(long_short_deployed)
                long_index = index[direction[index] > 0]
                if long_index.size:
                    long_capacity = np.minimum(
                        events["capacity_brl"][long_index],
                        NAME_CAP_FRACTION_OF_GROSS * NAV_BRL,
                    )
                    allocation = _allocate_priority(
                        long_capacity,
                        np.abs(events["blended_rank"][long_index]),
                        name[long_index],
                        NAV_BRL,
                    )
                    deployed = float(allocation.sum())
                    accum["long_only_tails_plus_cash"]["pnl"] += float(
                        np.sum(allocation * expected_net[long_index]) / 10_000.0
                    )
                    accum["long_only_tails_plus_cash"]["drift"] += (
                        deployed * market / 10_000.0
                    )
                    accum["long_only_tails_plus_cash"]["gross"].append(deployed)
                else:
                    accum["long_only_tails_plus_cash"]["gross"].append(0.0)
            for construction, values in accum.items():
                deployed = float(np.mean(values["gross"])) if values["gross"] else 0.0
                residual = max(NAV_BRL - MARGIN_FRACTION_OF_GROSS * deployed, 0.0)
                cdi = residual * float(daily_cdi_rate[day_index])
                daily_values[construction]["trading"].append(values["pnl"])
                daily_values[construction]["cdi"].append(cdi)
                daily_values[construction]["total"].append(values["pnl"] + cdi)
                daily_values[construction]["drift"].append(values["drift"])
                daily_values[construction]["deployed"].append(deployed)
        for construction, values in daily_values.items():
            trading = np.asarray(values["trading"])
            cdi = np.asarray(values["cdi"])
            total = np.asarray(values["total"])
            drift = np.asarray(values["drift"])
            beta = 1.0 if construction == "long_only_tails_plus_cash" else 0.0
            rows.append(
                {
                    "fold": fold,
                    "horizon_minutes": horizon,
                    "construction": construction,
                    "date_count": len(dates),
                    "mean_trading_net_nav_bps_per_day": float(
                        np.mean(trading) / NAV_BRL * 10_000.0
                    ),
                    "mean_residual_cdi_nav_bps_per_day": float(
                        np.mean(cdi) / NAV_BRL * 10_000.0
                    ),
                    "mean_total_net_nav_bps_per_day": float(
                        np.mean(total) / NAV_BRL * 10_000.0
                    ),
                    "mean_deployed_gross_brl": float(np.mean(values["deployed"])),
                    "market_beta_approx": beta,
                    "mean_intraday_drift_credit_nav_bps_per_day": float(
                        np.mean(drift) / NAV_BRL * 10_000.0
                    ),
                    "market_beta_variance_nav_bps2_per_day": float(
                        np.var(drift / NAV_BRL * 10_000.0, ddof=1) * beta**2
                    ),
                    "drift_credit_is_component_not_additional_pnl": True,
                }
            )
    return pl.DataFrame(rows)


def taker_decision(summary: pl.DataFrame) -> dict[str, object]:
    threshold = summary.filter(pl.col("threshold_bps") == TAKER_DECISION_THRESHOLD_BPS)
    best = (
        threshold.group_by("fold")
        .agg(
            pl.col("mean_net_nav_bps_per_day")
            .max()
            .alias("maximum_registered_horizon_nav_bps_per_day")
        )
        .sort("fold")
    )
    if best.height != len(FOLDS) or set(best["fold"]) != set(FOLDS):
        raise ValueError("Taker decision lacks one frontier for every fold")
    values = best["maximum_registered_horizon_nav_bps_per_day"].to_numpy()
    if np.all(values < FRONTIER_HURDLE_BPS_PER_DAY):
        outcome = "CLOSED"
        implication = "learned policy actions target maker execution"
    elif int(np.sum(values >= FRONTIER_HURDLE_BPS_PER_DAY)) >= 2:
        outcome = "VIABLE"
        implication = "taker actions remain viable; reward hurdle is all-cash CDI"
    else:
        outcome = "INCONCLUSIVE"
        implication = "no action-set change is authorized"
    return {
        "schema": "EXPERIMENT54_TAKER_DECISION_V1",
        "frozen_rule_verbatim": (
            "if the taker frontier at the 7 bps threshold is below 8 bps/day "
            "of NAV on every fold, taker execution at R$10m is declared CLOSED "
            "for the learned-policy stage; if the frontier clears 8 bps/day on "
            "at least two folds, the taker-action learned policy remains viable "
            "and its reward hurdle is set to all-cash CDI"
        ),
        "registered_horizon_reduction": "maximum across 15/30/60/120 minutes",
        "fold_frontiers": best.to_dicts(),
        "outcome": outcome,
        "implication": implication,
        "official_validation_accessed": False,
        "test_accessed": False,
    }


def _write_table(path: Path, tables: Sequence[pl.DataFrame]) -> dict[str, object]:
    pl.concat(tables, how="diagonal_relaxed").write_parquet(path)
    return _artifact(path)


def run_program(*, output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    design_path = output_dir / "frozen_design.json"
    manifest_path = output_dir / "program_manifest.json"
    design = _read_json(design_path)
    manifest = _read_json(manifest_path)
    if (
        manifest.get("status") != "frozen"
        or manifest.get("frozen_design", {}).get("sha256") != _sha256(design_path)
        or design.get("repository_commit") != repository_commit()
        or design.get("horizons") != list(HORIZONS)
        or design.get("entries") != list(ENTRIES)
        or design.get("thresholds_bps") != list(THRESHOLDS_BPS)
        or design.get("waits") != list(WAITS)
        or design.get("limit_variants") != list(LIMIT_VARIANTS)
    ):
        raise ValueError("Experiment 54 frozen state differs")
    _verify_experiment53(Path(str(design["source_experiment53"]["root"])))
    source52_root = Path(str(design["source_experiment52_root"]))
    _verify_experiment52(source52_root)
    _verify_cache(output_dir / "raw_ohlc")
    store = Path(str(design["store"]["path"])).resolve()
    source52_design = _read_json(source52_root / "frozen_design.json")
    buckets = _read_json(output_dir / "bucket_definitions.json")
    cache_idx = np.asarray(_load_cache_array(source52_root, "date_idx.npy"))
    cache_position = {int(value): index for index, value in enumerate(cache_idx)}
    cache_dates = pl.read_parquet(source52_root / "market_inputs" / "dates.parquet")
    raw_root = output_dir / "raw_ohlc"
    high_all = np.load(raw_root / "high_price.npy", mmap_mode="r", allow_pickle=False)
    low_all = np.load(raw_root / "low_price.npy", mmap_mode="r", allow_pickle=False)
    close_all = np.load(raw_root / "close_price.npy", mmap_mode="r", allow_pickle=False)

    latency_tables = []
    conditional_tables = []
    taker_daily_tables = []
    taker_summary_tables = []
    maker_tables = []
    maker_daily_tables = []
    maker_summary_tables = []
    positioning_tables = []
    exclusion_rows = []
    for fold in FOLDS:
        archive = _fold_archive(design, fold, store)
        positions = np.asarray(
            [cache_position[int(value)] for value in archive.date_idx], dtype=np.int64
        )
        date_table = cache_dates.filter(
            pl.col("date_idx").is_in(archive.date_idx)
        ).sort("date_idx")
        dates = tuple(date_table["trade_date"])
        if len(dates) != positions.size:
            raise ValueError(f"Experiment 54 fold dates differ: {fold}")
        open_price = np.asarray(
            _load_cache_array(source52_root, "open_price.npy")[positions]
        )
        observed = np.asarray(
            _load_cache_array(source52_root, "open_observed.npy")[positions]
        )
        high = np.asarray(high_all[positions])
        low = np.asarray(low_all[positions])
        close = np.asarray(close_all[positions])
        adv = np.asarray(_load_cache_array(source52_root, "adv20_brl.npy")[positions])
        spread = np.asarray(
            _load_cache_array(source52_root, "full_spread.npy")[positions]
        )
        sigma = np.asarray(
            _load_cache_array(source52_root, "sigma_daily.npy")[positions]
        )
        minute_capacity = np.asarray(
            _load_cache_array(source52_root, "minute_notional20_brl.npy")[positions]
        )
        events, exclusions = build_state_events(
            ranks=archive.ranks,
            valid=archive.valid,
            refresh_minutes=archive.refresh_minutes,
            adv20_brl=adv,
            full_spread=spread,
            sigma_daily=sigma,
            minute_notional20_brl=minute_capacity,
            buckets=buckets,
        )
        exclusion_rows.append({"fold": fold, **exclusions})
        open_sums, open_counts = _window_mean_open(open_price, observed)
        edges = {
            (horizon, entry): forward_edge_bps(
                events,
                open_price=open_price,
                close_price=close,
                observed=observed,
                horizon=horizon,
                entry=entry,
                open_sums=open_sums,
                open_counts=open_counts,
            )
            for horizon in HORIZONS
            for entry in ENTRIES
        }
        latency_tables.append(latency_decay_table(fold=fold, dates=dates, edges=edges))
        conditional_tables.append(
            conditional_edge_table(fold=fold, dates=dates, events=events, edges=edges)
        )
        taker_daily, taker_summary = taker_frontier_tables(
            fold=fold, dates=dates, events=events, edges=edges
        )
        taker_daily_tables.append(taker_daily)
        taker_summary_tables.append(taker_summary)
        last_close = last_observed_close_before(close, observed)
        for horizon in HORIZONS:
            for wait in WAITS:
                for variant in LIMIT_VARIANTS:
                    maker, maker_daily, maker_summary = _maker_metrics(
                        fold=fold,
                        dates=dates,
                        events=events,
                        unconditional_edge=edges[(horizon, "next_open")],
                        open_price=open_price,
                        close_price=close,
                        high_price=high,
                        low_price=low,
                        observed=observed,
                        last_close=last_close,
                        horizon=horizon,
                        wait=wait,
                        variant=variant,
                    )
                    maker_tables.append(maker)
                    maker_daily_tables.append(maker_daily)
                    maker_summary_tables.append(maker_summary)
        cdi = load_daily_cdi_rates(
            Path(source52_design["cdi"]["parquet"]["path"]),
            dates,
            str(source52_design["cdi"]["parquet"]["sha256"]),
        )
        positioning_tables.append(
            positioning_comparison(
                fold=fold,
                dates=dates,
                events=events,
                edges=edges,
                daily_cdi_rate=cdi,
            )
        )

    paths = {
        "event_exclusions": output_dir / "event_exclusions.parquet",
        "latency_decay": output_dir / "latency_decay.parquet",
        "taker_conditional_edges": output_dir / "taker_conditional_edges.parquet",
        "taker_frontier_daily": output_dir / "taker_frontier_daily.parquet",
        "taker_frontier": output_dir / "taker_frontier.parquet",
        "maker_conditional": output_dir / "maker_conditional.parquet",
        "maker_frontier_daily": output_dir / "maker_frontier_daily.parquet",
        "maker_frontier": output_dir / "maker_frontier.parquet",
        "positioning_comparison": output_dir / "positioning_comparison.parquet",
    }
    pl.DataFrame(exclusion_rows).write_parquet(paths["event_exclusions"])
    _write_table(paths["latency_decay"], latency_tables)
    _write_table(paths["taker_conditional_edges"], conditional_tables)
    _write_table(paths["taker_frontier_daily"], taker_daily_tables)
    _write_table(paths["taker_frontier"], taker_summary_tables)
    _write_table(paths["maker_conditional"], maker_tables)
    _write_table(paths["maker_frontier_daily"], maker_daily_tables)
    _write_table(paths["maker_frontier"], maker_summary_tables)
    _write_table(paths["positioning_comparison"], positioning_tables)
    taker_summary = pl.read_parquet(paths["taker_frontier"])
    decision = taker_decision(taker_summary)
    decision_path = output_dir / "taker_decision.json"
    _atomic_json(decision_path, decision)
    statements_path = output_dir / "frontier_definitions.json"
    _atomic_json(
        statements_path,
        {
            "schema": "EXPERIMENT54_FRONTIER_DEFINITIONS_V1",
            "taker": (
                "same-fold conditional-cell mean next-open gross edge above a "
                "fixed threshold; measured name cost; 10% causal minute capacity; "
                "5% of gross name cap; gross <= 2; no neutrality"
            ),
            "maker": (
                "same allocator using positive same-fold conditional-cell mean "
                "complete composite net edge; strict-through fills and taker fallback"
            ),
            "maker_decision_rule": None,
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    maker_summary = pl.read_parquet(paths["maker_frontier"])
    best_maker = maker_summary.sort("mean_net_nav_bps_per_day", descending=True).row(
        0, named=True
    )
    result_path = output_dir / "experiment54_result.json"
    _atomic_json(
        result_path,
        {
            "schema": "EXPERIMENT54_RESULT_V1",
            "taker_decision": decision["outcome"],
            "taker_implication": decision["implication"],
            "best_maker_frontier": best_maker,
            "maker_interpretation": "user decision required; no automatic rule",
            "simulator_run": False,
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    completed = {
        **manifest,
        "status": "completed",
        "completed_at": _now(),
        "operational_commit": os.environ.get(
            "EXPERIMENT54_OPERATIONAL_COMMIT", repository_commit()
        ),
        "artifacts": {name: _artifact(path) for name, path in paths.items()},
        "taker_decision": _artifact(decision_path),
        "frontier_definitions": _artifact(statements_path),
        "result": _artifact(result_path),
        "simulator_run": False,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(manifest_path, completed)
    audit_program(output_dir)
    return result_path


def audit_program(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    manifest = _read_json(output_dir / "program_manifest.json")
    decision = _read_json(output_dir / "taker_decision.json")
    result = _read_json(output_dir / "experiment54_result.json")
    latency = pl.read_parquet(output_dir / "latency_decay.parquet")
    taker = pl.read_parquet(output_dir / "taker_frontier.parquet")
    maker = pl.read_parquet(output_dir / "maker_frontier.parquet")
    positioning = pl.read_parquet(output_dir / "positioning_comparison.parquet")
    exclusions = pl.read_parquet(output_dir / "event_exclusions.parquet")
    if (
        manifest.get("status") != "completed"
        or manifest.get("simulator_run") is not False
        or latency.height != len(FOLDS) * len(HORIZONS) * len(ENTRIES)
        or taker.height != len(FOLDS) * len(HORIZONS) * len(THRESHOLDS_BPS)
        or maker.height != len(FOLDS) * len(HORIZONS) * len(WAITS) * len(LIMIT_VARIANTS)
        or positioning.height != len(FOLDS) * len(HORIZONS) * 2
        or exclusions.height != len(FOLDS)
        or set(latency["fold"]) != set(FOLDS)
        or set(taker["fold"]) != set(FOLDS)
        or set(maker["fold"]) != set(FOLDS)
        or decision.get("outcome") not in {"CLOSED", "VIABLE", "INCONCLUSIVE"}
        or result.get("taker_decision") != decision.get("outcome")
        or any(
            payload.get("official_validation_accessed") is not False
            or payload.get("test_accessed") is not False
            for payload in (manifest, decision, result)
        )
    ):
        raise ValueError("Experiment 54 completion audit failed")
    artifacts = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "final_audit.json":
            record = _artifact(path)
            record["path"] = str(path.relative_to(output_dir)).replace("\\", "/")
            artifacts.append(record)
    audit_path = output_dir / "final_audit.json"
    _atomic_json(
        audit_path,
        {
            "schema": "EXPERIMENT54_FINAL_AUDIT_V1",
            "status": "passed",
            "created_at": _now(),
            "file_count": len(artifacts),
            "total_bytes": sum(int(row["bytes"]) for row in artifacts),
            "latency_rows": latency.height,
            "taker_frontier_rows": taker.height,
            "maker_frontier_rows": maker.height,
            "positioning_rows": positioning.height,
            "taker_decision": decision["outcome"],
            "artifacts": artifacts,
            "simulator_run": False,
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    return audit_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 54 edge/maker analysis")
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--experiment53-root", type=Path, required=True)
    freeze.add_argument("--preregistration", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--output-dir", type=Path, required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "freeze":
        path = freeze_program(
            experiment53_root=args.experiment53_root,
            preregistration=args.preregistration,
            output_dir=args.output_dir,
        )
    elif args.command == "run":
        path = run_program(output_dir=args.output_dir)
    else:
        path = audit_program(args.output_dir)
    print(path)


if __name__ == "__main__":
    main()
