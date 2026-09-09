from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from brazil_rv.modeling.contract import workspace_path
from brazil_rv.preprocessing.io import (
    SOURCE_COLUMNS,
    dense_grid,
    validate_session_bars,
)

from .artifacts import inventory, sha256_file, write_json_atomic
from .decision_clock import SessionDefinition, load_session_schedule
from .intraday_features import NATIVE_FAST_FEATURES
from .store import STORE_SCHEMA


AUDIT_SCHEMA = "BRAZIL_RV_V2_NATIVE_FAST_RAW_AUDIT_V1"
_ACTION_SOURCE = "inferred_cotahist_dismes_v1"
_SCHEDULE_SOURCE = "reconstructed_v1"


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _verified_path(root: Path, record: Mapping[str, object]) -> Path:
    path = root / str(record["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(record["bytes"])
        or sha256_file(path) != record["sha256"]
    ):
        raise ValueError(f"immutable store artifact hash mismatch: {path}")
    return path


def _source_tiers(manifest: Mapping[str, object]) -> dict[str, str]:
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("store lacks metadata")
    result = {
        "action_terms_source": str(metadata.get("action_terms_source")),
        "schedule_source": str(metadata.get("schedule_source")),
    }
    if result != {
        "action_terms_source": _ACTION_SOURCE,
        "schedule_source": _SCHEDULE_SOURCE,
    }:
        raise ValueError("native-fast audit requires the development-grade source tier")
    return result


def _select_panel(
    dates: NDArray[np.datetime64],
    patch_mask: NDArray[np.bool_],
    feature_valid: NDArray[np.bool_],
    *,
    name_count: int,
    session_count: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    first = int(np.searchsorted(dates, np.datetime64("2021-08-16")))
    last = int(np.searchsorted(dates, np.datetime64("2025-01-01")))
    history = 20
    for start in range(first + history, last - session_count + 1):
        raw_rows = np.arange(start - history, start + session_count, dtype=np.int64)
        audited_rows = raw_rows[history:]
        supported = patch_mask[raw_rows, :, 0].all(axis=0)
        candidates = np.flatnonzero(supported)
        if candidates.size < name_count:
            continue
        candidate_valid = feature_valid[audited_rows][:, candidates]
        coverage = candidate_valid.sum(axis=(0, 2, 3))
        order = np.lexsort((candidates, -coverage))
        chosen = np.sort(candidates[order[:name_count]])
        channel_support = feature_valid[audited_rows][:, chosen].sum(axis=(0, 1, 2))
        if np.all(channel_support > 0):
            return audited_rows, chosen.astype(np.int64)
    raise ValueError(
        "no development window supports the fixed 20-name/20-session audit"
    )


def _clock_minutes(session: SessionDefinition) -> tuple[int, int, int]:
    opening = session.continuous_open.hour * 60 + session.continuous_open.minute
    decision = session.decision_time.hour * 60 + session.decision_time.minute
    close = session.continuous_close.hour * 60 + session.continuous_close.minute
    prefix = decision - opening
    continuous = close - opening
    if prefix <= 0 or continuous <= prefix:
        raise ValueError(f"invalid session clocks on {session.trade_date}")
    return opening, prefix, continuous


def _grid_raw_names(
    assignments: pl.DataFrame,
    mapping: pl.DataFrame,
    selected_fast: NDArray[np.int64],
    raw_sessions: Sequence[SessionDefinition],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.bool_],
    NDArray[np.bool_],
    list[dict[str, object]],
]:
    selected = mapping.filter(pl.col("fast_index").is_in(selected_fast.tolist())).join(
        assignments, on=["security_id", "isin"], how="left", validate="1:1"
    )
    if selected.height != selected_fast.size or selected["source_file"].null_count():
        raise ValueError("selected native-fast identities do not map uniquely to M1")
    by_fast = {int(row["fast_index"]): row for row in selected.iter_rows(named=True)}
    dates = tuple(item.trade_date for item in raw_sessions)
    max_minutes = max(_clock_minutes(item)[2] for item in raw_sessions)
    market = np.zeros(
        (len(dates), len(selected_fast), max_minutes, 5), dtype=np.float64
    )
    observed = np.zeros(market.shape[:3], dtype=np.bool_)
    supported = np.zeros(market.shape[:2], dtype=np.bool_)
    sources: list[dict[str, object]] = []
    cache: dict[Path, pl.DataFrame] = {}
    for local_name, fast_index in enumerate(selected_fast.tolist()):
        row = by_fast[fast_index]
        recorded = Path(str(row["source_file"]))
        resolved = recorded if recorded.is_file() else workspace_path(recorded)
        resolved = resolved.resolve(strict=True)
        if resolved not in cache:
            cache[resolved] = (
                pl.scan_parquet(resolved)
                .filter(pl.col("ts_exchange").dt.date().is_between(dates[0], dates[-1]))
                .select(SOURCE_COLUMNS)
                .collect()
            )
        first = row.get("first_overlap_date")
        last = row.get("last_overlap_date")
        if isinstance(first, str):
            first = date.fromisoformat(first)
        if isinstance(last, str):
            last = date.fromisoformat(last)
        allowed = frozenset(
            value
            for value in dates
            if (first is None or value >= first) and (last is None or value <= last)
        )
        schedule_index = pl.DataFrame(
            {
                "trade_date": pl.Series(dates, dtype=pl.Date),
                "date_idx": pl.Series(range(len(dates)), dtype=pl.Int32),
                "continuous_open_minute": pl.Series(
                    [_clock_minutes(item)[0] for item in raw_sessions], dtype=pl.Int16
                ),
                "continuous_minute_count": pl.Series(
                    [_clock_minutes(item)[2] for item in raw_sessions], dtype=pl.Int16
                ),
            }
        )
        source = cache[resolved]
        bars = (
            source.with_columns(
                pl.col("ts_exchange").dt.date().alias("trade_date"),
                (
                    pl.col("ts_exchange").dt.hour().cast(pl.Int16) * 60
                    + pl.col("ts_exchange").dt.minute().cast(pl.Int16)
                ).alias("clock_minute"),
            )
            .filter(pl.col("trade_date").is_in(tuple(allowed)))
            .join(schedule_index, on="trade_date", how="inner")
            .with_columns(
                (pl.col("clock_minute") - pl.col("continuous_open_minute"))
                .cast(pl.Int16)
                .alias("minute_idx")
            )
            .filter(
                pl.col("minute_idx") >= 0,
                pl.col("minute_idx") < pl.col("continuous_minute_count"),
            )
            .sort("ts_exchange")
        )
        validate_session_bars(bars, resolved)
        grid, seen = dense_grid(bars, len(dates), max_minutes)
        market[:, local_name] = grid
        observed[:, local_name] = seen
        supported[:, local_name] = seen.any(axis=1)
        sources.append(
            {
                "fast_index": fast_index,
                "security_id": str(row["security_id"]),
                "isin": str(row["isin"]),
                "recorded_path": str(recorded),
                "resolved_path": str(resolved),
                "sha256": sha256_file(resolved),
                "selected_raw_session_count": int(seen.any(axis=1).sum()),
            }
        )
    return market, observed, supported, sources


def _hand_compute(
    market: NDArray[np.float64],
    observed: NDArray[np.bool_],
    supported: NDArray[np.bool_],
    sigma: NDArray[np.float64],
    sessions: Sequence[SessionDefinition],
    *,
    max_patches: int,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.bool_],
    NDArray[np.bool_],
    NDArray[np.float32],
    NDArray[np.bool_],
]:
    day_count, name_count = supported.shape
    values = np.zeros((day_count, name_count, max_patches, 7), dtype=np.float32)
    valid = np.zeros_like(values, dtype=np.bool_)
    patch_mask = np.zeros(values.shape[:3], dtype=np.bool_)
    age = np.zeros(values.shape[:3], dtype=np.float32)
    age_valid = np.zeros(values.shape[:3], dtype=np.bool_)
    activity_history: list[
        dict[int, tuple[NDArray[np.float64], NDArray[np.bool_]]]
    ] = []
    for day, session in enumerate(sessions):
        opening, prefix, continuous = _clock_minutes(session)
        patch_count = prefix // 5
        last_price = np.full(name_count, -1, dtype=np.int64)
        previous = np.zeros(name_count, dtype=np.float64)
        previous_valid = np.zeros(name_count, dtype=np.bool_)
        current_activity: dict[int, tuple[NDArray[np.float64], NDArray[np.bool_]]] = {}
        risk_valid = np.isfinite(sigma[day]) & (sigma[day] > 1e-8)
        scale = sigma[day] * np.sqrt(5.0 / continuous)
        for patch in range(patch_count):
            start = patch * 5
            stop = start + 5
            patch_mask[day, :, patch] = supported[day]
            high = market[day, :, start:stop, 1]
            low = market[day, :, start:stop, 2]
            close = market[day, :, start:stop, 3]
            seen = observed[day, :, start:stop]
            price_valid = (
                seen
                & np.isfinite(high)
                & np.isfinite(low)
                & np.isfinite(close)
                & (low > 0.0)
                & (high >= low)
                & (close >= low)
                & (close <= high)
            )
            complete = supported[day] & price_valid.all(axis=1)
            block_high = high.max(axis=1)
            block_low = low.min(axis=1)
            endpoint = close[:, -1]
            mask = complete & risk_valid
            valid[day, mask, patch, 1] = True
            values[day, mask, patch, 1] = np.asarray(
                np.log(block_high[mask] / block_low[mask]) / scale[mask],
                dtype=np.float32,
            )
            valid[day, complete, patch, 2] = True
            ranged = complete & (block_high > block_low)
            values[day, ranged, patch, 2] = np.asarray(
                2.0
                * (endpoint[ranged] - block_low[ranged])
                / (block_high[ranged] - block_low[ranged])
                - 1.0,
                dtype=np.float32,
            )
            endpoint_valid = (
                supported[day] & seen[:, -1] & np.isfinite(endpoint) & (endpoint > 0.0)
            )
            if patch:
                mask = endpoint_valid & previous_valid & risk_valid
                valid[day, mask, patch, 0] = True
                values[day, mask, patch, 0] = np.asarray(
                    np.log(endpoint[mask] / previous[mask]) / scale[mask],
                    dtype=np.float32,
                )
            previous = endpoint.copy()
            previous_valid = endpoint_valid
            valid[day, supported[day], patch, 4] = True
            values[day, supported[day], patch, 4] = np.asarray(
                seen[supported[day]].sum(axis=1) / 5.0, dtype=np.float32
            )
            valid[day, supported[day], patch, 5] = True
            values[day, supported[day], patch, 5] = np.float32(stop / prefix)
            close_seen = seen & np.isfinite(close) & (close > 0.0)
            for minute in range(5):
                last_price[supported[day] & close_seen[:, minute]] = start + minute
            mask = supported[day] & (last_price >= 0)
            raw_age = stop - 1 - last_price
            age_valid[day, mask, patch] = True
            age[day, mask, patch] = raw_age[mask].astype(np.float32)
            valid[day, mask, patch, 6] = True
            values[day, mask, patch, 6] = np.asarray(
                np.clip(raw_age[mask] / continuous, 0.0, 1.0), dtype=np.float32
            )
            volume = market[day, :, start:stop, 4]
            activity_valid = seen & np.isfinite(volume) & (volume >= 0.0)
            block_valid = supported[day] & activity_valid.all(axis=1)
            block_volume = np.where(activity_valid, volume, 0.0).sum(axis=1)
            clock = opening + start
            if len(activity_history) == 20:
                history_values = np.zeros((20, name_count), dtype=np.float64)
                history_valid = np.zeros((20, name_count), dtype=np.bool_)
                for index, prior_day in enumerate(activity_history):
                    prior = prior_day.get(clock)
                    if prior is not None:
                        history_values[index], history_valid[index] = prior
                candidates = block_valid & (history_valid.sum(axis=0) >= 16)
                for name in np.flatnonzero(candidates):
                    baseline = np.median(history_values[history_valid[:, name], name])
                    if np.isfinite(baseline) and baseline > 0.0:
                        valid[day, name, patch, 3] = True
                        values[day, name, patch, 3] = np.float32(
                            np.clip(
                                np.log1p(block_volume[name] / baseline) - np.log(2),
                                -5,
                                5,
                            )
                        )
            current_activity[clock] = (block_volume.copy(), block_valid.copy())
        activity_history.append(current_activity)
        if len(activity_history) > 20:
            activity_history.pop(0)
    return values, valid, patch_mask, age, age_valid


def audit_native_fast(
    *,
    store_root: Path,
    assignments_path: Path,
    schedule_path: Path,
    output_root: Path,
    name_count: int = 20,
    session_count: int = 20,
) -> str:
    if name_count != 20 or session_count != 20:
        raise ValueError("canonical real-data audit is fixed at 20 names x 20 sessions")
    store = store_root.resolve(strict=True)
    output = output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest_path = store / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != STORE_SCHEMA:
        raise ValueError("native-fast audit requires the current store schema")
    source_tiers = _source_tiers(manifest)
    arrays = manifest.get("arrays")
    tables = manifest.get("tables")
    if not isinstance(arrays, Mapping) or not isinstance(tables, Mapping):
        raise ValueError("store manifest lacks arrays or tables")
    loaded: dict[str, NDArray[np.generic]] = {}
    for name in (
        "fast_patch_values",
        "fast_patch_valid",
        "fast_patch_mask",
        "fast_last_price_age_minutes",
        "fast_last_price_age_valid",
        "target_scale_sigma",
    ):
        record = arrays.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"store lacks {name}")
        loaded[name] = np.load(
            _verified_path(store, record), mmap_mode="r", allow_pickle=False
        )
    dates_record = manifest.get("indices", {}).get("date_index.npy")
    if not isinstance(dates_record, Mapping):
        raise ValueError("store lacks date axis")
    dates_path = store / "date_index.npy"
    if (
        dates_path.stat().st_size != int(dates_record["bytes"])
        or sha256_file(dates_path) != dates_record["sha256"]
    ):
        raise ValueError("store date axis hash mismatch")
    dates = np.load(dates_path, allow_pickle=False)
    mapping_record = tables.get("native_fast_security_mapping")
    if not isinstance(mapping_record, Mapping):
        raise ValueError("store lacks native-fast security mapping")
    mapping_path = _verified_path(store, mapping_record)
    mapping = pl.read_parquet(mapping_path)
    assignments_source = assignments_path.resolve(strict=True)
    assignments = pl.read_parquet(assignments_source)
    schedule_source = schedule_path.resolve(strict=True)
    sessions = load_session_schedule(schedule_source)
    schedule_by_date = {item.trade_date: item for item in sessions}
    audited_rows, selected_fast = _select_panel(
        dates,
        np.asarray(loaded["fast_patch_mask"]),
        np.asarray(loaded["fast_patch_valid"]),
        name_count=name_count,
        session_count=session_count,
    )
    raw_rows = np.arange(audited_rows[0] - 20, audited_rows[-1] + 1, dtype=np.int64)
    raw_dates = dates[raw_rows].astype("datetime64[D]").astype(object).tolist()
    raw_sessions = tuple(schedule_by_date[value] for value in raw_dates)
    market, observed, supported, raw_sources = _grid_raw_names(
        assignments, mapping, selected_fast, raw_sessions
    )
    store_name_indices = (
        mapping.filter(pl.col("fast_index").is_in(selected_fast.tolist()))
        .sort("fast_index")
        .get_column("store_name_index")
        .to_numpy()
        .astype(np.int64)
    )
    sigma = np.asarray(loaded["target_scale_sigma"])[raw_rows][:, store_name_indices]
    expected = _hand_compute(
        market,
        observed,
        supported,
        np.asarray(sigma, dtype=np.float64),
        raw_sessions,
        max_patches=int(loaded["fast_patch_values"].shape[2]),
    )
    local = np.arange(20, 20 + session_count, dtype=np.int64)
    actual = (
        np.asarray(loaded["fast_patch_values"])[audited_rows][:, selected_fast],
        np.asarray(loaded["fast_patch_valid"])[audited_rows][:, selected_fast],
        np.asarray(loaded["fast_patch_mask"])[audited_rows][:, selected_fast],
        np.asarray(loaded["fast_last_price_age_minutes"])[audited_rows][
            :, selected_fast
        ],
        np.asarray(loaded["fast_last_price_age_valid"])[audited_rows][:, selected_fast],
    )
    expected_audit = tuple(value[local] for value in expected)
    if not np.array_equal(actual[1], expected_audit[1]):
        raise ValueError("independent native-fast feature-valid masks differ")
    if not np.array_equal(actual[2], expected_audit[2]):
        raise ValueError("independent native-fast patch masks differ")
    if not np.array_equal(actual[4], expected_audit[4]):
        raise ValueError("independent native-fast age-valid masks differ")
    feature_mask = actual[1]
    age_mask = actual[4]
    feature_error = np.abs(actual[0].astype(np.float64) - expected_audit[0])
    age_error = np.abs(actual[3].astype(np.float64) - expected_audit[3])
    max_feature_error = float(feature_error[feature_mask].max(initial=0.0))
    max_age_error = float(age_error[age_mask].max(initial=0.0))
    if max_feature_error > 2e-6 or max_age_error != 0.0:
        raise ValueError("independent native-fast numeric values differ")
    output.mkdir(parents=True, exist_ok=False)
    result_path = output / "native_fast_audit.json"
    result = {
        "schema": AUDIT_SCHEMA,
        "status": "passed",
        "research_claim": False,
        "official_validation_accessed": False,
        "test_accessed": False,
        **source_tiers,
        "store": {"root": str(store), "manifest_sha256": sha256_file(manifest_path)},
        "inputs": {
            "assignments": {
                "path": str(assignments_source),
                "sha256": sha256_file(assignments_source),
            },
            "schedule": {
                "path": str(schedule_source),
                "sha256": sha256_file(schedule_source),
            },
            "store_mapping": {
                "path": str(mapping_path),
                "sha256": sha256_file(mapping_path),
            },
            "raw_sources": raw_sources,
        },
        "selection": {
            "name_count": name_count,
            "session_count": session_count,
            "history_session_count": 20,
            "fast_indices": selected_fast.tolist(),
            "store_name_indices": store_name_indices.tolist(),
            "dates": [str(dates[index]) for index in audited_rows],
        },
        "comparison": {
            "feature_names": list(NATIVE_FAST_FEATURES),
            "feature_valid_counts": actual[1].sum(axis=(0, 1, 2)).tolist(),
            "feature_mask_exact": True,
            "patch_mask_exact": True,
            "age_valid_exact": True,
            "max_abs_feature_error": max_feature_error,
            "max_abs_age_minutes_error": max_age_error,
            "absolute_tolerance": 2e-6,
            "independent_formula_implementation": True,
        },
    }
    digest = write_json_atomic(result_path, result)
    rows = inventory(
        output, exclude={"artifact_inventory.json", "artifact_inventory.json.sha256"}
    )
    write_json_atomic(
        output / "artifact_inventory.json",
        {
            "schema": "BRAZIL_RV_V2_NATIVE_FAST_AUDIT_INVENTORY_V1",
            "status": "passed",
            "official_validation_accessed": False,
            "test_accessed": False,
            "files": rows,
        },
    )
    return digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit native fast patches from raw M1"
    )
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--m1-assignments", type=Path, required=True)
    parser.add_argument("--session-schedule", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    print(
        audit_native_fast(
            store_root=arguments.store,
            assignments_path=arguments.m1_assignments,
            schedule_path=arguments.session_schedule,
            output_root=arguments.output_root,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
