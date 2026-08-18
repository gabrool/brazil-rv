from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import polars as pl
from ..modeling.contract import (
    EXPECTED_SPLIT_DATE_COUNTS,
    EXPECTED_SPLIT_SAMPLE_COUNTS,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
    workspace_path,
)

from .contract import (
    CANONICAL_OUTPUT_POINTER,
    DECISION_GLOBAL_INDICES,
    EXPOSURE_BETA_CONTEXT_SYMBOLS,
    FIXED_RATE_CONTEXT_SYMBOLS,
    GLOBAL_CONTEXT_SYMBOLS,
    GLOBAL_SLOW_CHANNELS,
    GLOBAL_UNUSED_SLOW_CHANNEL_INDICES,
    LOCAL_CONTEXT_SYMBOLS,
    LIQUIDITY_SELECTED_RATE_CONTEXT_SYMBOL,
    LIQUIDITY_SELECTED_RATE_ZERO_SLOW_CHANNEL_INDICES,
    DECISION_CONTEXT_INDICES,
    DECISION_EQUITY_INDICES,
    DYNAMIC_CHANNELS,
    EXPECTED_DATE_COUNT,
    EXPECTED_ELIGIBLE_DATE_COUNT,
    EXPECTED_ELIGIBLE_DATES_WITH_UNAVAILABLE_LOCAL_CONTEXT,
    EXPECTED_FIRST_ELIGIBLE_DATE,
    EXPECTED_LAST_ELIGIBLE_DATE,
    EXPECTED_SAMPLE_COUNT,
    LOCAL_CONTEXT_AVAILABILITY_RULE,
    SAMPLE_ELIGIBILITY_RULE,
    HORIZONS,
    MIN_ACTIVE_EQUITIES,
    SLOW_CHANNELS,
    output_array_specs,
)
from .global_source import load_global_symbol, validate_normalized_bars
from .intraday_normalization import validate_profile_artifact
from .transforms import centered_midranks

AUDIT_BASE = CANONICAL_OUTPUT_POINTER.parent.parent / "feature_audits"
EQUITY_VISIBLE_MINUTES = max(DECISION_EQUITY_INDICES)
CONTEXT_VISIBLE_MINUTES = max(DECISION_CONTEXT_INDICES)
GLOBAL_VISIBLE_MINUTES = max(DECISION_GLOBAL_INDICES)
DATE_CHUNK = 8
TARGET_MEAN_TOLERANCE = 2e-6


def local_context_readiness_rows(
    context_ready: np.ndarray,
    trade_dates: list[object] | tuple[object, ...],
    eligible_dates: np.ndarray,
) -> list[dict[str, object]]:
    """Summarize date-level local readiness without treating it as eligibility."""
    if context_ready.shape != (len(trade_dates), len(LOCAL_CONTEXT_SYMBOLS)):
        raise ValueError("Local context readiness does not match the date/symbol axes")
    eligible_dates = np.asarray(eligible_dates, dtype=np.int64)
    rows: list[dict[str, object]] = []
    for slot, symbol in enumerate(LOCAL_CONTEXT_SYMBOLS):
        ready = np.asarray(context_ready[:, slot], dtype=bool)
        ready_indices = np.flatnonzero(ready)
        eligible_ready = ready[eligible_dates]
        row: dict[str, object] = {
            "context_slot": slot,
            "symbol": symbol,
            "research_ready_date_count": int(ready.sum()),
            "research_date_count": len(trade_dates),
            "research_readiness_fraction": float(ready.mean()),
            "eligible_ready_date_count": int(eligible_ready.sum()),
            "eligible_date_count": int(eligible_dates.size),
            "eligible_readiness_fraction": float(eligible_ready.mean()),
            "first_ready_date": (
                str(trade_dates[int(ready_indices[0])]) if ready_indices.size else None
            ),
            "last_ready_date": (
                str(trade_dates[int(ready_indices[-1])]) if ready_indices.size else None
            ),
        }
        for split, start, end in (
            ("train", TRAIN_START, TRAIN_END),
            ("validation", VALIDATION_START, VALIDATION_END),
            ("test", TEST_START, TEST_END),
        ):
            indices = np.asarray(
                [
                    index
                    for index in eligible_dates
                    if start <= trade_dates[int(index)] <= end
                ],
                dtype=np.int64,
            )
            count = int(ready[indices].sum())
            row[f"{split}_ready_date_count"] = count
            row[f"{split}_date_count"] = int(indices.size)
            row[f"{split}_readiness_fraction"] = (
                count / int(indices.size) if indices.size else 0.0
            )
        rows.append(row)
    return rows


DYNAMIC_BOUNDS: tuple[tuple[float, float], ...] = (
    (-10.0, 10.0),
    (-10.0, 10.0),
    (-10.0, 10.0),
    (-10.0, 10.0),
    (-6.0, 6.0),
    (0.0, 1.0),
    (-10.0, 10.0),
    (-10.0, 10.0),
    (-10.0, 10.0),
    (-10.0, 10.0),
    (-4.0, 4.0),
    (-4.0, 4.0),
    (-4.0, 4.0),
    (-6.0, 6.0),
    (-1.0, 1.0),
    (0.0, 1.0),
    (-10.0, 10.0),
    (-10.0, 10.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (0.0, 10.0),
    (0.0, 10.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
)
SLOW_BOUNDS: tuple[tuple[float, float] | None, ...] = (
    (-4.0, 4.0),
    (-10.0, 10.0),
    (-10.0, 10.0),
    (-10.0, 10.0),
    (-10.0, 10.0),
    (-4.0, 4.0),
    (-6.0, 6.0),
    (-10.0, 10.0),
    (-10.0, 10.0),
    (-4.0, 4.0),
    (-4.0, 4.0),
    (0.0, 4.0),
    None,
    None,
    (-6.0, 6.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (-5.0, 5.0),
    (-5.0, 5.0),
    (-5.0, 5.0),
    (-5.0, 5.0),
    (-5.0, 5.0),
    (-5.0, 5.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (-1.0, 3.0),
    (0.0, 1.0),
)


@dataclass
class StreamingStats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = np.inf
    maximum: float = -np.inf
    zero_count: int = 0

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).ravel()
        if values.size == 0:
            return
        if not np.isfinite(values).all():
            raise ValueError("Non-finite value encountered during statistical audit")
        self.count += int(values.size)
        self.total += float(values.sum(dtype=np.float64))
        self.total_sq += float(np.square(values).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        self.zero_count += int(np.count_nonzero(values == 0.0))

    def row(self, scope: str, feature: str) -> dict[str, object]:
        mean = self.total / self.count if self.count else None
        variance = (
            max(self.total_sq / self.count - mean * mean, 0.0)
            if mean is not None
            else None
        )
        return {
            "scope": scope,
            "feature": feature,
            "count": self.count,
            "mean": mean,
            "std": None if variance is None else float(np.sqrt(variance)),
            "min": None if self.count == 0 else self.minimum,
            "max": None if self.count == 0 else self.maximum,
            "zero_rate": None if self.count == 0 else self.zero_count / self.count,
        }


def _opening_feature_row(feature: str, stats: StreamingStats) -> dict[str, object]:
    row = stats.row("equity_active_opening", feature)
    return {
        "feature": feature,
        "valid_observation_count": stats.count,
        "nonzero_count": stats.count - stats.zero_count,
        "nonzero_rate": (
            None if stats.count == 0 else (stats.count - stats.zero_count) / stats.count
        ),
        "mean": row["mean"],
        "std": row["std"],
        "min": row["min"],
        "max": row["max"],
    }


def opening_feature_family_stats(
    arrays: dict[str, np.ndarray], eligible_dates: np.ndarray
) -> dict[str, object]:
    """Audit equity opening features and fail on an expected but dead gap rank."""
    eligible_dates = np.asarray(eligible_dates, dtype=np.int64)
    stats = {
        "return_since_open_normalized": StreamingStats(),
        "overnight_gap_normalized": StreamingStats(),
        "previous_open_to_close_return_normalized": StreamingStats(),
        "overnight_gap_cross_section_rank": StreamingStats(),
    }
    early_open_equity_days = 0
    computed_dates = 0
    below_minimum_dates = 0
    first_cutoff = DECISION_EQUITY_INDICES[0]

    for start in range(0, eligible_dates.size, DATE_CHUNK):
        indices = eligible_dates[start : start + DATE_CHUNK]
        dynamic = np.asarray(
            arrays["equity_features.npy"][indices, :, :EQUITY_VISIBLE_MINUTES],
            dtype=np.float32,
        )
        slow = np.asarray(arrays["equity_slow.npy"][indices], dtype=np.float32)
        active = np.asarray(
            arrays["equity_membership.npy"][indices]
            & arrays["equity_data_ready.npy"][indices],
            dtype=bool,
        )
        observed = dynamic[..., 5] > 0.5
        dynamic_valid = observed & active[:, :, None]
        stats["return_since_open_normalized"].update(dynamic[..., 6][dynamic_valid])

        early_open_valid = active & observed[:, :, :first_cutoff].any(axis=2)
        early_open_equity_days += int(early_open_valid.sum())
        # Readiness is fixed from completed prior sessions before the current-day
        # update. Dynamic channel 5 is absent during warmup and cannot reconstruct
        # those observations, so active early opens are the exact valid gap set.
        gap_valid = early_open_valid
        stats["overnight_gap_normalized"].update(slow[..., 1][gap_valid])
        stats["previous_open_to_close_return_normalized"].update(slow[..., 3][active])

        valid_population = gap_valid.sum(axis=1)
        rank_expected = valid_population >= MIN_ACTIVE_EQUITIES
        computed_dates += int(rank_expected.sum())
        below_minimum_dates += int((~rank_expected).sum())
        rank_valid = gap_valid & rank_expected[:, None]
        rank_values = slow[..., 17]
        stats["overnight_gap_cross_section_rank"].update(rank_values[rank_valid])
        if np.any(rank_values[~rank_valid] != 0.0):
            raise ValueError(
                "overnight_gap_cross_section_rank must be neutral outside valid "
                "early-open cross-sections"
            )
        for local_date in np.flatnonzero(rank_expected):
            values = rank_values[local_date, rank_valid[local_date]]
            expected = centered_midranks(
                slow[local_date, rank_valid[local_date], 1].astype(np.float64)
            )
            if not np.allclose(values, expected, atol=TARGET_MEAN_TOLERANCE, rtol=0.0):
                raise ValueError(
                    "overnight_gap_cross_section_rank does not match centered "
                    "midranks of overnight_gap_normalized"
                )
            if abs(float(values.mean(dtype=np.float64))) > TARGET_MEAN_TOLERANCE:
                raise ValueError(
                    "overnight_gap_cross_section_rank is not centered on a valid "
                    "early-open cross-section"
                )

    rank_stats = stats["overnight_gap_cross_section_rank"]
    if computed_dates and (
        rank_stats.count == 0 or rank_stats.minimum == rank_stats.maximum
    ):
        raise ValueError(
            "overnight_gap_cross_section_rank is constant or entirely zero despite "
            f"{computed_dates} dates meeting the {MIN_ACTIVE_EQUITIES}-equity "
            "early-open population requirement"
        )

    decisions_per_date = len(DECISION_EQUITY_INDICES)
    return {
        "features": [
            _opening_feature_row(feature, feature_stats)
            for feature, feature_stats in stats.items()
        ],
        "equity_days_with_valid_early_open_proxy": early_open_equity_days,
        "minimum_rank_population": MIN_ACTIVE_EQUITIES,
        "rank_decision_cross_sections_computed": (computed_dates * decisions_per_date),
        "rank_decision_cross_sections_below_minimum_population": (
            below_minimum_dates * decisions_per_date
        ),
    }


def _load_arrays(features_dir: Path) -> dict[str, np.ndarray]:
    return {
        filename: np.load(features_dir / filename, mmap_mode="r", allow_pickle=False)
        for filename in output_array_specs(EXPECTED_DATE_COUNT)
    }


def _validate_shapes(
    arrays: dict[str, np.ndarray], manifest: dict[str, object]
) -> None:
    for filename, spec in output_array_specs(EXPECTED_DATE_COUNT).items():
        array = arrays[filename]
        if array.shape != spec.shape or array.dtype != spec.dtype:
            raise ValueError(
                f"Output contract mismatch for {filename}: {array.shape}/{array.dtype}"
            )
        manifest_spec = manifest["outputs"][filename]
        if (
            list(array.shape) != manifest_spec["shape"]
            or array.dtype.name != manifest_spec["dtype"]
        ):
            raise ValueError(f"Manifest output mismatch for {filename}")


def _check_bounds(values: np.ndarray, bounds: tuple[float, float], name: str) -> None:
    low, high = bounds
    if values.size and (values.min() < low - 1e-5 or values.max() > high + 1e-5):
        raise ValueError(f"{name} is outside [{low}, {high}]")


def validate_global_slow_fields(
    global_slow: np.ndarray, global_ready: np.ndarray
) -> None:
    if np.any(global_slow[..., GLOBAL_UNUSED_SLOW_CHANNEL_INDICES] != 0):
        raise ValueError("Global unused slow channels must be zero")
    if np.any(global_slow[~global_ready] != 0):
        raise ValueError("Unready global slow rows must be exactly zero")


def validate_liquidity_selected_rate_slow_fields(context_slow: np.ndarray) -> None:
    slot = LOCAL_CONTEXT_SYMBOLS.index(LIQUIDITY_SELECTED_RATE_CONTEXT_SYMBOL)
    values = context_slow[:, slot][
        ..., LIQUIDITY_SELECTED_RATE_ZERO_SLOW_CHANNEL_INDICES
    ]
    if np.any(values != 0):
        raise ValueError("DI1$N inapplicable slow channels must be zero")


def _validate_family_fields(arrays: dict[str, np.ndarray]) -> None:
    context_dynamic = arrays["context_features.npy"]
    context_slow = arrays["context_slow.npy"]
    equity_slow = arrays["equity_slow.npy"]
    global_dynamic = arrays["global_features.npy"]
    global_slow = arrays["global_slow.npy"]
    global_ready = arrays["global_data_ready.npy"]
    validate_liquidity_selected_rate_slow_fields(context_slow)
    if np.any(context_dynamic[..., 16:26] != 0):
        raise ValueError("Context cross-sectional dynamic channels must be zero")
    if np.any(equity_slow[..., 30:32] != 0):
        raise ValueError("Equity DI-only slow channels must be zero")
    if np.any(context_slow[..., 13:15] != 0):
        raise ValueError("Context equity-dollar-volume channels must be zero")
    if np.any(context_slow[..., 17:20] != 0):
        raise ValueError("Context cross-sectional slow ranks must be zero")
    if np.any(context_slow[..., 20:26] != 0):
        raise ValueError("Context exposure-beta channels must be zero")
    if np.any(context_slow[:, :2, 30:32] != 0):
        raise ValueError("WIN/WDO DI-only slow channels must be zero")
    if np.any(global_dynamic[..., 16:26] != 0):
        raise ValueError("Global equity-only dynamic channels must be zero")
    validate_global_slow_fields(global_slow, global_ready)


def _validate_targets(arrays: dict[str, np.ndarray]) -> None:
    targets = arrays["targets.npy"]
    raw_returns = arrays["raw_returns.npy"]
    label_mask = arrays["label_mask.npy"]
    medians = arrays["cross_section_median.npy"]
    horizon_mask = arrays["horizon_mask.npy"]
    equity_features = arrays["equity_features.npy"]
    membership = arrays["equity_membership.npy"]
    ready = arrays["equity_data_ready.npy"]

    for start in range(0, EXPECTED_DATE_COUNT, DATE_CHUNK):
        stop = min(start + DATE_CHUNK, EXPECTED_DATE_COUNT)
        chunk_targets = np.asarray(targets[start:stop], dtype=np.float32)
        chunk_raw = np.asarray(raw_returns[start:stop], dtype=np.float32)
        chunk_mask = np.asarray(label_mask[start:stop], dtype=bool)
        if not np.isfinite(chunk_targets).all() or not np.isfinite(chunk_raw).all():
            raise ValueError("Targets or raw returns contain non-finite values")
        if np.any(chunk_targets[~chunk_mask] != 0):
            raise ValueError("Invalid targets are not exactly zero")
        if np.any(chunk_raw[~chunk_mask] != 0):
            raise ValueError("Invalid raw returns are not exactly zero")
        valid_targets = chunk_targets[chunk_mask]
        if valid_targets.size and (
            valid_targets.min() <= -1.0 or valid_targets.max() >= 1.0
        ):
            raise ValueError("Valid rank targets must be strictly inside (-1, 1)")
        counts = chunk_mask.sum(axis=1)
        means = np.divide(
            (chunk_targets * chunk_mask).sum(axis=1, dtype=np.float64),
            counts,
            out=np.zeros_like(counts, dtype=np.float64),
            where=counts > 0,
        )
        if np.any(np.abs(means[counts > 0]) > TARGET_MEAN_TOLERANCE):
            raise ValueError("A valid target cross-section is not centered at zero")
        expected_horizon = counts >= 30
        if not np.array_equal(expected_horizon, horizon_mask[start:stop]):
            raise ValueError("horizon_mask disagrees with valid-label counts")

        observed = np.asarray(equity_features[start:stop, :, :, 5], dtype=bool)
        entry = observed[:, :, DECISION_EQUITY_INDICES]
        exits = np.stack(
            [
                observed[:, :, np.asarray(DECISION_EQUITY_INDICES) + horizon - 1]
                for horizon in HORIZONS
            ],
            axis=3,
        )
        required = (
            membership[start:stop, :, None, None]
            & ready[start:stop, :, None, None]
            & entry[:, :, :, None]
            & exits
            & horizon_mask[start:stop, None, :, :]
        )
        if np.any(chunk_mask & ~required):
            raise ValueError("label_mask violates membership, readiness, or endpoints")

        for local_date in range(stop - start):
            for decision_idx in range(len(DECISION_EQUITY_INDICES)):
                for horizon_idx in range(len(HORIZONS)):
                    valid = chunk_mask[local_date, :, decision_idx, horizon_idx]
                    if not valid.any():
                        continue
                    group_targets = chunk_targets[
                        local_date, valid, decision_idx, horizon_idx
                    ]
                    np.testing.assert_allclose(
                        group_targets,
                        centered_midranks(group_targets),
                        atol=1e-6,
                        rtol=0.0,
                        err_msg="Stored targets are not exact centered midranks",
                    )
                    group_raw = chunk_raw[local_date, valid, decision_idx, horizon_idx]
                    stored_median = medians[
                        start + local_date, decision_idx, horizon_idx
                    ]
                    if not np.isclose(np.median(group_raw), stored_median, atol=1e-7):
                        raise ValueError(
                            "Stored cross-sectional median is inconsistent"
                        )


def _collect_feature_stats(
    arrays: dict[str, np.ndarray], eligible_dates: np.ndarray
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray, np.ndarray]:
    equity_dynamic_stats = [StreamingStats() for _ in DYNAMIC_CHANNELS]
    equity_slow_stats = [StreamingStats() for _ in SLOW_CHANNELS]
    context_dynamic_stats = [
        [StreamingStats() for _ in DYNAMIC_CHANNELS] for _ in LOCAL_CONTEXT_SYMBOLS
    ]
    context_slow_stats = [
        [StreamingStats() for _ in SLOW_CHANNELS] for _ in LOCAL_CONTEXT_SYMBOLS
    ]
    global_dynamic_stats = [
        [StreamingStats() for _ in DYNAMIC_CHANNELS] for _ in GLOBAL_CONTEXT_SYMBOLS
    ]
    global_slow_stats = [
        [StreamingStats() for _ in GLOBAL_SLOW_CHANNELS] for _ in GLOBAL_CONTEXT_SYMBOLS
    ]
    security_observed = np.zeros(arrays["equity_features.npy"].shape[1], dtype=np.int64)
    security_possible = np.zeros_like(security_observed)
    security_active_days = np.zeros_like(security_observed)

    for start in range(0, EXPECTED_DATE_COUNT, DATE_CHUNK):
        stop = min(start + DATE_CHUNK, EXPECTED_DATE_COUNT)
        equity_dynamic = np.asarray(
            arrays["equity_features.npy"][start:stop], dtype=np.float32
        )
        equity_slow = np.asarray(
            arrays["equity_slow.npy"][start:stop], dtype=np.float32
        )
        context_dynamic = np.asarray(
            arrays["context_features.npy"][start:stop], dtype=np.float32
        )
        context_slow = np.asarray(
            arrays["context_slow.npy"][start:stop], dtype=np.float32
        )
        global_dynamic = np.asarray(
            arrays["global_features.npy"][start:stop], dtype=np.float32
        )
        global_slow = np.asarray(
            arrays["global_slow.npy"][start:stop], dtype=np.float32
        )
        for name, values in (
            ("equity_features", equity_dynamic),
            ("equity_slow", equity_slow),
            ("context_features", context_dynamic),
            ("context_slow", context_slow),
            ("global_features", global_dynamic),
            ("global_slow", global_slow),
        ):
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite value in {name} dates {start}:{stop}")
        for channel, bounds in enumerate(DYNAMIC_BOUNDS):
            _check_bounds(
                equity_dynamic[..., channel], bounds, f"equity dynamic {channel}"
            )
            _check_bounds(
                context_dynamic[..., channel], bounds, f"context dynamic {channel}"
            )
            _check_bounds(
                global_dynamic[..., channel], bounds, f"global dynamic {channel}"
            )
        for channel, bounds in enumerate(SLOW_BOUNDS):
            if bounds is not None:
                _check_bounds(
                    equity_slow[..., channel], bounds, f"equity slow {channel}"
                )
                _check_bounds(
                    context_slow[..., channel], bounds, f"context slow {channel}"
                )
                _check_bounds(
                    global_slow[..., channel], bounds, f"global slow {channel}"
                )

    eligible_set = set(eligible_dates.tolist())
    for start in range(0, EXPECTED_DATE_COUNT, DATE_CHUNK):
        indices = np.asarray(
            [
                index
                for index in range(start, min(start + DATE_CHUNK, EXPECTED_DATE_COUNT))
                if index in eligible_set
            ],
            dtype=np.int64,
        )
        if indices.size == 0:
            continue
        equity_dynamic = np.asarray(
            arrays["equity_features.npy"][indices, :, :EQUITY_VISIBLE_MINUTES],
            dtype=np.float32,
        )
        equity_slow = np.asarray(arrays["equity_slow.npy"][indices], dtype=np.float32)
        active = np.asarray(
            arrays["equity_membership.npy"][indices]
            & arrays["equity_data_ready.npy"][indices],
            dtype=bool,
        )
        dynamic_use = np.broadcast_to(active[:, :, None], equity_dynamic.shape[:-1])
        for channel, stats in enumerate(equity_dynamic_stats):
            stats.update(equity_dynamic[..., channel][dynamic_use])
        for channel, stats in enumerate(equity_slow_stats):
            stats.update(equity_slow[..., channel][active])

        observed = (equity_dynamic[..., 5] > 0.5) & dynamic_use
        security_observed += observed.sum(axis=(0, 2), dtype=np.int64)
        security_possible += active.sum(axis=0, dtype=np.int64) * EQUITY_VISIBLE_MINUTES
        security_active_days += active.sum(axis=0, dtype=np.int64)

        for slot in range(len(LOCAL_CONTEXT_SYMBOLS)):
            dynamic = np.asarray(
                arrays["context_features.npy"][indices, slot, :CONTEXT_VISIBLE_MINUTES],
                dtype=np.float32,
            )
            slow = np.asarray(
                arrays["context_slow.npy"][indices, slot], dtype=np.float32
            )
            ready = np.asarray(
                arrays["context_data_ready.npy"][indices, slot], dtype=bool
            )
            use = np.broadcast_to(ready[:, None], dynamic.shape[:-1])
            for channel, stats in enumerate(context_dynamic_stats[slot]):
                stats.update(dynamic[..., channel][use])
            for channel, stats in enumerate(context_slow_stats[slot]):
                stats.update(slow[..., channel][ready])
        for slot in range(len(GLOBAL_CONTEXT_SYMBOLS)):
            dynamic = np.asarray(
                arrays["global_features.npy"][indices, slot, :GLOBAL_VISIBLE_MINUTES],
                dtype=np.float32,
            )
            slow = np.asarray(
                arrays["global_slow.npy"][indices, slot], dtype=np.float32
            )
            ready = np.asarray(
                arrays["global_data_ready.npy"][indices, slot], dtype=bool
            )
            dynamic_use = np.broadcast_to(
                ready.any(axis=1)[:, None], dynamic.shape[:-1]
            )
            for channel, stats in enumerate(global_dynamic_stats[slot]):
                stats.update(dynamic[..., channel][dynamic_use])
            for channel, stats in enumerate(global_slow_stats[slot]):
                stats.update(slow[..., channel][ready])

    rows = [
        stats.row("equity_active", name)
        for name, stats in zip(DYNAMIC_CHANNELS, equity_dynamic_stats, strict=True)
    ]
    rows.extend(
        stats.row("equity_active", name)
        for name, stats in zip(SLOW_CHANNELS, equity_slow_stats, strict=True)
    )
    for slot, symbol in enumerate(LOCAL_CONTEXT_SYMBOLS):
        rows.extend(
            stats.row(f"context:{symbol}", name)
            for name, stats in zip(
                DYNAMIC_CHANNELS, context_dynamic_stats[slot], strict=True
            )
        )
        rows.extend(
            stats.row(f"context:{symbol}", name)
            for name, stats in zip(SLOW_CHANNELS, context_slow_stats[slot], strict=True)
        )
    for slot, symbol in enumerate(GLOBAL_CONTEXT_SYMBOLS):
        rows.extend(
            stats.row(f"global:{symbol}", name)
            for name, stats in zip(
                DYNAMIC_CHANNELS, global_dynamic_stats[slot], strict=True
            )
        )
        rows.extend(
            stats.row(f"global:{symbol}", name)
            for name, stats in zip(
                GLOBAL_SLOW_CHANNELS, global_slow_stats[slot], strict=True
            )
        )
    return rows, security_observed, security_possible, security_active_days


def _target_stats(
    arrays: dict[str, np.ndarray],
    trade_dates: list[object],
    eligible_dates: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    years = sorted({trade_dates[index].year for index in eligible_dates})
    target_rows: list[dict[str, object]] = []
    yearly_rows: list[dict[str, object]] = []
    for year in years:
        year_indices = eligible_dates[
            np.asarray([trade_dates[index].year == year for index in eligible_dates])
        ]
        year_row: dict[str, object] = {
            "year": year,
            "eligible_dates": int(year_indices.size),
            "sample_count": int(year_indices.size * len(DECISION_EQUITY_INDICES)),
        }
        for horizon_idx, horizon in enumerate(HORIZONS):
            target_stats = StreamingStats()
            raw_stats = StreamingStats()
            median_stats = StreamingStats()
            valid_count = 0
            opportunity_count = 0
            horizon_count = 0
            for start in range(0, year_indices.size, 32):
                chunk = year_indices[start : start + 32]
                mask = np.asarray(
                    arrays["label_mask.npy"][chunk, :, :, horizon_idx], dtype=bool
                )
                target_stats.update(
                    np.asarray(
                        arrays["targets.npy"][chunk, :, :, horizon_idx],
                        dtype=np.float32,
                    )[mask]
                )
                raw_stats.update(
                    10_000.0
                    * np.asarray(
                        arrays["raw_returns.npy"][chunk, :, :, horizon_idx],
                        dtype=np.float32,
                    )[mask]
                )
                horizon_mask = np.asarray(
                    arrays["horizon_mask.npy"][chunk, :, horizon_idx], dtype=bool
                )
                median_stats.update(
                    10_000.0
                    * np.asarray(
                        arrays["cross_section_median.npy"][chunk, :, horizon_idx],
                        dtype=np.float32,
                    )[horizon_mask]
                )
                valid_count += int(mask.sum())
                horizon_count += int(horizon_mask.sum())
                active = np.asarray(
                    arrays["equity_membership.npy"][chunk]
                    & arrays["equity_data_ready.npy"][chunk],
                    dtype=bool,
                )
                opportunity_count += int(active.sum()) * len(DECISION_EQUITY_INDICES)
            for metric, unit, stats in (
                ("rank_target", "centered rank", target_stats),
                ("raw_return", "basis points", raw_stats),
                ("cross_section_median", "basis points", median_stats),
            ):
                row = stats.row(str(year), metric)
                row.update({"horizon_minutes": horizon, "unit": unit})
                target_rows.append(row)
            year_row[f"target_{horizon}_mean"] = (
                target_stats.total / target_stats.count if target_stats.count else None
            )
            year_row[f"horizon_{horizon}_sample_coverage"] = horizon_count / (
                year_indices.size * len(DECISION_EQUITY_INDICES)
            )
            year_row[f"label_{horizon}_opportunity_coverage"] = (
                valid_count / opportunity_count if opportunity_count else 0.0
            )
        yearly_rows.append(year_row)
    return target_rows, yearly_rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_label(value: object) -> str:
    if TRAIN_START <= value <= TRAIN_END:
        return "train"
    if TRAIN_END < value < VALIDATION_START:
        return "embargo_1"
    if VALIDATION_START <= value <= VALIDATION_END:
        return "validation"
    if VALIDATION_END < value < TEST_START:
        return "embargo_2"
    if TEST_START <= value <= TEST_END:
        return "test"
    if value < TRAIN_START:
        return "warmup"
    return "post_test"


def _validated_split_counts(
    sample_trade_dates: list[object],
) -> dict[str, dict[str, int]]:
    split_labels = [_split_label(value) for value in sample_trade_dates]
    split_counts: dict[str, dict[str, int]] = {}
    for split, expected_dates in EXPECTED_SPLIT_DATE_COUNTS.items():
        positions = [
            index for index, label in enumerate(split_labels) if label == split
        ]
        row_count = len(positions)
        date_count = len({sample_trade_dates[index] for index in positions})
        if (
            date_count != expected_dates
            or row_count != EXPECTED_SPLIT_SAMPLE_COUNTS[split]
        ):
            raise ValueError(f"{split} split counts do not match the contract")
        split_counts[split] = {"date_count": date_count, "sample_count": row_count}
    return split_counts


def _date_groups(date_index: pl.DataFrame) -> pl.DataFrame:
    trade_dates = date_index["trade_date"].to_list()
    return date_index.select("date_idx", "trade_date").with_columns(
        pl.col("trade_date").dt.year().alias("year"),
        pl.Series("split", [_split_label(value) for value in trade_dates]),
    )


def _audit_global_source(
    source_dir: Path, source_manifest: dict[str, object]
) -> list[dict[str, object]]:
    for path_text, expected in source_manifest["source_hashes"].items():
        path = workspace_path(path_text)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Global source hash mismatch: {path}")
    for relative, expected in source_manifest["normalized_hashes"].items():
        path = source_dir / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Normalized global store hash mismatch: {path}")
        partition = pl.read_parquet(path)
        if partition.is_empty():
            raise ValueError(f"Normalized global partition is empty: {path}")
        validate_normalized_bars(
            partition, expected_symbol=str(partition.item(0, "continuous_symbol"))
        )

    rows: list[dict[str, object]] = []
    for symbol in GLOBAL_CONTEXT_SYMBOLS:
        frame = load_global_symbol(source_dir, symbol)
        timestamps = frame["ts_event_utc"].cast(pl.Int64).to_numpy()
        delta_minutes = np.diff(timestamps) // 60_000_000_000
        rows.append(
            {
                "continuous_symbol": symbol,
                "row_count": frame.height,
                "first_ts_event_utc": str(frame.item(0, "ts_event_utc")),
                "last_ts_event_utc": str(frame.item(-1, "ts_event_utc")),
                "gap_count": int(np.count_nonzero(delta_minutes > 1)),
                "missing_minutes_between_observations": int(
                    np.maximum(delta_minutes - 1, 0).sum()
                ),
                "mapping_change_count": int(frame["mapping_changed"].sum()),
                "receipt_timestamp_fraction": float(
                    frame["received_at_utc"].is_not_null().mean()
                ),
            }
        )
    return rows


def _global_feature_stats(
    arrays: dict[str, np.ndarray],
    date_groups: pl.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    canonical = date_groups.filter(
        pl.col("split").is_in(["train", "validation", "test"])
    )
    for group in (
        canonical.select("year", "split")
        .unique()
        .sort("year", "split")
        .iter_rows(named=True)
    ):
        indices = (
            canonical.filter(
                (pl.col("year") == group["year"]) & (pl.col("split") == group["split"])
            )["date_idx"]
            .to_numpy()
            .astype(np.int64)
        )
        for slot, symbol in enumerate(GLOBAL_CONTEXT_SYMBOLS):
            dynamic_stats = [StreamingStats() for _ in DYNAMIC_CHANNELS]
            slow_stats = [StreamingStats() for _ in GLOBAL_SLOW_CHANNELS]
            for start in range(0, indices.size, DATE_CHUNK):
                chunk = indices[start : start + DATE_CHUNK]
                dynamic = np.asarray(
                    arrays["global_features.npy"][chunk, slot, :GLOBAL_VISIBLE_MINUTES],
                    dtype=np.float32,
                )
                observed = dynamic[..., 5] > 0.5
                slow = np.asarray(
                    arrays["global_slow.npy"][chunk, slot], dtype=np.float32
                )
                ready = np.asarray(
                    arrays["global_data_ready.npy"][chunk, slot], dtype=bool
                )
                for channel, stats in enumerate(dynamic_stats):
                    stats.update(dynamic[..., channel][observed])
                for channel, stats in enumerate(slow_stats):
                    stats.update(slow[..., channel][ready])
            for kind, names, stats_by_channel in (
                ("dynamic", DYNAMIC_CHANNELS, dynamic_stats),
                ("slow", GLOBAL_SLOW_CHANNELS, slow_stats),
            ):
                for name, stats in zip(names, stats_by_channel, strict=True):
                    row = stats.row(f"global:{symbol}", name)
                    row.update(
                        {
                            "continuous_symbol": symbol,
                            "year": int(group["year"]),
                            "split": str(group["split"]),
                            "feature_kind": kind,
                        }
                    )
                    rows.append(row)
    return rows


def _global_coverage_reports(
    features_dir: Path,
    global_coverage: pl.DataFrame,
    date_groups: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    groups = date_groups.select("date_idx", "year", "split")
    coverage = global_coverage.join(groups, on="date_idx", how="left")
    decision_summary = (
        coverage.group_by(
            "global_slot",
            "continuous_symbol",
            "year",
            "split",
            "decision_idx",
        )
        .agg(
            pl.len().alias("date_count"),
            pl.col("observed_fraction").mean().alias("mean_observed_fraction"),
            pl.col("staleness_minutes").mean().alias("mean_staleness_minutes"),
            pl.col("staleness_minutes").max().alias("maximum_staleness_minutes"),
            pl.col("ready").mean().alias("readiness_rate"),
            pl.col("roll_count").sum().alias("roll_count"),
        )
        .sort("global_slot", "year", "split", "decision_idx")
    )
    date_counts = groups.group_by("year", "split").agg(
        pl.col("date_idx").n_unique().alias("possible_dates")
    )
    index = pl.scan_parquet(features_dir / "global_context_index.parquet")
    minute_summary = (
        index.join(groups.lazy(), on="date_idx", how="inner")
        .group_by(
            "global_slot",
            "continuous_symbol",
            "year",
            "split",
            "minute_idx",
        )
        .agg(pl.len().alias("observed_dates"))
        .join(date_counts.lazy(), on=["year", "split"], how="left")
        .with_columns(
            (pl.col("observed_dates") / pl.col("possible_dates")).alias(
                "observed_fraction"
            )
        )
        .sort("global_slot", "year", "split", "minute_idx")
        .collect()
    )
    roll_rows = (
        index.filter(pl.col("mapping_changed"))
        .join(groups.lazy(), on="date_idx", how="left")
        .select(
            "global_slot",
            "continuous_symbol",
            "family",
            "year",
            "split",
            "trade_date",
            "minute_idx",
            "ts_event_utc",
            "bar_end_utc",
            "instrument_id",
            "raw_symbol",
            "expiration_utc",
        )
        .sort("global_slot", "ts_event_utc")
        .collect()
    )
    return coverage, decision_summary, minute_summary, roll_rows


def _generate_feature_audit(
    features_dir: Path,
    output_dir: Path,
    final_output_dir: Path,
    created_at: datetime,
) -> None:
    features_dir = Path(features_dir)
    if not features_dir.is_dir():
        raise FileNotFoundError(f"Feature directory does not exist: {features_dir}")
    manifest = json.loads((features_dir / "manifest.json").read_text(encoding="utf-8"))
    constants = manifest["constants"]
    if tuple(constants["dynamic_channels"]) != DYNAMIC_CHANNELS:
        raise ValueError("Manifest dynamic-channel order is stale")
    if tuple(constants["equity_slow_channels"]) != SLOW_CHANNELS:
        raise ValueError("Manifest equity slow-channel order is stale")
    if tuple(constants["context_slow_channels"]) != SLOW_CHANNELS:
        raise ValueError("Manifest context slow-channel order is stale")
    if tuple(constants["global_slow_channels"]) != GLOBAL_SLOW_CHANNELS:
        raise ValueError("Manifest global slow-channel order is stale")
    if tuple(constants["global_context_symbols"]) != GLOBAL_CONTEXT_SYMBOLS:
        raise ValueError("Manifest global symbol order is stale")
    if tuple(constants["local_context_symbols"]) != LOCAL_CONTEXT_SYMBOLS:
        raise ValueError("Manifest local symbol order is stale")
    if tuple(constants["fixed_rate_context_symbols"]) != FIXED_RATE_CONTEXT_SYMBOLS:
        raise ValueError("Manifest fixed-rate symbol set is stale")
    if (
        constants["liquidity_selected_rate_context_symbol"]
        != LIQUIDITY_SELECTED_RATE_CONTEXT_SYMBOL
    ):
        raise ValueError("Manifest liquidity-selected rate symbol is stale")
    if (
        tuple(constants["exposure_beta_context_symbols"])
        != EXPOSURE_BETA_CONTEXT_SYMBOLS
    ):
        raise ValueError("Manifest exposure-beta sources are stale")
    if manifest.get("sample_eligibility_rule") != SAMPLE_ELIGIBILITY_RULE:
        raise ValueError("Manifest sample-eligibility rule is stale")
    if (
        manifest.get("local_context_availability_rule")
        != LOCAL_CONTEXT_AVAILABILITY_RULE
    ):
        raise ValueError("Manifest local-context availability rule is stale")

    date_index = pl.read_parquet(features_dir / "date_index.parquet")
    equity_index = pl.read_parquet(features_dir / "equity_index.parquet")
    context_index = pl.read_parquet(features_dir / "context_index.parquet")
    global_axis = (
        pl.scan_parquet(features_dir / "global_context_index.parquet")
        .select("global_slot", "continuous_symbol", "family", "quote_direction")
        .unique()
        .sort("global_slot")
        .collect()
    )
    sample_index = pl.read_parquet(features_dir / "sample_index.parquet")
    daily_audit = pl.read_parquet(features_dir / "daily_audit.parquet")
    global_coverage = pl.read_parquet(features_dir / "global_coverage.parquet")
    if (
        date_index.height != EXPECTED_DATE_COUNT
        or daily_audit.height != EXPECTED_DATE_COUNT
    ):
        raise ValueError("Date metadata does not preserve the 1,248-date contract")
    if sample_index.height != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Sample metadata does not preserve the {EXPECTED_SAMPLE_COUNT:,}-sample contract"
        )
    if tuple(context_index.get_column("symbol")) != LOCAL_CONTEXT_SYMBOLS:
        raise ValueError("Context index order does not match the feature contract")
    liquidity_selected = context_index.filter(
        pl.col("symbol") == LIQUIDITY_SELECTED_RATE_CONTEXT_SYMBOL
    )
    if liquidity_selected.height != 1:
        raise ValueError("Context index must contain exactly one DI1$N row")
    row = liquidity_selected.row(0, named=True)
    if (
        row["rate_representation"] != "liquidity_selected_unadjusted"
        or row["fixed_expiry_applicable"]
        or row["cross_session_price_features_applicable"]
        or row["absolute_rate_level_applicable"]
        or not row["session_boundary_price_state_reset"]
        or row["expiry_date"] is not None
    ):
        raise ValueError("DI1$N context applicability metadata is inconsistent")
    fixed = context_index.filter(pl.col("symbol").is_in(FIXED_RATE_CONTEXT_SYMBOLS))
    if (
        fixed.height != len(FIXED_RATE_CONTEXT_SYMBOLS)
        or set(fixed["rate_representation"]) != {"fixed_maturity"}
        or not fixed["fixed_expiry_applicable"].all()
        or not fixed["cross_session_price_features_applicable"].all()
        or not fixed["absolute_rate_level_applicable"].all()
        or fixed["session_boundary_price_state_reset"].any()
        or fixed["expiry_date"].null_count()
    ):
        raise ValueError("Fixed-DI context applicability metadata is inconsistent")

    arrays = _load_arrays(features_dir)
    _validate_shapes(arrays, manifest)
    _validate_family_fields(arrays)
    _validate_targets(arrays)
    required_metadata = {
        "global_context_index.parquet",
        "global_coverage.parquet",
        "equity_tod_profile.npy",
        "equity_tod_profile.json",
    }
    if not required_metadata <= set(manifest["metadata_files"]):
        raise ValueError("Manifest does not require every audit metadata artifact")
    if any(not (features_dir / name).is_file() for name in required_metadata):
        raise FileNotFoundError("A required audit metadata artifact is missing")
    if tuple(global_axis["continuous_symbol"]) != GLOBAL_CONTEXT_SYMBOLS:
        raise ValueError("Global context index order does not match the contract")
    expected_coverage_rows = (
        EXPECTED_DATE_COUNT * len(GLOBAL_CONTEXT_SYMBOLS) * len(DECISION_GLOBAL_INDICES)
    )
    if global_coverage.height != expected_coverage_rows:
        raise ValueError("Global coverage does not contain every symbol/date/decision")
    if (
        global_coverage.select("global_slot", "date_idx", "decision_idx")
        .is_duplicated()
        .any()
    ):
        raise ValueError("Global coverage keys are not unique")
    global_metadata = manifest["global_context"]
    source_dir = Path(global_metadata["normalized_source_path"])
    source_manifest = json.loads(
        (source_dir / "manifest.json").read_text(encoding="utf-8")
    )
    for feature_field, source_field in (
        ("provider", "provider"),
        ("dataset", "dataset"),
        ("schema", "schema"),
        ("databento_version", "databento_version"),
        ("symbols", "symbols"),
        ("source_hashes", "source_hashes"),
        ("normalized_store_hashes", "normalized_hashes"),
        ("continuous_roll_rule", "continuous_roll_rule"),
    ):
        if global_metadata[feature_field] != source_manifest[source_field]:
            raise ValueError(f"Global source manifest mismatch: {feature_field}")
    source_rows = _audit_global_source(source_dir, source_manifest)
    coverage_ready = (
        global_coverage.sort("date_idx", "global_slot", "decision_idx")["ready"]
        .to_numpy()
        .reshape(
            EXPECTED_DATE_COUNT,
            len(GLOBAL_CONTEXT_SYMBOLS),
            len(DECISION_GLOBAL_INDICES),
        )
    )
    if not np.array_equal(coverage_ready, arrays["global_data_ready.npy"]):
        raise ValueError("Global readiness array disagrees with coverage")
    future = global_coverage.filter(
        pl.col("last_observed_bar_end_utc").is_not_null()
        & (pl.col("last_observed_bar_end_utc") > pl.col("decision_time_utc"))
    )
    if future.height:
        raise ValueError("A global decision uses an unavailable future bar")
    roll_rows = (
        pl.scan_parquet(features_dir / "global_context_index.parquet")
        .filter(pl.col("mapping_changed"))
        .select("date_idx", "global_slot", "minute_idx")
        .collect()
    )
    if roll_rows.height:
        roll_moves = arrays["global_features.npy"][
            roll_rows["date_idx"].to_numpy(),
            roll_rows["global_slot"].to_numpy(),
            roll_rows["minute_idx"].to_numpy(),
            :4,
        ]
        if np.any(roll_moves != 0):
            raise ValueError("A cross-contract price move survived a mapping change")

    eligible_dates = np.sort(
        sample_index.get_column("date_idx").unique().to_numpy().astype(np.int64)
    )
    if sample_index.height != eligible_dates.size * len(DECISION_EQUITY_INDICES):
        raise ValueError("Eligible dates do not contain exactly 55 samples each")
    if eligible_dates.size != EXPECTED_ELIGIBLE_DATE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ELIGIBLE_DATE_COUNT} eligible dates, "
            f"found {eligible_dates.size}"
        )
    if int(manifest["sample_count"]) != sample_index.height:
        raise ValueError("Manifest sample_count does not match sample_index")
    if int(manifest["eligible_date_count"]) != eligible_dates.size:
        raise ValueError("Manifest eligible_date_count does not match sample_index")
    sample_ids = sample_index.get_column("sample_id").to_numpy()
    if not np.array_equal(sample_ids, np.arange(sample_index.height)):
        raise ValueError("sample_id must be unique and contiguous from zero")
    expected_decisions = list(range(len(DECISION_EQUITY_INDICES)))
    decisions_by_date = sample_index.group_by("date_idx").agg(
        pl.col("decision_idx").sort()
    )
    if any(
        decisions != expected_decisions
        for decisions in decisions_by_date.get_column("decision_idx").to_list()
    ):
        raise ValueError("Every eligible date must contain decision_idx exactly 0..54")
    sample_dates = sample_index.get_column("date_idx").to_numpy().astype(np.int64)
    active_by_date = (
        arrays["equity_membership.npy"] & arrays["equity_data_ready.npy"]
    ).sum(axis=1)
    expected_eligible_dates = np.flatnonzero(active_by_date >= MIN_ACTIVE_EQUITIES)
    if not np.array_equal(eligible_dates, expected_eligible_dates):
        raise ValueError(
            "Sample dates must equal the dates meeting only the active-equity rule"
        )
    sample_active = active_by_date[sample_dates]
    if np.any(sample_active < MIN_ACTIVE_EQUITIES):
        raise ValueError(
            f"sample_index contains a date with fewer than {MIN_ACTIVE_EQUITIES} active equities"
        )
    if not np.array_equal(
        sample_active,
        sample_index.get_column("active_equity_count").to_numpy(),
    ):
        raise ValueError("sample_index active-equity counts are inconsistent")
    trade_dates = date_index.get_column("trade_date").to_list()
    first_eligible_date = trade_dates[int(eligible_dates[0])]
    last_eligible_date = trade_dates[int(eligible_dates[-1])]
    if (
        first_eligible_date != EXPECTED_FIRST_ELIGIBLE_DATE
        or last_eligible_date != EXPECTED_LAST_ELIGIBLE_DATE
    ):
        raise ValueError("Eligible-date boundaries do not match the contract")
    if manifest["first_feature_eligible_date"] != str(first_eligible_date) or manifest[
        "last_feature_eligible_date"
    ] != str(last_eligible_date):
        raise ValueError("Manifest eligible-date boundaries are inconsistent")
    sample_trade_dates = sample_index.get_column("trade_date").to_list()
    split_counts = _validated_split_counts(sample_trade_dates)
    unavailable_local_dates = int(
        (~arrays["context_data_ready.npy"][eligible_dates].all(axis=1)).sum()
    )
    if (
        unavailable_local_dates
        != EXPECTED_ELIGIBLE_DATES_WITH_UNAVAILABLE_LOCAL_CONTEXT
    ):
        raise ValueError(
            "Eligible-date local-context unavailability does not match the contract"
        )
    local_readiness = local_context_readiness_rows(
        arrays["context_data_ready.npy"], trade_dates, eligible_dates
    )
    if manifest.get("local_context_readiness") != local_readiness:
        raise ValueError("Manifest local-context readiness summary is inconsistent")
    if (
        manifest.get("eligible_dates_with_unavailable_local_context")
        != unavailable_local_dates
    ):
        raise ValueError("Manifest unavailable-local-date count is inconsistent")

    tod_profile = validate_profile_artifact(features_dir, tuple(trade_dates))
    if manifest.get("equity_normalization") != tod_profile:
        raise ValueError("Manifest causal TOD profile lineage is inconsistent")

    feature_rows, security_observed, security_possible, security_active_days = (
        _collect_feature_stats(arrays, eligible_dates)
    )
    opening_feature_stats = opening_feature_family_stats(arrays, eligible_dates)
    target_rows, yearly_rows = _target_stats(arrays, trade_dates, eligible_dates)

    date_groups = _date_groups(date_index)
    eligible_groups = date_groups.filter(pl.col("date_idx").is_in(eligible_dates))
    global_feature_rows = _global_feature_stats(arrays, eligible_groups)
    (
        global_coverage_report,
        global_decision_summary,
        global_minute_summary,
        global_roll_rows,
    ) = _global_coverage_reports(features_dir, global_coverage, date_groups)
    global_readiness = (
        global_coverage_report.group_by("global_slot", "continuous_symbol")
        .agg(
            pl.col("observed_fraction").mean().alias("mean_observed_fraction"),
            pl.col("staleness_minutes").max().alias("maximum_staleness_minutes"),
            pl.col("ready").mean().alias("readiness_rate"),
            pl.col("roll_count").sum().alias("roll_count"),
        )
        .sort("global_slot")
        .to_dicts()
    )

    security_label_counts = np.zeros(
        (equity_index.height, len(HORIZONS)), dtype=np.int64
    )
    for start in range(0, EXPECTED_DATE_COUNT, DATE_CHUNK):
        stop = min(start + DATE_CHUNK, EXPECTED_DATE_COUNT)
        security_label_counts += arrays["label_mask.npy"][start:stop].sum(
            axis=(0, 2), dtype=np.int64
        )
    security_stats = equity_index.select(
        "equity_slot", "security_id", "latest_ticker"
    ).with_columns(
        pl.Series("active_days", security_active_days, dtype=pl.Int32),
        pl.Series("observed_input_bars", security_observed, dtype=pl.Int64),
        pl.Series("possible_input_bars", security_possible, dtype=pl.Int64),
        pl.Series(
            "observed_input_fraction",
            np.divide(
                security_observed,
                security_possible,
                out=np.zeros_like(security_observed, dtype=np.float64),
                where=security_possible > 0,
            ),
        ),
        pl.Series("valid_labels_30", security_label_counts[:, 0], dtype=pl.Int64),
        pl.Series("valid_labels_60", security_label_counts[:, 1], dtype=pl.Int64),
        pl.Series("valid_labels_120", security_label_counts[:, 2], dtype=pl.Int64),
    )

    active_counts = (
        arrays["equity_membership.npy"][eligible_dates]
        & arrays["equity_data_ready.npy"][eligible_dates]
    ).sum(axis=1)
    context_density: dict[str, float] = {}
    for slot, symbol in enumerate(LOCAL_CONTEXT_SYMBOLS):
        ready = arrays["context_data_ready.npy"][eligible_dates, slot]
        observed = (
            arrays["context_features.npy"][
                eligible_dates, slot, :CONTEXT_VISIBLE_MINUTES, 5
            ]
            > 0.5
        )
        denominator = int(ready.sum()) * CONTEXT_VISIBLE_MINUTES
        context_density[symbol] = (
            int((observed & ready[:, None]).sum()) / denominator if denominator else 0.0
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    store_size = sum(
        path.stat().st_size for path in features_dir.iterdir() if path.is_file()
    )
    max_target_mean = max(
        abs(float(row["mean"]))
        for row in target_rows
        if row["feature"] == "rank_target" and row["mean"] is not None
    )
    summary = {
        "created_at_utc": created_at.isoformat(),
        "features_dir": str(features_dir),
        "audit_output_dir": str(final_output_dir),
        "contract_version": manifest["contract_version"],
        "date_count": EXPECTED_DATE_COUNT,
        "eligible_date_count": int(eligible_dates.size),
        "first_eligible_date": str(first_eligible_date),
        "last_eligible_date": str(last_eligible_date),
        "sample_count": EXPECTED_SAMPLE_COUNT,
        "sample_eligibility_rule": SAMPLE_ELIGIBILITY_RULE,
        "local_context_availability_rule": LOCAL_CONTEXT_AVAILABILITY_RULE,
        "eligible_dates_with_unavailable_local_context": unavailable_local_dates,
        "split_counts": split_counts,
        "local_context_readiness": local_readiness,
        "store_size_bytes": store_size,
        "active_equities": {
            "min": int(active_counts.min()),
            "median": float(np.median(active_counts)),
            "mean": float(active_counts.mean()),
            "max": int(active_counts.max()),
        },
        "context_observed_input_fraction": context_density,
        "opening_feature_family": opening_feature_stats,
        "equity_normalization": tod_profile,
        "global_context": global_readiness,
        "global_source": source_rows,
        "candidate_6l_audit_command": (
            "python -m brazil_rv.preprocessing.global_source audit-6l "
            "--input-parquet <path> --output <path>"
        ),
        "maximum_absolute_year_horizon_rank_target_mean": max_target_mean,
        "checks": [
            "exact shape and dtype contract",
            "finite dynamic, slow, raw-return, median, and target arrays",
            "all declared channel bounds",
            "family-inapplicable fields are exactly zero",
            "invalid targets and raw returns are exactly zero",
            "valid targets are exact centered midranks inside (-1, 1)",
            "every valid target cross-section is centered at zero",
            "raw-return medians and horizon masks are consistent",
            "membership, equity readiness, and exact label endpoints are enforced",
            "local and global context readiness never gates B3 samples",
            "unavailable local contexts are explicitly audited for ingress masking",
            "eligible-date, sample, boundary, and split counts match the contract",
            "opening-feature validity and activation statistics are reported",
            "expected overnight-gap ranks are finite, centered, and non-degenerate",
            "global source timestamps, OHLCV, hashes, and slot order are valid",
            "global decision slices contain no future bar",
            "global mapping changes suppress cross-contract returns",
            "global readiness does not gate B3 sample eligibility",
            "global feature distributions are reported by symbol, year, and split",
            "causal TOD profile is bounded, frozen after training, and hash verified",
        ],
        "output_files": [
            "audit_summary.json",
            "feature_stats.csv",
            "local_context_readiness.csv",
            "target_stats.csv",
            "yearly_stats.csv",
            "security_stats.csv",
            "global_source_stats.csv",
            "global_feature_stats.csv",
            "global_decision_coverage.csv",
            "global_minute_coverage.csv",
            "global_coverage.parquet",
            "global_rolls.parquet",
        ],
    }
    pl.DataFrame(feature_rows).write_csv(output_dir / "feature_stats.csv")
    pl.DataFrame(local_readiness).write_csv(output_dir / "local_context_readiness.csv")
    pl.DataFrame(target_rows).write_csv(output_dir / "target_stats.csv")
    pl.DataFrame(yearly_rows).write_csv(output_dir / "yearly_stats.csv")
    security_stats.write_csv(output_dir / "security_stats.csv")
    pl.DataFrame(source_rows).write_csv(output_dir / "global_source_stats.csv")
    pl.DataFrame(global_feature_rows).write_csv(output_dir / "global_feature_stats.csv")
    global_decision_summary.write_csv(output_dir / "global_decision_coverage.csv")
    global_minute_summary.write_csv(output_dir / "global_minute_coverage.csv")
    global_coverage_report.write_parquet(output_dir / "global_coverage.parquet")
    global_roll_rows.write_parquet(output_dir / "global_rolls.parquet")
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def audit_feature_store(features_dir: Path) -> Path:
    """Run the complete store audit and atomically publish its output directory."""
    created_at = datetime.now(timezone.utc)
    output_dir = AUDIT_BASE / f"m1_features_audit_{created_at:%Y%m%dT%H%M%S%fZ}"
    partial = output_dir.with_name(f"{output_dir.name}.{uuid4().hex}.partial")
    if output_dir.exists():
        raise FileExistsError(f"Feature audit output already exists: {output_dir}")
    try:
        _generate_feature_audit(features_dir, partial, output_dir, created_at)
        os.replace(partial, output_dir)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return output_dir


def main() -> None:
    features_dir = Path(CANONICAL_OUTPUT_POINTER.read_text(encoding="utf-8").strip())
    audit_feature_store(features_dir)


if __name__ == "__main__":
    main()
