from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from .contract import (
    CANONICAL_OUTPUT_POINTER,
    CONTEXT_SYMBOLS,
    DECISION_CONTEXT_INDICES,
    DECISION_EQUITY_INDICES,
    DYNAMIC_CHANNELS,
    EXPECTED_DATE_COUNT,
    EXPECTED_SAMPLE_COUNT,
    HORIZONS,
    SLOW_CHANNELS,
    output_array_specs,
)
from .transforms import centered_midranks

AUDIT_BASE = CANONICAL_OUTPUT_POINTER.parent.parent / "feature_audits"
EQUITY_VISIBLE_MINUTES = max(DECISION_EQUITY_INDICES)
CONTEXT_VISIBLE_MINUTES = max(DECISION_CONTEXT_INDICES)
DATE_CHUNK = 8
TARGET_MEAN_TOLERANCE = 2e-6

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


def _validate_family_fields(arrays: dict[str, np.ndarray]) -> None:
    context_dynamic = arrays["context_features.npy"]
    context_slow = arrays["context_slow.npy"]
    equity_slow = arrays["equity_slow.npy"]
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
        [StreamingStats() for _ in DYNAMIC_CHANNELS] for _ in CONTEXT_SYMBOLS
    ]
    context_slow_stats = [
        [StreamingStats() for _ in SLOW_CHANNELS] for _ in CONTEXT_SYMBOLS
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
        for name, values in (
            ("equity_features", equity_dynamic),
            ("equity_slow", equity_slow),
            ("context_features", context_dynamic),
            ("context_slow", context_slow),
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
        for channel, bounds in enumerate(SLOW_BOUNDS):
            if bounds is not None:
                _check_bounds(
                    equity_slow[..., channel], bounds, f"equity slow {channel}"
                )
                _check_bounds(
                    context_slow[..., channel], bounds, f"context slow {channel}"
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

        for slot in range(len(CONTEXT_SYMBOLS)):
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

    rows = [
        stats.row("equity_active", name)
        for name, stats in zip(DYNAMIC_CHANNELS, equity_dynamic_stats, strict=True)
    ]
    rows.extend(
        stats.row("equity_active", name)
        for name, stats in zip(SLOW_CHANNELS, equity_slow_stats, strict=True)
    )
    for slot, symbol in enumerate(CONTEXT_SYMBOLS):
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


def audit_feature_store(features_dir: Path) -> Path:
    """Run the complete store audit and return its immutable output directory."""
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

    date_index = pl.read_parquet(features_dir / "date_index.parquet")
    equity_index = pl.read_parquet(features_dir / "equity_index.parquet")
    context_index = pl.read_parquet(features_dir / "context_index.parquet")
    sample_index = pl.read_parquet(features_dir / "sample_index.parquet")
    daily_audit = pl.read_parquet(features_dir / "daily_audit.parquet")
    if (
        date_index.height != EXPECTED_DATE_COUNT
        or daily_audit.height != EXPECTED_DATE_COUNT
    ):
        raise ValueError("Date metadata does not preserve the 1,248-date contract")
    if sample_index.height != EXPECTED_SAMPLE_COUNT:
        raise ValueError("Sample metadata does not preserve the 59,565-sample contract")
    if tuple(context_index.get_column("symbol")) != CONTEXT_SYMBOLS:
        raise ValueError("Context index order does not match the feature contract")

    arrays = _load_arrays(features_dir)
    _validate_shapes(arrays, manifest)
    _validate_family_fields(arrays)
    _validate_targets(arrays)

    eligible_dates = np.sort(
        sample_index.get_column("date_idx").unique().to_numpy().astype(np.int64)
    )
    if sample_index.height != eligible_dates.size * len(DECISION_EQUITY_INDICES):
        raise ValueError("Eligible dates do not contain exactly 55 samples each")
    if int(manifest["sample_count"]) != sample_index.height:
        raise ValueError("Manifest sample_count does not match sample_index")
    sample_dates = sample_index.get_column("date_idx").to_numpy().astype(np.int64)
    if not arrays["context_data_ready.npy"][sample_dates].all():
        raise ValueError("sample_index contains a context-unready date")
    sample_active = (
        arrays["equity_membership.npy"][sample_dates]
        & arrays["equity_data_ready.npy"][sample_dates]
    ).sum(axis=1)
    if np.any(sample_active < 30):
        raise ValueError(
            "sample_index contains a date with fewer than 30 active equities"
        )
    if not np.array_equal(
        sample_active,
        sample_index.get_column("active_equity_count").to_numpy(),
    ):
        raise ValueError("sample_index active-equity counts are inconsistent")

    feature_rows, security_observed, security_possible, security_active_days = (
        _collect_feature_stats(arrays, eligible_dates)
    )
    trade_dates = date_index.get_column("trade_date").to_list()
    target_rows, yearly_rows = _target_stats(arrays, trade_dates, eligible_dates)

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
    for slot, symbol in enumerate(CONTEXT_SYMBOLS):
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

    created_at = datetime.now(timezone.utc)
    output_dir = AUDIT_BASE / f"m1_features_v1_audit_{created_at:%Y%m%dT%H%M%S%fZ}"
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
        "audit_output_dir": str(output_dir),
        "contract_version": manifest["contract_version"],
        "date_count": EXPECTED_DATE_COUNT,
        "eligible_date_count": int(eligible_dates.size),
        "first_eligible_date": str(trade_dates[int(eligible_dates[0])]),
        "last_eligible_date": str(trade_dates[int(eligible_dates[-1])]),
        "sample_count": EXPECTED_SAMPLE_COUNT,
        "store_size_bytes": store_size,
        "active_equities": {
            "min": int(active_counts.min()),
            "median": float(np.median(active_counts)),
            "mean": float(active_counts.mean()),
            "max": int(active_counts.max()),
        },
        "context_observed_input_fraction": context_density,
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
            "membership, readiness, and exact label endpoints are enforced",
            "date and sample counts are unchanged",
        ],
        "output_files": [
            "audit_summary.json",
            "feature_stats.csv",
            "target_stats.csv",
            "yearly_stats.csv",
            "security_stats.csv",
        ],
    }
    pl.DataFrame(feature_rows).write_csv(output_dir / "feature_stats.csv")
    pl.DataFrame(target_rows).write_csv(output_dir / "target_stats.csv")
    pl.DataFrame(yearly_rows).write_csv(output_dir / "yearly_stats.csv")
    security_stats.write_csv(output_dir / "security_stats.csv")
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return output_dir


def main() -> None:
    features_dir = Path(CANONICAL_OUTPUT_POINTER.read_text(encoding="utf-8").strip())
    audit_feature_store(features_dir)


if __name__ == "__main__":
    main()
