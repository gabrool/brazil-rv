from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl

from .contract import (
    CANONICAL_OUTPUT_POINTER,
    CONTEXT_SYMBOLS,
    DECISION_CONTEXT_INDICES,
    DECISION_EQUITY_INDICES,
    HORIZONS,
    PRICE_FEATURE_CLIP,
    VOL_REGIME_CLIP,
    VOLUME_FEATURE_CLIP,
)

AUDIT_BASE = CANONICAL_OUTPUT_POINTER.parent.parent / "feature_audits"
EQUITY_VISIBLE_MINUTES = max(DECISION_EQUITY_INDICES)
CONTEXT_VISIBLE_MINUTES = max(DECISION_CONTEXT_INDICES)
DATE_CHUNK = 16


@dataclass
class BoundedStats:
    low: float
    high: float
    bins: int = 2000

    def __post_init__(self) -> None:
        self.edges = np.linspace(self.low, self.high, self.bins + 1, dtype=np.float64)
        self.hist = np.zeros(self.bins, dtype=np.int64)
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.minimum = np.inf
        self.maximum = -np.inf
        self.zero_count = 0
        self.low_clip_count = 0
        self.high_clip_count = 0

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).ravel()
        if values.size == 0:
            return
        if not np.isfinite(values).all():
            raise ValueError("Non-finite value encountered during statistical audit")
        if values.min() < self.low - 1e-5 or values.max() > self.high + 1e-5:
            raise ValueError(
                f"Value outside declared feature bounds [{self.low}, {self.high}]"
            )
        self.count += int(values.size)
        self.total += float(values.sum(dtype=np.float64))
        self.total_sq += float(
            np.square(values, dtype=np.float64).sum(dtype=np.float64)
        )
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        self.zero_count += int(np.count_nonzero(values == 0.0))
        tolerance = 1e-6
        self.low_clip_count += int(np.count_nonzero(values <= self.low + tolerance))
        self.high_clip_count += int(np.count_nonzero(values >= self.high - tolerance))
        self.hist += np.histogram(values, bins=self.edges)[0]

    def quantile(self, probability: float) -> float | None:
        if self.count == 0:
            return None
        if probability <= 0:
            return self.minimum
        if probability >= 1:
            return self.maximum
        target = int(np.ceil(probability * self.count))
        index = int(np.searchsorted(np.cumsum(self.hist), target, side="left"))
        index = min(index, self.bins - 1)
        return float((self.edges[index] + self.edges[index + 1]) / 2.0)

    def row(self, scope: str, feature: str) -> dict[str, object]:
        mean = self.total / self.count if self.count else None
        variance = (
            max(self.total_sq / self.count - mean * mean, 0.0)
            if self.count and mean is not None
            else None
        )
        return {
            "scope": scope,
            "feature": feature,
            "count": self.count,
            "mean": mean,
            "std": float(np.sqrt(variance)) if variance is not None else None,
            "min": None if self.count == 0 else self.minimum,
            "p001": self.quantile(0.001),
            "p01": self.quantile(0.01),
            "p05": self.quantile(0.05),
            "p50": self.quantile(0.50),
            "p95": self.quantile(0.95),
            "p99": self.quantile(0.99),
            "p999": self.quantile(0.999),
            "max": None if self.count == 0 else self.maximum,
            "zero_rate": self.zero_count / self.count if self.count else None,
            "low_clip_rate": self.low_clip_count / self.count if self.count else None,
            "high_clip_rate": self.high_clip_count / self.count if self.count else None,
        }


def _chunks(indices: np.ndarray, size: int = DATE_CHUNK) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), size):
        yield indices[start : start + size]


def _exact_stats(
    values: np.ndarray,
    *,
    period: str,
    metric: str,
    horizon: int,
    unit: str,
) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "period": period,
            "metric": metric,
            "horizon_minutes": horizon,
            "unit": unit,
            "count": 0,
        }
    if not np.isfinite(values).all():
        raise ValueError(
            f"Non-finite values in {metric}, horizon={horizon}, period={period}"
        )
    quantiles = np.quantile(
        values,
        [0.0, 0.001, 0.01, 0.05, 0.50, 0.95, 0.99, 0.999, 1.0],
    )
    return {
        "period": period,
        "metric": metric,
        "horizon_minutes": horizon,
        "unit": unit,
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(quantiles[0]),
        "p001": float(quantiles[1]),
        "p01": float(quantiles[2]),
        "p05": float(quantiles[3]),
        "p50": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "p99": float(quantiles[6]),
        "p999": float(quantiles[7]),
        "max": float(quantiles[8]),
        "abs_gt_5_rate": float(np.mean(np.abs(values) > 5.0))
        if unit == "volatility units"
        else None,
        "abs_gt_10_rate": float(np.mean(np.abs(values) > 10.0))
        if unit == "volatility units"
        else None,
    }


def _collect_masked(
    array: np.ndarray,
    mask: np.ndarray,
    date_indices: np.ndarray,
    horizon_index: int,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for chunk in _chunks(date_indices, 32):
        chunk_mask = np.asarray(mask[chunk, :, :, horizon_index], dtype=bool)
        values = np.asarray(array[chunk, :, :, horizon_index], dtype=np.float32)[
            chunk_mask
        ]
        if values.size:
            pieces.append(values.astype(np.float64, copy=False) * scale)
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float64)


def main() -> None:
    created_at = datetime.now(timezone.utc)
    features_dir = Path(CANONICAL_OUTPUT_POINTER.read_text(encoding="utf-8").strip())
    if not features_dir.is_dir():
        raise FileNotFoundError(
            f"Canonical feature pointer resolves to missing directory: {features_dir}"
        )

    output_dir = AUDIT_BASE / f"m1_features_v1_audit_{created_at:%Y%m%dT%H%M%S%fZ}"
    output_dir.mkdir(parents=True, exist_ok=False)

    manifest = json.loads((features_dir / "manifest.json").read_text(encoding="utf-8"))
    date_index = pl.read_parquet(features_dir / "date_index.parquet")
    equity_index = pl.read_parquet(features_dir / "equity_index.parquet")
    context_index = pl.read_parquet(features_dir / "context_index.parquet")
    if tuple(context_index.get_column("symbol").to_list()) != CONTEXT_SYMBOLS:
        raise ValueError("Context index order does not match the feature contract")
    sample_index = pl.read_parquet(features_dir / "sample_index.parquet")
    daily_audit = pl.read_parquet(features_dir / "daily_audit.parquet")

    arrays = {
        name: np.load(features_dir / name, mmap_mode="r", allow_pickle=False)
        for name in (
            "equity_features.npy",
            "equity_slow.npy",
            "equity_membership.npy",
            "equity_data_ready.npy",
            "context_features.npy",
            "context_slow.npy",
            "context_data_ready.npy",
            "raw_returns.npy",
            "targets.npy",
            "label_mask.npy",
            "cross_section_median.npy",
            "horizon_mask.npy",
        )
    }

    for filename, spec in manifest["outputs"].items():
        array = arrays[filename]
        if list(array.shape) != spec["shape"] or array.dtype.name != spec["dtype"]:
            raise ValueError(
                f"Manifest mismatch for {filename}: "
                f"actual={array.shape}/{array.dtype.name}, "
                f"manifest={spec['shape']}/{spec['dtype']}"
            )

    eligible_dates = np.sort(
        sample_index.get_column("date_idx").unique().to_numpy().astype(np.int64)
    )
    if eligible_dates.size == 0:
        raise ValueError("No feature-eligible dates in sample_index")
    expected_samples = int(eligible_dates.size * len(DECISION_EQUITY_INDICES))
    if sample_index.height != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} samples from eligible dates, "
            f"found {sample_index.height}"
        )
    if int(manifest["sample_count"]) != sample_index.height:
        raise ValueError("Manifest sample_count does not match sample_index")

    trade_dates = date_index.get_column("trade_date").to_list()
    years = np.asarray([trade_dates[index].year for index in range(len(trade_dates))])
    year_groups = {
        int(year): eligible_dates[years[eligible_dates] == year]
        for year in np.unique(years[eligible_dates])
    }

    equity_feature_stats = {
        "open_move_normalized": BoundedStats(-PRICE_FEATURE_CLIP, PRICE_FEATURE_CLIP),
        "high_move_normalized": BoundedStats(-PRICE_FEATURE_CLIP, PRICE_FEATURE_CLIP),
        "low_move_normalized": BoundedStats(-PRICE_FEATURE_CLIP, PRICE_FEATURE_CLIP),
        "close_move_normalized": BoundedStats(-PRICE_FEATURE_CLIP, PRICE_FEATURE_CLIP),
        "volume_surprise": BoundedStats(-VOLUME_FEATURE_CLIP, VOLUME_FEATURE_CLIP),
        "vol_regime": BoundedStats(-VOL_REGIME_CLIP, VOL_REGIME_CLIP),
    }
    yearly_close_stats = {
        year: BoundedStats(-PRICE_FEATURE_CLIP, PRICE_FEATURE_CLIP)
        for year in year_groups
    }
    yearly_volume_stats = {
        year: BoundedStats(-VOLUME_FEATURE_CLIP, VOLUME_FEATURE_CLIP)
        for year in year_groups
    }

    security_observed = np.zeros(equity_index.height, dtype=np.int64)
    security_possible = np.zeros(equity_index.height, dtype=np.int64)
    security_active_days = np.zeros(equity_index.height, dtype=np.int64)
    yearly_observed: dict[int, int] = {year: 0 for year in year_groups}
    yearly_possible: dict[int, int] = {year: 0 for year in year_groups}

    equity_features = arrays["equity_features.npy"]
    equity_slow = arrays["equity_slow.npy"]
    equity_membership = arrays["equity_membership.npy"]
    equity_ready = arrays["equity_data_ready.npy"]

    for year, indices in year_groups.items():
        for chunk in _chunks(indices):
            features = np.asarray(
                equity_features[chunk, :, :EQUITY_VISIBLE_MINUTES, :],
                dtype=np.float32,
            )
            active = np.asarray(
                equity_membership[chunk] & equity_ready[chunk],
                dtype=bool,
            )
            observed = features[..., 5] > 0.5
            use = observed & active[:, :, None]

            for channel, name in enumerate(
                (
                    "open_move_normalized",
                    "high_move_normalized",
                    "low_move_normalized",
                    "close_move_normalized",
                    "volume_surprise",
                )
            ):
                equity_feature_stats[name].update(features[..., channel][use])
            yearly_close_stats[year].update(features[..., 3][use])
            yearly_volume_stats[year].update(features[..., 4][use])

            slow_values = np.asarray(equity_slow[chunk, :, 0], dtype=np.float32)
            equity_feature_stats["vol_regime"].update(slow_values[active])

            observed_by_security = use.sum(axis=(0, 2), dtype=np.int64)
            possible_by_security = (
                active.sum(axis=0, dtype=np.int64) * EQUITY_VISIBLE_MINUTES
            )
            security_observed += observed_by_security
            security_possible += possible_by_security
            security_active_days += active.sum(axis=0, dtype=np.int64)
            yearly_observed[year] += int(observed_by_security.sum())
            yearly_possible[year] += int(possible_by_security.sum())

    feature_rows = [
        stats.row("equity_active", feature)
        for feature, stats in equity_feature_stats.items()
    ]

    context_features = arrays["context_features.npy"]
    context_slow = arrays["context_slow.npy"]
    context_ready = arrays["context_data_ready.npy"]
    context_density: dict[str, float] = {}

    for slot, symbol in enumerate(CONTEXT_SYMBOLS):
        dynamic_stats = {
            "open_move_normalized": BoundedStats(
                -PRICE_FEATURE_CLIP, PRICE_FEATURE_CLIP
            ),
            "high_move_normalized": BoundedStats(
                -PRICE_FEATURE_CLIP, PRICE_FEATURE_CLIP
            ),
            "low_move_normalized": BoundedStats(
                -PRICE_FEATURE_CLIP, PRICE_FEATURE_CLIP
            ),
            "close_move_normalized": BoundedStats(
                -PRICE_FEATURE_CLIP, PRICE_FEATURE_CLIP
            ),
            "volume_surprise": BoundedStats(-VOLUME_FEATURE_CLIP, VOLUME_FEATURE_CLIP),
        }
        vol_stats = BoundedStats(-VOL_REGIME_CLIP, VOL_REGIME_CLIP)
        observed_count = 0
        possible_count = 0

        for chunk in _chunks(eligible_dates):
            features = np.asarray(
                context_features[chunk, slot, :CONTEXT_VISIBLE_MINUTES, :],
                dtype=np.float32,
            )
            ready = np.asarray(context_ready[chunk, slot], dtype=bool)
            observed = features[..., 5] > 0.5
            use = observed & ready[:, None]
            for channel, name in enumerate(dynamic_stats):
                dynamic_stats[name].update(features[..., channel][use])
            vol_stats.update(
                np.asarray(context_slow[chunk, slot, 0], dtype=np.float32)[ready]
            )
            observed_count += int(use.sum())
            possible_count += int(ready.sum()) * CONTEXT_VISIBLE_MINUTES

        for feature, stats in dynamic_stats.items():
            row = stats.row(f"context:{symbol}", feature)
            feature_rows.append(row)
        vol_row = vol_stats.row(f"context:{symbol}", "vol_regime")
        feature_rows.append(vol_row)

        if slot >= 2:
            prior_stats = BoundedStats(-1.0, 3.0, bins=1600)
            expiry_stats = BoundedStats(0.0, 1.0, bins=1000)
            ready_values = np.asarray(context_ready[eligible_dates, slot], dtype=bool)
            prior_stats.update(
                np.asarray(context_slow[eligible_dates, slot, 1], dtype=np.float32)[
                    ready_values
                ]
            )
            expiry_stats.update(
                np.asarray(context_slow[eligible_dates, slot, 2], dtype=np.float32)[
                    ready_values
                ]
            )
            for feature, stats in (
                ("prior_rate_level_scaled", prior_stats),
                ("time_to_expiry_scaled", expiry_stats),
            ):
                row = stats.row(f"context:{symbol}", feature)
                feature_rows.append(row)

        context_density[symbol] = (
            observed_count / possible_count if possible_count else 0.0
        )

    target_rows: list[dict[str, object]] = []
    yearly_rows: dict[int, dict[str, object]] = {}
    label_mask = arrays["label_mask.npy"]
    targets = arrays["targets.npy"]
    raw_returns = arrays["raw_returns.npy"]
    cross_section_median = arrays["cross_section_median.npy"]
    horizon_mask = arrays["horizon_mask.npy"]
    security_label_counts = np.zeros(
        (equity_index.height, len(HORIZONS)), dtype=np.int64
    )

    daily = daily_audit.with_columns(pl.col("trade_date").dt.year().alias("year"))
    for year, indices in year_groups.items():
        daily_year = daily.filter(
            (pl.col("year") == year) & (pl.col("sample_count") > 0)
        )
        active_counts = daily_year.get_column("active_equity_count").to_numpy()
        yearly_rows[year] = {
            "year": year,
            "eligible_dates": len(indices),
            "sample_count": int(
                sample_index.filter(pl.col("trade_date").dt.year() == year).height
            ),
            "active_equities_min": int(active_counts.min()),
            "active_equities_median": float(np.median(active_counts)),
            "active_equities_mean": float(active_counts.mean()),
            "active_equities_max": int(active_counts.max()),
            "equity_observed_input_fraction": (
                yearly_observed[year] / yearly_possible[year]
                if yearly_possible[year]
                else 0.0
            ),
            "close_move_mean": yearly_close_stats[year].row("", "")["mean"],
            "close_move_std": yearly_close_stats[year].row("", "")["std"],
            "volume_surprise_mean": yearly_volume_stats[year].row("", "")["mean"],
            "volume_surprise_std": yearly_volume_stats[year].row("", "")["std"],
            "volume_surprise_p50": yearly_volume_stats[year].quantile(0.5),
            "volume_surprise_p99": yearly_volume_stats[year].quantile(0.99),
        }

    for horizon_index, horizon in enumerate(HORIZONS):
        overall_target_parts: list[np.ndarray] = []
        overall_raw_parts: list[np.ndarray] = []
        overall_median_parts: list[np.ndarray] = []

        for year, indices in year_groups.items():
            year_targets = _collect_masked(targets, label_mask, indices, horizon_index)
            year_raw_bps = _collect_masked(
                raw_returns,
                label_mask,
                indices,
                horizon_index,
                scale=10_000.0,
            )
            year_horizon_mask = np.asarray(
                horizon_mask[indices, :, horizon_index], dtype=bool
            )
            year_medians_bps = (
                np.asarray(
                    cross_section_median[indices, :, horizon_index], dtype=np.float64
                )[year_horizon_mask]
                * 10_000.0
            )

            target_rows.append(
                _exact_stats(
                    year_targets,
                    period=str(year),
                    metric="target",
                    horizon=horizon,
                    unit="volatility units",
                )
            )
            target_rows.append(
                _exact_stats(
                    year_raw_bps,
                    period=str(year),
                    metric="raw_return",
                    horizon=horizon,
                    unit="basis points",
                )
            )
            target_rows.append(
                _exact_stats(
                    year_medians_bps,
                    period=str(year),
                    metric="cross_section_median",
                    horizon=horizon,
                    unit="basis points",
                )
            )

            mask_year = np.asarray(label_mask[indices, :, :, horizon_index], dtype=bool)
            security_label_counts[:, horizon_index] += mask_year.sum(
                axis=(0, 2), dtype=np.int64
            )
            active_opportunities = (
                np.asarray(
                    equity_membership[indices] & equity_ready[indices],
                    dtype=bool,
                ).sum(axis=1, dtype=np.int64)
                * len(DECISION_EQUITY_INDICES)
            ).sum(dtype=np.int64)
            yearly_rows[year][f"target_{horizon}_mean"] = (
                float(year_targets.mean()) if year_targets.size else None
            )
            yearly_rows[year][f"target_{horizon}_std"] = (
                float(year_targets.std()) if year_targets.size else None
            )
            yearly_rows[year][f"target_{horizon}_p01"] = (
                float(np.quantile(year_targets, 0.01)) if year_targets.size else None
            )
            yearly_rows[year][f"target_{horizon}_p99"] = (
                float(np.quantile(year_targets, 0.99)) if year_targets.size else None
            )
            yearly_rows[year][f"horizon_{horizon}_sample_coverage"] = float(
                year_horizon_mask.mean()
            )
            yearly_rows[year][f"label_{horizon}_opportunity_coverage"] = (
                int(mask_year.sum()) / int(active_opportunities)
                if active_opportunities
                else 0.0
            )

            overall_target_parts.append(year_targets)
            overall_raw_parts.append(year_raw_bps)
            overall_median_parts.append(year_medians_bps)

        overall_targets = (
            np.concatenate(overall_target_parts)
            if overall_target_parts
            else np.empty(0, dtype=np.float64)
        )
        overall_raw_bps = (
            np.concatenate(overall_raw_parts)
            if overall_raw_parts
            else np.empty(0, dtype=np.float64)
        )
        overall_medians_bps = (
            np.concatenate(overall_median_parts)
            if overall_median_parts
            else np.empty(0, dtype=np.float64)
        )
        target_rows.extend(
            (
                _exact_stats(
                    overall_targets,
                    period="ALL",
                    metric="target",
                    horizon=horizon,
                    unit="volatility units",
                ),
                _exact_stats(
                    overall_raw_bps,
                    period="ALL",
                    metric="raw_return",
                    horizon=horizon,
                    unit="basis points",
                ),
                _exact_stats(
                    overall_medians_bps,
                    period="ALL",
                    metric="cross_section_median",
                    horizon=horizon,
                    unit="basis points",
                ),
            )
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

    active_counts_all = (
        daily_audit.filter(pl.col("sample_count") > 0)
        .get_column("active_equity_count")
        .to_numpy()
    )
    equity_observed_fraction = (
        int(security_observed.sum()) / int(security_possible.sum())
        if security_possible.sum()
        else 0.0
    )
    sample_dates = [trade_dates[index] for index in eligible_dates]
    summary = {
        "created_at_utc": created_at.isoformat(),
        "canonical_features_dir": str(features_dir),
        "audit_output_dir": str(output_dir),
        "contract_version": manifest["contract_version"],
        "date_count": int(arrays["equity_features.npy"].shape[0]),
        "eligible_date_count": int(eligible_dates.size),
        "first_eligible_date": str(sample_dates[0]),
        "last_eligible_date": str(sample_dates[-1]),
        "sample_count": int(sample_index.height),
        "active_equities": {
            "min": int(active_counts_all.min()),
            "median": float(np.median(active_counts_all)),
            "mean": float(active_counts_all.mean()),
            "max": int(active_counts_all.max()),
        },
        "equity_observed_input_fraction": equity_observed_fraction,
        "context_observed_input_fraction": context_density,
        "security_observed_input_fraction": {
            "min": float(security_stats.get_column("observed_input_fraction").min()),
            "median": float(
                security_stats.get_column("observed_input_fraction").median()
            ),
            "max": float(security_stats.get_column("observed_input_fraction").max()),
        },
        "output_files": [
            "audit_summary.json",
            "feature_stats.csv",
            "target_stats.csv",
            "yearly_stats.csv",
            "security_stats.csv",
        ],
        "notes": [
            "Feature statistics use only point-in-time members that are data-ready on feature-eligible dates.",
            f"Dynamic-feature statistics use only model-visible minutes: equity [0,{EQUITY_VISIBLE_MINUTES}) and context [0,{CONTEXT_VISIBLE_MINUTES}).",
            "Quantiles for clipped feature channels are histogram approximations; target and return quantiles are exact.",
            "No model tensors are modified by this audit.",
        ],
    }

    pl.DataFrame(feature_rows).write_csv(output_dir / "feature_stats.csv")
    pl.DataFrame(target_rows).write_csv(output_dir / "target_stats.csv")
    pl.DataFrame(list(yearly_rows.values())).sort("year").write_csv(
        output_dir / "yearly_stats.csv"
    )
    security_stats.write_csv(output_dir / "security_stats.csv")
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
