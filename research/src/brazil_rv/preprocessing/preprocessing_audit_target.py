from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from .analyze_preprocessing import (
    TARGET_PARITY_ATOL,
    AuditArrays,
    AuditDates,
    DistributionAccumulator,
    target_group_metrics,
    validate_audit_indices,
)
from .contract import (
    DECISION_EQUITY_INDICES,
    HORIZONS,
    MIN_ACTIVE_EQUITIES,
    VOL_REGIME_CLIP,
)


@dataclass
class PairMoments:
    count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_xx: float = 0.0
    sum_yy: float = 0.0
    sum_xy: float = 0.0

    def update(self, left: np.ndarray, right: np.ndarray) -> None:
        x = np.asarray(left, dtype=np.float64)
        y = np.asarray(right, dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        self.count += int(x.size)
        self.sum_x += float(x.sum())
        self.sum_y += float(y.sum())
        self.sum_xx += float(x @ x)
        self.sum_yy += float(y @ y)
        self.sum_xy += float(x @ y)

    def correlation(self) -> float | None:
        if self.count < 2:
            return None
        mean_x = self.sum_x / self.count
        mean_y = self.sum_y / self.count
        var_x = self.sum_xx / self.count - mean_x**2
        var_y = self.sum_yy / self.count - mean_y**2
        denominator = math.sqrt(max(var_x, 0.0) * max(var_y, 0.0))
        if denominator == 0.0:
            return None
        covariance = self.sum_xy / self.count - mean_x * mean_y
        return covariance / denominator


def _target_stats(
    registry: dict[tuple[str, int, str, str], DistributionAccumulator],
    stage: str,
    horizon: int,
    scope_kind: str,
    scope_value: str,
) -> DistributionAccumulator:
    key = stage, horizon, scope_kind, scope_value
    if key not in registry:
        registry[key] = DistributionAccumulator(
            f"target:{stage}:{horizon}:{scope_kind}:{scope_value}"
        )
    return registry[key]


def run_target_audit(
    arrays: AuditArrays,
    dates: AuditDates,
    equity_index: pl.DataFrame,
    *,
    causal_sigma: np.ndarray | None = None,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    """Audit target coverage and exact pre-rank reconstruction on training only."""
    validate_audit_indices(dates.train, dates.trade_dates, allow_validation=False)
    raw_returns = arrays.array("raw_returns.npy", target_training_only=True)
    targets = arrays.array("targets.npy", target_training_only=True)
    label_mask = arrays.array("label_mask.npy", target_training_only=True)
    medians = arrays.array("cross_section_median.npy", target_training_only=True)
    horizon_mask = arrays.array("horizon_mask.npy", target_training_only=True)
    slow = arrays.array("equity_slow.npy")
    dynamic = arrays.array("equity_features.npy")
    membership = arrays.array("equity_membership.npy")
    readiness = arrays.array("equity_data_ready.npy")
    if causal_sigma is not None and (
        causal_sigma.ndim != 2
        or causal_sigma.shape[0] <= int(dates.train[-1])
        or causal_sigma.shape[1] != slow.shape[1]
    ):
        raise ValueError("Causal sigma must cover training dates and equities")

    stats: dict[tuple[str, int, str, str], DistributionAccumulator] = {}
    coverage_rows: list[dict[str, object]] = []
    security_valid = np.zeros((equity_index.height, len(HORIZONS)), dtype=np.int64)
    security_candidates = np.zeros_like(security_valid)
    security_prerank = {
        (slot, horizon_idx): DistributionAccumulator(
            f"target_security:{slot}:{HORIZONS[horizon_idx]}"
        )
        for slot in range(equity_index.height)
        for horizon_idx in range(len(HORIZONS))
    }
    pairs = {
        (left, right): PairMoments()
        for left in range(len(HORIZONS))
        for right in range(left + 1, len(HORIZONS))
    }
    parity_max = 0.0
    parity_mismatches = 0
    parity_count = 0
    sigma_clipped_count = 0
    total_ties = np.zeros(len(HORIZONS), dtype=np.int64)
    total_valid = np.zeros(len(HORIZONS), dtype=np.int64)
    degenerate = np.zeros(len(HORIZONS), dtype=np.int64)
    below_minimum = np.zeros(len(HORIZONS), dtype=np.int64)
    cross_section_count = np.zeros(len(HORIZONS), dtype=np.int64)

    for date_idx in dates.train:
        index = int(date_idx)
        trade_date = dates.trade_dates[index]
        year = trade_date.year
        active = np.asarray(membership[index] & readiness[index], dtype=bool)
        observed = np.asarray(dynamic[index, :, :, 5], dtype=bool)
        date_raw = np.asarray(raw_returns[index], dtype=np.float32)
        date_targets = np.asarray(targets[index], dtype=np.float32)
        date_mask = np.asarray(label_mask[index], dtype=bool)
        date_medians = np.asarray(medians[index], dtype=np.float32)
        date_horizon = np.asarray(horizon_mask[index], dtype=bool)
        vol_regime = np.asarray(slow[index, :, 0], dtype=np.float32)
        date_sigma = (
            None
            if causal_sigma is None
            else np.asarray(causal_sigma[index], dtype=np.float64)
        )
        for decision_idx, entry in enumerate(DECISION_EQUITY_INDICES):
            time_bin = str((entry - DECISION_EQUITY_INDICES[0]) // 30)
            for horizon_idx, horizon in enumerate(HORIZONS):
                exit_index = entry + horizon - 1
                candidate = active & observed[:, entry] & observed[:, exit_index]
                stored_valid = date_mask[:, decision_idx, horizon_idx]
                candidate_count = int(candidate.sum())
                valid_count = int(stored_valid.sum())
                usable = candidate_count >= MIN_ACTIVE_EQUITIES
                if usable != bool(date_horizon[decision_idx, horizon_idx]):
                    raise ValueError(
                        "Stored horizon mask disagrees with target candidates"
                    )
                if usable and not np.array_equal(candidate, stored_valid):
                    raise ValueError(
                        "Stored label mask disagrees with target candidates"
                    )
                if not usable and stored_valid.any():
                    raise ValueError("Labels exist below the minimum cross-section")
                security_candidates[:, horizon_idx] += candidate
                security_valid[:, horizon_idx] += stored_valid
                cross_section_count[horizon_idx] += 1
                below_minimum[horizon_idx] += int(not usable)
                group_degenerate = False
                unique_count = 0
                tie_fraction: float | None = None
                prerank_std: float | None = None
                if valid_count:
                    if date_sigma is not None and (
                        not np.isfinite(date_sigma[stored_valid]).all()
                        or np.any(date_sigma[stored_valid] <= 0.0)
                    ):
                        raise ValueError(
                            "Exact causal sigma is unavailable for a label"
                        )
                    median = float(date_medians[decision_idx, horizon_idx])
                    group = target_group_metrics(
                        date_raw[:, decision_idx, horizon_idx],
                        date_targets[:, decision_idx, horizon_idx],
                        stored_valid,
                        median,
                        vol_regime,
                        horizon,
                        causal_sigma=date_sigma,
                    )
                    prerank = np.asarray(group["prerank"], dtype=np.float64)
                    stored = date_targets[stored_valid, decision_idx, horizon_idx]
                    raw_bps = 10_000.0 * date_raw[
                        stored_valid, decision_idx, horizon_idx
                    ].astype(np.float64)
                    parity_max = max(parity_max, float(group["parity_max_abs_error"]))
                    parity_mismatches += int(group["parity_mismatch_count"])
                    parity_count += valid_count
                    unique_count = int(group["unique_count"])
                    tie_fraction = float(group["tie_fraction"])
                    group_degenerate = bool(group["degenerate"])
                    prerank_std = float(np.std(prerank))
                    total_ties[horizon_idx] += valid_count - unique_count
                    total_valid[horizon_idx] += valid_count
                    degenerate[horizon_idx] += int(group_degenerate)
                    sigma_clipped_count += int(
                        np.count_nonzero(
                            np.isclose(
                                np.abs(vol_regime[stored_valid]),
                                VOL_REGIME_CLIP,
                                atol=1e-6,
                            )
                        )
                    )
                    stages = (
                        ("raw_return_bps", raw_bps),
                        ("prerank_scaled_residual", prerank),
                        ("final_target", stored),
                    )
                    for stage, values in stages:
                        for scope_kind, scope_value in (
                            ("overall", "train"),
                            ("year", str(year)),
                            ("decision_time_bin_30m", time_bin),
                        ):
                            _target_stats(
                                stats, stage, horizon, scope_kind, scope_value
                            ).update(values)
                    valid_slots = np.flatnonzero(stored_valid)
                    for offset, slot in enumerate(valid_slots):
                        security_prerank[int(slot), horizon_idx].update(
                            np.asarray([prerank[offset]])
                        )
                coverage_rows.append(
                    {
                        "trade_date": trade_date,
                        "date_idx": index,
                        "year": year,
                        "decision_idx": decision_idx,
                        "decision_time_bin_30m": time_bin,
                        "horizon_minutes": horizon,
                        "candidate_count": candidate_count,
                        "valid_label_count": valid_count,
                        "valid_fraction_of_candidates": (
                            valid_count / candidate_count if candidate_count else 0.0
                        ),
                        "model_usable": usable,
                        "below_minimum_usable_size": not usable,
                        "degenerate": group_degenerate,
                        "unique_target_count": unique_count,
                        "tie_fraction": tie_fraction,
                        "prerank_scaled_residual_std": prerank_std,
                    }
                )

        flat_target = date_targets.reshape(-1, len(HORIZONS))
        flat_mask = date_mask.reshape(-1, len(HORIZONS))
        for (left, right), moments in pairs.items():
            pair_valid = flat_mask[:, left] & flat_mask[:, right]
            moments.update(
                flat_target[pair_valid, left], flat_target[pair_valid, right]
            )

    if sigma_clipped_count and causal_sigma is None:
        raise ValueError(
            "Exact pre-rank target reconstruction is unavailable because a causal "
            f"volatility state is clipped for {sigma_clipped_count} valid labels"
        )
    if parity_mismatches:
        raise ValueError(
            f"Target reconstruction failed parity for {parity_mismatches} values; "
            f"maximum absolute error={parity_max}"
        )
    stat_rows = [
        {
            "stage": stage,
            "horizon_minutes": horizon,
            "scope_kind": scope_kind,
            "scope_value": scope_value,
            **accumulator.row(),
        }
        for (stage, horizon, scope_kind, scope_value), accumulator in sorted(
            stats.items()
        )
    ]
    security_metadata = equity_index.sort("equity_slot").to_dicts()
    security_rows: list[dict[str, object]] = []
    for slot, metadata in enumerate(security_metadata):
        for horizon_idx, horizon in enumerate(HORIZONS):
            candidate_count = int(security_candidates[slot, horizon_idx])
            valid_count = int(security_valid[slot, horizon_idx])
            security_rows.append(
                {
                    "equity_slot": slot,
                    "security_id": metadata["security_id"],
                    "latest_ticker": metadata["latest_ticker"],
                    "horizon_minutes": horizon,
                    "candidate_count": candidate_count,
                    "valid_label_count": valid_count,
                    "valid_label_fraction": (
                        valid_count / candidate_count if candidate_count else 0.0
                    ),
                    **{
                        f"prerank_{key}": value
                        for key, value in security_prerank[slot, horizon_idx]
                        .row()
                        .items()
                        if key
                        in {
                            "valid_count",
                            "mean",
                            "std",
                            "median",
                            "mad",
                            "p01",
                            "p99",
                        }
                    },
                }
            )

    coverage_frame = pl.DataFrame(coverage_rows)
    coverage_by_year = (
        coverage_frame.group_by("year", "horizon_minutes")
        .agg(
            pl.col("valid_label_count").sum(),
            pl.col("candidate_count").sum(),
            pl.col("model_usable").mean().alias("usable_cross_section_fraction"),
            pl.col("prerank_scaled_residual_std").mean().alias("mean_prerank_std"),
        )
        .with_columns(
            (pl.col("valid_label_count") / pl.col("candidate_count")).alias(
                "valid_label_fraction"
            )
        )
        .sort("year", "horizon_minutes")
        .to_dicts()
    )
    coverage_by_time = (
        coverage_frame.group_by("decision_time_bin_30m", "horizon_minutes")
        .agg(
            pl.col("valid_label_count").sum(),
            pl.col("candidate_count").sum(),
            pl.col("model_usable").mean().alias("usable_cross_section_fraction"),
            pl.col("prerank_scaled_residual_std").mean().alias("mean_prerank_std"),
        )
        .with_columns(
            (pl.col("valid_label_count") / pl.col("candidate_count")).alias(
                "valid_label_fraction"
            )
        )
        .sort("decision_time_bin_30m", "horizon_minutes")
        .to_dicts()
    )
    cross_section_distributions: list[dict[str, object]] = []
    security_summaries: list[dict[str, object]] = []
    for horizon in HORIZONS:
        counts = coverage_frame.filter(pl.col("horizon_minutes") == horizon)[
            "valid_label_count"
        ].to_numpy()
        quantiles = np.quantile(counts, (0.01, 0.05, 0.5, 0.95, 0.99))
        cross_section_distributions.append(
            {
                "horizon_minutes": horizon,
                "cross_section_count": int(counts.size),
                "mean": float(counts.mean()),
                "std": float(counts.std()),
                "p01": float(quantiles[0]),
                "p05": float(quantiles[1]),
                "median": float(quantiles[2]),
                "p95": float(quantiles[3]),
                "p99": float(quantiles[4]),
            }
        )
        horizon_rows = [
            row for row in security_rows if row["horizon_minutes"] == horizon
        ]
        for metric in ("valid_label_fraction", "prerank_std"):
            values = np.asarray(
                [
                    float(row[metric])
                    for row in horizon_rows
                    if row.get(metric) is not None
                    and (metric != "valid_label_fraction" or row["candidate_count"])
                ],
                dtype=np.float64,
            )
            if values.size:
                p10, median, p90 = np.quantile(values, (0.1, 0.5, 0.9))
                security_summaries.append(
                    {
                        "horizon_minutes": horizon,
                        "metric": metric,
                        "security_count": int(values.size),
                        "p10": float(p10),
                        "median": float(median),
                        "p90": float(p90),
                    }
                )
    worst_coverage = sorted(
        [row for row in security_rows if row["candidate_count"]],
        key=lambda row: float(row["valid_label_fraction"]),
    )[:10]
    finite_dispersion = [
        row for row in security_rows if row.get("prerank_std") is not None
    ]
    worst_dispersion = sorted(
        finite_dispersion,
        key=lambda row: abs(math.log(max(float(row["prerank_std"]), 1e-12))),
        reverse=True,
    )[:10]
    summary: dict[str, object] = {
        "split": "train",
        "train_start": str(dates.trade_dates[int(dates.train[0])]),
        "train_end": str(dates.trade_dates[int(dates.train[-1])]),
        "validation_targets_loaded": False,
        "test_indices_accessed": False,
        "minimum_usable_cross_section": MIN_ACTIVE_EQUITIES,
        "target_stages": [
            "raw log return (reported in basis points)",
            "contemporaneous cross-sectional median residual",
            "causal equity volatility and sqrt(horizon) scaling",
            "centered average midrank",
        ],
        "distribution_rows": stat_rows,
        "coverage_by_training_year": coverage_by_year,
        "coverage_by_decision_time_bin_30m": coverage_by_time,
        "cross_section_valid_count_distribution": cross_section_distributions,
        "security_distribution_summary": security_summaries,
        "horizon_summary": [
            {
                "horizon_minutes": horizon,
                "candidate_count": int(security_candidates[:, index].sum()),
                "valid_label_count": int(total_valid[index]),
                "valid_label_fraction": (
                    float(total_valid[index] / security_candidates[:, index].sum())
                    if security_candidates[:, index].sum()
                    else None
                ),
                "tie_fraction": (
                    float(total_ties[index] / total_valid[index])
                    if total_valid[index]
                    else None
                ),
                "degenerate_cross_section_fraction": (
                    float(degenerate[index] / cross_section_count[index])
                    if cross_section_count[index]
                    else None
                ),
                "below_minimum_cross_section_fraction": (
                    float(below_minimum[index] / cross_section_count[index])
                    if cross_section_count[index]
                    else None
                ),
            }
            for index, horizon in enumerate(HORIZONS)
        ],
        "target_parity": {
            "checked_value_count": parity_count,
            "mismatch_count": parity_mismatches,
            "maximum_absolute_error": parity_max,
            "absolute_tolerance": TARGET_PARITY_ATOL,
        },
        "vol_regime_values_at_clip_used_for_reconstruction": sigma_clipped_count,
        "causal_sigma_source": (
            "accepted_raw_identity_segments_reconstructed_in_audit_memory"
            if causal_sigma is not None
            else "invertible_unclipped_stored_vol_regime"
        ),
        "cross_horizon_final_target_correlation": [
            {
                "left_horizon": HORIZONS[left],
                "right_horizon": HORIZONS[right],
                "paired_count": moments.count,
                "correlation": moments.correlation(),
            }
            for (left, right), moments in pairs.items()
        ],
        "worst_security_coverage": worst_coverage,
        "worst_security_prerank_dispersion": worst_dispersion,
    }
    return summary, coverage_rows, security_rows
