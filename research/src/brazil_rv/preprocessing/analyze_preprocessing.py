from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from brazil_rv.modeling.contract import (
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
)

from .contract import (
    BETA_MIN_PAIRED_SESSIONS,
    DYNAMIC_CHANNEL_COUNT,
    PRICE_VOL_REFERENCE,
)
from .transforms import causal_exposure_betas, centered_midranks

AUDIT_SEED = 20260815
QUANTILE_SAMPLE_CAPACITY = 4096
PAIR_THRESHOLD = 0.95
ROBUST_SHIFT_FLOOR = 1e-6
TARGET_PARITY_ATOL = 1e-6
PERSISTENCE_LEVELS = (0.5, 0.8, 0.95)

OUTPUT_FILES = (
    "audit_manifest.json",
    "normalization_effectiveness.csv",
    "normalization_security_summary.csv",
    "target_audit.json",
    "target_coverage.csv",
    "target_security_summary.csv",
    "redundancy_pairs.csv",
    "redundancy_pairwise.csv",
    "redundancy_feature_summary.csv",
    "redundancy_family_pca.csv",
    "train_validation_shift.csv",
    "di_contract_coverage.csv",
    "di_factor_fit_summary.csv",
    "di_factor_beta_summary.csv",
    "di_feasibility.json",
    "preprocessing_audit_summary.json",
    "preprocessing_audit_summary.md",
)

TARGET_ARRAYS = frozenset(
    {
        "raw_returns.npy",
        "targets.npy",
        "label_mask.npy",
        "cross_section_median.npy",
        "horizon_mask.npy",
    }
)

BASE_CHANNELS = (0, 1, 2, 3, 4)
EQUITY_DECISION_DYNAMIC_CHANNELS = tuple(range(6, DYNAMIC_CHANNEL_COUNT))
CONTEXT_DECISION_DYNAMIC_CHANNELS = tuple(range(6, 16))
VOLATILITY_STATE_CHANNELS = (0, 5, 9, 10, 11)
VOLUME_STATE_CHANNELS = (6, 12, 13, 14)

PCA_FAMILIES: dict[str, tuple[int, ...]] = {
    "dynamic_price_moves": (0, 1, 2, 3),
    "dynamic_volume": (4, 13),
    "dynamic_returns": (6, 7, 8, 9),
    "dynamic_realized_volatility": (10, 11, 12),
    "dynamic_market_return_summary": (16, 17, 18, 19, 20, 21),
    "dynamic_cross_section_ranks": (22, 23, 24, 25),
    "slow_returns": (1, 2, 3, 4, 7, 8),
    "slow_volatility": (0, 5, 9, 10, 11),
    "slow_volume_liquidity": (6, 12, 13, 14, 15, 16),
    "slow_cross_section_ranks": (17, 18, 19),
    "slow_exposure_betas": (20, 21, 22, 23, 24, 25),
    "slow_calendar": (26, 27, 28, 29),
    "slow_rate_state": (30, 31),
}


def _stable_seed(label: str) -> int:
    digest = hashlib.sha256(f"{AUDIT_SEED}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


@dataclass
class DistributionAccumulator:
    """Exact moments plus a bounded deterministic sample for robust quantiles."""

    label: str
    lower_clip: float | None = None
    upper_clip: float | None = None
    sample_capacity: int = QUANTILE_SAMPLE_CAPACITY
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    zero_count: int = 0
    lower_clip_count: int = 0
    upper_clip_count: int = 0
    possible_count: int = 0
    _sample: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64), repr=False
    )
    _priorities: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64), repr=False
    )
    _update_count: int = 0

    def update(
        self, values: NDArray[np.floating], *, possible_count: int | None = None
    ) -> None:
        array = np.asarray(values, dtype=np.float64).ravel()
        if possible_count is None:
            possible_count = int(array.size)
        if possible_count < array.size:
            raise ValueError("possible_count cannot be smaller than valid values")
        self.possible_count += int(possible_count)
        if not array.size:
            return
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite audit value in {self.label}")
        self.count += int(array.size)
        self.total += float(array.sum(dtype=np.float64))
        self.total_sq += float(np.square(array).sum(dtype=np.float64))
        self.zero_count += int(np.count_nonzero(array == 0.0))
        if self.lower_clip is not None:
            self.lower_clip_count += int(
                np.count_nonzero(np.isclose(array, self.lower_clip, atol=1e-6))
            )
        if self.upper_clip is not None:
            self.upper_clip_count += int(
                np.count_nonzero(np.isclose(array, self.upper_clip, atol=1e-6))
            )
        self._update_sample(array)

    def _update_sample(self, array: NDArray[np.float64]) -> None:
        self._update_count += 1
        take = min(array.size, 512)
        positions = np.linspace(0, array.size - 1, take, dtype=np.int64)
        candidates = array[positions]
        rng = np.random.default_rng(_stable_seed(f"{self.label}:{self._update_count}"))
        priorities = rng.random(take)
        values = np.concatenate((self._sample, candidates))
        keys = np.concatenate((self._priorities, priorities))
        if values.size > self.sample_capacity:
            keep = np.argpartition(keys, self.sample_capacity - 1)[
                : self.sample_capacity
            ]
            values = values[keep]
            keys = keys[keep]
        self._sample = values
        self._priorities = keys

    def row(self) -> dict[str, object]:
        if not self.count:
            return {
                "valid_count": 0,
                "possible_count": self.possible_count,
                "mean": None,
                "std": None,
                "median": None,
                "mad": None,
                "p01": None,
                "p05": None,
                "p95": None,
                "p99": None,
                "zero_fraction": None,
                "lower_clipping_fraction": None,
                "upper_clipping_fraction": None,
                "observed_fraction": (0.0 if self.possible_count else None),
                "quantile_sample_count": 0,
            }
        mean = self.total / self.count
        variance = max(self.total_sq / self.count - mean * mean, 0.0)
        sample = self._sample
        median = float(np.median(sample))
        quantiles = np.quantile(sample, (0.01, 0.05, 0.95, 0.99))
        return {
            "valid_count": self.count,
            "possible_count": self.possible_count,
            "mean": mean,
            "std": float(np.sqrt(variance)),
            "median": median,
            "mad": float(np.median(np.abs(sample - median))),
            "p01": float(quantiles[0]),
            "p05": float(quantiles[1]),
            "p95": float(quantiles[2]),
            "p99": float(quantiles[3]),
            "zero_fraction": self.zero_count / self.count,
            "lower_clipping_fraction": self.lower_clip_count / self.count,
            "upper_clipping_fraction": self.upper_clip_count / self.count,
            "observed_fraction": (
                self.count / self.possible_count if self.possible_count else None
            ),
            "quantile_sample_count": int(sample.size),
        }


def distribution_metrics(
    values: NDArray[np.floating],
    valid: NDArray[np.bool_],
    *,
    lower_clip: float | None = None,
    upper_clip: float | None = None,
) -> dict[str, object]:
    values = np.asarray(values)
    mask = np.asarray(valid, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError("values and validity mask must align")
    stats = DistributionAccumulator("synthetic", lower_clip, upper_clip)
    stats.update(values[mask], possible_count=mask.size)
    return stats.row()


@dataclass(frozen=True)
class AuditDates:
    trade_dates: tuple[date, ...]
    train: NDArray[np.int64]
    validation: NDArray[np.int64]

    @classmethod
    def from_frame(cls, frame: pl.DataFrame) -> AuditDates:
        ordered = frame.sort("date_idx")
        trade_dates = tuple(ordered.get_column("trade_date").to_list())
        train = np.asarray(
            [
                index
                for index, value in enumerate(trade_dates)
                if TRAIN_START <= value <= TRAIN_END
            ],
            dtype=np.int64,
        )
        validation = np.asarray(
            [
                index
                for index, value in enumerate(trade_dates)
                if VALIDATION_START <= value <= VALIDATION_END
            ],
            dtype=np.int64,
        )
        validate_audit_indices(train, trade_dates, allow_validation=False)
        validate_audit_indices(validation, trade_dates, allow_validation=True)
        if not train.size or not validation.size:
            raise ValueError("Training and validation audit windows must be non-empty")
        return cls(trade_dates, train, validation)


def validate_audit_indices(
    indices: NDArray[np.integer],
    trade_dates: Sequence[date],
    *,
    allow_validation: bool,
) -> None:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("Audit indices must be one-dimensional")
    if indices.size and (indices.min() < 0 or indices.max() >= len(trade_dates)):
        raise ValueError("Audit index falls outside the date axis")
    boundary = VALIDATION_END if allow_validation else TRAIN_END
    invalid = [
        trade_dates[int(index)]
        for index in indices
        if trade_dates[int(index)] > boundary
    ]
    if invalid:
        raise ValueError(f"Audit index enters a prohibited split: {min(invalid)}")
    if any(trade_dates[int(index)] >= TEST_START for index in indices):
        raise ValueError("Held-out test indices are forbidden in preprocessing audits")


class AuditArrays:
    """Memory-map store arrays while enforcing target-specific access rules."""

    def __init__(self, store: Path, dates: AuditDates):
        self.store = store
        self.dates = dates
        self._arrays: dict[str, NDArray[np.generic]] = {}

    def array(
        self, filename: str, *, target_training_only: bool = False
    ) -> NDArray[np.generic]:
        if filename in TARGET_ARRAYS and not target_training_only:
            raise ValueError(
                f"{filename} may be opened only by the training-only target audit"
            )
        if filename not in self._arrays:
            self._arrays[filename] = np.load(
                self.store / filename, mmap_mode="r", allow_pickle=False
            )
        return self._arrays[filename]

    def target_slice(
        self, filename: str, indices: NDArray[np.integer]
    ) -> NDArray[np.generic]:
        if filename not in TARGET_ARRAYS:
            raise ValueError(f"{filename} is not a target array")
        validate_audit_indices(indices, self.dates.trade_dates, allow_validation=False)
        return self.array(filename, target_training_only=True)[indices]


def reconstruct_prerank_target(
    raw_return: NDArray[np.floating],
    median: float,
    vol_regime: NDArray[np.floating],
    horizon: int,
    *,
    causal_sigma: NDArray[np.floating] | None = None,
) -> NDArray[np.float64]:
    sigma = (
        np.asarray(causal_sigma, dtype=np.float64)
        if causal_sigma is not None
        else PRICE_VOL_REFERENCE * np.exp(np.asarray(vol_regime, dtype=np.float64))
    )
    return (np.asarray(raw_return, dtype=np.float64) - median) / (
        sigma * math.sqrt(horizon)
    )


def target_group_metrics(
    raw_return: NDArray[np.floating],
    stored_target: NDArray[np.floating],
    valid: NDArray[np.bool_],
    median: float,
    vol_regime: NDArray[np.floating],
    horizon: int,
    *,
    causal_sigma: NDArray[np.floating] | None = None,
) -> dict[str, object]:
    mask = np.asarray(valid, dtype=bool)
    values = reconstruct_prerank_target(
        np.asarray(raw_return)[mask],
        median,
        np.asarray(vol_regime)[mask],
        horizon,
        causal_sigma=(None if causal_sigma is None else np.asarray(causal_sigma)[mask]),
    )
    reconstructed = centered_midranks(values)
    stored = np.asarray(stored_target, dtype=np.float32)[mask]
    difference = np.abs(reconstructed.astype(np.float64) - stored.astype(np.float64))
    unique = np.unique(stored).size
    return {
        "valid_count": int(mask.sum()),
        "prerank": values,
        "reconstructed": reconstructed,
        "parity_max_abs_error": float(difference.max()) if difference.size else 0.0,
        "parity_mismatch_count": int(np.count_nonzero(difference > TARGET_PARITY_ATOL)),
        "unique_count": int(unique),
        "tie_fraction": (1.0 - unique / stored.size) if stored.size else None,
        "degenerate": bool(stored.size and unique <= 1),
    }


def average_ranks(values: NDArray[np.floating]) -> NDArray[np.float64]:
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman_correlation(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
    valid: NDArray[np.bool_] | None = None,
) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan")
    x_rank = average_ranks(x)
    y_rank = average_ranks(y)
    x_rank -= x_rank.mean()
    y_rank -= y_rank.mean()
    denominator = float(np.sqrt((x_rank @ x_rank) * (y_rank @ y_rank)))
    return float(x_rank @ y_rank / denominator) if denominator else float("nan")


def redundancy_tables(
    values: NDArray[np.floating],
    valid: NDArray[np.bool_],
    feature_names: Sequence[str],
    *,
    minimum_absolute_correlation: float = PAIR_THRESHOLD,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    matrix = np.asarray(values, dtype=np.float64)
    masks = np.asarray(valid, dtype=bool)
    if matrix.shape != masks.shape or matrix.shape[1] != len(feature_names):
        raise ValueError("Redundancy inputs do not share the feature axis")
    pairs: list[dict[str, object]] = []
    maximum = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    for left in range(matrix.shape[1]):
        for right in range(left + 1, matrix.shape[1]):
            pair_mask = masks[:, left] & masks[:, right]
            rho = spearman_correlation(matrix[:, left], matrix[:, right], pair_mask)
            if math.isfinite(rho):
                absolute = abs(rho)
                maximum[left] = (
                    absolute
                    if not math.isfinite(maximum[left])
                    else max(maximum[left], absolute)
                )
                maximum[right] = (
                    absolute
                    if not math.isfinite(maximum[right])
                    else max(maximum[right], absolute)
                )
                if absolute >= minimum_absolute_correlation:
                    pairs.append(
                        {
                            "feature_left": feature_names[left],
                            "feature_right": feature_names[right],
                            "valid_pair_count": int(pair_mask.sum()),
                            "spearman_rho": rho,
                            "absolute_spearman_rho": absolute,
                        }
                    )
    pairs.sort(
        key=lambda row: (
            -float(row["absolute_spearman_rho"]),
            row["feature_left"],
            row["feature_right"],
        )
    )
    summaries: list[dict[str, object]] = []
    for index, name in enumerate(feature_names):
        feature_values = matrix[masks[:, index], index]
        std = float(np.std(feature_values)) if feature_values.size else None
        unique = np.unique(feature_values).size if feature_values.size else 0
        summaries.append(
            {
                "feature": name,
                "valid_count": int(feature_values.size),
                "applicable_fraction": float(masks[:, index].mean()),
                "std": std,
                "unique_count": int(unique),
                "near_constant": bool(
                    feature_values.size and (std is not None and std <= 1e-8)
                ),
                "effectively_unused": bool(feature_values.size == 0 or unique <= 1),
                "maximum_absolute_spearman": (
                    float(maximum[index]) if math.isfinite(maximum[index]) else None
                ),
            }
        )
    return pairs, summaries


def pca_summary(
    values: NDArray[np.floating], valid: NDArray[np.bool_]
) -> dict[str, object]:
    matrix = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if matrix.shape != mask.shape:
        raise ValueError("PCA values and masks must align")
    complete = mask.all(axis=1) & np.isfinite(matrix).all(axis=1)
    matrix = matrix[complete]
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return {
            "fit_row_count": int(matrix.shape[0]),
            "feature_count": int(matrix.shape[1]),
            "components_90": None,
            "components_95": None,
            "explained_variance_ratio": [],
        }
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    nonconstant = scale > 1e-12
    matrix = (matrix[:, nonconstant] - mean[nonconstant]) / scale[nonconstant]
    if matrix.shape[1] < 2:
        return {
            "fit_row_count": int(matrix.shape[0]),
            "feature_count": int(mask.shape[1]),
            "nonconstant_feature_count": int(matrix.shape[1]),
            "components_90": 1 if matrix.shape[1] else None,
            "components_95": 1 if matrix.shape[1] else None,
            "explained_variance_ratio": [1.0] if matrix.shape[1] else [],
        }
    singular = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
    variance = singular**2
    ratio = variance / variance.sum()
    cumulative = np.cumsum(ratio)
    return {
        "fit_row_count": int(matrix.shape[0]),
        "feature_count": int(mask.shape[1]),
        "nonconstant_feature_count": int(matrix.shape[1]),
        "components_90": int(np.searchsorted(cumulative, 0.90) + 1),
        "components_95": int(np.searchsorted(cumulative, 0.95) + 1),
        "explained_variance_ratio": ratio.tolist(),
    }


def ks_statistic(
    training: NDArray[np.floating], validation: NDArray[np.floating]
) -> float:
    left = np.sort(np.asarray(training, dtype=np.float64))
    right = np.sort(np.asarray(validation, dtype=np.float64))
    if not left.size or not right.size:
        return float("nan")
    combined = np.sort(np.concatenate((left, right)))
    left_cdf = np.searchsorted(left, combined, side="right") / left.size
    right_cdf = np.searchsorted(right, combined, side="right") / right.size
    return float(np.max(np.abs(left_cdf - right_cdf)))


def shift_metrics(
    training: NDArray[np.floating],
    validation: NDArray[np.floating],
    *,
    training_possible: int,
    validation_possible: int,
    lower_clip: float | None = None,
    upper_clip: float | None = None,
) -> dict[str, object]:
    train = np.asarray(training, dtype=np.float64)
    valid = np.asarray(validation, dtype=np.float64)
    if not np.isfinite(train).all() or not np.isfinite(valid).all():
        raise ValueError("Shift inputs must be finite observed values")
    training_observed = train.size / training_possible if training_possible else None
    validation_observed = (
        valid.size / validation_possible if validation_possible else None
    )
    availability_change = (
        validation_observed - training_observed
        if training_observed is not None and validation_observed is not None
        else None
    )
    if not train.size or not valid.size:
        return {
            "training_count": int(train.size),
            "validation_count": int(valid.size),
            "training_observed_fraction": training_observed,
            "validation_observed_fraction": validation_observed,
            "observed_fraction_change": availability_change,
            "availability_shift_magnitude": (
                abs(availability_change) if availability_change is not None else None
            ),
            "dominant_shift_component": (
                "availability" if availability_change is not None else None
            ),
        }
    train_mean = float(train.mean())
    valid_mean = float(valid.mean())
    train_std = float(train.std())
    valid_std = float(valid.std())
    train_median = float(np.median(train))
    valid_median = float(np.median(valid))
    train_mad = float(np.median(np.abs(train - train_median)))
    valid_mad = float(np.median(np.abs(valid - valid_median)))
    p01, p99 = np.quantile(train, (0.01, 0.99))

    def clip_fraction(values: NDArray[np.float64]) -> float:
        count = 0
        if lower_clip is not None:
            count += int(np.count_nonzero(np.isclose(values, lower_clip, atol=1e-6)))
        if upper_clip is not None:
            count += int(np.count_nonzero(np.isclose(values, upper_clip, atol=1e-6)))
        return count / values.size

    train_clip = clip_fraction(train)
    valid_clip = clip_fraction(valid)
    standardized_mean = abs(valid_mean - train_mean) / max(
        train_std, ROBUST_SHIFT_FLOOR
    )
    robust_median = abs(valid_median - train_median) / max(
        train_mad, ROBUST_SHIFT_FLOOR
    )
    ks = ks_statistic(train, valid)
    outside = float(np.mean((valid < p01) | (valid > p99)))
    components = {
        "center": max(standardized_mean, robust_median),
        "scale": abs(
            math.log(
                (valid_std + ROBUST_SHIFT_FLOOR) / (train_std + ROBUST_SHIFT_FLOOR)
            )
        ),
        "tails": max(ks, outside),
        "clipping": abs(valid_clip - train_clip),
        "availability": abs(availability_change or 0.0),
    }
    return {
        "training_count": int(train.size),
        "validation_count": int(valid.size),
        "training_mean": train_mean,
        "validation_mean": valid_mean,
        "training_std": train_std,
        "validation_std": valid_std,
        "training_median": train_median,
        "validation_median": valid_median,
        "training_mad": train_mad,
        "validation_mad": valid_mad,
        "absolute_standardized_mean_difference": standardized_mean,
        "absolute_robust_median_shift": robust_median,
        "robust_mad_floor": ROBUST_SHIFT_FLOOR,
        "ks_statistic": ks,
        "validation_outside_training_p01_p99_fraction": outside,
        "training_clipping_fraction": train_clip,
        "validation_clipping_fraction": valid_clip,
        "clipping_fraction_change": valid_clip - train_clip,
        "training_observed_fraction": training_observed,
        "validation_observed_fraction": validation_observed,
        "observed_fraction_change": availability_change,
        "center_shift_score": components["center"],
        "scale_shift_score": components["scale"],
        "tail_shift_score": components["tails"],
        "clipping_shift_magnitude": components["clipping"],
        "availability_shift_magnitude": components["availability"],
        "dominant_shift_component": max(components, key=components.get),
    }


@dataclass(frozen=True)
class DICurveFit:
    contract_count: int
    level: float
    tilt: float
    raw_maturity_design_condition_number: float
    maturity_span_years: float
    minimum_distinct_maturity_separation_years: float
    residual_rmse: float
    explained_variance_fraction: float
    curvature: float | None
    curvature_residual_rmse: float | None
    curvature_incremental_residual_reduction: float | None


def fit_di_curve(
    changes_bp: NDArray[np.floating],
    maturity_years: NDArray[np.floating],
    ready: NDArray[np.bool_],
) -> DICurveFit | None:
    changes = np.asarray(changes_bp, dtype=np.float64)
    maturity = np.asarray(maturity_years, dtype=np.float64)
    mask = np.asarray(ready, dtype=bool) & np.isfinite(changes) & np.isfinite(maturity)
    if changes.shape != maturity.shape or changes.shape != mask.shape:
        raise ValueError("DI changes, maturities, and readiness must align")
    if int(mask.sum()) < 3:
        return None
    y = changes[mask]
    x = maturity[mask]
    scale = float(x.std())
    if scale <= 0.0:
        return None
    z = (x - x.mean()) / scale
    design = np.column_stack((np.ones(z.size), z))
    raw_design = np.column_stack((np.ones(x.size), x))
    distinct_maturities = np.unique(x)
    minimum_separation = float(np.diff(distinct_maturities).min())
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    sse = float(residual @ residual)
    centered = y - y.mean()
    sst = float(centered @ centered)
    explained = 1.0 - sse / sst if sst > 0.0 else (1.0 if sse == 0.0 else 0.0)
    curvature: float | None = None
    curvature_rmse: float | None = None
    incremental: float | None = None
    if y.size == 4:
        quadratic = z**2
        projection, *_ = np.linalg.lstsq(design, quadratic, rcond=None)
        orthogonal = quadratic - design @ projection
        orthogonal_scale = float(np.sqrt(np.mean(orthogonal**2)))
        if orthogonal_scale > 0.0:
            orthogonal /= orthogonal_scale
            design_three = np.column_stack((design, orthogonal))
            three, *_ = np.linalg.lstsq(design_three, y, rcond=None)
            residual_three = y - design_three @ three
            sse_three = float(residual_three @ residual_three)
            curvature = float(three[2])
            curvature_rmse = float(np.sqrt(sse_three / y.size))
            incremental = (sse - sse_three) / sse if sse > 0.0 else 0.0
    return DICurveFit(
        contract_count=int(mask.sum()),
        level=float(coefficients[0]),
        tilt=float(coefficients[1]),
        raw_maturity_design_condition_number=float(np.linalg.cond(raw_design)),
        maturity_span_years=float(x.max() - x.min()),
        minimum_distinct_maturity_separation_years=minimum_separation,
        residual_rmse=float(np.sqrt(sse / y.size)),
        explained_variance_fraction=float(explained),
        curvature=curvature,
        curvature_residual_rmse=curvature_rmse,
        curvature_incremental_residual_reduction=incremental,
    )


def maturity_hull_intersection(
    maturity_years: NDArray[np.floating], ready: NDArray[np.bool_]
) -> dict[str, object]:
    maturity = np.asarray(maturity_years, dtype=np.float64)
    masks = np.asarray(ready, dtype=bool)
    if maturity.shape != masks.shape or maturity.ndim != 2:
        raise ValueError("Maturity hull inputs must be [observation, contract]")
    lower: list[float] = []
    upper: list[float] = []
    incomplete = 0
    for values, mask in zip(maturity, masks, strict=True):
        if int(mask.sum()) < 2:
            incomplete += 1
            continue
        lower.append(float(values[mask].min()))
        upper.append(float(values[mask].max()))
    intersection_min = max(lower) if lower else None
    intersection_max = min(upper) if upper else None
    feasible = bool(
        lower
        and incomplete == 0
        and intersection_min is not None
        and intersection_max is not None
        and intersection_min <= intersection_max
    )
    return {
        "observation_count": int(maturity.shape[0]),
        "insufficient_contract_observation_count": incomplete,
        "point_in_time_minimum_maturity_years": min(lower) if lower else None,
        "point_in_time_maximum_maturity_years": max(upper) if upper else None,
        "intersection_minimum_maturity_years": intersection_min,
        "intersection_maximum_maturity_years": intersection_max,
        "constant_maturity_without_extrapolation_full_interval": feasible,
    }


def beta_readiness(
    equity_valid: NDArray[np.bool_], factor_valid: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    equities = np.asarray(equity_valid, dtype=bool)
    factors = np.asarray(factor_valid, dtype=bool)
    if equities.ndim != 2 or factors.ndim != 2 or equities.shape[0] != factors.shape[0]:
        raise ValueError("Equity and factor validity must share the date axis")
    counts = np.zeros((equities.shape[1], factors.shape[1]), dtype=np.int32)
    ready = np.zeros((equities.shape[0], *counts.shape), dtype=bool)
    for date_idx in range(equities.shape[0]):
        ready[date_idx] = counts >= BETA_MIN_PAIRED_SESSIONS
        counts += equities[date_idx, :, None] & factors[date_idx, None, :]
    return ready


def causal_factor_betas(
    equity_change: NDArray[np.floating],
    equity_valid: NDArray[np.bool_],
    factor_change: NDArray[np.floating],
    factor_valid: NDArray[np.bool_],
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    betas = causal_exposure_betas(
        np.asarray(equity_change, dtype=np.float64),
        np.asarray(equity_valid, dtype=bool),
        np.asarray(factor_change, dtype=np.float64),
        np.asarray(factor_valid, dtype=bool),
    )
    return betas, beta_readiness(equity_valid, factor_valid)
