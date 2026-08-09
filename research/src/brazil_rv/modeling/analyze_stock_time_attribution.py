from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from brazil_rv.preprocessing.contract import (
    DECISION_EQUITY_INDICES,
    DECISION_GLOBAL_INDICES,
    DECISION_TIMES,
    DYNAMIC_CHANNELS,
    GLOBAL_CONTEXT_SYMBOLS,
    HORIZONS,
    LOCAL_CONTEXT_SYMBOLS,
    SLOW_CHANNELS,
)

from .analyze_context_ablation import (
    BOOTSTRAP_BLOCK_TRADING_DAYS,
    BOOTSTRAP_REPLICATIONS,
    BOOTSTRAP_SEED,
)
from .context_ablation import get_context_ablation
from .contract import (
    EXPECTED_DECISIONS_PER_DATE,
    MIN_IC_EQUITIES,
)
from .data import select_sample_split
from .metrics import average_ranks
from .process_lock import PRODUCTION_TRAINING_LOCK, exclusive_process_lock
from .stage3_context_addition import (
    STAGE3_LOGICAL_CONFIGURATION_ORDER,
    STAGE3_SEEDS,
    _reject_test_derived_metadata,
)
from .stock_time_cache import (
    adopt_or_infer_caches as _adopt_or_infer_caches,
    atomic_write_json as _atomic_write_json,
    cache_directory as _cache_directory,
    default_cache_directory,
    job_cache_identity as _job_cache_identity,
    sha256 as _sha256,
    shared_validation_directory,
    split_boundaries as _split_boundaries,
    validate_cache_manifest as _validate_cache_manifest,
    write_or_validate_shared_cache as _write_or_validate_shared_cache,
)
from .stock_time_inference import (
    AnalysisInputs,
    assert_repository_identity as _assert_repository_identity,
    reject_test_derived_path as _reject_test_derived_path,
    validate_analysis_inputs,
)

ANALYSIS_NAME = "stock_time_attribution"
ANALYSIS_VERSION = 4
RECONSTRUCTION_ABSOLUTE_TOLERANCE = 5e-12
ECONOMIC_RECONSTRUCTION_ABSOLUTE_TOLERANCE = 5e-9
MIN_STOCK_SKILL_DAYS = 30
MIN_STOCK_SKILL_COVERAGE = 0.20
RECENT_OBSERVED_MINUTES = 30
OVERNIGHT_LARGE_ABSOLUTE_QUANTILE = 0.80
OVERNIGHT_SIGNED_TAIL_QUANTILE = 0.10
SCOPE_CHOICES = ("core", "full-stage3")
FINAL_ARTIFACT_NAMES = (
    "analysis_manifest.json",
    "summary.json",
    "stock_attribution.parquet",
    "stock_attribution.csv",
    "stock_time_attribution.parquet",
    "liquidity_attribution.parquet",
    "liquidity_time_attribution.parquet",
    "time_of_day_5m.parquet",
    "time_of_day_5m.csv",
    "time_of_day_bins.parquet",
    "time_of_day_bins.csv",
    "opening_regimes.parquet",
    "context_time_deltas.parquet",
    "context_time_deltas.csv",
)
EXPECTED_RETAINED_LOCAL_CONTEXTS = (
    "WDO$",
    "DI1F27",
    "DI1F28",
    "DI1F29",
    "DI1F31",
    "DI1$N",
)
EXPECTED_RETAINED_GLOBAL_CONTEXTS = ("ZT.v.0", "ZN.v.0")
ADDED_CONTEXT_BY_LOGICAL_CONFIGURATION = {
    "core_plus_win": "WIN$",
    "core_plus_es": "ES.v.0",
    "core_plus_nq": "NQ.v.0",
    "core_plus_cl": "CL.v.0",
    "core_plus_hg": "HG.v.0",
    "core_plus_6e": "6E.v.0",
    "core_plus_6m": "6M.v.0",
}


@dataclass(frozen=True)
class AdditiveSpearmanResult:
    contributions: np.ndarray
    sample_ic: np.ndarray


@dataclass(frozen=True)
class EconomicAttributionResult:
    weights: np.ndarray
    return_contributions: np.ndarray
    intraday_turnover: np.ndarray
    flat_entry_turnover: np.ndarray
    flat_exit_turnover: np.ndarray
    top_selected: np.ndarray
    bottom_selected: np.ndarray
    signed_selected_return: np.ndarray


@dataclass(frozen=True)
class EconomicWindowAccounting:
    dates: np.ndarray
    daily_gross_contribution: np.ndarray
    daily_entry_turnover: np.ndarray
    daily_intraday_turnover: np.ndarray
    daily_exit_turnover: np.ndarray

    @property
    def daily_total_turnover(self) -> np.ndarray:
        return (
            self.daily_entry_turnover
            + self.daily_intraday_turnover
            + self.daily_exit_turnover
        )


@dataclass(frozen=True)
class OpeningCondition:
    condition_type: str
    category: str
    sample_mask: np.ndarray
    overnight_regime: str | None = None
    freshness_category: str | None = None
    category_lower_bound: float | None = None
    category_upper_bound: float | None = None
    category_lower_bound_inclusive: bool | None = None
    category_upper_bound_inclusive: bool | None = None
    category_unit: str | None = None
    median_observed_bar_count: np.ndarray | None = None
    median_observed_fraction: np.ndarray | None = None
    fraction_eligible_meeting_expected_history: np.ndarray | None = None


def _atomic_write_parquet(path: Path, frame: pl.DataFrame) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        frame.write_parquet(temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_csv(path: Path, frame: pl.DataFrame) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        frame.write_csv(temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _finite_mean_or_none(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else None


def _with_economic_ratios(
    frame: pl.DataFrame,
    *,
    gross_column: str = "mean_daily_gross_contribution",
    total_turnover_column: str = "mean_daily_total_one_way_turnover",
) -> pl.DataFrame:
    return frame.with_columns(
        pl.when(pl.col(total_turnover_column) > 0)
        .then(pl.col(gross_column) / pl.col(total_turnover_column))
        .otherwise(None)
        .alias("gross_contribution_per_unit_turnover"),
        pl.when(pl.col(total_turnover_column) > 0)
        .then(10_000.0 * pl.col(gross_column) / pl.col(total_turnover_column))
        .otherwise(None)
        .alias("break_even_one_way_cost_bps"),
    )


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    left_centered = left64 - left64.mean()
    right_centered = right64 - right64.mean()
    denominator = math.sqrt(float(np.sum(left_centered**2) * np.sum(right_centered**2)))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(left_centered * right_centered) / denominator)


def primary_time_bins(
    decision_count: int = EXPECTED_DECISIONS_PER_DATE,
    width: int = 6,
) -> tuple[tuple[int, ...], ...]:
    if decision_count <= 0 or width <= 0:
        raise ValueError("Time-bin dimensions must be positive")
    bins = [
        tuple(range(start, min(start + width, decision_count)))
        for start in range(0, decision_count, width)
    ]
    if len(bins) >= 2 and len(bins[-1]) == 1:
        bins[-2] = (*bins[-2], *bins[-1])
        bins.pop()
    flattened = tuple(index for group in bins for index in group)
    if flattened != tuple(range(decision_count)):
        raise RuntimeError("Primary time bins do not partition the decision axis")
    return tuple(bins)


def named_time_scopes() -> dict[str, tuple[int, ...]]:
    decisions = tuple(range(EXPECTED_DECISIONS_PER_DATE))
    return {
        "opening_30": tuple(range(6)),
        "opening_60": tuple(range(12)),
        "rest_of_day": tuple(range(12, EXPECTED_DECISIONS_PER_DATE)),
        "midday": tuple(range(12, 43)),
        "late_session": tuple(range(43, EXPECTED_DECISIONS_PER_DATE)),
        "all_day": decisions,
    }


def moving_block_bootstrap_indices(
    date_count: int,
    *,
    replications: int = BOOTSTRAP_REPLICATIONS,
    block_length: int = BOOTSTRAP_BLOCK_TRADING_DAYS,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    if date_count < block_length or block_length <= 0 or replications <= 0:
        raise ValueError("Moving-block bootstrap dimensions are invalid")
    blocks = math.ceil(date_count / block_length)
    generator = np.random.default_rng(seed)
    starts = generator.integers(
        0,
        date_count - block_length + 1,
        size=(replications, blocks),
    )
    offsets = np.arange(block_length, dtype=np.int64)
    return (starts[..., None] + offsets).reshape(replications, -1)[:, :date_count]


def _bootstrap_date_counts(indices: np.ndarray, date_count: int) -> np.ndarray:
    counts = np.zeros((indices.shape[0], date_count), dtype=np.int16)
    rows = np.repeat(np.arange(indices.shape[0]), indices.shape[1])
    np.add.at(counts, (rows, indices.ravel()), 1)
    return counts


def moving_block_bootstrap_matrix(
    daily_values: np.ndarray,
    *,
    replications: int = BOOTSTRAP_REPLICATIONS,
    block_length: int = BOOTSTRAP_BLOCK_TRADING_DAYS,
    seed: int = BOOTSTRAP_SEED,
    chunk_size: int = 512,
) -> dict[str, np.ndarray]:
    values = np.asarray(daily_values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("Bootstrap values must be date by statistic")
    indices = moving_block_bootstrap_indices(
        values.shape[0],
        replications=replications,
        block_length=block_length,
        seed=seed,
    )
    counts = _bootstrap_date_counts(indices, values.shape[0])
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    replicated = np.full((replications, values.shape[1]), np.nan, dtype=np.float64)
    for start in range(0, replications, chunk_size):
        stop = min(start + chunk_size, replications)
        weights = counts[start:stop].astype(np.float64, copy=False)
        numerators = weights @ filled
        denominators = weights @ finite.astype(np.float64)
        np.divide(
            numerators,
            denominators,
            out=replicated[start:stop],
            where=denominators > 0,
        )
    finite_counts = finite.sum(axis=0)
    estimate = np.divide(
        filled.sum(axis=0),
        finite_counts,
        out=np.full(values.shape[1], np.nan, dtype=np.float64),
        where=finite_counts > 0,
    )
    lower = np.full(values.shape[1], np.nan, dtype=np.float64)
    upper = np.full_like(lower, np.nan)
    probability_positive = np.full_like(lower, np.nan)
    probability_negative = np.full_like(lower, np.nan)
    finite_replication_count = np.isfinite(replicated).sum(axis=0).astype(np.int64)
    for statistic in range(values.shape[1]):
        finite_replicates = replicated[np.isfinite(replicated[:, statistic]), statistic]
        if not finite_replicates.size:
            continue
        lower[statistic], upper[statistic] = np.quantile(
            finite_replicates, (0.025, 0.975)
        )
        probability_positive[statistic] = np.mean(finite_replicates > 0.0)
        probability_negative[statistic] = np.mean(finite_replicates < 0.0)
    return {
        "estimate": estimate,
        "lower_95": lower,
        "upper_95": upper,
        "probability_positive": probability_positive,
        "probability_negative": probability_negative,
        "finite_replication_count": finite_replication_count,
    }


def moving_block_bootstrap(
    daily_values: np.ndarray,
    *,
    replications: int = BOOTSTRAP_REPLICATIONS,
    block_length: int = BOOTSTRAP_BLOCK_TRADING_DAYS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    result = moving_block_bootstrap_matrix(
        daily_values,
        replications=replications,
        block_length=block_length,
        seed=seed,
    )
    return {
        "estimate": float(result["estimate"][0]),
        "interval_lower_95": float(result["lower_95"][0]),
        "interval_upper_95": float(result["upper_95"][0]),
        "probability_positive": float(result["probability_positive"][0]),
        "probability_negative": float(result["probability_negative"][0]),
        "finite_replication_count": int(result["finite_replication_count"][0]),
        "block_trading_days": block_length,
        "replications": replications,
        "bootstrap_seed": seed,
    }


def additive_spearman_contributions(
    predictions: np.ndarray,
    targets: np.ndarray,
    label_mask: np.ndarray,
    *,
    minimum_equities: int = MIN_IC_EQUITIES,
) -> AdditiveSpearmanResult:
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    label_mask = np.asarray(label_mask, dtype=bool)
    if (
        predictions.ndim != 3
        or predictions.shape != targets.shape
        or predictions.shape != label_mask.shape
    ):
        raise ValueError("Spearman arrays must share sample/equity/horizon shape")
    sample_count, equity_count, horizon_count = predictions.shape
    contributions = np.full(
        (sample_count, equity_count, horizon_count), np.nan, dtype=np.float64
    )
    sample_ic = np.full((sample_count, horizon_count), np.nan, dtype=np.float64)
    for sample in range(sample_count):
        for horizon in range(horizon_count):
            valid = np.flatnonzero(label_mask[sample, :, horizon])
            if valid.size < minimum_equities:
                continue
            predicted_scores = standardized_rank_scores(
                predictions[sample, valid, horizon]
            )
            target_scores = standardized_rank_scores(targets[sample, valid, horizon])
            if (
                not np.isfinite(predicted_scores).all()
                or not np.isfinite(target_scores).all()
            ):
                continue
            values = predicted_scores * target_scores / valid.size
            cell = np.zeros(equity_count, dtype=np.float64)
            cell[valid] = values
            ic = float(values.sum())
            contributions[sample, :, horizon] = cell
            sample_ic[sample, horizon] = ic
            if not math.isclose(
                float(cell.sum()),
                ic,
                rel_tol=0.0,
                abs_tol=RECONSTRUCTION_ABSOLUTE_TOLERANCE,
            ):
                raise RuntimeError(
                    "Per-stock Spearman contribution failed to reconstruct"
                )
    return AdditiveSpearmanResult(contributions, sample_ic)


def stock_contribution_opportunity_accounting(
    additive: AdditiveSpearmanResult, label_mask: np.ndarray
) -> dict[str, np.ndarray]:
    label_mask = np.asarray(label_mask, dtype=bool)
    if label_mask.shape != additive.contributions.shape:
        raise ValueError("Stock opportunity mask is misaligned")
    portfolio_valid = np.isfinite(additive.sample_ic)
    stock_valid_support = label_mask & portfolio_valid[:, None, :]
    valid_opportunity_count = stock_valid_support.sum(axis=(0, 2))
    conditional_numerator = np.where(
        stock_valid_support, additive.contributions, 0.0
    ).sum(axis=(0, 2))
    conditional_contribution = np.divide(
        conditional_numerator,
        valid_opportunity_count,
        out=np.full(label_mask.shape[1], np.nan, dtype=np.float64),
        where=valid_opportunity_count > 0,
    )
    portfolio_valid_support = np.broadcast_to(
        portfolio_valid[:, None, :], label_mask.shape
    )
    portfolio_valid_cell_count = portfolio_valid_support.sum(axis=(0, 2))
    unconditional_numerator = np.where(
        portfolio_valid_support,
        np.nan_to_num(additive.contributions, nan=0.0),
        0.0,
    ).sum(axis=(0, 2))
    unconditional_contribution = np.divide(
        unconditional_numerator,
        portfolio_valid_cell_count,
        out=np.full(label_mask.shape[1], np.nan, dtype=np.float64),
        where=portfolio_valid_cell_count > 0,
    )
    return {
        "valid_opportunity_count": valid_opportunity_count,
        "conditional_contribution_numerator": conditional_numerator,
        "conditional_contribution": conditional_contribution,
        "portfolio_valid_cell_count": portfolio_valid_cell_count,
        "unconditional_contribution_numerator": unconditional_numerator,
        "unconditional_contribution": unconditional_contribution,
    }


def aggregate_additive_contributions(
    contributions: np.ndarray,
    sample_ic: np.ndarray,
    date_idx: np.ndarray,
) -> dict[str, np.ndarray | float]:
    contributions = np.asarray(contributions, dtype=np.float64)
    sample_ic = np.asarray(sample_ic, dtype=np.float64)
    date_idx = np.asarray(date_idx, dtype=np.int64)
    if (
        contributions.ndim != 3
        or sample_ic.shape != (contributions.shape[0], contributions.shape[2])
        or date_idx.shape != (contributions.shape[0],)
    ):
        raise ValueError("Contribution aggregation arrays are misaligned")
    dates = np.unique(date_idx)
    daily = np.full(
        (dates.size, contributions.shape[1], contributions.shape[2]),
        np.nan,
        dtype=np.float64,
    )
    valid_decision_counts = np.zeros(
        (dates.size, contributions.shape[2]), dtype=np.int64
    )
    for date_position, date_value in enumerate(dates):
        on_date = date_idx == date_value
        for horizon in range(contributions.shape[2]):
            valid_samples = on_date & np.isfinite(sample_ic[:, horizon])
            count = int(valid_samples.sum())
            valid_decision_counts[date_position, horizon] = count
            if count:
                daily[date_position, :, horizon] = contributions[
                    valid_samples, :, horizon
                ].mean(axis=0)
    horizon_contributions = np.nanmean(daily, axis=0)
    primary_contributions = horizon_contributions.mean(axis=1)
    horizon_ic = np.nanmean(
        np.asarray(
            [
                [
                    np.nanmean(sample_ic[date_idx == date_value, horizon])
                    for horizon in range(sample_ic.shape[1])
                ]
                for date_value in dates
            ],
            dtype=np.float64,
        ),
        axis=0,
    )
    primary_ic = float(horizon_ic.mean())
    if not math.isclose(
        float(primary_contributions.sum()),
        primary_ic,
        rel_tol=0.0,
        abs_tol=RECONSTRUCTION_ABSOLUTE_TOLERANCE,
    ):
        raise RuntimeError(
            "Aggregated stock contributions failed to reconstruct primary IC"
        )
    return {
        "dates": dates,
        "daily_contributions": daily,
        "valid_decision_counts": valid_decision_counts,
        "horizon_contributions": horizon_contributions,
        "primary_contributions": primary_contributions,
        "horizon_ic": horizon_ic,
        "primary_ic": primary_ic,
    }


def economic_stock_attribution(
    predictions: np.ndarray,
    raw_returns: np.ndarray,
    label_mask: np.ndarray,
    date_idx: np.ndarray,
    decision_idx: np.ndarray,
    *,
    minimum_equities: int = MIN_IC_EQUITIES,
) -> EconomicAttributionResult:
    predictions = np.asarray(predictions)
    raw_returns = np.asarray(raw_returns)
    label_mask = np.asarray(label_mask, dtype=bool)
    date_idx = np.asarray(date_idx, dtype=np.int64)
    decision_idx = np.asarray(decision_idx, dtype=np.int64)
    if (
        predictions.ndim != 3
        or predictions.shape != raw_returns.shape
        or predictions.shape != label_mask.shape
        or date_idx.shape != (predictions.shape[0],)
        or decision_idx.shape != (predictions.shape[0],)
    ):
        raise ValueError("Economic attribution arrays are misaligned")
    weights = np.zeros_like(predictions, dtype=np.float64)
    return_contributions = np.zeros_like(weights)
    intraday_turnover = np.zeros_like(weights)
    flat_entry_turnover = np.zeros_like(weights)
    flat_exit_turnover = np.zeros_like(weights)
    top_selected = np.zeros_like(label_mask)
    bottom_selected = np.zeros_like(label_mask)
    signed_selected_return = np.zeros_like(weights)
    order = np.lexsort((decision_idx, date_idx))
    last_valid: dict[tuple[int, int], int] = {}
    previous_weights: dict[tuple[int, int], np.ndarray] = {}
    for sample in order:
        for horizon in range(predictions.shape[2]):
            valid = np.flatnonzero(label_mask[sample, :, horizon])
            if valid.size < minimum_equities:
                continue
            k = max(1, valid.size // 10)
            ranked = valid[
                np.argsort(predictions[sample, valid, horizon], kind="mergesort")
            ]
            bottom = ranked[:k]
            top = ranked[-k:]
            current = np.zeros(predictions.shape[1], dtype=np.float64)
            current[top] = 1.0 / k
            current[bottom] = -1.0 / k
            weights[sample, :, horizon] = current
            return_contributions[sample, :, horizon] = (
                current * raw_returns[sample, :, horizon]
            )
            top_selected[sample, top, horizon] = True
            bottom_selected[sample, bottom, horizon] = True
            signed_selected_return[sample, top, horizon] = raw_returns[
                sample, top, horizon
            ]
            signed_selected_return[sample, bottom, horizon] = -raw_returns[
                sample, bottom, horizon
            ]
            key = (int(date_idx[sample]), horizon)
            previous = previous_weights.get(key)
            if previous is None:
                flat_entry_turnover[sample, :, horizon] = 0.5 * np.abs(current)
            else:
                intraday_turnover[sample, :, horizon] = 0.5 * np.abs(current - previous)
            previous_weights[key] = current
            last_valid[key] = int(sample)
    for key, sample in last_valid.items():
        horizon = key[1]
        flat_exit_turnover[sample, :, horizon] = 0.5 * np.abs(
            weights[sample, :, horizon]
        )
    return EconomicAttributionResult(
        weights=weights,
        return_contributions=return_contributions,
        intraday_turnover=intraday_turnover,
        flat_entry_turnover=flat_entry_turnover,
        flat_exit_turnover=flat_exit_turnover,
        top_selected=top_selected,
        bottom_selected=bottom_selected,
        signed_selected_return=signed_selected_return,
    )


def economic_window_accounting(
    economic: EconomicAttributionResult,
    date_idx: np.ndarray,
    decision_idx: np.ndarray,
    *,
    decisions: tuple[int, ...] | None = None,
    selected_samples: np.ndarray | None = None,
    gross_contributions: np.ndarray | None = None,
) -> EconomicWindowAccounting:
    date_idx = np.asarray(date_idx, dtype=np.int64)
    decision_idx = np.asarray(decision_idx, dtype=np.int64)
    if date_idx.shape != decision_idx.shape or date_idx.shape != (
        economic.weights.shape[0],
    ):
        raise ValueError("Economic window indices are misaligned")
    selected = np.ones(date_idx.shape, dtype=bool)
    if decisions is not None:
        selected &= np.isin(decision_idx, decisions)
    if selected_samples is not None:
        sample_mask = np.asarray(selected_samples, dtype=bool)
        if sample_mask.shape != selected.shape:
            raise ValueError("Economic window sample selector is misaligned")
        selected &= sample_mask
    gross = (
        economic.return_contributions
        if gross_contributions is None
        else np.asarray(gross_contributions, dtype=np.float64)
    )
    if gross.shape != economic.weights.shape:
        raise ValueError("Economic gross contributions are misaligned")
    dates = np.unique(date_idx)
    shape = (dates.size, economic.weights.shape[1], economic.weights.shape[2])
    daily_gross = np.full(shape, np.nan, dtype=np.float64)
    daily_entry = np.full(shape, np.nan, dtype=np.float64)
    daily_intraday = np.full(shape, np.nan, dtype=np.float64)
    daily_exit = np.full(shape, np.nan, dtype=np.float64)
    valid_position = (economic.top_selected | economic.bottom_selected).any(axis=1)
    for day_position, day in enumerate(dates):
        on_day = selected & (date_idx == day)
        for horizon in range(economic.weights.shape[2]):
            positions = np.flatnonzero(on_day & valid_position[:, horizon])
            if not positions.size:
                continue
            positions = positions[np.argsort(decision_idx[positions], kind="stable")]
            weights = economic.weights[positions, :, horizon]
            daily_gross[day_position, :, horizon] = gross[positions, :, horizon].sum(
                axis=0
            )
            daily_entry[day_position, :, horizon] = 0.5 * np.abs(weights[0])
            daily_intraday[day_position, :, horizon] = (
                0.5 * np.abs(np.diff(weights, axis=0)).sum(axis=0)
                if positions.size > 1
                else 0.0
            )
            daily_exit[day_position, :, horizon] = 0.5 * np.abs(weights[-1])
    return EconomicWindowAccounting(
        dates=dates,
        daily_gross_contribution=daily_gross,
        daily_entry_turnover=daily_entry,
        daily_intraday_turnover=daily_intraday,
        daily_exit_turnover=daily_exit,
    )


def _finite_axis_mean(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    count = finite.sum(axis=axis)
    return np.divide(
        np.where(finite, values, 0.0).sum(axis=axis),
        count,
        out=np.full(count.shape, np.nan, dtype=np.float64),
        where=count > 0,
    )


def _window_stock_means(
    accounting: EconomicWindowAccounting,
) -> dict[str, np.ndarray]:
    return {
        "gross": _finite_axis_mean(accounting.daily_gross_contribution, axis=0),
        "entry": _finite_axis_mean(accounting.daily_entry_turnover, axis=0),
        "intraday": _finite_axis_mean(accounting.daily_intraday_turnover, axis=0),
        "exit": _finite_axis_mean(accounting.daily_exit_turnover, axis=0),
        "total": _finite_axis_mean(accounting.daily_total_turnover, axis=0),
    }


def _window_portfolio_daily(
    accounting: EconomicWindowAccounting,
) -> dict[str, np.ndarray]:
    def total(values: np.ndarray) -> np.ndarray:
        result = np.nansum(values, axis=1)
        result[~np.isfinite(values).any(axis=1)] = np.nan
        return result

    return {
        "gross": total(accounting.daily_gross_contribution),
        "entry": total(accounting.daily_entry_turnover),
        "intraday": total(accounting.daily_intraday_turnover),
        "exit": total(accounting.daily_exit_turnover),
        "total": total(accounting.daily_total_turnover),
    }


def _window_bucket_daily(
    accounting: EconomicWindowAccounting,
    buckets: np.ndarray,
    metadata: dict[str, object],
    bucket: int,
) -> dict[str, np.ndarray]:
    positions = _metadata_date_positions(metadata, accounting.dates)
    membership = np.asarray(buckets, dtype=np.int8)[positions] == bucket

    def total(values: np.ndarray) -> np.ndarray:
        valid_day = np.isfinite(values).any(axis=1)
        result = np.nansum(np.where(membership[:, :, None], values, 0.0), axis=1)
        result[~valid_day] = np.nan
        return result

    return {
        "gross": total(accounting.daily_gross_contribution),
        "entry": total(accounting.daily_entry_turnover),
        "intraday": total(accounting.daily_intraday_turnover),
        "exit": total(accounting.daily_exit_turnover),
        "total": total(accounting.daily_total_turnover),
    }


def standardized_rank_scores(values: np.ndarray) -> np.ndarray:
    ranks = average_ranks(np.asarray(values))
    centered = ranks - ranks.mean()
    population_std = float(np.std(ranks, ddof=0))
    if population_std == 0.0:
        return np.full(ranks.shape, np.nan, dtype=np.float64)
    return centered / population_std


def per_stock_time_series_skill(
    predictions: np.ndarray,
    targets: np.ndarray,
    label_mask: np.ndarray,
    date_idx: np.ndarray,
    *,
    minimum_equities: int = MIN_IC_EQUITIES,
    minimum_days: int = MIN_STOCK_SKILL_DAYS,
    minimum_coverage: float = MIN_STOCK_SKILL_COVERAGE,
    bootstrap_replications: int = BOOTSTRAP_REPLICATIONS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, np.ndarray]:
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    label_mask = np.asarray(label_mask, dtype=bool)
    date_idx = np.asarray(date_idx, dtype=np.int64)
    if predictions.shape != targets.shape or predictions.shape != label_mask.shape:
        raise ValueError("Stock-skill arrays must have identical shape")
    dates = np.unique(date_idx)
    date_position = {int(value): index for index, value in enumerate(dates)}
    shape = (dates.size, predictions.shape[1], predictions.shape[2])
    prediction_sum = np.zeros(shape, dtype=np.float64)
    target_sum = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.int32)
    for sample in range(predictions.shape[0]):
        day = date_position[int(date_idx[sample])]
        for horizon in range(predictions.shape[2]):
            valid = np.flatnonzero(label_mask[sample, :, horizon])
            if valid.size < minimum_equities:
                continue
            predicted_scores = standardized_rank_scores(
                predictions[sample, valid, horizon]
            )
            target_scores = standardized_rank_scores(targets[sample, valid, horizon])
            if (
                not np.isfinite(predicted_scores).all()
                or not np.isfinite(target_scores).all()
            ):
                continue
            prediction_sum[day, valid, horizon] += predicted_scores
            target_sum[day, valid, horizon] += target_scores
            counts[day, valid, horizon] += 1
    daily_predictions = np.full(shape, np.nan, dtype=np.float64)
    daily_targets = np.full(shape, np.nan, dtype=np.float64)
    np.divide(
        prediction_sum,
        counts,
        out=daily_predictions,
        where=counts > 0,
    )
    np.divide(target_sum, counts, out=daily_targets, where=counts > 0)
    valid_days = counts > 0
    day_counts = valid_days.sum(axis=0)
    coverage = day_counts / dates.size
    skill = np.full(day_counts.shape, np.nan, dtype=np.float64)
    lower = np.full_like(skill, np.nan)
    upper = np.full_like(skill, np.nan)
    probability_positive = np.full_like(skill, np.nan)
    probability_negative = np.full_like(skill, np.nan)
    finite_replication_count = np.zeros(day_counts.shape, dtype=np.int64)
    indices = moving_block_bootstrap_indices(
        dates.size,
        replications=bootstrap_replications,
        seed=bootstrap_seed,
    )
    bootstrap_counts = _bootstrap_date_counts(indices, dates.size).astype(np.float64)
    for horizon in range(predictions.shape[2]):
        x = daily_predictions[:, :, horizon]
        y = daily_targets[:, :, horizon]
        valid = np.isfinite(x) & np.isfinite(y)
        accepted = (valid.sum(axis=0) >= minimum_days) & (
            valid.mean(axis=0) >= minimum_coverage
        )
        for equity in np.flatnonzero(accepted):
            skill[equity, horizon] = _correlation(
                x[valid[:, equity], equity], y[valid[:, equity], equity]
            )
        filled_x = np.where(valid, x, 0.0)
        filled_y = np.where(valid, y, 0.0)
        n = bootstrap_counts @ valid.astype(np.float64)
        sx = bootstrap_counts @ filled_x
        sy = bootstrap_counts @ filled_y
        sxx = bootstrap_counts @ (filled_x * filled_x)
        syy = bootstrap_counts @ (filled_y * filled_y)
        sxy = bootstrap_counts @ (filled_x * filled_y)
        numerator = sxy - np.divide(sx * sy, n, out=np.zeros_like(sxy), where=n > 0)
        variance_x = sxx - np.divide(sx * sx, n, out=np.zeros_like(sxx), where=n > 0)
        variance_y = syy - np.divide(sy * sy, n, out=np.zeros_like(syy), where=n > 0)
        denominator = np.sqrt(np.maximum(variance_x * variance_y, 0.0))
        replicated = np.full_like(numerator, np.nan)
        np.divide(
            numerator,
            denominator,
            out=replicated,
            where=(n >= 2) & (denominator > 0),
        )
        finite_replication_count[:, horizon] = np.isfinite(replicated).sum(axis=0)
        for equity in np.flatnonzero(accepted):
            finite_replicates = replicated[np.isfinite(replicated[:, equity]), equity]
            if not finite_replicates.size:
                continue
            lower[equity, horizon], upper[equity, horizon] = np.quantile(
                finite_replicates, (0.025, 0.975)
            )
            probability_positive[equity, horizon] = np.mean(finite_replicates > 0.0)
            probability_negative[equity, horizon] = np.mean(finite_replicates < 0.0)
        finite_replication_count[~accepted, horizon] = 0
    return {
        "dates": dates,
        "daily_prediction_scores": daily_predictions,
        "daily_target_scores": daily_targets,
        "valid_day_count": day_counts,
        "coverage": coverage,
        "skill": skill,
        "interval_lower_95": lower,
        "interval_upper_95": upper,
        "probability_positive": probability_positive,
        "probability_negative": probability_negative,
        "finite_replication_count": finite_replication_count,
    }


def deterministic_liquidity_buckets(
    values: np.ndarray,
    eligible: np.ndarray,
    *,
    bucket_count: int = 5,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    if values.ndim != 1 or eligible.shape != values.shape or bucket_count <= 0:
        raise ValueError("Liquidity bucket inputs are invalid")
    selected = np.flatnonzero(eligible & np.isfinite(values))
    result = np.full(values.shape, -1, dtype=np.int8)
    if not selected.size:
        return result
    ranks = average_ranks(values[selected])
    buckets = np.floor((ranks + 0.5) * bucket_count / selected.size).astype(np.int64)
    result[selected] = np.clip(buckets, 0, bucket_count - 1).astype(np.int8)
    return result


def adaptive_liquidity_buckets(
    values: np.ndarray,
    eligible: np.ndarray,
    *,
    maximum_buckets: int = 5,
    minimum_equities: int = MIN_IC_EQUITIES,
) -> tuple[np.ndarray, int]:
    eligible = np.asarray(eligible, dtype=bool) & np.isfinite(values)
    for count in range(maximum_buckets, 1, -1):
        buckets = deterministic_liquidity_buckets(values, eligible, bucket_count=count)
        sizes = np.bincount(buckets[buckets >= 0], minlength=count)
        if sizes.size == count and np.all(sizes >= minimum_equities):
            return buckets, count
    if int(eligible.sum()) >= minimum_equities:
        buckets = np.full(eligible.shape, -1, dtype=np.int8)
        buckets[eligible] = 0
        return buckets, 1
    return np.full(eligible.shape, -1, dtype=np.int8), 0


def learn_overnight_thresholds(training_market_gap: np.ndarray) -> dict[str, float]:
    values = np.asarray(training_market_gap, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < MIN_STOCK_SKILL_DAYS:
        raise ValueError("Training period has too few overnight observations")
    tail = OVERNIGHT_SIGNED_TAIL_QUANTILE
    return {
        "large_absolute": float(
            np.quantile(np.abs(values), OVERNIGHT_LARGE_ABSOLUTE_QUANTILE)
        ),
        "large_negative": float(np.quantile(values, tail)),
        "large_positive": float(np.quantile(values, 1.0 - tail)),
        "training_observation_count": int(values.size),
    }


def overnight_regimes(
    market_gap: np.ndarray, thresholds: dict[str, float]
) -> dict[str, np.ndarray]:
    values = np.asarray(market_gap, dtype=np.float64)
    finite = np.isfinite(values)
    large = finite & (np.abs(values) >= float(thresholds["large_absolute"]))
    return {
        "normal": finite & ~large,
        "large": large,
        "large_positive": finite & (values >= float(thresholds["large_positive"])),
        "large_negative": finite & (values <= float(thresholds["large_negative"])),
    }


def causal_observation_completeness(
    observed: np.ndarray,
    date_idx: np.ndarray,
    cutoffs: np.ndarray,
    *,
    readiness: np.ndarray | None = None,
    preopen_cutoff: int | None = None,
    recent_minutes: int = RECENT_OBSERVED_MINUTES,
) -> dict[str, np.ndarray]:
    observed = np.asarray(observed, dtype=bool)
    date_idx = np.asarray(date_idx, dtype=np.int64)
    cutoffs = np.asarray(cutoffs, dtype=np.int64)
    if observed.ndim != 3 or date_idx.shape != cutoffs.shape:
        raise ValueError("Completeness arrays are misaligned")
    if readiness is not None:
        readiness = np.asarray(readiness, dtype=bool)
        if readiness.shape != observed.shape[:2]:
            raise ValueError("Readiness must have date/instrument axes")
    shape = (date_idx.size, observed.shape[1])
    observed_fraction = np.zeros(shape, dtype=np.float64)
    recent_fraction = np.zeros(shape, dtype=np.float64)
    preopen_fraction = np.full(shape, np.nan, dtype=np.float64)
    observed_count = np.zeros(shape, dtype=np.int32)
    staleness = np.full(shape, np.nan, dtype=np.float64)
    ready = np.ones(shape, dtype=bool)
    for sample, (day, cutoff) in enumerate(zip(date_idx, cutoffs, strict=True)):
        if day < 0 or cutoff <= 0 or cutoff > observed.shape[2]:
            raise ValueError("Completeness cutoff is outside the causal grid")
        prefix = observed[day, :, :cutoff]
        observed_count[sample] = prefix.sum(axis=1)
        observed_fraction[sample] = observed_count[sample] / cutoff
        start = max(0, cutoff - recent_minutes)
        recent_fraction[sample] = prefix[:, start:cutoff].mean(axis=1)
        if preopen_cutoff is not None:
            width = min(cutoff, preopen_cutoff)
            if width > 0:
                preopen_fraction[sample] = prefix[:, :width].mean(axis=1)
        for instrument in range(observed.shape[1]):
            positions = np.flatnonzero(prefix[instrument])
            if positions.size:
                staleness[sample, instrument] = cutoff - 1 - int(positions[-1])
        if readiness is not None:
            ready[sample] = readiness[day]
    return {
        "scheduled_minutes": cutoffs.copy(),
        "observed_bars": observed_count,
        "observed_fraction": observed_fraction,
        "recent_observed_fraction": recent_fraction,
        "missing_history_fraction": 1.0 - observed_fraction,
        "preopen_observed_fraction": preopen_fraction,
        "minutes_since_most_recent_observed_bar": staleness,
        "ready": ready,
    }


def _analysis_configuration(
    inputs: AnalysisInputs, scope: str, cache_dir: Path
) -> dict[str, object]:
    return {
        "analysis_name": ANALYSIS_NAME,
        "analysis_version": ANALYSIS_VERSION,
        "scope": scope,
        "cache_dir": str(cache_dir),
        "split": "validation",
        "stage3_state_path": str(inputs.state_path),
        "stage3_state_sha256": inputs.state_sha256,
        "feature_store_identity": inputs.feature_identity,
        "analyzer_git_commit_sha": inputs.analyzer_git_commit_sha,
        "analyzer_worktree_clean": inputs.analyzer_worktree_clean,
        "analyzer_source_sha256": inputs.analyzer_source_sha256,
        "inference_code_sha256": inputs.inference_code_sha256,
        "jobs": [_job_cache_identity(inputs, job) for job in inputs.jobs],
    }


def _new_analysis_state(
    inputs: AnalysisInputs,
    scope: str,
    cache_dir: Path,
) -> dict[str, object]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "analysis_name": ANALYSIS_NAME,
        "analysis_version": ANALYSIS_VERSION,
        "status": "running",
        "configuration": _analysis_configuration(inputs, scope, cache_dir),
        "created_at_utc": now,
        "completed_at_utc": None,
        "jobs": [
            {
                "position": job.position,
                "logical_configuration": job.logical_configuration,
                "seed": job.seed,
                "status": "pending",
                "cache_manifest_path": None,
                "cache_manifest_sha256": None,
                "started_at_utc": None,
                "completed_at_utc": None,
                "error": None,
            }
            for job in inputs.jobs
        ],
        "artifacts": {},
        "pending_artifact": None,
    }


def _load_analysis_state(
    path: Path,
    inputs: AnalysisInputs,
    scope: str,
    cache_dir: Path,
) -> dict[str, object]:
    expected = _analysis_configuration(inputs, scope, cache_dir)
    if not path.exists():
        return _new_analysis_state(inputs, scope, cache_dir)
    state = json.loads(path.read_text(encoding="utf-8"))
    _reject_test_derived_metadata(state, "analysis state")
    if (
        state.get("analysis_name") != ANALYSIS_NAME
        or state.get("analysis_version") != ANALYSIS_VERSION
        or state.get("status") not in {"running", "inference_completed", "completed"}
        or state.get("configuration") != expected
    ):
        raise ValueError("Existing analysis state is incompatible")
    jobs = state.get("jobs")
    expected_jobs = [
        (job.position, job.logical_configuration, job.seed) for job in inputs.jobs
    ]
    actual_jobs = (
        [
            (job.get("position"), job.get("logical_configuration"), job.get("seed"))
            for job in jobs
            if isinstance(job, dict)
        ]
        if isinstance(jobs, list)
        else []
    )
    if actual_jobs != expected_jobs:
        raise ValueError("Existing analysis state has the wrong job matrix")
    return state


def _resolve_cache_directory(output_dir: Path, cache_dir: Path | None) -> Path:
    resolved = (
        default_cache_directory(output_dir, ANALYSIS_NAME)
        if cache_dir is None
        else cache_dir.resolve()
    )
    _reject_test_derived_path(resolved, "analysis cache path")
    return resolved


def dry_run_payload(
    stage3_state_path: Path,
    output_dir: Path,
    scope: str,
    cache_dir: Path | None = None,
) -> dict[str, object]:
    _reject_test_derived_path(output_dir.resolve(), "analysis output path")
    resolved_cache_dir = _resolve_cache_directory(output_dir, cache_dir)
    inputs = validate_analysis_inputs(stage3_state_path, scope)
    return {
        "analysis_name": ANALYSIS_NAME,
        "analysis_version": ANALYSIS_VERSION,
        "dry_run": True,
        "scope": scope,
        "split": "validation",
        "stage3_state_path": str(inputs.state_path),
        "stage3_state_sha256": inputs.state_sha256,
        "output_dir": str(output_dir.resolve()),
        "cache_dir": str(resolved_cache_dir),
        "feature_store_identity": inputs.feature_identity,
        "validation_sample_count": inputs.validation_rows.height,
        "inference_job_count": len(inputs.jobs),
        "analyzer_git_commit_sha": inputs.analyzer_git_commit_sha,
        "analyzer_worktree_clean": inputs.analyzer_worktree_clean,
        "analyzer_source_sha256": inputs.analyzer_source_sha256,
        "inference_code_sha256": inputs.inference_code_sha256,
        "models_loaded_onto_gpu": False,
        "artifacts_created": False,
        "jobs": [
            {
                "position": job.position,
                "logical_configuration": job.logical_configuration,
                "context_ablation": job.context_ablation,
                "seed": job.seed,
                "run_dir": str(job.run_dir),
                "run_manifest_sha256": job.run_manifest_sha256,
                "checkpoint_path": str(job.checkpoint_path),
                "checkpoint_sha256": job.checkpoint_sha256,
                "planned_cache_manifest": str(
                    _cache_directory(resolved_cache_dir, job) / "manifest.json"
                ),
            }
            for job in inputs.jobs
        ],
        "planned_artifacts": ["analysis_state.json", *FINAL_ARTIFACT_NAMES],
        "selection": None,
        "training_performed": False,
        "test_data_used": False,
    }


def _feature_axes(inputs: AnalysisInputs) -> dict[str, object]:
    schema = json.loads(
        (inputs.feature_store / "feature_schema.json").read_text(encoding="utf-8")
    )
    slow_rows = schema.get("slow_channels")
    dynamic_rows = schema.get("dynamic_channels")
    if not isinstance(slow_rows, list) or not isinstance(dynamic_rows, list):
        raise ValueError("Feature schema lacks channel axes")
    slow_names = tuple(row.get("name") for row in slow_rows if isinstance(row, dict))
    dynamic_names = tuple(
        row.get("name") for row in dynamic_rows if isinstance(row, dict)
    )
    if slow_names != SLOW_CHANNELS or dynamic_names != DYNAMIC_CHANNELS:
        raise ValueError("Feature schema channel order differs from the code contract")
    liquidity_name = "median_daily_dollar_volume_20d_log_scale"
    liquidity_index = slow_names.index(liquidity_name)
    if liquidity_index != 13:
        raise ValueError("Liquidity slow feature is not at expected position 13")
    observed_index = dynamic_names.index("observed")
    if observed_index != 5:
        raise ValueError("Observed dynamic channel is not at expected position 5")
    affine = inputs.feature_manifest.get("constants", {}).get(
        "dollar_volume_log_affine"
    )
    if (
        not isinstance(affine, dict)
        or not math.isfinite(float(affine.get("center", float("nan"))))
        or not math.isfinite(float(affine.get("scale", float("nan"))))
        or float(affine["scale"]) <= 0.0
    ):
        raise ValueError("Feature manifest lacks dollar-volume affine metadata")
    return {
        "schema": schema,
        "liquidity_channel_name": liquidity_name,
        "liquidity_channel_index": liquidity_index,
        "observed_channel_index": observed_index,
        "dollar_volume_log_affine": {
            "center": float(affine["center"]),
            "scale": float(affine["scale"]),
            "inverse": "expm1(normalized * scale + center)",
        },
    }


def _universe_liquidity_threshold(
    inputs: AnalysisInputs,
) -> dict[str, object] | None:
    canonical_inputs = inputs.feature_manifest.get("canonical_inputs")
    if not isinstance(canonical_inputs, dict):
        return None
    universe = canonical_inputs.get("point_in_time_universe")
    if not isinstance(universe, dict) or not isinstance(
        universe.get("resolved_path"), str
    ):
        return None
    manifest_path = Path(str(universe["resolved_path"])) / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest.get("config")
    if not isinstance(config, dict):
        return None
    threshold = config.get("minimum_median_daily_turnover_brl")
    if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
        return None
    return {
        "value_brl": float(threshold),
        "source_manifest_path": str(manifest_path.resolve()),
        "source_manifest_sha256": _sha256(manifest_path),
        "field": "config.minimum_median_daily_turnover_brl",
    }


def _load_date_partition(
    path: Path,
    date_indices: np.ndarray,
    *trailing_selection: object,
) -> np.ndarray:
    indices = np.asarray(date_indices, dtype=np.int64)
    if (
        indices.ndim != 1
        or not indices.size
        or np.any(indices < 0)
        or np.unique(indices).size != indices.size
    ):
        raise ValueError(f"Invalid date partition for {path}")
    source = np.load(path, mmap_mode="r", allow_pickle=False)
    if np.any(indices >= source.shape[0]):
        raise ValueError(f"Date partition exceeds {path.name}")
    return np.asarray(source[(indices, *trailing_selection)])


def _metadata_date_positions(
    metadata: dict[str, object], date_idx: np.ndarray
) -> np.ndarray:
    mapping = metadata.get("date_position_by_index")
    if not isinstance(mapping, dict):
        raw_indices = metadata.get("validation_date_indices")
        if raw_indices is None:
            return np.asarray(date_idx, dtype=np.int64)
        indices = np.asarray(raw_indices, dtype=np.int64)
        mapping = {int(value): position for position, value in enumerate(indices)}
    try:
        return np.asarray(
            [mapping[int(value)] for value in np.asarray(date_idx, dtype=np.int64)],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(
            f"Sample date is outside the validation partition: {error}"
        ) from error


def _market_overnight_gap(
    active: np.ndarray,
    observed: np.ndarray,
    overnight_gap: np.ndarray,
) -> np.ndarray:
    early_observed = observed[:, :, : DECISION_EQUITY_INDICES[0]].any(axis=2)
    result = np.full(active.shape[0], np.nan, dtype=np.float64)
    for day in range(active.shape[0]):
        valid = active[day] & early_observed[day] & np.isfinite(overnight_gap[day])
        if valid.any():
            result[day] = float(np.median(overnight_gap[day, valid]))
    return result


def _load_analysis_metadata(
    inputs: AnalysisInputs,
    shared: dict[str, np.ndarray],
) -> dict[str, object]:
    axes = _feature_axes(inputs)
    store = inputs.feature_store
    equity_index = pl.read_parquet(store / "equity_index.parquet").sort("equity_slot")
    if (
        equity_index.height != 158
        or equity_index.get_column("security_id").n_unique() != 158
    ):
        raise ValueError("Equity axis does not contain 158 unique permanent identities")
    date_index = pl.read_parquet(store / "date_index.parquet").sort("date_idx")
    date_values = date_index.get_column("trade_date").to_numpy()
    training_indices = (
        select_sample_split(inputs.sample_index, "train")
        .get_column("date_idx")
        .unique()
        .sort()
        .to_numpy()
        .astype(np.int64)
    )
    validation_indices = (
        inputs.validation_rows.get_column("date_idx")
        .unique()
        .sort()
        .to_numpy()
        .astype(np.int64)
    )
    sample_dates = np.asarray(shared["date_idx"], dtype=np.int64)
    if not np.isin(sample_dates, validation_indices).all():
        raise ValueError("Shared cache contains a non-validation date")
    date_position_by_index = {
        int(value): position for position, value in enumerate(validation_indices)
    }
    sample_date_positions = np.asarray(
        [date_position_by_index[int(value)] for value in sample_dates], dtype=np.int64
    )
    sample_decisions = np.asarray(shared["decision_idx"], dtype=np.int64)
    observed_channel = int(axes["observed_channel_index"])
    liquidity_channel = int(axes["liquidity_channel_index"])
    overnight_channel = SLOW_CHANNELS.index("overnight_gap_normalized")

    membership = _load_date_partition(
        store / "equity_membership.npy", validation_indices, slice(None)
    ).astype(bool, copy=False)
    equity_ready = _load_date_partition(
        store / "equity_data_ready.npy", validation_indices, slice(None)
    ).astype(bool, copy=False)
    active = membership & equity_ready
    normalized_liquidity = _load_date_partition(
        store / "equity_slow.npy",
        validation_indices,
        slice(None),
        liquidity_channel,
    ).astype(np.float64, copy=False)
    affine = axes["dollar_volume_log_affine"]
    if not isinstance(affine, dict):
        raise RuntimeError("Liquidity affine metadata is malformed")
    dollar_liquidity = np.expm1(
        normalized_liquidity * float(affine["scale"]) + float(affine["center"])
    )
    dollar_liquidity[~active] = np.nan
    liquidity_quintile = np.full(active.shape, -1, dtype=np.int8)
    adaptive_liquidity = np.full(active.shape, -1, dtype=np.int8)
    adaptive_counts = np.zeros(active.shape[0], dtype=np.int8)
    for day in range(active.shape[0]):
        liquidity_quintile[day] = deterministic_liquidity_buckets(
            dollar_liquidity[day], active[day]
        )
        adaptive_liquidity[day], adaptive_counts[day] = adaptive_liquidity_buckets(
            dollar_liquidity[day], active[day]
        )

    training_active = _load_date_partition(
        store / "equity_membership.npy", training_indices, slice(None)
    ).astype(bool, copy=False) & _load_date_partition(
        store / "equity_data_ready.npy", training_indices, slice(None)
    ).astype(bool, copy=False)
    training_observed = _load_date_partition(
        store / "equity_features.npy",
        training_indices,
        slice(None),
        slice(None),
        observed_channel,
    ).astype(bool, copy=False)
    training_overnight = _load_date_partition(
        store / "equity_slow.npy",
        training_indices,
        slice(None),
        overnight_channel,
    ).astype(np.float64, copy=False)
    training_market_gap = _market_overnight_gap(
        training_active, training_observed, training_overnight
    )
    thresholds = learn_overnight_thresholds(training_market_gap)

    equity_observed = _load_date_partition(
        store / "equity_features.npy",
        validation_indices,
        slice(None),
        slice(None),
        observed_channel,
    ).astype(bool, copy=False)
    validation_overnight = _load_date_partition(
        store / "equity_slow.npy",
        validation_indices,
        slice(None),
        overnight_channel,
    ).astype(np.float64, copy=False)
    market_overnight_gap = _market_overnight_gap(
        active, equity_observed, validation_overnight
    )
    regimes = overnight_regimes(market_overnight_gap, thresholds)
    equity_completeness = causal_observation_completeness(
        equity_observed,
        sample_date_positions,
        np.asarray(DECISION_EQUITY_INDICES, dtype=np.int64)[sample_decisions],
        readiness=active,
        preopen_cutoff=None,
    )

    context_observed = _load_date_partition(
        store / "context_features.npy",
        validation_indices,
        slice(None),
        slice(None),
        observed_channel,
    ).astype(bool, copy=False)
    context_ready = _load_date_partition(
        store / "context_data_ready.npy", validation_indices, slice(None)
    ).astype(bool, copy=False)
    local_completeness = causal_observation_completeness(
        context_observed,
        sample_date_positions,
        75 + 5 * sample_decisions,
        readiness=context_ready,
        preopen_cutoff=60,
    )

    global_observed = _load_date_partition(
        store / "global_features.npy",
        validation_indices,
        slice(None),
        slice(None),
        observed_channel,
    ).astype(bool, copy=False)
    global_ready = _load_date_partition(
        store / "global_data_ready.npy",
        validation_indices,
        slice(None),
        slice(None),
    ).astype(bool, copy=False)
    global_completeness = causal_observation_completeness(
        global_observed,
        sample_date_positions,
        np.asarray(DECISION_GLOBAL_INDICES, dtype=np.int64)[sample_decisions],
        readiness=None,
        preopen_cutoff=330,
    )
    global_completeness["ready"] = global_ready[
        sample_date_positions, :, sample_decisions
    ]
    return {
        "axes": axes,
        "equity_index": equity_index,
        "date_index": date_index.filter(
            pl.col("date_idx").is_in(validation_indices.tolist())
        ),
        "trade_dates": date_values[validation_indices],
        "validation_date_indices": validation_indices,
        "date_position_by_index": date_position_by_index,
        "active": active,
        "dollar_liquidity": dollar_liquidity,
        "liquidity_quintile": liquidity_quintile,
        "adaptive_liquidity": adaptive_liquidity,
        "adaptive_liquidity_bucket_count": adaptive_counts,
        "eligibility_liquidity_threshold": _universe_liquidity_threshold(inputs),
        "market_overnight_gap": market_overnight_gap,
        "overnight_thresholds": thresholds,
        "overnight_regimes": regimes,
        "equity_completeness": equity_completeness,
        "local_completeness": local_completeness,
        "global_completeness": global_completeness,
    }


def _sample_trade_dates(
    metadata: dict[str, object], date_idx: np.ndarray
) -> np.ndarray:
    trade_dates = np.asarray(metadata["trade_dates"])
    return trade_dates[_metadata_date_positions(metadata, date_idx)]


def _daily_grid(
    values: np.ndarray,
    date_idx: np.ndarray,
    decision_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    dates = np.unique(date_idx)
    result = np.full(
        (dates.size, EXPECTED_DECISIONS_PER_DATE, values.shape[1]),
        np.nan,
        dtype=np.float64,
    )
    date_position = {int(value): index for index, value in enumerate(dates)}
    for sample in range(values.shape[0]):
        result[date_position[int(date_idx[sample])], int(decision_idx[sample])] = (
            values[sample]
        )
    return dates, result


def _grid_from_stock_values(
    values: np.ndarray,
    date_idx: np.ndarray,
    decision_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    dates = np.unique(date_idx)
    result = np.full(
        (
            dates.size,
            EXPECTED_DECISIONS_PER_DATE,
            values.shape[1],
            values.shape[2],
        ),
        np.nan,
        dtype=np.float64,
    )
    date_position = {int(value): index for index, value in enumerate(dates)}
    for sample in range(values.shape[0]):
        result[date_position[int(date_idx[sample])], int(decision_idx[sample])] = (
            values[sample]
        )
    return dates, result


def _scope_daily_mean(grid: np.ndarray, decisions: tuple[int, ...]) -> np.ndarray:
    selected = np.asarray(grid[:, decisions], dtype=np.float64)
    finite = np.isfinite(selected)
    count = finite.sum(axis=1)
    return np.divide(
        np.where(finite, selected, 0.0).sum(axis=1),
        count,
        out=np.full(count.shape, np.nan, dtype=np.float64),
        where=count > 0,
    )


def _coverage_summary(
    daily_valid_count: np.ndarray,
    daily_coverage: np.ndarray,
    decisions: tuple[int, ...],
    horizon_index: int | None,
) -> dict[str, float | int | None]:
    counts = np.asarray(daily_valid_count[:, decisions], dtype=np.float64)
    coverage = np.asarray(daily_coverage[:, decisions], dtype=np.float64)
    if horizon_index is not None:
        counts = counts[..., horizon_index]
        coverage = coverage[..., horizon_index]
    else:
        counts = counts.reshape(counts.shape[0], -1)
        coverage = coverage.reshape(coverage.shape[0], -1)
    valid = np.isfinite(counts) & (counts >= MIN_IC_EQUITIES)
    daily_count = np.divide(
        np.where(valid, counts, 0.0).sum(axis=1),
        valid.sum(axis=1),
        out=np.full(counts.shape[0], np.nan, dtype=np.float64),
        where=valid.sum(axis=1) > 0,
    )
    daily_coverage_value = np.divide(
        np.where(valid, coverage, 0.0).sum(axis=1),
        valid.sum(axis=1),
        out=np.full(counts.shape[0], np.nan, dtype=np.float64),
        where=valid.sum(axis=1) > 0,
    )
    return {
        "mean_valid_equity_count": _finite_mean_or_none(daily_count),
        "label_coverage": _finite_mean_or_none(daily_coverage_value),
        "valid_decision_cell_count": int(valid.sum()),
        "valid_date_count": int(valid.any(axis=1).sum()),
    }


def _time_bin_metadata() -> list[dict[str, object]]:
    rows = []
    for index, decisions in enumerate(primary_time_bins()):
        rows.append(
            {
                "bin_index": index,
                "name": f"bin_{index + 1:02d}",
                "decision_indices": list(decisions),
                "start_time_brt": DECISION_TIMES[decisions[0]].isoformat(),
                "end_time_brt": DECISION_TIMES[decisions[-1]].isoformat(),
                "decision_count": len(decisions),
            }
        )
    return rows


def _sign_counts(values: np.ndarray, tolerance: float = 1e-15) -> tuple[int, int, int]:
    values = np.asarray(values, dtype=np.float64)
    return (
        int(np.sum(values > tolerance)),
        int(np.sum(np.abs(values) <= tolerance)),
        int(np.sum(values < -tolerance)),
    )


def _economic_reconstruction_checks(
    economic: EconomicAttributionResult,
    predictions: np.ndarray,
    raw_returns: np.ndarray,
    label_mask: np.ndarray,
    date_idx: np.ndarray,
    decision_idx: np.ndarray,
) -> dict[str, float | bool]:

    spread_from_stock = economic.return_contributions.sum(axis=1)
    intraday_from_stock = economic.intraday_turnover.sum(axis=1)
    maximum_spread_difference = 0.0
    maximum_turnover_difference = 0.0
    previous: dict[tuple[int, int], np.ndarray] = {}
    order = np.lexsort((decision_idx, date_idx))
    for sample in order:
        for horizon in range(predictions.shape[2]):
            valid = np.flatnonzero(label_mask[sample, :, horizon])
            if valid.size < MIN_IC_EQUITIES:
                continue
            k = max(1, valid.size // 10)
            ranked = valid[
                np.argsort(predictions[sample, valid, horizon], kind="mergesort")
            ]
            expected_spread = float(
                np.asarray(
                    raw_returns[sample, ranked[-k:], horizon], dtype=np.float64
                ).mean()
                - np.asarray(
                    raw_returns[sample, ranked[:k], horizon], dtype=np.float64
                ).mean()
            )
            maximum_spread_difference = max(
                maximum_spread_difference,
                abs(expected_spread - spread_from_stock[sample, horizon]),
            )
            key = (int(date_idx[sample]), horizon)
            current = economic.weights[sample, :, horizon]
            if key in previous:
                expected_turnover = 0.5 * float(np.abs(current - previous[key]).sum())
                maximum_turnover_difference = max(
                    maximum_turnover_difference,
                    abs(expected_turnover - intraday_from_stock[sample, horizon]),
                )
            previous[key] = current
    passed = (
        maximum_spread_difference <= ECONOMIC_RECONSTRUCTION_ABSOLUTE_TOLERANCE
        and maximum_turnover_difference <= RECONSTRUCTION_ABSOLUTE_TOLERANCE
    )
    if not passed:
        raise RuntimeError("Economic stock attribution failed reconstruction")
    return {
        "maximum_gross_spread_absolute_difference": maximum_spread_difference,
        "maximum_intraday_turnover_absolute_difference": maximum_turnover_difference,
        "passed": True,
    }


def _period_additive_summary(
    additive: AdditiveSpearmanResult,
    date_idx: np.ndarray,
    selected: np.ndarray,
) -> dict[str, np.ndarray | float]:
    if not selected.any():
        raise ValueError("Attribution period has no observations")
    return aggregate_additive_contributions(
        additive.contributions[selected],
        additive.sample_ic[selected],
        date_idx[selected],
    )


def _stock_identity_rows(equity_index: pl.DataFrame) -> list[dict[str, object]]:
    rows = []
    for row in equity_index.to_dicts():
        rows.append(
            {
                "equity_slot": int(row["equity_slot"]),
                "security_id": str(row["security_id"]),
                "isin": row.get("isin"),
                "display_ticker": row.get("latest_ticker"),
                "display_name": row.get("xp_symbol"),
            }
        )
    return rows


def _build_core_outputs(
    cache_paths: dict[tuple[str, int], Path],
    shared: dict[str, np.ndarray],
    metadata: dict[str, object],
    *,
    bootstrap_replications: int | None = None,
) -> tuple[
    dict[str, pl.DataFrame],
    dict[int, dict[str, object]],
    dict[str, object],
]:
    if bootstrap_replications is None:
        bootstrap_replications = BOOTSTRAP_REPLICATIONS
    sample_id = np.asarray(shared["sample_id"], dtype=np.int64)
    del sample_id
    date_idx = np.asarray(shared["date_idx"], dtype=np.int64)
    decision_idx = np.asarray(shared["decision_idx"], dtype=np.int64)
    date_positions = _metadata_date_positions(metadata, date_idx)
    targets = np.asarray(shared["targets"])
    raw_returns = np.asarray(shared["raw_returns"])
    label_mask = np.asarray(shared["label_mask"], dtype=bool)
    trade_dates = _sample_trade_dates(metadata, date_idx)
    identities = _stock_identity_rows(metadata["equity_index"])
    dollar_liquidity = np.asarray(metadata["dollar_liquidity"], dtype=np.float64)
    threshold_metadata = metadata["eligibility_liquidity_threshold"]
    threshold = (
        float(threshold_metadata["value_brl"])
        if isinstance(threshold_metadata, dict)
        else None
    )
    time_5m_rows: list[dict[str, object]] = []
    time_bin_rows: list[dict[str, object]] = []
    stock_time_rows: list[dict[str, object]] = []
    core_by_seed: dict[int, dict[str, object]] = {}
    stock_seed_arrays: dict[int, dict[str, np.ndarray]] = {}
    time_grids: dict[int, dict[str, np.ndarray]] = {}
    reconstruction: dict[str, object] = {}
    bin_metadata = _time_bin_metadata()
    scopes = {
        **{row["name"]: tuple(row["decision_indices"]) for row in bin_metadata},
        **named_time_scopes(),
    }
    valid_count = label_mask.sum(axis=1).astype(np.float64)
    _, daily_valid_count = _daily_grid(valid_count, date_idx, decision_idx)
    coverage_values = label_mask.mean(axis=1).astype(np.float64)
    _, daily_coverage = _daily_grid(coverage_values, date_idx, decision_idx)
    for seed in STAGE3_SEEDS:
        predictions = np.load(
            cache_paths[("core", seed)], mmap_mode="r", allow_pickle=False
        )
        additive = additive_spearman_contributions(predictions, targets, label_mask)
        aggregate = aggregate_additive_contributions(
            additive.contributions, additive.sample_ic, date_idx
        )
        economic = economic_stock_attribution(
            predictions,
            raw_returns,
            label_mask,
            date_idx,
            decision_idx,
        )
        economic_check = _economic_reconstruction_checks(
            economic,
            predictions,
            raw_returns,
            label_mask,
            date_idx,
            decision_idx,
        )
        _, daily_ic = _daily_grid(additive.sample_ic, date_idx, decision_idx)
        top_contributions = np.where(
            economic.top_selected, economic.return_contributions, 0.0
        )
        bottom_contributions = np.where(
            economic.bottom_selected, economic.return_contributions, 0.0
        )
        daily_spread = np.full_like(daily_ic, np.nan)
        daily_top_return = np.full_like(daily_ic, np.nan)
        daily_bottom_return = np.full_like(daily_ic, np.nan)
        daily_turnover = np.full_like(daily_ic, np.nan)
        daily_entry = np.full_like(daily_ic, np.nan)
        daily_exit = np.full_like(daily_ic, np.nan)
        for decision in range(EXPECTED_DECISIONS_PER_DATE):
            decision_accounting = economic_window_accounting(
                economic,
                date_idx,
                decision_idx,
                decisions=(decision,),
            )
            decision_values = _window_portfolio_daily(decision_accounting)
            top_values = _window_portfolio_daily(
                economic_window_accounting(
                    economic,
                    date_idx,
                    decision_idx,
                    decisions=(decision,),
                    gross_contributions=top_contributions,
                )
            )
            bottom_values = _window_portfolio_daily(
                economic_window_accounting(
                    economic,
                    date_idx,
                    decision_idx,
                    decisions=(decision,),
                    gross_contributions=bottom_contributions,
                )
            )
            daily_spread[:, decision] = decision_values["gross"]
            daily_top_return[:, decision] = top_values["gross"]
            daily_bottom_return[:, decision] = -bottom_values["gross"]
            daily_turnover[:, decision] = decision_values["intraday"]
            daily_entry[:, decision] = decision_values["entry"]
            daily_exit[:, decision] = decision_values["exit"]
        time_bootstrap = moving_block_bootstrap_matrix(
            daily_ic.reshape(daily_ic.shape[0], -1),
            replications=bootstrap_replications,
            seed=BOOTSTRAP_SEED + seed,
        )
        for decision in range(EXPECTED_DECISIONS_PER_DATE):
            for horizon_index, horizon in enumerate(HORIZONS):
                flat_index = decision * len(HORIZONS) + horizon_index
                values = daily_ic[:, decision, horizon_index]
                coverage = _coverage_summary(
                    daily_valid_count, daily_coverage, (decision,), horizon_index
                )
                time_5m_rows.append(
                    {
                        "aggregation": "seed",
                        "seed": seed,
                        "decision_idx": decision,
                        "decision_time_brt": DECISION_TIMES[decision].isoformat(),
                        "horizon_minutes": horizon,
                        "mean_spearman_ic": _finite_or_none(np.nanmean(values)),
                        "ic_interval_lower_95": _finite_or_none(
                            time_bootstrap["lower_95"][flat_index]
                        ),
                        "ic_interval_upper_95": _finite_or_none(
                            time_bootstrap["upper_95"][flat_index]
                        ),
                        "ic_bootstrap_finite_replication_count": int(
                            time_bootstrap["finite_replication_count"][flat_index]
                        ),
                        "mean_gross_top_return": _finite_or_none(
                            np.nanmean(daily_top_return[:, decision, horizon_index])
                        ),
                        "mean_gross_bottom_return": _finite_or_none(
                            np.nanmean(daily_bottom_return[:, decision, horizon_index])
                        ),
                        "mean_gross_top_minus_bottom": _finite_or_none(
                            np.nanmean(daily_spread[:, decision, horizon_index])
                        ),
                        "mean_intraday_one_way_turnover": _finite_or_none(
                            np.nanmean(daily_turnover[:, decision, horizon_index])
                        ),
                        "mean_flat_entry_turnover": _finite_or_none(
                            np.nanmean(daily_entry[:, decision, horizon_index])
                        ),
                        "mean_flat_exit_turnover": _finite_or_none(
                            np.nanmean(daily_exit[:, decision, horizon_index])
                        ),
                        **coverage,
                        "finite_ic_date_count": int(np.isfinite(values).sum()),
                    }
                )
            primary_values = np.nanmean(daily_ic[:, decision], axis=1)
            primary_bootstrap = moving_block_bootstrap(
                primary_values,
                replications=bootstrap_replications,
                seed=BOOTSTRAP_SEED + seed * 1000 + decision,
            )
            primary_coverage = _coverage_summary(
                daily_valid_count, daily_coverage, (decision,), None
            )
            time_5m_rows.append(
                {
                    "aggregation": "seed",
                    "seed": seed,
                    "decision_idx": decision,
                    "decision_time_brt": DECISION_TIMES[decision].isoformat(),
                    "horizon_minutes": 0,
                    "mean_spearman_ic": _finite_or_none(np.nanmean(primary_values)),
                    "ic_interval_lower_95": primary_bootstrap["interval_lower_95"],
                    "ic_interval_upper_95": primary_bootstrap["interval_upper_95"],
                    "ic_bootstrap_finite_replication_count": int(
                        primary_bootstrap["finite_replication_count"]
                    ),
                    "mean_gross_top_return": _finite_or_none(
                        np.nanmean(daily_top_return[:, decision])
                    ),
                    "mean_gross_bottom_return": _finite_or_none(
                        np.nanmean(daily_bottom_return[:, decision])
                    ),
                    "mean_gross_top_minus_bottom": _finite_or_none(
                        np.nanmean(daily_spread[:, decision])
                    ),
                    "mean_intraday_one_way_turnover": _finite_or_none(
                        np.nanmean(daily_turnover[:, decision])
                    ),
                    "mean_flat_entry_turnover": _finite_or_none(
                        np.nanmean(daily_entry[:, decision])
                    ),
                    "mean_flat_exit_turnover": _finite_or_none(
                        np.nanmean(daily_exit[:, decision])
                    ),
                    **primary_coverage,
                    "finite_ic_date_count": int(np.isfinite(primary_values).sum()),
                }
            )
        scope_daily: dict[str, dict[str, np.ndarray]] = {}
        for scope_index, (scope_name, decisions) in enumerate(scopes.items()):
            ic_values = _scope_daily_mean(daily_ic, decisions)
            scope_accounting = economic_window_accounting(
                economic, date_idx, decision_idx, decisions=decisions
            )
            scope_economic = _window_portfolio_daily(scope_accounting)
            spread_values = scope_economic["gross"]
            top_return_values = _window_portfolio_daily(
                economic_window_accounting(
                    economic,
                    date_idx,
                    decision_idx,
                    decisions=decisions,
                    gross_contributions=top_contributions,
                )
            )["gross"]
            bottom_return_values = -_window_portfolio_daily(
                economic_window_accounting(
                    economic,
                    date_idx,
                    decision_idx,
                    decisions=decisions,
                    gross_contributions=bottom_contributions,
                )
            )["gross"]
            turnover_values = scope_economic["intraday"]
            entry_values = scope_economic["entry"]
            exit_values = scope_economic["exit"]
            scope_daily[scope_name] = {
                "ic": ic_values,
                "spread": spread_values,
                "top_return": top_return_values,
                "bottom_return": bottom_return_values,
                "turnover": turnover_values,
                "entry": entry_values,
                "exit": exit_values,
            }
            scope_bootstrap = moving_block_bootstrap_matrix(
                ic_values,
                replications=bootstrap_replications,
                seed=BOOTSTRAP_SEED + seed * 100 + scope_index,
            )
            for horizon_index, horizon in enumerate(HORIZONS):
                coverage = _coverage_summary(
                    daily_valid_count, daily_coverage, decisions, horizon_index
                )
                time_bin_rows.append(
                    {
                        "aggregation": "seed",
                        "seed": seed,
                        "scope": scope_name,
                        "decision_indices": json.dumps(list(decisions)),
                        "start_time_brt": DECISION_TIMES[decisions[0]].isoformat(),
                        "end_time_brt": DECISION_TIMES[decisions[-1]].isoformat(),
                        "decision_count": len(decisions),
                        "horizon_minutes": horizon,
                        "mean_spearman_ic": _finite_or_none(
                            np.nanmean(ic_values[:, horizon_index])
                        ),
                        "ic_interval_lower_95": _finite_or_none(
                            scope_bootstrap["lower_95"][horizon_index]
                        ),
                        "ic_interval_upper_95": _finite_or_none(
                            scope_bootstrap["upper_95"][horizon_index]
                        ),
                        "ic_bootstrap_finite_replication_count": int(
                            scope_bootstrap["finite_replication_count"][horizon_index]
                        ),
                        "mean_gross_top_return": _finite_or_none(
                            np.nanmean(top_return_values[:, horizon_index])
                        ),
                        "mean_gross_bottom_return": _finite_or_none(
                            np.nanmean(bottom_return_values[:, horizon_index])
                        ),
                        "mean_gross_top_minus_bottom": _finite_or_none(
                            np.nanmean(spread_values[:, horizon_index])
                        ),
                        "mean_intraday_one_way_turnover": _finite_or_none(
                            np.nanmean(turnover_values[:, horizon_index])
                        ),
                        "mean_flat_entry_turnover": _finite_or_none(
                            np.nanmean(entry_values[:, horizon_index])
                        ),
                        "mean_flat_exit_turnover": _finite_or_none(
                            np.nanmean(exit_values[:, horizon_index])
                        ),
                        **coverage,
                        "finite_ic_date_count": int(
                            np.isfinite(ic_values[:, horizon_index]).sum()
                        ),
                    }
                )
            primary_daily = np.nanmean(ic_values, axis=1)
            primary_bootstrap = moving_block_bootstrap(
                primary_daily,
                replications=bootstrap_replications,
                seed=BOOTSTRAP_SEED + seed * 1000 + scope_index,
            )
            primary_coverage = _coverage_summary(
                daily_valid_count, daily_coverage, decisions, None
            )
            time_bin_rows.append(
                {
                    "aggregation": "seed",
                    "seed": seed,
                    "scope": scope_name,
                    "decision_indices": json.dumps(list(decisions)),
                    "start_time_brt": DECISION_TIMES[decisions[0]].isoformat(),
                    "end_time_brt": DECISION_TIMES[decisions[-1]].isoformat(),
                    "decision_count": len(decisions),
                    "horizon_minutes": 0,
                    "mean_spearman_ic": _finite_or_none(np.nanmean(primary_daily)),
                    "ic_interval_lower_95": _finite_or_none(
                        primary_bootstrap["interval_lower_95"]
                    ),
                    "ic_interval_upper_95": _finite_or_none(
                        primary_bootstrap["interval_upper_95"]
                    ),
                    "ic_bootstrap_finite_replication_count": int(
                        primary_bootstrap["finite_replication_count"]
                    ),
                    "mean_gross_top_return": _finite_or_none(
                        np.nanmean(top_return_values)
                    ),
                    "mean_gross_bottom_return": _finite_or_none(
                        np.nanmean(bottom_return_values)
                    ),
                    "mean_gross_top_minus_bottom": _finite_or_none(
                        np.nanmean(spread_values)
                    ),
                    "mean_intraday_one_way_turnover": _finite_or_none(
                        np.nanmean(turnover_values)
                    ),
                    "mean_flat_entry_turnover": _finite_or_none(
                        np.nanmean(entry_values)
                    ),
                    "mean_flat_exit_turnover": _finite_or_none(np.nanmean(exit_values)),
                    **primary_coverage,
                    "finite_ic_date_count": int(np.isfinite(primary_daily).sum()),
                }
            )
        bin_daily = []
        bin_counts = []
        for row in bin_metadata:
            decisions = tuple(row["decision_indices"])
            selected = daily_ic[:, decisions]
            bin_daily.append(np.nanmean(selected, axis=1))
            bin_counts.append(np.isfinite(selected).sum(axis=1))
        numerator = sum(
            values * counts
            for values, counts in zip(bin_daily, bin_counts, strict=True)
        )
        denominator = sum(bin_counts)
        reconstructed_daily = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 0,
        )
        all_day_daily = np.nanmean(daily_ic, axis=1)
        time_reconstruction_difference = float(
            np.nanmax(np.abs(reconstructed_daily - all_day_daily))
        )
        if time_reconstruction_difference > RECONSTRUCTION_ABSOLUTE_TOLERANCE:
            raise RuntimeError("Weighted time bins failed to reconstruct daily IC")
        _, contribution_grid = _grid_from_stock_values(
            additive.contributions, date_idx, decision_idx
        )
        for decision in range(EXPECTED_DECISIONS_PER_DATE):
            mean_contribution = _finite_axis_mean(
                contribution_grid[:, decision], axis=0
            )
            for horizon_index, horizon in enumerate(HORIZONS):
                for equity, identity in enumerate(identities):
                    stock_time_rows.append(
                        {
                            **identity,
                            "aggregation": "seed",
                            "seed": seed,
                            "scope_type": "decision_5m",
                            "scope": f"decision_{decision:02d}",
                            "decision_idx": decision,
                            "decision_time_brt": DECISION_TIMES[decision].isoformat(),
                            "horizon_minutes": horizon,
                            "additive_ic_contribution": float(
                                mean_contribution[equity, horizon_index]
                            ),
                            "exploratory": True,
                        }
                    )
            for horizon_index in range(len(HORIZONS)):
                expected_cell = float(
                    _finite_axis_mean(daily_ic[:, decision, horizon_index], axis=0)
                )
                if not math.isclose(
                    float(mean_contribution[:, horizon_index].sum()),
                    expected_cell,
                    rel_tol=0.0,
                    abs_tol=RECONSTRUCTION_ABSOLUTE_TOLERANCE,
                ):
                    raise RuntimeError(
                        "Stock/time decision cell failed additive reconstruction"
                    )
            decision_daily_stock = _finite_axis_mean(
                contribution_grid[:, decision], axis=2
            )
            primary_contribution = _finite_axis_mean(decision_daily_stock, axis=0)
            expected_primary_cell = float(
                _finite_axis_mean(
                    _finite_axis_mean(daily_ic[:, decision], axis=1), axis=0
                )
            )
            if not math.isclose(
                float(primary_contribution.sum()),
                expected_primary_cell,
                rel_tol=0.0,
                abs_tol=RECONSTRUCTION_ABSOLUTE_TOLERANCE,
            ):
                raise RuntimeError(
                    "Stock/time primary decision cell failed additive reconstruction"
                )
            for equity, identity in enumerate(identities):
                stock_time_rows.append(
                    {
                        **identity,
                        "aggregation": "seed",
                        "seed": seed,
                        "scope_type": "decision_5m",
                        "scope": f"decision_{decision:02d}",
                        "decision_idx": decision,
                        "decision_time_brt": DECISION_TIMES[decision].isoformat(),
                        "horizon_minutes": 0,
                        "additive_ic_contribution": float(primary_contribution[equity]),
                        "exploratory": True,
                    }
                )
        for scope_name, decisions in scopes.items():
            daily_stock = _finite_axis_mean(contribution_grid[:, decisions], axis=1)
            mean_stock = _finite_axis_mean(daily_stock, axis=0)
            for horizon_index, horizon in enumerate(HORIZONS):
                for equity, identity in enumerate(identities):
                    stock_time_rows.append(
                        {
                            **identity,
                            "aggregation": "seed",
                            "seed": seed,
                            "scope_type": (
                                "primary_bin"
                                if scope_name.startswith("bin_")
                                else "named_session_scope"
                            ),
                            "scope": scope_name,
                            "decision_idx": None,
                            "decision_time_brt": None,
                            "horizon_minutes": horizon,
                            "additive_ic_contribution": float(
                                mean_stock[equity, horizon_index]
                            ),
                            "exploratory": scope_name.startswith("bin_"),
                        }
                    )
            scope_ic = _scope_daily_mean(daily_ic, decisions)
            for horizon_index in range(len(HORIZONS)):
                expected_cell = float(
                    _finite_axis_mean(scope_ic[:, horizon_index], axis=0)
                )
                if not math.isclose(
                    float(mean_stock[:, horizon_index].sum()),
                    expected_cell,
                    rel_tol=0.0,
                    abs_tol=RECONSTRUCTION_ABSOLUTE_TOLERANCE,
                ):
                    raise RuntimeError(
                        "Stock/time scope cell failed additive reconstruction"
                    )
            primary_stock = _finite_axis_mean(
                _finite_axis_mean(daily_stock, axis=2), axis=0
            )
            expected_primary_cell = float(
                _finite_axis_mean(_finite_axis_mean(scope_ic, axis=1), axis=0)
            )
            if not math.isclose(
                float(primary_stock.sum()),
                expected_primary_cell,
                rel_tol=0.0,
                abs_tol=RECONSTRUCTION_ABSOLUTE_TOLERANCE,
            ):
                raise RuntimeError(
                    "Stock/time primary scope cell failed additive reconstruction"
                )
            for equity, identity in enumerate(identities):
                stock_time_rows.append(
                    {
                        **identity,
                        "aggregation": "seed",
                        "seed": seed,
                        "scope_type": (
                            "primary_bin"
                            if scope_name.startswith("bin_")
                            else "named_session_scope"
                        ),
                        "scope": scope_name,
                        "decision_idx": None,
                        "decision_time_brt": None,
                        "horizon_minutes": 0,
                        "additive_ic_contribution": float(primary_stock[equity]),
                        "exploratory": scope_name.startswith("bin_"),
                    }
                )
        first_half = trade_dates <= np.datetime64(date(2024, 12, 31))
        latest_half = trade_dates >= np.datetime64(date(2025, 1, 1))
        first_aggregate = _period_additive_summary(additive, date_idx, first_half)
        latest_aggregate = _period_additive_summary(additive, date_idx, latest_half)
        scope_contributions = {}
        for name in (
            "opening_30",
            "opening_60",
            "rest_of_day",
            "midday",
            "late_session",
        ):
            selected = np.isin(decision_idx, named_time_scopes()[name])
            scope_contributions[name] = _period_additive_summary(
                additive, date_idx, selected
            )["primary_contributions"]
        all_day_accounting = economic_window_accounting(
            economic, date_idx, decision_idx
        )
        all_day_means = _window_stock_means(all_day_accounting)
        long_means = _window_stock_means(
            economic_window_accounting(
                economic,
                date_idx,
                decision_idx,
                gross_contributions=top_contributions,
            )
        )
        short_means = _window_stock_means(
            economic_window_accounting(
                economic,
                date_idx,
                decision_idx,
                gross_contributions=bottom_contributions,
            )
        )
        economic_aggregate = {
            "primary": _finite_axis_mean(all_day_means["gross"], axis=1)
        }
        long_aggregate = {"primary": _finite_axis_mean(long_means["gross"], axis=1)}
        short_aggregate = {"primary": _finite_axis_mean(short_means["gross"], axis=1)}
        turnover_aggregate = {
            "primary": _finite_axis_mean(all_day_means["intraday"], axis=1)
        }
        entry_aggregate = {"primary": _finite_axis_mean(all_day_means["entry"], axis=1)}
        exit_aggregate = {"primary": _finite_axis_mean(all_day_means["exit"], axis=1)}
        skill = per_stock_time_series_skill(
            predictions,
            targets,
            label_mask,
            date_idx,
            bootstrap_replications=bootstrap_replications,
            bootstrap_seed=BOOTSTRAP_SEED + seed,
        )
        valid_sample_count = label_mask.any(axis=2).sum(axis=0)
        valid_decision_count = np.asarray(
            [
                np.unique(decision_idx[label_mask[:, equity].any(axis=1)]).size
                for equity in range(label_mask.shape[1])
            ],
            dtype=np.int64,
        )
        opportunity = stock_contribution_opportunity_accounting(additive, label_mask)
        valid_opportunity_count = opportunity["valid_opportunity_count"]
        conditional_numerator = opportunity["conditional_contribution_numerator"]
        conditional_contribution = opportunity["conditional_contribution"]
        portfolio_valid_cell_count = opportunity["portfolio_valid_cell_count"]
        unconditional_numerator = opportunity["unconditional_contribution_numerator"]
        unconditional_contribution = opportunity["unconditional_contribution"]
        valid_date_count = np.asarray(
            [
                np.unique(date_idx[label_mask[:, equity].any(axis=1)]).size
                for equity in range(label_mask.shape[1])
            ],
            dtype=np.int64,
        )
        top_count = economic.top_selected.sum(axis=(0, 2))
        bottom_count = economic.bottom_selected.sum(axis=(0, 2))
        selected_count = top_count + bottom_count
        signed_sum = economic.signed_selected_return.sum(axis=(0, 2))
        selected_hits = (
            (economic.signed_selected_return > 0)
            & (economic.top_selected | economic.bottom_selected)
        ).sum(axis=(0, 2))
        mean_liquidity = _finite_axis_mean(dollar_liquidity, axis=0)
        selected_liquidity_sum = np.zeros(label_mask.shape[1], dtype=np.float64)
        for sample in range(label_mask.shape[0]):
            selected_by_horizon = (
                economic.top_selected[sample] | economic.bottom_selected[sample]
            ).sum(axis=1)
            selected_liquidity_sum += (
                np.nan_to_num(dollar_liquidity[date_positions[sample]], nan=0.0)
                * selected_by_horizon
            )
        selected_liquidity = np.divide(
            selected_liquidity_sum,
            selected_count,
            out=np.full_like(selected_liquidity_sum, np.nan),
            where=selected_count > 0,
        )
        stock_seed_arrays[seed] = {
            "primary": np.asarray(aggregate["primary_contributions"]),
            "horizon": np.asarray(aggregate["horizon_contributions"]),
            "daily_primary": np.nanmean(
                np.asarray(aggregate["daily_contributions"]), axis=2
            ),
            "first_half": np.asarray(first_aggregate["primary_contributions"]),
            "latest_half": np.asarray(latest_aggregate["primary_contributions"]),
            "opening_30": np.asarray(scope_contributions["opening_30"]),
            "opening_60": np.asarray(scope_contributions["opening_60"]),
            "rest_of_day": np.asarray(scope_contributions["rest_of_day"]),
            "midday": np.asarray(scope_contributions["midday"]),
            "late_session": np.asarray(scope_contributions["late_session"]),
            "economic": np.asarray(economic_aggregate["primary"]),
            "long_economic": np.asarray(long_aggregate["primary"]),
            "short_economic": np.asarray(short_aggregate["primary"]),
            "turnover": np.asarray(turnover_aggregate["primary"]),
            "entry_turnover": np.asarray(entry_aggregate["primary"]),
            "exit_turnover": np.asarray(exit_aggregate["primary"]),
            "valid_sample_count": valid_sample_count,
            "valid_decision_count": valid_decision_count,
            "valid_opportunity_count": valid_opportunity_count,
            "conditional_contribution_numerator": conditional_numerator,
            "conditional_contribution": conditional_contribution,
            "portfolio_valid_cell_count": portfolio_valid_cell_count,
            "unconditional_contribution_numerator": unconditional_numerator,
            "unconditional_contribution": unconditional_contribution,
            "valid_date_count": valid_date_count,
            "top_count": top_count,
            "bottom_count": bottom_count,
            "selected_count": selected_count,
            "signed_return_sum": signed_sum,
            "selected_hit_count": selected_hits,
            "mean_liquidity": mean_liquidity,
            "selected_liquidity": selected_liquidity,
            "skill": _finite_axis_mean(skill["skill"], axis=1),
            "skill_valid_days": np.nanmin(skill["valid_day_count"], axis=1),
            "skill_coverage": np.nanmin(skill["coverage"], axis=1),
        }
        time_grids[seed] = {
            "ic": daily_ic,
            "spread": daily_spread,
            "top_return": daily_top_return,
            "bottom_return": daily_bottom_return,
            "turnover": daily_turnover,
            "entry": daily_entry,
            "exit": daily_exit,
        }
        core_by_seed[seed] = {
            "predictions_path": cache_paths[("core", seed)],
            "sample_ic": additive.sample_ic,
            "contributions": additive.contributions,
            "economic": economic,
            "daily_ic": daily_ic,
            "scope_daily": scope_daily,
        }
        reconstruction[str(seed)] = {
            "primary_ic": float(aggregate["primary_ic"]),
            "stock_contribution_sum": float(
                np.asarray(aggregate["primary_contributions"]).sum()
            ),
            "stock_absolute_difference": abs(
                float(aggregate["primary_ic"])
                - float(np.asarray(aggregate["primary_contributions"]).sum())
            ),
            "time_decomposition_maximum_absolute_difference": (
                time_reconstruction_difference
            ),
            "economic": economic_check,
        }
    stock_bootstrap_input = np.mean(
        np.stack([stock_seed_arrays[seed]["daily_primary"] for seed in STAGE3_SEEDS]),
        axis=0,
    )
    stock_bootstrap = moving_block_bootstrap_matrix(
        stock_bootstrap_input,
        replications=bootstrap_replications,
        seed=BOOTSTRAP_SEED,
    )
    stock_rows: list[dict[str, object]] = []
    primary_by_seed = np.stack(
        [stock_seed_arrays[seed]["primary"] for seed in STAGE3_SEEDS]
    )
    horizon_by_seed = np.stack(
        [stock_seed_arrays[seed]["horizon"] for seed in STAGE3_SEEDS]
    )
    mean_primary = primary_by_seed.mean(axis=0)
    net_primary = float(mean_primary.sum())
    portfolio_positive_mass = float(mean_primary[mean_primary > 0].sum())
    portfolio_negative_mass = float(mean_primary[mean_primary < 0].sum())
    for equity, identity in enumerate(identities):
        seed_values = primary_by_seed[:, equity]
        positive, zero, negative = _sign_counts(seed_values)
        economic_values = np.asarray(
            [stock_seed_arrays[seed]["economic"][equity] for seed in STAGE3_SEEDS]
        )
        long_economic_values = np.asarray(
            [stock_seed_arrays[seed]["long_economic"][equity] for seed in STAGE3_SEEDS]
        )
        short_economic_values = np.asarray(
            [stock_seed_arrays[seed]["short_economic"][equity] for seed in STAGE3_SEEDS]
        )
        turnover_values = np.asarray(
            [stock_seed_arrays[seed]["turnover"][equity] for seed in STAGE3_SEEDS]
        )
        entry_values = np.asarray(
            [stock_seed_arrays[seed]["entry_turnover"][equity] for seed in STAGE3_SEEDS]
        )
        exit_values = np.asarray(
            [stock_seed_arrays[seed]["exit_turnover"][equity] for seed in STAGE3_SEEDS]
        )
        total_turnover = float(np.nanmean(turnover_values + entry_values + exit_values))
        gross_contribution = float(np.nanmean(economic_values))
        selected_count_mean = float(
            np.mean(
                [
                    stock_seed_arrays[seed]["selected_count"][equity]
                    for seed in STAGE3_SEEDS
                ]
            )
        )
        signed_sum_mean = float(
            np.mean(
                [
                    stock_seed_arrays[seed]["signed_return_sum"][equity]
                    for seed in STAGE3_SEEDS
                ]
            )
        )
        hit_count_mean = float(
            np.mean(
                [
                    stock_seed_arrays[seed]["selected_hit_count"][equity]
                    for seed in STAGE3_SEEDS
                ]
            )
        )
        liquidity = float(
            np.nanmean(
                [
                    stock_seed_arrays[seed]["mean_liquidity"][equity]
                    for seed in STAGE3_SEEDS
                ]
            )
        )
        valid_opportunities = int(
            sum(
                stock_seed_arrays[seed]["valid_opportunity_count"][equity]
                for seed in STAGE3_SEEDS
            )
        )
        conditional_numerator = float(
            sum(
                stock_seed_arrays[seed]["conditional_contribution_numerator"][equity]
                for seed in STAGE3_SEEDS
            )
        )
        portfolio_valid_cells = int(
            sum(
                stock_seed_arrays[seed]["portfolio_valid_cell_count"][equity]
                for seed in STAGE3_SEEDS
            )
        )
        unconditional_numerator = float(
            sum(
                stock_seed_arrays[seed]["unconditional_contribution_numerator"][equity]
                for seed in STAGE3_SEEDS
            )
        )
        stock_rows.append(
            {
                **identity,
                "additive_primary_ic_contribution": float(mean_primary[equity]),
                "contribution_share_of_net_primary_ic": (
                    float(mean_primary[equity] / net_primary)
                    if net_primary != 0.0
                    else None
                ),
                "positive_contribution_mass": float(max(mean_primary[equity], 0.0)),
                "negative_contribution_drag": float(min(mean_primary[equity], 0.0)),
                "portfolio_positive_contribution_mass": portfolio_positive_mass,
                "portfolio_negative_contribution_mass": portfolio_negative_mass,
                "portfolio_net_primary_ic": net_primary,
                "contribution_per_valid_opportunity": (
                    conditional_numerator / valid_opportunities
                    if valid_opportunities > 0
                    else None
                ),
                "contribution_per_valid_opportunity_unit": (
                    "additive Spearman contribution per label-valid stock/horizon "
                    "in a portfolio-valid cross-section, pooled across seeds"
                ),
                "contribution_per_portfolio_valid_cell": (
                    unconditional_numerator / portfolio_valid_cells
                    if portfolio_valid_cells > 0
                    else None
                ),
                "contribution_per_portfolio_valid_cell_unit": (
                    "additive Spearman contribution per stock/horizon in a "
                    "portfolio-valid cross-section, pooled across seeds"
                ),
                "portfolio_valid_cell_count": portfolio_valid_cells,
                "valid_sample_count": int(
                    min(
                        stock_seed_arrays[seed]["valid_sample_count"][equity]
                        for seed in STAGE3_SEEDS
                    )
                ),
                "valid_decision_count": int(
                    min(
                        stock_seed_arrays[seed]["valid_decision_count"][equity]
                        for seed in STAGE3_SEEDS
                    )
                ),
                "valid_opportunity_count": valid_opportunities,
                "minimum_seed_valid_opportunity_count": int(
                    min(
                        stock_seed_arrays[seed]["valid_opportunity_count"][equity]
                        for seed in STAGE3_SEEDS
                    )
                ),
                "valid_date_count": int(
                    min(
                        stock_seed_arrays[seed]["valid_date_count"][equity]
                        for seed in STAGE3_SEEDS
                    )
                ),
                **{
                    f"contribution_{horizon}m": float(
                        horizon_by_seed[:, equity, horizon_index].mean()
                    )
                    for horizon_index, horizon in enumerate(HORIZONS)
                },
                "first_half_contribution": float(
                    np.mean(
                        [
                            stock_seed_arrays[seed]["first_half"][equity]
                            for seed in STAGE3_SEEDS
                        ]
                    )
                ),
                "latest_half_contribution": float(
                    np.mean(
                        [
                            stock_seed_arrays[seed]["latest_half"][equity]
                            for seed in STAGE3_SEEDS
                        ]
                    )
                ),
                **{
                    f"seed_{seed}_contribution": float(
                        stock_seed_arrays[seed]["primary"][equity]
                    )
                    for seed in STAGE3_SEEDS
                },
                "across_seed_mean_contribution": float(seed_values.mean()),
                "across_seed_median_contribution": float(np.median(seed_values)),
                "across_seed_minimum_contribution": float(seed_values.min()),
                "across_seed_maximum_contribution": float(seed_values.max()),
                "positive_seed_count": positive,
                "zero_seed_count": zero,
                "negative_seed_count": negative,
                "bootstrap_interval_lower_95": float(
                    stock_bootstrap["lower_95"][equity]
                ),
                "bootstrap_interval_upper_95": float(
                    stock_bootstrap["upper_95"][equity]
                ),
                "bootstrap_probability_positive": float(
                    stock_bootstrap["probability_positive"][equity]
                ),
                "bootstrap_probability_negative": float(
                    stock_bootstrap["probability_negative"][equity]
                ),
                "bootstrap_finite_replication_count": int(
                    stock_bootstrap["finite_replication_count"][equity]
                ),
                "opening_30_contribution": float(
                    np.mean(
                        [
                            stock_seed_arrays[seed]["opening_30"][equity]
                            for seed in STAGE3_SEEDS
                        ]
                    )
                ),
                "opening_60_contribution": float(
                    np.mean(
                        [
                            stock_seed_arrays[seed]["opening_60"][equity]
                            for seed in STAGE3_SEEDS
                        ]
                    )
                ),
                "midday_contribution": float(
                    np.mean(
                        [
                            stock_seed_arrays[seed]["midday"][equity]
                            for seed in STAGE3_SEEDS
                        ]
                    )
                ),
                "late_session_contribution": float(
                    np.mean(
                        [
                            stock_seed_arrays[seed]["late_session"][equity]
                            for seed in STAGE3_SEEDS
                        ]
                    )
                ),
                "rest_of_day_contribution": float(
                    np.mean(
                        [
                            stock_seed_arrays[seed]["rest_of_day"][equity]
                            for seed in STAGE3_SEEDS
                        ]
                    )
                ),
                "opening_minus_rest_contribution": float(
                    np.mean(
                        [
                            stock_seed_arrays[seed]["opening_60"][equity]
                            - stock_seed_arrays[seed]["rest_of_day"][equity]
                            for seed in STAGE3_SEEDS
                        ]
                    )
                ),
                "per_stock_time_series_skill": _finite_mean_or_none(
                    np.asarray(
                        [
                            stock_seed_arrays[seed]["skill"][equity]
                            for seed in STAGE3_SEEDS
                        ],
                        dtype=np.float64,
                    )
                ),
                "per_stock_time_series_skill_minimum_valid_days": int(
                    np.nanmin(
                        [
                            stock_seed_arrays[seed]["skill_valid_days"][equity]
                            for seed in STAGE3_SEEDS
                        ]
                    )
                ),
                "per_stock_time_series_skill_minimum_coverage": float(
                    np.nanmin(
                        [
                            stock_seed_arrays[seed]["skill_coverage"][equity]
                            for seed in STAGE3_SEEDS
                        ]
                    )
                ),
                "top_selection_frequency": (
                    float(
                        sum(
                            stock_seed_arrays[seed]["top_count"][equity]
                            for seed in STAGE3_SEEDS
                        )
                        / valid_opportunities
                    )
                    if valid_opportunities > 0
                    else None
                ),
                "bottom_selection_frequency": (
                    float(
                        sum(
                            stock_seed_arrays[seed]["bottom_count"][equity]
                            for seed in STAGE3_SEEDS
                        )
                        / valid_opportunities
                    )
                    if valid_opportunities > 0
                    else None
                ),
                "net_gross_spread_contribution": gross_contribution,
                "mean_daily_gross_contribution": gross_contribution,
                "long_gross_return_contribution": float(
                    np.nanmean(long_economic_values)
                ),
                "short_gross_return_contribution": float(
                    np.nanmean(short_economic_values)
                ),
                "mean_signed_return_when_selected": (
                    signed_sum_mean / selected_count_mean
                    if selected_count_mean > 0
                    else None
                ),
                "hit_rate_when_selected": (
                    hit_count_mean / selected_count_mean
                    if selected_count_mean > 0
                    else None
                ),
                "intraday_one_way_turnover_contribution": float(
                    np.nanmean(turnover_values)
                ),
                "flat_entry_turnover_contribution": float(np.nanmean(entry_values)),
                "flat_exit_turnover_contribution": float(np.nanmean(exit_values)),
                "mean_daily_flat_entry_turnover": float(np.nanmean(entry_values)),
                "mean_daily_intraday_one_way_turnover": float(
                    np.nanmean(turnover_values)
                ),
                "mean_daily_flat_exit_turnover": float(np.nanmean(exit_values)),
                "mean_daily_total_one_way_turnover": total_turnover,
                "gross_return_unit": "decimal return summed within date then averaged across dates",
                "turnover_unit": "one-way notional fraction summed within date then averaged across dates",
                "gross_contribution_per_unit_turnover": (
                    gross_contribution / total_turnover if total_turnover > 0 else None
                ),
                "break_even_one_way_cost_bps": (
                    10_000.0 * gross_contribution / total_turnover
                    if total_turnover > 0
                    else None
                ),
                "mean_point_in_time_dollar_liquidity_brl": liquidity,
                "mean_point_in_time_dollar_liquidity_when_selected_brl": _finite_or_none(
                    np.nanmean(
                        [
                            stock_seed_arrays[seed]["selected_liquidity"][equity]
                            for seed in STAGE3_SEEDS
                        ]
                    )
                ),
                "liquidity_distance_from_eligibility_threshold_brl": (
                    liquidity - threshold if threshold is not None else None
                ),
                "liquidity_ratio_to_eligibility_threshold": (
                    liquidity / threshold
                    if threshold is not None and threshold > 0
                    else None
                ),
            }
        )
    for decision in range(EXPECTED_DECISIONS_PER_DATE):
        for horizon_index, horizon in enumerate(HORIZONS):
            ic_by_seed = np.asarray(
                [
                    np.nanmean(time_grids[seed]["ic"][:, decision, horizon_index])
                    for seed in STAGE3_SEEDS
                ]
            )
            positive, zero, negative = _sign_counts(ic_by_seed)
            across_daily = np.mean(
                np.stack(
                    [
                        time_grids[seed]["ic"][:, decision, horizon_index]
                        for seed in STAGE3_SEEDS
                    ]
                ),
                axis=0,
            )
            bootstrap = moving_block_bootstrap(
                across_daily,
                replications=bootstrap_replications,
                seed=BOOTSTRAP_SEED + decision * 10 + horizon_index,
            )
            coverage = _coverage_summary(
                daily_valid_count, daily_coverage, (decision,), horizon_index
            )
            time_5m_rows.append(
                {
                    "aggregation": "across_seed",
                    "seed": None,
                    "decision_idx": decision,
                    "decision_time_brt": DECISION_TIMES[decision].isoformat(),
                    "horizon_minutes": horizon,
                    "mean_spearman_ic": float(ic_by_seed.mean()),
                    "ic_interval_lower_95": bootstrap["interval_lower_95"],
                    "ic_interval_upper_95": bootstrap["interval_upper_95"],
                    "ic_bootstrap_finite_replication_count": int(
                        bootstrap["finite_replication_count"]
                    ),
                    "mean_gross_top_return": float(
                        np.mean(
                            [
                                np.nanmean(
                                    time_grids[seed]["top_return"][
                                        :, decision, horizon_index
                                    ]
                                )
                                for seed in STAGE3_SEEDS
                            ]
                        )
                    ),
                    "mean_gross_bottom_return": float(
                        np.mean(
                            [
                                np.nanmean(
                                    time_grids[seed]["bottom_return"][
                                        :, decision, horizon_index
                                    ]
                                )
                                for seed in STAGE3_SEEDS
                            ]
                        )
                    ),
                    "mean_gross_top_minus_bottom": float(
                        np.mean(
                            [
                                np.nanmean(
                                    time_grids[seed]["spread"][
                                        :, decision, horizon_index
                                    ]
                                )
                                for seed in STAGE3_SEEDS
                            ]
                        )
                    ),
                    "mean_intraday_one_way_turnover": float(
                        np.mean(
                            [
                                np.nanmean(
                                    time_grids[seed]["turnover"][
                                        :, decision, horizon_index
                                    ]
                                )
                                for seed in STAGE3_SEEDS
                            ]
                        )
                    ),
                    "mean_flat_entry_turnover": float(
                        np.mean(
                            [
                                np.nanmean(
                                    time_grids[seed]["entry"][
                                        :, decision, horizon_index
                                    ]
                                )
                                for seed in STAGE3_SEEDS
                            ]
                        )
                    ),
                    "mean_flat_exit_turnover": float(
                        np.mean(
                            [
                                np.nanmean(
                                    time_grids[seed]["exit"][:, decision, horizon_index]
                                )
                                for seed in STAGE3_SEEDS
                            ]
                        )
                    ),
                    **coverage,
                    "finite_ic_date_count": int(np.isfinite(across_daily).sum()),
                    "across_seed_minimum_ic": float(ic_by_seed.min()),
                    "across_seed_maximum_ic": float(ic_by_seed.max()),
                    "positive_seed_count": positive,
                    "zero_seed_count": zero,
                    "negative_seed_count": negative,
                }
            )
        primary_seed_daily = np.stack(
            [
                np.nanmean(time_grids[seed]["ic"][:, decision], axis=1)
                for seed in STAGE3_SEEDS
            ]
        )
        ic_by_seed = np.nanmean(primary_seed_daily, axis=1)
        positive, zero, negative = _sign_counts(ic_by_seed)
        across_daily = np.nanmean(primary_seed_daily, axis=0)
        bootstrap = moving_block_bootstrap(
            across_daily,
            replications=bootstrap_replications,
            seed=BOOTSTRAP_SEED + decision * 10 + len(HORIZONS),
        )
        primary_coverage = _coverage_summary(
            daily_valid_count, daily_coverage, (decision,), None
        )
        time_5m_rows.append(
            {
                "aggregation": "across_seed",
                "seed": None,
                "decision_idx": decision,
                "decision_time_brt": DECISION_TIMES[decision].isoformat(),
                "horizon_minutes": 0,
                "mean_spearman_ic": float(np.nanmean(ic_by_seed)),
                "ic_interval_lower_95": bootstrap["interval_lower_95"],
                "ic_interval_upper_95": bootstrap["interval_upper_95"],
                "ic_bootstrap_finite_replication_count": int(
                    bootstrap["finite_replication_count"]
                ),
                "mean_gross_top_return": float(
                    np.mean(
                        [
                            np.nanmean(time_grids[seed]["top_return"][:, decision])
                            for seed in STAGE3_SEEDS
                        ]
                    )
                ),
                "mean_gross_bottom_return": float(
                    np.mean(
                        [
                            np.nanmean(time_grids[seed]["bottom_return"][:, decision])
                            for seed in STAGE3_SEEDS
                        ]
                    )
                ),
                "mean_gross_top_minus_bottom": float(
                    np.mean(
                        [
                            np.nanmean(time_grids[seed]["spread"][:, decision])
                            for seed in STAGE3_SEEDS
                        ]
                    )
                ),
                "mean_intraday_one_way_turnover": float(
                    np.mean(
                        [
                            np.nanmean(time_grids[seed]["turnover"][:, decision])
                            for seed in STAGE3_SEEDS
                        ]
                    )
                ),
                "mean_flat_entry_turnover": float(
                    np.mean(
                        [
                            np.nanmean(time_grids[seed]["entry"][:, decision])
                            for seed in STAGE3_SEEDS
                        ]
                    )
                ),
                "mean_flat_exit_turnover": float(
                    np.mean(
                        [
                            np.nanmean(time_grids[seed]["exit"][:, decision])
                            for seed in STAGE3_SEEDS
                        ]
                    )
                ),
                **primary_coverage,
                "finite_ic_date_count": int(np.isfinite(across_daily).sum()),
                "across_seed_minimum_ic": float(np.nanmin(ic_by_seed)),
                "across_seed_maximum_ic": float(np.nanmax(ic_by_seed)),
                "positive_seed_count": positive,
                "zero_seed_count": zero,
                "negative_seed_count": negative,
            }
        )
    for scope_index, (scope_name, decisions) in enumerate(scopes.items()):
        for horizon_position in range(len(HORIZONS) + 1):
            horizon = (
                HORIZONS[horizon_position] if horizon_position < len(HORIZONS) else 0
            )
            coverage = _coverage_summary(
                daily_valid_count,
                daily_coverage,
                decisions,
                horizon_position if horizon else None,
            )
            if horizon:
                ic_daily_by_seed = np.stack(
                    [
                        core_by_seed[seed]["scope_daily"][scope_name]["ic"][
                            :, horizon_position
                        ]
                        for seed in STAGE3_SEEDS
                    ]
                )
            else:
                ic_daily_by_seed = np.stack(
                    [
                        np.nanmean(
                            core_by_seed[seed]["scope_daily"][scope_name]["ic"],
                            axis=1,
                        )
                        for seed in STAGE3_SEEDS
                    ]
                )
            ic_by_seed = np.nanmean(ic_daily_by_seed, axis=1)
            positive, zero, negative = _sign_counts(ic_by_seed)
            across_daily = np.nanmean(ic_daily_by_seed, axis=0)
            bootstrap = moving_block_bootstrap(
                across_daily,
                replications=bootstrap_replications,
                seed=BOOTSTRAP_SEED + 10_000 + scope_index * 10 + horizon_position,
            )

            def scope_metric(name: str) -> float:
                values = []
                for seed in STAGE3_SEEDS:
                    metric = core_by_seed[seed]["scope_daily"][scope_name][name]
                    values.append(
                        np.nanmean(metric[:, horizon_position])
                        if horizon
                        else np.nanmean(metric)
                    )
                return float(np.nanmean(values))

            time_bin_rows.append(
                {
                    "aggregation": "across_seed",
                    "seed": None,
                    "scope": scope_name,
                    "decision_indices": json.dumps(list(decisions)),
                    "start_time_brt": DECISION_TIMES[decisions[0]].isoformat(),
                    "end_time_brt": DECISION_TIMES[decisions[-1]].isoformat(),
                    "decision_count": len(decisions),
                    "horizon_minutes": horizon,
                    "mean_spearman_ic": float(np.nanmean(ic_by_seed)),
                    "ic_interval_lower_95": bootstrap["interval_lower_95"],
                    "ic_interval_upper_95": bootstrap["interval_upper_95"],
                    "ic_bootstrap_finite_replication_count": int(
                        bootstrap["finite_replication_count"]
                    ),
                    "mean_gross_top_return": scope_metric("top_return"),
                    "mean_gross_bottom_return": scope_metric("bottom_return"),
                    "mean_gross_top_minus_bottom": scope_metric("spread"),
                    "mean_intraday_one_way_turnover": scope_metric("turnover"),
                    "mean_flat_entry_turnover": scope_metric("entry"),
                    "mean_flat_exit_turnover": scope_metric("exit"),
                    **coverage,
                    "finite_ic_date_count": int(np.isfinite(across_daily).sum()),
                    "across_seed_minimum_ic": float(np.nanmin(ic_by_seed)),
                    "across_seed_maximum_ic": float(np.nanmax(ic_by_seed)),
                    "positive_seed_count": positive,
                    "zero_seed_count": zero,
                    "negative_seed_count": negative,
                }
            )
    stock_time_frame = pl.DataFrame(
        stock_time_rows, infer_schema_length=None
    ).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("across_seed_minimum_contribution"),
        pl.lit(None, dtype=pl.Float64).alias("across_seed_maximum_contribution"),
        pl.lit(None, dtype=pl.Int64).alias("positive_seed_count"),
        pl.lit(None, dtype=pl.Int64).alias("zero_seed_count"),
        pl.lit(None, dtype=pl.Int64).alias("negative_seed_count"),
    )
    across_stock_time = (
        stock_time_frame.group_by(
            "equity_slot",
            "security_id",
            "isin",
            "display_ticker",
            "display_name",
            "scope_type",
            "scope",
            "decision_idx",
            "decision_time_brt",
            "horizon_minutes",
            "exploratory",
        )
        .agg(
            pl.col("additive_ic_contribution").mean(),
            pl.col("additive_ic_contribution")
            .min()
            .alias("across_seed_minimum_contribution"),
            pl.col("additive_ic_contribution")
            .max()
            .alias("across_seed_maximum_contribution"),
            (pl.col("additive_ic_contribution") > 1e-15)
            .sum()
            .alias("positive_seed_count"),
            (pl.col("additive_ic_contribution").abs() <= 1e-15)
            .sum()
            .alias("zero_seed_count"),
            (pl.col("additive_ic_contribution") < -1e-15)
            .sum()
            .alias("negative_seed_count"),
        )
        .with_columns(
            pl.lit("across_seed").alias("aggregation"),
            pl.lit(None, dtype=pl.Int64).alias("seed"),
        )
    )
    stock_time_frame = pl.concat(
        [stock_time_frame, across_stock_time.select(stock_time_frame.columns)],
        how="vertical_relaxed",
    )
    time_5m_frame = pl.DataFrame(time_5m_rows, infer_schema_length=None)
    time_bin_frame = pl.DataFrame(time_bin_rows, infer_schema_length=None)
    for name, frame in (
        ("time_5m", time_5m_frame),
        ("time_bin", time_bin_frame),
    ):
        frame = frame.with_columns(
            pl.col("mean_gross_top_minus_bottom").alias(
                "mean_daily_gross_contribution"
            ),
            pl.col("mean_flat_entry_turnover").alias("mean_daily_flat_entry_turnover"),
            pl.col("mean_intraday_one_way_turnover").alias(
                "mean_daily_intraday_one_way_turnover"
            ),
            pl.col("mean_flat_exit_turnover").alias("mean_daily_flat_exit_turnover"),
            (
                pl.col("mean_flat_entry_turnover")
                + pl.col("mean_intraday_one_way_turnover")
                + pl.col("mean_flat_exit_turnover")
            ).alias("mean_daily_total_one_way_turnover"),
            pl.lit(
                "decimal return summed within date then averaged across dates"
            ).alias("gross_return_unit"),
            pl.lit(
                "one-way notional fraction summed within date then averaged across dates"
            ).alias("turnover_unit"),
        )
        frame = _with_economic_ratios(frame)
        if name == "time_5m":
            time_5m_frame = frame
        else:
            time_bin_frame = frame
    return (
        {
            "stock_attribution": pl.DataFrame(stock_rows, infer_schema_length=None),
            "stock_time_attribution": stock_time_frame,
            "time_of_day_5m": time_5m_frame,
            "time_of_day_bins": time_bin_frame,
        },
        core_by_seed,
        {
            "by_seed": reconstruction,
            "portfolio_positive_contribution_mass": portfolio_positive_mass,
            "portfolio_negative_contribution_mass": portfolio_negative_mass,
            "portfolio_net_primary_ic": net_primary,
        },
    )


def _group_sample_sums(
    values: np.ndarray,
    groups_by_date: np.ndarray,
    date_idx: np.ndarray,
    group_count: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.full(
        (values.shape[0], group_count, values.shape[2]), np.nan, dtype=np.float64
    )
    valid_cell = np.isfinite(values).all(axis=1)
    for sample in range(values.shape[0]):
        groups = groups_by_date[date_idx[sample]]
        for group in range(group_count):
            members = groups == group
            valid_horizons = valid_cell[sample]
            result[sample, group, valid_horizons] = values[sample, members][
                :, valid_horizons
            ].sum(axis=0)
    return result


def _group_sample_counts(
    mask: np.ndarray,
    groups_by_date: np.ndarray,
    date_idx: np.ndarray,
    group_count: int,
) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    result = np.zeros((mask.shape[0], group_count, mask.shape[2]), dtype=np.float64)
    for sample in range(mask.shape[0]):
        groups = groups_by_date[date_idx[sample]]
        for group in range(group_count):
            result[sample, group] = mask[sample, groups == group].sum(axis=0)
    return result


def _independent_bucket_ic(
    predictions: np.ndarray,
    targets: np.ndarray,
    label_mask: np.ndarray,
    groups_by_date: np.ndarray,
    date_idx: np.ndarray,
    maximum_groups: int,
) -> np.ndarray:
    result = np.full(
        (predictions.shape[0], maximum_groups, predictions.shape[2]),
        np.nan,
        dtype=np.float64,
    )
    for sample in range(predictions.shape[0]):
        groups = groups_by_date[date_idx[sample]]
        for horizon in range(predictions.shape[2]):
            for group in range(maximum_groups):
                valid = label_mask[sample, :, horizon] & (groups == group)
                if int(valid.sum()) < MIN_IC_EQUITIES:
                    continue
                result[sample, group, horizon] = _correlation(
                    average_ranks(predictions[sample, valid, horizon]),
                    average_ranks(targets[sample, valid, horizon]),
                )
    return result


def _contribution_concentration(
    values: np.ndarray,
) -> tuple[float | None, float | None]:
    absolute = np.abs(np.asarray(values, dtype=np.float64))
    total = float(absolute.sum())
    if total == 0.0:
        return None, None
    shares = absolute / total
    top_count = max(1, math.ceil(shares.size * 0.10))
    return float(np.sum(shares**2)), float(np.sort(shares)[-top_count:].sum())


def _point_in_time_bucket_contribution_vector(
    contribution_grid: np.ndarray,
    buckets: np.ndarray,
    bucket: int,
    decisions: tuple[int, ...],
    horizon: int,
) -> np.ndarray:
    values = np.asarray(contribution_grid[:, :, :, horizon], dtype=np.float64)
    valid_cell = np.isfinite(values).all(axis=2)
    membership = np.asarray(buckets, dtype=np.int8)[:, None, :] == bucket
    masked = np.where(
        valid_cell[:, :, None],
        np.where(membership, values, 0.0),
        np.nan,
    )
    daily = _finite_axis_mean(masked[:, decisions], axis=1)
    return _finite_axis_mean(daily, axis=0)


def _build_liquidity_outputs(
    cache_paths: dict[tuple[str, int], Path],
    core_by_seed: dict[int, dict[str, object]],
    shared: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, object]]:
    date_idx = np.asarray(shared["date_idx"], dtype=np.int64)
    date_positions = _metadata_date_positions(metadata, date_idx)
    decision_idx = np.asarray(shared["decision_idx"], dtype=np.int64)
    targets = np.asarray(shared["targets"])
    label_mask = np.asarray(shared["label_mask"], dtype=bool)
    quintiles = np.asarray(metadata["liquidity_quintile"], dtype=np.int8)
    adaptive = np.asarray(metadata["adaptive_liquidity"], dtype=np.int8)
    adaptive_count_by_date = np.asarray(
        metadata["adaptive_liquidity_bucket_count"], dtype=np.int8
    )
    dollar_liquidity = np.asarray(metadata["dollar_liquidity"], dtype=np.float64)
    threshold_metadata = metadata["eligibility_liquidity_threshold"]
    threshold = (
        float(threshold_metadata["value_brl"])
        if isinstance(threshold_metadata, dict)
        else None
    )
    rows: list[dict[str, object]] = []
    time_rows: list[dict[str, object]] = []
    checks: dict[str, object] = {}
    all_decisions = tuple(range(EXPECTED_DECISIONS_PER_DATE))
    time_bins = {
        row["name"]: tuple(row["decision_indices"]) for row in _time_bin_metadata()
    }
    for seed in STAGE3_SEEDS:
        core = core_by_seed[seed]
        additive_values = np.asarray(core["contributions"])
        sample_ic = np.asarray(core["sample_ic"])
        _, sample_ic_grid = _daily_grid(sample_ic, date_idx, decision_idx)
        economic = core["economic"]
        if not isinstance(economic, EconomicAttributionResult):
            raise TypeError("Core economic attribution is malformed")
        predictions = np.load(
            cache_paths[("core", seed)], mmap_mode="r", allow_pickle=False
        )
        group_values = {
            "contribution": _group_sample_sums(
                additive_values, quintiles, date_positions, 5
            ),
            "positive": _group_sample_sums(
                np.maximum(additive_values, 0.0), quintiles, date_positions, 5
            ),
            "negative": _group_sample_sums(
                np.minimum(additive_values, 0.0), quintiles, date_positions, 5
            ),
        }
        group_coverage = _group_sample_counts(label_mask, quintiles, date_positions, 5)
        group_active = _group_sample_counts(
            np.broadcast_to(
                (quintiles[date_positions] >= 0)[..., None], label_mask.shape
            ),
            quintiles,
            date_positions,
            5,
        )
        group_tail = _group_sample_counts(
            economic.top_selected | economic.bottom_selected,
            quintiles,
            date_positions,
            5,
        )
        daily_metrics: dict[str, np.ndarray] = {}
        for name, values in group_values.items():
            _, grid = _daily_grid(
                values.reshape(values.shape[0], -1), date_idx, decision_idx
            )
            daily_metrics[name] = grid.reshape(
                grid.shape[0], grid.shape[1], 5, len(HORIZONS)
            )
        _, contribution_grid = _grid_from_stock_values(
            additive_values, date_idx, decision_idx
        )

        def independent_grid(groups: np.ndarray) -> np.ndarray:
            values = _independent_bucket_ic(
                predictions,
                targets,
                label_mask,
                groups,
                date_positions,
                5,
            )
            _, grid = _daily_grid(
                values.reshape(values.shape[0], -1), date_idx, decision_idx
            )
            return grid.reshape(grid.shape[0], grid.shape[1], 5, len(HORIZONS))

        fixed_grid = independent_grid(quintiles)
        adaptive_grid = independent_grid(adaptive)
        all_day_fixed_ic = _scope_daily_mean(fixed_grid, all_decisions)
        all_day_adaptive_ic = _scope_daily_mean(adaptive_grid, all_decisions)
        all_day_metrics = {
            name: _scope_daily_mean(values, all_decisions)
            for name, values in daily_metrics.items()
        }
        all_day_accounting = economic_window_accounting(
            economic, date_idx, decision_idx, decisions=all_decisions
        )
        maximum_difference = 0.0
        for group in range(5):
            economic_daily = _window_bucket_daily(
                all_day_accounting, quintiles, metadata, group
            )
            group_members = quintiles == group
            group_liquidity = dollar_liquidity[group_members]
            for horizon_index, horizon in enumerate(HORIZONS):
                contribution = float(
                    _finite_axis_mean(
                        all_day_metrics["contribution"][:, group, horizon_index],
                        axis=0,
                    )
                )
                gross = float(
                    _finite_axis_mean(economic_daily["gross"][:, horizon_index], axis=0)
                )
                entry = float(
                    _finite_axis_mean(economic_daily["entry"][:, horizon_index], axis=0)
                )
                intraday = float(
                    _finite_axis_mean(
                        economic_daily["intraday"][:, horizon_index], axis=0
                    )
                )
                exit_value = float(
                    _finite_axis_mean(economic_daily["exit"][:, horizon_index], axis=0)
                )
                total = float(
                    _finite_axis_mean(economic_daily["total"][:, horizon_index], axis=0)
                )
                stock_vector = _point_in_time_bucket_contribution_vector(
                    contribution_grid,
                    quintiles,
                    group,
                    all_decisions,
                    horizon_index,
                )
                herfindahl, top_decile_share = _contribution_concentration(
                    stock_vector[np.any(group_members, axis=0)]
                )
                rows.append(
                    {
                        "aggregation": "seed",
                        "seed": seed,
                        "bucket_kind": "daily_liquidity_quintile",
                        "bucket": group + 1,
                        "horizon_minutes": horizon,
                        "additive_ic_contribution": contribution,
                        "positive_contribution_mass": _finite_or_none(
                            float(
                                _finite_axis_mean(
                                    all_day_metrics["positive"][
                                        :, group, horizon_index
                                    ],
                                    axis=0,
                                )
                            )
                        ),
                        "negative_contribution_mass": _finite_or_none(
                            float(
                                _finite_axis_mean(
                                    all_day_metrics["negative"][
                                        :, group, horizon_index
                                    ],
                                    axis=0,
                                )
                            )
                        ),
                        "mean_daily_gross_contribution": gross,
                        "gross_spread_contribution": gross,
                        "mean_daily_intraday_one_way_turnover": intraday,
                        "intraday_one_way_turnover": intraday,
                        "mean_daily_flat_entry_turnover": entry,
                        "flat_entry_turnover": entry,
                        "mean_daily_flat_exit_turnover": exit_value,
                        "flat_exit_turnover": exit_value,
                        "mean_daily_total_one_way_turnover": total,
                        "gross_contribution_per_unit_turnover": (
                            gross / total if total > 0 else None
                        ),
                        "break_even_one_way_cost_bps": (
                            10_000.0 * gross / total if total > 0 else None
                        ),
                        "tail_selection_frequency": float(
                            group_tail[:, group, horizon_index].sum()
                            / max(1.0, group_coverage[:, group, horizon_index].sum())
                        ),
                        "label_coverage": float(
                            group_coverage[:, group, horizon_index].sum()
                            / max(1.0, group_active[:, group, horizon_index].sum())
                        ),
                        "stock_contribution_herfindahl": herfindahl,
                        "top_decile_absolute_contribution_share": top_decile_share,
                        "mean_point_in_time_dollar_liquidity_brl": _finite_or_none(
                            float(_finite_axis_mean(group_liquidity, axis=0))
                        ),
                        "mean_distance_from_eligibility_threshold_brl": (
                            float(
                                _finite_axis_mean(group_liquidity, axis=0) - threshold
                            )
                            if threshold is not None
                            else None
                        ),
                        "independently_reranked_within_bucket_ic": _finite_mean_or_none(
                            all_day_fixed_ic[:, group, horizon_index]
                        ),
                        "adaptive_bucket_count_minimum": int(
                            adaptive_count_by_date.min()
                        ),
                    }
                )
            primary_contribution = float(
                _finite_axis_mean(
                    _finite_axis_mean(
                        all_day_metrics["contribution"][:, group], axis=1
                    ),
                    axis=0,
                )
            )
            primary_positive = float(
                _finite_axis_mean(
                    _finite_axis_mean(all_day_metrics["positive"][:, group], axis=1),
                    axis=0,
                )
            )
            primary_negative = float(
                _finite_axis_mean(
                    _finite_axis_mean(all_day_metrics["negative"][:, group], axis=1),
                    axis=0,
                )
            )
            primary_economic = {
                name: float(
                    _finite_axis_mean(
                        _finite_axis_mean(economic_daily[name], axis=1), axis=0
                    )
                )
                for name in ("gross", "entry", "intraday", "exit", "total")
            }
            primary_total = primary_economic["total"]
            rows.append(
                {
                    "aggregation": "seed",
                    "seed": seed,
                    "bucket_kind": "daily_liquidity_quintile",
                    "bucket": group + 1,
                    "horizon_minutes": 0,
                    "additive_ic_contribution": primary_contribution,
                    "positive_contribution_mass": primary_positive,
                    "negative_contribution_mass": primary_negative,
                    "mean_daily_gross_contribution": primary_economic["gross"],
                    "gross_spread_contribution": primary_economic["gross"],
                    "mean_daily_intraday_one_way_turnover": primary_economic[
                        "intraday"
                    ],
                    "intraday_one_way_turnover": primary_economic["intraday"],
                    "mean_daily_flat_entry_turnover": primary_economic["entry"],
                    "flat_entry_turnover": primary_economic["entry"],
                    "mean_daily_flat_exit_turnover": primary_economic["exit"],
                    "flat_exit_turnover": primary_economic["exit"],
                    "mean_daily_total_one_way_turnover": primary_total,
                    "gross_contribution_per_unit_turnover": (
                        primary_economic["gross"] / primary_total
                        if primary_total > 0
                        else None
                    ),
                    "break_even_one_way_cost_bps": (
                        10_000.0 * primary_economic["gross"] / primary_total
                        if primary_total > 0
                        else None
                    ),
                    "tail_selection_frequency": float(
                        group_tail[:, group].sum()
                        / max(1.0, group_coverage[:, group].sum())
                    ),
                    "label_coverage": float(
                        group_coverage[:, group].sum()
                        / max(1.0, group_active[:, group].sum())
                    ),
                    "stock_contribution_herfindahl": None,
                    "top_decile_absolute_contribution_share": None,
                    "mean_point_in_time_dollar_liquidity_brl": _finite_or_none(
                        float(_finite_axis_mean(group_liquidity, axis=0))
                    ),
                    "mean_distance_from_eligibility_threshold_brl": (
                        float(_finite_axis_mean(group_liquidity, axis=0) - threshold)
                        if threshold is not None
                        else None
                    ),
                    "independently_reranked_within_bucket_ic": _finite_mean_or_none(
                        _finite_axis_mean(all_day_fixed_ic[:, group], axis=1)
                    ),
                    "adaptive_bucket_count_minimum": int(adaptive_count_by_date.min()),
                }
            )
        for group in range(5):
            for horizon_index, horizon in enumerate(HORIZONS):
                values = all_day_adaptive_ic[:, group, horizon_index]
                if not np.isfinite(values).any():
                    continue
                rows.append(
                    {
                        "aggregation": "seed",
                        "seed": seed,
                        "bucket_kind": "adaptive_independent_ic_bucket",
                        "bucket": group + 1,
                        "horizon_minutes": horizon,
                        "additive_ic_contribution": None,
                        "positive_contribution_mass": None,
                        "negative_contribution_mass": None,
                        "mean_daily_gross_contribution": None,
                        "gross_spread_contribution": None,
                        "mean_daily_intraday_one_way_turnover": None,
                        "intraday_one_way_turnover": None,
                        "mean_daily_flat_entry_turnover": None,
                        "flat_entry_turnover": None,
                        "mean_daily_flat_exit_turnover": None,
                        "flat_exit_turnover": None,
                        "mean_daily_total_one_way_turnover": None,
                        "gross_contribution_per_unit_turnover": None,
                        "break_even_one_way_cost_bps": None,
                        "tail_selection_frequency": None,
                        "label_coverage": None,
                        "stock_contribution_herfindahl": None,
                        "top_decile_absolute_contribution_share": None,
                        "mean_point_in_time_dollar_liquidity_brl": None,
                        "mean_distance_from_eligibility_threshold_brl": None,
                        "independently_reranked_within_bucket_ic": _finite_mean_or_none(
                            values
                        ),
                        "adaptive_bucket_count_minimum": int(
                            adaptive_count_by_date.min()
                        ),
                    }
                )
            primary_values = _finite_axis_mean(all_day_adaptive_ic[:, group], axis=1)
            if np.isfinite(primary_values).any():
                rows.append(
                    {
                        "aggregation": "seed",
                        "seed": seed,
                        "bucket_kind": "adaptive_independent_ic_bucket",
                        "bucket": group + 1,
                        "horizon_minutes": 0,
                        "additive_ic_contribution": None,
                        "positive_contribution_mass": None,
                        "negative_contribution_mass": None,
                        "mean_daily_gross_contribution": None,
                        "gross_spread_contribution": None,
                        "mean_daily_intraday_one_way_turnover": None,
                        "intraday_one_way_turnover": None,
                        "mean_daily_flat_entry_turnover": None,
                        "flat_entry_turnover": None,
                        "mean_daily_flat_exit_turnover": None,
                        "flat_exit_turnover": None,
                        "mean_daily_total_one_way_turnover": None,
                        "gross_contribution_per_unit_turnover": None,
                        "break_even_one_way_cost_bps": None,
                        "tail_selection_frequency": None,
                        "label_coverage": None,
                        "stock_contribution_herfindahl": None,
                        "top_decile_absolute_contribution_share": None,
                        "mean_point_in_time_dollar_liquidity_brl": None,
                        "mean_distance_from_eligibility_threshold_brl": None,
                        "independently_reranked_within_bucket_ic": _finite_mean_or_none(
                            primary_values
                        ),
                        "adaptive_bucket_count_minimum": int(
                            adaptive_count_by_date.min()
                        ),
                    }
                )
        for time_name, decisions in time_bins.items():
            scope_contribution = _scope_daily_mean(
                daily_metrics["contribution"], decisions
            )
            scope_fixed_ic = _scope_daily_mean(fixed_grid, decisions)
            scope_accounting = economic_window_accounting(
                economic, date_idx, decision_idx, decisions=decisions
            )
            for group in range(5):
                economic_daily = _window_bucket_daily(
                    scope_accounting, quintiles, metadata, group
                )
                for horizon_index, horizon in enumerate(HORIZONS):
                    contribution = float(
                        _finite_axis_mean(
                            scope_contribution[:, group, horizon_index], axis=0
                        )
                    )
                    gross = float(
                        _finite_axis_mean(
                            economic_daily["gross"][:, horizon_index], axis=0
                        )
                    )
                    entry = float(
                        _finite_axis_mean(
                            economic_daily["entry"][:, horizon_index], axis=0
                        )
                    )
                    intraday = float(
                        _finite_axis_mean(
                            economic_daily["intraday"][:, horizon_index], axis=0
                        )
                    )
                    exit_value = float(
                        _finite_axis_mean(
                            economic_daily["exit"][:, horizon_index], axis=0
                        )
                    )
                    total = float(
                        _finite_axis_mean(
                            economic_daily["total"][:, horizon_index], axis=0
                        )
                    )
                    stock_vector = _point_in_time_bucket_contribution_vector(
                        contribution_grid,
                        quintiles,
                        group,
                        decisions,
                        horizon_index,
                    )
                    herfindahl, top_share = _contribution_concentration(
                        stock_vector[np.any(quintiles == group, axis=0)]
                    )
                    time_rows.append(
                        {
                            "aggregation": "seed",
                            "seed": seed,
                            "liquidity_quintile": group + 1,
                            "time_bin": time_name,
                            "decision_indices": json.dumps(list(decisions)),
                            "horizon_minutes": horizon,
                            "additive_ic_contribution": contribution,
                            "mean_daily_gross_contribution": gross,
                            "gross_spread_contribution": gross,
                            "mean_daily_intraday_one_way_turnover": intraday,
                            "intraday_one_way_turnover": intraday,
                            "mean_daily_flat_entry_turnover": entry,
                            "flat_entry_turnover": entry,
                            "mean_daily_flat_exit_turnover": exit_value,
                            "flat_exit_turnover": exit_value,
                            "mean_daily_total_one_way_turnover": total,
                            "gross_contribution_per_unit_turnover": (
                                gross / total if total > 0 else None
                            ),
                            "break_even_one_way_cost_bps": (
                                10_000.0 * gross / total if total > 0 else None
                            ),
                            "stock_contribution_herfindahl": herfindahl,
                            "independently_reranked_within_bucket_ic": _finite_mean_or_none(
                                scope_fixed_ic[:, group, horizon_index]
                            ),
                            "top_decile_absolute_contribution_share": top_share,
                        }
                    )
            for horizon_index, horizon in enumerate(HORIZONS):
                reconstructed = sum(
                    row["additive_ic_contribution"]
                    for row in time_rows
                    if row["seed"] == seed
                    and row["time_bin"] == time_name
                    and row["horizon_minutes"] == horizon
                )
                expected = float(
                    _finite_axis_mean(
                        _scope_daily_mean(
                            sample_ic_grid,
                            decisions,
                        )[:, horizon_index],
                        axis=0,
                    )
                )
                difference = abs(reconstructed - expected)
                maximum_difference = max(maximum_difference, difference)
                if difference > RECONSTRUCTION_ABSOLUTE_TOLERANCE:
                    raise RuntimeError(
                        "Liquidity/time cells failed additive reconstruction"
                    )
        for horizon_index, horizon in enumerate(HORIZONS):
            reconstructed = sum(
                row["additive_ic_contribution"]
                for row in rows
                if row["seed"] == seed
                and row["bucket_kind"] == "daily_liquidity_quintile"
                and row["horizon_minutes"] == horizon
            )
            expected = float(
                _finite_axis_mean(
                    _scope_daily_mean(
                        sample_ic_grid,
                        all_decisions,
                    )[:, horizon_index],
                    axis=0,
                )
            )
            difference = abs(reconstructed - expected)
            maximum_difference = max(maximum_difference, difference)
            if difference > RECONSTRUCTION_ABSOLUTE_TOLERANCE:
                raise RuntimeError("Liquidity quintiles failed additive reconstruction")
        reconstructed_primary = sum(
            row["additive_ic_contribution"]
            for row in rows
            if row["seed"] == seed
            and row["bucket_kind"] == "daily_liquidity_quintile"
            and row["horizon_minutes"] == 0
        )
        expected_primary = float(
            _finite_axis_mean(
                _finite_axis_mean(
                    _scope_daily_mean(sample_ic_grid, all_decisions), axis=1
                ),
                axis=0,
            )
        )
        primary_difference = abs(reconstructed_primary - expected_primary)
        maximum_difference = max(maximum_difference, primary_difference)
        if primary_difference > RECONSTRUCTION_ABSOLUTE_TOLERANCE:
            raise RuntimeError(
                "Liquidity quintile h0 rows failed additive reconstruction"
            )
        checks[str(seed)] = {
            "maximum_cell_absolute_difference": maximum_difference,
            "passed": True,
        }
    liquidity = pl.DataFrame(rows, infer_schema_length=None)
    liquidity_time = pl.DataFrame(time_rows, infer_schema_length=None)
    group_columns = ["bucket_kind", "bucket", "horizon_minutes"]
    derived_economic_columns = {
        "gross_contribution_per_unit_turnover",
        "break_even_one_way_cost_bps",
    }
    numeric_columns = [
        name
        for name, dtype in liquidity.schema.items()
        if name
        not in {"seed", "aggregation", *group_columns, *derived_economic_columns}
        and dtype.is_numeric()
    ]
    across = (
        liquidity.group_by(group_columns)
        .agg([pl.col(column).mean().alias(column) for column in numeric_columns])
        .with_columns(
            pl.lit("across_seed").alias("aggregation"),
            pl.lit(None, dtype=pl.Int64).alias("seed"),
        )
    )
    across = _with_economic_ratios(across)
    for column in liquidity.columns:
        if column not in across.columns:
            across = across.with_columns(pl.lit(None).alias(column))
    liquidity = pl.concat(
        [liquidity, across.select(liquidity.columns)], how="vertical_relaxed"
    )
    time_group_columns = [
        "liquidity_quintile",
        "time_bin",
        "decision_indices",
        "horizon_minutes",
    ]
    time_numeric = [
        name
        for name, dtype in liquidity_time.schema.items()
        if name
        not in {"seed", "aggregation", *time_group_columns, *derived_economic_columns}
        and dtype.is_numeric()
    ]
    time_across = (
        liquidity_time.group_by(time_group_columns)
        .agg([pl.col(column).mean().alias(column) for column in time_numeric])
        .with_columns(
            pl.lit("across_seed").alias("aggregation"),
            pl.lit(None, dtype=pl.Int64).alias("seed"),
        )
    )
    time_across = _with_economic_ratios(time_across)
    for column in liquidity_time.columns:
        if column not in time_across.columns:
            time_across = time_across.with_columns(pl.lit(None).alias(column))
    liquidity_time = pl.concat(
        [liquidity_time, time_across.select(liquidity_time.columns)],
        how="vertical_relaxed",
    )
    return liquidity, liquidity_time, checks


def _daily_subset_mean(
    values: np.ndarray,
    selected_samples: np.ndarray,
    date_idx: np.ndarray,
    decision_idx: np.ndarray,
    decisions: tuple[int, ...],
) -> np.ndarray:
    selected = np.asarray(selected_samples, dtype=bool) & np.isin(
        decision_idx, decisions
    )
    masked = np.asarray(values, dtype=np.float64).copy()
    if masked.ndim == 1:
        masked = masked[:, None]
    masked[~selected] = np.nan
    _, grid = _daily_grid(masked, date_idx, decision_idx)
    return _scope_daily_mean(grid, decisions)


def _completeness_category(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.full(values.shape, -1, dtype=np.int8)
    result[(values >= 0.0) & (values < 0.80)] = 0
    result[(values >= 0.80) & (values < 0.95)] = 1
    result[values >= 0.95] = 2
    return result


def _freshness_categories(
    ready: np.ndarray, staleness: np.ndarray
) -> dict[str, np.ndarray]:
    ready = np.asarray(ready, dtype=bool)
    staleness = np.asarray(staleness, dtype=np.float64)
    return {
        "all_dates": np.ones(ready.shape, dtype=bool),
        "ready": ready,
        "fresh_0_5m": ready & np.isfinite(staleness) & (staleness <= 5.0),
        "moderate_6_30m": ready
        & np.isfinite(staleness)
        & (staleness > 5.0)
        & (staleness <= 30.0),
        "stale_over_30m": ready & np.isfinite(staleness) & (staleness > 30.0),
        "unready": ~ready,
    }


def _opening_condition_masks(
    shared: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> list[OpeningCondition]:
    sample_count = np.asarray(shared["date_idx"]).size
    date_positions = _metadata_date_positions(
        metadata, np.asarray(shared["date_idx"], dtype=np.int64)
    )
    equity = metadata["equity_completeness"]
    local = metadata["local_completeness"]
    global_values = metadata["global_completeness"]
    if not all(isinstance(value, dict) for value in (equity, local, global_values)):
        raise TypeError("Completeness metadata is malformed")
    result: list[OpeningCondition] = []
    active = np.asarray(metadata["active"], dtype=bool)[date_positions]
    observed_bars = np.asarray(equity["observed_bars"], dtype=np.float64)
    observed_fraction = np.asarray(equity["observed_fraction"], dtype=np.float64)
    scheduled_minutes = np.asarray(equity["scheduled_minutes"], dtype=np.float64)
    median_bar_count = np.full(sample_count, np.nan, dtype=np.float64)
    median_fraction = np.full(sample_count, np.nan, dtype=np.float64)
    meeting_expected_fraction = np.full(sample_count, np.nan, dtype=np.float64)
    for sample in range(sample_count):
        eligible = active[sample]
        if not eligible.any() or scheduled_minutes[sample] <= 0:
            continue
        median_bar_count[sample] = float(np.median(observed_bars[sample, eligible]))
        median_fraction[sample] = float(np.median(observed_fraction[sample, eligible]))
        meeting_expected_fraction[sample] = float(
            np.mean(observed_bars[sample, eligible] >= scheduled_minutes[sample])
        )
    missing_bar_count = scheduled_minutes - median_bar_count
    diagnostic_fields = {
        "median_observed_bar_count": median_bar_count,
        "median_observed_fraction": median_fraction,
        "fraction_eligible_meeting_expected_history": meeting_expected_fraction,
    }
    for category, mask, lower, upper, lower_inclusive in (
        (
            "0_to_30_bars",
            np.isfinite(median_bar_count)
            & (median_bar_count >= 0.0)
            & (median_bar_count <= 30.0),
            0.0,
            30.0,
            True,
        ),
        (
            "31_to_60_bars",
            (median_bar_count > 30.0) & (median_bar_count <= 60.0),
            30.0,
            60.0,
            False,
        ),
        (
            "61_to_90_bars",
            (median_bar_count > 60.0) & (median_bar_count <= 90.0),
            60.0,
            90.0,
            False,
        ),
        (
            "91_to_120_bars",
            (median_bar_count > 90.0) & (median_bar_count <= 120.0),
            90.0,
            120.0,
            False,
        ),
        (
            "121_to_180_bars",
            (median_bar_count > 120.0) & (median_bar_count <= 180.0),
            120.0,
            180.0,
            False,
        ),
        ("over_180_bars", median_bar_count > 180.0, 180.0, None, False),
    ):
        result.append(
            OpeningCondition(
                "b3_observed_bar_count",
                category,
                mask,
                category_lower_bound=lower,
                category_upper_bound=upper,
                category_lower_bound_inclusive=lower_inclusive,
                category_upper_bound_inclusive=(True if upper is not None else None),
                category_unit="observed_current_session_b3_equity_bars",
                **diagnostic_fields,
            )
        )
    for category, mask, lower, upper in (
        (
            "complete",
            np.isfinite(missing_bar_count) & (missing_bar_count <= 0.5),
            0.0,
            0.5,
        ),
        (
            "missing_1_to_5_bars",
            (missing_bar_count > 0.5) & (missing_bar_count <= 5.0),
            0.5,
            5.0,
        ),
        ("missing_over_5_bars", missing_bar_count > 5.0, 5.0, None),
    ):
        result.append(
            OpeningCondition(
                "b3_history_missingness",
                category,
                mask,
                category_lower_bound=lower,
                category_upper_bound=upper,
                category_lower_bound_inclusive=(category == "complete"),
                category_upper_bound_inclusive=(upper is not None),
                category_unit="missing_bars_from_expected_current_session_history",
                **diagnostic_fields,
            )
        )
    for category, mask, lower, upper in (
        ("below_80pct", (median_fraction >= 0.0) & (median_fraction < 0.80), 0.0, 0.80),
        (
            "80_to_95pct",
            (median_fraction >= 0.80) & (median_fraction < 0.95),
            0.80,
            0.95,
        ),
        ("at_least_95pct", median_fraction >= 0.95, 0.95, 1.0),
    ):
        result.append(
            OpeningCondition(
                "b3_history_completeness",
                category,
                mask,
                category_lower_bound=lower,
                category_upper_bound=upper,
                category_lower_bound_inclusive=True,
                category_upper_bound_inclusive=(category == "at_least_95pct"),
                category_unit="observed_fraction_of_expected_current_session_history",
                **diagnostic_fields,
            )
        )

    local_positions = np.asarray(
        [
            LOCAL_CONTEXT_SYMBOLS.index(symbol)
            for symbol in EXPECTED_RETAINED_LOCAL_CONTEXTS
        ]
    )
    global_positions = np.asarray(
        [
            GLOBAL_CONTEXT_SYMBOLS.index(symbol)
            for symbol in EXPECTED_RETAINED_GLOBAL_CONTEXTS
        ]
    )
    local_complete = np.all(
        np.asarray(local["ready"])[:, local_positions]
        & (
            np.asarray(local["observed_fraction"], dtype=np.float64)[:, local_positions]
            >= 0.95
        ),
        axis=1,
    )
    global_complete = np.all(
        np.asarray(global_values["ready"])[:, global_positions]
        & (
            np.asarray(global_values["observed_fraction"], dtype=np.float64)[
                :, global_positions
            ]
            >= 0.95
        ),
        axis=1,
    )
    result.extend(
        [
            OpeningCondition("retained_local_completeness", "complete", local_complete),
            OpeningCondition(
                "retained_local_completeness", "incomplete", ~local_complete
            ),
            OpeningCondition(
                "retained_global_completeness", "complete", global_complete
            ),
            OpeningCondition(
                "retained_global_completeness", "incomplete", ~global_complete
            ),
        ]
    )
    global_staleness = np.asarray(
        global_values["minutes_since_most_recent_observed_bar"], dtype=np.float64
    )[:, global_positions]
    global_fresh = (
        np.all(np.asarray(global_values["ready"])[:, global_positions], axis=1)
        & np.isfinite(global_staleness).all(axis=1)
        & (global_staleness <= RECENT_OBSERVED_MINUTES).all(axis=1)
    )
    result.extend(
        [
            OpeningCondition(
                "retained_global_freshness",
                "fresh_within_30m",
                global_fresh,
                freshness_category="fresh_within_30m",
            ),
            OpeningCondition(
                "retained_global_freshness",
                "stale_or_unready",
                ~global_fresh,
                freshness_category="stale_or_unready",
            ),
        ]
    )
    local_preopen = np.asarray(local["preopen_observed_fraction"], dtype=np.float64)[
        :, local_positions
    ]
    global_preopen = np.asarray(
        global_values["preopen_observed_fraction"], dtype=np.float64
    )[:, global_positions]
    local_ready = np.all(np.asarray(local["ready"])[:, local_positions], axis=1)
    global_ready = np.all(
        np.asarray(global_values["ready"])[:, global_positions], axis=1
    )
    local_preopen_complete = (
        local_ready
        & np.isfinite(local_preopen).all(axis=1)
        & (local_preopen >= 0.95).all(axis=1)
    )
    global_preopen_complete = (
        global_ready
        & np.isfinite(global_preopen).all(axis=1)
        & (global_preopen >= 0.95).all(axis=1)
    )
    for condition_type, complete in (
        ("retained_local_preopen", local_preopen_complete),
        ("retained_global_preopen", global_preopen_complete),
        (
            "retained_context_preopen",
            local_preopen_complete & global_preopen_complete,
        ),
    ):
        result.extend(
            [
                OpeningCondition(condition_type, "complete", complete),
                OpeningCondition(condition_type, "incomplete", ~complete),
            ]
        )
    overnight = metadata["overnight_regimes"]
    if not isinstance(overnight, dict):
        raise TypeError("Overnight regimes are malformed")
    for regime, day_mask in overnight.items():
        sample_mask = np.asarray(day_mask, dtype=bool)[date_positions]
        if sample_mask.shape != (sample_count,):
            raise ValueError("Overnight regime mask is misaligned")
        result.append(
            OpeningCondition(
                "overnight_regime",
                regime,
                sample_mask,
                overnight_regime=regime,
            )
        )
    return result


def _opening_condition_metadata(
    condition: OpeningCondition,
    date_idx: np.ndarray,
    decision_idx: np.ndarray,
    decisions: tuple[int, ...],
) -> dict[str, object]:
    mask = np.asarray(condition.sample_mask, dtype=bool) & np.isin(
        decision_idx, decisions
    )

    def selected_mean(values: np.ndarray | None) -> float | None:
        if values is None:
            return None
        selected = np.asarray(values, dtype=np.float64)[mask]
        return _finite_mean_or_none(selected)

    def selected_median(values: np.ndarray | None) -> float | None:
        if values is None:
            return None
        selected = np.asarray(values, dtype=np.float64)[mask]
        finite = selected[np.isfinite(selected)]
        return float(np.median(finite)) if finite.size else None

    return {
        "category_lower_bound": condition.category_lower_bound,
        "category_upper_bound": condition.category_upper_bound,
        "category_lower_bound_inclusive": condition.category_lower_bound_inclusive,
        "category_upper_bound_inclusive": condition.category_upper_bound_inclusive,
        "category_unit": condition.category_unit,
        "condition_decision_cell_count": int(mask.sum()),
        "condition_date_count": int(np.unique(date_idx[mask]).size),
        "mean_median_observed_bar_count": selected_mean(
            condition.median_observed_bar_count
        ),
        "median_observed_fraction": selected_median(condition.median_observed_fraction),
        "mean_fraction_eligible_meeting_expected_history": selected_mean(
            condition.fraction_eligible_meeting_expected_history
        ),
    }


def _build_opening_regimes(
    core_by_seed: dict[int, dict[str, object]],
    shared: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> pl.DataFrame:
    date_idx = np.asarray(shared["date_idx"], dtype=np.int64)
    decision_idx = np.asarray(shared["decision_idx"], dtype=np.int64)
    date_positions = _metadata_date_positions(metadata, date_idx)
    label_mask = np.asarray(shared["label_mask"], dtype=bool)
    liquidity = np.asarray(metadata["liquidity_quintile"], dtype=np.int8)
    equity = metadata["equity_completeness"]
    local = metadata["local_completeness"]
    global_values = metadata["global_completeness"]
    if not all(isinstance(value, dict) for value in (equity, local, global_values)):
        raise TypeError("Completeness metadata is malformed")
    equity_category = _completeness_category(equity["observed_fraction"])
    rows: list[dict[str, object]] = []
    bins = {row["name"]: tuple(row["decision_indices"]) for row in _time_bin_metadata()}
    for seed in STAGE3_SEEDS:
        core = core_by_seed[seed]
        predictions = np.load(
            core["predictions_path"], mmap_mode="r", allow_pickle=False
        )
        targets = np.asarray(shared["targets"])
        contributions = np.asarray(core["contributions"])
        sample_ic = np.asarray(core["sample_ic"])
        economic = core["economic"]
        if not isinstance(economic, EconomicAttributionResult):
            raise TypeError("Core economic attribution is malformed")
        for bin_name, decisions in bins.items():
            sample_selector = np.isin(decision_idx, decisions)
            for quintile in range(5):
                sample_groups = liquidity[date_positions] == quintile
                for horizon_index, horizon in enumerate(HORIZONS):
                    for category, category_name in enumerate(
                        ("below_80pct", "80_to_95pct", "at_least_95pct")
                    ):
                        sample_contribution = np.full(
                            contributions.shape[0], np.nan, dtype=np.float64
                        )
                        conditional_ic = np.full_like(sample_contribution, np.nan)
                        observed_values: list[float] = []
                        recent_values: list[float] = []
                        for sample in np.flatnonzero(sample_selector):
                            members = (
                                label_mask[sample, :, horizon_index]
                                & sample_groups[sample]
                                & (equity_category[sample] == category)
                            )
                            if members.any():
                                sample_contribution[sample] = float(
                                    contributions[sample, members, horizon_index].sum()
                                )
                                observed_values.extend(
                                    np.asarray(equity["observed_fraction"])[
                                        sample, members
                                    ].tolist()
                                )
                                recent_values.extend(
                                    np.asarray(equity["recent_observed_fraction"])[
                                        sample, members
                                    ].tolist()
                                )
                            if int(members.sum()) >= MIN_IC_EQUITIES:
                                conditional_ic[sample] = _correlation(
                                    average_ranks(
                                        predictions[sample, members, horizon_index]
                                    ),
                                    average_ranks(
                                        targets[sample, members, horizon_index]
                                    ),
                                )
                        daily_contribution = _daily_subset_mean(
                            sample_contribution,
                            np.isfinite(sample_contribution),
                            date_idx,
                            decision_idx,
                            decisions,
                        )[:, 0]
                        daily_conditional_ic = _daily_subset_mean(
                            conditional_ic,
                            np.isfinite(conditional_ic),
                            date_idx,
                            decision_idx,
                            decisions,
                        )[:, 0]
                        rows.append(
                            {
                                "diagnostic_type": "equity_history_stratification",
                                "seed": seed,
                                "instrument": None,
                                "time_scope": bin_name,
                                "decision_indices": json.dumps(list(decisions)),
                                "horizon_minutes": horizon,
                                "liquidity_quintile": quintile + 1,
                                "history_category": category_name,
                                "overnight_regime": None,
                                "additive_ic_contribution": _finite_mean_or_none(
                                    daily_contribution
                                ),
                                "independently_reranked_conditional_ic": _finite_mean_or_none(
                                    daily_conditional_ic
                                ),
                                "mean_observed_fraction": (
                                    float(np.mean(observed_values))
                                    if observed_values
                                    else None
                                ),
                                "mean_recent_observed_fraction": (
                                    float(np.mean(recent_values))
                                    if recent_values
                                    else None
                                ),
                                "readiness_fraction": None,
                                "preopen_observed_fraction": None,
                                "mean_staleness_minutes": None,
                                "gross_spread": None,
                                "intraday_turnover": None,
                                "valid_date_count": int(
                                    np.isfinite(daily_contribution).sum()
                                ),
                            }
                        )
        overnight = metadata["overnight_regimes"]
        if not isinstance(overnight, dict):
            raise TypeError("Overnight regimes are malformed")
        sample_date_position = date_positions
        for regime_name, date_mask in overnight.items():
            for scope_name in ("opening_30", "opening_60"):
                decisions = named_time_scopes()[scope_name]
                selected = np.asarray(date_mask, dtype=bool)[sample_date_position]
                daily_ic = _daily_subset_mean(
                    sample_ic,
                    selected,
                    date_idx,
                    decision_idx,
                    decisions,
                )
                daily_economic = _window_portfolio_daily(
                    economic_window_accounting(
                        economic,
                        date_idx,
                        decision_idx,
                        decisions=decisions,
                        selected_samples=selected,
                    )
                )
                daily_spread = daily_economic["gross"]
                daily_turnover = daily_economic["intraday"]
                for horizon_index, horizon in enumerate(HORIZONS):
                    rows.append(
                        {
                            "diagnostic_type": "overnight_regime",
                            "seed": seed,
                            "instrument": None,
                            "time_scope": scope_name,
                            "decision_indices": json.dumps(list(decisions)),
                            "horizon_minutes": horizon,
                            "liquidity_quintile": None,
                            "history_category": None,
                            "overnight_regime": regime_name,
                            "additive_ic_contribution": _finite_mean_or_none(
                                daily_ic[:, horizon_index]
                            ),
                            "independently_reranked_conditional_ic": None,
                            "mean_observed_fraction": None,
                            "mean_recent_observed_fraction": None,
                            "readiness_fraction": None,
                            "preopen_observed_fraction": None,
                            "mean_staleness_minutes": None,
                            "gross_spread": _finite_mean_or_none(
                                daily_spread[:, horizon_index]
                            ),
                            "intraday_turnover": _finite_mean_or_none(
                                daily_turnover[:, horizon_index]
                            ),
                            "valid_date_count": int(
                                np.isfinite(daily_ic[:, horizon_index]).sum()
                            ),
                        }
                    )
    for family, symbols, completeness, retained in (
        (
            "retained_local_context",
            LOCAL_CONTEXT_SYMBOLS,
            local,
            EXPECTED_RETAINED_LOCAL_CONTEXTS,
        ),
        (
            "retained_global_context",
            GLOBAL_CONTEXT_SYMBOLS,
            global_values,
            EXPECTED_RETAINED_GLOBAL_CONTEXTS,
        ),
    ):
        symbol_positions = {symbol: symbols.index(symbol) for symbol in retained}
        for bin_name, decisions in bins.items():
            selected_samples = np.isin(decision_idx, decisions)
            for symbol, position in symbol_positions.items():
                fractions = np.asarray(completeness["observed_fraction"])[
                    selected_samples, position
                ]
                preopen = np.asarray(completeness["preopen_observed_fraction"])[
                    selected_samples, position
                ]
                stale = np.asarray(
                    completeness["minutes_since_most_recent_observed_bar"]
                )[selected_samples, position]
                ready = np.asarray(completeness["ready"])[selected_samples, position]
                rows.append(
                    {
                        "diagnostic_type": family,
                        "seed": None,
                        "instrument": symbol,
                        "time_scope": bin_name,
                        "decision_indices": json.dumps(list(decisions)),
                        "horizon_minutes": None,
                        "liquidity_quintile": None,
                        "history_category": None,
                        "overnight_regime": None,
                        "additive_ic_contribution": None,
                        "independently_reranked_conditional_ic": None,
                        "mean_observed_fraction": float(np.mean(fractions)),
                        "mean_recent_observed_fraction": None,
                        "readiness_fraction": float(np.mean(ready)),
                        "preopen_observed_fraction": _finite_or_none(
                            np.nanmean(preopen)
                        ),
                        "mean_staleness_minutes": _finite_or_none(np.nanmean(stale)),
                        "gross_spread": None,
                        "intraday_turnover": None,
                        "valid_date_count": int(
                            np.unique(date_idx[selected_samples]).size
                        ),
                    }
                )
            retained_positions = np.asarray(list(symbol_positions.values()))
            retained_fractions = np.asarray(completeness["observed_fraction"])[
                selected_samples
            ][:, retained_positions]
            retained_stale = np.asarray(
                completeness["minutes_since_most_recent_observed_bar"]
            )[selected_samples][:, retained_positions]
            for statistic, values in (
                ("minimum", np.nanmin(retained_fractions, axis=1)),
                ("median", np.nanmedian(retained_fractions, axis=1)),
            ):
                rows.append(
                    {
                        "diagnostic_type": f"{family}_{statistic}",
                        "seed": None,
                        "instrument": f"retained_{family}_{statistic}",
                        "time_scope": bin_name,
                        "decision_indices": json.dumps(list(decisions)),
                        "horizon_minutes": None,
                        "liquidity_quintile": None,
                        "history_category": None,
                        "overnight_regime": None,
                        "additive_ic_contribution": None,
                        "independently_reranked_conditional_ic": None,
                        "mean_observed_fraction": float(np.nanmean(values)),
                        "mean_recent_observed_fraction": None,
                        "readiness_fraction": None,
                        "preopen_observed_fraction": None,
                        "mean_staleness_minutes": _finite_or_none(
                            np.nanmedian(retained_stale)
                        ),
                        "gross_spread": None,
                        "intraday_turnover": None,
                        "valid_date_count": int(
                            np.unique(date_idx[selected_samples]).size
                        ),
                    }
                )
    for seed in STAGE3_SEEDS:
        core = core_by_seed[seed]
        sample_ic = np.asarray(core["sample_ic"])
        economic = core["economic"]
        if not isinstance(economic, EconomicAttributionResult):
            raise TypeError("Core economic attribution is malformed")
        for condition in _opening_condition_masks(shared, metadata):
            sample_mask = np.asarray(condition.sample_mask, dtype=bool)
            for scope_name in ("opening_30", "opening_60"):
                decisions = named_time_scopes()[scope_name]
                condition_metadata = _opening_condition_metadata(
                    condition, date_idx, decision_idx, decisions
                )
                daily_ic = _daily_subset_mean(
                    sample_ic,
                    sample_mask,
                    date_idx,
                    decision_idx,
                    decisions,
                )
                daily_economic = _window_portfolio_daily(
                    economic_window_accounting(
                        economic,
                        date_idx,
                        decision_idx,
                        decisions=decisions,
                        selected_samples=sample_mask,
                    )
                )
                for horizon_position in range(len(HORIZONS) + 1):
                    if horizon_position < len(HORIZONS):
                        horizon = HORIZONS[horizon_position]
                        ic_values = daily_ic[:, horizon_position]
                        gross = daily_economic["gross"][:, horizon_position]
                        entry = daily_economic["entry"][:, horizon_position]
                        intraday = daily_economic["intraday"][:, horizon_position]
                        exit_values = daily_economic["exit"][:, horizon_position]
                        total = daily_economic["total"][:, horizon_position]
                    else:
                        horizon = 0
                        ic_values = _finite_axis_mean(daily_ic, axis=1)
                        gross = _finite_axis_mean(daily_economic["gross"], axis=1)
                        entry = _finite_axis_mean(daily_economic["entry"], axis=1)
                        intraday = _finite_axis_mean(daily_economic["intraday"], axis=1)
                        exit_values = _finite_axis_mean(daily_economic["exit"], axis=1)
                        total = _finite_axis_mean(daily_economic["total"], axis=1)
                    mean_gross = _finite_mean_or_none(gross)
                    mean_total = _finite_mean_or_none(total)
                    rows.append(
                        {
                            "diagnostic_type": f"core_{condition.condition_type}",
                            "condition_category": condition.category,
                            "seed": seed,
                            "instrument": None,
                            "time_scope": scope_name,
                            "decision_indices": json.dumps(list(decisions)),
                            "horizon_minutes": horizon,
                            "liquidity_quintile": None,
                            "history_category": None,
                            "overnight_regime": condition.overnight_regime,
                            "freshness_category": condition.freshness_category,
                            "additive_ic_contribution": _finite_mean_or_none(ic_values),
                            "independently_reranked_conditional_ic": None,
                            "mean_observed_fraction": None,
                            "mean_recent_observed_fraction": None,
                            "readiness_fraction": None,
                            "preopen_observed_fraction": None,
                            "mean_staleness_minutes": None,
                            "mean_daily_gross_contribution": mean_gross,
                            "gross_spread": _finite_mean_or_none(gross),
                            "mean_daily_flat_entry_turnover": _finite_mean_or_none(
                                entry
                            ),
                            "mean_daily_intraday_turnover": _finite_mean_or_none(
                                intraday
                            ),
                            "intraday_turnover": _finite_mean_or_none(intraday),
                            "mean_daily_flat_exit_turnover": _finite_mean_or_none(
                                exit_values
                            ),
                            "mean_daily_total_turnover": mean_total,
                            "gross_contribution_per_unit_turnover": (
                                mean_gross / mean_total
                                if mean_gross is not None
                                and mean_total is not None
                                and mean_total > 0
                                else None
                            ),
                            "break_even_one_way_cost_bps": (
                                10_000.0 * mean_gross / mean_total
                                if mean_gross is not None
                                and mean_total is not None
                                and mean_total > 0
                                else None
                            ),
                            **condition_metadata,
                            "valid_date_count": int(np.isfinite(ic_values).sum()),
                        }
                    )
    return pl.DataFrame(rows, infer_schema_length=None)


def _paired_delta_row(
    *,
    logical: str,
    seed: int,
    scope_type: str,
    scope_name: str,
    decisions: tuple[int, ...],
    horizon_minutes: int,
    current_ic: np.ndarray,
    core_ic: np.ndarray,
    current_spread: np.ndarray,
    core_spread: np.ndarray,
    current_turnover: np.ndarray,
    core_turnover: np.ndarray,
    regime: str | None,
    freshness: str | None,
    bootstrap_replications: int,
    bootstrap_seed: int,
    condition_type: str | None = None,
    condition_category: str | None = None,
    current_entry: np.ndarray | None = None,
    core_entry: np.ndarray | None = None,
    current_exit: np.ndarray | None = None,
    core_exit: np.ndarray | None = None,
    current_total_turnover: np.ndarray | None = None,
    core_total_turnover: np.ndarray | None = None,
) -> dict[str, object]:
    if current_ic.shape != core_ic.shape:
        raise ValueError("Same-seed context delta daily arrays are misaligned")
    valid = np.isfinite(current_ic) & np.isfinite(core_ic)
    delta = np.where(valid, current_ic - core_ic, np.nan)
    bootstrap = moving_block_bootstrap(
        delta,
        replications=bootstrap_replications,
        seed=bootstrap_seed,
    )

    def paired_mean(
        current: np.ndarray | None, core: np.ndarray | None
    ) -> float | None:
        if current is None or core is None:
            return None
        current = np.asarray(current, dtype=np.float64)
        core = np.asarray(core, dtype=np.float64)
        if current.shape != valid.shape or core.shape != valid.shape:
            raise ValueError("Paired economic daily arrays are misaligned")
        selected = valid & np.isfinite(current) & np.isfinite(core)
        return (
            float(np.mean(current[selected] - core[selected]))
            if selected.any()
            else None
        )

    return {
        "aggregation": "seed",
        "logical_configuration": logical,
        "added_context": ADDED_CONTEXT_BY_LOGICAL_CONFIGURATION[logical],
        "seed": seed,
        "scope_type": scope_type,
        "scope": scope_name,
        "decision_indices": json.dumps(list(decisions)),
        "horizon_minutes": horizon_minutes,
        "condition_type": condition_type,
        "condition_category": condition_category,
        "overnight_regime": regime,
        "freshness_category": freshness,
        "mean_paired_ic_delta": _finite_mean_or_none(delta),
        "ic_delta_interval_lower_95": _finite_or_none(
            float(bootstrap["interval_lower_95"])
        ),
        "ic_delta_interval_upper_95": _finite_or_none(
            float(bootstrap["interval_upper_95"])
        ),
        "bootstrap_finite_replication_count": int(
            bootstrap["finite_replication_count"]
        ),
        "mean_paired_gross_spread_delta": paired_mean(current_spread, core_spread),
        "mean_paired_flat_entry_turnover_delta": paired_mean(current_entry, core_entry),
        "mean_paired_intraday_turnover_delta": paired_mean(
            current_turnover, core_turnover
        ),
        "mean_paired_flat_exit_turnover_delta": paired_mean(current_exit, core_exit),
        "mean_paired_total_turnover_delta": paired_mean(
            current_total_turnover, core_total_turnover
        ),
        "valid_date_count": int(valid.sum()),
        "preregistered_primary_question": (
            scope_name == "opening_30"
            and horizon_minutes == 0
            and condition_type is None
        ),
        "exploratory": not (
            scope_name == "opening_30"
            and horizon_minutes == 0
            and condition_type is None
        ),
    }


def _build_context_time_deltas(
    cache_paths: dict[tuple[str, int], Path],
    core_by_seed: dict[int, dict[str, object]],
    shared: dict[str, np.ndarray],
    metadata: dict[str, object],
    *,
    bootstrap_replications: int | None = None,
) -> pl.DataFrame:
    if bootstrap_replications is None:
        bootstrap_replications = BOOTSTRAP_REPLICATIONS
    if not all(
        (logical, seed) in cache_paths
        for logical in STAGE3_LOGICAL_CONFIGURATION_ORDER
        for seed in STAGE3_SEEDS
    ):
        return pl.DataFrame(
            schema={
                "aggregation": pl.String,
                "logical_configuration": pl.String,
                "seed": pl.Int64,
                "scope": pl.String,
                "horizon_minutes": pl.Int64,
                "mean_paired_ic_delta": pl.Float64,
            }
        )
    date_idx = np.asarray(shared["date_idx"], dtype=np.int64)
    decision_idx = np.asarray(shared["decision_idx"], dtype=np.int64)
    targets = np.asarray(shared["targets"])
    raw_returns = np.asarray(shared["raw_returns"])
    label_mask = np.asarray(shared["label_mask"], dtype=bool)
    all_samples = np.ones(date_idx.shape, dtype=bool)
    rows: list[dict[str, object]] = []
    primary_bins = {
        row["name"]: tuple(row["decision_indices"]) for row in _time_bin_metadata()
    }
    scopes = {**primary_bins, **named_time_scopes()}
    opening_conditions = _opening_condition_masks(shared, metadata)

    def daily_economic(
        result: EconomicAttributionResult,
        decisions: tuple[int, ...],
        sample_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        return _window_portfolio_daily(
            economic_window_accounting(
                result,
                date_idx,
                decision_idx,
                decisions=decisions,
                selected_samples=sample_mask,
            )
        )

    for logical_index, logical in enumerate(STAGE3_LOGICAL_CONFIGURATION_ORDER[1:]):
        added = ADDED_CONTEXT_BY_LOGICAL_CONFIGURATION[logical]
        if added in LOCAL_CONTEXT_SYMBOLS:
            completeness = metadata["local_completeness"]
            instrument_position = LOCAL_CONTEXT_SYMBOLS.index(added)
        else:
            completeness = metadata["global_completeness"]
            instrument_position = GLOBAL_CONTEXT_SYMBOLS.index(added)
        if not isinstance(completeness, dict):
            raise TypeError("Added-context completeness is malformed")
        ready = np.asarray(completeness["ready"])[..., instrument_position]
        stale = np.asarray(completeness["minutes_since_most_recent_observed_bar"])[
            ..., instrument_position
        ]
        added_freshness = _freshness_categories(ready, stale)
        conditional_masks = [
            *opening_conditions,
            *[
                OpeningCondition(
                    "added_context_freshness",
                    category,
                    mask,
                    freshness_category=category,
                )
                for category, mask in added_freshness.items()
            ],
        ]
        for seed in STAGE3_SEEDS:
            predictions = np.load(
                cache_paths[(logical, seed)], mmap_mode="r", allow_pickle=False
            )
            additive = additive_spearman_contributions(predictions, targets, label_mask)
            economic = economic_stock_attribution(
                predictions,
                raw_returns,
                label_mask,
                date_idx,
                decision_idx,
            )
            core = core_by_seed[seed]
            core_economic = core["economic"]
            if not isinstance(core_economic, EconomicAttributionResult):
                raise TypeError("Core economic attribution is malformed")
            row_counter = 0

            def append_scope(
                scope_type: str,
                scope_name: str,
                decisions: tuple[int, ...],
                sample_mask: np.ndarray,
                condition_type: str | None = None,
                condition_category: str | None = None,
                regime: str | None = None,
                freshness: str | None = None,
            ) -> None:
                nonlocal row_counter
                current_ic = _daily_subset_mean(
                    additive.sample_ic,
                    sample_mask,
                    date_idx,
                    decision_idx,
                    decisions,
                )
                core_ic = _daily_subset_mean(
                    np.asarray(core["sample_ic"]),
                    sample_mask,
                    date_idx,
                    decision_idx,
                    decisions,
                )
                current_economic = daily_economic(economic, decisions, sample_mask)
                core_economic_daily = daily_economic(
                    core_economic, decisions, sample_mask
                )
                for horizon_position in range(len(HORIZONS) + 1):
                    if horizon_position < len(HORIZONS):
                        horizon = HORIZONS[horizon_position]

                        def select(values: np.ndarray) -> np.ndarray:
                            return values[:, horizon_position]

                    else:
                        horizon = 0

                        def select(values: np.ndarray) -> np.ndarray:
                            return _finite_axis_mean(values, axis=1)

                    rows.append(
                        _paired_delta_row(
                            logical=logical,
                            seed=seed,
                            scope_type=scope_type,
                            scope_name=scope_name,
                            decisions=decisions,
                            horizon_minutes=horizon,
                            current_ic=select(current_ic),
                            core_ic=select(core_ic),
                            current_spread=select(current_economic["gross"]),
                            core_spread=select(core_economic_daily["gross"]),
                            current_turnover=select(current_economic["intraday"]),
                            core_turnover=select(core_economic_daily["intraday"]),
                            current_entry=select(current_economic["entry"]),
                            core_entry=select(core_economic_daily["entry"]),
                            current_exit=select(current_economic["exit"]),
                            core_exit=select(core_economic_daily["exit"]),
                            current_total_turnover=select(current_economic["total"]),
                            core_total_turnover=select(core_economic_daily["total"]),
                            condition_type=condition_type,
                            condition_category=condition_category,
                            regime=regime,
                            freshness=freshness,
                            bootstrap_replications=bootstrap_replications,
                            bootstrap_seed=(
                                BOOTSTRAP_SEED
                                + logical_index * 10_000_000
                                + seed * 100_000
                                + row_counter
                            ),
                        )
                    )
                    row_counter += 1

            for scope_name, decisions in scopes.items():
                append_scope(
                    "decision_bin" if scope_name.startswith("bin_") else "named_scope",
                    scope_name,
                    decisions,
                    all_samples,
                )
            for decision in range(EXPECTED_DECISIONS_PER_DATE):
                append_scope(
                    "decision_5m",
                    f"decision_{decision:02d}",
                    (decision,),
                    all_samples,
                )
            for condition in conditional_masks:
                for scope_name in ("opening_30", "opening_60"):
                    append_scope(
                        condition.condition_type,
                        scope_name,
                        named_time_scopes()[scope_name],
                        np.asarray(condition.sample_mask, dtype=bool),
                        condition.condition_type,
                        condition.category,
                        condition.overnight_regime,
                        condition.freshness_category,
                    )

    identity_fields = (
        "logical_configuration",
        "added_context",
        "scope_type",
        "scope",
        "decision_indices",
        "horizon_minutes",
        "condition_type",
        "condition_category",
        "overnight_regime",
        "freshness_category",
        "preregistered_primary_question",
        "exploratory",
    )
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[field] for field in identity_fields)
        grouped.setdefault(key, []).append(row)
    across_rows: list[dict[str, object]] = []
    for key, seed_rows in grouped.items():
        if len(seed_rows) != len(STAGE3_SEEDS):
            raise RuntimeError("Across-seed delta cell lacks three matched seeds")
        values_by_seed = {
            int(row["seed"]): float(row["mean_paired_ic_delta"])
            for row in seed_rows
            if row["mean_paired_ic_delta"] is not None
        }
        if len(values_by_seed) != len(STAGE3_SEEDS):
            continue
        values = np.asarray([values_by_seed[seed] for seed in STAGE3_SEEDS])
        positive, zero, negative = _sign_counts(values)
        row = {field: value for field, value in zip(identity_fields, key, strict=True)}

        def mean_field(name: str) -> float | None:
            values_for_field = [
                value[name] for value in seed_rows if value[name] is not None
            ]
            return float(np.mean(values_for_field)) if values_for_field else None

        row.update(
            {
                "aggregation": "across_seed",
                "seed": None,
                "mean_paired_ic_delta": float(values.mean()),
                "ic_delta_interval_lower_95": None,
                "ic_delta_interval_upper_95": None,
                "bootstrap_finite_replication_count": None,
                "mean_paired_gross_spread_delta": mean_field(
                    "mean_paired_gross_spread_delta"
                ),
                "mean_paired_flat_entry_turnover_delta": mean_field(
                    "mean_paired_flat_entry_turnover_delta"
                ),
                "mean_paired_intraday_turnover_delta": mean_field(
                    "mean_paired_intraday_turnover_delta"
                ),
                "mean_paired_flat_exit_turnover_delta": mean_field(
                    "mean_paired_flat_exit_turnover_delta"
                ),
                "mean_paired_total_turnover_delta": mean_field(
                    "mean_paired_total_turnover_delta"
                ),
                "valid_date_count": int(
                    min(value["valid_date_count"] for value in seed_rows)
                ),
                **{f"delta_seed_{seed}": values_by_seed[seed] for seed in STAGE3_SEEDS},
                "across_seed_median_delta": float(np.median(values)),
                "across_seed_minimum_delta": float(values.min()),
                "across_seed_maximum_delta": float(values.max()),
                "positive_seed_count": positive,
                "zero_seed_count": zero,
                "negative_seed_count": negative,
            }
        )
        across_rows.append(row)
    return pl.DataFrame([*rows, *across_rows], infer_schema_length=None)


def _record_artifact(
    state: dict[str, object],
    state_path: Path,
    name: str,
    path: Path,
    writer: Any,
) -> None:
    artifacts = state.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Analysis artifact state is malformed")
    recorded = artifacts.get(name)
    if recorded is not None:
        if (
            not isinstance(recorded, dict)
            or recorded.get("path") != str(path)
            or not path.is_file()
            or recorded.get("sha256") != _sha256(path)
        ):
            raise ValueError(f"Recorded analysis artifact is invalid: {name}")
        return
    staged = path.with_name(f"{path.name}.staged")
    pending = state.get("pending_artifact")
    if pending is not None:
        if (
            not isinstance(pending, dict)
            or pending.get("name") != name
            or pending.get("path") != str(path)
            or pending.get("staged_path") != str(staged)
        ):
            raise ValueError("Pending analysis artifact does not match output order")
        candidate = staged if staged.is_file() else path
        if (
            not candidate.is_file()
            or pending.get("sha256") != _sha256(candidate)
            or pending.get("bytes") != candidate.stat().st_size
        ):
            raise ValueError(f"Pending analysis artifact is invalid: {name}")
        if candidate == staged:
            if path.exists():
                raise FileExistsError(
                    f"Ambiguous pending artifact target exists: {path}"
                )
            os.replace(staged, path)
        artifacts[name] = {
            "path": str(path),
            "sha256": pending["sha256"],
            "bytes": pending["bytes"],
        }
        state["pending_artifact"] = None
        _atomic_write_json(state_path, state)
        return
    if path.exists():
        raise FileExistsError(f"Refusing to adopt unrecorded analysis artifact: {path}")
    if staged.exists():
        staged.unlink()
    writer(staged)
    pending = {
        "name": name,
        "path": str(path),
        "staged_path": str(staged),
        "sha256": _sha256(staged),
        "bytes": staged.stat().st_size,
    }
    state["pending_artifact"] = pending
    _atomic_write_json(state_path, state)
    os.replace(staged, path)
    artifacts[name] = {
        "path": str(path),
        "sha256": pending["sha256"],
        "bytes": pending["bytes"],
    }
    state["pending_artifact"] = None
    _atomic_write_json(state_path, state)


def _validate_completed_artifacts(state: dict[str, object]) -> None:
    if state.get("pending_artifact") is not None:
        raise ValueError("Completed analysis state has a pending artifact")
    artifacts = state.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(FINAL_ARTIFACT_NAMES):
        raise ValueError("Completed analysis state has the wrong artifact matrix")
    for name, metadata in artifacts.items():
        if not isinstance(metadata, dict):
            raise ValueError(f"Completed artifact metadata is invalid: {name}")
        path = Path(str(metadata.get("path")))
        if not path.is_file() or _sha256(path) != metadata.get("sha256"):
            raise ValueError(f"Completed artifact hash is invalid: {name}")


def _cache_metric_gates(
    inputs: AnalysisInputs, cache_dir: Path
) -> list[dict[str, object]]:
    gates = []
    for job in inputs.jobs:
        manifest_path = _cache_directory(cache_dir, job) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        gates.append(
            {
                "logical_configuration": job.logical_configuration,
                "seed": job.seed,
                **manifest["metric_reproduction_gate"],
            }
        )
    return gates


def _analysis_summary(
    inputs: AnalysisInputs,
    scope: str,
    output_dir: Path,
    metadata: dict[str, object],
    cache_dir: Path,
    reconstruction: dict[str, object],
    liquidity_checks: dict[str, object],
    frames: dict[str, pl.DataFrame],
) -> dict[str, object]:
    prediction_bytes = sum(
        (_cache_directory(cache_dir, job) / "predictions.npy").stat().st_size
        for job in inputs.jobs
    )
    shared_bytes = sum(
        path.stat().st_size
        for path in shared_validation_directory(cache_dir).glob("*.npy")
    )
    return {
        "analysis_name": ANALYSIS_NAME,
        "analysis_version": ANALYSIS_VERSION,
        "scope": scope,
        "split": "validation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "stage3_state_path": str(inputs.state_path),
            "stage3_state_sha256": inputs.state_sha256,
            "feature_store_resolved_path": str(inputs.feature_store),
            "feature_store_identity": inputs.feature_identity,
            "analyzer_git_commit_sha": inputs.analyzer_git_commit_sha,
            "analyzer_worktree_clean": inputs.analyzer_worktree_clean,
            "analyzer_source_sha256": inputs.analyzer_source_sha256,
            "inference_code_sha256": inputs.inference_code_sha256,
            "split_boundaries": _split_boundaries(),
            "job_count": len(inputs.jobs),
            "jobs": [
                {
                    "logical_configuration": job.logical_configuration,
                    "context_ablation": job.context_ablation,
                    "context_ablation_specification": get_context_ablation(
                        job.context_ablation
                    ).metadata(),
                    "seed": job.seed,
                    "run_dir": str(job.run_dir),
                    "run_manifest_path": str(job.run_manifest_path),
                    "run_manifest_sha256": job.run_manifest_sha256,
                    "checkpoint_path": str(job.checkpoint_path),
                    "checkpoint_sha256": job.checkpoint_sha256,
                    "producing_git_commit_sha": job.producing_git_commit_sha,
                }
                for job in inputs.jobs
            ],
        },
        "metric_reproduction_checks": _cache_metric_gates(inputs, cache_dir),
        "stock_contribution_reconstruction_checks": reconstruction,
        "time_decomposition_reconstruction_checks": reconstruction["by_seed"],
        "gross_spread_and_turnover_reconstruction_checks": {
            seed: values["economic"]
            for seed, values in reconstruction["by_seed"].items()
        },
        "liquidity_reconstruction_checks": liquidity_checks,
        "time_bins": _time_bin_metadata(),
        "named_time_scopes": {
            name: list(decisions) for name, decisions in named_time_scopes().items()
        },
        "liquidity": {
            "feature_channel_name": metadata["axes"]["liquidity_channel_name"],
            "feature_channel_index": metadata["axes"]["liquidity_channel_index"],
            "affine_metadata": metadata["axes"]["dollar_volume_log_affine"],
            "eligibility_threshold": metadata["eligibility_liquidity_threshold"],
            "adaptive_bucket_count_minimum_validation": int(
                np.asarray(metadata["adaptive_liquidity_bucket_count"]).min()
            ),
        },
        "overnight_regime": {
            "training_only_thresholds": metadata["overnight_thresholds"],
            "threshold_source_split": "training",
            "validation_dates_used_to_fit_thresholds": 0,
        },
        "bootstrap": {
            "method": "moving blocks of aligned trading dates",
            "block_trading_days": BOOTSTRAP_BLOCK_TRADING_DAYS,
            "replications": BOOTSTRAP_REPLICATIONS,
            "deterministic_seed": BOOTSTRAP_SEED,
            "resampling_unit": "trading_date",
            "overlapping_decisions_and_horizons_are_not_independent_units": True,
        },
        "coverage_warnings": [
            "Individual stock by five-minute estimates are exploratory and noisy.",
            f"Fixed point-in-time liquidity-quintile IC is null when a quintile has fewer than {MIN_IC_EQUITIES} eligible labels in a decision cell (MIN_IC_EQUITIES={MIN_IC_EQUITIES}).",
            f"Adaptive fallback-bucket IC is null when a fallback bucket has fewer than {MIN_IC_EQUITIES} eligible labels in a decision cell (MIN_IC_EQUITIES={MIN_IC_EQUITIES}); empty aggregate estimates remain null.",
            "The existing intraday turnover metric has no previous portfolio at the first decision; flat entry and exit are reported separately.",
            "Gross signal-spread diagnostics use overlapping decisions/horizons and are not an executable annualized portfolio.",
            "Opening-history and context-completeness results are descriptive and do not identify a causal opening mechanism.",
        ],
        "artifact_row_counts": {name: frame.height for name, frame in frames.items()},
        "cache_storage": {
            "prediction_bytes": prediction_bytes,
            "shared_validation_bytes": shared_bytes,
            "total_bytes": prediction_bytes + shared_bytes,
        },
        "primary_outputs": [
            "stock_attribution",
            "time_of_day_bins",
            "liquidity_attribution",
            "liquidity_time_attribution",
            "opening_regimes broad strata",
            "opening_30 same-seed context deltas",
        ],
        "exploratory_outputs": [
            "stock by five-minute attribution",
            "full context by decision by horizon deltas",
            "fine context freshness cells",
        ],
        "transaction_cost_model": None,
        "break_even_one_way_costs_reported": True,
        "selection": None,
        "training_performed": False,
        "test_data_used": False,
        "final_test_remained_sealed": True,
    }


def run_analysis(
    stage3_state_path: Path,
    output_dir: Path,
    scope: str,
    cache_dir: Path | None = None,
) -> Path:
    output_dir = output_dir.resolve()
    _reject_test_derived_path(output_dir, "analysis output path")
    cache_root = _resolve_cache_directory(output_dir, cache_dir)
    inputs = validate_analysis_inputs(stage3_state_path, scope)
    if not inputs.analyzer_worktree_clean:
        raise RuntimeError("Non-dry analysis requires a clean worktree")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "analysis_state.json"
    lock_path = output_dir / "analysis.lock"
    with exclusive_process_lock(lock_path, ANALYSIS_NAME):
        state = _load_analysis_state(state_path, inputs, scope, cache_root)
        if state.get("status") == "completed":
            _validate_completed_artifacts(state)
            _write_or_validate_shared_cache(cache_root, inputs)
            for job in inputs.jobs:
                _validate_cache_manifest(
                    _cache_directory(cache_root, job) / "manifest.json",
                    _job_cache_identity(inputs, job),
                )
            if any(
                gate.get("passed") is not True
                for gate in _cache_metric_gates(inputs, cache_root)
            ):
                raise ValueError("Completed analysis cache lacks metric parity")
            return output_dir / "analysis_manifest.json"
        _atomic_write_json(state_path, state)
        with exclusive_process_lock(
            PRODUCTION_TRAINING_LOCK, "stock/time attribution inference"
        ) as production_lock:
            with exclusive_process_lock(
                cache_root / "cache.lock", "stock/time attribution cache"
            ):
                cache_paths, shared = _adopt_or_infer_caches(
                    cache_root,
                    inputs,
                    state,
                    state_path,
                    production_lock,
                )
        metadata = _load_analysis_metadata(inputs, shared)
        core_frames, core_by_seed, reconstruction = _build_core_outputs(
            cache_paths, shared, metadata
        )
        liquidity, liquidity_time, liquidity_checks = _build_liquidity_outputs(
            cache_paths, core_by_seed, shared, metadata
        )
        opening = _build_opening_regimes(core_by_seed, shared, metadata)
        context_deltas = _build_context_time_deltas(
            cache_paths, core_by_seed, shared, metadata
        )
        frames = {
            **core_frames,
            "liquidity_attribution": liquidity,
            "liquidity_time_attribution": liquidity_time,
            "opening_regimes": opening,
            "context_time_deltas": context_deltas,
        }
        _assert_repository_identity(inputs)
        artifact_specs = (
            (
                "stock_attribution.parquet",
                lambda path: _atomic_write_parquet(path, frames["stock_attribution"]),
            ),
            (
                "stock_attribution.csv",
                lambda path: _atomic_write_csv(path, frames["stock_attribution"]),
            ),
            (
                "stock_time_attribution.parquet",
                lambda path: _atomic_write_parquet(
                    path, frames["stock_time_attribution"]
                ),
            ),
            (
                "liquidity_attribution.parquet",
                lambda path: _atomic_write_parquet(
                    path, frames["liquidity_attribution"]
                ),
            ),
            (
                "liquidity_time_attribution.parquet",
                lambda path: _atomic_write_parquet(
                    path, frames["liquidity_time_attribution"]
                ),
            ),
            (
                "time_of_day_5m.parquet",
                lambda path: _atomic_write_parquet(path, frames["time_of_day_5m"]),
            ),
            (
                "time_of_day_5m.csv",
                lambda path: _atomic_write_csv(path, frames["time_of_day_5m"]),
            ),
            (
                "time_of_day_bins.parquet",
                lambda path: _atomic_write_parquet(path, frames["time_of_day_bins"]),
            ),
            (
                "time_of_day_bins.csv",
                lambda path: _atomic_write_csv(path, frames["time_of_day_bins"]),
            ),
            (
                "opening_regimes.parquet",
                lambda path: _atomic_write_parquet(path, frames["opening_regimes"]),
            ),
            (
                "context_time_deltas.parquet",
                lambda path: _atomic_write_parquet(path, frames["context_time_deltas"]),
            ),
            (
                "context_time_deltas.csv",
                lambda path: _atomic_write_csv(path, frames["context_time_deltas"]),
            ),
        )
        for name, writer in artifact_specs:
            _record_artifact(
                state,
                state_path,
                name,
                output_dir / name,
                writer,
            )
        summary = _analysis_summary(
            inputs,
            scope,
            output_dir,
            metadata,
            cache_root,
            reconstruction,
            liquidity_checks,
            frames,
        )
        _record_artifact(
            state,
            state_path,
            "summary.json",
            output_dir / "summary.json",
            lambda path: _atomic_write_json(path, summary),
        )
        manifest = {
            "analysis_name": ANALYSIS_NAME,
            "analysis_version": ANALYSIS_VERSION,
            "status": "completed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": _analysis_configuration(inputs, scope, cache_root),
            "artifacts": state["artifacts"],
            "prediction_cache_manifests": [
                {
                    "path": str(_cache_directory(cache_root, job) / "manifest.json"),
                    "sha256": _sha256(
                        _cache_directory(cache_root, job) / "manifest.json"
                    ),
                }
                for job in inputs.jobs
            ],
            "shared_validation_manifest": {
                "path": str(shared_validation_directory(cache_root) / "manifest.json"),
                "sha256": _sha256(
                    shared_validation_directory(cache_root) / "manifest.json"
                ),
            },
            "selection": None,
            "training_performed": False,
            "test_data_used": False,
        }
        _record_artifact(
            state,
            state_path,
            "analysis_manifest.json",
            output_dir / "analysis_manifest.json",
            lambda path: _atomic_write_json(path, manifest),
        )
        state["status"] = "completed"
        state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(state_path, state)
    return output_dir / "analysis_manifest.json"


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--scope", choices=SCOPE_CHOICES, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(arguments)


def main() -> None:
    args = parse_args()
    stage3_state = args.stage3_state.resolve()
    output_dir = args.output_dir.resolve()
    if args.dry_run:
        print(
            json.dumps(
                dry_run_payload(stage3_state, output_dir, args.scope, args.cache_dir),
                indent=2,
                allow_nan=False,
            ),
            flush=True,
        )
        return
    manifest = run_analysis(stage3_state, output_dir, args.scope, args.cache_dir)
    print(f"Completed stock/time attribution analysis: {manifest}", flush=True)


if __name__ == "__main__":
    main()
