from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch

from brazil_rv.preprocessing.contract import (
    DECISION_CONTEXT_INDICES,
    DECISION_EQUITY_INDICES,
    DECISION_GLOBAL_INDICES,
    DECISION_TIMES,
    DYNAMIC_CHANNELS,
    SLOW_CHANNELS,
)

from .contract import (
    CANONICAL_DROPPED_LOCAL_SLOTS,
    CANONICAL_RETAINED_GLOBAL_SLOTS,
    EXPECTED_DECISIONS_PER_DATE,
    GLOBAL_WINDOW_MINUTES,
    HORIZONS,
    MIN_IC_EQUITIES,
    TRAIN_END,
    TRAIN_START,
)
from .data import select_sample_split
from .evaluate import collect_neural_evaluation
from .metrics import average_ranks, ranking_diagnostics

BOOTSTRAP_BLOCK_TRADING_DAYS = 5
BOOTSTRAP_REPLICATIONS = 10_000
BOOTSTRAP_SEED = 20260805
OVERNIGHT_TAIL_QUANTILE = 0.10
OBSERVED_CHANNEL = DYNAMIC_CHANNELS.index("observed")
OVERNIGHT_CHANNEL = SLOW_CHANNELS.index("overnight_gap_normalized")
RETAINED_LOCAL_SLOTS = tuple(
    slot for slot in range(7) if slot not in CANONICAL_DROPPED_LOCAL_SLOTS
)


@dataclass(frozen=True)
class OpeningDiagnostics:
    market_overnight_gap: np.ndarray
    opening_thresholds: tuple[float, float]
    b3_observed_bars: np.ndarray
    local_observed_fraction: np.ndarray
    global_observed_fraction: np.ndarray
    global_staleness_minutes: np.ndarray
    preopen_context_complete: np.ndarray


@dataclass(frozen=True)
class AttributionInputs:
    run_name: str
    predictions: np.ndarray
    targets: np.ndarray
    raw_returns: np.ndarray
    label_mask: np.ndarray
    date_idx: np.ndarray
    decision_idx: np.ndarray
    security_ids: tuple[str, ...]
    opening: OpeningDiagnostics


@dataclass(frozen=True)
class RankDecomposition:
    contributions: np.ndarray
    predicted_coordinates: np.ndarray
    target_coordinates: np.ndarray
    sample_ic: np.ndarray
    eligible: np.ndarray


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-only stock/time attribution"
    )
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    return parser.parse_args(arguments)


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _cache_path(cache_dir: Path, run_dir: Path) -> Path:
    resolved = run_dir.resolve()
    return cache_dir.joinpath(*resolved.parts[1:], "predictions.npz")


def _load_cached(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as values:
        return {name: values[name] for name in values.files}


def _write_cache(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def _security_ids(store: Path) -> tuple[str, ...]:
    frame = pl.read_parquet(store / "equity_index.parquet").sort("equity_slot")
    return tuple(str(value) for value in frame.get_column("security_id"))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2:
        return float("nan")
    left -= left.mean()
    right -= right.mean()
    denominator = math.sqrt(float(np.sum(left**2) * np.sum(right**2)))
    return float(np.sum(left * right) / denominator) if denominator else float("nan")


def _finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


def rank_decomposition(
    predictions: np.ndarray,
    targets: np.ndarray,
    label_mask: np.ndarray,
) -> RankDecomposition:
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    label_mask = np.asarray(label_mask, dtype=bool)
    if predictions.ndim != 3 or predictions.shape != targets.shape:
        raise ValueError(
            "Predictions and targets must share sample/equity/horizon shape"
        )
    if label_mask.shape != predictions.shape:
        raise ValueError("Label mask must match predictions")
    sample_count, equity_count, horizon_count = predictions.shape
    shape = (sample_count, equity_count, horizon_count)
    contributions = np.full(shape, np.nan, dtype=np.float64)
    predicted_coordinates = np.full(shape, np.nan, dtype=np.float64)
    target_coordinates = np.full(shape, np.nan, dtype=np.float64)
    sample_ic = np.full((sample_count, horizon_count), np.nan, dtype=np.float64)
    eligible = np.zeros((sample_count, horizon_count), dtype=bool)
    for sample in range(sample_count):
        for horizon in range(horizon_count):
            valid = np.flatnonzero(label_mask[sample, :, horizon])
            if valid.size < MIN_IC_EQUITIES:
                continue
            predicted = average_ranks(predictions[sample, valid, horizon])
            actual = average_ranks(targets[sample, valid, horizon])
            predicted -= predicted.mean()
            actual -= actual.mean()
            predicted_sum_squares = float(np.sum(predicted**2))
            actual_sum_squares = float(np.sum(actual**2))
            denominator = math.sqrt(predicted_sum_squares * actual_sum_squares)
            if denominator == 0.0:
                continue
            eligible[sample, horizon] = True
            contributions[sample, :, horizon] = 0.0
            contributions[sample, valid, horizon] = predicted * actual / denominator
            predicted_coordinates[sample, valid, horizon] = predicted / math.sqrt(
                predicted_sum_squares
            )
            target_coordinates[sample, valid, horizon] = actual / math.sqrt(
                actual_sum_squares
            )
            sample_ic[sample, horizon] = float(contributions[sample, :, horizon].sum())
    return RankDecomposition(
        contributions,
        predicted_coordinates,
        target_coordinates,
        sample_ic,
        eligible,
    )


def stock_attribution(inputs: AttributionInputs) -> pl.DataFrame:
    decomposition = rank_decomposition(
        inputs.predictions, inputs.targets, inputs.label_mask
    )
    _, equity_count, _ = inputs.predictions.shape
    rows: list[dict[str, object]] = []
    for equity in range(equity_count):
        for horizon, minutes in enumerate(HORIZONS):
            eligible = decomposition.eligible[:, horizon]
            supported = eligible & inputs.label_mask[:, equity, horizon]
            rows.append(
                {
                    "run": inputs.run_name,
                    "security_id": inputs.security_ids[equity],
                    "equity_slot": equity,
                    "horizon_minutes": minutes,
                    "eligible_samples": int(eligible.sum()),
                    "observations": int(supported.sum()),
                    "coverage": float(supported.sum() / eligible.sum())
                    if eligible.any()
                    else float("nan"),
                    "mean_spearman_contribution": float(
                        np.mean(decomposition.contributions[eligible, equity, horizon])
                    )
                    if eligible.any()
                    else float("nan"),
                    "time_series_rank_skill": _correlation(
                        decomposition.predicted_coordinates[supported, equity, horizon],
                        decomposition.target_coordinates[supported, equity, horizon],
                    ),
                    "mean_raw_return": _finite_mean(
                        inputs.raw_returns[supported, equity, horizon]
                    ),
                }
            )
    return pl.DataFrame(rows)


def _daily_mean(
    values: np.ndarray, date_idx: np.ndarray, selected: np.ndarray
) -> np.ndarray:
    dates = np.unique(date_idx)
    result = np.full(dates.size, np.nan, dtype=np.float64)
    for position, date_value in enumerate(dates):
        on_date = selected & (date_idx == date_value)
        result[position] = _finite_mean(values[on_date])
    return result


def _group_metrics(
    inputs: AttributionInputs,
    group_name: str,
    group_values: np.ndarray,
    decomposition: RankDecomposition | None = None,
) -> pl.DataFrame:
    if decomposition is None:
        decomposition = rank_decomposition(
            inputs.predictions, inputs.targets, inputs.label_mask
        )
    diagnostics = ranking_diagnostics(
        inputs.predictions,
        inputs.raw_returns,
        inputs.label_mask,
        inputs.date_idx,
        inputs.decision_idx,
    )
    rows: list[dict[str, object]] = []
    for group in np.unique(group_values):
        selected = group_values == group
        for horizon, minutes in enumerate(HORIZONS):
            value = group.item() if hasattr(group, "item") else group
            rows.append(
                {
                    "run": inputs.run_name,
                    group_name: value,
                    "horizon_minutes": minutes,
                    "samples": int(selected.sum()),
                    "eligible_samples": int(
                        np.isfinite(decomposition.sample_ic[selected, horizon]).sum()
                    ),
                    "mean_spearman_ic": _finite_mean(
                        _daily_mean(
                            decomposition.sample_ic[:, horizon],
                            inputs.date_idx,
                            selected,
                        )
                    ),
                    "mean_top_minus_bottom": _finite_mean(
                        _daily_mean(
                            diagnostics["top_minus_bottom"][:, horizon],
                            inputs.date_idx,
                            selected,
                        )
                    ),
                    "mean_one_way_turnover": _finite_mean(
                        _daily_mean(
                            diagnostics["one_way_turnover"][:, horizon],
                            inputs.date_idx,
                            selected,
                        )
                    ),
                    "label_coverage": float(
                        inputs.label_mask[selected, :, horizon].mean()
                    ),
                }
            )
    return pl.DataFrame(rows)


def primary_time_bins(
    decision_count: int = EXPECTED_DECISIONS_PER_DATE, width: int = 6
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
    return tuple(bins)


def time_of_day_attribution(inputs: AttributionInputs) -> pl.DataFrame:
    frame = _group_metrics(inputs, "decision_idx", inputs.decision_idx)
    metadata = pl.DataFrame(
        {
            "decision_idx": np.arange(EXPECTED_DECISIONS_PER_DATE, dtype=np.int64),
            "decision_time": [value.isoformat() for value in DECISION_TIMES],
        }
    )
    return frame.join(metadata, on="decision_idx")


def time_of_day_30m_attribution(inputs: AttributionInputs) -> pl.DataFrame:
    bins = primary_time_bins()
    decision_to_bin = np.empty(EXPECTED_DECISIONS_PER_DATE, dtype=np.int8)
    for index, decisions in enumerate(bins):
        decision_to_bin[list(decisions)] = index
    frame = _group_metrics(inputs, "time_bin_30m", decision_to_bin[inputs.decision_idx])
    metadata = pl.DataFrame(
        {
            "time_bin_30m": np.arange(len(bins), dtype=np.int8),
            "bin_name": [f"bin_{index + 1:02d}" for index in range(len(bins))],
            "start_decision_time": [
                DECISION_TIMES[decisions[0]].isoformat() for decisions in bins
            ],
            "end_decision_time": [
                DECISION_TIMES[decisions[-1]].isoformat() for decisions in bins
            ],
            "decision_count": [len(decisions) for decisions in bins],
        }
    )
    return frame.join(metadata, on="time_bin_30m")


def horizon_attribution(inputs: AttributionInputs) -> pl.DataFrame:
    return _group_metrics(
        inputs, "all_samples", np.zeros(inputs.date_idx.size, dtype=np.int8)
    ).drop("all_samples")


def learn_overnight_thresholds(training_market_gap: np.ndarray) -> tuple[float, float]:
    values = np.asarray(training_market_gap, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < MIN_IC_EQUITIES:
        raise ValueError("Training period has too few overnight-gap observations")
    low, high = np.quantile(
        values, (OVERNIGHT_TAIL_QUANTILE, 1.0 - OVERNIGHT_TAIL_QUANTILE)
    )
    return float(low), float(high)


def opening_attribution(inputs: AttributionInputs) -> pl.DataFrame:
    low, high = inputs.opening.opening_thresholds
    gap = inputs.opening.market_overnight_gap
    regimes = np.where(
        ~np.isfinite(gap),
        "unavailable",
        np.where(
            gap <= low,
            "large_negative",
            np.where(gap >= high, "large_positive", "middle"),
        ),
    )
    return _group_metrics(inputs, "opening_regime", regimes)


def _fraction_categories(values: np.ndarray) -> np.ndarray:
    return np.select(
        (values < 0.80, values < 0.95, np.isfinite(values)),
        ("below_80pct", "80_to_95pct", "at_least_95pct"),
        default="unavailable",
    )


def opening_context_attribution(inputs: AttributionInputs) -> pl.DataFrame:
    opening = inputs.opening
    bar_categories = np.select(
        (
            opening.b3_observed_bars <= 30,
            opening.b3_observed_bars <= 60,
            opening.b3_observed_bars <= 90,
            opening.b3_observed_bars <= 120,
            opening.b3_observed_bars <= 180,
            np.isfinite(opening.b3_observed_bars),
        ),
        (
            "0_to_30_bars",
            "31_to_60_bars",
            "61_to_90_bars",
            "91_to_120_bars",
            "121_to_180_bars",
            "over_180_bars",
        ),
        default="unavailable",
    )
    staleness_categories = np.select(
        (
            opening.global_staleness_minutes <= 5,
            opening.global_staleness_minutes <= 30,
            np.isfinite(opening.global_staleness_minutes),
        ),
        ("fresh_0_5m", "moderate_6_30m", "stale_over_30m"),
        default="unready",
    )
    specifications = (
        ("b3_observed_bars", bar_categories, opening.b3_observed_bars, "bars"),
        (
            "local_observed_fraction",
            _fraction_categories(opening.local_observed_fraction),
            opening.local_observed_fraction,
            "fraction",
        ),
        (
            "global_observed_fraction",
            _fraction_categories(opening.global_observed_fraction),
            opening.global_observed_fraction,
            "fraction",
        ),
        (
            "global_staleness",
            staleness_categories,
            opening.global_staleness_minutes,
            "minutes",
        ),
        (
            "preopen_context",
            np.where(opening.preopen_context_complete, "complete", "incomplete"),
            opening.preopen_context_complete.astype(np.float64),
            "complete_fraction",
        ),
    )
    frames: list[pl.DataFrame] = []
    for name, categories, values, unit in specifications:
        frame = _group_metrics(inputs, "category", categories)
        summaries = pl.DataFrame(
            {
                "category": np.unique(categories),
                "mean_diagnostic_value": [
                    _finite_mean(values[categories == category])
                    for category in np.unique(categories)
                ],
            }
        )
        frames.append(
            frame.join(summaries, on="category").with_columns(
                pl.lit(name).alias("diagnostic"), pl.lit(unit).alias("unit")
            )
        )
    return pl.concat(frames, how="diagonal_relaxed")


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
    date_count = values.shape[0]
    if values.ndim != 2 or date_count < block_length or replications <= 0:
        raise ValueError("Moving-block bootstrap dimensions are invalid")
    block_count = math.ceil(date_count / block_length)
    generator = np.random.default_rng(seed)
    starts = generator.integers(
        0,
        date_count - block_length + 1,
        size=(replications, block_count),
    )
    indices = (starts[..., None] + np.arange(block_length, dtype=np.int64)).reshape(
        replications, -1
    )[:, :date_count]
    counts = np.zeros((replications, date_count), dtype=np.int16)
    rows = np.repeat(np.arange(replications), date_count)
    np.add.at(counts, (rows, indices.ravel()), 1)
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
    upper = np.full(values.shape[1], np.nan, dtype=np.float64)
    available = np.isfinite(replicated).any(axis=0)
    if available.any():
        lower[available] = np.nanquantile(replicated[:, available], 0.025, axis=0)
        upper[available] = np.nanquantile(replicated[:, available], 0.975, axis=0)
    return {"estimate": estimate, "lower_95": lower, "upper_95": upper}


def bootstrap_summary(inputs: AttributionInputs) -> pl.DataFrame:
    decomposition = rank_decomposition(
        inputs.predictions, inputs.targets, inputs.label_mask
    )
    dates = np.unique(inputs.date_idx)
    statistics: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []

    def add(scope: str, group: str, selected: np.ndarray) -> None:
        for horizon, minutes in enumerate(HORIZONS):
            statistics.append(
                _daily_mean(
                    decomposition.sample_ic[:, horizon], inputs.date_idx, selected
                )
            )
            metadata.append(
                {"scope": scope, "group": group, "horizon_minutes": minutes}
            )

    for decision in range(EXPECTED_DECISIONS_PER_DATE):
        add("time_of_day_5m", str(decision), inputs.decision_idx == decision)
    for index, decisions in enumerate(primary_time_bins()):
        add(
            "time_of_day_30m",
            f"bin_{index + 1:02d}",
            np.isin(inputs.decision_idx, decisions),
        )
    add("horizon_attribution", "all", np.ones(inputs.date_idx.size, dtype=bool))
    matrix = np.stack(statistics, axis=1)
    if matrix.shape[0] != dates.size:
        raise RuntimeError("Bootstrap date matrix is misaligned")
    result = moving_block_bootstrap_matrix(matrix)
    return pl.DataFrame(
        [
            {
                "run": inputs.run_name,
                **row,
                "point_estimate": float(result["estimate"][index]),
                "lower_95": float(result["lower_95"][index]),
                "upper_95": float(result["upper_95"][index]),
                "block_trading_days": BOOTSTRAP_BLOCK_TRADING_DAYS,
                "replications": BOOTSTRAP_REPLICATIONS,
                "seed": BOOTSTRAP_SEED,
            }
            for index, row in enumerate(metadata)
        ]
    )


def causal_observation_completeness(
    observed: np.ndarray,
    date_idx: np.ndarray,
    cutoffs: np.ndarray,
    *,
    readiness: np.ndarray | None = None,
    preopen_cutoff: int | None = None,
    current_window_length: int | None = None,
) -> dict[str, np.ndarray]:
    observed = np.asarray(observed, dtype=bool)
    date_idx = np.asarray(date_idx, dtype=np.int64)
    cutoffs = np.asarray(cutoffs, dtype=np.int64)
    if (
        observed.ndim != 3
        or date_idx.shape != cutoffs.shape
        or (current_window_length is not None and current_window_length <= 0)
    ):
        raise ValueError("Completeness arrays are misaligned")
    sample_count, instrument_count = date_idx.size, observed.shape[1]
    counts = np.zeros((sample_count, instrument_count), dtype=np.int32)
    fractions = np.zeros((sample_count, instrument_count), dtype=np.float64)
    preopen = np.full_like(fractions, np.nan)
    staleness = np.full_like(fractions, np.nan)
    ready = np.ones((sample_count, instrument_count), dtype=bool)
    for sample, (day, cutoff) in enumerate(zip(date_idx, cutoffs, strict=True)):
        if (
            day < 0
            or day >= observed.shape[0]
            or cutoff <= 0
            or cutoff > observed.shape[2]
        ):
            raise ValueError("Completeness cutoff is outside the causal grid")
        start = (
            max(0, cutoff - current_window_length)
            if current_window_length is not None
            else 0
        )
        current = observed[day, :, start:cutoff]
        counts[sample] = current.sum(axis=1)
        fractions[sample] = counts[sample] / current.shape[1]
        last = np.where(current, np.arange(current.shape[1]), -1).max(axis=1)
        present = last >= 0
        staleness[sample, present] = current.shape[1] - 1 - last[present]
        if preopen_cutoff is not None:
            width = min(cutoff, preopen_cutoff)
            preopen[sample] = observed[day, :, :width].mean(axis=1)
        if readiness is not None:
            ready[sample] = readiness[day]
    return {
        "observed_bars": counts,
        "observed_fraction": fractions,
        "preopen_observed_fraction": preopen,
        "minutes_since_most_recent_observed_bar": staleness,
        "ready": ready,
    }


def _market_gap_by_date(store: Path, date_indices: np.ndarray) -> np.ndarray:
    membership = np.load(
        store / "equity_membership.npy", mmap_mode="r", allow_pickle=False
    )
    ready = np.load(store / "equity_data_ready.npy", mmap_mode="r", allow_pickle=False)
    features = np.load(store / "equity_features.npy", mmap_mode="r", allow_pickle=False)
    slow = np.load(store / "equity_slow.npy", mmap_mode="r", allow_pickle=False)
    output = np.full(date_indices.size, np.nan, dtype=np.float64)
    for position, date_index in enumerate(date_indices):
        active = membership[date_index] & ready[date_index]
        early = np.asarray(
            features[date_index, :, : DECISION_EQUITY_INDICES[0], OBSERVED_CHANNEL],
            dtype=bool,
        ).any(axis=1)
        gaps = np.asarray(slow[date_index, :, OVERNIGHT_CHANNEL], dtype=np.float64)
        valid = active & early & np.isfinite(gaps)
        if valid.any():
            output[position] = float(np.median(gaps[valid]))
    return output


def _opening_diagnostics(
    store: Path, validation_dates: np.ndarray, decision_idx: np.ndarray
) -> OpeningDiagnostics:
    sample_index = pl.read_parquet(store / "sample_index.parquet")
    allowed_validation_dates = (
        select_sample_split(sample_index, "validation")
        .get_column("date_idx")
        .unique()
        .to_numpy()
    )
    if not np.isin(validation_dates, allowed_validation_dates).all():
        raise ValueError("Attribution inputs contain a non-validation date")
    training_dates = (
        sample_index.filter(pl.col("trade_date").is_between(TRAIN_START, TRAIN_END))
        .get_column("date_idx")
        .unique()
        .sort()
        .to_numpy()
    )
    thresholds = learn_overnight_thresholds(_market_gap_by_date(store, training_dates))
    unique_dates, inverse = np.unique(validation_dates, return_inverse=True)
    gaps = _market_gap_by_date(store, unique_dates)[inverse]
    membership = np.load(
        store / "equity_membership.npy", mmap_mode="r", allow_pickle=False
    )
    equity_ready = np.load(
        store / "equity_data_ready.npy", mmap_mode="r", allow_pickle=False
    )
    equity_features = np.load(
        store / "equity_features.npy", mmap_mode="r", allow_pickle=False
    )
    active = np.asarray(
        membership[unique_dates] & equity_ready[unique_dates], dtype=bool
    )[inverse]
    equity_observed = np.asarray(
        equity_features[unique_dates, :, :, OBSERVED_CHANNEL], dtype=bool
    )
    equity_cumulative = np.cumsum(equity_observed, axis=2, dtype=np.int16)
    equity_cutoffs = np.asarray(DECISION_EQUITY_INDICES)[decision_idx]
    equity_counts = equity_cumulative[inverse, :, equity_cutoffs - 1]
    b3_observed_bars = np.full(validation_dates.size, np.nan, dtype=np.float64)
    for sample in range(validation_dates.size):
        if active[sample].any():
            b3_observed_bars[sample] = float(
                np.median(equity_counts[sample, active[sample]])
            )

    context_features = np.load(
        store / "context_features.npy", mmap_mode="r", allow_pickle=False
    )
    context_ready = np.asarray(
        np.load(store / "context_data_ready.npy", mmap_mode="r", allow_pickle=False)[
            unique_dates
        ],
        dtype=bool,
    )
    local = causal_observation_completeness(
        np.asarray(context_features[unique_dates, :, :, OBSERVED_CHANNEL], dtype=bool),
        inverse,
        np.asarray(DECISION_CONTEXT_INDICES)[decision_idx],
        readiness=context_ready,
        preopen_cutoff=60,
    )
    local_fractions = local["observed_fraction"][:, RETAINED_LOCAL_SLOTS]
    local_observed_fraction = np.nanmedian(local_fractions, axis=1)
    local_preopen_complete = np.all(
        local["ready"][:, RETAINED_LOCAL_SLOTS]
        & (local["preopen_observed_fraction"][:, RETAINED_LOCAL_SLOTS] >= 0.95),
        axis=1,
    )

    global_features = np.load(
        store / "global_features.npy", mmap_mode="r", allow_pickle=False
    )
    global_ready_store = np.load(
        store / "global_data_ready.npy", mmap_mode="r", allow_pickle=False
    )
    global_values = causal_observation_completeness(
        np.asarray(global_features[unique_dates, :, :, OBSERVED_CHANNEL], dtype=bool),
        inverse,
        np.asarray(DECISION_GLOBAL_INDICES)[decision_idx],
        preopen_cutoff=330,
        current_window_length=GLOBAL_WINDOW_MINUTES,
    )
    global_values["ready"] = np.asarray(
        global_ready_store[unique_dates][inverse, :, decision_idx], dtype=bool
    )
    retained = np.asarray(CANONICAL_RETAINED_GLOBAL_SLOTS)
    global_fractions = global_values["observed_fraction"][:, retained]
    global_observed_fraction = np.nanmedian(global_fractions, axis=1)
    retained_staleness = global_values["minutes_since_most_recent_observed_bar"][
        :, retained
    ]
    global_staleness = np.full(validation_dates.size, np.nan, dtype=np.float64)
    for sample in range(validation_dates.size):
        finite = retained_staleness[sample, np.isfinite(retained_staleness[sample])]
        if finite.size:
            global_staleness[sample] = float(np.median(finite))
    global_preopen_complete = np.all(
        global_values["ready"][:, retained]
        & (global_values["preopen_observed_fraction"][:, retained] >= 0.95),
        axis=1,
    )
    return OpeningDiagnostics(
        gaps,
        thresholds,
        b3_observed_bars,
        local_observed_fraction,
        global_observed_fraction,
        global_staleness,
        local_preopen_complete & global_preopen_complete,
    )


def load_attribution_inputs(
    run_dir: Path, cache_dir: Path | None = None
) -> AttributionInputs:
    cached = (
        _load_cached(_cache_path(cache_dir, run_dir)) if cache_dir is not None else None
    )
    if cached is None:
        observations, _, _, _, store = collect_neural_evaluation(run_dir, "validation")
        values = {
            "predictions": observations.predictions,
            "targets": observations.targets,
            "raw_returns": observations.raw_returns,
            "label_mask": observations.label_mask,
            "date_idx": observations.date_idx,
            "decision_idx": observations.decision_idx,
        }
        if cache_dir is not None:
            _write_cache(_cache_path(cache_dir, run_dir), values)
    else:
        values = cached
        checkpoint = torch.load(
            run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False
        )
        store = Path(str(checkpoint["feature_store"]))
    dates = values["date_idx"].astype(np.int64)
    decisions = values["decision_idx"].astype(np.int64)
    return AttributionInputs(
        run_dir.name,
        values["predictions"],
        values["targets"],
        values["raw_returns"],
        values["label_mask"].astype(bool),
        dates,
        decisions,
        _security_ids(store),
        _opening_diagnostics(store, dates, decisions),
    )


def analyze_runs(
    run_dirs: list[Path], output_dir: Path, cache_dir: Path | None = None
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    inputs = [
        load_attribution_inputs(run_dir.resolve(), cache_dir) for run_dir in run_dirs
    ]
    outputs = {
        "stock_attribution": pl.concat([stock_attribution(value) for value in inputs]),
        "time_of_day_5m": pl.concat(
            [time_of_day_attribution(value) for value in inputs]
        ),
        "time_of_day_30m": pl.concat(
            [time_of_day_30m_attribution(value) for value in inputs]
        ),
        "horizon_attribution": pl.concat(
            [horizon_attribution(value) for value in inputs]
        ),
        "opening_regimes": pl.concat([opening_attribution(value) for value in inputs]),
        "opening_context": pl.concat(
            [opening_context_attribution(value) for value in inputs]
        ),
        "bootstrap_summary": pl.concat([bootstrap_summary(value) for value in inputs]),
    }
    for name, frame in outputs.items():
        frame.write_parquet(output_dir / f"{name}.parquet")
        frame.write_csv(output_dir / f"{name}.csv")
    _atomic_json(
        output_dir / "summary.json",
        {
            "split": "validation",
            "test_accessed": False,
            "runs": [str(path.resolve()) for path in run_dirs],
            "outputs": {name: frame.height for name, frame in outputs.items()},
            "bootstrap": {
                "block_trading_days": BOOTSTRAP_BLOCK_TRADING_DAYS,
                "replications": BOOTSTRAP_REPLICATIONS,
                "seed": BOOTSTRAP_SEED,
            },
            "opening_threshold_source": "training",
        },
    )
    return output_dir


def main() -> None:
    args = parse_args()
    print(analyze_runs(args.run_dir, args.output_dir, args.cache_dir))


if __name__ == "__main__":
    main()
