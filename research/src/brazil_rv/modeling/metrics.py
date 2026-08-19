from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .contract import HORIZONS, MIN_IC_EQUITIES


def average_ranks(values: NDArray[np.floating]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _rowwise_average_ranks(
    values: NDArray[np.float64], mask: NDArray[np.bool_]
) -> NDArray[np.float64]:
    value_order = np.argsort(values, axis=1, kind="stable")
    sorted_valid = np.take_along_axis(mask, value_order, axis=1)
    valid_order = np.argsort(~sorted_valid, axis=1, kind="stable")
    order = np.take_along_axis(value_order, valid_order, axis=1)
    sorted_values = np.take_along_axis(values, order, axis=1)
    sorted_valid = np.take_along_axis(mask, order, axis=1)

    columns = np.arange(values.shape[1])
    group_starts = sorted_valid.copy()
    group_starts[:, 1:] &= sorted_values[:, 1:] != sorted_values[:, :-1]
    starts = np.maximum.accumulate(np.where(group_starts, columns, -1), axis=1)
    group_ends = sorted_valid.copy()
    group_ends[:, :-1] &= (~sorted_valid[:, 1:]) | (
        sorted_values[:, :-1] != sorted_values[:, 1:]
    )
    ends = np.minimum.accumulate(
        np.where(group_ends, columns, values.shape[1])[:, ::-1], axis=1
    )[:, ::-1]
    sorted_ranks = np.where(sorted_valid, 0.5 * (starts + ends), 0.0)
    ranks = np.empty_like(sorted_ranks)
    np.put_along_axis(ranks, order, sorted_ranks, axis=1)
    return ranks


def _rowwise_correlation(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    counts = mask.sum(axis=1)
    safe_counts = np.maximum(counts, 1)
    left_mean = np.where(mask, left, 0.0).sum(axis=1) / safe_counts
    right_mean = np.where(mask, right, 0.0).sum(axis=1) / safe_counts
    left_centered = np.where(mask, left - left_mean[:, None], 0.0)
    right_centered = np.where(mask, right - right_mean[:, None], 0.0)
    covariance = (left_centered * right_centered).sum(axis=1)
    denominator = np.sqrt(
        (left_centered**2).sum(axis=1) * (right_centered**2).sum(axis=1)
    )
    result = np.full(left.shape[0], np.nan, dtype=np.float64)
    valid = (counts >= MIN_IC_EQUITIES) & (denominator != 0.0)
    result[valid] = covariance[valid] / denominator[valid]
    return result


def _metric_rows(
    values: NDArray[np.float32], label_mask: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    rows = values.transpose(0, 2, 1).reshape(-1, values.shape[1]).astype(np.float64)
    mask = label_mask.transpose(0, 2, 1).reshape(-1, values.shape[1])
    return rows, mask


def rank_average_predictions(
    members: Sequence[NDArray[np.float32]],
    label_mask: NDArray[np.bool_],
) -> NDArray[np.float32]:
    if not members:
        raise ValueError("At least one ensemble member is required")
    if any(member.shape != label_mask.shape for member in members):
        raise ValueError("Ensemble prediction shapes differ from the label mask")
    ranked = []
    for member in members:
        rows, mask = _metric_rows(member, label_mask)
        ranked.append(_rowwise_average_ranks(rows, mask))
    averaged = np.mean(np.stack(ranked), axis=0)
    result = averaged.reshape(
        label_mask.shape[0], label_mask.shape[2], label_mask.shape[1]
    ).transpose(0, 2, 1)
    result[~label_mask] = 0.0
    return result.astype(np.float32)


def sample_level_spearman_ic(
    predictions: NDArray[np.float32],
    targets: NDArray[np.float32],
    label_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    predicted, mask = _metric_rows(predictions, label_mask)
    actual, _ = _metric_rows(targets, label_mask)
    spearman = _rowwise_correlation(
        _rowwise_average_ranks(predicted, mask),
        _rowwise_average_ranks(actual, mask),
        mask,
    )
    return spearman.reshape(predictions.shape[0], predictions.shape[2])


def sample_level_ic(
    predictions: NDArray[np.float32],
    targets: NDArray[np.float32],
    label_mask: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    spearman = sample_level_spearman_ic(predictions, targets, label_mask)
    predicted, mask = _metric_rows(predictions, label_mask)
    actual, _ = _metric_rows(targets, label_mask)
    pearson = _rowwise_correlation(predicted, actual, mask)
    return spearman, pearson.reshape(predictions.shape[0], predictions.shape[2])


def finite_mean(values: NDArray[np.float64]) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _standard_deviation(values: NDArray[np.float64]) -> float:
    finite = values[np.isfinite(values)]
    return float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")


def daily_horizon_ic(
    sample_ic: NDArray[np.float64],
    date_idx: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    dates = np.unique(date_idx)
    values = np.asarray(
        [
            [
                finite_mean(sample_ic[date_idx == date_value, horizon])
                for horizon in range(sample_ic.shape[1])
            ]
            for date_value in dates
        ],
        dtype=np.float64,
    )
    return dates, values


def primary_score_from_sample_ic(
    sample_ic: NDArray[np.float64], date_idx: NDArray[np.int64]
) -> float:
    _, daily = daily_horizon_ic(sample_ic, date_idx)
    return finite_mean(np.nanmean(daily, axis=0))


def per_date_primary_ic(
    sample_ic: NDArray[np.float64], date_idx: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    dates = np.unique(date_idx)
    return dates, np.asarray(
        [finite_mean(sample_ic[date_idx == value].ravel()) for value in dates],
        dtype=np.float64,
    )


def primary_validation_score(
    predictions: NDArray[np.float32],
    targets: NDArray[np.float32],
    label_mask: NDArray[np.bool_],
    date_idx: NDArray[np.int64],
) -> float:
    return primary_score_from_sample_ic(
        sample_level_spearman_ic(predictions, targets, label_mask), date_idx
    )


def ranking_diagnostics(
    predictions: NDArray[np.float32],
    raw_returns: NDArray[np.float32],
    label_mask: NDArray[np.bool_],
    date_idx: NDArray[np.int64],
    decision_idx: NDArray[np.int64],
) -> dict[str, NDArray[np.float64]]:
    sample_count, equity_count, horizon_count = predictions.shape
    diagnostics = {
        name: np.full((sample_count, horizon_count), np.nan, dtype=np.float64)
        for name in (
            "top_return",
            "bottom_return",
            "top_minus_bottom",
            "long_only_top",
            "one_way_turnover",
        )
    }
    previous_weights: dict[tuple[int, int], NDArray[np.float64]] = {}
    order = np.lexsort((decision_idx, date_idx))
    for sample in order:
        for horizon in range(horizon_count):
            valid = np.flatnonzero(label_mask[sample, :, horizon])
            if valid.size < MIN_IC_EQUITIES:
                continue
            k = max(1, valid.size // 10)
            ranked = valid[
                np.argsort(predictions[sample, valid, horizon], kind="mergesort")
            ]
            bottom = ranked[:k]
            top = ranked[-k:]
            top_return = float(np.mean(raw_returns[sample, top, horizon]))
            bottom_return = float(np.mean(raw_returns[sample, bottom, horizon]))
            diagnostics["top_return"][sample, horizon] = top_return
            diagnostics["bottom_return"][sample, horizon] = bottom_return
            diagnostics["top_minus_bottom"][sample, horizon] = (
                top_return - bottom_return
            )
            diagnostics["long_only_top"][sample, horizon] = top_return
            weights = np.zeros(equity_count, dtype=np.float64)
            weights[top] = 1.0 / k
            weights[bottom] = -1.0 / k
            key = (int(date_idx[sample]), horizon)
            previous = previous_weights.get(key)
            if previous is not None:
                diagnostics["one_way_turnover"][sample, horizon] = 0.5 * float(
                    np.abs(weights - previous).sum()
                )
            previous_weights[key] = weights
    return diagnostics


def create_metric_table(
    predictions: NDArray[np.float32],
    targets: NDArray[np.float32],
    raw_returns: NDArray[np.float32],
    label_mask: NDArray[np.bool_],
    date_idx: NDArray[np.int64],
    decision_idx: NDArray[np.int64],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    spearman, pearson = sample_level_ic(predictions, targets, label_mask)
    diagnostics = ranking_diagnostics(
        predictions, raw_returns, label_mask, date_idx, decision_idx
    )
    daily_rows: list[dict[str, object]] = []
    unique_dates = np.unique(date_idx)
    for date_value in unique_dates:
        on_date = date_idx == date_value
        for horizon_index, horizon_minutes in enumerate(HORIZONS):
            daily_rows.append(
                {
                    "date_idx": int(date_value),
                    "horizon_minutes": horizon_minutes,
                    "spearman_ic": finite_mean(spearman[on_date, horizon_index]),
                    "rank_target_pearson_ic": finite_mean(
                        pearson[on_date, horizon_index]
                    ),
                    **{
                        name: finite_mean(values[on_date, horizon_index])
                        for name, values in diagnostics.items()
                    },
                }
            )

    horizons: list[dict[str, float | int]] = []
    for horizon_index, horizon_minutes in enumerate(HORIZONS):
        horizon_rows = [
            row for row in daily_rows if row["horizon_minutes"] == horizon_minutes
        ]
        daily_spearman = np.asarray(
            [row["spearman_ic"] for row in horizon_rows], dtype=np.float64
        )
        daily_pearson = np.asarray(
            [row["rank_target_pearson_ic"] for row in horizon_rows], dtype=np.float64
        )
        mean_spearman = finite_mean(daily_spearman)
        standard_deviation = _standard_deviation(daily_spearman)
        horizons.append(
            {
                "horizon_minutes": horizon_minutes,
                "mean_daily_spearman_ic": mean_spearman,
                "daily_spearman_standard_deviation": standard_deviation,
                "annualized_spearman_icir": (
                    mean_spearman / standard_deviation * math.sqrt(252.0)
                    if np.isfinite(standard_deviation) and standard_deviation > 0.0
                    else float("nan")
                ),
                "mean_daily_rank_target_pearson_ic": finite_mean(daily_pearson),
                **{
                    f"mean_{name}": finite_mean(
                        np.asarray(
                            [row[name] for row in horizon_rows], dtype=np.float64
                        )
                    )
                    for name in diagnostics
                },
            }
        )
    return (
        {
            "primary_score": primary_score_from_sample_ic(spearman, date_idx),
            "mean_valid_sample_spearman_ic": finite_mean(spearman.ravel()),
            "horizons": horizons,
        },
        daily_rows,
    )


def moving_block_bootstrap(
    daily_values: NDArray[np.floating],
    *,
    replications: int = 10_000,
    block_length: int = 5,
    seed: int = 20260815,
) -> dict[str, NDArray[np.float64]]:
    values = np.asarray(daily_values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] < block_length or replications <= 0:
        raise ValueError("Moving-block bootstrap dimensions are invalid")
    date_count = values.shape[0]
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
    replicated = np.nanmean(values[indices], axis=1)
    return {
        "estimate": np.nanmean(values, axis=0),
        "lower_95": np.nanquantile(replicated, 0.025, axis=0),
        "upper_95": np.nanquantile(replicated, 0.975, axis=0),
    }
