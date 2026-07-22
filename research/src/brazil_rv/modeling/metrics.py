from __future__ import annotations

import math

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


def _correlation(left: NDArray[np.floating], right: NDArray[np.floating]) -> float:
    left_centered = left.astype(np.float64) - float(np.mean(left))
    right_centered = right.astype(np.float64) - float(np.mean(right))
    denominator = math.sqrt(float(np.sum(left_centered**2) * np.sum(right_centered**2)))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(left_centered * right_centered) / denominator)


def sample_level_ic(
    predictions: NDArray[np.float32],
    targets: NDArray[np.float32],
    label_mask: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    sample_count, _, horizon_count = predictions.shape
    spearman = np.full((sample_count, horizon_count), np.nan, dtype=np.float64)
    pearson = np.full_like(spearman, np.nan)
    for sample in range(sample_count):
        for horizon in range(horizon_count):
            valid = label_mask[sample, :, horizon]
            if int(valid.sum()) < MIN_IC_EQUITIES:
                continue
            predicted = predictions[sample, valid, horizon]
            actual = targets[sample, valid, horizon]
            pearson[sample, horizon] = _correlation(predicted, actual)
            spearman[sample, horizon] = _correlation(
                average_ranks(predicted), average_ranks(actual)
            )
    return spearman, pearson


def _mean(values: NDArray[np.float64]) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _standard_deviation(values: NDArray[np.float64]) -> float:
    finite = values[np.isfinite(values)]
    return float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")


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
                    "spearman_ic": _mean(spearman[on_date, horizon_index]),
                    "pearson_ic": _mean(pearson[on_date, horizon_index]),
                    "top_return": _mean(
                        diagnostics["top_return"][on_date, horizon_index]
                    ),
                    "bottom_return": _mean(
                        diagnostics["bottom_return"][on_date, horizon_index]
                    ),
                    "top_minus_bottom": _mean(
                        diagnostics["top_minus_bottom"][on_date, horizon_index]
                    ),
                    "long_only_top": _mean(
                        diagnostics["long_only_top"][on_date, horizon_index]
                    ),
                    "one_way_turnover": _mean(
                        diagnostics["one_way_turnover"][on_date, horizon_index]
                    ),
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
            [row["pearson_ic"] for row in horizon_rows], dtype=np.float64
        )
        mean_spearman = _mean(daily_spearman)
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
                "mean_daily_pearson_ic": _mean(daily_pearson),
                "mean_top_return": _mean(
                    np.asarray(
                        [row["top_return"] for row in horizon_rows],
                        dtype=np.float64,
                    )
                ),
                "mean_bottom_return": _mean(
                    np.asarray(
                        [row["bottom_return"] for row in horizon_rows],
                        dtype=np.float64,
                    )
                ),
                "mean_top_minus_bottom": _mean(
                    np.asarray(
                        [row["top_minus_bottom"] for row in horizon_rows],
                        dtype=np.float64,
                    )
                ),
                "mean_long_only_top": _mean(
                    np.asarray(
                        [row["long_only_top"] for row in horizon_rows],
                        dtype=np.float64,
                    )
                ),
                "mean_one_way_turnover": _mean(
                    np.asarray(
                        [row["one_way_turnover"] for row in horizon_rows],
                        dtype=np.float64,
                    )
                ),
            }
        )
    primary_score = float(np.mean([row["mean_daily_spearman_ic"] for row in horizons]))
    return (
        {
            "primary_score": primary_score,
            "mean_valid_sample_spearman_ic": _mean(spearman.ravel()),
            "horizons": horizons,
        },
        daily_rows,
    )
