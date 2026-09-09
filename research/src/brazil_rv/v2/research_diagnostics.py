"""Descriptive sealed-panel diagnostics; none contributes to model selection."""

from __future__ import annotations

import numpy as np

from .contract import HORIZONS, PRIMARY_HORIZONS, TRADED_PRIMARY_HORIZONS
from .evaluate import (
    _economics_signal,
    _primary_population_components,
    _primary_daily_metrics,
    _spearman,
)
from .normalization import average_ranks


def _finite(value):
    return float(value) if np.isfinite(value) else None


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return {
        "mean": _finite(finite.mean()) if len(finite) else None,
        "sd": _finite(finite.std(ddof=1)) if len(finite) > 1 else None,
        "defined_sessions": len(finite),
        "possible_sessions": len(values),
    }


def pooled_momentum_diagnostics(folds: dict) -> dict:
    """Pool daily observations, rather than averaging unequal fold summaries."""
    daily = [row for result in folds.values() for row in result["daily"]]
    keys = tuple(key for key in daily[0] if key != "date")
    values = {
        key: np.asarray([np.nan if row[key] is None else row[key] for row in daily])
        for key in keys
    }
    contributions = [
        values[f"spread_contribution_{group}_bps_per_session"]
        for group in ("extreme_momentum", "middle_momentum", "unknown_momentum")
    ]
    denominator = sum(np.nansum(value) for value in contributions)
    return {
        "summary": {key: _summary(value) for key, value in values.items()},
        "extreme_momentum_spread_share": _finite(
            np.nansum(contributions[0]) / denominator
        )
        if abs(denominator) > 1e-12
        else None,
        "selection_weight": 0,
        "folds": {
            fold: {
                "summary": result["summary"],
                "extreme_momentum_spread_share": result[
                    "extreme_momentum_spread_share"
                ],
            }
            for fold, result in folds.items()
        },
    }


def rank_residual_ic(scores, momentum, targets, population):
    """Per-date OLS of score ranks on intercept/momentum ranks, then residual IC."""
    selected = (
        population & np.isfinite(scores) & np.isfinite(momentum) & np.isfinite(targets)
    )
    if selected.sum() < 20:
        return np.nan
    left = average_ranks(scores[selected])
    right = average_ranks(momentum[selected])
    left -= left.mean()
    right -= right.mean()
    denominator = np.dot(right, right)
    if denominator == 0:
        return np.nan
    residual = left - right * (np.dot(right, left) / denominator)
    if np.ptp(residual) <= 1e-12:
        return np.nan  # Pure momentum has no residual rank variation.
    return _spearman(residual, targets[selected], np.ones(len(residual), dtype=bool))


def momentum_diagnostics(inputs, momentum_scores, momentum_mask):
    momentum = np.asarray(momentum_scores, dtype=np.float64)[..., 0]
    supported = np.asarray(momentum_mask, dtype=bool).all(axis=-1) & np.isfinite(
        momentum
    )
    composite, composite_valid = _economics_signal(
        inputs, horizons=TRADED_PRIMARY_HORIZONS
    )
    series = {
        "composite_rank_correlation_with_momentum": np.full(len(inputs.dates), np.nan)
    }
    for label, heads in (
        ("primary_ic", TRADED_PRIMARY_HORIZONS),
        ("legacy_primary_ic_1235", PRIMARY_HORIZONS),
    ):
        scores, targets, outcomes, support = _primary_population_components(
            inputs, heads
        )
        _, series[label], _ = _primary_daily_metrics(
            scores, targets, outcomes, support, inputs.dates, heads
        )
        population = outcomes & support & supported
        raw = np.full((len(inputs.dates), len(heads)), np.nan)
        residual = np.full_like(raw, np.nan)
        for day, valid in enumerate(population):
            for column in range(len(heads)):
                raw[day, column] = _spearman(
                    scores[day, :, column], targets[day, :, column], valid
                )
                residual[day, column] = rank_residual_ic(
                    scores[day, :, column],
                    momentum[day],
                    targets[day, :, column],
                    valid,
                )
        series[label + "_on_momentum_supported_population"] = raw.mean(axis=1)
        series[label + "_residual"] = residual.mean(axis=1)
    legacy = _primary_population_components(inputs)
    legacy_population = legacy[2] & legacy[3]
    for head, horizon in enumerate(HORIZONS):
        forecast_pop = inputs.active & inputs.score_mask[..., head] & supported
        unconditioned_pop = (
            legacy_population
            if horizon in PRIMARY_HORIZONS
            else inputs.active
            & inputs.neutral_target_mask[..., head]
            & inputs.score_mask[..., head]
        )
        target_pop = unconditioned_pop & supported
        correlation, residual_ic, raw_ic, unconditioned_ic = [], [], [], []
        for day in range(len(inputs.dates)):
            unconditioned_ic.append(
                _spearman(
                    inputs.scores[day, :, head],
                    inputs.neutral_midrank_targets[day, :, head],
                    unconditioned_pop[day],
                )
            )
            correlation.append(
                _spearman(inputs.scores[day, :, head], momentum[day], forecast_pop[day])
            )
            raw_ic.append(
                _spearman(
                    inputs.scores[day, :, head],
                    inputs.neutral_midrank_targets[day, :, head],
                    target_pop[day],
                )
            )
            residual_ic.append(
                rank_residual_ic(
                    inputs.scores[day, :, head],
                    momentum[day],
                    inputs.neutral_midrank_targets[day, :, head],
                    target_pop[day],
                )
            )
        series[f"D{horizon}_rank_correlation_with_momentum"] = np.asarray(correlation)
        series[f"D{horizon}_neutral_ic_momentum_supported"] = np.asarray(raw_ic)
        series[f"D{horizon}_neutral_ic"] = np.asarray(unconditioned_ic)
        series[f"D{horizon}_residual_ic"] = np.asarray(residual_ic)
    for day in range(len(inputs.dates)):
        series["composite_rank_correlation_with_momentum"][day] = _spearman(
            composite[day], momentum[day], composite_valid[day] & supported[day]
        )
    # Attribute the unchanged D1/D2/D3/D5 decile-spread readout. We keep the
    # original tails and denominators; missing momentum remains unattributed.
    indexes = [HORIZONS.index(h) for h in PRIMARY_HORIZONS]
    scores = np.asarray(inputs.scores)[..., indexes]
    returns = np.asarray(inputs.shareholder_simple_returns)[..., indexes]
    valid = (
        inputs.active
        & inputs.shareholder_target_mask[..., indexes].all(axis=-1)
        & np.isfinite(returns).all(axis=-1)
        & inputs.score_mask[..., indexes].all(axis=-1)
        & np.isfinite(scores).all(axis=-1)
    )
    contributions = np.full((len(inputs.dates), 3), np.nan)
    for day in range(len(inputs.dates)):
        names = np.flatnonzero(valid[day])
        if len(names) < 20:
            continue
        known_names = np.flatnonzero(inputs.active[day] & supported[day])
        extreme = np.zeros(inputs.active.shape[1], dtype=bool)
        known = supported[day].copy()
        if len(known_names) >= 20:
            fractions = (average_ranks(momentum[day, known_names]) + 0.5) / len(
                known_names
            )
            extreme[known_names] = (fractions <= 0.2) | (fractions >= 0.8)
        else:
            known[:] = False  # Insufficient support cannot define quintiles.
        groups = (extreme & known, ~extreme & known, ~known)
        count = max(1, len(names) // 10)
        values = np.zeros(3)
        for column, horizon in enumerate(PRIMARY_HORIZONS):
            ordered = names[np.argsort(scores[day, names, column], kind="stable")]
            for group, membership in enumerate(groups):
                top, bottom = ordered[-count:], ordered[:count]
                values[group] += (
                    (
                        np.where(membership[top], returns[day, top, column], 0).sum()
                        - np.where(
                            membership[bottom], returns[day, bottom, column], 0
                        ).sum()
                    )
                    / count
                    * 10_000
                    / horizon
                    / len(PRIMARY_HORIZONS)
                )
        contributions[day] = values
    for column, label in enumerate(
        ("extreme_momentum", "middle_momentum", "unknown_momentum")
    ):
        series[f"spread_contribution_{label}_bps_per_session"] = contributions[
            :, column
        ]
    denominator = np.nansum(contributions)
    return {
        "selection_weight": 0,
        "regression": "per_date_score_rank_on_intercept_and_momentum_rank",
        "spread_definition": "unchanged_1235_decile_spread_not_constructed_book_pnl",
        "extreme_momentum_spread_share": _finite(
            np.nansum(contributions[:, 0]) / denominator
        )
        if abs(denominator) > 1e-12
        else None,
        "spread_share_note": "ratio_of_pooled_contributions_can_exceed_one_or_be_negative",
        "summary": {key: _summary(values) for key, values in series.items()},
        "daily": [
            {
                "date": day.isoformat(),
                **{key: _finite(values[i]) for key, values in series.items()},
            }
            for i, day in enumerate(inputs.dates)
        ],
    }
