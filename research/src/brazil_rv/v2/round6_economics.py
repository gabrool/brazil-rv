"""Registered hindsight borrow-cost sensitivity; never a model input."""

from dataclasses import replace
from datetime import date

import numpy as np
import polars as pl

from .lending_archive import LendingBorrowPanels


def placeholder_v2(
    panels: LendingBorrowPanels,
    observed_rates: pl.DataFrame,
    dates: np.ndarray,
    isins: tuple[str, ...],
    prior_adv20: np.ndarray,
) -> tuple[LendingBorrowPanels, dict]:
    """Only replace explicit flat-rate placeholders; all other cells are exact.

    Calibration uses 2023-2024 name medians and fixed liquidity boundaries.
    Earlier replay years therefore have hindsight; this scenario must never
    govern promotion or supply training inputs.
    """
    rates = (
        observed_rates.filter(
            pl.col("source_trade_date").is_between(date(2023, 1, 1), date(2024, 12, 30))
            & pl.col("annual_taker_rate").is_finite()
            & (pl.col("annual_taker_rate") >= 0)
        )
        .select("security_id", "source_trade_date", "annual_taker_rate")
        .unique()
    )
    if (
        rates.group_by("security_id", "source_trade_date")
        .len()
        .filter(pl.col("len") > 1)
        .height
    ):
        raise ValueError(
            "placeholder calibration has conflicting same-name/session observations"
        )
    names = rates.group_by("security_id").agg(
        pl.col("annual_taker_rate").median().alias("median"), pl.len().alias("n")
    )
    lookup = {
        r["security_id"].removeprefix("ISIN:"): r for r in names.iter_rows(named=True)
    }
    median = np.array([lookup.get(i, {}).get("median", np.nan) for i in isins])
    counts = np.array([lookup.get(i, {}).get("n", 0) for i in isins])
    if not np.isfinite(median).any():
        raise ValueError("placeholder sensitivity has no observed calibration rates")
    calibration = (dates >= np.datetime64("2023-01-01")) & (
        dates <= np.datetime64("2024-12-30")
    )
    # Missing liquidity is reported separately rather than converted to zero.
    liquidity = np.full(len(isins), np.nan)
    for j in range(len(isins)):
        x = prior_adv20[calibration, j]
        x = x[np.isfinite(x) & (x > 0)]
        if len(x):
            liquidity[j] = np.median(x)
    observed_liquidity = liquidity[np.isfinite(liquidity)]
    if not len(observed_liquidity):
        raise ValueError("placeholder sensitivity has no calibration liquidity")
    boundaries = np.quantile(observed_liquidity, [0.2, 0.4, 0.6, 0.8])
    groups = np.searchsorted(boundaries, liquidity, side="right")
    pooled = float(np.nanmedian(median))
    group_medians = []
    for q in range(5):
        x = median[(groups == q) & np.isfinite(liquidity) & np.isfinite(median)]
        group_medians.append(float(np.median(x)) if len(x) else pooled)
    current_groups = np.searchsorted(boundaries, prior_adv20, side="right")
    missing = ~np.isfinite(prior_adv20) | (prior_adv20 <= 0)
    centers = np.asarray(group_medians)[current_groups]
    centers[missing] = pooled
    weights = counts / (counts + 20.0)
    estimate = (
        weights[None, :] * np.nan_to_num(median)[None, :]
        + (1 - weights[None, :]) * centers
    )
    changed = panels.annual_taker_rate.copy()
    changed[panels.rate_placeholder] = estimate[panels.rate_placeholder]
    return replace(panels, annual_taker_rate=changed), {
        "label": "placeholder_v2_hindsight_cost_sensitivity",
        "promotion_weight": 0,
        "calibration_years": [2023, 2024],
        "shrinkage_pseudo_sessions": 20,
        "liquidity_boundaries_brl": boundaries.tolist(),
        "quintile_name_median_rates": group_medians,
        "pooled_name_median_rate": pooled,
        "observed_names": int(np.isfinite(median).sum()),
        "replaced_cells": int(panels.rate_placeholder.sum()),
        "missing_liquidity_placeholder_cells": int(
            (missing & panels.rate_placeholder).sum()
        ),
        "names": [
            {
                "isin": name,
                "observed_sessions": int(counts[j]),
                "median_rate": float(median[j]) if counts[j] else None,
                "calibration_liquidity": float(liquidity[j])
                if np.isfinite(liquidity[j])
                else None,
            }
            for j, name in enumerate(isins)
        ],
    }
