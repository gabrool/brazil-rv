"""Registered field-sign readouts; descriptive only, never feature selection."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from .artifacts import sha256_file, write_json_atomic
from .contract import DEVELOPMENT_END, FINETUNE_START, HORIZONS, PRETRAIN_END
from .research_rounds import (
    REGISTERED_PRIMARY_TARGET,
    REGISTERED_PRIMARY_TARGET_MASK,
    _git_identity,
)
from .store import open_store_for_dates, peak_rss_bytes

EXPECTED_SIGNS = {
    "log_market_cap": "negative_to_flat",
    "book_to_market": "positive_value_prior",
    "earnings_yield_ttm": "positive",
    "gross_profitability": "positive",
    "liabilities_to_assets": "negative",
    "accruals_to_assets": "negative",
    "revenue_growth_yoy": "positive_weak_prior",
    "sue": "positive",
    "statement_age_sessions": "small_no_fixed_sign",
    "fundamental_financial_flag": "structural_no_fixed_sign",
    "fundamental_consolidated_flag": "structural_no_fixed_sign",
    "valuation_available_flag": "coverage_no_fixed_sign",
    "sessions_since_financial_filing": "small_no_fixed_sign",
    "filing_is_dfp": "event_type_no_fixed_sign",
    "sessions_since_material_fact": "small_no_fixed_sign",
    "material_fact_count_20": "event_intensity_no_fixed_sign",
    "dividend_announcement_age": "small_no_fixed_sign",
    "offering_or_buyback_flag": "mixed_event_types_no_fixed_sign",
    "sessions_until_expected_filing": "small_no_fixed_sign",
}


def field_readout(values, targets, valid, dates, *, minimum_names=20):
    """Rank within the identical valid cross-section before averaging days."""
    support = valid & np.isfinite(values) & np.isfinite(targets)
    counts = support.sum(axis=1)
    x = rankdata(np.where(support, values, np.nan), axis=1, nan_policy="omit")
    y = rankdata(np.where(support, targets, np.nan), axis=1, nan_policy="omit")
    # Each supported rank vector has exact mean (n+1)/2, including tied ranks.
    center = (counts + 1)[:, None] / 2
    x = np.where(support, x - center, 0)
    y = np.where(support, y - center, 0)
    denominator = np.sqrt(np.sum(x * x, axis=1) * np.sum(y * y, axis=1))
    daily = np.full(len(dates), np.nan)
    defined = (counts >= minimum_names) & (denominator > 0)
    daily[defined] = np.sum(x * y, axis=1)[defined] / denominator[defined]

    def summarize(rows):
        mask = support[rows]
        finite = daily[rows][np.isfinite(daily[rows])]
        stacked = None
        if mask.sum() >= minimum_names:
            a, b = rankdata(values[rows][mask]), rankdata(targets[rows][mask])
            a, b = a - a.mean(), b - b.mean()
            scale = np.sqrt(np.dot(a, a) * np.dot(b, b))
            if scale > 0:
                stacked = float(np.dot(a, b) / scale)
        return {
            "mean_daily_cross_sectional_ic": float(finite.mean())
            if finite.size
            else None,
            "median_daily_cross_sectional_ic": float(np.median(finite))
            if finite.size
            else None,
            "pooled_name_day_spearman_secondary": stacked,
            "supported_name_days": int(mask.sum()),
            "supported_dates": int((counts[rows] > 0).sum()),
            "defined_daily_ics": int(finite.size),
            "constant_rank_dates_with_minimum_support": int(
                ((counts[rows] >= minimum_names) & (denominator[rows] == 0)).sum()
            ),
            "positive_daily_ics": int((finite > 0).sum()),
        }

    years = dates.astype("datetime64[Y]").astype(int) + 1970
    return {
        "pooled": summarize(np.ones(len(dates), bool)),
        "by_year": {str(year): summarize(years == year) for year in np.unique(years)},
        "daily_ic": [float(value) if np.isfinite(value) else None for value in daily],
        "daily_supported_names": counts.tolist(),
    }


def run(registration_path: Path, output: Path):
    started = time.monotonic()
    code = _git_identity()
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    root = Path(registration["base_store"]["root"])
    if (
        sha256_file(root / "manifest.json")
        != registration["base_store"]["manifest_sha256"]
    ):
        raise ValueError("diagnostic store differs from Round-6 registration")
    dates = np.load(root / "date_index.npy", allow_pickle=False)
    if dates[-1] > np.datetime64(DEVELOPMENT_END):
        raise PermissionError("diagnostic store contains held-out dates")
    rows = np.flatnonzero(
        (dates <= np.datetime64(PRETRAIN_END))
        | (dates >= np.datetime64(FINETUNE_START))
    )
    store, access = open_store_for_dates(root, rows, purpose="training")
    fields = {}
    try:
        target_mask = store.read(REGISTERED_PRIMARY_TARGET_MASK, rows)
        target = store.read_target(
            REGISTERED_PRIMARY_TARGET, rows, valid_mask=target_mask
        )
        head = HORIZONS.index(5)
        supported_target = (
            np.asarray(store.read("active", rows)) & target_mask[..., head]
        )
        for family in ("fundamentals", "events"):
            names = store.manifest["feature_names"][f"sidecar_{family}"]
            values = store.read(f"sidecar_{family}_values", rows)
            valid = store.read(f"sidecar_{family}_valid", rows)
            for index, name in enumerate(names):
                fields[name] = {
                    "family": family,
                    "registered_expected_sign": EXPECTED_SIGNS[name],
                    **field_readout(
                        values[..., index],
                        target[..., head],
                        valid[..., index] & supported_target,
                        dates[rows],
                    ),
                }
    finally:
        store.close()
    result = {
        "schema": "BRAZIL_RV_ROUND6_FIELD_SIGN_READOUT_V1",
        "code": code,
        "registration_sha256": sha256_file(registration_path),
        "store": registration["base_store"],
        "target": REGISTERED_PRIMARY_TARGET,
        "horizon": 5,
        "minimum_names_per_daily_ic": 20,
        "dates": dates[rows].astype(str).tolist(),
        "access": access.payload(),
        "selection_weight": 0,
        "interpretation": "Expected signs are diagnostic priors, not gates; contrary signs never justify a sign flip or source change without independent evidence.",
        "fields": fields,
        "wall_seconds": time.monotonic() - started,
        "peak_rss_bytes": peak_rss_bytes(),
    }
    if result["peak_rss_bytes"] >= 8 * 2**30:
        raise MemoryError("Round-6 CPU sign diagnostic exceeded 8 GiB")
    if output.exists():
        raise FileExistsError(output)
    write_json_atomic(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.registration, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "fields": len(result["fields"]),
                "wall_seconds": result["wall_seconds"],
                "peak_rss_bytes": result["peak_rss_bytes"],
            }
        )
    )


if __name__ == "__main__":
    main()
