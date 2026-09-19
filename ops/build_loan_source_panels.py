"""Bind recovered stock/hedge reference prices and causal hedge rates.

No PDF reparsing, model-cache replacement, outcome read or financial replay.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.lending_archive import load_lending_borrow_panels
from brazil_rv.v2.lending_recovery import prior_loan_references

PROJECT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def bind(path):
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def main():
    pointer = read(PROJECT / "docs/v2_economic_data_scaling_run.json")
    root = Path(pointer["root"])
    output = root / "loan_source_panels"
    if output.exists():
        raise FileExistsError(output)
    store_binding = read(PROJECT / "docs/v2_data_inputs.json")["store"]
    store = Path(store_binding["root"])
    if sha256_file(store / "manifest.json") != store_binding["manifest_sha256"]:
        raise ValueError("accepted model store differs")
    dates = np.load(store / "date_index.npy").astype("datetime64[D]")
    if dates[-1] > np.datetime64("2024-12-30"):
        raise PermissionError("development-only loan references")
    isins = np.load(store / "isin_index.npy").tolist()
    active = np.load(store / "active.npy", mmap_mode="r")
    stock_path = root / "daily_source_audit/loan_reference_quotes.parquet"
    stock_manifest = root / "daily_source_audit/result.json"
    if read(stock_manifest)["store"] != store_binding:
        raise ValueError("stock quote recovery belongs to a different store")
    bova_record = pointer["bova_loan_reference_audit"]
    bova_manifest = Path(bova_record["path"])
    if sha256_file(bova_manifest) != bova_record["sha256"]:
        raise ValueError("hedge quote recovery identity differs")
    bova_source = read(bova_manifest)["data"]
    bova_path = Path(bova_source["path"])
    if sha256_file(bova_path) != bova_source["sha256"]:
        raise ValueError("hedge published averages changed")
    hedge_isin = "BRBOVACTF003"
    columns = ["trade_date", "isin", "average_brl"]
    stock = pl.read_parquet(stock_path).select(columns)
    bova = pl.read_parquet(bova_path).select(columns)
    quotes = pl.concat([stock, bova])
    if quotes["trade_date"].max().year > 2024:
        raise PermissionError("held-out quote in source panel")
    references, published = prior_loan_references(quotes, dates, [*isins, hedge_isin])

    lending_record = pointer["recovered_lending"]
    lending_root = Path(lending_record["root"])
    # Separate single-ETF axes preserve the equity cross-sectional fallback.
    hedge = load_lending_borrow_panels(
        lending_root,
        expected_manifest_sha256=lending_record["manifest_sha256"],
        canonical_dates=dates.astype(object).tolist(),
        canonical_isins=[hedge_isin],
    )
    hedge_rates = np.where(
        hedge.rate_placeholder[:, 0], np.nan, hedge.annual_taker_rate[:, 0]
    )
    source_age = (dates[:, None] - published).astype("timedelta64[D]").astype(float)
    finite = np.isfinite(references)
    if np.any(finite & ~(published < dates[:, None])):
        raise ValueError("a loan reference uses a current or future publication")
    output.mkdir()
    panels_path = output / "panels.npz"
    np.savez_compressed(
        panels_path,
        dates=dates,
        isins=np.asarray([*isins, hedge_isin]),
        loan_reference_prices=references,
        reference_publication_dates=published,
        hedge_annual_borrow_rate=hedge_rates,
    )
    report = {
        "schema": "BRAZIL_RV_LOAN_SOURCE_PANELS_V1",
        "status": "source_panels_bound_event_overrides_pending",
        "store": store_binding,
        "sources": [
            bind(stock_manifest),
            bind(stock_path),
            bind(bova_manifest),
            bind(bova_path),
        ],
        "lending": lending_record,
        "panels": bind(panels_path),
        "reference_rule": "same-ISIN last published average strictly before session; literal contract reference, no inferred ex-date adjustment or current-mark substitution",
        "hedge_rate_rule": "own positive-flow taker rate, available D+1, at most 60 sessions; NaN explicitly delegates to configured 2% fallback",
        "contract_source": bind(root / "primary_sources/b3_loan_contract.pdf"),
        "current_feature_coordinates_changed": False,
        "historical_outcomes_read": False,
        "source_rows": quotes.height,
        "dates": len(dates),
        "equity_names": len(isins),
        "eligible_stock_days": int(active.sum()),
        "eligible_reference_missing": int((active & ~finite[:, :-1]).sum()),
        "eligible_prior_quote_older_than_7_calendar_days": int(
            (active & (source_age[:, :-1] > 7)).sum()
        ),
        "hedge_reference_missing": int((~finite[:, -1]).sum()),
        "hedge_observed_rate_sessions": int(np.isfinite(hedge_rates).sum()),
        "hedge_fallback_sessions": int((~np.isfinite(hedge_rates)).sum()),
        "median_observed_hedge_annual_rate": float(np.nanmedian(hedge_rates)),
        "events": "contractual successor/loan allocation and ex-date overrides require event-specific admission; no historical account or profitability acceptance claimed",
    }
    write_json_atomic(output / "manifest.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
