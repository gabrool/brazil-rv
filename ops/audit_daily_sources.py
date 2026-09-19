"""Independent COTAHIST unit/identity reconciliation to the accepted model store."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic

PROJECT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    pointer = read(PROJECT / "docs/v2_data_inputs.json")["store"]
    store = Path(pointer["root"])
    if sha256_file(store / "manifest.json") != pointer["manifest_sha256"]:
        raise ValueError("accepted store identity changed")
    manifest = read(store / "manifest.json")
    root = Path(read(PROJECT / "docs/v2_economic_data_scaling_run.json")["root"])
    output = root / "daily_source_audit"
    output.mkdir(exist_ok=True)
    dates = np.load(store / "date_index.npy")
    isins = np.load(store / "isin_index.npy").tolist()
    if dates[-1] > np.datetime64("2024-12-30"):
        raise PermissionError("development-only source audit")
    calendar = pl.DataFrame({"trade_date": dates, "day": np.arange(len(dates))})
    identities = pl.DataFrame({"isin": isins, "name": np.arange(len(isins))})
    fields = {
        "open_brl": "raw_open",
        "high_brl": "raw_high",
        "low_brl": "raw_low",
        "close_brl": "raw_close",
        "volume_brl": "volume_brl",
        "quantity": "quantity",
        "trades": "trade_count",
    }
    arrays = {
        k: np.load(store / manifest["arrays"][v]["path"], mmap_mode="r")
        for k, v in fields.items()
    }
    observed = np.load(store / "observed.npy", mmap_mode="r")
    active = np.load(store / "active.npy", mmap_mode="r")
    matched = np.zeros_like(observed)
    records, exceptions, quotes = [], [], []
    sources = [
        r
        for r in manifest["sources"]
        if Path(r["path"]).name.startswith("equities_daily_")
        and 2010 <= int(Path(r["path"]).stem.rsplit("_", 1)[1]) <= 2024
    ]
    continuation = manifest["metadata"]["round7_repair"]["source_audit"]
    if sha256_file(Path(continuation["path"])) != continuation["sha256"]:
        raise ValueError("continuation source manifest changed")
    sources.append(read(continuation["path"])["continued_prints"]["data"])
    for source in sources:
        path = Path(source["path"])
        if sha256_file(path) != source["sha256"]:
            raise ValueError(f"COTAHIST source changed: {path}")
        frame = (
            pl.read_parquet(path)
            .filter(pl.col("market_type") == 10)
            .join(calendar, on="trade_date")
            .join(identities, on="isin")
        )
        if frame.select(pl.struct("day", "name").is_duplicated().any()).item():
            raise ValueError("ambiguous source date/ISIN")
        day, name = frame["day"].to_numpy(), frame["name"].to_numpy()
        if matched[day, name].any():
            raise ValueError("multiple sources claim the same observed security/day")
        matched[day, name] = True
        seen = observed[day, name]
        mismatches = {}
        for column, values in arrays.items():
            expected = frame[column].to_numpy().astype(values.dtype)
            actual = values[day, name]
            different = seen & (expected != actual)
            mismatches[column] = int(different.sum())
            if different.any():
                exceptions.extend(
                    frame.filter(pl.Series(different))
                    .select("trade_date", "isin", "ticker", column)
                    .with_columns(pl.lit(column).alias("field"))
                    .head(20)
                    .to_dicts()
                )
        average = frame["average_brl"].to_numpy()
        quantity = frame["quantity"].to_numpy()
        volume = frame["volume_brl"].to_numpy()
        factor = frame["quote_factor"].to_numpy()
        vwap = np.divide(
            volume, quantity, out=np.full_like(volume, np.nan), where=quantity > 0
        )
        error = np.abs(vwap - average)
        # The source quotes averages in cents per FATCOT lot; allow one last
        # displayed decimal unit and cent rounding of the monetary total.
        tolerance = 0.010001 / factor + 0.011 / np.maximum(quantity, 1)
        bad_average = (quantity > 0) & (error > tolerance)
        bounds = (average < frame["low_brl"].to_numpy() - tolerance) | (
            average > frame["high_brl"].to_numpy() + tolerance
        )
        anomaly = frame.with_columns(
            pl.Series("implied_average_brl", vwap),
            pl.Series("average_error_brl", error),
            pl.Series("active", active[day, name]),
            pl.Series("observed", seen),
        ).filter(pl.Series(bad_average | bounds))
        if anomaly.height:
            anomaly.write_parquet(output / f"{path.stem}_average_exceptions.parquet")
        quotes.append(
            frame.select(
                "trade_date",
                "isin",
                "average_brl",
                "close_brl",
                "quantity",
                "volume_brl",
            )
        )
        records.append(
            {
                "source": source,
                "source_rows_on_store_axes": frame.height,
                "observed_rows": int(seen.sum()),
                "source_present_store_not_observed": int((~seen).sum()),
                "active_source_present_store_not_observed": int(
                    (~seen & active[day, name]).sum()
                ),
                "exact_after_store_dtype_mismatches": mismatches,
                "average_vs_turnover_exceptions": int(bad_average.sum()),
                "average_outside_ohlc": int(bounds.sum()),
            }
        )
    pl.concat(quotes).sort("trade_date", "isin").write_parquet(
        output / "loan_reference_quotes.parquet"
    )
    result = {
        "store": pointer,
        "sources": records,
        "observed_without_source": int((observed & ~matched).sum()),
        "active_observed_without_source": int((observed & active & ~matched).sum()),
        "mismatch_examples": exceptions,
        "scope": "parsed COTAHIST to raw store and independent average/quantity/turnover units; not a corporate-action or label acceptance",
        "heldout_accessed": False,
    }
    write_json_atomic(output / "result.json", result)
    print(
        {
            "years": len(records),
            "observed_without_source": result["observed_without_source"],
            "mismatches": sum(
                sum(r["exact_after_store_dtype_mismatches"].values()) for r in records
            ),
            "average_exceptions": sum(
                r["average_vs_turnover_exceptions"] for r in records
            ),
        }
    )


if __name__ == "__main__":
    main()
