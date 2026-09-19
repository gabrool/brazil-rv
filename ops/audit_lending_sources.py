"""Reconcile cached development BDI loan rows with the accepted rate archive."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import polars as pl

from brazil_rv.preprocessing import bdi_lending_strong as parser
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic

PROJECT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def parse_one(record, output, parser_sha):
    day = date.fromisoformat(record["report_date"])
    if day.year >= 2025:
        raise PermissionError("development-only source audit")
    path = Path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"BDI source identity differs: {path}")
    receipt = output / f"{day}.json"
    table = output / f"{day}.parquet"
    if receipt.exists():
        saved = read(receipt)
        if saved["source"] == record and saved["parser_sha256"] == parser_sha:
            return saved
    result = {"source": record, "parser_sha256": parser_sha}
    try:
        rows, pages = parser.parse_registered_pdf(path, day)
        result.update(status="parsed", rows=len(rows), pages=pages)
        if rows:
            pl.DataFrame([asdict(row) for row in rows]).write_parquet(table)
            result["table_sha256"] = sha256_file(table)
    except ValueError as error:
        # Preserve failed dates visibly; no partial table is accepted.
        result.update(status="extraction_failure", error=str(error))
    write_json_atomic(receipt, result)
    return result


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--workers", type=int, default=4)
    options = args.parse_args()
    root = Path(read(PROJECT / "docs/v2_economic_data_scaling_run.json")["root"])
    output = root / "lending_source_audit"
    output.mkdir(exist_ok=True)
    archive = read(PROJECT / "docs/v2_round6_inputs.json")["evaluation_sources"][
        "design"
    ]["lending_archive"]
    archive_root = Path(archive["root"])
    if sha256_file(archive_root / "manifest.json") != archive["manifest_sha256"]:
        raise ValueError("accepted lending archive differs")
    manifest = read(archive_root / "manifest.json")
    old = read(Path(manifest["old_rate"]["root"]) / "manifest.json")
    records = {}
    quarters = set()
    for row in old["pdf_audits"]:
        day = date.fromisoformat(row["report_date"])
        quarter = (day.year, (day.month - 1) // 3)
        # Complete later rate history, plus one earlier PDF per quarter to
        # check the claimed first table against original bulletin content.
        if day >= date(2023, 7, 1) or quarter not in quarters:
            records[day.isoformat()] = {
                key: row[key] for key in ("report_date", "path", "sha256")
            }
        quarters.add(quarter)
    new_root = Path(manifest["new_pdf_snapshot"]["root"])
    for row in read(new_root / "manifest.json")["files"]:
        if row["status"] == "downloaded":
            records[row["report_date"]] = {
                "report_date": row["report_date"],
                "path": str(new_root / row["filename"]),
                "sha256": row["sha256"],
            }
    parser_sha = sha256_file(Path(parser.__file__))
    write_json_atomic(
        output / "source_inventory.json",
        {
            "parser_sha256": parser_sha,
            "accepted_archive": archive,
            "records": list(records.values()),
            "heldout_accessed": False,
        },
    )
    results = []
    with ProcessPoolExecutor(options.workers) as pool:
        futures = [
            pool.submit(parse_one, r, output, parser_sha) for r in records.values()
        ]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 20 == 0 or len(results) == len(records):
                progress = {
                    "completed": len(results),
                    "total": len(records),
                    "failures": sum(r["status"] != "parsed" for r in results),
                }
                write_json_atomic(output / "progress.json", progress)
                print(progress, flush=True)
    good = sorted(
        (r for r in results if r["status"] == "parsed" and r["rows"]),
        key=lambda r: r["source"]["report_date"],
    )
    frames = []
    for record in good:
        path = output / f"{record['source']['report_date']}.parquet"
        if sha256_file(path) != record["table_sha256"]:
            raise ValueError("parsed loan table differs from its receipt")
        frames.append(pl.read_parquet(path))
    loans = pl.concat(frames)
    daily = (
        loans.group_by("report_date", "isin")
        .agg(
            pl.col("contracts").sum().alias("registered_contracts"),
            pl.col("quantity").sum().alias("registered_quantity"),
            pl.col("value_brl").sum().alias("registered_value_brl"),
            (pl.col("quantity") * pl.col("taker_avg") / 100)
            .sum()
            .alias("taker_weighted"),
            (pl.col("quantity") * pl.col("donor_avg") / 100)
            .sum()
            .alias("donor_weighted"),
        )
        .filter(pl.col("registered_quantity") > 0)
        .with_columns(
            (pl.col("taker_weighted") / pl.col("registered_quantity")).alias(
                "annual_taker_rate"
            ),
            (pl.col("donor_weighted") / pl.col("registered_quantity")).alias(
                "annual_donor_rate"
            ),
            pl.concat_str(pl.lit("ISIN:"), "isin").alias("security_id"),
        )
        .drop("taker_weighted", "donor_weighted")
        .rename({"report_date": "source_trade_date"})
    )
    daily = daily.sort("source_trade_date", "security_id")
    daily.write_parquet(output / "recovered_registered_loans.parquet")
    accepted = pl.read_parquet(archive_root / "lending_rates.parquet")
    comparison = accepted.join(
        daily,
        on=["source_trade_date", "security_id"],
        how="full",
        coalesce=True,
        suffix="_recovered",
    ).with_columns(
        (pl.col("annual_taker_rate_recovered") - pl.col("annual_taker_rate")).alias(
            "rate_delta"
        ),
        (pl.col("registered_quantity_recovered") - pl.col("registered_quantity")).alias(
            "quantity_delta"
        ),
    )
    comparison.write_parquet(output / "accepted_rate_comparison.parquet")
    overlap = comparison.filter(pl.col("rate_delta").is_not_null())
    result = {
        "parser_sha256": parser_sha,
        "source_pdf_count": len(results),
        "failed_extractions": [r for r in results if r["status"] != "parsed"],
        "printed_rows_reconciled": loans.height,
        "positive_flow_security_days": daily.height,
        "first_rate_date": str(daily["source_trade_date"].min()),
        "last_rate_date": str(daily["source_trade_date"].max()),
        "overlap_security_days": overlap.height,
        "overlap_rate_changed_gt_1e8": overlap.filter(
            pl.col("rate_delta").abs() > 1e-8
        ).height,
        "overlap_quantity_changed": overlap.filter(
            pl.col("quantity_delta") != 0
        ).height,
        "existing_not_recovered": comparison.filter(
            pl.col("annual_taker_rate_recovered").is_null()
        ).height,
        "new_security_days_all_b3_assets": comparison.filter(
            pl.col("annual_taker_rate").is_null()
        ).height,
        "status": "audit_only_not_admitted",
        "heldout_accessed": False,
    }
    write_json_atomic(output / "result.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
