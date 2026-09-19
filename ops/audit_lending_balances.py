"""Recover cached historical loan balances using contemporaneous quote identities.

Outputs are source-audit tables, not an admission into features or shortability.
Unquoted legacy tickers stay unmapped; no current ticker or future quote is used.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import numpy as np
import polars as pl

from brazil_rv.preprocessing import bdi_lending as parser
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic

PROJECT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def parse_one(source, output, parser_sha):
    path = Path(source["path"])
    if sha256_file(path) != source["sha256"]:
        raise ValueError(f"source changed: {path}")
    receipt = output / f"{source['report_date']}.json"
    table = receipt.with_suffix(".parquet")
    if receipt.exists():
        saved = read(receipt)
        if saved["source"] == source and saved["parser_sha256"] == parser_sha:
            return saved
    result = {"source": source, "parser_sha256": parser_sha}
    try:
        bulletin, pages = parser.parse_bdi_pdf(
            path, date.fromisoformat(source["report_date"])
        )
        result["pages"] = pages
        if bulletin is None:
            result["status"] = "no_table"
        else:
            frame = pl.DataFrame(
                [asdict(row) for row in bulletin.positions]
            ).with_columns(pl.lit(bulletin.report_date).alias("report_date"))
            frame.write_parquet(table)
            result.update(
                status="parsed",
                rows=frame.height,
                position_date=str(bulletin.position_date),
                layout=bulletin.layout,
                table_sha256=sha256_file(table),
            )
    except ValueError as error:
        result.update(status="extraction_failure", error=str(error))
    write_json_atomic(receipt, result)
    return result


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--workers", type=int, default=4)
    args = cli.parse_args()
    root = Path(read(PROJECT / "docs/v2_economic_data_scaling_run.json")["root"])
    output = root / "lending_balance_audit"
    output.mkdir(exist_ok=True)
    archive = read(PROJECT / "docs/v2_round6_inputs.json")["evaluation_sources"][
        "design"
    ]["lending_archive"]
    archive_root = Path(archive["root"])
    if sha256_file(archive_root / "manifest.json") != archive["manifest_sha256"]:
        raise ValueError("accepted lending archive differs")
    accepted_manifest = read(archive_root / "manifest.json")
    old_rate = accepted_manifest["old_rate"]
    old_manifest = Path(old_rate["root"]) / "manifest.json"
    if sha256_file(old_manifest) != old_rate["manifest_sha256"]:
        raise ValueError("source inventory differs")
    # The accepted inventory already binds these cached original bulletins.
    # Only the missing historical balance era is needed here.
    records = [
        {key: row[key] for key in ("report_date", "path", "sha256")}
        for row in read(old_manifest)["pdf_audits"]
        if row["report_date"] < "2022-03-21"
    ]
    parser_sha = sha256_file(Path(parser.__file__))
    write_json_atomic(
        output / "source_inventory.json",
        {"records": records, "parser_sha256": parser_sha, "accepted_archive": archive},
    )
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(parse_one, row, output, parser_sha) for row in records]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 25 == 0 or len(results) == len(records):
                progress = {
                    "completed": len(results),
                    "total": len(records),
                    "extraction_failures": sum(
                        row["status"] == "extraction_failure" for row in results
                    ),
                }
                write_json_atomic(output / "progress.json", progress)
                print(progress, flush=True)
    frames = []
    for receipt in results:
        if receipt["status"] != "parsed":
            continue
        path = output / f"{receipt['source']['report_date']}.parquet"
        if sha256_file(path) != receipt["table_sha256"]:
            raise ValueError("recovered balance table differs")
        frames.append(pl.read_parquet(path))
    balances = pl.concat(frames, how="vertical_relaxed")
    pointer = read(PROJECT / "docs/v2_data_inputs.json")["store"]
    store = Path(pointer["root"])
    if sha256_file(store / "manifest.json") != pointer["manifest_sha256"]:
        raise ValueError("accepted model store differs")
    manifest = read(store / "manifest.json")
    quotes = []
    sources = []
    for source in manifest["sources"]:
        path = Path(source["path"])
        if not path.name.startswith("equities_daily_"):
            continue
        if not 2019 <= int(path.stem.rsplit("_", 1)[1]) <= 2022:
            continue
        if sha256_file(path) != source["sha256"]:
            raise ValueError("COTAHIST identity source differs")
        sources.append(source)
        quotes.append(
            pl.read_parquet(path)
            .filter(pl.col("market_type") == 10)
            .select(
                pl.col("trade_date").alias("position_date"),
                "ticker",
                pl.col("isin").alias("quote_isin"),
            )
        )
    quotes = pl.concat(quotes).unique()
    if quotes.select(pl.struct("position_date", "ticker").is_duplicated().any()).item():
        raise ValueError("ambiguous same-date ticker identity in COTAHIST")
    mapped = balances.join(quotes, on=["position_date", "ticker"], how="left")
    conflicts = mapped.filter(
        pl.col("isin").is_not_null()
        & pl.col("quote_isin").is_not_null()
        & (pl.col("isin") != pl.col("quote_isin"))
    )
    mapped = mapped.with_columns(pl.coalesce("isin", "quote_isin").alias("isin"))
    dates = np.load(store / "date_index.npy")
    if dates[-1] > np.datetime64("2024-12-30"):
        raise PermissionError("development-only source audit")
    report_day = mapped["report_date"].to_numpy().astype("datetime64[D]")
    next_day = np.searchsorted(dates, report_day, side="right")
    mapped = mapped.with_columns(pl.Series("available_date", dates[next_day]))
    mapped.sort("report_date", "ticker").write_parquet(
        output / "recovered_balances.parquet"
    )
    mapped.filter(pl.col("isin").is_null()).write_parquet(output / "unmapped.parquet")
    conflicts.write_parquet(output / "identity_conflicts.parquet")
    isins = np.load(store / "isin_index.npy").tolist()
    result = {
        "source_pdf_count": len(records),
        "parser_sha256": parser_sha,
        "model_store": pointer,
        "identity_sources": sources,
        "failed_extractions": [
            r for r in results if r["status"] == "extraction_failure"
        ],
        "no_table": [r for r in results if r["status"] == "no_table"],
        "parsed_rows": mapped.height,
        "mapped_rows": mapped.filter(pl.col("isin").is_not_null()).height,
        "model_axis_rows": mapped.filter(pl.col("isin").is_in(isins)).height,
        "identity_conflicts": conflicts.height,
        "first_position_date": str(mapped["position_date"].min()),
        "last_position_date": str(mapped["position_date"].max()),
        "status": "audit_only_pending_printed_row_reconciliation_and_source_review",
        "heldout_accessed": False,
    }
    write_json_atomic(output / "result.json", result)
    print(
        {
            k: v
            for k, v in result.items()
            if k not in {"identity_sources", "failed_extractions", "no_table"}
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
