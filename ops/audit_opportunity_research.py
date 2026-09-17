"""Verify frozen inputs, causal fits, saved books and currency report consistency."""

import argparse
import json
from pathlib import Path

import numpy as np

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.opportunity_research import checked, benchmark_for
from brazil_rv.v2.performance import performance
from brazil_rv.v2.portfolio_program import read


def audit(root):
    design = read(root / "frozen_design.json")
    for key in (
        "source_design",
        "registration",
        "sector_source",
        "identity",
        "market",
        "effr",
        "benchmarks",
    ):
        checked(design[key])
    output = {"design_sha256": sha256_file(root / "frozen_design.json"), "arms": {}}
    for arm, entry in design["arms"].items():
        complete = read(root / arm / "complete.json")
        for key in ("supervised", "fits"):
            checked(complete[key])
        with np.load(checked(entry["prepared"]), allow_pickle=False) as values:
            dates = values["dates"]
            valid = values["valid"]
            ranks = values["ranks"]
            if (
                dates[-1] > np.datetime64("2024-12-31")
                or not np.isfinite(ranks[valid]).all()
            ):
                raise ValueError("invalid forecast population or development boundary")
        for fold, record in read(root / arm / "fits.json").items():
            if record["last_origin"] + 5 >= record["first_decision"]:
                raise ValueError(f"unmatured calibration outcome in {arm}/{fold}")
        books = {}
        for path in sorted((root / arm / "books").glob("*/book.json")):
            book = read(path)
            for name, digest in book["files"].items():
                if sha256_file(path.parent / name) != digest:
                    raise ValueError(
                        f"saved book artifact changed: {path.parent / name}"
                    )
            if book["provenance"]["frozen_design_sha256"] != output["design_sha256"]:
                raise ValueError("book does not bind this experiment")
            analysis = read(path.parent / "analysis.json")
            checked(analysis["book"])
            ds = np.asarray(book["dates"], dtype="datetime64[D]")
            first = np.searchsorted(dates, ds[0])
            if not np.array_equal(ds, dates[first : first + len(ds)]):
                raise ValueError("book omitted intervening trading dates")
            daily = book["daily"]
            metrics = performance(
                np.asarray(daily["absolute_bps"]) / 1e4,
                np.asarray(daily["cdi_bps"]) / 1e4,
                benchmark_for(root, ds, dates[first - 1]),
            )
            if metrics != analysis["metrics"]:
                raise ValueError("published performance differs from account/benchmark")
            # Exact P&L identity; sector contribution remains explicitly a proxy.
            reconstructed = (
                np.asarray(daily["equity_gross_bps"]) + daily["hedge_gross_bps"]
            )
            reconstructed -= np.asarray(daily["trading_cost_bps"]) + daily["borrow_bps"]
            reconstructed += np.asarray(daily["interest_bps"]) - daily["cdi_bps"]
            error = float(np.abs(reconstructed - daily["net_excess_bps"]).max())
            if (
                error > 1e-6
                or book["summary"]["max_absolute_reconciliation_error"] > 1e-7
            ):
                raise ValueError(f"unreconciled account {path}: {error}")
            books[path.parent.name] = {
                "book_sha256": sha256_file(path),
                "analysis_sha256": sha256_file(path.parent / "analysis.json"),
                "sessions": len(ds),
                "decomposition_max_error_bps": error,
                "fx_max_age_days": metrics["fx_max_age_days"],
            }
        if len(books) != 10 or complete["neutral_max_daily_error"] > 1e-7:
            raise ValueError("missing registered book or failed neutral parity")
        output["arms"][arm] = {
            "books": books,
            "neutral_max_daily_error": complete["neutral_max_daily_error"],
            "forecast_stock_days": int(valid.sum()),
            "heldout_accessed": False,
        }
    output["status"] = "passed"
    write_json_atomic(root / "audit.json", output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    print(
        json.dumps(
            {
                "status": result["status"],
                "books": sum(len(a["books"]) for a in result["arms"].values()),
            }
        )
    )
