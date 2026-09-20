"""Rank actual unquoted holdings in the saved matched books; never replay them."""

import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.portfolio_training import load_data

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = perf_counter()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    root = Path(run["stage_c_root"]) / "exposure_audit"
    root.mkdir(exist_ok=False)
    plan = bound_json(run["stage_c_plan"])
    replay = bound_json(run["stage_c_replays"])
    write_json_atomic(
        root / "plan.json",
        dict(
            replays=run["stage_c_replays"],
            scope="Every saved matched book, every nonzero stock holding on an unquoted date, including floating remnants; no PnL selection or book replay.",
            ranking="Maximum absolute marked holding at R10m; all capitals and every nonzero holding retained in the row table.",
            limits="An unquoted holding is a source investigation lead, not evidence of a particular corporate transition. Marks are saved account valuations, not observations. No contractual cohort inventory was saved.",
        ),
    )
    shutil.copyfile(__file__, root / "executed.py")
    frozen, cache = load_data(Path(plan["prior_root"]), "C6")
    dates = np.asarray(frozen.inputs.dates, dtype="datetime64[D]")
    ids = frozen.inputs.security_ids
    assert len(ids) == 933 and dates[-1] <= np.datetime64("2024-12-30")
    records, books = [], []
    for rec in replay["completed"]:
        book = bound_json(rec["book"])
        path = Path(rec["book"]["path"]).parent / "account.npz"
        assert sha256_file(path) == book["files"]["account.npz"]
        bd = np.asarray(book["state_dates"], dtype="datetime64[D]")
        rows = np.searchsorted(dates, bd)
        np.testing.assert_array_equal(dates[rows], bd)
        printed = np.isfinite(frozen.inputs.raw_close[rows]) & (
            frozen.inputs.raw_close[rows] > 0
        )
        with np.load(path) as z:
            shares, marks = z["signed_shares"], z["mark_price"]
            ii, jj = np.where((shares != 0) & ~printed)
            for i, j in zip(ii, jj):
                value = float(shares[i, j] * marks[i, j])
                records.append(
                    dict(
                        key=rec["key"],
                        capital=int(rec["key"].split("/")[1]),
                        date=str(bd[i]),
                        security_id=ids[j],
                        axis=int(j),
                        shares=float(shares[i, j]),
                        mark=float(marks[i, j]) if np.isfinite(marks[i, j]) else None,
                        signed_marked_brl=value if np.isfinite(value) else None,
                        absolute_marked_brl=abs(value) if np.isfinite(value) else None,
                        terminal=bool(i == len(bd) - 1),
                    )
                )
            books.append(
                dict(
                    key=rec["key"],
                    book=rec["book"],
                    account=binding(path),
                    unquoted_nonzero_cells=len(ii),
                    unquoted_nonzero_terminal_cells=int(
                        ((shares[-1] != 0) & ~printed[-1]).sum()
                    ),
                    economics_unresolved=rec["economics_unresolved"],
                    max_debit_brl=float(np.maximum(-z["free_cash"], 0).max()),
                    max_overdue_principal_brl=float(z["loan_overdue_principal"].max()),
                )
            )
    table = pl.DataFrame(records)
    table.write_parquet(root / "unquoted_holdings.parquet", compression="zstd")
    ranked = (
        table.filter(pl.col("capital") == 10000000)
        .group_by("security_id", "axis")
        .agg(
            pl.col("absolute_marked_brl").max().alias("max_absolute_marked_brl"),
            pl.col("date").min().alias("first_unquoted_held"),
            pl.col("date").max().alias("last_unquoted_held"),
            pl.len().alias("book_date_cells"),
            pl.col("key").n_unique().alias("books"),
            pl.col("terminal").sum().alias("terminal_book_cells"),
        )
        .sort("max_absolute_marked_brl", descending=True)
    )
    result = dict(
        plan=binding(root / "plan.json"),
        old_cache=cache,
        books=books,
        rows=binding(root / "unquoted_holdings.parquet"),
        ranked=ranked.to_dicts(),
        total_books=len(books),
        cells=table.height,
        seconds=perf_counter() - started,
        statement="Source-ranked exposure inventory only; no model profitability or new source admission.",
    )
    write_json_atomic(root / "manifest.json", result)
    print(
        json.dumps(
            {k: result[k] for k in ("total_books", "cells", "ranked", "seconds")}
        )
    )


if __name__ == "__main__":
    main()
