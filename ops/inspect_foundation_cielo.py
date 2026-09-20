"""Read Cielo inventory from the existing foundation book index; never replay."""

import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    tick = perf_counter()
    run = json.loads(
        (PROJECT / "docs/v2_economic_data_scaling_run.json").read_text(encoding="utf8")
    )
    source = PROJECT / "docs/v2_foundation_settlement_exposure.json"
    index = json.loads(source.read_text(encoding="utf8"))
    output = Path(run["root"]) / "cielo_foundation_exposure"
    output.mkdir(exist_ok=False)
    shutil.copyfile(__file__, output / "executed.py")
    cases, selected, values, fills = [], [], {}, []
    for item in index["books"]:
        book_ref = dict(path=item["book"], sha256=item["book_sha256"])
        book = bound_json(book_ref)
        dates = book["state_dates"]
        if "2024-08-30" not in dates:
            continue
        assert max(dates) <= "2024-12-30"
        folder = Path(item["book"]).parent
        paths = [folder / name for name in ("account.npz", "fills.parquet")]
        for path in paths:
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest()
                == book["files"][path.name]
            )
        selected.append(book_ref)
        k = len(cases)
        with np.load(paths[0]) as arrays:
            shares = arrays["signed_shares"][:, 235].copy()
            target = arrays["targets"][:, 235].copy()
            marks = arrays["mark_price"][:, 235].copy()
            assert shares.shape == (len(dates),)
            initial_nav = float(arrays["start_nav"][0])
            for name, value in (
                ("shares", shares),
                ("targets", target),
                ("marks", marks),
            ):
                values[f"book{k}_{name}"] = value
        trades = pl.read_parquet(paths[1]).filter(pl.col("security_index") == 235)
        for row in trades.to_dicts():
            assert str(row["fill_date"]) <= "2024-12-30"
            fills.append(dict(book=k, **row))
        negative = np.flatnonzero(shares < 0)
        boundary = (np.asarray(dates) >= "2024-08-26") & (
            np.asarray(dates) <= "2024-08-30"
        )
        rows = [
            dict(
                date=dates[i],
                shares=float(shares[i]),
                target=float(target[i]),
                signed_notional_initial_nav_bps=float(
                    shares[i] * marks[i] / initial_nav * 1e4
                ),
            )
            for i in np.flatnonzero(boundary)
        ]
        cases.append(
            dict(
                book=book_ref,
                account=dict(path=str(paths[0]), sha256=book["files"]["account.npz"]),
                fills=dict(path=str(paths[1]), sha256=book["files"]["fills.parquet"]),
                initial_nav=initial_nav,
                dates=dates,
                negative_inventory_days=len(negative),
                last_negative_date=dates[negative[-1]] if len(negative) else None,
                minimum_shares=float(shares.min()),
                boundary=rows,
                source_specific_loan_records="not_saved_by_this_old_account_contract",
            )
        )
    np.savez_compressed(output / "focus_arrays.npz", **values)
    write_json_atomic(output / "selected_books.json", selected)
    write_json_atomic(output / "cases.json", cases)
    write_json_atomic(output / "focus_fills.json", fills)
    report = dict(
        status="saved_old_inventory_scope_qualified_not_held_loan_cent_bound",
        source=binding(source),
        indexed_books=len(index["books"]),
        selected_books=len(cases),
        selection="Every indexed book whose saved state_dates contains 2024-08-30; no outcome or model-profit selection.",
        security="BRCIELACNOR3",
        axis=235,
        cases=binding(output / "cases.json"),
        arrays=binding(output / "focus_arrays.npz"),
        focus_array_cells=sum(v.size for v in values.values()),
        fills=binding(output / "focus_fills.json"),
        focus_fill_rows=len(fills),
        boundary_negative_books=sum(
            any(r["shares"] < 0 for r in c["boundary"]) for c in cases
        ),
        aug30_negative_books=sum(
            any(r["date"] == "2024-08-30" and r["shares"] < 0 for r in c["boundary"])
            for c in cases
        ),
        limitations=[
            "These old accounts saved economic inventory and synthetic terminal settlements, not contractual loan cohorts or pending returns. Neither long nor flat economic stock establishes absence of pending old loans.",
            "No replay, neural scoring, source/loan rate change, preference change, new locate, old-fit/store mutation or held-out consumer read. This is an exposure search, not a corrected model profit or a Cielo held-loan cent bound.",
            "Scope is the existing 140-book foundation settlement index, including its ensemble/reference books. It is not an exhaustive search of all seed books or future corrected adaptive paths.",
        ],
        seconds=perf_counter() - tick,
    )
    write_json_atomic(output / "manifest.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
