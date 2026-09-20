"""Bounded saved foundation exposure at the two newly admitted spot identities."""

import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    start = perf_counter()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    index = PROJECT / "docs/v2_foundation_settlement_exposure.json"
    source = json.loads(index.read_text())
    terms = bound_json(run["composed_primary_event_terms"])
    isins = np.load(Path(terms["store"]["root"]) / "isin_index.npy").tolist()
    out = (
        Path(run["root"]) / "integrated_admission" / "surviving_exposure" / "qualified"
    )
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / "executed.py")
    windows = [("BRSSBRACNOR1", "2019-08-06"), ("BRARZZACNOR3", "2024-08-01")]
    write_json_atomic(
        out / "plan.json",
        dict(
            index=binding(index),
            windows=windows,
            selection="Every indexed book containing either effect date; source axis, three preceding/two following dates, actual fills. No outcome selection or replay.",
        ),
    )
    cases = []
    cells = 0
    for item in source["books"]:
        ref = dict(path=item["book"], sha256=item["book_sha256"])
        meta = bound_json(ref)
        dates = meta["state_dates"]
        selected = [(sid, date) for sid, date in windows if date in dates]
        if not selected:
            continue
        assert max(dates) <= "2024-12-30"
        folder = Path(item["book"]).parent
        for filename in ("account.npz", "fills.parquet"):
            assert sha256_file(folder / filename) == meta["files"][filename]
        for sid, date in selected:
            n = isins.index(sid)
            t = dates.index(date)
            rows = np.arange(max(0, t - 3), min(len(dates), t + 3))
            with np.load(folder / "account.npz") as z:
                arrays = {
                    k: [float(x) if np.isfinite(x) else None for x in z[k][rows, n]]
                    for k in ("signed_shares", "targets", "mark_price")
                }
            cells += 3 * len(rows)
            fills = pl.read_parquet(folder / "fills.parquet").filter(
                (pl.col("security_index") == n)
                & pl.col("fill_session").is_in(rows.tolist())
            )
            cases.append(
                dict(
                    book=ref,
                    source=sid,
                    effect=date,
                    dates=[dates[i] for i in rows],
                    arrays=arrays,
                    fills=fills.to_dicts(),
                    exposed=any(x != 0 for x in arrays["signed_shares"]),
                    source_specific_loan_records="not saved; flat net stock does not prove absent pending loans",
                )
            )
    write_json_atomic(out / "cases.json", cases)
    report = dict(
        source=binding(index),
        cases=binding(out / "cases.json"),
        indexed_books=len(source["books"]),
        selected_book_event_pairs=len(cases),
        focus_cells=cells,
        fills=sum(len(c["fills"]) for c in cases),
        exposed_book_event_pairs=sum(c["exposed"] for c in cases),
        limitations="Existing ensemble/reference index only, not all seed or future corrected paths. Net inventory does not reconstruct missing contractual cohorts. No old book/model/store mutation or replay.",
        seconds=perf_counter() - start,
    )
    write_json_atomic(out / "manifest.json", report)
    print(json.dumps(report))
    print(
        json.dumps(
            [
                {k: v for k, v in c.items() if k != "fills"}
                for c in cases
                if c["exposed"]
            ]
        )
    )


if __name__ == "__main__":
    main()
