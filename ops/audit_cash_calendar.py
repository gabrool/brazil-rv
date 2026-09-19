"""Reconcile bound SGS observations and write an explicit corrected cash contract."""

import json
from pathlib import Path
import time

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.cash_calendar import close_interval_returns

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = time.perf_counter()
    pointer_path = PROJECT / "docs/v2_economic_data_scaling_run.json"
    pointer = json.loads(pointer_path.read_text())
    root = Path(pointer["root"]) / "cash_calendar"
    if root.exists():
        raise FileExistsError(root)
    source = json.loads((PROJECT / "docs/v2_checkpoint_cdi_evidence.json").read_text())[
        "development_extension"
    ]
    source_path = Path(source["path"])
    assert sha256_file(source_path) == source["sha256"]
    store = json.loads((PROJECT / "docs/v2_data_inputs.json").read_text())["store"]
    store_path = Path(store["root"])
    assert sha256_file(store_path / "manifest.json") == store["manifest_sha256"]
    calendar = np.load(store_path / "date_index.npy")
    if calendar[-1] > np.datetime64("2024-12-30"):
        raise ValueError("cash audit must stay on the development calendar")
    frame = pl.read_parquet(source_path).sort("trade_date")
    money = frame["trade_date"].to_numpy().astype("datetime64[D]")
    rates = frame["daily_cdi_rate"].to_numpy()
    corrected = close_interval_returns(money, rates, calendar)
    by_day = dict(zip(money, rates))
    old = np.array([by_day.get(day, np.nan) for day in calendar])
    valid = np.isfinite(old) & np.isfinite(corrected)
    change = valid & (np.abs(corrected - old) > 1e-14)
    omitted = (
        (money >= calendar[np.flatnonzero(valid)[0] - 1])
        & (money < calendar[-1])
        & ~np.isin(money, calendar)
    )
    # Independent recently retrieved BCB values also reconcile the bound vintage.
    check_path = (
        Path(pointer["root"]) / "primary_sources/cdi_sgs12_20221024_20230113.json"
    )
    check = json.loads(check_path.read_text())
    from datetime import datetime

    for row in check:
        day = np.datetime64(datetime.strptime(row["data"], "%d/%m/%Y").date())
        if by_day[day] != float(row["valor"]) / 100:
            raise ValueError(
                "retrieved monetary observations differ from bound vintage"
            )
    rows = [
        dict(
            date=str(calendar[i]),
            previous_equity_close=str(calendar[i - 1]),
            old_same_date_return=float(old[i]),
            corrected_interval_return=float(corrected[i]),
            delta_bps=float((corrected[i] - old[i]) * 1e4),
        )
        for i in np.flatnonzero(change)
    ]
    root.mkdir()
    panel_path = root / "returns.npz"
    np.savez_compressed(panel_path, dates=calendar, cdi_returns=corrected)
    receipt = dict(
        status="cash_calendar_defect_quantified_corrected_explicit_amendment_not_yet_scored",
        source=source,
        calendar={
            "path": str(store_path / "date_index.npy"),
            "sha256": sha256_file(store_path / "date_index.npy"),
        },
        panel={"path": str(panel_path), "sha256": sha256_file(panel_path)},
        source_rows=len(money),
        verified_overlapping_new_source_rows=len(check),
        comparison_sessions=int(valid.sum()),
        changed_sessions=int(change.sum()),
        omitted_monetary_dates=[str(day) for day in money[omitted]],
        omitted_monetary_date_count=int(omitted.sum()),
        mean_return_change_bps_per_equity_session=float(
            np.mean((corrected - old)[valid]) * 1e4
        ),
        old_cash_growth=float(np.prod(1 + old[valid]) - 1),
        corrected_cash_growth=float(np.prod(1 + corrected[valid]) - 1),
        rows=rows,
        contract="compound all sourced monetary returns on [previous equity close,current equity close), including money-only dates; neither current-date money rate nor future observation enters the ending interval",
        evidence="B3 Dommo contract starts CDI at approving AGE, and Oct24-inclusive/Jan13-exclusive SGS12 reproduces Jan6 issuer redemption 1.90432468607 to 6e-13",
        upstream_builder="execution/experiment52.py::_fetch_cdi retains raw monetary dates/percent-to-fraction units",
        defect="execution/inputs.py::load_daily_cdi_rates selects one same-date rate; v2/validate_pipeline.py::_load_development_cdi passes that into close-to-close accounting. Exact overlap with older experiments proved value preservation, not interval correctness",
        application="load new panel and apply_cash_calendar after frozen policy loading; old neural/static features and old results remain bound to original sources; benchmark, money income and debits must all use one interval series",
        limitations="source revision timestamps remain unavailable; first/pre-source intervals unresolved, later full loan/spot clearing-calendar treatment still separate; not model alpha or completion of deep audit",
        elapsed_seconds=time.perf_counter() - started,
    )
    manifest = root / "manifest.json"
    digest = write_json_atomic(manifest, receipt)
    doc = PROJECT / "docs/v2_cash_calendar_audit.json"
    doc_digest = write_json_atomic(doc, receipt)
    pointer["cash_calendar"] = {
        "path": str(manifest),
        "sha256": digest,
        "status": receipt["status"],
    }
    pointer["cash_calendar_audit"] = {
        "path": str(doc.relative_to(PROJECT)),
        "sha256": doc_digest,
    }
    write_json_atomic(pointer_path, pointer)
    print(
        json.dumps(
            {
                k: v
                for k, v in receipt.items()
                if k not in ("rows", "omitted_monetary_dates")
            }
        )
    )


if __name__ == "__main__":
    main()
