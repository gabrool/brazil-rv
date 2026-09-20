"""Saved-book custody qualification with independent physical-flow and Decimal fee arithmetic."""

from collections import defaultdict
from decimal import Decimal
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def fee_decimal(value, year):
    value = Decimal(str(value))
    if value < Decimal("23084.39" if year == "2023" else "24164.73"):
        return 0.0
    edges = [
        0,
        100000,
        200000,
        300000,
        1700000,
        17000000,
        170000000,
        1700000000,
        17000000000,
    ]
    rates = [
        ".0005",
        ".0004",
        ".0002",
        ".00013",
        ".000072",
        ".000032",
        ".000025",
        ".000015",
        ".000005",
    ]
    total = Decimal(0)
    for i, (lo, rate) in enumerate(zip(edges, rates)):
        amount = max(value - lo, Decimal(0))
        if i + 1 < len(edges):
            amount = min(amount, Decimal(edges[i + 1] - lo))
        total += amount * Decimal(rate) / 12
    return float(total)


def main():
    tick = perf_counter()
    run = json.loads(
        (PROJECT / "docs/v2_economic_data_scaling_run.json").read_text(encoding="utf8")
    )
    root = Path(run["root"]) / "custody_fees"
    out = root / "qualification"
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / "executed.py")
    audit = bound_json(binding(root / "manifest.json"))
    cases = bound_json(audit["completed"])
    plan = bound_json(audit["plan"])
    terms = bound_json(run["enat_settlement_terms"])
    store = Path(terms["store"]["root"])
    dates = np.load(store / "date_index.npy").astype("datetime64[D]")
    raw = np.load(store / "raw_close.npy", mmap_mode="r")
    qs = np.load(store / "action_shares_per_prior_share.npy", mmap_mode="r")
    ds = np.load(store / "action_cash_per_prior_share.npy", mmap_mode="r")
    resolved = np.load(store / "action_session_resolved.npy", mmap_mode="r")
    with np.load(bound_json(run["cash_calendar"])["panel"]["path"]) as z:
        cdi = z["cdi_returns"]
    hedge = pl.read_parquet(
        bound_json(run["bova_loan_reference_audit"])["data"]["path"]
    )
    hedge_map = dict(zip(hedge["trade_date"], hedge["close_brl"]))
    books, details, checks, contrasts = {}, {}, [], []
    errors = defaultdict(float)
    cells = fill_count = assessment_count = marked_without_print = 0
    for case in cases:
        folder = root / case["book"]
        book = json.loads((folder / "book.json").read_text(encoding="utf8"))
        assert not book["legacy_slot_diagnostics_applicable"]
        for path, digest in book["files"].items():
            assert sha256_file(folder / path) == digest
        with np.load(folder / "account.npz") as z:
            a = {k: z[k] for k in z.files}
        rows = json.loads(
            (folder / "funding_and_costs.json").read_text(encoding="utf8")
        )
        key = (case["start"], case["capital"], case["scenario"])
        books[key], details[key] = a, rows
        assert a["signed_shares"].shape == (48, 933)
        cells += sum(v.size for v in a.values())
        nav = (
            a["free_cash"]
            + a["restricted_cash"]
            + a["hedge_restricted_cash"]
            + a["unsettled_cash"]
            + a["receivables"]
            - a["payables"]
            + (a["signed_shares"] * np.nan_to_num(a["mark_price"])).sum(1)
            + a["hedge_signed_shares"] * np.nan_to_num(a["hedge_mark_price"])
            - a["loan_liability"]
            - a["custody_liability"]
        )
        errors["nav"] = max(errors["nav"], float(np.max(np.abs(nav - a["nav"]))))
        cfg = book["provenance"]["config"]
        schedule = {x["date"]: x["payment_date"] for x in cfg["custody_assessments"]}
        fill_rows = pl.read_parquet(folder / "fills.parquet").to_dicts()
        fill_count += len(fill_rows)
        by_day = defaultdict(list)
        for f in fill_rows:
            by_day[f["fill_session"]].append(f)
        # Independent event-flow accumulation: spot D2, new loans D1, ordinary
        # covering returns D2. Renewal has no transfer. No loan subledger reused.
        economic = np.zeros(934)
        physical = np.zeros(934)
        flows = defaultdict(lambda: np.zeros(934))
        first = int(np.searchsorted(dates, np.datetime64(case["start"])))
        prior = np.r_[
            np.load(store / "prior_reference_close.npy", mmap_mode="r")[first],
            hedge_map[dates[first - 1].astype(object)],
        ]
        invoices = []
        assessment_rows = []
        for day, row in enumerate(rows):
            index = first + day
            date = str(dates[index])
            q = np.r_[np.where(resolved[index], qs[index], 1), 1.0].astype(float)
            d = np.r_[np.where(resolved[index], ds[index], 0), 0.0].astype(float)
            assert np.all(q > 0), (
                "nonordinary action needs independent source disposition"
            )
            physical *= q
            economic *= q
            for due in flows:
                flows[due] *= q
            prior = (prior - d) / q
            physical += flows.pop(day, np.zeros(934))
            for f in by_day[day]:
                name = 933 if f["purpose"] == "hedge" else f["security_index"]
                quantity = f["quantity"] * (1 if f["side"] == "buy" else -1)
                cover = min(max(quantity, 0), max(-economic[name], 0))
                opening = max(-quantity - max(economic[name], 0), 0)
                economic[name] += quantity
                flows[day + 2][name] += quantity - cover
                flows[day + 1][name] += opening
            shares = np.r_[a["signed_shares"][day], a["hedge_signed_shares"][day]]
            errors["fill_share_conservation"] = max(
                errors["fill_share_conservation"],
                float(np.max(np.abs(economic - shares))),
            )
            close = np.r_[
                raw[index].astype(float), hedge_map[dates[index].astype(object)]
            ]
            observed = np.isfinite(close) & (close > 0)
            prior = np.where(observed, close, prior)
            if schedule:
                errors["independent_physical_flow"] = max(
                    errors["independent_physical_flow"],
                    float(np.max(np.abs(physical - a["physical_custody"][day]))),
                )
                errors["identical_intention_physical"] = max(
                    errors["identical_intention_physical"], row["physical_error"]
                )
            charged = 0.0
            if date in schedule:
                assert index + 1 < len(dates) and dates[index].astype(
                    "datetime64[M]"
                ) != dates[index + 1].astype("datetime64[M]")
                assessment_count += 1
                qty = np.maximum(
                    economic if cfg["custody_base"] == "economic_long" else physical, 0
                )
                marked_without_print += int(np.sum((qty > 1e-8) & ~observed))
                # Sparse Decimal dot product uses original stored source precision.
                value = sum(
                    (
                        Decimal(str(float(x))) * Decimal(str(float(p)))
                        for x, p in zip(qty, prior)
                        if x > 0
                    ),
                    Decimal(0),
                )
                errors["physical_source_price_base"] = max(
                    errors["physical_source_price_base"],
                    abs(float(value) - a["custody_base"][day]),
                )
                charged = fee_decimal(float(value), date[:4])
                errors["decimal_fee"] = max(
                    errors["decimal_fee"], abs(charged - a["custody_fee"][day])
                )
                invoices.append((schedule[date], charged))
                assessment_rows.append(
                    dict(
                        date=date,
                        payment=schedule[date],
                        base=float(value),
                        fee=charged,
                        economic_long_value=float(
                            np.maximum(economic, 0) @ np.nan_to_num(prior)
                        ),
                    )
                )
            paid = sum(f for due, f in invoices if due == date)
            invoices = [(due, f) for due, f in invoices if due != date]
            liability = sum(f for _, f in invoices)
            errors["payment"] = max(
                errors["payment"], abs(paid - a["custody_payment"][day])
            )
            errors["liability"] = max(
                errors["liability"], abs(liability - a["custody_liability"][day])
            )
            old = a["custody_liability"][day - 1] if day else 0
            errors["liability_rollforward"] = max(
                errors["liability_rollforward"],
                abs(
                    old
                    + a["custody_fee"][day]
                    - a["custody_payment"][day]
                    - a["custody_liability"][day]
                ),
            )
            funding = a["free_cash"][day - 1] if day else case["capital"]
            restricted = (
                a["restricted_cash"][day - 1] + a["hedge_restricted_cash"][day - 1]
                if day
                else 0
            )
            errors["prior_funding"] = max(
                errors["prior_funding"],
                abs(funding - row["funding_cash"]),
                abs(restricted - row["funding_restricted"]),
            )
            errors["income"] = max(
                errors["income"],
                abs(funding * cdi[index] + restricted * cdi[index] - row["interest"]),
            )
        checks.append(
            dict(
                book=case["book"],
                assessments=assessment_rows,
                terminal_liability=liability,
                positive_physical_cells=int(np.sum(a["physical_custody"] > 1e-8)),
            )
        )
    for start in plan["starts"]:
        for capital in plan["capital"]:
            base = books[start, capital, "primary"]
            for scenario in plan["variants"]:
                if scenario == "primary":
                    continue
                candidate = books[start, capital, scenario]
                different = (candidate["custody_fee"] != base["custody_fee"]) | (
                    candidate["custody_payment"] != base["custody_payment"]
                )
                first = int(np.flatnonzero(different)[0])
                np.testing.assert_array_equal(
                    base["nav"][:first], candidate["nav"][:first]
                )
                np.testing.assert_array_equal(
                    base["targets"][: first + 1], candidate["targets"][: first + 1]
                )
                diff = (candidate["nav"] - base["nav"]) / capital * 1e4
                contrasts.append(
                    dict(
                        start=start,
                        capital=capital,
                        scenario=scenario,
                        first_realization=first,
                        final_bps=float(diff[-1]),
                        max_path_bps=float(np.max(np.abs(diff))),
                        fee_difference=float(
                            candidate["custody_fee"].sum() - base["custody_fee"].sum()
                        ),
                    )
                )
    write_json_atomic(out / "checks.json", checks)
    assert len(cases) == 30 and max(errors.values()) < 1e-7, dict(errors)
    report = dict(
        status="qualified_ordinary_custody_not_final_economics",
        audit=binding(root / "manifest.json"),
        checks=binding(out / "checks.json"),
        sessions=1440,
        cells=cells,
        actual_fills=fill_count,
        assessments=assessment_count,
        positive_custody_without_current_print=marked_without_print,
        errors=dict(errors),
        contrasts=contrasts,
        adaptive_max_bps={
            str(cap): max(
                c["adaptive_nav_difference_bps"] for c in cases if c["capital"] == cap
            )
            for cap in plan["capital"]
        },
        max_identical_intention_nav=max(
            c["identical_intention_nav_error"] for c in cases
        ),
        max_adaptive_target=max(c["adaptive_target_difference"] for c in cases),
        seconds=perf_counter() - tick,
        limitation="Ordinary physical custody, explicit0/3/10close-payment and own-close valuation hypotheses; corporate physical base and earlier tariffs not admitted. Prior larger discrete fee uncertainty retained. No model profit.",
    )
    write_json_atomic(out / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "contrasts"}))


if __name__ == "__main__":
    main()
