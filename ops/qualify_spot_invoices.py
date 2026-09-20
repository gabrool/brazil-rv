"""Independent Decimal invoice and saved cash checks; no book replay."""

from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN
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
    tick = perf_counter()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    root = Path(run["root"]) / "spot_invoice"
    out = root / "qualification"
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / "executed.py")
    audit = bound_json(binding(root / "manifest.json"))
    cases = bound_json(audit["completed"])
    tariffs = {
        r["date"]: r for r in bound_json(run["historical_cost_sources"])["calendar"]
    }
    terms = bound_json(run["enat_settlement_terms"])
    dates = np.load(Path(terms["store"]["root"]) / "date_index.npy").astype(
        "datetime64[D]"
    )
    with np.load(bound_json(run["cash_calendar"])["panel"]["path"]) as z:
        cdi = z["cdi_returns"]
    errors = defaultdict(float)
    cells = fills_count = groups_count = days = 0
    checks = []

    def error(name, expected, actual):
        errors[name] = max(
            errors[name],
            float(np.max(np.abs(np.asarray(expected) - actual), initial=0)),
        )

    for case in cases:
        folder = root / case["book"]
        meta = json.loads((folder / "book.json").read_text())
        for file, digest in meta["files"].items():
            assert sha256_file(folder / file) == digest
        assert not meta["legacy_slot_diagnostics_applicable"]
        cfg = meta["provenance"]["config"]
        with np.load(folder / "account.npz") as z:
            a = {k: z[k] for k in z.files}
        cells += sum(x.size for x in a.values())
        rows = json.loads((folder / "funding_and_costs.json").read_text())
        fills = pl.read_parquet(folder / "fills.parquet").to_dicts()
        fills_count += len(fills)
        by_day = defaultdict(list)
        for f in fills:
            by_day[f["fill_session"]].append(f)
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
        error("saved_nav", nav, a["nav"])
        pending = []
        book_checks = []
        for day, row in enumerate(rows):
            date = meta["state_dates"][day]
            index = int(np.searchsorted(dates, np.datetime64(date)))
            trading = tariffs[date]["trading_bps"]
            rates = [
                Decimal(
                    str(
                        cfg["unrecovered_spot_trading_bps"]
                        if trading is None
                        else trading
                    )
                ),
                Decimal("2.75" if date < "2021-02-02" else "2.5"),
            ]
            groups = defaultdict(lambda: Decimal(0))
            directions = defaultdict(set)
            flow = cost = Decimal(0)
            for f in by_day[day]:
                key = f["security"]
                amount = Decimal(str(f["gross_notional"]))
                groups[key] += amount
                directions[key].add(f["side"])
                cost += Decimal(str(f["cost"]))
                flow += amount * (1 if f["side"] == "sell" else -1) - Decimal(
                    str(f["cost"])
                )
            assert all(len(v) == 1 for v in directions.values())
            groups_count += len(groups)
            invoiced = [
                sum(
                    (
                        (
                            v.quantize(Decimal(".000001"), rounding=ROUND_HALF_UP)
                            * rate
                            / 10000
                        ).quantize(Decimal(".000001"), rounding=ROUND_HALF_UP)
                        for v in groups.values()
                    ),
                    Decimal(0),
                ).quantize(Decimal(".01"), rounding=ROUND_DOWN)
                for rate in rates
            ]
            total = sum(groups.values(), Decimal(0))
            raw = [total * rate / 10000 for rate in rates]
            adjustment = [float(v - u) for v, u in zip(invoiced, raw)]
            # All actual invoiced category coordinates must equal Decimal cents.
            error(
                "decimal_b3_invoice",
                [float(v) for v in invoiced],
                a["execution_charges"][day, 1:3],
            )
            error("decimal_adjustment", adjustment, a["spot_invoice_adjustment"][day])
            error(
                "independent_account_adjustment",
                a["spot_invoice_adjustment"][day],
                np.array(row["adjustment"]),
            )
            error(
                "independent_account_components",
                a["execution_charges"][day],
                np.array(row["components"]),
            )
            components = [
                0,
                *map(float, invoiced),
                float(total * Decimal(str(cfg["execution_brokerage_bps"])) / 10000),
                float(total * Decimal(str(cfg["execution_shortfall_bps"])) / 10000),
            ]
            error("decimal_all_components", components, a["execution_charges"][day])
            error(
                "fill_cost_plus_adjustment", float(cost) + sum(adjustment), row["cost"]
            )
            error("cost_components", a["execution_charges"][day].sum(), row["cost"])
            # Independent net spot queue from actual fills. Internal free/restricted
            # corporate transfers sum to zero, and do not create spot invoices.
            due = day + (3 if date < "2019-05-27" else 2)
            pending.append(
                (due, flow - sum((v - u for v, u in zip(invoiced, raw)), Decimal(0)))
            )
            pending = [(d, v) for d, v in pending if d > day]
            error(
                "independent_spot_unsettled",
                float(sum((v for _, v in pending), Decimal(0))),
                a["unsettled_cash"][day],
            )
            cash = a["free_cash"][day - 1] if day else case["capital"]
            restricted = (
                a["restricted_cash"][day - 1] + a["hedge_restricted_cash"][day - 1]
                if day
                else 0
            )
            error(
                "prior_funding",
                [cash, restricted],
                np.array([row["funding_cash"], row["funding_restricted"]]),
            )
            error("income", (cash + restricted) * cdi[index], row["interest"])
            book_checks.append(
                dict(
                    date=date,
                    security_groups=len(groups),
                    trading=str(invoiced[0]),
                    clearing=str(invoiced[1]),
                    adjustment=adjustment,
                    unsettled=float(sum((v for _, v in pending), Decimal(0))),
                )
            )
            days += 1
        with np.load(case["parent"]["account"]["path"]) as old:
            first = case["first_difference"]
            np.testing.assert_array_equal(a["nav"][:first], old["nav"][:first])
            np.testing.assert_array_equal(
                a["targets"][: first + 1], old["targets"][: first + 1]
            )
        checks.append(
            dict(
                book=case["book"],
                days=book_checks,
                terminal_pending_spot=float(sum((v for _, v in pending), Decimal(0))),
            )
        )
    report = dict(
        status="qualified_spot_invoice_hypothesis_not_final_stage_a",
        audit=binding(root / "manifest.json"),
        books=len(cases),
        sessions=days,
        cells=cells,
        actual_fills=fills_count,
        security_day_groups=groups_count,
        errors=dict(errors),
        max_identical_intention_nav=max(
            c["identical_intention_nav_error"] for c in cases
        ),
        adaptive_max_bps={
            str(cap): max(
                c["adaptive_nav_difference_bps"] for c in cases if c["capital"] == cap
            )
            for cap in (10000000, 1000000, 5000000)
        },
        max_adaptive_target=max(c["adaptive_target_difference"] for c in cases),
        contrasts=[
            {
                k: c[k]
                for k in (
                    "book",
                    "capital",
                    "final_contrast_bps",
                    "max_contrast_bps",
                    "direct_invoice_adjustment",
                )
            }
            for c in cases
        ],
        cielo_exposed_books=[c["book"] for c in cases if c["cielo_loan_cash_quantity"]],
        limitation="Fractional research units, one normal cash-market account/phase and pre017/2023 backcast are hypotheses, not exact observed client invoices. No opposing same-ISIN/day fills exposed; existing rejection retained. All older larger adaptive fixed-fee uncertainty retained. Total synthetic path contrasts, not model profits or all-interior bounds.",
        seconds=perf_counter() - tick,
    )
    write_json_atomic(out / "checks.json", checks)
    write_json_atomic(out / "numerical_report.json", report)
    assert max(errors.values()) < 1e-7, dict(errors)
    report["checks"] = binding(out / "checks.json")
    write_json_atomic(out / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "contrasts"}))


if __name__ == "__main__":
    main()
