"""Independent Decimal invoice arithmetic and saved-book NAV qualification."""

from collections import defaultdict
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic, sha256_file
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def D(value):
    return Decimal(str(value))


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    audit = bound_json(run["loan_invoice_audit"])
    root = Path(audit["audit_root"])
    out = root / "qualification_qualified"
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / "executed.py")
    cases = bound_json(audit["completed"])
    assert len(cases) == 36
    books, checks, contrasts, precision_boundaries = {}, [], [], []
    errors = defaultdict(float)
    total_roots = total_invoices = cells = sessions = 0
    for case in cases:
        folder = root / case["book"]
        book = bound_json(binding(folder / "book.json"))
        assert (
            book["status"] == "completed"
            and not book["legacy_slot_diagnostics_applicable"]
        )
        for name, digest in book["files"].items():
            assert sha256_file(folder / name) == digest
        with np.load(folder / "account.npz") as z:
            a = {k: z[k] for k in z.files}
        books[case["start"], case["scenario"], case["capital"]] = a
        assert a["signed_shares"].shape == (64, 933)
        cells += sum(v.size for v in a.values())
        sessions += 64
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
        )
        errors["nav"] = max(errors["nav"], float(np.abs(nav - a["nav"]).max()))
        details = bound_json(binding(folder / "payments.json"))
        charges = pl.read_parquet(folder / "loan_charges.parquet")
        accrual = np.zeros(64)
        for row in charges.iter_rows(named=True):
            accrual[row["session"]] += row["rent"] + row["fee"]
        balance = 0.0
        roots = invoices = partial_minimums = grouped_merges = 0
        cents = case["scenario"] not in ("primary", "minimum_pro_rata")
        rounding = (
            ROUND_CEILING
            if case["scenario"] == "contract_up"
            else ROUND_FLOOR
            if case["scenario"] == "contract_down"
            else ROUND_HALF_UP
        )
        for day, row in enumerate(details):
            funding = a["free_cash"][day - 1] if day else case["capital"]
            restricted = (
                a["restricted_cash"][day - 1] + a["hedge_restricted_cash"][day - 1]
                if day
                else 0.0
            )
            errors["funding"] = max(
                errors["funding"],
                abs(row["funding_cash"] - funding),
                abs(row["funding_restricted"] - restricted),
            )
            liability = (
                row["rent_due"]
                + row["fees_due"]
                + row["minimum"]
                - row["credit"]
                + row["cash_liability"]
            )
            errors["liability"] = max(
                errors["liability"], abs(liability - a["loan_liability"][day])
            )
            errors["credit"] = max(
                errors["credit"], abs(row["credit"] - a["loan_minimum_credit"][day])
            )
            balance += (
                accrual[day]
                + a["loan_invoice_adjustment"][day].sum()
                - a["loan_payment"][day]
            )
            errors["expense_rollforward"] = max(
                errors["expense_rollforward"],
                abs(balance - (a["loan_liability"][day] - row["cash_liability"])),
            )
            errors["paid_account"] = max(
                errors["paid_account"], abs(row["paid"] - a["loan_payment"][day])
            )
            payment = row["payment"]
            if not payment:
                assert row["paid"] == 0
                continue
            raw_groups = []
            for r in payment["roots"]:
                fee, prior_credit, minimum = (
                    D(r["fee_due"]),
                    D(r["credit_before"]),
                    D(r["minimum"]),
                )
                used = min(fee, prior_credit)
                after = prior_credit - used
                if case["scenario"] == "minimum_pro_rata":
                    fraction = (
                        D(r["returned_principal"]) / D(r["principal"])
                        if r["principal"]
                        else D(0)
                    )
                    allocated = max(minimum - after, D(0)) * fraction
                    after = D(0) if r["completed"] else after + allocated
                else:
                    allocated = minimum if r["completed"] else D(0)
                raw_fee = fee - used + allocated
                for key, expected in (
                    ("credit_used", used),
                    ("allocated", allocated),
                    ("credit_after", after),
                    ("raw_fee", raw_fee),
                ):
                    errors["decimal_minimum"] = max(
                        errors["decimal_minimum"], float(abs(expected - D(r[key])))
                    )
                partial_minimums += int(not r["completed"] and allocated > 0)
                raw_groups.append((r["security"], D(r["rent"]), raw_fee))
                roots += 1
            if case["scenario"] == "security_day_nearest":
                groups = defaultdict(lambda: [D(0), D(0)])
                for security, rent, fee in raw_groups:
                    groups[security][0] += rent
                    groups[security][1] += fee
                grouped_merges += len(raw_groups) - len(groups)
                raw_groups = [
                    (security, *values) for security, values in sorted(groups.items())
                ]
            assert len(raw_groups) == len(payment["invoices"])
            paid_total = D(0)
            adjustment = [D(0), D(0)]
            for (security, rent, fee), invoice in zip(raw_groups, payment["invoices"]):
                assert security == invoice["security"]
                for component, raw in enumerate((rent, fee)):
                    field = "rent" if component == 0 else "fee"
                    errors["decimal_grouping"] = max(
                        errors["decimal_grouping"], float(abs(raw - D(invoice[field])))
                    )
                    # Recomposition of separately printed operands can straddle
                    # an exact-cent minimum by ulps. Verify components above, then
                    # independently round the actual Float64 invoice coordinate.
                    coordinate = D(invoice[field])
                    paid = (
                        coordinate.quantize(D(".01"), rounding=rounding)
                        if cents
                        else coordinate
                    )
                    recomposed = (
                        raw.quantize(D(".01"), rounding=rounding) if cents else raw
                    )
                    if abs(recomposed - paid) > D(".000001"):
                        assert abs(raw - coordinate) < D(".000000001")
                        precision_boundaries.append(
                            dict(
                                book=case["book"],
                                day=day,
                                security=security,
                                field=field,
                                recomposed=str(raw),
                                coordinate=str(coordinate),
                                recomposed_paid=str(recomposed),
                                coordinate_paid=str(paid),
                            )
                        )
                    errors["decimal_invoice"] = max(
                        errors["decimal_invoice"],
                        float(abs(paid - D(invoice["paid_" + field]))),
                    )
                    paid_total += paid
                    adjustment[component] += paid - raw
                invoices += 1
            errors["decimal_paid"] = max(
                errors["decimal_paid"], float(abs(paid_total - D(row["paid"])))
            )
            errors["rounding_expense"] = max(
                errors["rounding_expense"],
                max(
                    float(abs(x - D(y)))
                    for x, y in zip(adjustment, a["loan_invoice_adjustment"][day])
                ),
            )
        assert max(errors.values()) < 1e-6, (case["book"], dict(errors))
        if case["start"].startswith("2019") and case["scenario"] == "minimum_pro_rata":
            assert partial_minimums > 0 and a["loan_minimum_credit"].max() > 0
        if case["scenario"] == "security_day_nearest":
            assert grouped_merges > 0
        checks.append(
            dict(
                book=case["book"],
                root_payments=roots,
                invoices=invoices,
                partial_minimum_allocations=partial_minimums,
                merged_invoice_roots=grouped_merges,
                maximum_credit=float(a["loan_minimum_credit"].max()),
            )
        )
        total_roots += roots
        total_invoices += invoices
    for case in cases:
        if case["scenario"] == "primary":
            continue
        a = books[case["start"], case["scenario"], case["capital"]]
        for base in (
            ["primary", "contract_nearest"]
            if case["scenario"] == "security_day_nearest"
            else ["primary"]
        ):
            b = books[case["start"], base, case["capital"]]
            delta = (a["nav"] - b["nav"]) / case["capital"] * 1e4
            # Earlier decisions must agree until the first actual invoice/credit difference.
            changes = np.flatnonzero(
                np.any(
                    a["loan_invoice_adjustment"] != b["loan_invoice_adjustment"], axis=1
                )
                | (a["loan_minimum_credit"] != b["loan_minimum_credit"])
            )
            first = int(changes[0]) if len(changes) else 64
            prefix = (
                float(np.abs(a["nav"][:first] - b["nav"][:first]).max())
                if first
                else 0.0
            )
            errors["prefix"] = max(errors["prefix"], prefix)
            assert prefix < 1e-6
            target_prefix = float(
                np.abs(
                    a["targets"][: min(first + 1, 64)]
                    - b["targets"][: min(first + 1, 64)]
                ).max()
            )
            errors["target_prefix"] = max(errors["target_prefix"], target_prefix)
            contrasts.append(
                dict(
                    start=case["start"],
                    scenario=case["scenario"],
                    base=base,
                    capital=case["capital"],
                    final_bps=float(delta[-1]),
                    max_path_bps=float(np.abs(delta).max()),
                    first_difference_session=first,
                    rent_rounding_brl=float(a["loan_invoice_adjustment"][:, 0].sum()),
                    fee_rounding_brl=float(a["loan_invoice_adjustment"][:, 1].sum()),
                )
            )
    runtime = {
        p.name: binding(p)
        for folder in ("execution", "v2")
        for p in (PROJECT / "research/src/brazil_rv" / folder).glob("*.py")
    }
    for folder in ("execution", "v2"):
        for p in (PROJECT / "research/src/brazil_rv" / folder).glob("*.py"):
            assert sha256_file(p) == sha256_file(root / f"executed_{folder}_{p.name}")
    q = dict(
        status="qualified",
        precision_boundaries=precision_boundaries,
        audit=run["loan_invoice_audit"],
        books=36,
        sessions=sessions,
        account_array_cells=cells,
        root_payments=total_roots,
        invoice_groups=total_invoices,
        errors=dict(errors),
        checks=checks,
        contrasts=contrasts,
        identical_intention_max_brl=max(
            c["identical_intention_nav_error"] for c in cases
        ),
        adaptive_target_max=max(c["adaptive_target_difference"] for c in cases),
        independent_adaptive_max_path_bps={
            str(capital): max(
                c["adaptive_nav_difference_bps"]
                for c in cases
                if c["capital"] == capital
            )
            for capital in (10000000, 1000000, 5000000)
        },
        runtime=runtime,
        seconds=perf_counter() - tick,
    )
    write_json_atomic(out / "manifest.json", q)
    run["loan_invoice_qualification"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {k: v for k, v in q.items() if k not in ("checks", "contrasts", "runtime")}
        )
    )


if __name__ == "__main__":
    main()
