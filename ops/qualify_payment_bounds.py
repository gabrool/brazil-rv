"""Independent saved-book cash, entitlement and original-loan qualification."""

from decimal import Decimal, ROUND_FLOOR
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic, sha256_file
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def D(x):
    return Decimal(str(x))


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    audit = bound_json(run["payment_bounds_audit"])
    root = Path(audit["audit_root"])
    out = root / "qualification"
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / "executed.py")
    cases = bound_json(audit["completed"])
    assert len(cases) == 54
    books, details, charges = {}, {}, {}
    checks, contrasts = [], []
    days = cells = 0
    identity_error = funding_error = claim_error = quantity_error = Decimal(0)
    for case in cases:
        folder = root / case["book"]
        book = bound_json(binding(folder / "book.json"))
        assert (
            book["status"] == "completed"
            and not book["legacy_slot_diagnostics_applicable"]
        )
        for filename, digest in book["files"].items():
            assert sha256_file(folder / filename) == digest
        with np.load(folder / "account.npz") as z:
            a = {k: z[k] for k in z.files}
        key = case["event"], case["scenario"], case["sign"], case["capital"]
        books[key] = a
        snapshots = bound_json(binding(folder / "payments.json"))
        details[key] = snapshots
        charges[key] = pl.read_parquet(folder / "loan_charges.parquet")
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
        error = D(np.max(np.abs(nav - a["nav"])))
        identity_error = max(identity_error, error)
        assert error < D("5e-8")
        days += len(nav)
        cells += sum(v.size for v in a.values())
        assert a["signed_shares"].shape == (case["sessions"], 933)
        for day, row in enumerate(snapshots):
            expected_cash = a["free_cash"][day - 1] if day else case["capital"]
            expected_restricted = (
                a["restricted_cash"][day - 1] + a["hedge_restricted_cash"][day - 1]
                if day
                else 0
            )
            error = max(
                abs(D(row["funding_cash"]) - D(expected_cash)),
                abs(D(row["funding_restricted"]) - D(expected_restricted)),
            )
            assert error < D("5e-8"), (case["book"], day, error)
            funding_error = max(funding_error, error)
        name, recognition, payment = (
            case["source"],
            case["recognition"],
            case["payment"],
        )
        terms = bound_json(
            binding(root / f"terms_{case['event']}_{case['scenario']}.json")
        )
        fills = pl.read_parquet(folder / "fills.parquet").filter(
            pl.col("security_index") == name
        )
        assert not fills.filter(pl.col("fill_session") >= 6).height
        if case["event"] == "CIEL":
            term = next(
                e for e in terms["loan_cash_settlements"] if e["isin"] == "BRCIELACNOR3"
            )
            quantity = sum(
                (D(c["quantity"]) for c in snapshots[6]["source_loans"]), D(0)
            )
            cash = quantity * D(term["cash_per_share"])
            error = abs(cash - D(snapshots[6]["loan_cash_settlement_payment"]))
            assert error < D("5e-8"), (case["book"], error)
            claim_error = max(claim_error, error)
            checks.append(
                dict(
                    book=case["book"],
                    signed_loan_quantity=str(-quantity),
                    decimal_loan_cash=str(cash),
                    realization_session=6,
                )
            )
            continue
        event = next(
            e
            for e in terms["share_distributions"]
            if e["isin"]
            == ("BRBRMLACNOR9" if case["event"] == "BRML" else "BRDMMOACNOR0")
        )
        leg = event["legs"][0]
        ratio = D(leg["shares_per_prior_share"])
        auction = leg["fractional_auction"]
        original = D(a["signed_shares"][5, name])
        if case["sign"] > 0:
            entitlement = original * ratio
            fraction = entitlement - entitlement.to_integral_value(rounding=ROUND_FLOOR)
        elif auction.get("provision_loan_fractions", False):
            cohorts = []
            for fill in fills.sort("fill_session").to_dicts():
                q = D(fill["quantity"])
                if fill["side"] == "sell":
                    cohorts.append(q)
                else:
                    total = sum(cohorts, D(0))
                    cohorts = [c * (1 - q / total) for c in cohorts]
            assert abs(sum(cohorts, D(0)) + original) < D("1e-8")
            fraction = -sum(
                (
                    c * ratio - (c * ratio).to_integral_value(rounding=ROUND_FLOOR)
                    for c in cohorts
                ),
                D(0),
            )
        else:
            fraction = D(0)
        amount = fraction * D(auction["cash_per_share"])
        error = abs(
            D(snapshots[recognition]["prepared_source_quantity"]) * ratio - fraction
        )
        assert error < D("1e-8"), (case["book"], error)
        quantity_error = max(quantity_error, error)
        dates = np.load(terms["calendar"]["path"])
        first = np.searchsorted(dates, np.datetime64(case["dates"][0]))
        cash_pay = (
            int(np.searchsorted(dates, np.datetime64(event["payment_date"]))) - first
        )
        ordinary_cash = original * D(event["cash_per_prior_share"])
        for day, row in enumerate(snapshots):
            expected = ordinary_cash if 6 <= day < cash_pay else D(0)
            if recognition <= day < payment:
                expected += amount
            error = abs(D(row["source_claim"]) - expected)
            assert error < D("5e-8"), (case["book"], day, error)
            claim_error = max(claim_error, error)
            if day >= recognition:
                assert row["source_quantity"] == 0
            if day >= payment:
                assert abs(row["source_restricted"]) < 1e-8
        checks.append(
            dict(
                book=case["book"],
                decimal_original_quantity=str(original),
                decimal_residual=str(fraction),
                decimal_fraction_cash=str(amount),
                decimal_ordinary_cash=str(ordinary_cash),
                ordinary_payment=int(cash_pay),
                auction_recognition=recognition,
                fraction_payment=payment,
            )
        )
    tiny_checks = []
    for case in cases:
        label, scenario, sign, capital = (
            case["event"],
            case["scenario"],
            case["sign"],
            case["capital"],
        )
        if scenario == "primary":
            continue
        key, base = (label, scenario, sign, capital), (label, "primary", sign, capital)
        a, b = books[key], books[base]
        prefix = 8 if scenario == "tiny_rent" else case["recognition"]
        np.testing.assert_array_equal(a["nav"][:prefix], b["nav"][:prefix])
        np.testing.assert_array_equal(
            a["targets"][: prefix + 1], b["targets"][: prefix + 1]
        )
        equality = (
            (label == "CIEL" and sign == 1)
            or (label == "BRML" and sign == -1)
            or (scenario == "tiny_rent" and sign == 1)
        )
        if equality:
            for field in a:
                np.testing.assert_array_equal(a[field], b[field])
        delta = (a["nav"] - b["nav"]) / capital * 1e4
        contrasts.append(
            dict(
                event=label,
                scenario=scenario,
                sign=sign,
                capital=capital,
                final_bps=float(delta[-1]),
                max_path_bps=float(np.max(np.abs(delta))),
                exact_prefix=prefix,
                all_fields_equal=equality,
            )
        )
        if scenario == "tiny_rent" and sign < 0:
            tiny = [c for c in details[key][8]["source_loans"] if c["quantity"] == 0]
            expected = sum(
                (
                    D(c["principal"])
                    * (
                        (1 + D(c["rate"])) ** (D(case["payment"] - c["opened"]) / 252)
                        - (1 + D(c["rate"])) ** (D(7 - c["opened"]) / 252)
                    )
                    for c in tiny
                ),
                D(0),
            )
            assert all(
                c["value_lag"] == 1 and c["return_day"] == case["payment"] for c in tiny
            )
            actual = D(
                charges[key]
                .filter(pl.col("security_index") == case["source"])["rent"]
                .sum()
            ) - D(
                charges[base]
                .filter(pl.col("security_index") == case["source"])["rent"]
                .sum()
            )
            assert abs(expected - actual) < D("1e-7"), (key, expected, actual)
            tiny_checks.append(
                dict(
                    capital=capital,
                    original_tiny_loans=len(tiny),
                    decimal_additional_rent=str(expected),
                    actual_additional_rent=str(actual),
                )
            )
    result = dict(
        status="payment_sweep_precision_engineering_qualified_final_A_pending",
        audit=run["payment_bounds_audit"],
        books=len(cases),
        sessions=days,
        account_array_cells=cells,
        independent_nav_identity_max_brl=float(identity_error),
        independent_prior_funding_max_brl=float(funding_error),
        decimal_cash_max_error=str(claim_error),
        decimal_fraction_max_error=str(quantity_error),
        checks=checks,
        tiny_loan_checks=tiny_checks,
        contrasts=contrasts,
        identical_intention_max_brl=max(
            c["identical_intention_nav_error"] for c in cases
        ),
        independent_adaptive_max_path_bps={
            str(capital): max(
                c["adaptive_nav_difference_bps"]
                for c in cases
                if c["capital"] == capital
            )
            for capital in (10000000, 1000000, 5000000)
        },
        adaptive_target_max=max(c["adaptive_target_difference"] for c in cases),
        seconds=perf_counter() - tick,
    )
    write_json_atomic(out / "manifest.json", result)
    run["payment_bounds_qualification"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("checks", "contrasts")}
        )
    )
    print(json.dumps([c for c in contrasts if c["capital"] == 10000000]))


if __name__ == "__main__":
    main()
