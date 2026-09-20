"""Qualify saved corporate custody books without replaying either account."""

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
from qualify_tariff_coverage import fee_decimal

PROJECT = Path(__file__).resolve().parents[1]


def main():
    tick = perf_counter()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    root = Path(run["root"]) / "tariff_coverage/corporate_books"
    out = root / "saved_qualification"
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / "executed.py")
    shutil.copyfile(PROJECT / "ops/qualify_tariff_coverage.py", out / "fee_oracle.py")
    audit = bound_json(binding(root / "manifest.json"))
    cases, plan = bound_json(audit["completed"]), bound_json(audit["plan"])
    terms = bound_json(plan["dependencies"]["enat_settlement_terms"])
    store = Path(terms["store"]["root"])
    dates = np.load(store / "date_index.npy").astype("datetime64[D]")
    isins = np.load(store / "isin_index.npy").tolist()
    raw = np.load(store / "raw_close.npy", mmap_mode="r")
    qs = np.load(store / "action_shares_per_prior_share.npy", mmap_mode="r")
    resolved = np.load(store / "action_session_resolved.npy", mmap_mode="r")
    tariffs = {
        r["date"]: r for r in bound_json(plan["source_qualification"])["calendar"]
    }
    with np.load(bound_json(run["cash_calendar"])["panel"]["path"]) as z:
        cdi = z["cdi_returns"]
    errors = defaultdict(float)
    arrays, checks, contrasts = {}, [], []
    cells = fills_count = assessments = ordinary_flow_cells = entitlement_checks = 0

    def error(name, expected, actual):
        errors[name] = max(
            errors[name], float(np.max(np.abs(np.asarray(expected) - actual), initial=0))
        )

    for case in cases:
        folder = root / case["book"]
        book = json.loads((folder / "book.json").read_text())
        assert not book["legacy_slot_diagnostics_applicable"]
        for file, digest in book["files"].items():
            assert sha256_file(folder / file) == digest
        with np.load(folder / "account.npz") as z:
            a = {k: z[k] for k in z.files}
        cells += sum(x.size for x in a.values())
        key = case["start"], case["sign"], case["capital"], case["scenario"]
        arrays[key] = a
        cfg = book["provenance"]["config"]
        rows = json.loads((folder / "funding_and_costs.json").read_text())
        fills = pl.read_parquet(folder / "fills.parquet").to_dicts()
        fills_count += len(fills)
        by_day = defaultdict(list)
        signed_fills = np.zeros((case["sessions"], 934))
        for f in fills:
            by_day[f["fill_session"]].append(f)
            name = 933 if f["purpose"] == "hedge" else f["security_index"]
            signed_fills[f["fill_session"], name] += f["quantity"] * (
                1 if f["side"] == "buy" else -1
            )
        first = int(np.searchsorted(dates, np.datetime64(case["start"])))
        last = first + case["sessions"]
        shares = np.column_stack((a["signed_shares"], a["hedge_signed_shares"]))
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
        error("saved_nav_identity", nav, a["nav"])

        # Physical-flow oracle on every noncorporate axis, independent of saved
        # pending queues and loan cohorts. Corporate axes get separate sourced
        # entitlement and saved-component checks below; do not label these alike.
        ordinary = np.ones(934, dtype=bool)
        events = []
        for event in terms["share_distributions"]:
            effect = (
                int(np.searchsorted(dates, np.datetime64(event["effective_date"])))
                - first
            )
            if not 0 <= effect < case["sessions"]:
                continue
            source = isins.index(event["isin"])
            ordinary[source] = False
            for leg in event["legs"]:
                ordinary[isins.index(leg["successor_isin"])] = False
            events.append((event, source, effect))
        for event in terms["scalar_actions"] + terms["cash_cancellations"]:
            date = event.get("effective_date", event.get("recognition_date"))
            if dates[first] <= np.datetime64(date) <= dates[last - 1]:
                ordinary[isins.index(event["isin"])] = False
        for event in terms["loan_cash_settlements"]:
            if (
                dates[first]
                <= np.datetime64(event["settlement_date"])
                <= dates[last - 1]
            ):
                ordinary[isins.index(event["isin"])] = False
        economic, physical = np.zeros(934), np.zeros(934)
        flows = defaultdict(lambda: np.zeros(934))
        schedule = {r["date"]: r["payment_date"] for r in cfg["custody_assessments"]}
        invoices, monthly = [], []
        claims = (
            json.loads((folder / "share_claim_positions.json").read_text())
            if (folder / "share_claim_positions.json").exists()
            else []
        )
        claim_by_day = defaultdict(list)
        for claim in claims:
            claim_by_day[claim["session"]].append(claim)
        for day, row in enumerate(rows):
            index, date = first + day, str(dates[first + day])
            q = np.r_[np.where(resolved[index], qs[index], 1), 1].astype(float)
            q[~ordinary] = 1
            assert np.all(q > 0)
            economic *= q
            physical *= q
            for due in flows:
                flows[due] *= q
            physical += flows.pop(day, np.zeros(934))
            cost = [Decimal(0)] * 5
            directions = defaultdict(set)
            for f in by_day[day]:
                name = 933 if f["purpose"] == "hedge" else f["security_index"]
                amount = f["quantity"] * (1 if f["side"] == "buy" else -1)
                cover = min(max(amount, 0), max(-economic[name], 0))
                opened = max(-amount - max(economic[name], 0), 0)
                economic[name] += amount
                flows[day + 2][name] += amount - cover
                if date < "2020-10-26":
                    physical[name] += opened
                else:
                    flows[day + 1][name] += opened
                directions[name].add(f["side"])
                tr = tariffs[date]["trading_bps"]
                rates = [
                    0,
                    Decimal(
                        str(cfg["unrecovered_spot_trading_bps"] if tr is None else tr)
                    ),
                    Decimal("2.75") if date < "2021-02-02" else Decimal("2.5"),
                    0,
                    Decimal(1),
                ]
                cost = [
                    old + Decimal(str(f["gross_notional"])) * rate / 10000
                    for old, rate in zip(cost, rates)
                ]
            assert all(len(v) == 1 for v in directions.values())
            error(
                "decimal_execution",
                [float(x) for x in cost],
                a["execution_charges"][day],
            )
            error("ordinary_share_flow", economic[ordinary], shares[day, ordinary])
            error(
                "independent_ordinary_physical_flow",
                physical[ordinary],
                a["physical_custody"][day, ordinary],
            )
            ordinary_flow_cells += int(ordinary.sum())
            error("identical_intention_physical", 0, row["physical_error"])
            pending = sum(
                (np.array(p["quantity"]) for p in row["pending"]), np.zeros(934)
            )
            bonus = sum(
                (
                    np.array(p["quantity"])
                    for p in row["corporate_pending"]
                    if p["start"] <= day
                ),
                np.zeros(934),
            )
            gross = shares[day] + np.array(row["loan_quantity"]) - pending
            expected_physical = gross - bonus
            expected_rights = np.maximum(bonus, 0)
            for name in row["claim_names"]:
                expected_rights[name] = max(expected_physical[name], 0)
                expected_physical[name] = 0
            error(
                "saved_corporate_component_identity",
                np.maximum(expected_physical, 0),
                a["physical_custody"][day],
            )
            # On source axes without a pending purchase or loan, positive saved
            # successor entitlements independently value the entire right.
            claim_value = Decimal(0)
            claim_sources = {c["source_index"] for c in claim_by_day[day]}
            for source in claim_sources:
                if row["loan_quantity"][source] == 0 and pending[source] == 0:
                    value = sum(
                        (
                            Decimal(str(max(c["signed_quantity"], 0)))
                            * Decimal(str(c["mark"]))
                            for c in claim_by_day[day]
                            if c["source_index"] == source
                        ),
                        Decimal(0),
                    )
                    source_value = Decimal(str(expected_rights[source])) * Decimal(
                        str(row["custody_marks"][source])
                    )
                    error(
                        "successor_right_valuation", float(value), float(source_value)
                    )
                    claim_value += value
            if date in schedule:
                assessments += 1
                assert dates[index].astype("datetime64[M]") != dates[index + 1].astype(
                    "datetime64[M]"
                )
                marks = np.asarray(row["custody_marks"])
                # Original current prints independently qualify valuation where
                # present. Carried local claim marks retain the V33 hypothesis.
                source_prices = raw[index].astype(float)
                printed = (
                    np.isfinite(source_prices)
                    & (source_prices > 0)
                    & (a["physical_custody"][day, :933] > 1e-8)
                )
                error(
                    "physical_current_print",
                    source_prices[printed],
                    marks[:933][printed],
                )
                physical_value = float(
                    sum(
                        (
                            Decimal(str(float(qty))) * Decimal(str(float(price)))
                            for qty, price in zip(a["physical_custody"][day], marks)
                            if qty > 0
                        ),
                        Decimal(0),
                    )
                )
                rights_value = float(
                    sum(
                        (
                            Decimal(str(float(qty))) * Decimal(str(float(price)))
                            for qty, price in zip(expected_rights, marks)
                            if qty > 0
                        ),
                        Decimal(0),
                    )
                )
                error(
                    "decimal_physical_base",
                    physical_value,
                    a["custody_physical_base"][day],
                )
                error("decimal_right_base", rights_value, a["custody_claim_base"][day])
                base = physical_value + rights_value * cfg["custody_claim_fraction"]
                error("base_composition", base, a["custody_base"][day])
                fee, maintenance = fee_decimal(base, date)
                error("decimal_custody_fee", fee, a["custody_fee"][day])
                error("decimal_maintenance", maintenance, a["custody_maintenance"][day])
                invoices.append((schedule[date], fee))
                monthly.append(
                    dict(
                        date=date,
                        physical_base=physical_value,
                        rights_base=rights_value,
                        independently_valued_unencumbered_rights=float(claim_value),
                        fee=fee,
                    )
                )
            paid = sum(f for d, f in invoices if d == date)
            invoices = [(d, f) for d, f in invoices if d != date]
            error("custody_payment", paid, a["custody_payment"][day])
            error(
                "custody_liability",
                sum(f for _, f in invoices),
                a["custody_liability"][day],
            )
            old_liability = a["custody_liability"][day - 1] if day else 0
            error(
                "custody_rollforward",
                old_liability + a["custody_fee"][day] - a["custody_payment"][day],
                a["custody_liability"][day],
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
                [row["funding_cash"], row["funding_restricted"]],
            )
            error("income", (cash + restricted) * cdi[index], row["interest"])

        event_checks = []
        for event, source, effect in events:
            source_fills = [
                f
                for f in fills
                if f["purpose"] != "hedge"
                and f["security_index"] == source
                and f["fill_session"] >= effect
            ]
            assert not source_fills
            original = Decimal(str(float(a["signed_shares"][effect - 1, source])))
            held = float(original)
            for leg in event["legs"]:
                target = isins.index(leg["successor_isin"])
                delivery = (
                    None
                    if leg["delivery_date"] is None
                    else int(
                        np.searchsorted(dates, np.datetime64(leg["delivery_date"]))
                    )
                    - first
                )
                ratio = Decimal(str(leg["shares_per_prior_share"]))
                before = [
                    c
                    for c in claims
                    if c["source_index"] == source
                    and c["successor_index"] == target
                    and (delivery is None or c["session"] < delivery)
                ]
                for c in before:
                    error(
                        "decimal_original_entitlements",
                        float(original * ratio),
                        c["signed_quantity"],
                    )
                    entitlement_checks += 1
                if (
                    delivery is not None
                    and 0 < delivery < case["sessions"]
                    and held > 0
                ):
                    incoming = original * ratio
                    expected = (
                        float(incoming // 1)
                        if leg.get("fractional_auction")
                        else float(incoming)
                    )
                    # Current successor q/cash actions are separately excluded from this
                    # narrow delivery conservation check, not assigned invented units.
                    assert qs[first + delivery, target] == 1
                    arrived = (
                        shares[delivery, target]
                        - shares[delivery - 1, target]
                        - signed_fills[delivery, target]
                    )
                    error("decimal_long_delivery_plus_fills", expected, arrived)
                    entitlement_checks += 1
                if delivery is None:
                    assert np.all(a["physical_custody"][effect:, source] == 0)
                    error(
                        "unknown_custody_original_units_retained",
                        float(original),
                        a["signed_shares"][effect:, source],
                    )
                event_checks.append(
                    dict(
                        source=event["isin"],
                        successor=leg["successor_isin"],
                        original_units=held,
                        delivery_session=delivery,
                        precredit_checks=len(before),
                    )
                )
        payments = json.loads((folder / "loan_cash_payments.json").read_text())
        checks.append(
            dict(
                book=case["book"],
                monthly=monthly,
                events=event_checks,
                loan_cash_payments=payments,
            )
        )

    for case in cases:
        if case["scenario"] != "exclude_rights":
            continue
        key = case["start"], case["sign"], case["capital"]
        base, candidate = arrays[*key, "primary"], arrays[*key, "exclude_rights"]
        changed = np.flatnonzero(base["custody_fee"] != candidate["custody_fee"])
        if len(changed):
            first = int(changed[0])
            np.testing.assert_array_equal(base["nav"][:first], candidate["nav"][:first])
            np.testing.assert_array_equal(
                base["targets"][: first + 1], candidate["targets"][: first + 1]
            )
        else:
            np.testing.assert_array_equal(base["nav"], candidate["nav"])
            np.testing.assert_array_equal(base["targets"], candidate["targets"])
        diff = (candidate["nav"] - base["nav"]) / case["capital"] * 1e4
        contrasts.append(
            dict(
                start=case["start"],
                sign=case["sign"],
                capital=case["capital"],
                final_bps=float(diff[-1]),
                max_path_bps=float(np.max(np.abs(diff))),
                total_fee_difference=float(
                    candidate["custody_fee"].sum() - base["custody_fee"].sum()
                ),
            )
        )
    report = dict(
        status="saved_corporate_custody_interactions_qualified_not_final_stage_a",
        audit=binding(root / "manifest.json"),
        books=len(cases),
        skipped=bound_json(audit["skipped"]),
        sessions=sum(c["sessions"] for c in cases),
        cells=cells,
        fills=fills_count,
        assessments=assessments,
        ordinary_physical_flow_cells=ordinary_flow_cells,
        source_entitlement_checks=entitlement_checks,
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
        cielo_cash_loan_exposed_books=[
            c["book"] for c in cases if c["cielo_loan_cash_quantity"]
        ],
        limitation="Corporate saved component reconciliation relies on the separately qualified loan subledger; the independent fill-only physical oracle covers noncorporate axes. Source entitlement/delivery and rights valuation checks are separate. Seventeen corporate fixtures cover signed pending flows, bonus, offsets, fractions, multiple legs, cash closeout and independent training copies. Valuation and rights fee inclusion are explicit hypotheses, not observed invoices. Preserve larger prior adaptive uncertainty. Synthetic paths are not model profit. An empty Cielo exposure list is not a held-loan cent bound.",
        seconds=perf_counter() - tick,
    )
    write_json_atomic(out / "checks.json", checks)
    write_json_atomic(out / "numerical_report.json", report)
    assert max(errors.values()) < 1e-7, dict(errors)
    report["checks"] = binding(out / "checks.json")
    write_json_atomic(out / "report.json", report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("contrasts", "skipped")}
        )
    )


if __name__ == "__main__":
    main()
