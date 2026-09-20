"""Qualify saved ENAT accounts and newly recovered ALSC auction evidence."""

from datetime import datetime
from decimal import Decimal, ROUND_FLOOR
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic, sha256_file
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.decision_clock import normalize_publication_time

PROJECT = Path(__file__).resolve().parents[1]


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    audit = bound_json(run["enat_settlement_audit"])
    root = Path(audit["audit_root"])
    out = root / "qualification"
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / "executed.py")
    cases = bound_json(audit["completed"])
    assert len(cases) == 18
    name, effect, credit = (
        audit["name"],
        audit["effect"],
        audit["credit"],
    )
    books = {}
    checks = []
    identity_error = 0.0
    cells = days = 0
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
        books[case["scenario"], case["sign"], case["capital"]] = a
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
        error = float(np.max(np.abs(nav - a["nav"])))
        assert error < 5e-8
        identity_error = max(identity_error, error)
        days += len(nav)
        cells += sum(x.size for x in a.values())
        assert a["signed_shares"].shape == (30, 933)
        fills = pl.read_parquet(folder / "fills.parquet").filter(
            pl.col("security_index") == name
        )
        assert not fills.filter(pl.col("fill_session") >= effect).height
        if case["scenario"] == "unresolved_source":
            continue
        claims = json.loads(
            (folder / "share_claim_positions.json").read_text(encoding="utf8")
        )
        own = [c for c in claims if c["source_index"] == name]
        q0 = Decimal(str(a["signed_shares"][effect - 1, name]))
        ratio = Decimal(".805012676")
        expected = q0 * ratio
        before = [c for c in own if c["session"] < credit]
        assert [c["session"] for c in before] == list(range(effect, credit))
        assert all(
            abs(Decimal(str(c["signed_quantity"])) - expected) < Decimal("1e-8")
            and c["delivery_session"] == credit
            for c in before
        )
        if case["sign"] > 0:
            residual = expected - expected.to_integral_value(rounding=ROUND_FLOOR)
        elif case["scenario"] == "provisioned_loan":
            # Independent Decimal original cohorts; buy fills consume each live
            # original pro rata, as registered. No subledger implementation reuse.
            cohorts = []
            for f in fills.sort("fill_session").to_dicts():
                quantity = Decimal(str(f["quantity"]))
                if f["side"] == "sell":
                    cohorts.append(quantity)
                else:
                    total = sum(cohorts, Decimal(0))
                    fraction = quantity / total
                    cohorts = [q * (1 - fraction) for q in cohorts]
            assert abs(sum(cohorts) + q0) < Decimal("1e-8")
            residual = -sum(
                (
                    q * ratio - (q * ratio).to_integral_value(rounding=ROUND_FLOOR)
                    for q in cohorts
                ),
                Decimal(0),
            )
        else:
            residual = Decimal(0)
        after = [c for c in own if c["session"] >= credit]
        assert all(
            c["delivery_session"] is None
            and abs(Decimal(str(c["signed_quantity"])) - residual) < Decimal("1e-8")
            for c in after
        )
        assert len(after) == (len(nav) - credit if residual else 0)
        assert abs(a["signed_shares"][-1, name] * float(ratio) - float(residual)) < 1e-8
        checks.append(
            dict(
                book=case["book"],
                prior_shares=str(q0),
                decimal_entitlement=str(expected),
                decimal_residual=str(residual),
                first_custody=credit,
                residual_days=len(after),
                final_claim_value=float(a["undelivered_share_notional"][-1]),
            )
        )
    contrasts = []
    for sign in (1, -1):
        for capital in (10_000_000, 1_000_000, 5_000_000):
            pairs = [
                ("source", "primary", "unresolved_source", effect),
                ("loan_fraction", "provisioned_loan", "primary", credit),
            ]
            for label, lhs, rhs, prefix in pairs:
                a, b = books[lhs, sign, capital], books[rhs, sign, capital]
                np.testing.assert_array_equal(a["nav"][:prefix], b["nav"][:prefix])
                np.testing.assert_array_equal(
                    a["targets"][:prefix], b["targets"][:prefix]
                )
                if label == "loan_fraction" and sign == 1:
                    np.testing.assert_array_equal(a["nav"], b["nav"])
                delta = (a["nav"] - b["nav"]) / capital * 1e4
                contrasts.append(
                    dict(
                        contrast=label,
                        lhs=lhs,
                        rhs=rhs,
                        sign=sign,
                        capital=capital,
                        final_bps=float(delta[-1]),
                        max_absolute_path_bps=float(np.max(np.abs(delta))),
                        identical_prefix_sessions=prefix,
                    )
                )
    source_root = Path(run["root"]) / "remaining_held_sources"
    row = next(
        x
        for x in bound_json(binding(source_root / "allos_2020_index.json"))["rows"]
        if x["protocol"] == "732740"
    )
    text_path = source_root / "732740.txt"
    text = text_path.read_text(encoding="utf8")
    for phrase in [
        "54,26688776859",
        "2.076",
        "15 de",
        "7 (sete) dias úteis",
        "líquidos de taxas",
    ]:
        assert phrase in text
    stamp = normalize_publication_time(
        datetime.strptime(row["receipt"][9:], "%d/%m/%Y %H:%M"),
        naive_timezone="America/Sao_Paulo",
        precision="minute",
    ).isoformat()
    dates = np.load(
        Path(bound_json(audit["terms"])["store"]["root"]) / "date_index.npy"
    )
    deadline = str(dates[np.searchsorted(dates, np.datetime64("2020-01-30")) + 7])
    assert deadline == "2020-02-10"
    source = dict(
        enat_sources=bound_json(audit["plan"])["original_sources"],
        enat_index=binding(source_root / "enat_index.json"),
        allos_2020_index=binding(source_root / "allos_2020_index.json"),
        new_original_pdfs=5,
        visually_reviewed_pages={
            "1260409": [1, 2],
            "1264467": [1, 2],
            "1265227": [1, 2],
            "1263891": [1],
            "732740": [1],
        },
        render_disposition="Poppler warnings about optional display fonts; all listed original pages rendered legibly and visually matched extracted numerical/date terms.",
        alsc_auction=dict(
            protocol="732740",
            pdf=binding(source_root / "732740.pdf"),
            text=binding(text_path),
            receipt=binding(source_root / "732740_receipt.json"),
            auction_date="2020-01-15",
            available_at=stamp,
            available_decision="2020-01-31",
            shares=2076,
            printed_per_share="54.26688776859",
            price_semantics="Auction proceeds stated; distribution net of unspecified fees. Do not silently treat printed proceeds as exact net investor cash.",
            payment_deadline_business_days=7,
            derived_deadline=deadline,
            deadline_convention="Seven following accepted B3 sessions after Jan30 notice, not observed payment receipt.",
            physical_share_credit=None,
            account_admission="No ALSC books changed. Whole claim stays locked while physical credit is unknown; exact net fraction proceeds remain unresolved.",
        ),
        search_scope="Prior216 NATU/ALSC/SOMA receipts reused; new153 ENAT/3R 2024 and41 issuer22357 2020 records selected from existing original RAD responses. Current issuer labels are not historical legal identity. Search engines initially surfaced unrelated 2023 BRML fractions; not admitted.",
        failed_attempt="2020 issuer-selection local variable shadowed output list; failed before index/PDF output. Exact failed recipe retained; fixed narrow selection completed.",
        negative_source_limits="No ENAT fraction-auction date/price/payment found in selected2024 notices; no event-specific ENAT loan instruction. No ALSC physical credit receipt recovered. These bounded searches do not prove disclosure completeness.",
    )
    write_json_atomic(out / "sources.json", source)
    result = dict(
        status="enat_undated_fraction_engineering_qualified_stage_A_incomplete",
        audit=run["enat_settlement_audit"],
        sources=binding(out / "sources.json"),
        books=len(cases),
        sessions=days,
        account_array_cells=cells,
        independent_nav_identity_max_brl=identity_error,
        entitlements=checks,
        contrasts=contrasts,
        identical_intention_max_brl=max(
            x["identical_intention_nav_error"] for x in cases
        ),
        independent_adaptive_max_path_bps={
            str(c): max(
                x["adaptive_nav_difference_bps"] for x in cases if x["capital"] == c
            )
            for c in (10_000_000, 1_000_000, 5_000_000)
        },
        adaptive_target_max=max(x["adaptive_target_difference"] for x in cases),
        seconds=perf_counter() - tick,
        limits="Synthetic30-session full933 books, old4bp bridge, frozen old policy coordinates; no neural forward/model alpha/new fit. Earlier V32-V34 path uncertainties persist. Source loan/fraction hypotheses and final corporate pricing still open. New ENAT model-target/history propagation is not inferred from account admission; both accepted stores remain immutable.",
    )
    write_json_atomic(out / "manifest.json", result)
    run["enat_settlement_qualification"] = binding(out / "manifest.json")
    run["remaining_held_source_audit"] = binding(out / "sources.json")
    write_json_atomic(pointer, run)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
