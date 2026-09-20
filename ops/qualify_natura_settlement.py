"""Saved-book identities, independent entitlements and one-factor attribution."""

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


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    output = Path(run["root"]) / "natura_settlement/qualification"
    output.mkdir(exist_ok=False)
    shutil.copyfile(__file__, output / "executed.py")
    audits = {
        k: bound_json(run[k])
        for k in ("natura_settlement_audit", "natura_bonus_disposal_audit")
    }
    identity_error = 0.0
    days = cells = 0
    entitlements, contrasts, paths = [], [], []
    for key, audit in audits.items():
        root = Path(audit["audit_root"])
        cases = bound_json(audit["completed"])
        e, name = audit["effects"], audit["name"]
        books = {}
        for case in cases:
            folder = root / case["book"]
            book = json.loads((folder / "book.json").read_text())
            for file, digest in book["files"].items():
                assert sha256_file(folder / file) == digest
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
            assert error < 5e-8, (case["book"], error)
            identity_error = max(identity_error, error)
            assert a["signed_shares"].shape == (case["sessions"], 933)
            days += len(nav)
            cells += sum(v.size for v in a.values())
            paths.append(
                {
                    k: case[k]
                    for k in (
                        "book",
                        "capital",
                        "sign",
                        "identical_intention_nav_error",
                        "adaptive_nav_difference_bps",
                        "adaptive_target_difference",
                    )
                }
            )
            if case["scenario"] == "legacy_scalar":
                continue
            fills = pl.read_parquet(folder / "fills.parquet").filter(
                pl.col("security_index") == name
            )
            q0 = Decimal(str(a["signed_shares"][e["bonus"] - 1, name]))
            signed_fills = sum(
                (
                    Decimal(str(f["quantity"])) * (1 if f["side"] == "buy" else -1)
                    for f in fills.filter(
                        pl.col("fill_session") == e["bonus"]
                    ).to_dicts()
                ),
                Decimal(0),
            )
            expected_quantity = q0 * 2 + signed_fills
            assert (
                abs(float(expected_quantity) - a["signed_shares"][e["bonus"], name])
                < 1e-9
            )
            if case["scenario"] in ("primary", "net_compensation"):
                claims = json.loads((folder / "share_claim_positions.json").read_text())
                bonus = [
                    c
                    for c in claims
                    if c["source_index"] == name and c["successor_index"] == name
                ]
                assert [c["session"] for c in bonus] == list(
                    range(e["bonus"], e["credit"])
                )
                assert all(
                    c["delivery_session"] == e["credit"]
                    and c["signed_quantity"] == float(q0)
                    for c in bonus
                )
                if key == "natura_bonus_disposal_audit":
                    np.testing.assert_allclose(
                        a["signed_shares"][e["bonus"] : e["credit"], name],
                        float(q0),
                        rtol=0,
                        atol=1e-9,
                    )
            if e["jcp"] < len(nav):
                held = Decimal(str(a["signed_shares"][e["jcp"] - 1, name]))
                gross = held * Decimal("0.12784527353")
                withheld = max(gross, Decimal(0)) * (
                    Decimal(0)
                    if case["scenario"] == "gross_instant"
                    else Decimal(".15")
                )
                net = (
                    gross - withheld
                    if gross >= 0
                    else gross
                    * (
                        Decimal(".85")
                        if case["scenario"] == "net_compensation"
                        else Decimal(1)
                    )
                )
                assert abs(float(withheld) - a["withholding_accrual"].sum()) < 1e-9
                entitlements.append(
                    dict(
                        book=case["book"],
                        record_close="2019-11-06",
                        gross_brl=str(gross),
                        withholding_brl=str(withheld),
                        signed_cash_brl=str(net),
                        payment="2020-02-26",
                        tax_credit_asset=0,
                    )
                )
        pairs = [("primary", "immediate_credit", "bonus_custody")]
        if key == "natura_settlement_audit":
            pairs += [
                ("gross_instant", "legacy_scalar", "source_gross"),
                ("immediate_credit", "gross_instant", "account_withholding"),
                ("net_compensation", "primary", "lender_compensation"),
            ]
        for left, right, label in pairs:
            for sign in (1, -1):
                for capital in (10_000_000, 1_000_000, 5_000_000):
                    a, b = books[left, sign, capital], books[right, sign, capital]
                    effect = (
                        e["jcp"]
                        if label in ("account_withholding", "lender_compensation")
                        else e["bonus"]
                    )
                    np.testing.assert_array_equal(a["nav"][:effect], b["nav"][:effect])
                    np.testing.assert_array_equal(
                        a["targets"][: effect + 1], b["targets"][: effect + 1]
                    )
                    if label == "lender_compensation" and sign < 0:
                        held = Decimal(str(b["signed_shares"][effect - 1, name]))
                        expected = -held * Decimal(".12784527353") * Decimal(".15")
                        assert (
                            abs(
                                float(expected)
                                - (
                                    b["lender_compensation"][effect]
                                    - a["lender_compensation"][effect]
                                )
                            )
                            < 1e-9
                        )
                    if label == "lender_compensation" and sign > 0:
                        np.testing.assert_array_equal(a["nav"], b["nav"])
                    diff = (a["nav"] - b["nav"]) / capital * 1e4
                    contrasts.append(
                        dict(
                            scope=key,
                            contrast=label,
                            sign=sign,
                            capital=capital,
                            final_bps=float(diff[-1]),
                            max_path_bps=float(np.abs(diff).max()),
                        )
                    )
    result = dict(
        status="passed_saved_book_qualification",
        sources={k: run[k] for k in audits},
        books=len(paths),
        daily_identities=days,
        account_cells=cells,
        identity_max_error_brl=identity_error,
        identical_intention_max_error_brl=max(
            x["identical_intention_nav_error"] for x in paths
        ),
        independent_adaptive_max_bps={
            str(c): max(
                x["adaptive_nav_difference_bps"] for x in paths if x["capital"] == c
            )
            for c in (10_000_000, 1_000_000, 5_000_000)
        },
        independent_adaptive_target_max=max(
            x["adaptive_target_difference"] for x in paths
        ),
        entitlements=entitlements,
        contrasts=contrasts,
        scope="All933-name synthetic preference/fixed-risk actual adaptive allocator; no model profitability. Original account/source data immutable. No consumer or book reruns.",
        seconds=perf_counter() - tick,
    )
    write_json_atomic(output / "manifest.json", result)
    run["natura_settlement_qualification"] = binding(output / "manifest.json")
    run["natura_settlement_terms"] = audits["natura_settlement_audit"]["terms"]
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("contrasts", "entitlements")}
        )
    )


if __name__ == "__main__":
    main()
