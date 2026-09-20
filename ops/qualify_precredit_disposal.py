"""Independent arithmetic on saved precredit books; no account replay."""

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


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    audit = bound_json(run["precredit_disposal_audit"])
    root = Path(audit["audit_root"])
    out = root / "qualification"
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / "executed.py")
    plan = bound_json(audit["plan"])
    cases = bound_json(audit["completed"])
    assert len(cases) == 54
    arrays, checks = {}, []
    cells = days = 0
    identity_error = quantity_error = 0.0
    for case in cases:
        folder = root / case["book"]
        meta = bound_json(binding(folder / "book.json"))
        assert (
            meta["status"] == "completed"
            and not meta["legacy_slot_diagnostics_applicable"]
        )
        for filename, digest in meta["files"].items():
            assert sha256_file(folder / filename) == digest
        with np.load(folder / "account.npz") as z:
            a = {k: z[k] for k in z.files}
        arrays[case["event"], case["scenario"], case["sign"], case["capital"]] = a
        assert a["signed_shares"].shape == (14, 933)
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
        fills = pl.read_parquet(folder / "fills.parquet")
        name = case["source"]
        assert not fills.filter(
            (pl.col("security_index") == name) & (pl.col("fill_session") >= 6)
        ).height
        event = plan["events"][case["event"]]
        q0 = Decimal(str(a["signed_shares"][5, name]))
        assert q0 * case["sign"] > 0
        arrival = (
            8
            if case["scenario"] == "custody" or case["sign"] < 0
            else 6 + (case["scenario"] == "next")
        )
        own = []
        if case["sign"] > 0:
            for destination, leg in zip(case["successors"], event["legs"]):
                entitlement = q0 * Decimal(str(leg["shares_per_prior_share"]))
                whole = (
                    entitlement.to_integral_value(rounding=ROUND_FLOOR)
                    if leg.get("fractional_auction")
                    else entitlement
                )
                subset = fills.filter(
                    pl.col("security_index") == destination
                ).to_dicts()
                for day in range(6, 14):
                    traded = sum(
                        (
                            Decimal(str(f["quantity"]))
                            * (1 if f["side"] == "buy" else -1)
                            for f in subset
                            if f["fill_session"] == day
                        ),
                        Decimal(0),
                    )
                    gained = (
                        Decimal(str(a["signed_shares"][day, destination]))
                        - Decimal(str(a["signed_shares"][day - 1, destination]))
                        - traded
                    )
                    expected = whole if day == arrival else Decimal(0)
                    err = abs(gained - expected)
                    quantity_error = max(quantity_error, float(err))
                    assert err < Decimal("1e-7"), (
                        case["book"],
                        day,
                        destination,
                        gained,
                        expected,
                    )
                early_fills = [f for f in subset if arrival <= f["fill_session"] < 8]
                # Historical windows are post-2019T+2. This independently binds
                # each possible early sale to physical credit, not to an early
                # synthetic delivery assumption.
                assert all(f["fill_session"] + 2 >= 8 for f in early_fills)
                own.append(
                    dict(
                        successor=destination,
                        decimal_entitlement=str(entitlement),
                        decimal_whole=str(whole),
                        decimal_fraction=str(entitlement - whole),
                        economic_arrival=arrival,
                        physical_credit=8,
                        precredit_fill_rows=len(early_fills),
                        precredit_sell_quantity=sum(
                            float(f["quantity"])
                            for f in early_fills
                            if f["side"] == "sell"
                        ),
                    )
                )
            if event["legs"][0].get("fractional_auction"):
                ratio = Decimal(str(event["legs"][0]["shares_per_prior_share"]))
                expected = q0 * ratio - (q0 * ratio).to_integral_value(
                    rounding=ROUND_FLOOR
                )
                claims = bound_json(binding(folder / "share_claim_positions.json"))
                residuals = [
                    c
                    for c in claims
                    if c["source_index"] == name and c["session"] >= arrival
                ]
                assert len(residuals) == 14 - arrival
                assert all(
                    c["delivery_session"] is None
                    and abs(Decimal(str(c["signed_quantity"])) - expected)
                    < Decimal("1e-7")
                    for c in residuals
                )
        custody = bound_json(binding(folder / "custody.json"))
        for row in custody:
            if row["day"] < 8:
                assert all(due >= row["day"] for due, _ in row["after_receipts"])
        checks.append(
            dict(
                book=case["book"],
                source_quantity=str(q0),
                legs=own,
                nav_identity_error=error,
                custody=binding(folder / "custody.json"),
            )
        )
    contrasts = []
    for label in plan["events"]:
        for sign in plan["signs"]:
            for capital in plan["capital"]:
                b = arrays[label, "custody", sign, capital]
                for scenario in ("effect", "next"):
                    a = arrays[label, scenario, sign, capital]
                    prefix = 6 + (scenario == "next")
                    np.testing.assert_array_equal(a["nav"][:prefix], b["nav"][:prefix])
                    np.testing.assert_array_equal(a["targets"][:7], b["targets"][:7])
                    if sign < 0:
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
                            max_absolute_path_bps=float(np.abs(delta).max()),
                            identical_nav_prefix=prefix,
                        )
                    )
    result = dict(
        status="precredit_disposal_engineering_qualified_stage_A_incomplete",
        audit=run["precredit_disposal_audit"],
        books=54,
        daily_navs=days,
        account_array_cells=cells,
        independent_nav_identity_max_brl=identity_error,
        decimal_quantity_max_error=quantity_error,
        entitlements=checks,
        contrasts=contrasts,
        identical_intention_max_brl=max(
            x["identical_intention_nav_error"] for x in cases
        ),
        independent_adaptive_max_path_bps={
            str(c): max(
                x["adaptive_nav_difference_bps"] for x in cases if x["capital"] == c
            )
            for c in plan["capital"]
        },
        adaptive_target_max=max(x["adaptive_target_difference"] for x in cases),
        seconds=perf_counter() - tick,
        limitations="Synthetic14-session books/full933, frozen OLDPolicyData/old4bp bridge. Early permission for unencumbered positive holdings only; source loans stay at sourced credit. Existing successor purchases/offsets keep custody clocks. Known-source liquidation/rounding/payment/cost/clearing and actual-model final admission remain. No source retrieval, store amendment, neural scoring or fit.",
    )
    write_json_atomic(out / "manifest.json", result)
    run["precredit_disposal_qualification"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("entitlements", "contrasts")}
        )
    )
    print(json.dumps([r for r in contrasts if r["capital"] == 10_000_000]))


if __name__ == "__main__":
    main()
