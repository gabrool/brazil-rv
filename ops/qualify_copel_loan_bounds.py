"""Qualify saved Copel books with Decimal cohort and independent cash arithmetic."""

from decimal import Decimal, localcontext
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic, sha256_file
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def dec(value):
    return Decimal(str(value))


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    audit = bound_json(run["copel_loan_bounds_audit"])
    root = Path(audit["audit_root"])
    out = root / "qualification"
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / "executed.py")
    cases = bound_json(audit["completed"])
    plan = bound_json(audit["plan"])
    assert len(cases) == 30
    arrays, checks = {}, []
    cells = days = loan_days = 0
    identity_error = cohort_error = rent_error = fee_error = share_error = 0.0
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
            a = {key: z[key] for key in z.files}
        arrays[case["scenario"], case["sign"], case["capital"]] = a
        assert a["signed_shares"].shape == (30, 933)
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
        error = float(np.abs(nav - a["nav"]).max())
        assert error < 5e-8
        identity_error = max(identity_error, error)
        days += len(nav)
        cells += sum(x.size for x in a.values())
        name, successors = case["source"], case["successors"]
        fills = pl.read_parquet(folder / "fills.parquet")
        assert not fills.filter(
            (pl.col("security_index") == name) & (pl.col("fill_session") >= 6)
        ).height
        q0 = dec(a["signed_shares"][5, name])
        assert q0 * case["sign"] > 0
        delivery = 8 if case["sign"] > 0 else case["conversion"]
        for successor, ratio in zip(successors, (1, 4)):
            trades = fills.filter(pl.col("security_index") == successor).to_dicts()
            for day in range(6, 30):
                traded = sum(
                    (
                        dec(f["quantity"]) * (1 if f["side"] == "buy" else -1)
                        for f in trades
                        if f["fill_session"] == day
                    ),
                    Decimal(0),
                )
                arrived = (
                    dec(a["signed_shares"][day, successor])
                    - dec(a["signed_shares"][day - 1, successor])
                    - traded
                )
                expected = q0 * ratio if day == delivery else Decimal(0)
                share_error = max(share_error, float(abs(arrived - expected)))
                assert abs(arrived - expected) < dec("1e-7")
        cohorts = bound_json(binding(folder / "cohorts.json"))
        conversion = cohorts[case["conversion"]]
        before, prepared = conversion["before"], conversion["prepared"]
        source_rows = []
        if case["sign"] < 0:
            assert before and all(r["name"] == name for r in before)
            for loan_root in sorted({r["root"] for r in before}):
                old = [r for r in before if r["root"] == loan_root]
                for successor, ratio, k in zip(
                    successors, (1, 4), (case["k"], 1 - case["k"])
                ):
                    new = [
                        r
                        for r in prepared
                        if r["root"] == loan_root and r["name"] == successor
                    ]
                    assert new
                    for key, factor in (
                        ("quantity", ratio),
                        ("principal", k),
                        ("rent_due", k),
                    ):
                        expected = sum((dec(r[key]) for r in old), Decimal(0)) * dec(
                            factor
                        )
                        actual = sum((dec(r[key]) for r in new), Decimal(0))
                        cohort_error = max(cohort_error, float(abs(actual - expected)))
                        assert abs(actual - expected) < dec("1e-7")
                    for component in (0, 1):
                        expected = sum(
                            (dec(r["fees_due"][component]) for r in old), Decimal(0)
                        ) * dec(k)
                        actual = sum(
                            (dec(r["fees_due"][component]) for r in new), Decimal(0)
                        )
                        assert abs(actual - expected) < dec("1e-7")
                    for r in new:
                        assert (
                            r["return_day"] < 0 or r["return_day"] >= case["conversion"]
                        )
                        for key in (
                            "annual_rate",
                            "opened",
                            "fee_rate",
                            "fee_growth",
                            "value_lag",
                            "minimum",
                            "root_fees",
                        ):
                            assert r[key] == old[0][key], (case["book"], key)
                source_rows.append(
                    dict(
                        root=loan_root,
                        original_principal=str(
                            sum((dec(r["principal"]) for r in old), Decimal(0))
                        ),
                        original_quantity=str(
                            sum((dec(r["quantity"]) for r in old), Decimal(0))
                        ),
                        rate=old[0]["annual_rate"],
                        minimum=old[0]["minimum"],
                    )
                )
        else:
            assert not before and not prepared
        charges = pl.read_parquet(folder / "loan_charges.parquet").filter(
            pl.col("security_index") == name
        )
        daily = []
        with localcontext() as context:
            context.prec = 40
            for row in cohorts[case["conversion"] :]:
                day = row["day"]
                rent = fees = Decimal(0)
                for r in row["prepared"]:
                    assert r["value_lag"] == 1 and r["minimum"] == 0
                    age = day - r["opened"]
                    growth = (1 + dec(r["annual_rate"])) ** (Decimal(1) / 252)
                    rent += dec(r["principal"]) * (growth ** (age - 1)) * (growth - 1)
                    fees += dec(r["principal"]) * sum(
                        (
                            dec(g) * ((1 + dec(rate)) ** (Decimal(1) / 252) - 1)
                            for g, rate in zip(r["fee_growth"], r["fee_rate"])
                        ),
                        Decimal(0),
                    )
                observed = charges.filter(pl.col("session") == day)
                actual_rent = dec(observed["rent"].sum())
                actual_fee = dec(observed["fee"].sum())
                rent_error = max(rent_error, float(abs(rent - actual_rent)))
                fee_error = max(fee_error, float(abs(fees - actual_fee)))
                assert abs(rent - actual_rent) < dec("1e-7")
                assert abs(fees - actual_fee) < dec("1e-7")
                loan_days += 1
                daily.append(
                    dict(
                        day=day,
                        decimal_rent=str(rent),
                        actual_rent=str(actual_rent),
                        decimal_fee=str(fees),
                        actual_fee=str(actual_fee),
                    )
                )
        checks.append(
            dict(
                book=case["book"],
                custody=8,
                effective=6,
                net_claim_delivery=delivery,
                nav_error=error,
                cohorts=binding(folder / "cohorts.json"),
                originals=source_rows,
                daily_source_loan_charges=daily,
            )
        )
    contrasts = []
    for sign in plan["signs"]:
        for capital in plan["capital"]:
            base = arrays["primary", sign, capital]
            for scenario in plan["scenarios"][1:]:
                a = arrays[scenario, sign, capital]
                prefix = 7 if scenario == "loan_early" else 8
                np.testing.assert_array_equal(a["nav"][:prefix], base["nav"][:prefix])
                np.testing.assert_array_equal(a["targets"][:7], base["targets"][:7])
                if sign > 0:
                    for key in a:
                        np.testing.assert_array_equal(a[key], base[key])
                delta = (a["nav"] - base["nav"]) / capital * 1e4
                contrasts.append(
                    dict(
                        scenario=scenario,
                        sign=sign,
                        capital=capital,
                        final_bps=float(delta[-1]),
                        max_path_bps=float(np.abs(delta).max()),
                        source_rent_delta_brl=float(
                            pl.read_parquet(
                                root
                                / f"CPLE_{scenario}_{sign}_{capital}"
                                / "loan_charges.parquet"
                            )
                            .filter(pl.col("security_index") == 262)["rent"]
                            .sum()
                            - pl.read_parquet(
                                root
                                / f"CPLE_primary_{sign}_{capital}"
                                / "loan_charges.parquet"
                            )
                            .filter(pl.col("security_index") == 262)["rent"]
                            .sum()
                        ),
                    )
                )
    result = dict(
        status="copel_loan_bounds_engineering_qualified_stage_A_incomplete",
        audit=run["copel_loan_bounds_audit"],
        books=30,
        daily_navs=days,
        account_array_cells=cells,
        independent_nav_identity_max_brl=identity_error,
        decimal_cohort_max_error=cohort_error,
        decimal_share_max_error=share_error,
        decimal_rent_max_error=rent_error,
        decimal_fee_max_error=fee_error,
        source_loan_daily_checks=loan_days,
        checks=checks,
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
            for capital in plan["capital"]
        },
        adaptive_target_max=max(c["adaptive_target_difference"] for c in cases),
        seconds=perf_counter() - tick,
        limitations="Synthetic30-session/full933, OLD frozen PolicyData, old4bp bridge. Net borrowed timing only; flat/positive source pending returns retain primary. Long custody fixed; K unknown. No old books/store/source audits repeated, no model scores/fit/profitability, no final corporate pricing/admission.",
    )
    write_json_atomic(out / "manifest.json", result)
    run["copel_loan_bounds_qualification"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("checks", "contrasts")}
        )
    )
    print(json.dumps([c for c in contrasts if c["sign"] < 0]))


if __name__ == "__main__":
    main()
