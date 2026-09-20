"""Independent saved-fill Decimal charges, settled funding and NAV attribution."""

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


def main():
    tick = perf_counter()
    run = json.loads(
        (PROJECT / "docs/v2_economic_data_scaling_run.json").read_text(encoding="utf8")
    )
    root = Path(run["root"]) / "historical_spot"
    out = root / "qualification"
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / "executed.py")
    audit = bound_json(binding(root / "manifest.json"))
    plan = bound_json(audit["plan"])
    cases = bound_json(audit["completed"])
    assert len(cases) == 30
    cash = bound_json(run["cash_calendar"])["panel"]
    with np.load(cash["path"]) as z:
        cdi = z["cdi_returns"]
    terms = bound_json(run["enat_settlement_terms"])
    dates = np.load(Path(terms["store"]["root"]) / "date_index.npy").astype(
        "datetime64[D]"
    )
    books, details, errors, checks, contrasts = {}, {}, defaultdict(float), [], []
    cells = count_fills = 0
    for case in cases:
        folder = root / case["book"]
        book = bound_json(binding(folder / "book.json"))
        assert not book["legacy_slot_diagnostics_applicable"]
        for name, digest in book["files"].items():
            assert sha256_file(folder / name) == digest
        with np.load(folder / "account.npz") as z:
            a = {k: z[k] for k in z.files}
        key = case["capital"], case["scenario"]
        books[key] = a
        detail = bound_json(binding(folder / "funding_and_costs.json"))
        details[key] = detail
        assert a["signed_shares"].shape == (32, 933)
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
        )
        errors["nav"] = max(errors["nav"], float(np.abs(nav - a["nav"]).max()))
        cfg = book["provenance"]["config"]
        rates = (
            [Decimal(4), Decimal(0), Decimal(0), Decimal(0), Decimal(0)]
            if case["scenario"] == "bundled_control"
            else [
                Decimal(0),
                Decimal(".7")
                if cfg["spot_execution_phase"] == "auction"
                else Decimal(".5"),
                Decimal("2.5"),
                Decimal(str(cfg["execution_brokerage_bps"])),
                Decimal(str(cfg["execution_shortfall_bps"])),
            ]
        )
        fills = pl.read_parquet(folder / "fills.parquet")
        count_fills += fills.height
        for day, row in enumerate(detail):
            selected = fills.filter(pl.col("fill_session") == day)
            for group in selected.partition_by("security_index"):
                assert group["side"].n_unique() == 1
            traded = sum(
                (Decimal(str(v)) for v in selected["gross_notional"]), Decimal(0)
            )
            expected = np.array([float(traded * rate / 10000) for rate in rates])
            errors["decimal_charges"] = max(
                errors["decimal_charges"],
                float(np.abs(expected - a["execution_charges"][day]).max()),
            )
            errors["account_components"] = max(
                errors["account_components"],
                float(np.abs(a["execution_charges"][day] - row["components"]).max()),
            )
            errors["cost_sum"] = max(
                errors["cost_sum"], abs(float(expected.sum()) - row["cost"])
            )
            funding = a["free_cash"][day - 1] if day else case["capital"]
            restricted = (
                a["restricted_cash"][day - 1] + a["hedge_restricted_cash"][day - 1]
                if day
                else 0
            )
            errors["funding"] = max(
                errors["funding"],
                abs(funding - row["funding_cash"]),
                abs(restricted - row["funding_restricted"]),
            )
            rate = float(
                cdi[
                    int(np.searchsorted(dates, np.datetime64(book["state_dates"][day])))
                ]
            )
            expected_income = [
                max(funding, 0) * rate,
                restricted * rate * cfg["short_proceeds_remuneration"],
                -min(funding, 0) * (rate + cfg["annual_debit_spread"] / 252),
            ]
            for name, value in zip(
                ("free_income", "proceeds_income", "debit_financing"), expected_income
            ):
                errors["funding_income"] = max(
                    errors["funding_income"], abs(value - row[name])
                )
            assert (
                abs(
                    row["interest"]
                    - (expected_income[0] + expected_income[1] - expected_income[2])
                )
                < 1e-7
            )
        checks.append(
            dict(
                book=case["book"],
                fills=fills.height,
                debit_sessions=case["debit_sessions"],
                components=case["execution_components"],
            )
        )
    for capital in plan["capital"]:
        base = books[capital, "primary"]
        for field, array in base.items():
            if field != "execution_charges":
                np.testing.assert_array_equal(
                    array, books[capital, "bundled_control"][field]
                )
        for scenario in plan["variants"]:
            if scenario in {"primary", "bundled_control"}:
                continue
            candidate = books[capital, scenario]
            # New cost/cash realizes after the decision and prior-close funding.
            first = next(
                (
                    i
                    for i, (x, y) in enumerate(
                        zip(details[capital, "primary"], details[capital, scenario])
                    )
                    if x["cost"] != y["cost"] or x["interest"] != y["interest"]
                ),
                32,
            )
            np.testing.assert_array_equal(base["nav"][:first], candidate["nav"][:first])
            np.testing.assert_array_equal(
                base["targets"][: min(first + 1, 32)],
                candidate["targets"][: min(first + 1, 32)],
            )
            diff = (candidate["nav"] - base["nav"]) / capital * 10000
            contrasts.append(
                dict(
                    capital=capital,
                    scenario=scenario,
                    first_different_realization=first if first < 32 else None,
                    final_bps=float(diff[-1]),
                    max_path_bps=float(np.abs(diff).max()),
                    direct_component_difference=(
                        candidate["execution_charges"].sum(0)
                        - base["execution_charges"].sum(0)
                    ).tolist(),
                )
            )
            if scenario.startswith("debit_"):
                assert all(
                    row["funding_cash"] >= 0 for row in details[capital, scenario]
                )
                for field in base:
                    np.testing.assert_array_equal(base[field], candidate[field])
    assert max(errors.values()) < 1e-7, errors
    report = dict(
        status="qualified_bounded_spot_not_final_economics",
        audit=binding(root / "manifest.json"),
        sessions=960,
        cells=cells,
        actual_fills=count_fills,
        errors=dict(errors),
        cases=checks,
        contrasts=contrasts,
        limitation="Unrounded ordinary net fills only, fractional research units; no debit exposure in these adaptive books. Custody, older tariffs, spot invoice rounding, daytrade and actual-held-Cielo remain outside this acceptance.",
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
    )
    write_json_atomic(out / "report.json", report)
    print(
        json.dumps({k: v for k, v in report.items() if k not in {"cases", "contrasts"}})
    )


if __name__ == "__main__":
    main()
