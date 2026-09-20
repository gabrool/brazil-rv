"""Saved actual-model sensitivity arithmetic and paired path attribution."""

from collections import defaultdict
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
import json
from pathlib import Path
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
    plan = bound_json(run["stage_c_sensitivity_plan"])
    root = Path(plan["output_root"])
    completed = bound_json(binding(root / "replays.json"))
    assert completed["status"] == "complete"
    assert len(completed["completed"]) == plan["planned_new_books"] == 1062
    out = root / "qualification"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    cases = {c["key"]: c for c in plan["cases"]}
    phases = {(p["name"], p["capital"]): p for p in plan["phases"]}
    source_tariffs = {
        c["date"]: c for c in bound_json(run["historical_cost_sources"])["calendar"]
    }
    accepted = bound_json(run["economic_refit_inputs"])["store"]
    dates = np.load(Path(accepted["root"]) / "date_index.npy")
    with np.load(bound_json(run["cash_calendar"])["panel"]["path"]) as z:
        cdi = z["cdi_returns"]
    counts, maxima = defaultdict(int), defaultdict(float)
    baselines, records, cash_payments = {}, [], []

    def check(label, expected, actual, tolerance=1e-6):
        a, b = np.asarray(expected), np.asarray(actual)
        error = float(np.max(np.abs(a - b), initial=0))
        assert np.isfinite(error) and error < tolerance, (label, error)
        maxima[label] = max(maxima[label], error)
        counts[label] += int(np.broadcast_arrays(a, b)[0].size)

    for rec in completed["completed"]:
        key = rec["key"]
        variant, capital_text, arm, fold, member = key.split("/")
        capital = int(capital_text)
        phase = phases[variant, capital]
        book = bound_json(rec["book"])
        folder = Path(rec["book"]["path"]).parent
        for filename, digest in book["files"].items():
            assert sha256_file(folder / filename) == digest
        with np.load(folder / "account.npz") as z:
            a = {k: z[k] for k in z.files}
        assert a["signed_shares"].shape == (len(a["nav"]), 933)
        counts["saved_array_cells"] += sum(v.size for v in a.values())
        cfg, daily, days = (
            book["provenance"]["config"],
            book["daily"],
            book["state_dates"],
        )
        primary = cases[key]["baseline"]
        if primary["key"] not in baselines:
            primary_book = bound_json(primary["book"])
            with np.load(Path(primary["book"]["path"]).parent / "account.npz") as z:
                saved = {
                    k: z[k]
                    for k in (
                        "nav",
                        "targets",
                        "execution_charges",
                        "spot_invoice_adjustment",
                        "loan_invoice_adjustment",
                    )
                }
            baselines[primary["key"]] = primary_book, saved
        base, previous = baselines[primary["key"]]
        assert days == base["state_dates"]
        for provenance in ("forecast_sources", "mapping", "member", "old_policy_cache"):
            assert book["provenance"][provenance] == base["provenance"][provenance]
        assert not book["provenance"]["heldout_accessed"]
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
        check("nav_brl", nav, a["nav"])
        check(
            "custody_liability_rollforward",
            np.r_[0, a["custody_liability"][:-1]]
            + a["custody_fee"]
            - a["custody_payment"],
            a["custody_liability"],
        )
        indices = np.searchsorted(dates, np.array(days, dtype="datetime64[D]"))
        np.testing.assert_array_equal(dates[indices].astype(str), days)
        check("cdi_return", cdi[indices], np.array(daily["cdi_bps"]) / 1e4, 1e-15)
        free = np.r_[capital, a["free_cash"][:-1]]
        restricted = np.r_[0, (a["restricted_cash"] + a["hedge_restricted_cash"])[:-1]]
        free_income = np.maximum(free, 0) * cdi[indices]
        debit = -np.minimum(free, 0) * (
            cdi[indices] + cfg["annual_debit_spread"] / cfg["annual_sessions"]
        )
        proceeds = restricted * cdi[indices] * cfg["short_proceeds_remuneration"]
        scale = a["start_nav"] / 1e4
        for label, expected in (
            ("free_cash_income_bps", free_income),
            ("debit_financing_bps", debit),
            ("short_proceeds_income_bps", proceeds),
            ("interest_bps", free_income - debit + proceeds),
        ):
            check(label, expected, np.asarray(daily[label]) * scale)
        fills = pl.read_parquet(folder / "fills.parquet").to_dicts()
        by_day = defaultdict(list)
        for f in fills:
            by_day[f["fill_session"]].append(f)
        counts["fills"] += len(fills)
        pending = []
        for day, date in enumerate(days):
            groups, directions = defaultdict(lambda: Decimal(0)), defaultdict(set)
            flow, unrounded_cost = Decimal(0), Decimal(0)
            for f in by_day[day]:
                amount = Decimal(str(f["gross_notional"]))
                groups[f["security_index"]] += amount
                directions[f["security_index"]].add(f["side"])
                cost = Decimal(str(f["cost"]))
                flow += amount * (1 if f["side"] == "sell" else -1) - cost
                unrounded_cost += cost
            assert all(len(sides) == 1 for sides in directions.values()), (
                key,
                date,
                "opposing spot fills",
            )
            total = sum(groups.values(), Decimal(0))
            trading = source_tariffs[date]["trading_bps"]
            if trading is None:
                trading = cfg["unrecovered_spot_trading_bps"]
            if cfg["spot_execution_phase"] == "auction":
                trading = 0.7
            rates = [
                Decimal(str(trading)),
                Decimal("2.75" if date < "2021-02-02" else "2.5"),
            ]
            raw = [total * rate / 10000 for rate in rates]
            invoice = raw
            if cfg["spot_invoice_convention"] == "security_day_6dp_cent":
                invoice = [
                    sum(
                        (
                            (
                                amount.quantize(
                                    Decimal(".000001"), rounding=ROUND_HALF_UP
                                )
                                * rate
                                / 10000
                            ).quantize(Decimal(".000001"), rounding=ROUND_HALF_UP)
                            for amount in groups.values()
                        ),
                        Decimal(0),
                    ).quantize(Decimal(".01"), rounding=ROUND_DOWN)
                    for rate in rates
                ]
            adjustment = sum(
                (v - u for v, u in zip(invoice, raw, strict=True)), Decimal(0)
            )
            components = [
                0,
                *map(float, invoice),
                float(total * Decimal(str(cfg["execution_brokerage_bps"])) / 10000),
                float(total * Decimal(str(cfg["execution_shortfall_bps"])) / 10000),
            ]
            check("decimal_spot_components", components, a["execution_charges"][day])
            check(
                "decimal_spot_adjustment",
                [float(v - u) for v, u in zip(invoice, raw, strict=True)],
                a["spot_invoice_adjustment"][day],
            )
            check(
                "fill_cost_plus_adjustment",
                float(unrounded_cost + adjustment),
                a["execution_charges"][day].sum(),
            )
            check(
                "daily_trading_cost",
                a["execution_charges"][day].sum(),
                daily["trading_cost_bps"][day] * scale[day],
            )
            pending.append((day + (3 if date < "2019-05-27" else 2), flow - adjustment))
            pending = [(d, v) for d, v in pending if d > day]
            check(
                "independent_spot_cash_queue",
                float(sum((v for _, v in pending), Decimal(0))),
                a["unsettled_cash"][day],
            )
            counts["security_day_groups"] += len(groups)
        payments = bound_json(rec["loan_cash_payments"])
        if payments:
            cash_payments.append(dict(key=key, payments=payments))
        change = a["nav"] - previous["nav"]
        first = np.flatnonzero(change != 0)
        first_target = np.flatnonzero(
            np.any(a["targets"] != previous["targets"], axis=1)
        )
        if variant in {
            "shortfall0",
            "shortfall2",
            "brokerhalf",
            "brokerone",
            "auction",
            "proceeds95",
            "debit50",
            "debit100",
            "spot_invoice",
            "contract_nearest",
            "contract_down",
            "contract_up",
            "security_day_nearest",
            "minimum_pro_rata",
            "custody_economic_long",
            "custody_exclude_rights",
            "custody_payment3",
            "custody_payment10",
            "missing_tariff_lower",
        }:
            np.testing.assert_array_equal(a["targets"][0], previous["targets"][0])
        row = dict(
            key=key,
            baseline=primary["book"],
            scenario=rec["book"],
            capital=capital,
            variant=variant,
            arm=arm,
            fold=fold,
            sessions=len(days),
            final_path_bps=float(change[-1] / capital * 1e4),
            max_abs_path_bps=float(np.max(np.abs(change)) / capital * 1e4),
            mean_net_cdI_delta_bps=float(
                np.mean(daily["net_excess_bps"])
                - np.mean(base["daily"]["net_excess_bps"])
            ),
            first_nav_difference=None if not len(first) else int(first[0]),
            first_target_difference=None
            if not len(first_target)
            else int(first_target[0]),
            execution_components_brl=a["execution_charges"].sum(0).tolist(),
            primary_execution_components_brl=previous["execution_charges"]
            .sum(0)
            .tolist(),
            loan_invoice_adjustment_brl=a["loan_invoice_adjustment"].sum(0).tolist(),
            free_income_brl=float(free_income.sum()),
            proceeds_income_brl=float(proceeds.sum()),
            debit_financing_brl=float(debit.sum()),
            overdue_sessions=int((a["loan_overdue_principal"] > 0).sum()),
            maximum_overdue_principal_brl=float(a["loan_overdue_principal"].max()),
            unresolved_economics=rec["economics_unresolved"],
            phase_hypothesis=phase,
        )
        records.append(row)
        # A failure after a completed book does not require repeating its checks.
        with (out / "completed.jsonl").open("a", encoding="utf8") as stream:
            stream.write(json.dumps(row) + "\n")
        if len(records) % 100 == 0:
            print(
                json.dumps(dict(qualified=len(records), seconds=perf_counter() - tick)),
                flush=True,
            )
    write_json_atomic(out / "cash_payment_exposures.json", cash_payments)
    write_json_atomic(out / "results.json", records)
    by_key = {c["key"]: c for c in records}
    grouping = []
    for row in records:
        if row["variant"] == "security_day_nearest":
            other = by_key[
                row["key"].replace("security_day_nearest/", "contract_nearest/")
            ]
            grouping.append(
                dict(
                    key=row["key"],
                    final_grouping_minus_contract_bps=row["final_path_bps"]
                    - other["final_path_bps"],
                    mean_net_cdi_grouping_delta_bps=row["mean_net_cdI_delta_bps"]
                    - other["mean_net_cdI_delta_bps"],
                )
            )
    write_json_atomic(out / "grouping_attribution.json", grouping)
    report = dict(
        passed=True,
        plan=run["stage_c_sensitivity_plan"],
        completed=binding(root / "replays.json"),
        books=len(records),
        primary_books=len(baselines),
        skipped=len(plan["skipped"]),
        counts=dict(counts),
        maximum_errors=dict(maxima),
        results=binding(out / "results.json"),
        grouping=binding(out / "grouping_attribution.json"),
        cash_payments=binding(out / "cash_payment_exposures.json"),
        seconds=perf_counter() - tick,
        limits="Actual old-coordinate model portfolios, not corrected-data refit results. One-factor adaptive total paths; no joint/worst-interior bound. Retain all prior fixed-minimum adaptive uncertainty. General loan/custody mechanics reuse prior independent proofs; no new independent full-cohort reconstruction. New loan cash exposures require conditional source/cent assessment before final admission.",
    )
    write_json_atomic(out / "report.json", report)
    run = json.loads(pointer.read_text())
    run["stage_c_sensitivities"] = binding(out / "report.json")
    write_json_atomic(pointer, run)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
