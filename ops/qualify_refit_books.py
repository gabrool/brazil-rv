"""Saved fresh-fit account arithmetic and actual unresolved-exposure inventory."""

from collections import defaultdict
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main(sensitivities=False, expanded_attention=False, source_attention=False):
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    reference = (
        run["scaling_expanded_source_plan"]
        if source_attention
        else run["scaling_expanded_evaluation_plan"]
        if expanded_attention
        else run["stage_c_refit_sensitivity_plan"]
        if sensitivities
        else run["stage_c_data_replay_plan"]
    )
    outer = bound_json(reference)
    plan = bound_json(outer["primary"]) if sensitivities else outer
    root = Path(reference["path"]).parent
    replays = bound_json(binding(root / "replays.json"))
    out = root / "qualification"
    out.mkdir(exist_ok=True)
    recipe = out / "executed.py"
    if recipe.exists() and sha256_file(recipe) != sha256_file(Path(__file__)):
        recipe = out / ("executed_" + sha256_file(Path(__file__))[:12] + ".py")
    if not recipe.exists():
        recipe.write_bytes(Path(__file__).read_bytes())
    store = Path(plan["store"]["root"])
    names = np.load(store / "isin_index.npy").tolist()
    dates = np.load(store / "date_index.npy")
    raw = np.load(store / "raw_close.npy", mmap_mode="r")
    cash = bound_json(run["cash_calendar"])["panel"]
    assert sha256_file(Path(cash["path"])) == cash["sha256"]
    with np.load(cash["path"]) as z:
        cdi = z["cdi_returns"]
    tariffs = {
        c["date"]: c for c in bound_json(run["historical_cost_sources"])["calendar"]
    }
    terms = bound_json(plan["terms"])
    inherited = {}
    if expanded_attention:
        original = bound_json(run["stage_c_data_replay_plan"])
        for field in ("store", "inputs", "terms", "configuration"):
            if field in original:
                assert plan[field] == original[field]
        for proof in bound_json(run["stage_c_data_replay_qualification"])["reports"]:
            saved = bound_json(proof)
            inherited[saved["key"]] = (proof, saved)
    records = []
    for rec in replays["completed"]:
        if rec["key"] in inherited:
            proof, saved = inherited[rec["key"]]
            assert saved["passed"] and saved["book"] == rec["book"]
            records.append(proof)
            continue
        target = out / (rec["key"].replace("/", "_") + ".json")
        if target.exists():
            saved = bound_json(binding(target))
            assert saved["book"] == rec["book"]
            records.append(binding(target))
            continue
        started = perf_counter()
        book = bound_json(rec["book"])
        folder = Path(rec["book"]["path"]).parent
        for name, digest in book["files"].items():
            assert sha256_file(folder / name) == digest
        with np.load(folder / "account.npz") as z:
            a = {k: z[k] for k in z.files}
        counts, maxima = defaultdict(int), defaultdict(float)

        def check(label, expected, actual, tolerance=1e-6):
            x, y = np.asarray(expected), np.asarray(actual)
            error = float(np.max(np.abs(x - y), initial=0))
            assert np.isfinite(error) and error < tolerance, (rec["key"], label, error)
            maxima[label] = max(maxima[label], error)
            counts[label] += int(np.broadcast_arrays(x, y)[0].size)

        days, daily, cfg = (
            book["state_dates"],
            book["daily"],
            book["provenance"]["config"],
        )
        indices = np.searchsorted(dates, np.asarray(days, dtype="datetime64[D]"))
        np.testing.assert_array_equal(dates[indices].astype(str), days)
        assert a["signed_shares"].shape == (len(days), 933)
        assert book["provenance"]["policy_inputs"] == plan["inputs"]
        assert not book["provenance"]["heldout_accessed"]
        counts["saved_cells"] = sum(v.size for v in a.values())
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
        check("nav_brl", nav, a["nav"], 1e-7)
        check(
            "custody_liability",
            np.r_[0, a["custody_liability"][:-1]]
            + a["custody_fee"]
            - a["custody_payment"],
            a["custody_liability"],
        )
        free = np.r_[cfg["initial_capital_brl"], a["free_cash"][:-1]]
        restricted = np.r_[0, (a["restricted_cash"] + a["hedge_restricted_cash"])[:-1]]
        scale = a["start_nav"] / 1e4
        check("cdi", cdi[indices], np.asarray(daily["cdi_bps"]) / 1e4, 1e-15)
        free_income = np.maximum(free, 0) * cdi[indices]
        debit = -np.minimum(free, 0) * (
            cdi[indices] + cfg["annual_debit_spread"] / cfg["annual_sessions"]
        )
        proceeds = restricted * cdi[indices] * cfg["short_proceeds_remuneration"]
        for label, expected in (
            ("free_cash_income_bps", free_income),
            ("debit_financing_bps", debit),
            ("short_proceeds_income_bps", proceeds),
            ("interest_bps", free_income - debit + proceeds),
        ):
            check(label, expected, np.asarray(daily[label]) * scale)
        fills_path = folder / "fills.parquet"
        fills = pl.read_parquet(fills_path).to_dicts() if fills_path.exists() else []
        by_day = defaultdict(list)
        signed = defaultdict(lambda: Decimal(0))
        for fill in fills:
            by_day[fill["fill_session"]].append(fill)
            if fill["purpose"] != "hedge":
                signed[fill["fill_session"], fill["security_index"]] += Decimal(
                    str(fill["quantity"])
                ) * (1 if fill["side"] == "buy" else -1)
        counts["fills"] = len(fills)
        pending = []
        for day, date in enumerate(days):
            groups = defaultdict(set)
            amounts = defaultdict(lambda: Decimal(0))
            total, flow, cost = Decimal(0), Decimal(0), Decimal(0)
            for fill in by_day[day]:
                amount, charge = (
                    Decimal(str(fill["gross_notional"])),
                    Decimal(str(fill["cost"])),
                )
                total += amount
                cost += charge
                flow += amount * (1 if fill["side"] == "sell" else -1) - charge
                groups[fill["security_index"]].add(fill["side"])
                amounts[fill["security_index"]] += amount
            assert all(len(sides) == 1 for sides in groups.values()), (
                rec["key"],
                date,
                "opposing spot fills need dated daytrade disposition",
            )
            trading = tariffs[date]["trading_bps"]
            if trading is None:
                trading = cfg["unrecovered_spot_trading_bps"]
            if cfg["spot_execution_phase"] == "auction":
                trading = 0.7
            rates = [
                trading,
                2.75 if date < "2021-02-02" else 2.5,
                cfg["execution_brokerage_bps"],
                cfg["execution_shortfall_bps"],
            ]
            expected = [
                0,
                *[float(total * Decimal(str(rate)) / 10000) for rate in rates],
            ]
            adjustment = np.zeros(2)
            if cfg["spot_invoice_convention"] == "security_day_6dp_cent":
                for category, rate in enumerate(rates[:2]):
                    charge = sum(
                        (
                            (
                                amount.quantize(
                                    Decimal(".000001"), rounding=ROUND_HALF_UP
                                )
                                * Decimal(str(rate))
                                / 10000
                            ).quantize(Decimal(".000001"), rounding=ROUND_HALF_UP)
                            for amount in amounts.values()
                        ),
                        Decimal(0),
                    ).quantize(Decimal(".01"), rounding=ROUND_DOWN)
                    adjustment[category] = float(charge) - expected[category + 1]
                    expected[category + 1] = float(charge)
            check("decimal_spot_components", expected, a["execution_charges"][day])
            check(
                "invoice_adjustment",
                adjustment,
                a["spot_invoice_adjustment"][day],
            )
            check("fill_charges", float(cost) + adjustment.sum(), sum(expected))
            pending.append(
                (
                    day + (3 if date < "2019-05-27" else 2),
                    flow - Decimal(str(float(adjustment.sum()))),
                )
            )
            pending = [(d, v) for d, v in pending if d > day]
            check(
                "spot_cash_queue",
                float(sum((v for _, v in pending), Decimal(0))),
                a["unsettled_cash"][day],
            )
            counts["spot_groups"] += len(groups)

        identities = []
        for event in terms["identity_actions"]:
            if event["effective_date"] not in days:
                continue
            day = days.index(event["effective_date"])
            source, destination = (
                names.index(event["predecessor_isin"]),
                names.index(event["successor_isin"]),
            )
            before = (
                Decimal(str(float(a["signed_shares"][day - 1, source])))
                if day
                else Decimal(0)
            )
            destination_before = (
                Decimal(str(float(a["signed_shares"][day - 1, destination])))
                if day
                else Decimal(0)
            )
            after = Decimal(str(float(a["signed_shares"][day, destination])))
            check(
                "identity_arrival",
                float(before),
                float(after - destination_before - signed[day, destination]),
            )
            check("retired_identity_quantity", 0, a["signed_shares"][day, source])
            identities.append(
                dict(
                    event=event["effective_date"],
                    source=source,
                    destination=destination,
                    held=float(before),
                )
            )

        printed = np.isfinite(raw[indices]) & (raw[indices] > 0)
        ii, jj = np.where((a["signed_shares"] != 0) & ~printed)
        exposure = []
        for day, name in zip(ii, jj, strict=True):
            value = float(a["signed_shares"][day, name] * a["mark_price"][day, name])
            exposure.append(
                dict(
                    date=days[day],
                    isin=names[name],
                    axis=int(name),
                    shares=float(a["signed_shares"][day, name]),
                    absolute_marked_brl=abs(value) if np.isfinite(value) else None,
                    terminal=bool(day == len(days) - 1),
                )
            )
        exposure_path = target.with_suffix(".exposures.json")
        write_json_atomic(exposure_path, exposure)
        payments = bound_json(rec["loan_cash_payments"])
        contrast = None
        if sensitivities:
            baseline = bound_json(rec["baseline"]["book"])
            assert baseline["state_dates"] == days
            for field in ("forecast_sources", "mapping", "member", "policy_inputs"):
                assert baseline["provenance"][field] == book["provenance"][field]
            with np.load(
                Path(rec["baseline"]["book"]["path"]).parent / "account.npz"
            ) as z:
                difference = (a["nav"] - z["nav"]) / cfg["initial_capital_brl"] * 1e4
            contrast = dict(
                final_path_bps=float(difference[-1]),
                max_abs_path_bps=float(np.max(np.abs(difference))),
                mean_net_cdi_delta_bps_day=float(
                    np.mean(
                        np.asarray(daily["net_excess_bps"])
                        - baseline["daily"]["net_excess_bps"]
                    )
                ),
                baseline=rec["baseline"]["book"],
            )
        result = dict(
            passed=True,
            executed_recipe=binding(recipe),
            key=rec["key"],
            book=rec["book"],
            counts=dict(counts),
            maxima=dict(maxima),
            identities=identities,
            unquoted_holdings=binding(exposure_path),
            unquoted_cells=len(exposure),
            loan_cash_payments=payments,
            loan_cash_bounds_pending=bool(payments),
            prior_debit_sessions=int((free < 0).sum()),
            maximum_overdue_principal=float(np.max(a["loan_overdue_principal"])),
            economics_unresolved=rec["economics_unresolved"],
            contrast=contrast,
            seconds=perf_counter() - started,
            limits="Saved independent NAV/funding/spot-cash/fee and unit-identity checks; original loan/corporate mechanics reused. Unquoted holdings are evidence for exposure review, not observed quotes or proof of a new corporate event. No new forecasts or account replay ran in qualification.",
        )
        write_json_atomic(target, result)
        records.append(binding(target))
    summary = dict(
        status="complete"
        if replays["status"] == "complete" and len(records) == len(replays["completed"])
        else "partial",
        planned=len(replays["completed"]) if sensitivities else plan["planned_books"],
        qualified=len(records),
        reports=records,
        plan=reference,
        executed_recipe=binding(recipe),
        reused_reports=sum(r["key"] in inherited for r in replays["completed"]),
        seconds=perf_counter() - tick,
    )
    write_json_atomic(out / "manifest.json", summary)
    run = json.loads(pointer.read_text())
    run[
        "scaling_expanded_source_qualification"
        if source_attention
        else "scaling_expanded_qualification"
        if expanded_attention
        else "stage_c_refit_sensitivity_qualification"
        if sensitivities
        else "stage_c_data_replay_qualification"
    ] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(json.dumps({k: v for k, v in summary.items() if k != "reports"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sensitivities", action="store_true")
    group.add_argument("--expanded-attention", action="store_true")
    group.add_argument("--source-attention", action="store_true")
    args = parser.parse_args()
    main(args.sensitivities, args.expanded_attention, args.source_attention)
