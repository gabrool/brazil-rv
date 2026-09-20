"""Frozen adaptive fraction sweep and cash-precision bounds."""

from copy import copy, deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import polars as pl
import torch

from brazil_rv.execution.portfolio_policy import (
    CalibratedPolicy,
    PolicyData,
    account_decision,
    exact_replay,
    policy_ledger_config,
)
from brazil_rv.v2.artifacts import write_json_atomic, sha256_file
from brazil_rv.v2.cash_calendar import apply_cash_calendar
from brazil_rv.v2.corporate_replay import apply_corporate_replay, load_corporate_replay
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.lending_archive import load_lending_borrow_panels
from brazil_rv.v2.portfolio_inputs import Calibration
from brazil_rv.v2.portfolio_readouts import book_summary, save_book
from verify_corporate_replay import inputs_on_axes

PROJECT = Path(__file__).resolve().parents[1]


def main():
    tick = perf_counter()
    torch.set_num_threads(1)
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    root = Path(run["root"]) / "payment_bounds"
    resuming = root.exists()
    root.mkdir(exist_ok=True)
    shutil.copyfile(
        __file__, root / ("resumed_executed.py" if resuming else "executed.py")
    )
    if not resuming:
        shutil.copyfile(
            PROJECT / "research/preregistrations/v2_economic_data_scaling.md",
            root / "registration.md",
        )
    for folder in ("execution", "v2"):
        for path in (PROJECT / "research/src/brazil_rv" / folder).glob("*.py"):
            saved = root / f"executed_{folder}_{path.name}"
            if resuming:
                assert sha256_file(path) == sha256_file(saved)
            else:
                shutil.copyfile(path, saved)
    terms, dates = load_corporate_replay(
        run["enat_settlement_terms"]["path"], run["enat_settlement_terms"]["sha256"]
    )
    events = {
        label: next(e for e in terms["share_distributions"] if e["isin"] == isin)
        for label, isin in [
            ("BRML", "BRBRMLACNOR9"),
            ("DMMO", "BRDMMOACNOR0"),
        ]
    }
    events["CIEL"] = dict(isin="BRCIELACNOR3", effective_date="2024-08-30", legs=[])
    scenarios = {
        "BRML": ["primary", "early"],
        "DMMO": ["primary", "early", "precision", "tiny_rent"],
        "CIEL": ["primary", "cent_low", "cent_high"],
    }
    ends = {"BRML": "2023-02-06", "DMMO": "2023-04-12", "CIEL": "2024-10-02"}
    plan = dict(
        events=events,
        capital=[10_000_000, 1_000_000, 5_000_000],
        signs=[1, -1],
        scenarios=scenarios,
        sessions="6 pre-effect + through fixed endpoints; all933",
        endpoints=ends,
        preferences="sin(axis*.31+localday*.003+head*.2); source +/-4 before effect and opposite thereafter; successor opposite from effect. No model scoring.",
        risks="fixed beta1/idio.0004/market.0001; original .002/.001/.0005 calibration; frozen OLD PolicyData shallow copies",
        costs="old bundled4bp/no extraB3spot; correctedCDI/qualifiedloans/exactpriorrefs; not final corporate pricing",
        execution="V38 same-day newly-known auction releases omit an already-due queue; other V37 mechanics unchanged. BRML Jan26/Feb2 fraction payment; DMMO Mar31/Apr6 fraction payment and separate printed31.94031 versus565024/17690 precision; separate zero-whole original loan rent stop-at-credit versus through known Apr6 payment. CIEL sourced continuous loan cash5.842895570784521 +/-0.01/share. Dates/knowledge/cash legs/custody unchanged, no stacking. No invoice receipt or broker sweep claimed.",
        dependencies={
            k: run[k]
            for k in (
                "enat_settlement_terms",
                "qualified_lending",
                "loan_source_panels",
                "cash_calendar",
                "bova_loan_reference_audit",
            )
        },
    )
    if resuming:
        assert bound_json(binding(root / "plan.json")) == plan
    else:
        write_json_atomic(root / "plan.json", plan)
    store = Path(terms["store"]["root"])
    isins = np.load(store / "isin_index.npy").tolist()
    names = np.arange(len(isins))
    assert len(names) == 933 and dates[-1] <= np.datetime64("2024-12-30")
    lend = run["qualified_lending"]
    lending = load_lending_borrow_panels(
        Path(lend["root"]),
        expected_manifest_sha256=lend["manifest_sha256"],
        canonical_dates=dates.astype(object).tolist(),
        canonical_isins=isins,
    )
    panels = bound_json(run["loan_source_panels"])["panels"]
    assert sha256_file(Path(panels["path"])) == panels["sha256"]
    with np.load(panels["path"]) as z:
        references, hedge_rates = (
            z["loan_reference_prices"],
            z["hedge_annual_borrow_rate"],
        )
    cash = bound_json(run["cash_calendar"])["panel"]
    assert sha256_file(Path(cash["path"])) == cash["sha256"]
    with np.load(cash["path"]) as z:
        cdi = z["cdi_returns"]
    hedge = pl.read_parquet(
        bound_json(run["bova_loan_reference_audit"])["data"]["path"]
    )
    hmap = dict(zip(hedge["trade_date"], hedge["close_brl"]))
    hedge_close = np.array([hmap.get(d, np.nan) for d in dates.astype(object)])
    reports = bound_json(binding(root / "completed.json")) if resuming else []
    completed_books = {c["book"] for c in reports}
    for label, event in events.items():
        name = isins.index(event["isin"])
        successors = [isins.index(leg["successor_isin"]) for leg in event["legs"]]
        effect = int(np.searchsorted(dates, np.datetime64(event["effective_date"])))
        rows = np.arange(
            effect - 6, np.searchsorted(dates, np.datetime64(ends[label])) + 1
        )
        inputs = inputs_on_axes(store, dates, names, rows)
        prior = np.load(store / "prior_reference_close.npy", mmap_mode="r")[rows]
        for sign in plan["signs"]:
            scores = np.sin(
                names[None, :, None] * 0.31
                + np.arange(len(rows))[:, None, None] * 0.003
                + np.arange(5)[None, None, :] * 0.2
            ).astype(np.float32)
            scores[:6, name, :] = 4 * sign
            scores[6:, name, :] = -4 * sign
            for successor in successors:
                scores[6:, successor, :] = -4 * sign
            base = replace(
                inputs,
                scores=scores,
                score_mask=np.repeat(inputs.active[..., None], 5, -1),
                annual_borrow_rate_by_name=lending.annual_taker_rate[rows],
                borrow_rate_imputed=lending.rate_imputed[rows],
                borrow_rate_placeholder=lending.rate_placeholder[rows],
                shortable_by_borrow_source={
                    k: v[rows] for k, v in lending.availability.items()
                },
                loan_reference_prices=references[rows],
                hedge_annual_borrow_rate=hedge_rates[rows],
                bova11_close=hedge_close[rows],
                initial_hedge_reference_price=hedge_close[rows[0] - 1],
            )
            frozen = PolicyData(
                base,
                np.ones(base.active.shape),
                np.full(base.active.shape, 0.0004),
                np.full(len(rows), 0.0001),
                cdi[rows - 1],
                prior,
            )
            for scenario in scenarios[label]:
                variant = deepcopy(terms)
                variant["status"] = (
                    "explicit_fraction_sweep_and_cash_precision_engineering"
                )
                if label == "CIEL":
                    amended = next(
                        e
                        for e in variant["loan_cash_settlements"]
                        if e["isin"] == event["isin"]
                    )
                    amended["cash_per_share"] += (
                        -0.01
                        if scenario == "cent_low"
                        else 0.01
                        if scenario == "cent_high"
                        else 0
                    )
                    recognition = 6
                    payment = 6
                else:
                    amended = next(
                        e
                        for e in variant["share_distributions"]
                        if e["isin"] == event["isin"]
                    )
                    auction = amended["legs"][0]["fractional_auction"]
                    if scenario == "early":
                        auction["payment_date"] = auction["available_date"]
                        auction["payment_convention"] = (
                            "Explicit earliest-known signed settlement bound; not an observed receipt. Contractual ordinary cash-leg date unchanged."
                        )
                    elif scenario == "precision":
                        auction["cash_per_share"] = 565024 / 17690
                    elif scenario == "tiny_rent":
                        auction["zero_quantity_rent_through_payment"] = True
                    recognition = int(
                        np.searchsorted(dates, np.datetime64(auction["available_date"]))
                    ) - int(rows[0])
                    payment = int(
                        np.searchsorted(dates, np.datetime64(auction["payment_date"]))
                    ) - int(rows[0])
                variant_path = root / f"terms_{label}_{scenario}.json"
                if not variant_path.exists():
                    write_json_atomic(variant_path, variant)
                data = copy(frozen)
                data.inputs = apply_cash_calendar(
                    apply_corporate_replay(
                        base, variant, dates, sha256_file(variant_path)
                    ),
                    dates,
                    cdi,
                    run["cash_calendar"]["sha256"],
                )
                assert (
                    data.static is frozen.static
                    and data.references is frozen.references
                )
                model = CalibratedPolicy(
                    Calibration(
                        np.zeros(3), np.ones(3), np.array([0.002, 0.001, 0.0005]), 0
                    )
                )
                for capital in plan["capital"]:
                    started = perf_counter()
                    book = f"{label}_{scenario}_{sign}_{capital}"
                    if book in completed_books:
                        continue
                    config = policy_ledger_config(initial_capital_brl=capital)
                    result, targets, previous = exact_replay(
                        data, model, 0, len(rows), config=config
                    )
                    save_book(
                        root / book,
                        data,
                        result,
                        targets,
                        previous,
                        0,
                        0,
                        dict(
                            policy="synthetic_payment_bounds_optimizer",
                            scenario="base",
                            config=asdict(config),
                        ),
                    )
                    fixed, adaptive = (
                        data.initial_account(0, config),
                        data.initial_account(0, config),
                    )
                    fixed_error = adaptive_error = decision_error = 0.0
                    payment_rows = []
                    with torch.no_grad():
                        for day, target in enumerate(targets):
                            actual = account_decision(data, model, adaptive, day)
                            if day < len(rows) - 1:
                                decision_error = max(
                                    decision_error,
                                    float(np.max(np.abs(actual.numpy() - target))),
                                )
                            fixed.prepare_day(day)
                            before_claim = float(fixed.claims[name])
                            before_cash = float(fixed.trade_cash)
                            before_restricted = float(fixed.restricted[name])
                            before_quantity = float(fixed.shares[name])
                            original = fixed.loans.root_name[fixed.loans.root] == name
                            source_loans = [
                                dict(
                                    opened=int(fixed.loans.opened[i]),
                                    quantity=float(fixed.loans.quantity[i]),
                                    principal=float(fixed.loans.principal[i]),
                                    rent=float(fixed.loans.rent_due[i]),
                                    value_lag=int(fixed.loans.value_lag[i]),
                                    rate=float(fixed.loans.annual_rate[i]),
                                    accrual_end=int(fixed.loans.accrual_end[i]),
                                    return_day=int(fixed.loans.return_day[i]),
                                )
                                for i in np.flatnonzero(original)
                            ]
                            row = data.step(
                                fixed,
                                torch.from_numpy(target),
                                day,
                                terminal=day == len(rows) - 1,
                            )
                            data.step(
                                adaptive, actual, day, terminal=day == len(rows) - 1
                            )
                            fixed_error = max(
                                fixed_error, abs(float(row["nav"]) - result.nav[day])
                            )
                            adaptive_error = max(
                                adaptive_error,
                                abs(float(adaptive.nav) - result.nav[day]),
                            )
                            payment_rows.append(
                                dict(
                                    day=day,
                                    source_loans=source_loans,
                                    prepared_source_claim=before_claim,
                                    prepared_trade_cash=before_cash,
                                    prepared_restricted=before_restricted,
                                    prepared_source_quantity=before_quantity,
                                    source_claim=float(fixed.claims[name]),
                                    loan_cash_settlement_payment=float(
                                        row["loan_cash_settlement_payment"]
                                    ),
                                    source_quantity=float(fixed.shares[name]),
                                    source_restricted=float(fixed.restricted[name]),
                                    funding_cash=float(fixed.funding_cash),
                                    funding_restricted=float(fixed.funding_restricted),
                                    payments=[
                                        dict(due=due, amount=float(amount[name]))
                                        for due, amount in fixed.payments
                                        if float(amount[name]) != 0
                                    ],
                                )
                            )
                    write_json_atomic(root / book / "payments.json", payment_rows)
                    summary, _ = book_summary(data, result, previous, 0, 0)
                    report = dict(
                        book=book,
                        event=label,
                        scenario=scenario,
                        sign=sign,
                        capital=capital,
                        sessions=len(rows),
                        source=name,
                        successors=successors,
                        recognition=recognition,
                        payment=payment,
                        dates=[str(dates[rows[0]]), str(dates[rows[-1]])],
                        source_quantity_before=float(result.signed_shares[5, name]),
                        identical_intention_nav_error=fixed_error,
                        adaptive_nav_difference_bps=adaptive_error / capital * 1e4,
                        adaptive_target_difference=decision_error,
                        summary=summary,
                        seconds=perf_counter() - started,
                    )
                    reports.append(report)
                    write_json_atomic(root / "completed.json", reports)
                    print(
                        json.dumps({k: v for k, v in report.items() if k != "summary"}),
                        flush=True,
                    )
                    assert fixed_error < capital * 1e-8 and (
                        label == "CIEL" or sign * report["source_quantity_before"] > 0
                    )
    manifest = dict(
        status="payment_bounds_books_complete_pending_qualification",
        audit_root=str(root),
        plan=binding(root / "plan.json"),
        completed=binding(root / "completed.json"),
        final_invocation_seconds=perf_counter() - tick,
        summed_case_seconds=sum(c["seconds"] for c in reports),
        initial_invocation_wall_seconds=None,
        resumed_completed_books=len(completed_books),
    )
    write_json_atomic(root / "manifest.json", manifest)
    run["payment_bounds_audit"] = binding(root / "manifest.json")
    write_json_atomic(pointer, run)


if __name__ == "__main__":
    main()
