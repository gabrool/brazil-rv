"""Frozen adaptive invoice and historical-minimum payment bounds."""

from copy import copy
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
    root = Path(run["root"]) / "loan_invoices"
    root.mkdir(exist_ok=False)
    shutil.copyfile(__file__, root / "executed.py")
    shutil.copyfile(
        PROJECT / "research/preregistrations/v2_economic_data_scaling.md",
        root / "registration.md",
    )
    for folder in ("execution", "v2"):
        for path in (PROJECT / "research/src/brazil_rv" / folder).glob("*.py"):
            shutil.copyfile(path, root / f"executed_{folder}_{path.name}")
    variants = {
        "primary": {},
        "contract_nearest": {"loan_invoice_convention": "contract_nearest"},
        "contract_down": {"loan_invoice_convention": "contract_down"},
        "contract_up": {"loan_invoice_convention": "contract_up"},
        "security_day_nearest": {"loan_invoice_convention": "security_day_nearest"},
        "minimum_pro_rata": {"loan_minimum_allocation": "pro_rata"},
    }
    plan = dict(
        starts=["2019-02-01", "2024-02-01"],
        sessions=64,
        capital=[10000000, 1000000, 5000000],
        variants=variants,
        preferences="sin(axis*.31+localday*.07+head*.2), all933; no model scores",
        risks="beta1/idio.0004/market.0001, original .002/.001/.0005 calibration",
        contract="V39 invoice hypotheses: separately round rent/totalB3 fees per original contract/payday or original security/payday; nearest/down/up cents. No registration grouping. Residual minimum pro-rata to returned original principal with fee credit, versus final primary. One-factor comparisons; no sourced invoice precision.",
        costs="old bundled4bp/no added B3spot, not final corporate pricing; correctedCDI/qualifiedloans/strictpriorrefs; OLD PolicyData shallow copies",
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
    write_json_atomic(root / "plan.json", plan)
    terms, dates = load_corporate_replay(
        run["enat_settlement_terms"]["path"], run["enat_settlement_terms"]["sha256"]
    )
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
    reports = []
    for start in plan["starts"]:
        first = int(np.searchsorted(dates, np.datetime64(start)))
        rows = np.arange(first, first + plan["sessions"])
        inputs = inputs_on_axes(store, dates, names, rows)
        scores = np.sin(
            names[None, :, None] * 0.31
            + np.arange(len(rows))[:, None, None] * 0.07
            + np.arange(5)[None, None, :] * 0.2
        ).astype(np.float32)
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
            initial_hedge_reference_price=hedge_close[first - 1],
        )
        frozen = PolicyData(
            base,
            np.ones(base.active.shape),
            np.full(base.active.shape, 0.0004),
            np.full(len(rows), 0.0001),
            cdi[rows - 1],
            np.load(store / "prior_reference_close.npy", mmap_mode="r")[rows],
        )
        data = copy(frozen)
        data.inputs = apply_cash_calendar(
            apply_corporate_replay(
                base, terms, dates, run["enat_settlement_terms"]["sha256"]
            ),
            dates,
            cdi,
            run["cash_calendar"]["sha256"],
        )
        assert data.static is frozen.static and data.references is frozen.references
        model = CalibratedPolicy(
            Calibration(np.zeros(3), np.ones(3), np.array([0.002, 0.001, 0.0005]), 0)
        )
        for scenario, changes in variants.items():
            for capital in plan["capital"]:
                started = perf_counter()
                config = replace(
                    policy_ledger_config(initial_capital_brl=capital), **changes
                )
                book = f"{start}_{scenario}_{capital}"
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
                        policy="synthetic_invoice_bounds_optimizer",
                        scenario="base",
                        config=asdict(config),
                    ),
                )
                fixed, adaptive = (
                    data.initial_account(0, config),
                    data.initial_account(0, config),
                )
                fixed.loans.record_payments = True
                fixed_error = adaptive_error = decision_error = 0.0
                details = []
                with torch.no_grad():
                    for day, target in enumerate(targets):
                        actual = account_decision(data, model, adaptive, day)
                        if day < len(rows) - 1:
                            decision_error = max(
                                decision_error,
                                float(np.max(np.abs(actual.numpy() - target))),
                            )
                        row = data.step(
                            fixed,
                            torch.from_numpy(target),
                            day,
                            terminal=day == len(rows) - 1,
                        )
                        data.step(adaptive, actual, day, terminal=day == len(rows) - 1)
                        fixed_error = max(
                            fixed_error, abs(float(row["nav"]) - result.nav[day])
                        )
                        adaptive_error = max(
                            adaptive_error, abs(float(adaptive.nav) - result.nav[day])
                        )
                        details.append(
                            dict(
                                day=day,
                                payment=fixed.loans.last_payment,
                                rent_due=float(fixed.loans.rent_due.sum()),
                                fees_due=float(fixed.loans.fees_due.sum()),
                                minimum=float(fixed.loans.minimum_provision.sum()),
                                credit=float(fixed.loans.minimum_credit.sum()),
                                cash_liability=float(fixed.loans.cash_liability),
                                funding_cash=float(fixed.funding_cash),
                                funding_restricted=float(fixed.funding_restricted),
                                borrowed_expense=float(row["borrow"]),
                                paid=float(row["borrow_paid"]),
                            )
                        )
                write_json_atomic(root / book / "payments.json", details)
                summary, _ = book_summary(data, result, previous, 0, 0)
                report = dict(
                    book=book,
                    start=start,
                    scenario=scenario,
                    capital=capital,
                    sessions=len(rows),
                    dates=[str(dates[rows[0]]), str(dates[rows[-1]])],
                    identical_intention_nav_error=fixed_error,
                    adaptive_nav_difference_bps=adaptive_error / capital * 1e4,
                    adaptive_target_difference=decision_error,
                    max_minimum_credit=float(result.loan_minimum_credit.max()),
                    rounding_rent_fee=result.loan_invoice_adjustment.sum(0).tolist(),
                    summary=summary,
                    seconds=perf_counter() - started,
                )
                reports.append(report)
                write_json_atomic(root / "completed.json", reports)
                print(
                    json.dumps({k: v for k, v in report.items() if k != "summary"}),
                    flush=True,
                )
                assert fixed_error < 1e-6
    manifest = dict(
        status="complete_pending_qualification",
        audit_root=str(root),
        plan=binding(root / "plan.json"),
        completed=binding(root / "completed.json"),
        seconds=perf_counter() - tick,
        summed_case_seconds=sum(c["seconds"] for c in reports),
    )
    write_json_atomic(root / "manifest.json", manifest)
    run["loan_invoice_audit"] = binding(root / "manifest.json")
    write_json_atomic(pointer, run)


if __name__ == "__main__":
    main()
