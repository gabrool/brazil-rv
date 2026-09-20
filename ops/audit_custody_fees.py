"""Frozen ordinary physical-custody base and monthly payment bounds."""

from copy import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import polars as pl
import torch

from brazil_rv.execution.custody_fees import custody_schedule
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
    root = Path(run["root"]) / "custody_fees"
    root.mkdir(exist_ok=True)
    assert not (root / "plan.json").exists()
    shutil.copyfile(__file__, root / "executed.py")
    shutil.copyfile(
        PROJECT / "research/preregistrations/v2_economic_data_scaling.md",
        root / "registration.md",
    )
    for folder in ("execution", "v2"):
        for path in (PROJECT / "research/src/brazil_rv" / folder).glob("*.py"):
            shutil.copyfile(path, root / f"executed_{folder}_{path.name}")
    corporate = dict(
        spot_cost_model="b3_spot_2021", cost_bps_per_side=0, hedge_cost_bps_per_side=0
    )
    variants = {
        "no_custody": {},
        "primary": {"payment_lag": 0},
        "payment_3": {"payment_lag": 3},
        "payment_10": {"payment_lag": 10},
        "economic_long": {"payment_lag": 0, "custody_base": "economic_long"},
    }
    plan = dict(
        starts=["2023-05-02", "2024-05-02"],
        sessions=48,
        capital=[10000000, 1000000, 5000000],
        variants=variants,
        preferences="sin(axis*.31+localday*.07+head*.2), all933; no model scores",
        risks="beta1/idio.0004/market.0001, original .002/.001/.0005 calibration",
        contract="Ordinary physical stock equals economic signed stock plus all outstanding contract loan quantities minus unsettled signed spot purchases/sales and undelivered new-loan receipts. No loan renewal delivery; partial and delayed returns remain physical until actual return. Immediate unit splits transform pending quantities. Corporate share claims/delayed bonus/loan redemption require separate custody admission and fail on actual exposure; no names deleted.",
        costs="V41: monthly progressive annual brackets divided by12, 2023 exemption23084.39/2024 24164.73, active resident maintenance exempt. Last-session monthly calendar explicit from full accepted dates; not window terminal. One own-account CNPJ with one custodian, no fund/DR discount. Own observed close/last available own close valuation hypothesis, not a quote or loan reference. Unrounded fractional research units. Primary assessment-close debit versus +3/+10 equity sessions are client-payment HYPOTHESES, not observed receipts. Economic-long base is a separate one-factor diagnostic; all variants keep primary regular .5+2.5bp B3 and zero brokerage/1bp shortfall, replacing4bp. All remuneration/allocator estimates frozen.",
        source_receipts=binding(
            Path(run["root"]) / "custody_tariff_sources/qualification.json"
        ),
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
                selected = dict(changes)
                schedule = (
                    ()
                    if scenario == "no_custody"
                    else custody_schedule(
                        dates, start, str(dates[rows[-1]]), selected.pop("payment_lag")
                    )
                )
                config = replace(
                    policy_ledger_config(initial_capital_brl=capital),
                    **corporate,
                    custody_assessments=schedule,
                    **selected,
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
                        policy="synthetic_custody_bounds_optimizer",
                        scenario="base",
                        config=asdict(config),
                    ),
                )
                fixed, adaptive = (
                    data.initial_account(0, config),
                    data.initial_account(0, config),
                )
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
                                funding_cash=float(fixed.funding_cash),
                                funding_restricted=float(fixed.funding_restricted),
                                components=row["execution_charges"].tolist(),
                                cost=float(row["cost"]),
                                free_income=float(row["free_cash_income"]),
                                proceeds_income=float(row["short_proceeds_interest"]),
                                debit_financing=float(row["debit_financing"]),
                                interest=float(row["interest"]),
                                custody_base=float(row["custody_base"]),
                                custody_fee=float(row["custody_fee"]),
                                custody_paid=float(row["custody_payment"]),
                                custody_liability=float(row["custody_liability"]),
                                physical_error=float(
                                    np.max(
                                        np.abs(
                                            row["physical_custody"].numpy()
                                            - result.physical_custody[day]
                                        )
                                    )
                                ),
                                custody_marks=fixed.marks.tolist(),
                                loan_quantity=fixed.loans._by_name(
                                    fixed.loans.quantity
                                    * torch.as_tensor(fixed.loans.accrual_end < 0)
                                ).tolist(),
                                pending=[
                                    dict(due=d, quantity=q.tolist())
                                    for d, q in fixed.custody_fees.pending
                                ],
                            )
                        )
                write_json_atomic(root / book / "funding_and_costs.json", details)
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
                    execution_components=result.execution_charges.sum(0).tolist(),
                    custody_fee=float(result.custody_fee.sum()),
                    custody_payment=float(result.custody_payment.sum()),
                    terminal_custody_liability=float(result.custody_liability[-1]),
                    debit_sessions=sum(d["funding_cash"] < 0 for d in details),
                    funding_totals={
                        k: sum(d[k] for d in details)
                        for k in ("free_income", "proceeds_income", "debit_financing")
                    },
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


if __name__ == "__main__":
    main()
