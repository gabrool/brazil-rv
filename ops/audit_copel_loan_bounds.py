"""Frozen adaptive Copel principal-allocation and net-loan conversion bounds."""

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


def source_cohorts(account, source):
    loans = account.loans
    rows = []
    for i in np.flatnonzero(loans.root_name[loans.root] == source):
        root = int(loans.root[i])
        rows.append(
            dict(
                root=root,
                name=int(loans.name[i]),
                opened=int(loans.opened[i]),
                quantity=float(loans.quantity[i]),
                principal=float(loans.principal[i]),
                rent_due=float(loans.rent_due[i]),
                fees_due=loans.fees_due[i].tolist(),
                return_day=int(loans.return_day[i]),
                annual_rate=float(loans.annual_rate[i]),
                fee_rate=loans.fee_rate[i].tolist(),
                fee_growth=loans.fee_growth[i].tolist(),
                value_lag=int(loans.value_lag[i]),
                minimum=float(loans.minimum[root]),
                root_fees=float(loans.root_fees[root]),
            )
        )
    return rows


def main():
    tick = perf_counter()
    torch.set_num_threads(1)
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    root = Path(run["root"]) / "copel_loan_bounds"
    root.mkdir(exist_ok=False)
    shutil.copyfile(__file__, root / "executed.py")
    shutil.copyfile(
        PROJECT / "research/preregistrations/v2_economic_data_scaling.md",
        root / "registration.md",
    )
    for folder in ("execution", "v2"):
        for path in (PROJECT / "research/src/brazil_rv" / folder).glob("*.py"):
            shutil.copyfile(path, root / f"executed_{folder}_{path.name}")
    terms, dates = load_corporate_replay(
        run["enat_settlement_terms"]["path"], run["enat_settlement_terms"]["sha256"]
    )
    events = {
        label: next(e for e in terms["share_distributions"] if e["isin"] == isin)
        for label, isin in [
            ("CPLE", "BRCPLECDAM13"),
        ]
    }
    plan = dict(
        events=events,
        capital=[10_000_000, 1_000_000, 5_000_000],
        signs=[1, -1],
        scenarios=["primary", "k0", "k1", "loan_early", "loan_late"],
        sessions="6 pre-effect +24 effect/following, all933",
        preferences="sin(axis*.31+localday*.003+head*.2), source +/-4 pre-effect then opposite; ON opposite source sign from effect; PN source sign until credit+2 then opposite. This forces staggered economic reductions without model scores.",
        risks="fixed beta1/idio.0004/market.0001; original .002/.001/.0005 calibration; frozen OLD PolicyData shallow copies",
        costs="old bundled4bp/no extraB3spot; correctedCDI/qualifiedloans/exactpriorrefs; not final corporate pricing",
        execution="Custody-first long December28 is fixed. Net borrowed source claims convert December27/28/29, preserving principal/rates/fees; existing owned successor custody still bounds returns. Flat/positive source pending returns retain the original convention, not covered by this net-short timing bound. K0/.2/1 are separate from timing; no stacking or recovered issuer K claim.",
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
    for label, event in events.items():
        name = isins.index(event["isin"])
        successors = [isins.index(leg["successor_isin"]) for leg in event["legs"]]
        effect = int(np.searchsorted(dates, np.datetime64(event["effective_date"])))
        rows = np.arange(effect - 6, effect + 24)
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
            scores[6:, successors[0], :] = -4 * sign
            scores[6:10, successors[1], :] = 4 * sign
            scores[10:, successors[1], :] = -4 * sign
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
            for scenario in plan["scenarios"]:
                variant = deepcopy(terms)
                variant["status"] = "explicit_copel_loan_allocation_timing_engineering"
                amended = next(
                    e
                    for e in variant["share_distributions"]
                    if e["isin"] == event["isin"]
                )
                k = 0 if scenario == "k0" else 1 if scenario == "k1" else 0.2
                conversion = (
                    7
                    if scenario == "loan_early"
                    else 9
                    if scenario == "loan_late"
                    else 8
                )
                for leg, fraction in zip(amended["legs"], (k, 1 - k)):
                    leg["loan_principal_fraction"] = fraction
                    if scenario.startswith("loan_"):
                        leg["loan_conversion_date"] = str(dates[rows[conversion]])
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
                            policy="synthetic_staggered_copel_optimizer",
                            scenario="base",
                            config=asdict(config),
                        ),
                    )
                    fixed, adaptive = (
                        data.initial_account(0, config),
                        data.initial_account(0, config),
                    )
                    fixed_error = adaptive_error = decision_error = 0.0
                    cohort_rows = []
                    with torch.no_grad():
                        for day, target in enumerate(targets):
                            actual = account_decision(data, model, adaptive, day)
                            if day < len(rows) - 1:
                                decision_error = max(
                                    decision_error,
                                    float(np.max(np.abs(actual.numpy() - target))),
                                )
                            before = source_cohorts(fixed, name)
                            fixed.prepare_day(day)
                            prepared = source_cohorts(fixed, name)
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
                            cohort_rows.append(
                                dict(
                                    day=day,
                                    before=before,
                                    prepared=prepared,
                                    after=source_cohorts(fixed, name),
                                    receipts=[
                                        (due, [float(q[i]) for i in successors])
                                        for due, q in fixed.custody.receipts
                                        if any(float(q[i]) != 0 for i in successors)
                                    ],
                                    restricted=[
                                        float(fixed.restricted[i])
                                        for i in [name, *successors]
                                    ],
                                )
                            )
                    write_json_atomic(root / book / "cohorts.json", cohort_rows)
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
                        k=k,
                        conversion=conversion,
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
                    assert (
                        fixed_error < capital * 1e-8
                        and sign * report["source_quantity_before"] > 0
                    )
    manifest = dict(
        status="copel_loan_bounds_books_complete_pending_qualification",
        audit_root=str(root),
        plan=binding(root / "plan.json"),
        completed=binding(root / "completed.json"),
        seconds=perf_counter() - tick,
    )
    write_json_atomic(root / "manifest.json", manifest)
    run["copel_loan_bounds_audit"] = binding(root / "manifest.json")
    write_json_atomic(pointer, run)


if __name__ == "__main__":
    main()
