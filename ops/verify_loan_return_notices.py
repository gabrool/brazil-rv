"""Adaptive lifecycle engineering books on bounded original market paths.

Synthetic, frozen preferences exercise the actual constrained allocator, not a
forecast fit or corrected model profitability. Original stores remain immutable.
"""

from copy import copy
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import time

import numpy as np
import polars as pl
import torch

from brazil_rv.execution.loan_contracts import LoanRecall
from brazil_rv.execution.portfolio_policy import (
    CalibratedPolicy,
    PolicyData,
    account_decision,
    exact_replay,
    policy_ledger_config,
)
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.cash_calendar import apply_cash_calendar
from brazil_rv.v2.corporate_replay import apply_corporate_replay, load_corporate_replay
from brazil_rv.v2.lending_archive import load_lending_borrow_panels
from brazil_rv.v2.portfolio_inputs import Calibration
from brazil_rv.v2.portfolio_readouts import book_summary, save_book
from verify_corporate_replay import inputs_on_axes

PROJECT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def resolved(pointer, key):
    item = pointer[key]
    path = Path(item["path"])
    if not path.is_absolute():
        path = PROJECT / path
    assert sha256_file(path) == item["sha256"], key
    return read(path)


def main(label=None):
    started = time.perf_counter()
    torch.set_num_threads(1)
    pointer_path = PROJECT / "docs/v2_economic_data_scaling_run.json"
    pointer = read(pointer_path)
    root = Path(pointer["root"]) / "loan_return_notices"
    if label:
        root = root / label
    root.mkdir(exist_ok=False)
    shutil.copyfile(__file__, root / "executed.py")
    for name in (
        "loan_contracts",
        "portfolio_account",
        "portfolio_policy",
        "stateful_ledger",
        "allocation",
    ):
        shutil.copyfile(
            PROJECT / f"research/src/brazil_rv/execution/{name}.py",
            root / f"executed_{name}.py",
        )
    terms, calendar = load_corporate_replay(
        **{
            "path": pointer["corporate_replay"]["path"],
            "expected_sha256": pointer["corporate_replay"]["sha256"],
        }
    )
    store = Path(terms["store"]["root"])
    isins = np.load(store / "isin_index.npy").tolist()
    names = np.arange(len(isins))
    assert calendar[-1] <= np.datetime64("2024-12-30")
    binding = pointer["qualified_lending"]
    lending = load_lending_borrow_panels(
        Path(binding["root"]),
        expected_manifest_sha256=binding["manifest_sha256"],
        canonical_dates=calendar.astype(object).tolist(),
        canonical_isins=isins,
    )
    panels = resolved(pointer, "loan_source_panels")["panels"]
    assert sha256_file(Path(panels["path"])) == panels["sha256"]
    with np.load(panels["path"]) as archive:
        references = archive["loan_reference_prices"]
        hedge_rates = archive["hedge_annual_borrow_rate"]
    cash = resolved(pointer, "cash_calendar")["panel"]
    assert sha256_file(Path(cash["path"])) == cash["sha256"]
    with np.load(cash["path"]) as archive:
        cdi = archive["cdi_returns"]
    hedge_source = resolved(pointer, "bova_loan_reference_audit")["data"]
    hedge_frame = pl.read_parquet(hedge_source["path"])
    hedge_by_date = dict(zip(hedge_frame["trade_date"], hedge_frame["close_brl"]))
    hedge_close = np.array(
        [hedge_by_date.get(d, np.nan) for d in calendar.astype(object)]
    )
    # Freeze every case, preference coefficient and notice before any outcomes.
    plan = {
        "windows": ["2019-08-01", "2024-05-02"],
        "sessions": 128,
        "capital": [10_000_000, 1_000_000, 5_000_000],
        "scenarios": ["base", "denied", "recall2", "recall4"],
        "recall_notice_offset": 50,
        "term": 63,
        "preferences": "sin(axis*.31 + session*.003 + head*.2); coefficients .002/.001/.0005",
        "risk": "fixed engineering beta1, idiosyncratic variance .0004, market variance .0001",
        "purpose": "adaptive accounting engineering, no calibrated or neural forecast, no alpha inference",
        "execution_cost": "unchanged bundled4bp bridge; no additional B3 spot component",
        "source_bindings": {
            k: pointer[k]
            for k in (
                "corporate_replay",
                "qualified_lending",
                "loan_source_panels",
                "cash_calendar",
                "bova_loan_reference_audit",
            )
        },
    }
    write_json_atomic(root / "frozen_plan.json", plan)
    reports = []
    for window in plan["windows"]:
        first = int(np.searchsorted(calendar, np.datetime64(window)))
        rows = np.arange(first, first + plan["sessions"])
        inputs = inputs_on_axes(store, calendar, names, rows)
        scores = np.sin(
            names[None, :, None] * 0.31
            + np.arange(len(rows))[:, None, None] * 0.003
            + np.arange(5)[None, None, :] * 0.2
        ).astype(np.float32)
        inputs = replace(
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
        shape = inputs.active.shape
        prior_prices = np.load(store / "prior_reference_close.npy", mmap_mode="r")[rows]
        frozen = PolicyData(
            inputs,
            np.ones(shape),
            np.full(shape, 0.0004),
            np.full(len(rows), 0.0001),
            cdi[rows - 1],
            prior_prices,
        )
        data = copy(frozen)
        data.inputs = apply_cash_calendar(
            apply_corporate_replay(
                inputs, terms, calendar, pointer["corporate_replay"]["sha256"]
            ),
            calendar,
            cdi,
            pointer["cash_calendar"]["sha256"],
        )
        assert data.static is frozen.static and data.references is frozen.references
        model = CalibratedPolicy(
            Calibration(np.zeros(3), np.ones(3), np.array([0.002, 0.001, 0.0005]), 0)
        )
        baseline_targets = None
        for capital in plan["capital"]:
            for scenario in plan["scenarios"]:
                label = f"{window}_{capital}_{scenario}"
                started_case = time.perf_counter()
                recalls = (
                    tuple(
                        LoanRecall(
                            int(i),
                            50,
                            50 + int(scenario[-1]),
                            scenario + " analyst hypothesis",
                        )
                        for i in range(len(names) + 1)
                    )
                    if scenario.startswith("recall")
                    else ()
                )
                config = policy_ledger_config(
                    initial_capital_brl=capital,
                    approve_loan_renewals=scenario != "denied",
                    loan_recalls=recalls,
                )
                result, targets, previous = exact_replay(
                    data, model, 0, len(rows), config=config
                )
                account = data.initial_account(0, config)
                errors = dict(
                    nav=0.0,
                    cash=0.0,
                    restricted=0.0,
                    shares=0.0,
                    decision=0.0,
                    overdue=0.0,
                )
                with torch.no_grad():
                    for day in range(len(rows)):
                        target = account_decision(data, model, account, day)
                        if day < len(rows) - 1:
                            errors["decision"] = max(
                                errors["decision"],
                                float(np.max(np.abs(target.numpy() - targets[day]))),
                            )
                        row = data.step(
                            account, target, day, terminal=day == len(rows) - 1
                        )
                        for key, value in {
                            "nav": abs(account.nav.item() - result.nav[day]),
                            "cash": abs(account.cash.item() - result.free_cash[day]),
                            "restricted": abs(
                                account.restricted.sum().item()
                                - result.restricted_cash[day]
                                - result.hedge_restricted_cash[day]
                            ),
                            "shares": np.max(
                                np.abs(
                                    account.shares.numpy()
                                    - np.r_[
                                        result.signed_shares[day],
                                        result.hedge_signed_shares[day],
                                    ]
                                )
                            ),
                            "overdue": abs(
                                row["loan_overdue_principal"].item()
                                - result.loan_overdue_principal[day]
                            ),
                        }.items():
                            errors[key] = max(errors[key], float(value))
                # Isolate accounting from independently re-solved adaptive QPs.
                identical_intentions = data.initial_account(0, config)
                fixed_error = 0.0
                with torch.no_grad():
                    for day, target in enumerate(targets):
                        row = data.step(
                            identical_intentions,
                            torch.from_numpy(target),
                            day,
                            terminal=day == len(rows) - 1,
                        )
                        difference = abs(float(row["nav"]) - result.nav[day])
                        fixed_error = max(fixed_error, difference)
                        if difference > capital * 1e-8:
                            evidence = dict(
                                case=label,
                                first_day=day,
                                fixed_error=difference,
                                adaptive_errors=errors,
                                original_loans=[
                                    asdict(c)
                                    for c in result.loan_charges
                                    if c.session == day
                                ],
                                account_roots=identical_intentions.loans.root_name.tolist(),
                                account_root_opened=identical_intentions.loans.root_opened.tolist(),
                                account_principal=identical_intentions.loans.principal.tolist(),
                                original_renewals=[asdict(r) for r in result.loan_renewals if r.session == day],
                                account_renewals=[asdict(r) for r in identical_intentions.loans.renewals if r.session == day],
                            )
                            write_json_atomic(root / f"{label}_failure.json", evidence)
                            raise ValueError((label, day, difference))
                if scenario == "base":
                    baseline_targets = targets
                else:
                    prefix = 50 if recalls else 60
                    np.testing.assert_array_equal(
                        targets[:prefix], baseline_targets[:prefix]
                    )
                summary, _ = book_summary(data, result, previous, 0, 0)
                save_book(
                    root / label,
                    data,
                    result,
                    targets,
                    previous,
                    0,
                    0,
                    {
                        "scenario": scenario,
                        "policy": "synthetic_calibrated_optimizer",
                        "config": asdict(config),
                    },
                )
                report = dict(
                    case=label,
                    summary=summary,
                    account_errors=errors,
                    identical_intentions_nav_error=fixed_error,
                    adaptive_target_difference_le_2e_6=errors["decision"] <= 2e-6,
                    adaptive_nav_difference_bps=errors["nav"] / capital * 1e4,
                    notices=len(result.loan_return_notices),
                    overdue_sessions=int((result.loan_overdue_principal > 0).sum()),
                    seconds=time.perf_counter() - started_case,
                )
                reports.append(report)
                write_json_atomic(root / "completed_cases.json", reports)
                print(
                    json.dumps({k: v for k, v in report.items() if k != "summary"}),
                    flush=True,
                )
    report = {
        "status": "identical_intention_money_verified_adaptive_solver_difference_measured_not_economic_admission",
        "plan": plan,
        "cases": reports,
        "seconds": time.perf_counter() - started,
        "reproducer_sha256": sha256_file(Path(__file__)),
    }
    digest = write_json_atomic(root / "manifest.json", report)
    pointer["loan_return_notice_audit"] = {
        "path": str(root / "manifest.json"),
        "sha256": digest,
    }
    write_json_atomic(pointer_path, pointer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label")
    main(parser.parse_args().label)
