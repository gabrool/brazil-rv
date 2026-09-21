"""Reuse frozen one-factor terms on actually exposed fresh-fit ensembles."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import pickle
from time import perf_counter

import numpy as np
import torch

from brazil_rv.execution.custody_fees import CustodyAssessment
from brazil_rv.execution.loan_contracts import LoanRecall
from brazil_rv.execution.portfolio_policy import CalibratedPolicy, exact_replay
from brazil_rv.execution.spot_costs import MonthlySpotTariff
from brazil_rv.execution.stateful_ledger import LedgerConfig
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.foundation_readouts import new_panel
from brazil_rv.v2.objective_readouts import calibration, rank_view
from brazil_rv.v2.opportunity_research import benchmark_for
from brazil_rv.v2.performance import performance
from brazil_rv.v2.portfolio_readouts import save_book
from brazil_rv.v2.portfolio_training import windows
from brazil_rv.v2.research_rounds import _git_identity
from run_economic_replays import phase_data

PROJECT = Path(__file__).resolve().parents[1]


def exposed(phase, book, arrays, terms, names):
    days, daily = book["state_dates"], book["daily"]
    loans = bool(np.any(arrays["loan_outstanding_principal"] > 0))
    for field, short in (
        ("delivery_delays", False),
        ("loan_fraction_conventions", True),
    ):
        if field in phase:
            for event in terms["share_distributions"]:
                if (
                    event["isin"] not in phase[field]
                    or event["effective_date"] not in days
                ):
                    continue
                day = days.index(event["effective_date"])
                held = (
                    arrays["signed_shares"][day - 1, names.index(event["isin"])]
                    if day
                    else 0
                )
                if (held < 0) if short else (held != 0):
                    return True
            return False
    if "recall_deadline" in phase:
        return loans
    config = phase["config"]
    assert len(config) == 1
    field = next(iter(config))
    if field in {
        "execution_shortfall_bps",
        "execution_brokerage_bps",
        "spot_execution_phase",
        "spot_invoice_convention",
    }:
        return bool(np.any(np.asarray(daily["turnover"]) > 0))
    if field == "short_proceeds_remuneration":
        return bool(np.any(np.asarray(daily["short_proceeds_income_bps"]) != 0))
    if field == "annual_debit_spread":
        return bool(np.any(np.asarray(daily["debit_financing_bps"]) != 0))
    if field == "loan_invoice_convention":
        return bool(np.any(arrays["loan_payment"] != 0))
    if field == "loan_minimum_allocation":
        return loans and days[0] < "2020-10-26"
    if field in {"loan_term_sessions", "approve_loan_renewals"}:
        return loans
    if field in {"custody_base", "custody_assessments"}:
        return bool(np.any(arrays["custody_fee"] != 0))
    if field == "custody_claim_fraction":
        return bool(
            np.any((arrays["custody_fee"] > 0) & (arrays["custody_claim_base"] > 0))
        )
    assert field == "unrecovered_spot_trading_bps", field
    tariffs = book["provenance"]["config"]["monthly_spot_tariffs"]
    return any(
        d < "2021-02-02"
        and not any(t["valid_from"] <= d <= t["valid_to"] for t in tariffs)
        for d in days
    )


def freeze(run):
    parent = bound_json(run["stage_c_sensitivity_plan"])
    primary = bound_json(run["stage_c_data_replay_plan"])
    out = Path(run["root"]) / "matched_refit_sensitivities"
    out.mkdir(exist_ok=False)
    plan = dict(
        primary=run["stage_c_data_replay_plan"],
        original_hypotheses=run["stage_c_sensitivity_plan"],
        phases=parent["phases"],
        scope="All48 fresh-fit ensemble/capital/fold books, same previously frozen one-factor hypotheses. Evaluate only actual exposure under deterministic predicates; preserve every unexposed skip. No new training, combined worst case, price/availability change or manufactured exposure.",
        planned_primary_ensembles=len(primary["arms"])
        * len(primary["folds"])
        * len(primary["capitals"]),
        additional_conditional="Cielo cent and opposing-fill daytrade branches stay conditional on actual new exposure; previous zero-exposure controls are not bounds. Existing January/Natura/2023 corporate windows do not intersect these flat-start screen folds.",
        driver=binding(Path(__file__)),
    )
    write_json_atomic(out / "plan.json", plan)
    run["stage_c_refit_sensitivity_plan"] = binding(out / "plan.json")
    write_json_atomic(PROJECT / "docs/v2_economic_data_scaling_run.json", run)
    print(
        json.dumps(
            dict(plan=run["stage_c_refit_sensitivity_plan"], phases=len(plan["phases"]))
        ),
        flush=True,
    )


def execute(run):
    torch.set_num_threads(1)
    code = _git_identity()
    reference = run["stage_c_refit_sensitivity_plan"]
    plan = bound_json(reference)
    assert sha256_file(Path(__file__)) == plan["driver"]["sha256"]
    primary = bound_json(plan["primary"])
    parent = Path(plan["primary"]["path"]).parent
    replays = bound_json(binding(parent / "replays.json"))
    inputs = bound_json(primary["inputs"])
    assert sha256_file(Path(inputs["cache"]["path"])) == inputs["cache"]["sha256"]
    with Path(inputs["cache"]["path"]).open("rb") as f:
        data = pickle.load(f)
    names = list(data.inputs.security_ids)
    terms = bound_json(primary["terms"])
    out = Path(reference["path"]).parent
    progress = out / "replays.json"
    old = (
        json.loads(progress.read_text())
        if progress.exists()
        else dict(completed=[], skipped=[], processed_primary=[])
    )
    completed, skipped, processed = (
        old["completed"],
        old["skipped"],
        old["processed_primary"],
    )
    done = {r["key"] for r in [*completed, *skipped]}
    prior = Path(primary["prior_root"])
    benchmark = Path(
        json.loads((PROJECT / "docs/v2_opportunity_run.json").read_text())["root"]
    )
    source = bound_json(primary["economic_account"])["primary_config"].copy()
    source["monthly_spot_tariffs"] = tuple(
        MonthlySpotTariff(**v) for v in source["monthly_spot_tariffs"]
    )
    source["custody_assessments"] = tuple(
        CustodyAssessment(**v) for v in source["custody_assessments"]
    )
    primary_config = LedgerConfig(**source)

    def save_progress():
        write_json_atomic(
            progress,
            dict(
                status="complete"
                if len(processed) == plan["planned_primary_ensembles"]
                else "waiting_for_qualified_primary",
                plan=reference,
                completed=completed,
                skipped=skipped,
                processed_primary=processed,
            ),
        )

    panels = {}
    for base in replays["completed"]:
        if not base["key"].endswith("/ensemble") or base["key"] in processed:
            continue
        proof = parent / "qualification" / (base["key"].replace("/", "_") + ".json")
        if not proof.exists():
            continue
        qualified = bound_json(binding(proof))
        assert qualified["passed"] and qualified["book"] == base["book"]
        assert not qualified["loan_cash_bounds_pending"], (
            "Actual loan cash requires the separately frozen cash-price exposure bound first"
        )
        _, capital, arm, fold, _ = base["key"].split("/")
        capital = int(capital)
        book = bound_json(base["book"])
        with np.load(Path(base["book"]["path"]).parent / "account.npz") as z:
            arrays = {
                k: z[k]
                for k in (
                    "signed_shares",
                    "loan_outstanding_principal",
                    "loan_payment",
                    "custody_fee",
                    "custody_claim_base",
                )
            }
        rows = windows(prior, data, fold)["evaluation"]
        start, stop = int(rows[0]), int(rows[-1]) + 1
        if (arm, fold) not in panels:
            forecasts, valid, sources = new_panel(
                Path(primary["fit_root"]), data, arm, fold, rows, "raw"
            )
            panels[arm, fold] = forecasts["ensemble"], valid, sources
        ranks, valid, sources = panels[arm, fold]
        assert sources == book["provenance"]["forecast_sources"]
        mapping = calibration(
            bound_json(book["provenance"]["mapping"])["arms"][
                "C6" if arm == "C6" else "TE_all"
            ]
        )
        bench = benchmark_for(
            benchmark, data.inputs.dates[start:stop], data.inputs.dates[start - 1]
        )
        for phase in plan["phases"]:
            if phase["capital"] != capital:
                continue
            key = f"{phase['name']}/{capital}/{arm}/{fold}/ensemble"
            if key in done:
                continue
            if not exposed(phase, book, arrays, terms, names):
                skipped.append(
                    dict(
                        key=key,
                        baseline=base,
                        reason="Frozen exposure predicate false; not a numerical bound",
                    )
                )
                done.add(key)
                save_progress()
                continue
            tick = perf_counter()
            path = out / "books" / key
            assert not path.exists(), (
                "Preserve partial output and resume its exact saved boundary"
            )
            values = phase.get("config", {}).copy()
            if "custody_assessments" in values:
                values["custody_assessments"] = tuple(
                    CustodyAssessment(**v) for v in values["custody_assessments"]
                )
            if "recall_deadline" in phase:
                values["loan_recalls"] = tuple(
                    LoanRecall(
                        i,
                        start + 50,
                        start + 50 + phase["recall_deadline"],
                        "registered universal recall at local session50; analyst hypothesis",
                    )
                    for i in range(934)
                )
            config = replace(primary_config, initial_capital_brl=capital, **values)
            view = rank_view(phase_data(data, phase), rows, ranks, valid)
            provenance = dict(
                book["provenance"],
                implementation=code,
                phase=phase["name"],
                phase_hypothesis=phase,
                config=asdict(config),
                sensitivity_plan=reference,
                baseline=base["book"],
            )
            result, targets, previous = exact_replay(
                view, CalibratedPolicy(mapping), start, stop, config=config
            )
            save_book(path, view, result, targets, previous, start, start, provenance)
            write_json_atomic(
                path / "performance.json",
                performance(
                    result.daily_net_return, view.inputs.cdi_returns[start:stop], bench
                ),
            )
            write_json_atomic(
                path / "loan_cash_payments.json",
                [asdict(p) for p in result.loan_cash_payments],
            )
            completed.append(
                dict(
                    key=key,
                    book=binding(path / "book.json"),
                    baseline=base,
                    performance=binding(path / "performance.json"),
                    loan_cash_payments=binding(path / "loan_cash_payments.json"),
                    seconds=perf_counter() - tick,
                    economics_unresolved=bool(result.economics_unresolved),
                )
            )
            done.add(key)
            save_progress()
            print(
                json.dumps(dict(completed=key, seconds=completed[-1]["seconds"])),
                flush=True,
            )
        processed.append(base["key"])
        save_progress()
    save_progress()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    freeze(run) if args.freeze else execute(run)
