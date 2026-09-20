"""Frozen Stage C accounting/source contrasts on original forecast coordinates."""

import argparse
from copy import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import polars as pl
import torch

from brazil_rv.execution.custody_fees import CustodyAssessment
from brazil_rv.execution.loan_contracts import LoanRecall
from brazil_rv.execution.portfolio_policy import CalibratedPolicy, exact_replay
from brazil_rv.execution.spot_costs import MonthlySpotTariff
from brazil_rv.execution.stateful_ledger import LedgerConfig
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.corporate_replay import apply_corporate_replay, load_corporate_replay
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.foundation_readouts import new_panel
from brazil_rv.v2.lending_archive import load_lending_borrow_panels
from brazil_rv.v2.objective_readouts import calibration, rank_view, read_panel
from brazil_rv.v2.opportunity_research import benchmark_for
from brazil_rv.v2.performance import performance
from brazil_rv.v2.portfolio_readouts import save_book
from brazil_rv.v2.portfolio_training import load_data, windows

PROJECT = Path(__file__).resolve().parents[1]


def phase_data(data, phase):
    """Change only explicit delivery/fraction hypotheses on frozen PolicyData."""
    delays = phase.get("delivery_delays", {})
    provisioned = phase.get("loan_fraction_conventions", {})
    if not delays and not provisioned:
        return data
    amended = copy(data)
    events = []
    for event in data.inputs.share_distributions:
        name = data.inputs.security_ids[event.source_index]
        if name in delays or name in provisioned:
            legs = []
            for leg in event.legs:
                if name in delays:
                    assert leg.delivery_session is not None
                    leg = replace(
                        leg, delivery_session=leg.delivery_session + delays[name]
                    )
                if name in provisioned:
                    assert leg.fractional_auction is not None
                    leg = replace(
                        leg,
                        fractional_auction=replace(
                            leg.fractional_auction,
                            provision_loan_fractions=provisioned[name],
                        ),
                    )
                legs.append(leg)
            event = replace(event, legs=tuple(legs))
        events.append(event)
    amended.inputs = replace(data.inputs, share_distributions=tuple(events))
    return amended


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    stage_root = Path(run["stage_c_root"])
    plan_binding = run["stage_c_plan"] if args.plan is None else binding(args.plan)
    plan = bound_json(plan_binding)
    root = Path(plan.get("output_root", stage_root))
    root.mkdir(exist_ok=True)
    (root / "replay_attempts").mkdir(exist_ok=True)
    assert bound_json(binding(stage_root / "baseline_control/manifest.json"))["passed"]
    attempt = (
        root
        / "replay_attempts"
        / str(len(list((root / "replay_attempts").glob("*"))) + 1)
    )
    attempt.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, attempt / "executed.py")
    runtime = {
        str(p.relative_to(PROJECT)): binding(p)
        for directory in ("execution", "v2")
        for p in (PROJECT / "research/src/brazil_rv" / directory).glob("*.py")
    }
    write_json_atomic(attempt / "runtime.json", runtime)
    account = bound_json(plan["economic_account"])
    cfg = account["primary_config"].copy()
    cfg["monthly_spot_tariffs"] = tuple(
        MonthlySpotTariff(**x) for x in cfg["monthly_spot_tariffs"]
    )
    cfg["custody_assessments"] = tuple(
        CustodyAssessment(**x) for x in cfg["custody_assessments"]
    )
    primary = LedgerConfig(**cfg)
    corporate_terms = plan.get("corporate_terms", account["corporate_terms"])
    terms, dates = load_corporate_replay(
        corporate_terms["path"], corporate_terms["sha256"]
    )
    prior, foundation = Path(plan["prior_root"]), Path(plan["foundation_root"])
    frozen, cache = load_data(prior, "C6")
    indices = frozen.inputs.session_indices
    assert len(frozen.inputs.security_ids) == 933
    np.testing.assert_array_equal(
        dates[indices], np.asarray(frozen.inputs.dates, dtype="datetime64[D]")
    )
    assert dates[indices[-1]] <= np.datetime64("2024-12-30")
    panels = bound_json(run["loan_source_panels"])["panels"]
    assert sha256_file(Path(panels["path"])) == panels["sha256"]
    with np.load(panels["path"]) as z:
        references, hedge_rates = (
            z["loan_reference_prices"][indices],
            z["hedge_annual_borrow_rate"][indices],
        )
    cash = bound_json(run["cash_calendar"])["panel"]
    assert sha256_file(Path(cash["path"])) == cash["sha256"]
    with np.load(cash["path"]) as z:
        cdi = z["cdi_returns"]
    source = run["qualified_lending"]
    lending = load_lending_borrow_panels(
        Path(source["root"]),
        expected_manifest_sha256=source["manifest_sha256"],
        canonical_dates=dates.astype(object).tolist(),
        canonical_isins=list(frozen.inputs.security_ids),
    )
    hedge_source = bound_json(run["bova_loan_reference_audit"])["data"]
    assert sha256_file(Path(hedge_source["path"])) == hedge_source["sha256"]
    hedge = pl.read_parquet(hedge_source["path"])
    hmap = dict(zip(hedge["trade_date"], hedge["close_brl"]))
    closes = np.array([hmap.get(d, np.nan) for d in dates.astype(object)])
    accounting = copy(frozen)
    accounting.inputs = apply_corporate_replay(
        replace(frozen.inputs, loan_reference_prices=references),
        terms,
        dates,
        corporate_terms["sha256"],
    )
    # The frozen cache includes pre-source warmup outside every requested book.
    # Preserve that unused prefix; every actual replay requires corrected coverage.
    corrected_cash = frozen.inputs.cdi_returns.copy()
    covered = np.isfinite(cdi[indices])
    corrected_cash[covered] = cdi[indices][covered]
    accounting.inputs = replace(
        accounting.inputs,
        cdi_returns=corrected_cash,
        source_artifact_hashes={
            **(accounting.inputs.source_artifact_hashes or {}),
            "cash_calendar": run["cash_calendar"]["sha256"],
        },
    )
    write_json_atomic(
        attempt / "cash_scope.json",
        dict(
            unchanged_outside_source_dates=[str(d) for d in dates[indices][~covered]],
            actual_books_require_full_corrected_coverage=True,
        ),
    )
    sourced = copy(accounting)
    sourced.inputs = replace(
        accounting.inputs,
        annual_borrow_rate_by_name=lending.annual_taker_rate[indices],
        borrow_rate_imputed=lending.rate_imputed[indices],
        borrow_rate_placeholder=lending.rate_placeholder[indices],
        shortable_by_borrow_source={
            k: v[indices] for k, v in lending.availability.items()
        },
        hedge_annual_borrow_rate=hedge_rates,
        bova11_close=closes[indices],
        initial_hedge_reference_price=closes[indices[0] - 1],
    )
    sourced.shortable = sourced.inputs.shortable_by_borrow_source["borrow_balance"]
    for data in (accounting, sourced):
        for key in (
            "static",
            "references",
            "beta",
            "diagonal",
            "factor",
            "volatility",
            "prior_cdi",
        ):
            assert getattr(data, key) is getattr(frozen, key)
        assert data.inputs.active is frozen.inputs.active
    benchmark = Path(
        json.loads((PROJECT / "docs/v2_opportunity_run.json").read_text())["root"]
    )
    progress_file = root / "replays.json"
    completed = (
        json.loads(progress_file.read_text())["completed"]
        if progress_file.exists()
        else []
    )
    done = {c["key"] for c in completed}
    phase_views = {}
    for fold in plan["folds"]:
        rows = windows(prior, frozen, fold)["evaluation"]
        assert covered[rows].all(), "requested book lacks corrected cash coverage"
        start, stop = int(rows[0]), int(rows[-1] + 1)
        mappings = json.loads((prior / "phase3/mappings" / f"{fold}.json").read_text())
        bench = benchmark_for(
            benchmark, frozen.inputs.dates[start:stop], frozen.inputs.dates[start - 1]
        )
        for arm in plan["arms"]:
            if arm == "C6":
                forecasts, _, valid, sources = read_panel(
                    prior, frozen, arm, "neutral", fold, rows
                )
            else:
                forecasts, valid, sources = new_panel(
                    foundation, frozen, arm, fold, rows, "raw"
                )
            mapping = calibration(mappings["arms"]["C6" if arm == "C6" else "TE_all"])
            for phase in plan["phases"]:
                members = plan["members"] if phase["members"] == "all" else ["ensemble"]
                base_data = accounting if phase["name"] == "accounting" else sourced
                if phase["name"] not in phase_views:
                    phase_views[phase["name"]] = phase_data(base_data, phase)
                data = phase_views[phase["name"]]
                config_values = phase.get("config", {}).copy()
                if "custody_assessments" in config_values:
                    config_values["custody_assessments"] = tuple(
                        CustodyAssessment(**x)
                        for x in config_values["custody_assessments"]
                    )
                if "recall_deadline" in phase:
                    config_values["loan_recalls"] = tuple(
                        LoanRecall(
                            i,
                            start + 50,
                            start + 50 + phase["recall_deadline"],
                            "registered universal recall at local session50; analyst hypothesis",
                        )
                        for i in range(934)
                    )
                config = replace(
                    primary, initial_capital_brl=phase["capital"], **config_values
                )
                for member in members:
                    key = f"{phase['name']}/{phase['capital']}/{arm}/{fold}/{member}"
                    if "included_keys" in plan and key not in plan["included_keys"]:
                        continue
                    if key in done:
                        record = next(c for c in completed if c["key"] == key)
                        bound_json(record["book"])
                        continue
                    tick = perf_counter()
                    output = root / "books" / key
                    assert not output.exists(), (
                        "retain partial output; resume saved work explicitly"
                    )
                    view = rank_view(data, rows, forecasts[member], valid)
                    provenance = dict(
                        scenario="base",
                        policy="equal_rank",
                        phase=phase["name"],
                        phase_hypothesis=phase,
                        config=asdict(config),
                        stage_c_plan=plan_binding,
                        economic_account=plan["economic_account"],
                        corporate_terms=corporate_terms,
                        forecast_sources=sources,
                        old_policy_cache=cache,
                        mapping=binding(prior / "phase3/mappings" / f"{fold}.json"),
                        member=member,
                        runtime=binding(attempt / "runtime.json"),
                        heldout_accessed=False,
                    )
                    result, targets, previous = exact_replay(
                        view, CalibratedPolicy(mapping), start, stop, config=config
                    )
                    meta = save_book(
                        output,
                        view,
                        result,
                        targets,
                        previous,
                        start,
                        start,
                        provenance,
                    )
                    nav = (
                        result.free_cash
                        + result.restricted_cash
                        + result.hedge_restricted_cash
                        + result.unsettled_cash
                        + result.receivables
                        - result.payables
                        + (result.signed_shares * np.nan_to_num(result.mark_price)).sum(
                            1
                        )
                        + result.hedge_signed_shares
                        * np.nan_to_num(result.hedge_mark_price)
                        - result.loan_liability
                        - result.custody_liability
                    )
                    err = float(np.max(np.abs(nav - result.nav)))
                    assert (
                        err < 1e-6
                        and np.max(np.abs(result.reconciliation_error)) < 1e-6
                    )
                    write_json_atomic(
                        output / "loan_cash_payments.json",
                        [asdict(p) for p in result.loan_cash_payments],
                    )
                    write_json_atomic(
                        output / "performance.json",
                        performance(
                            result.daily_net_return,
                            view.inputs.cdi_returns[start:stop],
                            bench,
                        ),
                    )
                    old = (
                        prior / "phase3/books" / fold / "C6/neutral" / member
                        if arm == "C6"
                        else foundation / "books" / arm / "raw" / fold / member
                    ) / "book.json"
                    report = dict(
                        key=key,
                        book=binding(output / "book.json"),
                        original=binding(old),
                        performance=binding(output / "performance.json"),
                        loan_cash_payments=binding(output / "loan_cash_payments.json"),
                        saved_nav_error_brl=err,
                        seconds=perf_counter() - tick,
                        economics_unresolved=bool(result.economics_unresolved),
                        net_excess_bps=meta["summary"]["mean"]["net_excess_bps"],
                    )
                    completed.append(report)
                    done.add(key)
                    write_json_atomic(
                        progress_file,
                        dict(
                            status="complete"
                            if len(completed) == plan["planned_new_books"]
                            else "running",
                            plan=plan_binding,
                            completed=completed,
                            planned=plan["planned_new_books"],
                        ),
                    )
                    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
