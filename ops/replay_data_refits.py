"""Frozen neutral allocation of fresh compatible matched-data forecasts."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import pickle
from time import perf_counter

import numpy as np
import torch

from brazil_rv.execution.custody_fees import CustodyAssessment
from brazil_rv.execution.portfolio_policy import CalibratedPolicy, exact_replay
from brazil_rv.execution.spot_costs import MonthlySpotTariff
from brazil_rv.execution.stateful_ledger import LedgerConfig
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.evaluate import _holding_audit
from brazil_rv.v2.foundation_readouts import new_panel
from brazil_rv.v2.objective_readouts import calibration, forecast_readout, rank_view
from brazil_rv.v2.opportunity_research import benchmark_for
from brazil_rv.v2.performance import performance
from brazil_rv.v2.portfolio_readouts import save_book
from brazil_rv.v2.portfolio_training import windows
from brazil_rv.v2.research_rounds import _git_identity

PROJECT = Path(__file__).resolve().parents[1]


def freeze(run):
    source = bound_json(run["stage_c_plan"])
    refits = bound_json(run["stage_c_refit_plan"])
    inputs = bound_json(run["stage_c_refit_economics"])
    qualification = bound_json(run["stage_c_refit_economics_qualification"])
    assert (
        qualification["passed"]
        and qualification["inputs"] == run["stage_c_refit_economics"]
    )
    out = Path(run["stage_c_root"]) / "data_refit_replays"
    out.mkdir(exist_ok=False)
    plan = dict(
        status="frozen_before_fresh_fold_economic_outcomes",
        store=refits["store"],
        fit_root=run["stage_c_refit_root"],
        prior_root=source["prior_root"],
        arms=source["arms"],
        folds=source["folds"],
        seeds=refits["seeds"],
        capitals=[10000000, 1000000, 5000000],
        primary_seed_books_only_at_10m=True,
        policy="original frozen neutral equal-rank calibration and constrained allocator; no new policy training, scaling, deadband or selector",
        attribution="Corrected data plus required matched refits versus the saved corrected-account/source old-coordinate books. Neither isolated pure-data attribution nor new accounting terms are inferred.",
        inputs=run["stage_c_refit_economics"],
        input_qualification=run["stage_c_refit_economics_qualification"],
        economic_account=run["economic_account"],
        terms=run["stage_c_event_candidate_terms"],
        refits=run["stage_c_refit_plan"],
        old_source_results=run["stage_c_account_results"],
        planned_books=len(source["arms"]) * len(source["folds"]) * 6,
        readouts="All seeds and ensembles, three currency-consistent Sharpes, utility, turnover, win/loss, drawdown, gross/net/beta, costs/income, observed holding ages and completed/censored holding spells. Later one-factor scenarios depend on actual exposures; earlier proofs are reused.",
        registration=binding(
            PROJECT / "research/preregistrations/v2_economic_data_scaling.md"
        ),
        driver=binding(Path(__file__)),
    )
    assert inputs["store"] == refits["store"]
    write_json_atomic(
        Path(run["stage_c_refit_root"]) / "frozen_design.json",
        dict(store=refits["store"], refit_plan=run["stage_c_refit_plan"]),
    )
    write_json_atomic(out / "plan.json", plan)
    run["stage_c_data_replay_plan"] = binding(out / "plan.json")
    write_json_atomic(PROJECT / "docs/v2_economic_data_scaling_run.json", run)
    print(
        json.dumps(
            dict(plan=run["stage_c_data_replay_plan"], books=plan["planned_books"])
        ),
        flush=True,
    )


def holding_spells(result, names):
    """Observed continuous signed stock spells; unresolved claims stay separate."""
    shares = result.signed_shares
    spells = []
    for n, name in enumerate(names):
        start = None
        side = 0
        for t in range(len(shares) + 1):
            current = 0 if t == len(shares) else int(np.sign(shares[t, n]))
            if current == side:
                continue
            if side:
                spells.append(
                    dict(
                        isin=name,
                        side=side,
                        first_close=str(result.dates[start]),
                        last_close=str(result.dates[t - 1]),
                        observed_close_sessions=t - start,
                        censored_at_end=t == len(shares),
                    )
                )
            start = t if current else None
            side = current
    return dict(
        spells=spells,
        definition="Continuous nonzero same-sign stock inventory at observed session closes; partial resizing does not restart. A sourced identity conversion ends one ISIN spell and begins another. This is not FIFO fill-level or cross-corporate beneficial-claim duration; locked claims are reported separately.",
    )


def execute(run):
    torch.set_num_threads(1)
    code = _git_identity()
    ref = run["stage_c_data_replay_plan"]
    plan = bound_json(ref)
    assert plan["driver"]["sha256"] == sha256_file(Path(__file__))
    inputs = bound_json(plan["inputs"])
    assert sha256_file(Path(inputs["cache"]["path"])) == inputs["cache"]["sha256"]
    with Path(inputs["cache"]["path"]).open("rb") as f:
        data = pickle.load(f)
    out = Path(ref["path"]).parent
    prior = Path(plan["prior_root"])
    account = bound_json(plan["economic_account"])
    values = account["primary_config"].copy()
    values["monthly_spot_tariffs"] = tuple(
        MonthlySpotTariff(**x) for x in values["monthly_spot_tariffs"]
    )
    values["custody_assessments"] = tuple(
        CustodyAssessment(**x) for x in values["custody_assessments"]
    )
    benchmark_root = Path(
        json.loads((PROJECT / "docs/v2_opportunity_run.json").read_text())["root"]
    )
    cash_scope = bound_json(inputs["cash_scope"])
    progress = out / "replays.json"
    completed = (
        json.loads(progress.read_text())["completed"] if progress.exists() else []
    )
    done = {r["key"] for r in completed}
    pending = []
    for fold in plan["folds"]:
        rows = windows(prior, data, fold)["evaluation"]
        start, stop = int(rows[0]), int(rows[-1]) + 1
        days = data.inputs.dates[start:stop]
        assert not set(map(str, days)).intersection(
            cash_scope["uncovered_unused_prelude"]
        )
        bench = benchmark_for(benchmark_root, days, data.inputs.dates[start - 1])
        mapping_path = prior / "phase3/mappings" / f"{fold}.json"
        mappings = json.loads(mapping_path.read_text())
        for arm in plan["arms"]:
            keys = {
                f"data_refit/{capital}/{arm}/{fold}/{member}"
                for capital in plan["capitals"]
                for member in (
                    [*map(str, plan["seeds"]), "ensemble"]
                    if capital == 10000000
                    else ["ensemble"]
                )
            }
            if keys <= done:
                continue
            fits = [
                Path(plan["fit_root"]) / "fits" / arm / f"{fold}_seed_{seed}"
                for seed in plan["seeds"]
            ]
            if not all(
                (fit / "run_manifest.json").exists()
                and bound_json(binding(fit / "run_manifest.json"))["status"]
                == "completed"
                for fit in fits
            ):
                pending.append(dict(arm=arm, fold=fold))
                continue
            panels, valid, sources = new_panel(
                Path(plan["fit_root"]), data, arm, fold, rows, "raw"
            )
            mapping = calibration(mappings["arms"]["C6" if arm == "C6" else "TE_all"])
            for capital in plan["capitals"]:
                for member in (
                    [*map(str, plan["seeds"]), "ensemble"]
                    if capital == 10000000
                    else ["ensemble"]
                ):
                    key = f"data_refit/{capital}/{arm}/{fold}/{member}"
                    if key in done:
                        bound_json(
                            next(r["book"] for r in completed if r["key"] == key)
                        )
                        continue
                    tick = perf_counter()
                    target = out / "books" / key
                    assert not target.exists(), (
                        "Preserve partial output and resume its exact boundary explicitly"
                    )
                    view = rank_view(data, rows, panels[member], valid)
                    config = LedgerConfig(
                        **(values | dict(initial_capital_brl=capital))
                    )
                    provenance = dict(
                        implementation=code,
                        scenario="base",
                        policy="equal_rank",
                        phase="data_refit",
                        config=asdict(config),
                        plan=ref,
                        policy_inputs=plan["inputs"],
                        mapping=binding(mapping_path),
                        forecast_sources=sources,
                        member=member,
                        heldout_accessed=False,
                    )
                    result, targets, previous = exact_replay(
                        view, CalibratedPolicy(mapping), start, stop, config=config
                    )
                    book = save_book(
                        target,
                        view,
                        result,
                        targets,
                        previous,
                        start,
                        start,
                        provenance,
                    )
                    write_json_atomic(
                        target / "performance.json",
                        performance(
                            result.daily_net_return,
                            view.inputs.cdi_returns[start:stop],
                            bench,
                        ),
                    )
                    write_json_atomic(
                        target / "holding_ages.json",
                        _holding_audit(result, view.inputs.security_ids),
                    )
                    write_json_atomic(
                        target / "holding_spells.json",
                        holding_spells(result, view.inputs.security_ids),
                    )
                    write_json_atomic(
                        target / "loan_cash_payments.json",
                        [asdict(p) for p in result.loan_cash_payments],
                    )
                    write_json_atomic(
                        target / "forecast_readout.json",
                        forecast_readout(data, rows, panels[member], valid, mapping),
                    )
                    completed.append(
                        dict(
                            key=key,
                            book=binding(target / "book.json"),
                            performance=binding(target / "performance.json"),
                            holding_ages=binding(target / "holding_ages.json"),
                            holding_spells=binding(target / "holding_spells.json"),
                            loan_cash_payments=binding(
                                target / "loan_cash_payments.json"
                            ),
                            forecast=binding(target / "forecast_readout.json"),
                            seconds=perf_counter() - tick,
                            economics_unresolved=bool(result.economics_unresolved),
                            net_excess_bps=book["summary"]["mean"]["net_excess_bps"],
                        )
                    )
                    done.add(key)
                    write_json_atomic(
                        progress,
                        dict(
                            status="complete"
                            if len(completed) == plan["planned_books"]
                            else "running",
                            plan=ref,
                            completed=completed,
                            planned=plan["planned_books"],
                        ),
                    )
                    print(json.dumps(completed[-1]), flush=True)
    write_json_atomic(
        progress,
        dict(
            status="complete"
            if len(completed) == plan["planned_books"]
            else "waiting_for_fits",
            plan=ref,
            completed=completed,
            planned=plan["planned_books"],
            pending_groups=pending,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    freeze(run) if args.freeze else execute(run)
