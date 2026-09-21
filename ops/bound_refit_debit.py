"""Existing debit spreads on actually funded denied-renewal refit portfolios."""

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
from brazil_rv.v2.foundation_readouts import new_panel
from brazil_rv.v2.objective_readouts import calibration, rank_view
from brazil_rv.v2.opportunity_research import benchmark_for
from brazil_rv.v2.performance import performance
from brazil_rv.v2.portfolio_readouts import save_book
from brazil_rv.v2.portfolio_training import windows
from brazil_rv.v2.research_rounds import _git_identity
import qualify_refit_books

PROJECT = Path(__file__).resolve().parents[1]


def freeze(run):
    parent = bound_json(run["stage_c_refit_sensitivity_plan"])
    root = Path(run["root"]) / "matched_refit_debit_bounds"
    root.mkdir(exist_ok=False)
    phases = [
        p for p in parent["phases"] if "annual_debit_spread" in p.get("config", {})
    ]
    plan = dict(
        primary=parent["primary"],
        denied_sensitivities=run["stage_c_refit_sensitivity_plan"],
        phases=phases,
        trigger="Every completed and qualified denied_renewal ensemble with actual prior-close negative free cash. Preserve explicit unexposed skips and unresolved status. Selection uses exposure, never PnL.",
        contrast="Change annual debit spread from the denied-renewal baseline's zero to the already frozen50/100annual basis points. Keep denied renewal, forecasts, allocation, capital, source/account terms and all other settings fixed. This is a conditional one-factor financing contrast within the stated denial scenario, not a joint worst case or a change to the approved-renewal primary.",
        limits="Primary debit variants remain independently governed by their own exposure predicates. No new lender quote, borrowing guarantee, combined interior bound or executable resolution of overdue loans is inferred.",
        driver=binding(Path(__file__)),
    )
    write_json_atomic(root / "plan.json", plan)
    run["stage_c_refit_debit_plan"] = binding(root / "plan.json")
    write_json_atomic(PROJECT / "docs/v2_economic_data_scaling_run.json", run)
    print(
        json.dumps(dict(plan=run["stage_c_refit_debit_plan"], phases=len(phases))),
        flush=True,
    )


def execute(run):
    torch.set_num_threads(1)
    code = _git_identity()
    reference = run["stage_c_refit_debit_plan"]
    plan = bound_json(reference)
    assert sha256_file(Path(__file__)) == plan["driver"]["sha256"]
    root = Path(reference["path"]).parent
    parent = Path(plan["denied_sensitivities"]["path"]).parent
    upstream = bound_json(binding(parent / "replays.json"))
    primary = bound_json(plan["primary"])
    inputs = bound_json(primary["inputs"])
    assert sha256_file(Path(inputs["cache"]["path"])) == inputs["cache"]["sha256"]
    with Path(inputs["cache"]["path"]).open("rb") as f:
        data = pickle.load(f)
    progress = root / "replays.json"
    saved = (
        json.loads(progress.read_text())
        if progress.exists()
        else dict(completed=[], skipped=[])
    )
    completed, skipped = saved["completed"], saved["skipped"]
    done = {r["key"] for r in [*completed, *skipped]}
    prior = Path(primary["prior_root"])
    bench_root = Path(
        json.loads((PROJECT / "docs/v2_opportunity_run.json").read_text())["root"]
    )
    panels = {}

    def save_progress():
        write_json_atomic(
            progress,
            dict(
                status="complete"
                if upstream["status"] == "complete"
                else "waiting_for_denial_baselines",
                plan=reference,
                completed=completed,
                skipped=skipped,
            ),
        )

    for base in upstream["completed"]:
        if not base["key"].startswith("denied_renewal/"):
            continue
        proof = parent / "qualification" / (base["key"].replace("/", "_") + ".json")
        if not proof.exists():
            continue
        quality = bound_json(binding(proof))
        assert quality["passed"] and quality["book"] == base["book"]
        book = bound_json(base["book"])
        _, capital, arm, fold, _ = base["key"].split("/")
        phases = [p for p in plan["phases"] if p["capital"] == int(capital)]
        with np.load(Path(base["book"]["path"]).parent / "account.npz") as z:
            original = {k: z[k] for k in ("free_cash", "nav", "targets")}
        for phase in phases:
            key = "denied_" + phase["name"] + "/" + base["key"].split("/", 1)[1]
            if key in done:
                continue
            if quality["prior_debit_sessions"] == 0:
                skipped.append(
                    dict(
                        key=key,
                        baseline=base,
                        reason="No actual prior-close debit; not a financing bound",
                    )
                )
                done.add(key)
                save_progress()
                continue
            tick = perf_counter()
            rows = windows(prior, data, fold)["evaluation"]
            start, stop = int(rows[0]), int(rows[-1]) + 1
            if (arm, fold) not in panels:
                panels[arm, fold] = new_panel(
                    Path(primary["fit_root"]), data, arm, fold, rows, "raw"
                )
            forecasts, valid, sources = panels[arm, fold]
            assert sources == book["provenance"]["forecast_sources"]
            view = rank_view(data, rows, forecasts["ensemble"], valid)
            mapping = calibration(
                bound_json(book["provenance"]["mapping"])["arms"][
                    "C6" if arm == "C6" else "TE_all"
                ]
            )
            cfg = book["provenance"]["config"].copy()
            assert not cfg["approve_loan_renewals"] and cfg["annual_debit_spread"] == 0
            assert not cfg["loan_recalls"]
            cfg["monthly_spot_tariffs"] = tuple(
                MonthlySpotTariff(**x) for x in cfg["monthly_spot_tariffs"]
            )
            cfg["custody_assessments"] = tuple(
                CustodyAssessment(**x) for x in cfg["custody_assessments"]
            )
            cfg["annual_debit_spread"] = phase["config"]["annual_debit_spread"]
            config = LedgerConfig(**cfg)
            path = root / "books" / key
            assert not path.exists(), (
                "Preserve any partial output and resume its saved boundary explicitly"
            )
            provenance = dict(
                book["provenance"],
                implementation=code,
                config=asdict(config),
                phase=key.split("/")[0],
                conditional_debit_plan=reference,
                baseline=base["book"],
            )
            result, targets, previous = exact_replay(
                view, CalibratedPolicy(mapping), start, stop, config=config
            )
            save_book(path, view, result, targets, previous, start, start, provenance)
            bench = benchmark_for(
                bench_root, data.inputs.dates[start:stop], data.inputs.dates[start - 1]
            )
            write_json_atomic(
                path / "performance.json",
                performance(
                    result.daily_net_return, data.inputs.cdi_returns[start:stop], bench
                ),
            )
            write_json_atomic(
                path / "loan_cash_payments.json",
                [asdict(p) for p in result.loan_cash_payments],
            )
            first = int(np.flatnonzero(np.r_[0, original["free_cash"][:-1]] < 0)[0])
            with np.load(path / "account.npz") as z:
                prefix = dict(
                    first_debit_index=first,
                    first_debit_date=book["state_dates"][first],
                    nav_prefix_exact=bool(
                        np.array_equal(z["nav"][:first], original["nav"][:first])
                    ),
                    target_prefix_exact=bool(
                        np.array_equal(
                            z["targets"][:first], original["targets"][:first]
                        )
                    ),
                    first_debit_target_max_distance=float(
                        np.max(np.abs(z["targets"][first] - original["targets"][first]))
                    ),
                )
            write_json_atomic(path / "prefix.json", prefix)
            assert prefix["nav_prefix_exact"] and prefix["target_prefix_exact"]
            completed.append(
                dict(
                    key=key,
                    book=binding(path / "book.json"),
                    baseline=base,
                    performance=binding(path / "performance.json"),
                    loan_cash_payments=binding(path / "loan_cash_payments.json"),
                    prefix=binding(path / "prefix.json"),
                    economics_unresolved=bool(result.economics_unresolved),
                    seconds=perf_counter() - tick,
                )
            )
            done.add(key)
            save_progress()
            print(
                json.dumps(dict(completed=key, seconds=completed[-1]["seconds"])),
                flush=True,
            )
    save_progress()
    # Reuse the unchanged qualified checker in a private pointer context.
    context = root / "qualifier_context"
    (context / "docs").mkdir(parents=True, exist_ok=True)
    local = dict(run, stage_c_refit_sensitivity_plan=reference)
    write_json_atomic(context / "docs/v2_economic_data_scaling_run.json", local)
    qualify_refit_books.PROJECT = context
    qualify_refit_books.main(sensitivities=True)
    run["stage_c_refit_debit_qualification"] = binding(
        root / "qualification/manifest.json"
    )
    write_json_atomic(PROJECT / "docs/v2_economic_data_scaling_run.json", run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    freeze(run) if args.freeze else execute(run)
