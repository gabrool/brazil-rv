"""Frozen forecast/portfolio-input decomposition; never refit or rescore weights."""

from copy import copy
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
from brazil_rv.v2.objective_readouts import calibration, forecast_readout, rank_view
from brazil_rv.v2.portfolio_readouts import save_book
from brazil_rv.v2.portfolio_training import load_data, windows

PROJECT = Path(__file__).resolve().parents[1]


def nav_check(path):
    with np.load(path / "account.npz") as a:
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
        error = float(np.max(np.abs(nav - a["nav"])))
        assert error < 1e-7, error
        return dict(
            independent_saved_nav_max_error_brl=error,
            maximum_overdue_principal=float(np.max(a["loan_overdue_principal"])),
            days=len(nav),
            saved_cells=sum(a[key].size for key in a.files),
        )


def main():
    tick = perf_counter()
    torch.set_num_threads(1)
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    plan = bound_json(run["scaling_investigation"])
    out = Path(run["scaling_investigation"]["path"]).parent / "decomposition"
    out.mkdir(exist_ok=True)
    recipe = out / "executed.py"
    if recipe.exists():
        assert sha256_file(recipe) == sha256_file(Path(__file__))
    else:
        recipe.write_bytes(Path(__file__).read_bytes())
        write_json_atomic(
            out / "runtime.json",
            {
                str(path.relative_to(PROJECT)): binding(path)
                for folder in ["v2", "execution"]
                for path in (PROJECT / "research/src/brazil_rv" / folder).glob("*.py")
            },
        )
    source = bound_json(plan["old_plan"])
    prior = Path(source["prior_root"])
    old, old_cache = load_data(prior, "C6")
    inputs = bound_json(plan["new_inputs"])
    cache = Path(inputs["cache"]["path"])
    assert sha256_file(cache) == inputs["cache"]["sha256"]
    with cache.open("rb") as stream:
        new = pickle.load(stream)
    np.testing.assert_array_equal(old.inputs.security_ids, new.inputs.security_ids)
    np.testing.assert_array_equal(old.inputs.dates, new.inputs.dates)
    assert len(new.inputs.security_ids) == 933
    assert str(new.inputs.dates[-1]) <= "2024-12-30"
    values = bound_json(plan["account"])["primary_config"].copy()
    values["initial_capital_brl"] = plan["capital"]
    values["monthly_spot_tariffs"] = tuple(
        MonthlySpotTariff(**x) for x in values["monthly_spot_tariffs"]
    )
    values["custody_assessments"] = tuple(
        CustodyAssessment(**x) for x in values["custody_assessments"]
    )
    config = LedgerConfig(**values)
    old_index = bound_json(bound_json(plan["old_result"])["primary"])
    old_books = {row["key"]: row for row in old_index["completed"]}
    new_results = bound_json(bound_json(plan["new_result"])["books"])
    saved_new = {
        (r["arm"], r["fold"]): r
        for r in new_results
        if r["capital"] == plan["capital"] and r["member"] == "ensemble"
    }
    progress = out / "replays.json"
    completed = bound_json(binding(progress))["completed"] if progress.exists() else []
    done = {row["key"] for row in completed}
    summaries = []
    for fold in plan["screen_folds"]:
        rows = windows(prior, new, fold)["evaluation"]
        start, stop = int(rows[0]), int(rows[-1]) + 1
        mapping_path = prior / "phase3/mappings" / f"{fold}.json"
        mapping = calibration(bound_json(binding(mapping_path))["arms"]["TE_all"])
        for arm in plan["arms"]:
            old_panels, old_valid, old_sources = new_panel(
                Path(source["foundation_root"]), old, arm, fold, rows, "raw"
            )
            new_panels, new_valid, new_sources = new_panel(
                Path(run["stage_c_refit_root"]), new, arm, fold, rows, "raw"
            )
            valid = old_valid & new.inputs.active[rows]
            common = valid & new_valid
            assert np.isfinite(old_panels["ensemble"][valid]).all()
            assert np.isfinite(new_panels["ensemble"][new_valid]).all()
            rank_a, rank_b = (
                old_panels["ensemble"][common],
                new_panels["ensemble"][common],
            )
            support = dict(
                old_signals=int(old_valid.sum()),
                old_signals_on_new_membership=int(valid.sum()),
                new_signals=int(new_valid.sum()),
                unsupported_new_signals=int((new_valid & ~valid).sum()),
                retired_old_signals=int((old_valid & ~valid).sum()),
                common_rank_correlation=float(
                    np.corrcoef(rank_a.ravel(), rank_b.ravel())[0, 1]
                ),
                old_forecast_new_targets=forecast_readout(
                    new, rows, old_panels["ensemble"], valid, mapping
                ),
                new_forecast_new_targets=forecast_readout(
                    new, rows, new_panels["ensemble"], new_valid, mapping
                ),
            )
            scope = out / "inputs" / arm / fold
            scope.mkdir(parents=True, exist_ok=True)
            support_path = scope / "support.json"
            if not support_path.exists():
                write_json_atomic(support_path, support)
            for contrast in plan["counterfactuals"]:
                key = f"{arm}/{fold}/{contrast}"
                if key in done:
                    continue
                old_forecast = contrast.startswith("old_forecast")
                old_risk = contrast.endswith("old_risk")
                base = copy(new)
                if old_risk:
                    for field in ["beta", "diagonal", "factor", "volatility"]:
                        setattr(base, field, getattr(old, field))
                assert base.static is new.static and base.inputs is new.inputs
                ranks = (
                    old_panels["ensemble"] if old_forecast else new_panels["ensemble"]
                )
                mask = valid if old_forecast else new_valid
                view = rank_view(base, rows, ranks, mask)
                for field in ["beta", "diagonal", "volatility"]:
                    assert np.isfinite(getattr(view, field)[rows][mask]).all(), field
                target = out / "books" / key
                assert not target.exists(), "Preserve partial book before any resume"
                started = perf_counter()
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
                    dict(
                        scenario="base",
                        plan=run["scaling_investigation"],
                        diagnostic=contrast,
                        config=asdict(config),
                        policy="equal_rank",
                        policy_inputs=plan["new_inputs"],
                        allocation_risk="old" if old_risk else "new",
                        old_policy_cache_sha256=old_cache,
                        forecast_sources=old_sources if old_forecast else new_sources,
                        mapping=binding(mapping_path),
                        heldout_accessed=False,
                    ),
                )
                quality = nav_check(target)
                write_json_atomic(
                    target / "loan_cash_payments.json",
                    [asdict(x) for x in result.loan_cash_payments],
                )
                record = dict(
                    key=key,
                    book=binding(target / "book.json"),
                    support=binding(support_path),
                    mean=book["summary"]["mean"],
                    quality=quality,
                    economics_unresolved=bool(result.economics_unresolved),
                    seconds=perf_counter() - started,
                )
                completed.append(record)
                done.add(key)
                write_json_atomic(
                    progress,
                    dict(
                        status="running",
                        completed=completed,
                        plan=run["scaling_investigation"],
                    ),
                )
                print(
                    json.dumps(
                        dict(
                            key=key,
                            net=record["mean"]["net_excess_bps"],
                            seconds=record["seconds"],
                        )
                    ),
                    flush=True,
                )
            saved_old = old_books[f"sources/{plan['capital']}/{arm}/{fold}/ensemble"]
            old_book = bound_json(saved_old["book"])
            records = {
                r["key"].split("/")[-1]: r
                for r in completed
                if r["key"].startswith(f"{arm}/{fold}/")
            }
            net = {k: r["mean"]["net_excess_bps"] for k, r in records.items()}
            old_net = old_book["summary"]["mean"]["net_excess_bps"]
            new_net = saved_new[arm, fold]["mean"]["net_excess_bps"]
            summaries.append(
                dict(
                    arm=arm,
                    fold=fold,
                    saved_old=saved_old["book"],
                    saved_new=saved_new[arm, fold]["source"]["book"],
                    old=old_net,
                    new=new_net,
                    **net,
                    input_effect_old_forecasts=net["old_forecast_old_risk"] - old_net,
                    risk_effect_old_forecasts=net["old_forecast_new_risk"]
                    - net["old_forecast_old_risk"],
                    forecast_effect_new_risk=new_net - net["old_forecast_new_risk"],
                    forecast_effect_old_risk=net["new_forecast_old_risk"]
                    - net["old_forecast_old_risk"],
                    risk_forecast_interaction=(new_net - net["old_forecast_new_risk"])
                    - (net["new_forecast_old_risk"] - net["old_forecast_old_risk"]),
                )
            )
    assert len(completed) == plan["new_books"]
    write_json_atomic(
        progress,
        dict(status="complete", completed=completed, plan=run["scaling_investigation"]),
    )
    report = dict(
        status="complete",
        comparisons=summaries,
        replays=binding(progress),
        seconds=perf_counter() - tick,
        limits="Diagnostic adaptive paths, not new model admission. Forecast contrasts include changed eligibility/support; common support correlation is diagnostic only. No old checkpoint consumed new model inputs. Existing controls reused, no source/account campaign repeated.",
    )
    write_json_atomic(out / "report.json", report)
    run["scaling_portfolio_decomposition"] = binding(out / "report.json")
    write_json_atomic(pointer, run)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
