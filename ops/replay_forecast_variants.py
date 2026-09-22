"""Post-hoc selection views and rank-average ensembles on saved forecasts; no fit.

Every book here is a frozen-policy replay of forecasts that already exist: the
trainer's own selected epoch, another saved epoch of the same executed trajectory
(scored once, on CPU by default, through the registered scoring path on a
hash-bound derived checkpoint), or the equal-weight rank average of several such
forecasts across epochs, seeds, stopping views, widths or model families. Nothing
is retrained and no evaluation label is read before the books.

    uv run --project research --no-sync python ops/replay_forecast_variants.py \
        --freeze --policy neutral
    uv run --project research --no-sync python ops/replay_forecast_variants.py \
        --policy neutral
    uv run --project research --no-sync python ops/replay_forecast_variants.py \
        --policy neutral --summarize

A variant is ``NAME=arm@rule+arm@rule`` or ``arm@rule``; rules are ``raw``
(the trainer's selection), ``centreK`` (centre epoch of the best trailing-K
selection window), ``topK`` (rank average of the K best selection epochs) and
``aroundK`` (rank average of the K epochs around the raw selection). Default
variants cover the matched-stopping attention arms and, once the common-model
fits exist, the cross-family ensembles. Paired summaries compare each variant
with its reference arm's existing books fold by fold (block 40 primary).

Scoring an alternative epoch on CPU is safe while a GPU worker owns the queue;
pass ``--device cuda`` only when no fit is running.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import pickle
from time import perf_counter

import numpy as np
import torch

from brazil_rv.execution.allocation import AllocationConfig
from brazil_rv.execution.custody_fees import CustodyAssessment
from brazil_rv.execution.portfolio_policy import CalibratedPolicy, exact_replay
from brazil_rv.execution.spot_costs import MonthlySpotTariff
from brazil_rv.execution.stateful_ledger import LedgerConfig
from brazil_rv.v2 import forecast_variants
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.evaluate import _holding_audit
from brazil_rv.v2.forecast_variants import (
    completed_fit,
    compose,
    load_panel,
    member_epochs,
    parse_variant,
    scores_for_epoch,
)
from brazil_rv.v2.foundation_readouts import paired_interval
from brazil_rv.v2.objective_readouts import calibration, forecast_readout, rank_view
from brazil_rv.v2.opportunity_research import benchmark_for
from brazil_rv.v2.performance import maximum_drawdown, performance, sharpe
from brazil_rv.v2.portfolio_readouts import save_book
from brazil_rv.v2.portfolio_training import windows
from brazil_rv.v2.research_rounds import _git_identity
import replay_data_refits

PROJECT = Path(__file__).resolve().parents[1]
POINTER = PROJECT / "docs/v2_economic_data_scaling_run.json"
PLAN_KEY = "scaling_posthoc_variant_plans"
SUMMARY_KEY = "scaling_posthoc_variant_summaries"
DEFAULT_RULES = ("centre3", "around3", "top3")
PRIMARY_CAPITAL = 10_000_000


def fit_roots(run):
    """Every arm with completed-fit directories ``<root>/fits/<arm>/<fold>_seed_<seed>``."""
    stopping = bound_json(run["scaling_matched_stopping_plan"])
    roots = {arm: stopping["root"] for arm in stopping["arms"]}
    for key, field in (
        ("scaling_common_model_plan", "cells"),
        ("scaling_smoothed_views_plan", "arms"),
        ("scaling_f_schedule_plan", "arms"),
    ):
        if key in run:
            plan = bound_json(run[key])
            roots.update({arm: plan["root"] for arm in plan[field]})
    return roots, stopping


def default_variants(stopping, common_arms):
    variants = [f"{arm}@{rule}" for arm in stopping["arms"] for rule in DEFAULT_RULES]
    for cell in stopping["cells"]:
        variants.append(f"ENS_{cell}_p5p20={cell}_p5@raw+{cell}_p20@raw")
    cells = list(stopping["cells"])
    if len(cells) > 1:
        variants.append(
            "ENS_widths_p20=" + "+".join(f"{cell}_p20@raw" for cell in cells)
        )
    if common_arms:
        lead = f"{cells[0]}_p20"
        variants.append(f"ENS_C6_attention=C6@raw+{lead}@raw")
        variants.append(
            "ENS_families=" + "+".join(f"{a}@raw" for a in (*common_arms, lead))
        )
    return variants


def reference_plan(run, policy, arm):
    """The frozen evaluation plan whose existing books serve as this arm's reference."""
    candidates = []
    if policy == "neutral":
        candidates.append(run["scaling_matched_stopping_evaluation_plan"])
        if "scaling_common_evaluation_plans" in run:
            candidates.append(run["scaling_common_evaluation_plans"]["neutral"])
    elif "scaling_common_evaluation_plans" in run:
        candidates.append(run["scaling_common_evaluation_plans"]["flexible"])
    for reference in candidates:
        if arm in bound_json(reference)["arms"]:
            return reference
    return None


def freeze(run, args):
    policy = args.policy
    source_ref = (
        run["scaling_matched_stopping_evaluation_plan"]
        if policy == "neutral"
        else run["scaling_common_evaluation_plans"]["flexible"]
    )
    source = bound_json(source_ref)
    roots, stopping = fit_roots(run)
    common_arms = (
        list(bound_json(run["scaling_common_model_plan"])["cells"])
        if "scaling_common_model_plan" in run
        else []
    )
    variants = {}
    for text in args.variant or default_variants(stopping, common_arms):
        name, members = parse_variant(text)
        unknown = sorted({arm for arm, _ in members if arm not in roots})
        if unknown:
            raise SystemExit(f"variant {name} names arms without fits: {unknown}")
        variants[name] = {
            "members": [{"arm": arm, "rule": rule} for arm, rule in members],
            "mapping_key": "C6" if all(arm == "C6" for arm, _ in members) else "TE_all",
            "reference": {
                "arm": members[0][0],
                "plan": reference_plan(run, policy, members[0][0]),
            },
        }
    used = sorted({m["arm"] for v in variants.values() for m in v["members"]})
    root = Path(stopping["root"]) / f"posthoc_variants_{policy}"
    root.mkdir(exist_ok=False)
    capitals = list(source["capitals"]) if args.all_capitals else [PRIMARY_CAPITAL]
    books_per_fold = sum(
        len(source["seeds"]) + 1 if c == PRIMARY_CAPITAL else 1 for c in capitals
    )
    plan = dict(
        source,
        status="frozen_before_variant_outcomes",
        exposure_policy=policy,
        source_plan=source_ref,
        root=str(root),
        arms=list(variants),
        variants=variants,
        fit_roots={arm: roots[arm] for arm in used},
        scores_cache=str(root / "scores"),
        capitals=capitals,
        primary_seed_books_only_at_10m=True,
        device=args.device,
        planned_books=len(variants) * len(source["folds"]) * books_per_fold,
        driver=binding(Path(__file__)),
        freezer=binding(Path(__file__)),
        library=binding(Path(forecast_variants.__file__)),
        runtime=_git_identity(),
        registration=binding(
            PROJECT / "research/preregistrations/v2_posthoc_selection_and_ensembles.md"
        ),
        scope=(
            "Saved forecasts only: the trainer's selected epochs, other saved epochs "
            "of the same executed trajectories scored once through the registered "
            "path on hash-bound derived checkpoints, and equal-weight rank averages "
            "of those panels. Same store, dates, account, calibration mapping and "
            "allocation as the source evaluation plan. No retraining, no reserved "
            "fold, no held-out or 2025/2026 consumer, no learned weights."
        ),
        attribution=(
            "Within one frozen evaluation plan a variant differs from its reference "
            "arm only through which saved forecasts enter the policy. Paired fold "
            "deltas against the reference arm's existing books are the primary "
            "readout (block 40, 90/95 percent); IC is diagnostic."
        ),
        selection=(
            "Report every variant. Retention of a selection rule or ensemble needs the "
            "pooled 95 percent lower bound above zero on the primary policy, a "
            "majority of folds and seeds positive and no drawdown deterioration; "
            "otherwise the trainer's raw selection and single-family ensembles stand."
        ),
    )
    write_json_atomic(root / "plan.json", plan)
    run.setdefault(PLAN_KEY, {})[policy] = binding(root / "plan.json")
    write_json_atomic(POINTER, run)
    print(
        json.dumps(
            {
                "plan": run[PLAN_KEY][policy],
                "variants": list(variants),
                "planned_books": plan["planned_books"],
            }
        ),
        flush=True,
    )


def _ledger_values(plan):
    account = bound_json(plan["economic_account"])
    values = account["primary_config"].copy()
    allocation = AllocationConfig(**plan.get("allocation", {}))
    if "allocation" in plan:
        values["planned_absolute_net_cap"] = allocation.net_cap
        values["planned_absolute_beta_cap"] = allocation.beta_cap
    values["monthly_spot_tariffs"] = tuple(
        MonthlySpotTariff(**x) for x in values["monthly_spot_tariffs"]
    )
    values["custody_assessments"] = tuple(
        CustodyAssessment(**x) for x in values["custody_assessments"]
    )
    return values, allocation


def _members(plan, capital):
    return (
        [*map(str, plan["seeds"]), "ensemble"]
        if capital == PRIMARY_CAPITAL
        else ["ensemble"]
    )


def variant_panels(plan, name, fold, *, data, rows, store, schema, device):
    """Per-seed and ensemble rank panels of one variant, or None while fits are pending."""
    spec = plan["variants"][name]
    dates = np.asarray(data.inputs.dates, dtype="datetime64[D]")[rows]
    isins = data.inputs.security_ids
    active = data.inputs.active[rows]
    seed_panels, sources, valid = {}, {}, None
    for seed in plan["seeds"]:
        member_panels = []
        for member in spec["members"]:
            fit = (
                Path(plan["fit_roots"][member["arm"]])
                / "fits"
                / member["arm"]
                / f"{fold}_seed_{seed}"
            )
            if not (fit / "run_manifest.json").exists():
                return None, {"pending": str(fit)}
            manifest, history = completed_fit(fit)
            if (
                manifest["seed"] != seed
                or manifest["fold"] != fold
                or manifest["contract"]["store_manifest_sha256"]
                != plan["store"]["manifest_sha256"]
            ):
                raise ValueError(f"fit provenance differs from the plan: {fit}")
            view = member_epochs(history, member["rule"])
            epoch_panels = []
            for epoch in view["epochs"]:
                scores, record = scores_for_epoch(
                    store,
                    fit,
                    manifest,
                    epoch,
                    Path(plan["scores_cache"]) / member["arm"] / f"{fold}_seed_{seed}",
                    view=view,
                    device=device,
                )
                panel, this_valid = load_panel(
                    scores,
                    dates=dates,
                    isins=isins,
                    feature_schema_sha256=schema,
                    active=active,
                )
                if valid is None:
                    valid = this_valid
                elif not np.array_equal(valid, this_valid):
                    raise ValueError("variant members disagree on the eligible names")
                epoch_panels.append(panel)
                sources[
                    f"{member['arm']}@{member['rule']}/{seed}/epoch_{epoch:03d}"
                ] = {
                    **record,
                    "fit": binding(fit / "run_manifest.json"),
                    "scores": binding(scores / "score_manifest.json"),
                    "view": {k: v for k, v in view.items() if k != "scores"},
                }
            member_panels.append(compose(epoch_panels))
        seed_panels[str(seed)] = compose(member_panels)
    panels = dict(seed_panels, ensemble=compose(list(seed_panels.values())))
    return (panels, valid), sources


def execute(run, policy):
    torch.set_num_threads(1)
    code = _git_identity()
    ref = run[PLAN_KEY][policy]
    plan = bound_json(ref)
    assert plan["driver"]["sha256"] == sha256_file(Path(__file__))
    assert plan["library"]["sha256"] == sha256_file(Path(forecast_variants.__file__))
    inputs = bound_json(plan["inputs"])
    assert sha256_file(Path(inputs["cache"]["path"])) == inputs["cache"]["sha256"]
    with Path(inputs["cache"]["path"]).open("rb") as handle:
        data = pickle.load(handle)
    root, store = Path(plan["root"]), Path(plan["store"]["root"])
    assert sha256_file(store / "manifest.json") == plan["store"]["manifest_sha256"]
    schema = json.loads((store / "manifest.json").read_text(encoding="utf-8"))[
        "metadata"
    ]["feature_schema"]["sha256"]
    device = torch.device(plan["device"])
    prior = Path(plan["prior_root"])
    values, allocation = _ledger_values(plan)
    benchmark_root = Path(
        json.loads((PROJECT / "docs/v2_opportunity_run.json").read_text())["root"]
    )
    cash_scope = bound_json(inputs["cash_scope"])
    progress = root / "replays.json"
    completed = (
        json.loads(progress.read_text())["completed"] if progress.exists() else []
    )
    done = {r["key"] for r in completed}
    pending = []

    def checkpoint(status):
        write_json_atomic(
            progress,
            {
                "status": status,
                "plan": ref,
                "completed": completed,
                "planned": plan["planned_books"],
                "pending_groups": pending,
            },
        )

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
        for name, spec in plan["variants"].items():
            keys = {
                f"variant/{capital}/{name}/{fold}/{member}"
                for capital in plan["capitals"]
                for member in _members(plan, capital)
            }
            if keys <= done:
                continue
            outcome, sources = variant_panels(
                plan,
                name,
                fold,
                data=data,
                rows=rows,
                store=store,
                schema=schema,
                device=device,
            )
            if outcome is None:
                pending.append({"variant": name, "fold": fold, **sources})
                continue
            panels, valid = outcome
            mapping = calibration(mappings["arms"][spec["mapping_key"]])
            for capital in plan["capitals"]:
                for member in _members(plan, capital):
                    key = f"variant/{capital}/{name}/{fold}/{member}"
                    if key in done:
                        continue
                    tick = perf_counter()
                    target = root / "books" / key
                    assert not target.exists(), (
                        "Preserve partial output and resume its exact boundary explicitly"
                    )
                    view = rank_view(data, rows, panels[member], valid)
                    config = LedgerConfig(**(values | {"initial_capital_brl": capital}))
                    provenance = {
                        "implementation": code,
                        "scenario": "base",
                        "policy": "equal_rank",
                        "phase": "posthoc_variant",
                        "config": asdict(config),
                        "plan": ref,
                        "policy_inputs": plan["inputs"],
                        "mapping": binding(mapping_path),
                        "mapping_key": spec["mapping_key"],
                        "variant": {"name": name, **spec},
                        "forecast_sources": sources,
                        "allocation": asdict(allocation),
                        "member": member,
                        "heldout_accessed": False,
                    }
                    result, targets, previous = exact_replay(
                        view,
                        CalibratedPolicy(mapping),
                        start,
                        stop,
                        config=config,
                        allocation=allocation,
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
                        replay_data_refits.holding_spells(
                            result, view.inputs.security_ids
                        ),
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
                        {
                            "key": key,
                            "book": binding(target / "book.json"),
                            "performance": binding(target / "performance.json"),
                            "holding_ages": binding(target / "holding_ages.json"),
                            "holding_spells": binding(target / "holding_spells.json"),
                            "loan_cash_payments": binding(
                                target / "loan_cash_payments.json"
                            ),
                            "forecast": binding(target / "forecast_readout.json"),
                            "seconds": perf_counter() - tick,
                            "economics_unresolved": bool(result.economics_unresolved),
                            "net_excess_bps": book["summary"]["mean"]["net_excess_bps"],
                        }
                    )
                    done.add(key)
                    checkpoint(
                        "complete"
                        if len(completed) == plan["planned_books"]
                        else "running"
                    )
                    print(json.dumps(completed[-1]), flush=True)
    checkpoint(
        "complete" if len(completed) == plan["planned_books"] else "waiting_for_fits"
    )


def _daily(book_record):
    book = bound_json(book_record["book"])
    daily = book["daily"]
    return np.asarray(daily["net_excess_bps"], float), np.asarray(
        daily["cdi_bps"], float
    )


def reference_daily(plan, books, spec, fold, member):
    """Reference book series: the source plan's existing books, else this plan's raw variant."""
    arm, reference = spec["reference"]["arm"], spec["reference"]["plan"]
    if reference is not None:
        source_progress = Path(reference["path"]).parent / "replays.json"
        if source_progress.exists():
            key = f"data_refit/{PRIMARY_CAPITAL}/{arm}/{fold}/{member}"
            for record in json.loads(source_progress.read_text())["completed"]:
                if record["key"] == key:
                    return _daily(record), {"plan": reference, "key": key}
    key = f"variant/{PRIMARY_CAPITAL}/{arm}@raw/{fold}/{member}"
    if key in books:
        return _daily(books[key]), {"plan": None, "key": key}
    return None, None


def summarize(run, policy):
    ref = run[PLAN_KEY][policy]
    plan = bound_json(ref)
    root = Path(plan["root"])
    progress = json.loads((root / "replays.json").read_text())
    books = {r["key"]: r for r in progress["completed"]}
    rows = []
    for name, spec in plan["variants"].items():
        for member in _members(plan, PRIMARY_CAPITAL):
            own, deltas, references, sources = [], [], [], []
            for fold in plan["folds"]:
                key = f"variant/{PRIMARY_CAPITAL}/{name}/{fold}/{member}"
                if key not in books:
                    continue
                excess, cdi = _daily(books[key])
                own.append((excess, cdi))
                reference, source = reference_daily(plan, books, spec, fold, member)
                if reference is not None and len(reference[0]) == len(excess):
                    deltas.append(excess - reference[0])
                    references.append(reference)
                    sources.append(source)
            if not own:
                continue
            excess = np.concatenate([e for e, _ in own])
            absolute = (excess + np.concatenate([c for _, c in own])) / 1e4
            row = {
                "variant": name,
                "member": member,
                "members": spec["members"],
                "folds": len(own),
                "mean_excess_bps": float(excess.mean()),
                "sharpe_zero_rate": sharpe(absolute),
                "maximum_drawdown": maximum_drawdown(absolute),
            }
            if deltas:
                interval = paired_interval(tuple(deltas), block_length=40)
                reference_absolute = np.concatenate(
                    [(e + c) / 1e4 for e, c in references]
                )
                row.update(
                    {
                        "reference_arm": spec["reference"]["arm"],
                        "paired_folds": len(deltas),
                        "positive_folds": int(sum(d.mean() > 0 for d in deltas)),
                        "delta_bps": interval["estimate"],
                        "delta_lower_90": interval["lower_90"],
                        "delta_upper_90": interval["upper_90"],
                        "delta_lower_95": interval["lower_95"],
                        "delta_upper_95": interval["upper_95"],
                        "reference_sharpe_zero_rate": sharpe(reference_absolute),
                        "sharpe_zero_rate_delta": (
                            None
                            if row["sharpe_zero_rate"] is None
                            or sharpe(reference_absolute) is None
                            else row["sharpe_zero_rate"] - sharpe(reference_absolute)
                        ),
                        "reference_sources": sources,
                    }
                )
            rows.append(row)
    report = {
        "plan": ref,
        "status": progress["status"],
        "books": len(books),
        "block_length": 40,
        "rows": rows,
        "reading": (
            "Deltas are paired fold-by-fold daily net-excess differences against the "
            "reference arm's existing books at R$10m, pooled with circular block-40 "
            "draws; intervals are nominal development intervals, not held-out tests."
        ),
    }
    write_json_atomic(root / "summary.json", report)
    run.setdefault(SUMMARY_KEY, {})[policy] = binding(root / "summary.json")
    write_json_atomic(POINTER, run)
    print(
        f"{'variant':>34} {'member':>8} {'folds':>5} {'bps/day':>8} {'Sharpe0':>8} {'delta':>7} {'95% interval':>20} {'+folds':>6}"
    )
    for row in rows:
        delta = (
            f"{row['delta_bps']:+7.3f} [{row['delta_lower_95']:+7.3f}, {row['delta_upper_95']:+7.3f}] "
            f"{row['positive_folds']:>3}/{row['paired_folds']}"
            if "delta_bps" in row
            else "unpaired"
        )
        sharpe0 = (
            "n/a"
            if row["sharpe_zero_rate"] is None
            else f"{row['sharpe_zero_rate']:8.3f}"
        )
        print(
            f"{row['variant']:>34} {row['member']:>8} {row['folds']:>5} "
            f"{row['mean_excess_bps']:8.3f} {sharpe0:>8} {delta}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("neutral", "flexible"), required=True)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--variant", action="append", help="NAME=arm@rule+arm@rule")
    parser.add_argument("--all-capitals", action="store_true")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    run = json.loads(POINTER.read_text())
    if args.freeze:
        freeze(run, args)
    elif args.summarize:
        summarize(run, args.policy)
    else:
        execute(run, args.policy)
