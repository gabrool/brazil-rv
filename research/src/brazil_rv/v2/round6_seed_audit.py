"""Fixed Round-6 seed omissions and descriptive individual-seed forecasts; no fits."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import research_rounds as rr
from .artifacts import sha256_file, write_json_atomic
from .checkpoint_readouts import paired_readouts, retained
from .contract import ALLOWED_SEEDS, DEVELOPMENT_FOLDS, TRADED_PRIMARY_HORIZONS
from .evaluate import _primary_daily_metrics, _primary_population_components
from .research_checkpoint import _context_arguments, _finish_cell
from .round4_seed_audit import isolated_occupancy_failure
from .round6 import SESSION1
from .round6_decisions import (
    IC,
    NET,
    leader_comparisons,
    promotion_trace,
    roster_sensitivity,
    seed_stability,
)
from .round6_readouts import ensemble, evaluation_design, informative_folds


def _values(values):
    return [float(v) if np.isfinite(v) else None for v in values]


def _pool(folds):
    return rr._folded_bootstrap(tuple(np.asarray(v, float) for v in folds))


def panel(root: Path, review: Path, output: Path, omitted: int) -> str:
    full = rr._read_json(review / "result.json")
    design = rr._read_json(root / "frozen_design.json")
    design_hash = sha256_file(root / "frozen_design.json")
    if (
        full["frozen_design_sha256"] != design_hash
        or full["status"] != "provisional_requires_fixed_seed_audit"
    ):
        raise ValueError("seed audit differs from the reviewed experiment")
    roster = full["registered_C6_roster"]
    arms = tuple(full["readouts"])
    seeds = tuple(s for s in ALLOWED_SEEDS if s != omitted)
    destination = output / f"omit_{omitted}"
    destination.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        destination / "source.json",
        {
            "implementation": rr._git_identity(),
            "review_sha256": sha256_file(review / "result.json"),
            "frozen_design_sha256": design_hash,
            "seeds": list(seeds),
            "omitted_seed": omitted,
            "new_training_runs": 0,
        },
    )
    original = evaluation_design(design)
    context = rr._open_ledger_replay(original)
    policy, _ = rr.load_selected_policy(
        Path(original["execution_policy"]["root"]),
        expected_result_sha256=original["execution_policy"]["result_sha256"],
    )
    paths = {a: {} for a in arms}
    levels, solo, single_pairs, rejected = ({a: {} for a in arms} for _ in range(4))
    try:
        for fold in DEVELOPMENT_FOLDS:
            singles = {}
            ix = context.evaluation[fold]
            for arm in arms:
                scores, mask, records = ensemble(
                    design,
                    root,
                    arm,
                    fold,
                    seeds,
                    context.store.dates[ix],
                    context.store.isins,
                    roster=roster,
                )
                dest = destination / arm / fold
                paths[arm][fold] = dest
                rr._persist_scores(
                    dest,
                    {"scores": scores, "score_mask": mask},
                    {
                        **rr._source_tier_labels(context.store.manifest),
                        "arm": arm,
                        "fold": fold,
                        "seeds": list(seeds),
                        "evaluation_date_indices": ix.tolist(),
                        "trajectories": records,
                        "round6_frozen_design_sha256": design_hash,
                    },
                )
                try:
                    evaluated = rr._evaluate(
                        **_context_arguments(
                            context, {"round6_frozen_design": design_hash}
                        ),
                        indices=ix,
                        scores=scores,
                        score_mask=mask,
                        fold=fold,
                        output=dest / "evaluation.json",
                        execution_policy=policy,
                        settle_terminal_residuals=True,
                    )
                    _finish_cell(dest, evaluated, name=arm, fold=fold)
                except RuntimeError as error:
                    if not isolated_occupancy_failure(error, arm, baseline="S0"):
                        raise
                    rejected[arm][fold] = str(error)
                    write_json_atomic(
                        dest / "rejected.json",
                        {
                            "engineering_acceptance": "failed",
                            "error": str(error),
                            "excluded_from_all_four_panel_choices": True,
                            "evaluation_sha256": sha256_file(dest / "evaluation.json"),
                            "score_manifest_sha256": sha256_file(
                                dest / "score_manifest.json"
                            ),
                        },
                    )
                    evaluated = retained(context, dest, fold)
                fields = rr._daily_series(evaluated)
                levels[arm][fold] = {key: _values(fields[key]) for key in (IC, NET)}
                member, member_mask, member_records = ensemble(
                    design,
                    root,
                    arm,
                    fold,
                    (omitted,),
                    context.store.dates[ix],
                    context.store.isins,
                    roster=roster,
                )
                np.testing.assert_array_equal(mask, member_mask)
                inputs = replace(
                    evaluated.inputs, scores=member, score_mask=member_mask
                )
                components = _primary_population_components(
                    inputs, TRADED_PRIMARY_HORIZONS
                )
                _, daily, _ = _primary_daily_metrics(
                    *components, inputs.dates, TRADED_PRIMARY_HORIZONS
                )
                solo[arm][fold] = _values(daily)
                singles[arm] = replace(
                    evaluated.result,
                    daily_primary_ic=daily,
                    primary_scores=components[0],
                    primary_targets=components[1],
                    primary_outcome_mask=components[2],
                    primary_score_mask=components[3],
                )
                write_json_atomic(
                    dest / "single_seed.json",
                    {
                        "seed": omitted,
                        "score_sources": member_records,
                        "daily_primary_ic": solo[arm][fold],
                        "promotion_weight": 0,
                    },
                )
                del evaluated, inputs
            for arm in arms:
                if arm == "S0":
                    continue
                delta, population = rr._paired_primary_daily(
                    singles[arm], singles["S0"]
                )
                single_pairs[arm][fold] = _values(delta)
                write_json_atomic(
                    destination / "single_paired" / arm / f"{fold}.json",
                    {
                        "daily_primary_ic_delta": _values(delta),
                        "population": population,
                    },
                )
        readouts = {
            a: {k: _pool(levels[a][f][k] for f in DEVELOPMENT_FOLDS) for k in (IC, NET)}
            for a in arms
        }
        support = {a: informative_folds(design, a, roster) for a in arms}
        pairs = {}
        for arm in arms:
            if arm != "S0":
                pairs.update(
                    paired_readouts(
                        context,
                        {arm: paths[arm], "S0": paths["S0"]},
                        destination / "paired",
                        informative_folds=support,
                    )
                )
        return write_json_atomic(
            destination / "result.json",
            {
                "status": "completed",
                "review_sha256": sha256_file(review / "result.json"),
                "seeds": list(seeds),
                "omitted_seed": omitted,
                "readouts": readouts,
                "paired": pairs,
                "daily_levels": levels,
                "rejected_books": {a: v for a, v in rejected.items() if v},
                "single_seed_primary_ic": {
                    a: {
                        "all_folds": _pool(solo[a].values()),
                        "informative": _pool(solo[a][f] for f in support[a]),
                        "daily_by_fold": solo[a],
                        "paired_to_S0": None
                        if a == "S0"
                        else {
                            "all_folds": _pool(single_pairs[a].values()),
                            "informative": _pool(
                                single_pairs[a][f] for f in support[a]
                            ),
                        },
                    }
                    for a in arms
                },
                "C6_roster_sensitivity": roster_sensitivity(pairs),
                "roster_changes_applied": False,
                "new_training_runs": 0,
                **rr.RESEARCH_FLAGS,
            },
        )
    finally:
        context.store.close()


def finish(root: Path, review: Path, output: Path) -> str:
    full = rr._read_json(review / "result.json")
    if full["frozen_design_sha256"] != sha256_file(root / "frozen_design.json"):
        raise ValueError("seed audit differs from the reviewed experiment")
    results = {
        f"omit_{s}": rr._read_json(output / f"omit_{s}" / "result.json")
        for s in ALLOWED_SEEDS
    }
    for seed in ALLOWED_SEEDS:
        panel_result = results[f"omit_{seed}"]
        if (
            panel_result["status"] != "completed"
            or panel_result["seeds"] != [s for s in ALLOWED_SEEDS if s != seed]
            or panel_result["review_sha256"] != sha256_file(review / "result.json")
        ):
            raise ValueError("seed audit requires all three matched fixed omissions")
    excluded = sorted({a for row in results.values() for a in row["rejected_books"]})
    panels = {"full": full, **results}
    design = rr._read_json(root / "frozen_design.json")
    roster = full["registered_C6_roster"]
    context = rr._open_ledger_replay(evaluation_design(design))
    traces = {}
    try:
        for name, result in panels.items():
            paths = {}
            for arm in full["readouts"]:
                group = "session1" if arm in ("S0", *SESSION1) else "session2"
                base = root / "aggregates" / group if name == "full" else output / name
                paths[arm] = {f: base / arm / f for f in DEVELOPMENT_FOLDS}
            pairs = leader_comparisons(
                context,
                paths,
                result["readouts"],
                dict(result["paired"]),
                output / "decision_pairs" / name,
                {a: informative_folds(design, a, roster) for a in paths},
                excluded=excluded,
            )
            traces[name] = promotion_trace(result["readouts"], pairs, excluded=excluded)
            write_json_atomic(output / "decision_pairs" / name / "result.json", pairs)
        path = output / "seed_audit_result.json"
        if path.exists():
            raise FileExistsError(path)
        return write_json_atomic(
            path,
            {
                "status": "completed",
                "review_sha256": sha256_file(review / "result.json"),
                "omission_result_sha256": {
                    str(s): sha256_file(output / f"omit_{s}" / "result.json")
                    for s in ALLOWED_SEEDS
                },
                "excluded_arms": excluded,
                "decision_traces": traces,
                "decision": seed_stability(traces),
                "C6_roster_sensitivity": {
                    k: v["C6_roster_sensitivity"] for k, v in results.items()
                },
                "roster_changes_applied": False,
                "new_training_runs": 0,
                "limitation": "overlapping_seed_sensitivity; not_independent_replication; temporal_CIs_conditional_on_fitted_ensemble",
                **rr.RESEARCH_FLAGS,
            },
        )
    finally:
        context.store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("panel", "finish"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--omitted", type=int, choices=ALLOWED_SEEDS)
    args = parser.parse_args()
    if args.action == "panel":
        if args.omitted is None:
            parser.error("panel requires --omitted")
        print(panel(args.root, args.review, args.output, args.omitted))
    else:
        print(finish(args.root, args.review, args.output))


if __name__ == "__main__":
    main()
