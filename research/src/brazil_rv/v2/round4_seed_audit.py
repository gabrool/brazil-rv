"""Fixed saved-score seed sensitivity after the user-authorized Round-4 budget amendment."""

from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import combinations
from multiprocessing import get_context
from pathlib import Path
import time

import numpy as np

from . import research_rounds as rr
from .artifacts import sha256_file, write_json_atomic
from .config import PROJECT_ROOT
from .contract import ALLOWED_SEEDS, DEVELOPMENT_FOLDS, TRADED_PRIMARY_HORIZONS
from .evaluate import _primary_daily_metrics, _primary_population_components
from .research_checkpoint import _context_arguments, _finish_cell
from .round4 import ARMS, completed, promotion_trace

REGISTRATION = PROJECT_ROOT / "research/preregistrations/v2_round4_budget_amendment.md"
METRICS = ("primary_neutral_target_ic", "headline_net_excess_bps")


def development_decision(full: dict, omissions: dict[int, dict]) -> dict:
    """Unanimity is a sensitivity condition, not an independent replication test."""
    panels = [full, *(omissions[seed] for seed in ALLOWED_SEEDS)]
    designations = [p["economics_override"] or p["ic_leader"] for p in panels]
    parents = [p["provisional_parent"] for p in panels]
    stable_designation = designations[0] is not None and len(set(designations)) == 1
    stable_parent = len(set(parents)) == 1 and all(
        parent in panel["eligible"]
        for parent, panel in zip(parents, panels, strict=True)
    )
    return {
        "panel_order": ["full_11_29_47", "omit_11", "omit_29", "omit_47"],
        "provisional_designations": designations,
        "provisional_parents": parents,
        "designation_stable": stable_designation,
        "research_designation": designations[0] if stable_designation else None,
        "parent_stable": stable_parent,
        "working_research_parent": parents[0] if stable_parent else "fast_off",
        "parent_inconclusive": not stable_parent,
        "confirmed_six_seed_panel": False,
        "independent_replication": False,
        "read_2025_authorized": False,
        "deployment_authorized": False,
    }


def _values(values):
    return [float(v) if np.isfinite(v) else None for v in values]


def _pooled(folds):
    return rr._folded_bootstrap(tuple(np.asarray(v, dtype=float) for v in folds))


def _audit_omission(source: Path, output: Path, omitted: int) -> dict:
    design = rr._read_json(source / "frozen_design.json")
    _, dates = rr._read_store_header(Path(design["store"]["root"]))
    context = rr._open_ledger_replay(design)
    policy, _ = rr.load_selected_policy(
        Path(design["execution_policy"]["root"]),
        expected_result_sha256=design["execution_policy"]["result_sha256"],
    )
    source_hashes = rr._read_json(
        source / "cpu_replay/baselines/momentum_12_1/F1/evaluation.json"
    )["source_artifact_hashes"]
    seeds = tuple(seed for seed in ALLOWED_SEEDS if seed != omitted)
    levels = {arm: {} for arm in ARMS}
    solo = {arm: {} for arm in ARMS}
    pair_days = {f"{a}_minus_{b}": {} for a, b in combinations(ARMS, 2)}
    try:
        for fold in DEVELOPMENT_FOLDS:
            results, singles = {}, {}
            for arm in ARMS:
                members = {}
                reference_mask = None
                manifests = {}
                for seed in ALLOWED_SEEDS:
                    run = source / "trajectories" / arm / f"{fold}_seed_{seed}"
                    score, mask = rr._score_artifact(
                        run / "scores",
                        require_clean_transfer=True,
                        expected_dates=dates[context.evaluation[fold]],
                        expected_isins=context.store.isins,
                        expected_feature_schema_sha256=design["feature_schema_sha256"],
                    )
                    if reference_mask is not None and not np.array_equal(
                        reference_mask, mask
                    ):
                        raise ValueError("seed score support differs")
                    reference_mask = mask
                    members[seed] = score
                    manifests[str(seed)] = sha256_file(run / "run_manifest.json")
                scores = rr.rank_average_ensemble(
                    [members[s] for s in seeds], reference_mask
                )
                dest = output / f"omit_{omitted}" / arm / fold
                rr._persist_scores(
                    dest,
                    {"scores": scores, "score_mask": reference_mask},
                    {
                        **rr._source_tier_labels(context.store.manifest),
                        "arm": arm,
                        "fold": fold,
                        "seeds": list(seeds),
                        "evaluation_date_indices": context.evaluation[fold].tolist(),
                        "source_run_manifests": manifests,
                        "budget_amendment_sha256": sha256_file(REGISTRATION),
                    },
                )
                evaluated = rr._evaluate(
                    **_context_arguments(context, source_hashes),
                    indices=context.evaluation[fold],
                    scores=scores,
                    score_mask=reference_mask,
                    fold=fold,
                    output=dest / "evaluation.json",
                    execution_policy=policy,
                    settle_terminal_residuals=True,
                )
                _finish_cell(dest, evaluated, name=arm, fold=fold)
                results[arm] = evaluated.result
                fields = rr._daily_series(evaluated)
                levels[arm][fold] = {key: _values(fields[key]) for key in METRICS}
                # Each worker records the one excluded member as a single-seed diagnostic.
                inputs = replace(evaluated.inputs, scores=members[omitted])
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
                del evaluated, inputs, members
            single_pairs = {}
            for arm in ARMS[1:]:
                delta, population = rr._paired_primary_daily(
                    singles[arm], singles["fast_off"]
                )
                single_pairs[arm] = {"delta": _values(delta), "population": population}
            write_json_atomic(
                output / f"single_{omitted}" / f"{fold}.json", single_pairs
            )
            for a, b in combinations(ARMS, 2):
                delta, population = rr._paired_primary_daily(results[a], results[b])
                key = f"{a}_minus_{b}"
                net = np.asarray(levels[a][fold][METRICS[1]], float) - np.asarray(
                    levels[b][fold][METRICS[1]], float
                )
                pair_days[key][fold] = {
                    METRICS[0]: _values(delta),
                    METRICS[1]: _values(net),
                }
                write_json_atomic(
                    output / f"omit_{omitted}/paired" / key / f"{fold}.json",
                    {"primary_population": population, "daily": pair_days[key][fold]},
                )
        readouts = {
            arm: {
                metric: _pooled([levels[arm][f][metric] for f in DEVELOPMENT_FOLDS])
                for metric in METRICS
            }
            for arm in ARMS
        }
        informative = design["round4_protocol"]["informative_folds"]["folds"]
        comparisons = {}
        for key, by_fold in pair_days.items():
            a, b = key.split("_minus_")
            for left, right, sign in ((a, b, 1), (b, a, -1)):
                comparisons[f"{left}_minus_{right}"] = {
                    "pooled": {
                        metric: _pooled(
                            [
                                sign * np.asarray(by_fold[f][metric], float)
                                for f in DEVELOPMENT_FOLDS
                            ]
                        )
                        for metric in METRICS
                    },
                    "informative_subsets": {
                        arm: {
                            "folds": informative[arm],
                            "pooled": {
                                metric: _pooled(
                                    [
                                        sign * np.asarray(by_fold[f][metric], float)
                                        for f in informative[arm]
                                    ]
                                )
                                for metric in METRICS
                            },
                        }
                        for arm in (left, right)
                    },
                }
        trace = promotion_trace(
            readouts,
            comparisons,
            confirmed=False,
            s0_informative=comparisons["S0_minus_fast_off"]["informative_subsets"][
                "S0"
            ],
        )
        result = {
            "omitted_seed": omitted,
            "ensemble_seeds": list(seeds),
            "readouts": readouts,
            "paired": comparisons,
            "daily_levels": levels,
            "single_seed_primary_ic": {
                arm: {
                    "pooled": _pooled([solo[arm][f] for f in DEVELOPMENT_FOLDS]),
                    "daily_by_fold": solo[arm],
                }
                for arm in ARMS
            },
            "promotion_trace": trace,
            "accepted_books": len(ARMS) * len(DEVELOPMENT_FOLDS),
            **rr.RESEARCH_FLAGS,
        }
        write_json_atomic(output / f"omit_{omitted}_result.json", result)
        return result
    finally:
        context.store.close()


def run(source: Path, output: Path) -> str:
    code = rr._git_identity()
    source = source.resolve(strict=True)
    seal = rr._read_json(source / "artifact_inventory.json")
    if (
        seal["status"] != "passed"
        or rr.inventory(source, exclude=set(seal["excluded_self"])) != seal["files"]
    ):
        raise ValueError(
            "seed audit requires the complete unchanged screening inventory"
        )
    screening = rr._read_json(source / "screening_result.json")
    if screening["status"] != "screened_requires_confirmation":
        raise ValueError("seed audit requires completed original screening")
    design = rr._read_json(source / "frozen_design.json")
    for arm in ARMS:
        for fold in DEVELOPMENT_FOLDS:
            for seed in ALLOWED_SEEDS:
                completed(
                    source / "trajectories" / arm / f"{fold}_seed_{seed}",
                    design,
                    arm,
                    "F",
                    seed,
                    fold,
                )
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        output / "frozen_design.json",
        {
            "schema": "BRAZIL_RV_V2_ROUND4_SEED_AUDIT_V1",
            "implementation": code,
            "source_root": str(source),
            "source_inventory_sha256": sha256_file(source / "artifact_inventory.json"),
            "source_screening_sha256": sha256_file(source / "screening_result.json"),
            "source_frozen_design_sha256": sha256_file(source / "frozen_design.json"),
            "budget_amendment_sha256": sha256_file(REGISTRATION),
            "new_training_runs": 0,
            "seed_panels": [
                [s for s in ALLOWED_SEEDS if s != omit] for omit in ALLOWED_SEEDS
            ],
            "all_arms": list(ARMS),
            "all_folds": list(DEVELOPMENT_FOLDS),
            "frozen_before_seed_audit": True,
            **rr.RESEARCH_FLAGS,
        },
    )
    try:
        processes = [
            get_context("spawn").Process(
                target=_audit_omission, args=(source, output, seed)
            )
            for seed in ALLOWED_SEEDS
        ]
        try:
            for process in processes:
                process.start()
            while any(process.is_alive() for process in processes):
                if any(process.exitcode not in (None, 0) for process in processes):
                    raise RuntimeError(
                        "seed audit worker failed; preserve partial root"
                    )
                time.sleep(0.25)
            if any(process.exitcode != 0 for process in processes):
                raise RuntimeError("seed audit worker failed; preserve partial root")
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                if process.pid is not None:
                    process.join()
        results = {
            seed: rr._read_json(output / f"omit_{seed}_result.json")
            for seed in ALLOWED_SEEDS
        }
        decision = development_decision(
            screening["promotion_trace"],
            {seed: result["promotion_trace"] for seed, result in results.items()},
        )
        digest = write_json_atomic(
            output / "seed_audit_result.json",
            {
                "status": "three_seed_development_complete",
                "source_screening_sha256": sha256_file(
                    source / "screening_result.json"
                ),
                "accepted_leave_one_out_books": sum(
                    r["accepted_books"] for r in results.values()
                ),
                "new_training_runs": 0,
                "decision": decision,
                "omission_result_sha256": {
                    str(seed): sha256_file(output / f"omit_{seed}_result.json")
                    for seed in ALLOWED_SEEDS
                },
                "limitation": "post_training_budget_amendment; overlapping_sensitivity_panels; not_independent_replication; time_intervals_conditional_on_fitted_ensemble",
                **rr.RESEARCH_FLAGS,
            },
        )
        return digest
    except BaseException as error:
        write_json_atomic(
            output / "seed_audit_stop.json",
            {"error": str(error), "at_utc": rr._utc_now()},
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.source, args.output), flush=True)


if __name__ == "__main__":
    main()
