"""Bounded controller dispatch and paired chronological screen decisions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from .artifacts import sha256_file, write_json_atomic
from .contract import ALLOWED_SEEDS, DEVELOPMENT_FOLDS, RUN_MANY_PLAN_SCHEMA
from .portfolio_program import PROJECT, read
from .portfolio_readouts import interval, verify_book
from .research_rounds import _git_identity
from .round7 import SCREEN_FOLDS

ARMS = ("TE_all", "C6")
KINDS = ("reliability", "stateful")
METRICS = ("net_excess_bps", "utility_bps")


def seeds_for(kind):
    return ALLOWED_SEEDS[:1] if kind == "reliability" else ALLOWED_SEEDS


def policy_path(root, arm, fold, kind, seed):
    return root / "phase2/policies" / arm / fold / kind / f"seed_{seed}"


def plan(root, *, requested_kind, confirmation=False, max_parallel=6):
    engineering = {}
    for kind in (requested_kind,):
        for seed in seeds_for(kind):
            path = (
                root
                / "phase2/behavioral_acceptance"
                / kind
                / f"seed_{seed}/acceptance.json"
            )
            if not read(path)["passed"]:
                raise ValueError(f"behavioral learning not accepted: {kind}/{seed}")
            engineering[str(path.relative_to(root))] = sha256_file(path)
    if confirmation:
        screen = read(root / "phase2/screen_summary.json")
        cells = [tuple(cell) for cell in screen["survivors"]]
        if requested_kind == "reliability":
            cells = sorted({(arm, "reliability") for arm, _ in cells})
        folds = tuple(f for f in DEVELOPMENT_FOLDS if f not in SCREEN_FOLDS)
    else:
        cells = [
            (arm, kind)
            for arm in ARMS
            for kind in read(root / "phase2/engineering_acceptance.json")[
                "financial_kinds"
            ]
        ]
        folds = SCREEN_FOLDS
    jobs = []
    for fold in folds:
        for arm, kind in cells:
            if kind != requested_kind:
                continue
            for seed in seeds_for(kind):
                jobs.append(
                    {
                        "name": f"{arm}_{fold}_{kind}_{seed}",
                        "seed": seed,
                        "fold": fold,
                        "run_dir": str(policy_path(root, arm, fold, kind, seed)),
                        "cwd": str(PROJECT),
                        "resume": True,
                        "command": [
                            sys.executable,
                            "-m",
                            "brazil_rv.v2.controller_training",
                            "fit",
                            "--root",
                            str(root),
                            "--arm",
                            arm,
                            "--fold",
                            fold,
                            "--kind",
                            kind,
                            "--seed",
                            str(seed),
                        ],
                    }
                )
    result = {
        "schema": RUN_MANY_PLAN_SCHEMA,
        "implementation": _git_identity(),
        "registration_sha256": sha256_file(
            PROJECT / "research/preregistrations/v2_decision_phase2.md"
        ),
        "engineering": engineering,
        "max_parallel": max_parallel,
        "jobs": jobs,
        "conditional_linear_seeds_are_exact_aliases": True,
    }
    path = (
        root
        / "phase2"
        / f"{'confirmation' if confirmation else 'screen'}_{requested_kind}_plan.json"
    )
    write_json_atomic(path, result)
    return {"plan": str(path), "unique_fits": len(jobs)}


def paired(candidate, control, metric):
    if candidate["dates"] != control["dates"]:
        raise ValueError("paired controller books have different dates")
    return np.asarray(candidate["daily"][metric]) - control["daily"][metric]


def checked_book(path):
    record = read(path / "book.json")
    return verify_book(path, record["provenance"])


def summarize(root, *, confirmation=False):
    folds = (
        tuple(f for f in DEVELOPMENT_FOLDS if f not in SCREEN_FOLDS)
        if confirmation
        else SCREEN_FOLDS
    )
    cells = (
        [tuple(c) for c in read(root / "phase2/screen_summary.json")["survivors"]]
        if confirmation
        else [
            (a, k)
            for a in ARMS
            for k in read(root / "phase2/engineering_acceptance.json")[
                "financial_kinds"
            ]
        ]
    )
    output = {
        "cells": {},
        "survivors": [],
        "heldout_accessed": False,
        "engineering": read(root / "phase2/engineering_acceptance.json"),
    }
    for arm, kind in cells:
        records, contrasts, seed_points = {}, {}, {}
        for seed in seeds_for(kind):
            records[str(seed)] = {}
            for fold in folds:
                path = policy_path(root, arm, fold, kind, seed)
                record = read(path / "run_manifest.json")
                if sha256_file(path / "selected.pt") != record["selected_sha256"]:
                    raise ValueError("selected controller checkpoint changed")
                records[str(seed)][fold] = {
                    "manifest_sha256": sha256_file(path / "run_manifest.json"),
                    "selected_epoch": record["selected_epoch"],
                    "history": record["history"],
                    "fallback": record["fallback"],
                    "summaries": record["books"],
                }
        for policy in ("candidate", "fallback"):
            contrasts[policy] = {}
            references = ("benchmark", "equal_rank", "cash") + (
                ("conditional",) if kind == "stateful" else ()
            )
            for reference in references:
                contrasts[policy][reference] = {}
                for metric in METRICS:
                    arrays, fold_points, by_seed = (
                        [],
                        {},
                        {s: [] for s in seeds_for(kind)},
                    )
                    for fold in folds:
                        control = checked_book(
                            policy_path(root, arm, fold, "reliability", 11)
                            / "candidate"
                            if reference == "conditional"
                            else root / "phase2/references" / arm / fold / reference
                        )
                        differences = []
                        for seed in seeds_for(kind):
                            book = checked_book(
                                policy_path(root, arm, fold, kind, seed) / policy
                            )
                            difference = paired(book, control, metric)
                            differences.append(difference)
                            by_seed[seed].append(difference)
                        average = np.mean(differences, axis=0)
                        arrays.append(average)
                        fold_points[fold] = float(average.mean())
                    contrasts[policy][reference][metric] = {
                        "mean": float(np.concatenate(arrays).mean()),
                        "fold_means": fold_points,
                        "intervals": {
                            str(b): interval(arrays, block_length=b)
                            for b in (20, 40, 60)
                        },
                    }
                    if policy == "candidate" and reference == "benchmark":
                        seed_points[metric] = {
                            str(s): float(np.concatenate(v).mean())
                            for s, v in by_seed.items()
                        }
        primary = contrasts["candidate"]["benchmark"]
        positive_folds = sum(
            v > 0 for v in primary["utility_bps"]["fold_means"].values()
        )
        checks = {
            "positive_paired_net": primary["net_excess_bps"]["mean"] > 0,
            "positive_paired_utility": primary["utility_bps"]["mean"] > 0,
            "positive_utility_three_quarters_of_folds": positive_folds
            >= int(np.ceil(0.75 * len(folds))),
            "no_material_seed_reversal": min(
                v for metric in seed_points.values() for v in metric.values()
            )
            >= -0.25,
        }
        if kind == "stateful":
            additional = contrasts["candidate"]["conditional"]
            checks["incremental_net_and_utility"] = all(
                additional[m]["mean"] > 0 for m in METRICS
            )
            checks["incremental_utility_three_quarters_of_folds"] = sum(
                v > 0 for v in additional["utility_bps"]["fold_means"].values()
            ) >= int(np.ceil(0.75 * len(folds)))
        if all(checks.values()):
            output["survivors"].append([arm, kind])
        output["cells"][f"{arm}/{kind}"] = {
            "seeds": records,
            "seed_paired_means": seed_points,
            "contrasts": contrasts,
            "gate": checks,
            "controller_seed_replications": 1 if kind == "reliability" else 3,
        }
    output["interpretation"] = (
        "Nominal development inference; seed-mean portfolios are descriptive, not an executable ensemble. Candidate gate is separate from the prior-selection fallback."
    )
    write_json_atomic(
        root
        / "phase2"
        / ("confirmation_summary.json" if confirmation else "screen_summary.json"),
        output,
    )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--kind", choices=KINDS)
    args = parser.parse_args()
    result = (
        plan(
            args.root,
            requested_kind=args.kind,
            confirmation=args.confirmation,
            max_parallel=args.max_parallel,
        )
        if args.command == "plan"
        else summarize(args.root, confirmation=args.confirmation)
    )
    print({k: v for k, v in result.items() if k != "cells"})


if __name__ == "__main__":
    main()
