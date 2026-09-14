"""Source-bound forecast jobs for the cash-aware portfolio experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .artifacts import sha256_file, write_json_atomic
from .contract import ALLOWED_SEEDS, DEVELOPMENT_FOLDS, RUN_MANY_PLAN_SCHEMA
from .data_roots import resolve_external_root
from .post_data_program import CELLS, RECIPES
from .research_rounds import _git_identity
from .round6 import training_command
from .round7 import SCREEN_FOLDS
from .round7_data import PROJECT


ARMS = ("S0", "TE_all", "C6")
RECIPES_BY_ARM = {"S0": "incumbent", "TE_all": "asam20", "C6": "incumbent"}
C6_ROSTER = {"c6_families": ["fundamentals", "magnitudes"]}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_root():
    if os.name == "nt":
        archive = read(PROJECT / "docs/v2_post_data_recovery.json")[
            "archives_in_restore_order"
        ][0]["archive"]
        return Path(archive).parent.parent
    path = read(PROJECT / "docs/v2_post_data_screen.json")["aggregate_paths"]["S0"][
        "F2"
    ]
    return Path(path).parents[2]


def fit_path(root, arm, fold, seed):
    return root / "forecasters" / arm / f"{fold}_seed_{seed}"


def frozen_fit_roster():
    """Original screen fits are reused, never resubmitted under a new identity."""
    roster = [
        (arm, fold, seed)
        for arm in ARMS
        for fold in DEVELOPMENT_FOLDS
        if arm == "C6" or fold not in SCREEN_FOLDS
        for seed in ALLOWED_SEEDS
    ]
    # Complete the first out-of-fit block, then the informative C6 bridge,
    # while later independent folds continue on the same paid instance.
    return sorted(
        roster,
        key=lambda item: (
            0
            if item[1] == "F1"
            else 1
            if item[0] == "C6" and item[1] in SCREEN_FOLDS
            else 2,
            int(item[1][1:]),
            ARMS.index(item[0]),
            item[2],
        ),
    )


def freeze(root):
    implementation = _git_identity()
    pointer = PROJECT / "docs/v2_data_inputs.json"
    inputs = read(pointer)
    store, resolution = resolve_external_root(inputs["store"]["root"])
    if (
        not inputs["accepted"]
        or sha256_file(store / "manifest.json") != inputs["store"]["manifest_sha256"]
    ):
        raise ValueError("accepted data pointer does not bind the forecast store")
    source = source_root()
    inventory = {
        r["path"]: r["sha256"]
        for r in read(PROJECT / "docs/v2_post_data_final_inventory.json")
    }

    def bound(relative):
        path = source / relative
        if sha256_file(path) != inventory[relative]:
            raise ValueError(f"sealed reuse source changed: {relative}")
        return {"path": str(path), "sha256": inventory[relative]}

    old = read(bound("frozen_design.json")["path"])
    if old["store"]["manifest_sha256"] != inputs["store"]["manifest_sha256"]:
        raise ValueError("A-C weights do not bind accepted current coordinates")
    parents = {}
    reused = {}
    for arm in ("S0", "TE_all"):
        checkpoint = "raw_patience.pt" if arm == "S0" else "selected.pt"
        parents[arm] = {}
        reused[arm] = {}
        for seed in ALLOWED_SEEDS:
            relative = f"parents/{arm}/seed_{seed}"
            parents[arm][str(seed)] = {
                "checkpoint": bound(f"{relative}/{checkpoint}"),
                "manifest": bound(f"{relative}/run_manifest.json"),
            }
        for fold in SCREEN_FOLDS:
            reused[arm][fold] = {}
            for seed in ALLOWED_SEEDS:
                relative = f"fits/{arm}/{RECIPES_BY_ARM[arm]}/{fold}_seed_{seed}"
                reused[arm][fold][str(seed)] = {
                    "manifest": bound(f"{relative}/run_manifest.json"),
                    "scores": bound(f"{relative}/scores/score_manifest.json"),
                }
    registration = PROJECT / "research/preregistrations/v2_portfolio_policy.md"
    design = {
        "implementation": implementation,
        "registration_sha256": sha256_file(registration),
        "pointer_sha256": sha256_file(pointer),
        "store": {**inputs["store"], "root": str(store)},
        "store_resolution": resolution.payload(),
        "s0_store": {**inputs["store"], "root": str(store)},
        "feature_names": old["feature_names"],
        "feature_schema_sha256": old["feature_schema_sha256"],
        "economic_source": old["economic_source"],
        "fast_initialization": old["fast_initialization"],
        "source_root": str(source),
        "source_inventory_sha256": sha256_file(
            PROJECT / "docs/v2_post_data_final_inventory.json"
        ),
        "parents": parents,
        "reused_fits": reused,
        "new_fits": frozen_fit_roster(),
        "c6_roster": C6_ROSTER,
        "seeds": ALLOWED_SEEDS,
        "folds": DEVELOPMENT_FOLDS,
        "forecast_ensemble": "equal seed average of within-date/name/head midranks",
        "forward_capture": False,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    root.mkdir(parents=True, exist_ok=False)
    write_json_atomic(root / "TE_all.json", CELLS["TE_all"])
    from dataclasses import asdict

    write_json_atomic(root / "asam20.json", asdict(RECIPES["asam20"]))
    write_json_atomic(root / "frozen_design.json", design)
    return {
        "root": str(root),
        "new_P": 3,
        "new_F": len(design["new_fits"]),
        "reused_F": 24,
    }


def parent_binding(root, design, arm, seed, *, prelude=False):
    if arm == "C6" and prelude:
        directory = root / "parents" / "C6_bootstrap" / f"seed_{seed}"
        record = read(directory / "run_manifest.json")
        path = directory / "raw_patience.pt"
        digest = record["artifacts"][path.name]
    else:
        record = design["parents"]["S0" if arm == "C6" else arm][str(seed)][
            "checkpoint"
        ]
        path, digest = Path(record["path"]), record["sha256"]
    if sha256_file(path) != digest:
        raise ValueError("forecast parent weights changed")
    return path, digest


def plan(root, phase, *, max_parallel=6):
    design = read(root / "frozen_design.json")
    if design["implementation"]["commit"] != _git_identity()["commit"]:
        raise ValueError("forecast checkout differs from frozen implementation")
    if (
        sha256_file(PROJECT / "research/preregistrations/v2_portfolio_policy.md")
        != design["registration_sha256"]
    ):
        raise ValueError("forecast registration changed after freeze")
    tasks = (
        [("C6", "pretrain_internal", seed) for seed in ALLOWED_SEEDS]
        if phase == "parents"
        else [(arm, "parent_prelude", seed) for arm in ARMS for seed in ALLOWED_SEEDS]
        if phase == "prelude"
        else design["new_fits"]
    )
    jobs = []
    for arm, fold, seed in tasks:
        stage = "P" if phase == "parents" else "F"
        output = (
            root / "parents" / "C6_bootstrap" / f"seed_{seed}"
            if phase == "parents"
            else root / "prelude" / arm / f"seed_{seed}"
            if phase == "prelude"
            else fit_path(root, arm, fold, seed)
        )
        if phase == "prelude":
            parent_binding(root, design, arm, seed, prelude=True)
            command = [
                sys.executable,
                "-m",
                "brazil_rv.v2.portfolio_program",
                "score-prelude",
                "--root",
                str(root),
                "--arm",
                arm,
                "--seed",
                str(seed),
            ]
        else:
            parent, digest = (
                (None, None)
                if stage == "P"
                else parent_binding(root, design, arm, seed)
            )
            if arm in ("S0", "C6"):
                command = training_command(
                    design,
                    output,
                    arm,
                    seed,
                    fold,
                    stage,
                    checkpoint=parent,
                    digest=digest,
                    reused_s0=arm == "C6" and stage == "F",
                    roster=C6_ROSTER,
                )
                # Representation/policy comparison needs actual forecasts once;
                # do not rerun per-family inference ablations for every fold.
                if "--score-sidecar-ablations" in command:
                    command.remove("--score-sidecar-ablations")
            else:
                command = [
                    sys.executable,
                    "-m",
                    "brazil_rv.v2.round7_training",
                    "--store",
                    design["store"]["root"],
                    "--output",
                    str(output),
                    "--cell-spec",
                    str(root / "TE_all.json"),
                    "--recipe",
                    str(root / "asam20.json"),
                    "--stage",
                    "F",
                    "--fold",
                    fold,
                    "--seed",
                    str(seed),
                    "--epochs",
                    "60",
                    "--parent",
                    str(parent),
                    "--parent-sha256",
                    digest,
                    "--export-scores",
                ]
        jobs.append(
            {
                "name": f"{phase}_{arm}_{fold}_{seed}",
                "seed": seed,
                "fold": fold,
                "run_dir": str(output),
                "cwd": str(PROJECT),
                "command": command,
                "expected_manifest": {
                    "status": "completed",
                    "stage": "forecast" if phase == "prelude" else stage,
                },
                "resume": arm == "TE_all" and phase == "forecasters",
            }
        )
    path = root / f"{phase}_plan.json"
    write_json_atomic(
        path,
        {"schema": RUN_MANY_PLAN_SCHEMA, "max_parallel": max_parallel, "jobs": jobs},
    )
    return {"path": str(path), "jobs": len(jobs)}


def score_prelude(root, arm, seed):
    design = read(root / "frozen_design.json")
    checkpoint, digest = parent_binding(root, design, arm, seed, prelude=True)
    output = root / "prelude" / arm / f"seed_{seed}"
    if arm == "TE_all":
        from .round7_score import score

        score(
            Path(design["store"]["root"]),
            checkpoint,
            output / "scores",
            expected_sha256=digest,
            parent_prelude=True,
        )
    else:
        from .score import main as score_main

        score_main(
            [
                "--store",
                design["store"]["root"],
                "--checkpoint",
                str(checkpoint),
                "--checkpoint-sha256",
                digest,
                "--output-dir",
                str(output / "scores"),
                "--parent-prelude",
                "--device",
                "cuda",
            ]
        )
    return write_json_atomic(
        output / "run_manifest.json",
        {
            "status": "completed",
            "stage": "forecast",
            "fold": "parent_prelude",
            "seed": seed,
            "arm": arm,
            "parent_sha256": digest,
            "score_manifest_sha256": sha256_file(output / "scores/score_manifest.json"),
            "frozen_design_sha256": sha256_file(root / "frozen_design.json"),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "plan", "score-prelude"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--phase", choices=("parents", "prelude", "forecasters"))
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=ALLOWED_SEEDS)
    args = parser.parse_args()
    result = (
        freeze(args.root)
        if args.command == "freeze"
        else plan(args.root, args.phase, max_parallel=args.max_parallel)
        if args.command == "plan"
        else score_prelude(args.root, args.arm, args.seed)
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
