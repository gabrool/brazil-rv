"""Plan triggered confirmation using the unchanged frozen training environment.

This file can be executed by absolute path outside the training checkout. Its
imports must resolve to that checkout's frozen package; the planner's own hash
is recorded separately. No training-code update or frozen-design rewrite is needed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from brazil_rv.v2 import research_rounds as rr
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.contract import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_FOLDS,
    RUN_MANY_PLAN_SCHEMA,
)
from brazil_rv.v2.round6 import (
    arm_config,
    completed,
    design_at,
    families_for,
    session2_roster,
    training_command,
    trajectory,
)
from brazil_rv.v2.train import model_config_contract


def confirmation_tasks(arms, phase):
    own_p = [a for a in arms if a in ("S0", "mlp") or a.endswith("fresh_p")]
    if phase == "smoke":
        return [(a, 11, "pretrain_internal", "P") for a in own_p if a in ("S0", "mlp")]
    if phase == "p":
        return [
            (a, s, "pretrain_internal", "P") for a in own_p for s in CONFIRMATION_SEEDS
        ]
    if phase == "f":
        return [
            (a, s, f, "F")
            for a in arms
            for f in DEVELOPMENT_FOLDS
            for s in CONFIRMATION_SEEDS
        ]
    raise ValueError("unknown confirmation phase")


def initialization_completed(run, design, arm, seed, *, roster=None):
    if arm not in ("S0", "mlp"):
        return completed(
            run, design, arm, "P", seed, "pretrain_internal", roster=roster
        )
    manifest = rr._read_json(run / "run_manifest.json")
    rr._assert_current_clean_training(manifest, path=run / "run_manifest.json")
    expected_config = model_config_contract(
        replace(arm_config(design["feature_names"], arm, stage="P"), use_bf16=False)
    )
    if (manifest["stage"], manifest["seed"], manifest["fold"]) != (
        "P",
        seed,
        "pretrain_internal",
    ):
        raise ValueError("confirmation parent identity differs")
    if (
        manifest["model_config"] != expected_config
        or manifest["checkpoint_input_contract"]["implementation_commit"]
        != design["initialization_recipe"]["commit"]
        or manifest["optimizer"]["effective_batch_date_pairs"] != 8
        or manifest["patience"]["maximum_epochs"] not in (1, 20)
        or manifest["compiled_graphs"] != {"training": 1, "selection": 1, "total": 2}
    ):
        raise ValueError(
            "confirmation parent differs from the screening initialization recipe"
        )
    for name, digest in manifest["artifacts"].items():
        if sha256_file(run / name) != digest:
            raise ValueError("confirmation parent artifact changed")
    return manifest


def initialization_command(command, checkout: Path, *, smoke: bool):
    arguments = list(command[command.index("-m") + 2 :])
    arguments.remove("--use-bf16")
    position = arguments.index("--selection-interval")
    del arguments[position : position + 2]
    arguments[arguments.index("--maximum-epochs") + 1] = "1" if smoke else "20"
    # The installed environment is shared. Resolve the historical package
    # explicitly in a fresh interpreter instead of mutating its installation.
    bootstrap = (
        f"import runpy,sys;sys.path.insert(0,{str(checkout / 'research/src')!r});"
        "runpy.run_module('brazil_rv.v2.train',run_name='__main__')"
    )
    return [sys.executable, "-c", bootstrap, *arguments]


def write_plan(
    root: Path,
    decision_path: Path,
    phase: str,
    initialization_checkout: Path | None = None,
) -> str:
    design = design_at(root)
    roster = session2_roster(root)
    decision = rr._read_json(decision_path)
    if decision["status"] != "completed" or decision[
        "frozen_design_sha256"
    ] != sha256_file(root / "frozen_design.json"):
        raise ValueError(
            "confirmation requires the completed matching fixed seed audit"
        )
    trace = decision["decision_traces"]["full"]
    arms = trace["confirmation_arms"]
    if not trace["confirmation_reasons"] or "S0" not in arms or len(arms) < 2:
        raise ValueError("registered conditional confirmation did not trigger")
    if trace["confirmation_seeds"] != list(CONFIRMATION_SEEDS):
        raise ValueError("confirmation seeds differ from registration")
    required = {"S0", trace["provisional_designation"]}
    if trace["economics_override"]:
        required.add(trace["ic_leader"])
    if set(arms) != required or len(arms) != len(required):
        raise ValueError("confirmation roster differs from both matched claims")
    for arm in arms:
        families_for(arm, roster)
    tasks = confirmation_tasks(arms, phase)
    smoke = phase == "smoke"
    if any(stage == "P" and arm in ("S0", "mlp") for arm, _, _, stage in tasks):
        if initialization_checkout is None:
            raise ValueError("confirmation requires the frozen initialization checkout")
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=initialization_checkout, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=initialization_checkout, text=True
        ).strip()
        if source_commit != design["initialization_recipe"]["commit"] or dirty:
            raise ValueError(
                "initialization checkout differs from its clean frozen commit"
            )
    if not smoke:
        for arm, stage in sorted({(a, st) for a, _, _, st in tasks}):
            run = root / "smoke" / f"{arm}_{stage}"
            manifest = (
                initialization_completed(run, design, arm, 11, roster=roster)
                if stage == "P"
                else completed(
                    run,
                    design,
                    arm,
                    stage,
                    11,
                    "pretrain_internal" if stage == "P" else "F14",
                    roster=roster,
                )
            )
            if manifest["epochs_completed"] != 1 or (run / "scores").exists():
                raise ValueError("confirmation requires score-free one-epoch smokes")
    jobs = []
    for arm, seed, fold, stage in tasks:
        run = (
            root / "smoke" / f"{arm}_{stage}"
            if smoke
            else trajectory(root, arm, seed, fold, stage)
        )
        checkpoint = digest = None
        if stage == "F" and not smoke:
            parent = arm if arm == "mlp" or arm.endswith("fresh_p") else "S0"
            pretrain = trajectory(root, parent, seed, "pretrain_internal", "P")
            manifest = initialization_completed(
                pretrain, design, parent, seed, roster=roster
            )
            checkpoint = pretrain / "raw_patience.pt"
            digest = manifest["artifacts"]["raw_patience.pt"]
        command = training_command(
            {**design, "store": design["s0_store"]}
            if stage == "P" and arm == "S0"
            else design,
            run,
            arm,
            seed,
            fold,
            stage,
            smoke=smoke,
            checkpoint=checkpoint,
            digest=digest,
            reused_s0=stage == "F" and parent == "S0",
            roster=roster,
        )
        if stage == "P" and arm in ("S0", "mlp"):
            command = initialization_command(
                command, initialization_checkout, smoke=smoke
            )
        job = rr._plan_job(
            name=f"{arm}_{stage}_{fold}_{seed}",
            seed=seed,
            fold=fold,
            run_dir=run,
            command=command,
            stage=stage,
            source_tiers=rr._source_tier_labels({"metadata": design}),
        )
        if not smoke and stage == "F":
            job["expected_manifest"]["scoring_complete"] = True
        jobs.append(job)
    path = root / f"round6_plan_confirmation_{phase}.json"
    if path.exists():
        raise FileExistsError(path)
    return write_json_atomic(
        path,
        {
            "schema": RUN_MANY_PLAN_SCHEMA,
            "phase": f"confirmation_{phase}",
            "max_parallel": 6,
            "first_failure_stop": True,
            "jobs": jobs,
            "research_candidate_score": not smoke and phase == "f",
            "smoke_reused": False,
            "decision_sha256": sha256_file(decision_path),
            "frozen_design_sha256": sha256_file(root / "frozen_design.json"),
            "planner_sha256": sha256_file(Path(__file__)),
            "confirmation_arms": arms,
            "confirmation_seeds": list(CONFIRMATION_SEEDS),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "p", "f"), required=True)
    parser.add_argument("--initialization-checkout", type=Path)
    args = parser.parse_args()
    print(
        write_plan(args.root, args.decision, args.phase, args.initialization_checkout)
    )


if __name__ == "__main__":
    main()
