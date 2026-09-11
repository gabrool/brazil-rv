"""Plan triggered confirmation using the unchanged frozen training environment.

This file can be executed by absolute path outside the training checkout. Its
imports must resolve to that checkout's frozen package; the planner's own hash
is recorded separately. No training-code update or frozen-design rewrite is needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from brazil_rv.v2 import research_rounds as rr
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.contract import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_FOLDS,
    RUN_MANY_PLAN_SCHEMA,
)
from brazil_rv.v2.round6 import (
    completed,
    design_at,
    families_for,
    session2_roster,
    training_command,
    trajectory,
)


def confirmation_tasks(arms, phase):
    own_p = [a for a in arms if a in ("S0", "mlp") or a.endswith("fresh_p")]
    if phase == "smoke":
        # All non-S0 configurations already passed their registered full runs.
        return [("S0", 11, "pretrain_internal", "P"), ("S0", 11, "F14", "F")]
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


def write_plan(root: Path, decision_path: Path, phase: str) -> str:
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
    if not smoke:
        for arm, stage in sorted({(a, st) for a, _, _, st in tasks}):
            run = root / "smoke" / f"{arm}_{stage}"
            manifest = completed(
                run,
                design,
                arm,
                stage,
                11,
                "pretrain_internal" if stage == "P" else "F14",
                roster=roster,
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
            manifest = completed(
                pretrain, design, parent, "P", seed, "pretrain_internal", roster=roster
            )
            checkpoint = pretrain / "raw_patience.pt"
            digest = manifest["artifacts"]["raw_patience.pt"]
        command = training_command(
            design,
            run,
            arm,
            seed,
            fold,
            stage,
            smoke=smoke,
            checkpoint=checkpoint,
            digest=digest,
            roster=roster,
        )
        # Fresh confirmation S0 P is fitted on the current store schema. Do not
        # send it through the older archived-store transfer path.
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
    args = parser.parse_args()
    print(write_plan(args.root, args.decision, args.phase))


if __name__ == "__main__":
    main()
