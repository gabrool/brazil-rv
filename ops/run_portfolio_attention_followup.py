"""Bounded, source-bound rich early/late comparison using the existing trainer."""

import argparse
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.contract import RUN_MANY_PLAN_SCHEMA
from brazil_rv.v2.portfolio_program import read
from brazil_rv.v2.post_data_program import CELLS, RECIPES
from brazil_rv.v2.research_rounds import _git_identity
from brazil_rv.v2.round7 import SCREEN_FOLDS, SEEDS
from brazil_rv.v2.round7_data import PROJECT


def run(root, source):
    registration = (
        PROJECT / "research/preregistrations/v2_portfolio_attention_followup.md"
    )
    design = {
        "implementation": _git_identity(),
        "registration_sha256": sha256_file(registration),
        "source_design": str(source / "frozen_design.json"),
        "source_design_sha256": sha256_file(source / "frozen_design.json"),
        "store": read(source / "frozen_design.json")["store"],
        "cell": {**CELLS["TE_all"], "cell": "TL_all", "peer_timing": "late"},
        "parent_recipe": asdict(RECIPES["sam125"]),
        "fine_recipe": asdict(RECIPES["asam20"]),
        "folds": list(SCREEN_FOLDS),
        "seeds": list(SEEDS),
        "forward_capture": False,
        "heldout_accessed": False,
    }
    if root.exists():
        if read(root / "frozen_design.json") != design:
            raise ValueError("follow-up source or design changed")
    else:
        root.mkdir(parents=True)
        write_json_atomic(root / "frozen_design.json", design)
        write_json_atomic(root / "TL_all.json", design["cell"])
        for phase in ("parent", "fine"):
            write_json_atomic(root / f"{phase}_recipe.json", design[f"{phase}_recipe"])
    for stage, phase, folds in (
        ("P", "parent", ("pretrain_internal",)),
        ("F", "fine", SCREEN_FOLDS),
    ):
        jobs = []
        for fold in folds:
            for seed in SEEDS:
                output = root / phase / f"{fold}_seed_{seed}"
                command = [
                    sys.executable,
                    "-m",
                    "brazil_rv.v2.round7_training",
                    "--store",
                    design["store"]["root"],
                    "--output",
                    str(output),
                    "--cell-spec",
                    str(root / "TL_all.json"),
                    "--recipe",
                    str(root / f"{phase}_recipe.json"),
                    "--stage",
                    stage,
                    "--fold",
                    fold,
                    "--seed",
                    str(seed),
                    "--epochs",
                    "60",
                ]
                if stage == "F":
                    parent_dir = root / "parent" / f"pretrain_internal_seed_{seed}"
                    parent = parent_dir / "selected.pt"
                    digest = read(parent_dir / "run_manifest.json")["artifacts"][
                        "selected.pt"
                    ]
                    if sha256_file(parent) != digest:
                        raise ValueError("follow-up parent hash changed")
                    command += [
                        "--parent",
                        str(parent),
                        "--parent-sha256",
                        digest,
                        "--export-scores",
                    ]
                jobs.append(
                    {
                        "name": f"TL_all_{stage}_{fold}_{seed}",
                        "seed": seed,
                        "fold": fold,
                        "run_dir": str(output),
                        "cwd": str(PROJECT),
                        "command": command,
                        "resume": True,
                        "expected_manifest": {"status": "completed", "stage": stage},
                    }
                )
        plan = root / f"{phase}_plan.json"
        write_json_atomic(
            plan, {"schema": RUN_MANY_PLAN_SCHEMA, "max_parallel": 6, "jobs": jobs}
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "brazil_rv.v2.run_many",
                "--plan",
                str(plan),
                "--manifest",
                str(root / f"{phase}_result.json"),
            ],
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.source)
