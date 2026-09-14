"""Execute the prewritten remaining-fold gate without changing the frozen fits."""

import argparse
from pathlib import Path
import subprocess
import sys

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.contract import DEVELOPMENT_FOLDS, RUN_MANY_PLAN_SCHEMA
from brazil_rv.v2.portfolio_program import read
from brazil_rv.v2.round7 import SCREEN_FOLDS, SEEDS
from brazil_rv.v2.round7_score import score


def run(root):
    screen = read(root / "timing_comparison.json")
    if (
        not screen["point_estimate_gate"]
        or screen["accounting_acceptance_requires_review"]
    ):
        raise ValueError("registered timing screen gate did not pass")
    design = read(root / "frozen_design.json")
    jobs = []
    for fold in DEVELOPMENT_FOLDS:
        if fold in SCREEN_FOLDS:
            continue
        for seed in SEEDS:
            parent_dir = root / "parent" / f"pretrain_internal_seed_{seed}"
            parent = parent_dir / "selected.pt"
            digest = read(parent_dir / "run_manifest.json")["artifacts"]["selected.pt"]
            if sha256_file(parent) != digest:
                raise ValueError("compatible parent changed")
            output = root / "fine" / f"{fold}_seed_{seed}"
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
                str(root / "fine_recipe.json"),
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
                    "name": f"TL_all_F_{fold}_{seed}",
                    "seed": seed,
                    "fold": fold,
                    "run_dir": str(output),
                    "cwd": str(Path.cwd()),
                    "command": command,
                    "resume": True,
                    "expected_manifest": {"status": "completed", "stage": "F"},
                }
            )
    plan = root / "remaining_plan.json"
    payload = {
        "schema": RUN_MANY_PLAN_SCHEMA,
        "max_parallel": 6,
        "jobs": jobs,
        "screen_sha256": sha256_file(root / "timing_comparison.json"),
        "dispatch_script_sha256": sha256_file(Path(__file__)),
    }
    if plan.exists() and read(plan) != payload:
        raise ValueError("remaining-fold plan changed")
    write_json_atomic(plan, payload)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "brazil_rv.v2.run_many",
            "--plan",
            str(plan),
            "--manifest",
            str(root / "remaining_result.json"),
        ],
        check=True,
    )
    for seed in SEEDS:
        directory = root / "parent" / f"pretrain_internal_seed_{seed}"
        checkpoint = directory / "selected.pt"
        score(
            Path(design["store"]["root"]),
            checkpoint,
            root / "prelude" / f"seed_{seed}" / "scores",
            expected_sha256=read(directory / "run_manifest.json")["artifacts"][
                "selected.pt"
            ],
            parent_prelude=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
