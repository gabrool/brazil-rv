"""Run the user-authorized ASAM .5 extension without changing the active freeze."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic

ATTENTION = ("TE_slow", "TL_slow", "TE_family", "TE_all")


def prepare(root: Path, registration: Path) -> dict:
    original = json.loads((root / "c_screen_plan.json").read_text())
    design = json.loads((root / "frozen_design.json").read_text())
    jobs = []
    for source in original["jobs"]:
        cell = (
            source["expected_manifest"].get("contract", {}).get("cell", {}).get("cell")
        )
        if cell not in ATTENTION:
            continue
        job = copy.deepcopy(source)
        old = Path(job["run_dir"])
        destination = old.parent.parent / "asam50" / old.name
        command = job["command"]
        command[command.index("--output") + 1] = str(destination)
        command[command.index("--recipe") + 1] = str(root / "recipes/asam50.json")
        job["run_dir"] = str(destination)
        job["name"] = f"{cell}_asam50_F_{job['fold']}_{job['seed']}"
        job["expected_manifest"]["contract"]["recipe"] = design["recipes"]["asam50"]
        jobs.append(job)
    expected = {
        (c, f, s)
        for c in ATTENTION
        for f in design["screen_folds"]
        for s in design["seeds"]
    }
    actual = {
        (j["expected_manifest"]["contract"]["cell"]["cell"], j["fold"], j["seed"])
        for j in jobs
    }
    if actual != expected or len(jobs) != len(expected):
        raise ValueError("Extension is not the exact four-cell matched roster")
    plan = {**original, "jobs": jobs, "max_parallel": 6}
    plan_path = root / "attention_asam50_plan.json"
    amendment_path = root / "attention_asam50_amendment.json"
    if plan_path.exists() or amendment_path.exists():
        raise FileExistsError("Attention extension is already frozen")
    write_json_atomic(plan_path, plan)
    amendment = {
        "authorization": "User explicitly requested attention with .5 in addition to current .2 on 2026-09-13 America/Sao_Paulo",
        "cells": ATTENTION,
        "recipe": "asam50",
        "folds": design["screen_folds"],
        "seeds": design["seeds"],
        "scored_fits": len(jobs),
        "new_fits": len(jobs) - 4,
        "reused_fits": "TE_slow F2/F14 seeds11/29 from Stage B; score attachment only",
        "parents": "Same selected SAM .125 parents as the corresponding .2 fits; no new P fits",
        "frozen_design_sha256": sha256_file(root / "frozen_design.json"),
        "original_screen_plan_sha256": sha256_file(root / "c_screen_plan.json"),
        "calibration_choice_sha256": sha256_file(root / "calibration_choice.json"),
        "extension_plan_sha256": sha256_file(plan_path),
        "registration_sha256": sha256_file(registration),
        "launcher_sha256": sha256_file(Path(__file__)),
        "interpretation": "Additional matched development comparison, not a replacement calibration choice or automatic promotion; all other training and book contracts unchanged",
    }
    write_json_atomic(amendment_path, amendment)
    return amendment


def wait_completed(path: Path) -> None:
    deadline = time.monotonic() + 6 * 3600
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Dependency did not complete: {path}")
        time.sleep(30)
    if json.loads(path.read_text())["status"] != "completed":
        raise RuntimeError(f"Dependency failed: {path}")


def run(root: Path, readout_repo: Path) -> None:
    amendment = json.loads((root / "attention_asam50_amendment.json").read_text())
    plan = root / "attention_asam50_plan.json"
    if sha256_file(plan) != amendment["extension_plan_sha256"]:
        raise ValueError("Extension plan changed after registration")
    # Keep the aggregate GPU concurrency at six; overlap with the original CPU readouts.
    wait_completed(root / "c_screen_result.json")
    start = time.time()
    print(json.dumps({"starting": "attention_asam50", "unix_time": start}), flush=True)
    subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "brazil_rv.v2.run_many",
            "--plan",
            str(plan),
            "--manifest",
            str(root / "attention_asam50_result.json"),
        ],
        check=True,
    )
    print(
        json.dumps(
            {"completed": "attention_asam50_fits", "seconds": time.time() - start}
        ),
        flush=True,
    )
    wait_completed(root / "screen_result.json")
    # A separate clean checkout supplies only the extended evaluation implementation.
    # The training checkout/commands remain frozen throughout every fit.
    environment = {**os.environ, "PYTHONPATH": str(readout_repo / "research/src")}
    subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "brazil_rv.v2.post_data_readouts",
            "--root",
            str(root),
            "--attention-asam50",
        ],
        cwd=readout_repo,
        env=environment,
        check=True,
    )
    print(json.dumps({"completed": "attention_asam50_readouts"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--readout-repo", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.root, args.registration)))
    else:
        run(args.root, args.readout_repo)
