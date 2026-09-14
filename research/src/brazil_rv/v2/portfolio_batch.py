"""Bounded offline CPU jobs; forecaster GPU dispatch remains separate."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
from pathlib import Path
import time

from .artifacts import sha256_file, write_json_atomic
from .contract import ALLOWED_SEEDS, DEVELOPMENT_FOLDS
from .portfolio_program import ARMS, SCREEN_FOLDS
from .portfolio_readouts import evaluate_books, summarize
from .portfolio_training import fit_policy, load_data, prepare
from .research_rounds import _git_identity


def jobs(phase):
    if phase == "continuous":
        return [(arm, None) for arm in ARMS]
    ordered = sorted(
        DEVELOPMENT_FOLDS, key=lambda f: (f not in SCREEN_FOLDS, int(f[1:]))
    )
    return [(arm, fold) for fold in ordered for arm in ARMS]


def run_job(root, arm, fold):
    started = time.monotonic()
    data, binding = load_data(root, arm)
    if fold is not None:
        for seed in ALLOWED_SEEDS:
            fit_policy(data, root, arm, fold, seed, binding)
    evaluate_books(
        root, arm, fold=fold, continuous=fold is None, loaded=(data, binding)
    )
    label = "continuous" if fold is None else fold
    books = root / "books" / arm / label
    record = {
        "status": "completed",
        "arm": arm,
        "fold": label,
        "seeds": list(ALLOWED_SEEDS),
        "seconds": time.monotonic() - started,
        "policy_data_sha256": binding,
        "books": {
            str(p.relative_to(books)): sha256_file(p)
            for p in books.glob("*/*/book.json")
        },
    }
    write_json_atomic(root / "policy_jobs" / arm / label / "run_manifest.json", record)
    return record


def run(root, phase, workers):
    implementation = _git_identity()
    # Child processes own one small sparse solve at a time. Oversubscribing BLAS
    # harms these sequential kernels and also competes with GPU data staging.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    if phase == "prepare":
        # Economics/beta are shared. Build them once before independent readers.
        for arm in ARMS:
            prepare(root, arm)
        return
    if phase == "summarize":
        summarize(root)
        return
    tasks = jobs(phase)
    progress = {
        "status": "running",
        "phase": phase,
        "implementation": implementation,
        "workers": min(workers, len(tasks)),
        "completed": [],
        "failed": [],
    }
    output = root / f"portfolio_{phase}_dispatch.json"
    write_json_atomic(output, progress)
    with ProcessPoolExecutor(
        max_workers=progress["workers"], mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {
            pool.submit(run_job, root, arm, fold): (arm, fold) for arm, fold in tasks
        }
        for future in as_completed(futures):
            try:
                progress["completed"].append(future.result())
            except Exception as error:
                progress["failed"].append(
                    {"job": futures[future], "error": repr(error)}
                )
                for pending in futures:
                    pending.cancel()
            write_json_atomic(output, progress)
    progress["status"] = "failed" if progress["failed"] else "completed"
    write_json_atomic(output, progress)
    if progress["failed"]:
        raise RuntimeError(f"portfolio phase failed; inspect {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("prepare", "policies", "continuous", "summarize")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    run(args.root, args.phase, args.workers)


if __name__ == "__main__":
    main()
