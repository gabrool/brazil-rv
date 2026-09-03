from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .artifacts import write_json_atomic
from .contract import ALLOWED_SEEDS

MAX_PARALLEL_TRAJECTORIES = 6


@dataclass(frozen=True)
class TrajectoryJob:
    name: str
    seed: int
    fold: str
    run_dir: Path
    command: tuple[str, ...]
    cwd: Path | None = None
    expected_manifest: Mapping[str, object] | None = None

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "run_manifest.json"

    @property
    def stdout_path(self) -> Path:
        return self.run_dir / "launcher.stdout.log"

    @property
    def stderr_path(self) -> Path:
        return self.run_dir / "launcher.stderr.log"


@dataclass(frozen=True)
class TrajectoryOutcome:
    name: str
    status: str
    seed: int
    fold: str
    run_dir: Path
    manifest_sha256: str

    def payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "seed": self.seed,
            "fold": self.fold,
            "run_dir": str(self.run_dir),
            "manifest_sha256": self.manifest_sha256,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON manifest must contain an object: {path}")
    return payload


def _contains(actual: object, expected: object) -> bool:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return False
        return all(
            key in actual and _contains(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, (list, tuple)):
        return (
            isinstance(actual, (list, tuple))
            and len(actual) == len(expected)
            and all(
                _contains(actual_value, expected_value)
                for actual_value, expected_value in zip(actual, expected, strict=True)
            )
        )
    return actual == expected


def _expected(job: TrajectoryJob) -> dict[str, object]:
    expected = dict(job.expected_manifest or {})
    expected.update({"seed": job.seed, "fold": job.fold})
    return expected


def _completed_outcome(job: TrajectoryJob, *, status: str) -> TrajectoryOutcome:
    manifest = _read_json(job.manifest_path)
    if manifest.get("status") != "completed":
        raise ValueError(f"trajectory did not complete: {job.name}")
    expected = _expected(job)
    if not _contains(manifest, expected):
        raise ValueError(f"trajectory manifest differs from launch plan: {job.name}")
    return TrajectoryOutcome(
        name=job.name,
        status=status,
        seed=job.seed,
        fold=job.fold,
        run_dir=job.run_dir,
        manifest_sha256=_sha256(job.manifest_path),
    )


def _preflight(
    jobs: Sequence[TrajectoryJob], max_parallel: int
) -> tuple[list[TrajectoryOutcome], list[TrajectoryJob]]:
    if not 1 <= max_parallel <= MAX_PARALLEL_TRAJECTORIES:
        raise ValueError("max_parallel must be between one and six")
    if not jobs:
        raise ValueError("at least one trajectory is required")
    if len({job.name for job in jobs}) != len(jobs):
        raise ValueError("trajectory names must be unique")
    resolved_dirs = [job.run_dir.resolve() for job in jobs]
    if len(set(resolved_dirs)) != len(resolved_dirs):
        raise ValueError("trajectory run directories must be unique")
    if any(
        left != right and left.is_relative_to(right)
        for left in resolved_dirs
        for right in resolved_dirs
    ):
        raise ValueError("trajectory run directories must not contain each other")
    skipped: list[TrajectoryOutcome] = []
    pending: list[TrajectoryJob] = []
    for job in jobs:
        if not job.name or not job.fold or not job.command or not job.command[0]:
            raise ValueError("trajectory name, fold, and command must be nonempty")
        if job.seed not in ALLOWED_SEEDS:
            raise ValueError("trajectory seed differs from the frozen v2 roster")
        if job.run_dir.exists():
            if not job.run_dir.is_dir() or not job.manifest_path.is_file():
                raise FileExistsError(
                    f"existing trajectory root is incomplete and will not be reused: "
                    f"{job.run_dir}"
                )
            skipped.append(_completed_outcome(job, status="skipped_completed"))
        else:
            pending.append(job)
    return skipped, pending


def _run_one(job: TrajectoryJob) -> TrajectoryOutcome:
    job.run_dir.mkdir(parents=True, exist_ok=False)
    with job.stdout_path.open("xb") as stdout, job.stderr_path.open("xb") as stderr:
        completed = subprocess.run(
            list(job.command),
            cwd=job.cwd,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            check=False,
            shell=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"trajectory {job.name} exited with status {completed.returncode}"
        )
    return _completed_outcome(job, status="completed")


def run_many(
    jobs: Sequence[TrajectoryJob],
    *,
    max_parallel: int,
    launcher_manifest_path: Path | None = None,
) -> tuple[TrajectoryOutcome, ...]:
    """Run isolated child processes once, with immutable-root resume semantics."""
    job_list = tuple(jobs)
    skipped, pending = _preflight(job_list, max_parallel)
    completed: list[TrajectoryOutcome] = []
    failures: dict[str, BaseException] = {}
    if pending:
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {executor.submit(_run_one, job): job for job in pending}
            done, not_done = wait(futures, return_when=FIRST_EXCEPTION)
            if any(future.exception() is not None for future in done):
                for future in not_done:
                    future.cancel()
        by_name = {job.name: future for future, job in futures.items()}
        for job in pending:
            future = by_name[job.name]
            if future.cancelled():
                continue
            try:
                completed.append(future.result())
            except BaseException as error:
                failures[job.name] = error
    failed_job = next((job.name for job in job_list if job.name in failures), None)
    outcomes_by_name = {item.name: item for item in skipped + completed}
    ordered = tuple(
        outcomes_by_name[job.name] for job in job_list if job.name in outcomes_by_name
    )
    if launcher_manifest_path is not None:
        write_json_atomic(
            launcher_manifest_path,
            {
                "schema": "BRAZIL_RV_V2_RUN_MANY_V1",
                "status": "failed" if failed_job is not None else "completed",
                "max_parallel": max_parallel,
                "planned_jobs": [
                    {
                        "name": job.name,
                        "seed": job.seed,
                        "fold": job.fold,
                        "run_dir": str(job.run_dir),
                        "expected_manifest": _expected(job),
                    }
                    for job in job_list
                ],
                "jobs": [item.payload() for item in ordered],
                "failed_job": failed_job,
            },
        )
    if failed_job is not None:
        raise failures[failed_job]
    return ordered


def load_plan(path: Path) -> tuple[tuple[TrajectoryJob, ...], int]:
    payload = _read_json(path)
    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list):
        raise ValueError("run-many plan must contain a jobs list")
    jobs: list[TrajectoryJob] = []
    for raw_job in raw_jobs:
        if not isinstance(raw_job, dict):
            raise ValueError("run-many job must be an object")
        command = raw_job.get("command")
        if not isinstance(command, list) or not all(
            isinstance(value, str) for value in command
        ):
            raise ValueError("run-many command must be a string list")
        expected = raw_job.get("expected_manifest")
        if expected is not None and not isinstance(expected, dict):
            raise ValueError("expected_manifest must be an object")
        jobs.append(
            TrajectoryJob(
                name=str(raw_job["name"]),
                seed=int(raw_job["seed"]),
                fold=str(raw_job["fold"]),
                run_dir=Path(str(raw_job["run_dir"])),
                command=tuple(command),
                cwd=(None if raw_job.get("cwd") is None else Path(str(raw_job["cwd"]))),
                expected_manifest=expected,
            )
        )
    return tuple(jobs), int(payload.get("max_parallel", 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v2 trajectories concurrently")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    arguments = parser.parse_args()
    jobs, max_parallel = load_plan(arguments.plan)
    run_many(
        jobs,
        max_parallel=max_parallel,
        launcher_manifest_path=arguments.manifest,
    )


if __name__ == "__main__":
    main()
