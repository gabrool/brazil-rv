from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .context_ablation import (
    STAGE1_CONTEXT_ABLATION_ORDER,
    get_context_ablation,
    resolve_context_ablation_for_store,
)
from .contract import (
    FEATURE_CONTRACT_VERSION,
    GLOBAL_CONTEXT_SYMBOLS,
    LOCAL_CONTEXT_SYMBOLS,
    PROJECT_ROOT,
    RUN_OUTPUT_BASE,
    SplitBoundaries,
    TCNSettings,
    architecture_for_model,
    expected_trainable_parameter_count,
)
from .data import resolve_feature_store, validate_feature_store
from .engine import objective_metadata, sam_metadata
from .process_lock import (
    PRODUCTION_TRAINING_LOCK,
    active_lock_owner,
    exclusive_process_lock,
)

SWEEP_NAME = "stage1_context_ablation_seed29"
STATE_VERSION = 1
DEFAULT_STATE_DIR = RUN_OUTPUT_BASE / SWEEP_NAME
STAGE1_TCN_SETTINGS = TCNSettings("context_pooled", 64, "full", "swiglu")
STAGE1_SEED = 29
STAGE1_TEMPERATURE = 0.50
STAGE1_SAM_RHO = 0.125
_REPOSITORY = PROJECT_ROOT / "quant" / "b3-quant"
_RESEARCH = _REPOSITORY / "research"


def build_stage1_command(key: str) -> tuple[str, ...]:
    get_context_ablation(key)
    return (
        sys.executable,
        "-m",
        "brazil_rv.modeling.train",
        "--model",
        "tcn",
        "--tcn-fusion",
        "context_pooled",
        "--tcn-width",
        "64",
        "--tcn-receptive-field",
        "full",
        "--tcn-block",
        "swiglu",
        "--optimizer",
        "sam_adamw",
        "--objective",
        "soft_spearman",
        "--soft-rank-temperature",
        "0.50",
        "--sam-rho",
        "0.125",
        "--global-context",
        "enabled",
        "--seed",
        "29",
        "--context-ablation",
        key,
    )


def stage1_jobs() -> tuple[dict[str, object], ...]:
    jobs = tuple(
        {
            "context_ablation": key,
            "seed": STAGE1_SEED,
            "command": list(build_stage1_command(key)),
        }
        for key in STAGE1_CONTEXT_ABLATION_ORDER
    )
    if len(jobs) != 25 or len({job["context_ablation"] for job in jobs}) != 25:
        raise RuntimeError("Stage-1 matrix must contain 25 unique ablations")
    if {job["seed"] for job in jobs} != {STAGE1_SEED}:
        raise RuntimeError("Stage-1 matrix must use only seed 29")
    return jobs


def _git_identity(*, require_clean: bool) -> tuple[str, bool]:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=_REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    clean = not bool(status)
    if require_clean and not clean:
        raise RuntimeError("Stage-1 execution requires a clean Git worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, clean


def _configuration(commit: str, feature_store: Path) -> dict[str, object]:
    architecture = architecture_for_model("tcn", STAGE1_TCN_SETTINGS)
    return {
        "git_commit_sha": commit,
        "resolved_feature_store_path": str(feature_store),
        "feature_contract": FEATURE_CONTRACT_VERSION,
        "local_context_symbols": list(LOCAL_CONTEXT_SYMBOLS),
        "global_context_symbols": list(GLOBAL_CONTEXT_SYMBOLS),
        "model_name": "tcn",
        "tcn_settings": asdict(STAGE1_TCN_SETTINGS),
        "parameter_count": expected_trainable_parameter_count("tcn", architecture),
        "optimizer_variant": "sam_adamw",
        "objective": objective_metadata("soft_spearman", STAGE1_TEMPERATURE),
        "sam": sam_metadata("sam_adamw", STAGE1_SAM_RHO),
        "global_context": "enabled",
        "seed": STAGE1_SEED,
        "split_boundaries": {
            key: str(value) for key, value in asdict(SplitBoundaries()).items()
        },
        "ablation_order": list(STAGE1_CONTEXT_ABLATION_ORDER),
    }


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _new_state(configuration: dict[str, object]) -> dict[str, object]:
    return {
        "state_version": STATE_VERSION,
        "sweep_name": SWEEP_NAME,
        "status": "running",
        "configuration": configuration,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_at_utc": None,
        "jobs": [
            {
                **job,
                "status": "pending",
                "run_dir": None,
                "started_at_utc": None,
                "completed_at_utc": None,
                "primary_validation_ic": None,
                "error": None,
            }
            for job in stage1_jobs()
        ],
    }


def _load_state(path: Path, configuration: dict[str, object]) -> dict[str, object]:
    if not path.exists():
        return _new_state(configuration)
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("state_version") != STATE_VERSION:
        raise ValueError("Stage-1 state version is incompatible")
    if state.get("sweep_name") != SWEEP_NAME:
        raise ValueError("Stage-1 state belongs to another sweep")
    if state.get("configuration") != configuration:
        raise ValueError("Stage-1 state configuration does not match this invocation")
    jobs = state.get("jobs")
    if (
        not isinstance(jobs, list)
        or tuple(job.get("context_ablation") for job in jobs if isinstance(job, dict))
        != STAGE1_CONTEXT_ABLATION_ORDER
    ):
        raise ValueError("Stage-1 state does not contain the canonical 25-job order")
    return state


def validate_completed_run(
    run_dir: Path,
    configuration: dict[str, object],
    key: str,
) -> float:
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "status": "completed",
        "model_name": configuration["model_name"],
        "model_family": "tcn",
        "tcn_settings": configuration["tcn_settings"],
        "parameter_count": configuration["parameter_count"],
        "optimizer_variant": configuration["optimizer_variant"],
        "objective": configuration["objective"],
        "sam": configuration["sam"],
        "global_context": configuration["global_context"],
        "seed": configuration["seed"],
        "git_commit_sha": configuration["git_commit_sha"],
        "resolved_feature_store_path": configuration["resolved_feature_store_path"],
        "feature_manifest_contract_version": configuration["feature_contract"],
        "split_boundaries": configuration["split_boundaries"],
        "context_ablation": get_context_ablation(key).metadata(),
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"Completed run {run_dir} has incompatible manifest field: {field}"
            )
    if not isinstance(manifest.get("architecture_constants"), dict):
        raise ValueError(
            f"Completed run {run_dir} is missing TCN architecture metadata"
        )
    if manifest["architecture_constants"].get("width") != 64:
        raise ValueError(f"Completed run {run_dir} has the wrong TCN width")
    metrics_path = run_dir / "validation_metrics.json"
    daily_path = run_dir / "validation_daily_metrics.parquet"
    for artifact in (
        metrics_path,
        daily_path,
        run_dir / "best.pt",
        run_dir / "history.csv",
    ):
        if not artifact.is_file():
            raise ValueError(f"Completed run is missing artifact: {artifact}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    primary = float(metrics["primary_score"])
    if not math.isfinite(primary):
        raise ValueError(f"Completed run {run_dir} has nonfinite primary IC")
    if not math.isclose(
        primary,
        float(manifest["best_validation_primary_score"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"Completed run {run_dir} has inconsistent primary IC")
    if int(manifest.get("best_epoch", 0)) <= 0:
        raise ValueError(f"Completed run {run_dir} has no valid best epoch")
    return primary


def _candidate_run_dirs(configuration: dict[str, object], key: str) -> tuple[Path, ...]:
    candidates: list[Path] = []
    if not RUN_OUTPUT_BASE.is_dir():
        return ()
    for run_dir in RUN_OUTPUT_BASE.iterdir():
        manifest_path = run_dir / "run_manifest.json"
        if not run_dir.is_dir() or not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        ablation = manifest.get("context_ablation")
        if (
            manifest.get("git_commit_sha") == configuration["git_commit_sha"]
            and manifest.get("resolved_feature_store_path")
            == configuration["resolved_feature_store_path"]
            and manifest.get("seed") == STAGE1_SEED
            and isinstance(ablation, dict)
            and ablation.get("key") == key
        ):
            candidates.append(run_dir)
    return tuple(sorted(candidates))


def _adopt_completed_run(
    job: dict[str, object], configuration: dict[str, object]
) -> bool:
    key = str(job["context_ablation"])
    recorded = job.get("run_dir")
    candidates = (
        (Path(str(recorded)),) if recorded else _candidate_run_dirs(configuration, key)
    )
    completed: list[tuple[Path, float]] = []
    for run_dir in candidates:
        try:
            score = validate_completed_run(run_dir, configuration, key)
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            if recorded and job.get("status") == "completed":
                raise
            if recorded:
                return False
            continue
        completed.append((run_dir, score))
    if len(completed) > 1:
        raise ValueError(f"Multiple completed runs match Stage-1 ablation {key}")
    if not completed:
        return False
    run_dir, score = completed[0]
    job.update(
        {
            "status": "completed",
            "run_dir": str(run_dir),
            "completed_at_utc": job.get("completed_at_utc")
            or datetime.now(timezone.utc).isoformat(),
            "primary_validation_ic": score,
            "error": None,
        }
    )
    return True


def _prepare(*, require_clean: bool) -> tuple[str, bool, Path, dict[str, object]]:
    commit, clean = _git_identity(require_clean=require_clean)
    feature_store = resolve_feature_store().resolve()
    validate_feature_store(feature_store)
    for key in STAGE1_CONTEXT_ABLATION_ORDER:
        resolve_context_ablation_for_store(feature_store, key)
    return commit, clean, feature_store, _configuration(commit, feature_store)


def _production_run_directories() -> set[Path]:
    if not RUN_OUTPUT_BASE.is_dir():
        return set()
    return {
        path.resolve()
        for path in RUN_OUTPUT_BASE.iterdir()
        if path.is_dir() and path.name != "_ops"
    }


def dry_run_payload() -> dict[str, object]:
    commit, clean, feature_store, configuration = _prepare(require_clean=False)
    return {
        "sweep_name": SWEEP_NAME,
        "dry_run": True,
        "worktree_clean": clean,
        "git_commit_sha": commit,
        "resolved_feature_store_path": str(feature_store),
        "configuration": configuration,
        "job_count": 25,
        "jobs": list(stage1_jobs()),
    }


def run_sweep(state_dir: Path) -> Path:
    commit, _, feature_store, configuration = _prepare(require_clean=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.json"
    with exclusive_process_lock(state_dir / "sweep.lock", SWEEP_NAME):
        if owner := active_lock_owner(PRODUCTION_TRAINING_LOCK):
            raise RuntimeError(f"Another production training run is active: {owner}")
        state = _load_state(state_path, configuration)
        _atomic_write_json(state_path, state)
        jobs = state["jobs"]
        for position, job in enumerate(jobs, start=1):
            if _adopt_completed_run(job, configuration):
                _atomic_write_json(state_path, state)
                print(
                    f"[{position}/25] verified {job['context_ablation']}: "
                    f"{job['run_dir']}",
                    flush=True,
                )
                continue
            current_commit, _ = _git_identity(require_clean=True)
            if (
                current_commit != commit
                or resolve_feature_store().resolve() != feature_store
            ):
                raise RuntimeError(
                    "Git commit or canonical feature store changed mid-sweep"
                )
            if owner := active_lock_owner(PRODUCTION_TRAINING_LOCK):
                raise RuntimeError(
                    f"Another production training run is active: {owner}"
                )
            before = _production_run_directories()
            job.update(
                {
                    "status": "running",
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "completed_at_utc": None,
                    "primary_validation_ic": None,
                    "error": None,
                }
            )
            _atomic_write_json(state_path, state)
            print(f"[{position}/25] starting {job['context_ablation']}", flush=True)
            try:
                result = subprocess.run(job["command"], cwd=_RESEARCH, check=False)
            except OSError as error:
                job.update({"status": "failed", "error": str(error)})
                _atomic_write_json(state_path, state)
                raise RuntimeError(
                    f"Could not start Stage-1 training for {job['context_ablation']}"
                ) from error
            created = tuple(
                sorted(
                    path for path in _production_run_directories() if path not in before
                )
            )
            if len(created) == 1:
                job["run_dir"] = str(created[0])
            if result.returncode != 0:
                job.update(
                    {
                        "status": "failed",
                        "error": f"training exited with code {result.returncode}",
                    }
                )
                _atomic_write_json(state_path, state)
                raise RuntimeError(
                    f"Stage-1 training failed for {job['context_ablation']}"
                )
            if len(created) != 1:
                job.update(
                    {
                        "status": "failed",
                        "error": f"expected one new run directory, found {len(created)}",
                    }
                )
                _atomic_write_json(state_path, state)
                raise RuntimeError(str(job["error"]))
            try:
                score = validate_completed_run(
                    created[0], configuration, str(job["context_ablation"])
                )
            except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
                job.update(
                    {
                        "status": "failed",
                        "error": f"completed-run validation failed: {error}",
                    }
                )
                _atomic_write_json(state_path, state)
                raise RuntimeError(
                    "Stage-1 completed-run validation failed for "
                    f"{job['context_ablation']}"
                ) from error
            job.update(
                {
                    "status": "completed",
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "primary_validation_ic": score,
                    "error": None,
                }
            )
            _atomic_write_json(state_path, state)
            print(
                f"[{position}/25] completed {job['context_ablation']} IC={score:.8f}",
                flush=True,
            )
        completed_keys = {
            str(job["context_ablation"])
            for job in jobs
            if job.get("status") == "completed"
            and _adopt_completed_run(job, configuration)
        }
        if completed_keys != set(STAGE1_CONTEXT_ABLATION_ORDER):
            raise RuntimeError("Stage-1 did not finish with exactly 25 verified runs")
        state["status"] = "completed"
        state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(state_path, state)
    return state_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(json.dumps(dry_run_payload(), indent=2, allow_nan=False), flush=True)
        return
    state_path = run_sweep(args.state_dir.resolve())
    print(f"Stage-1 sweep completed: {state_path}", flush=True)


if __name__ == "__main__":
    main()
