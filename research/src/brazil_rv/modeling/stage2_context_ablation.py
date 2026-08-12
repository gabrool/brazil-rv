from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from .context_ablation import (
    STAGE1_CONTEXT_ABLATION_ORDER,
    get_context_ablation,
    resolve_context_ablation_for_store,
)
from .contract import (
    AdamWConstants,
    EFFECTIVE_BATCH_SIZE,
    FEATURE_CONTRACT_VERSION,
    GH200_RUNTIME,
    GLOBAL_CONTEXT_SYMBOLS,
    HORIZONS,
    LOCAL_CONTEXT_SYMBOLS,
    PROJECT_ROOT,
    RUN_OUTPUT_BASE,
    SchedulerConstants,
    SplitBoundaries,
    TCNSettings,
    TrainingConstants,
    VALIDATION_END,
    VALIDATION_START,
    architecture_for_model,
    expected_trainable_parameter_count,
    peer_feature_metadata,
)
from .data import resolve_feature_store, validate_feature_store
from .engine import objective_metadata, sam_metadata
from .process_lock import (
    PRODUCTION_TRAINING_LOCK,
    active_lock_owner,
    exclusive_process_lock,
)

SWEEP_NAME = "stage2_context_ablation_matched_seeds"
STATE_VERSION = 1
STAGE1_PRODUCING_COMMIT = "169c9b2979858d14fc6a3dd12123b0cf5a9e7576"
STAGE2_CONTEXT_ABLATION_ORDER = (
    "none",
    "drop_fixed_di",
    "drop_all_di",
    "drop_di1n",
    "drop_all_global",
    "drop_global_non_rates",
)
STAGE2_SEEDS = (11, 29, 47)
STAGE2_TCN_SETTINGS = TCNSettings("context_pooled", 64, "full", "swiglu")
STAGE2_TEMPERATURE = 0.50
STAGE2_SAM_RHO = 0.125
STAGE2_UPDATES_PER_EPOCH = 77
ADOPTED_STAGE1_KEYS = STAGE2_CONTEXT_ABLATION_ORDER[:-1]
_REPOSITORY = PROJECT_ROOT / "quant" / "b3-quant"
_RESEARCH = _REPOSITORY / "research"
_VALIDATION_DAILY_FILENAME = "validation_daily_metrics.parquet"


def _json_value(value: object) -> object:
    return json.loads(json.dumps(value))


def build_stage2_command(key: str, seed: int) -> tuple[str, ...]:
    get_context_ablation(key)
    if key not in STAGE2_CONTEXT_ABLATION_ORDER or seed not in STAGE2_SEEDS:
        raise ValueError("Stage-2 job is outside the canonical matrix")
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
        "--peer-features",
        "none",
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
        str(seed),
        "--context-ablation",
        key,
    )


def stage2_jobs() -> tuple[dict[str, object], ...]:
    jobs = tuple(
        {
            "context_ablation": key,
            "seed": seed,
            "command": list(build_stage2_command(key, seed)),
        }
        for key in STAGE2_CONTEXT_ABLATION_ORDER
        for seed in STAGE2_SEEDS
    )
    if (
        len(jobs) != 18
        or len({(job["context_ablation"], job["seed"]) for job in jobs}) != 18
    ):
        raise RuntimeError("Stage-2 matrix must contain 18 unique jobs")
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
        raise RuntimeError("Stage-2 execution requires a clean Git worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, clean


def _feature_store_identity(store: Path) -> dict[str, object]:
    manifest_path = store / "manifest.json"
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("contract_version") != FEATURE_CONTRACT_VERSION:
        raise ValueError("Feature manifest has the wrong contract version")
    global_context = manifest.get("global_context")
    if not isinstance(global_context, dict):
        raise ValueError("Feature manifest is missing global-context identity")
    return {
        "resolved_path": str(store),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "contract_version": manifest["contract_version"],
        "global_context_source_hashes": global_context.get("source_hashes"),
        "global_context_normalized_store_hashes": global_context.get(
            "normalized_store_hashes"
        ),
        "canonical_inputs": manifest.get("canonical_inputs"),
    }


def feature_stores_equivalent(left: Path, right: Path) -> bool:
    try:
        if left.samefile(right):
            return True
    except OSError:
        pass
    try:
        return (
            _feature_store_identity(left)["manifest_sha256"]
            == _feature_store_identity(right)["manifest_sha256"]
        )
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        return False


def _training_semantics() -> dict[str, object]:
    architecture = architecture_for_model("tcn", STAGE2_TCN_SETTINGS)
    return {
        "model_name": "tcn",
        "model_family": "tcn",
        "tcn_settings": asdict(STAGE2_TCN_SETTINGS),
        "architecture_constants": _json_value(asdict(architecture)),
        "parameter_count": expected_trainable_parameter_count("tcn", architecture),
        "peer_features": peer_feature_metadata("tcn", architecture, "none"),
        "optimizer_variant": "sam_adamw",
        "objective": objective_metadata("soft_spearman", STAGE2_TEMPERATURE),
        "sam": sam_metadata("sam_adamw", STAGE2_SAM_RHO),
        "global_context": "enabled",
        "training_constants": _json_value(asdict(TrainingConstants())),
        "optimizer_constants": {"adamw": _json_value(asdict(AdamWConstants()))},
        "scheduler_constants": _json_value(asdict(SchedulerConstants())),
        "physical_microbatch_size": GH200_RUNTIME.microbatch_size,
        "accumulation_steps": GH200_RUNTIME.accumulation_steps,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "evaluation_batch_size": GH200_RUNTIME.evaluation_batch_size,
        "num_workers": GH200_RUNTIME.num_workers,
        "prefetch_factor": GH200_RUNTIME.prefetch_factor,
        "precision": "bf16",
        "bf16": True,
        "grad_scaler_used": False,
    }


def _configuration(
    commit: str, feature_store: Path, stage1_state_path: Path
) -> dict[str, object]:
    return {
        "orchestrator_git_commit_sha": commit,
        "feature_store": _feature_store_identity(feature_store),
        "feature_contract": FEATURE_CONTRACT_VERSION,
        "local_context_symbols": list(LOCAL_CONTEXT_SYMBOLS),
        "global_context_symbols": list(GLOBAL_CONTEXT_SYMBOLS),
        "training_semantics": _training_semantics(),
        "split_boundaries": {
            key: str(value) for key, value in asdict(SplitBoundaries()).items()
        },
        "configuration_order": list(STAGE2_CONTEXT_ABLATION_ORDER),
        "seeds": list(STAGE2_SEEDS),
        "source_stage1_state": str(stage1_state_path),
        "required_stage1_producing_commit": STAGE1_PRODUCING_COMMIT,
    }


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_validation_artifacts(run_dir: Path) -> float:
    metrics_path = run_dir / "validation_metrics.json"
    daily_path = run_dir / _VALIDATION_DAILY_FILENAME
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
    daily = pl.read_parquet(daily_path)
    required = {
        "trade_date",
        "date_idx",
        "horizon_minutes",
        "spearman_ic",
        "top_minus_bottom",
        "one_way_turnover",
    }
    if not required <= set(daily.columns):
        raise ValueError(f"Validation daily metrics are incomplete: {run_dir}")
    if daily.height != 244 * len(HORIZONS):
        raise ValueError(
            f"Validation daily metrics have the wrong row count: {run_dir}"
        )
    dates = daily.get_column("trade_date")
    if (
        dates.n_unique() != 244
        or dates.min() != VALIDATION_START
        or dates.max() != VALIDATION_END
    ):
        raise ValueError(f"Validation daily metrics have the wrong dates: {run_dir}")
    return primary


def validate_stage2_completed_run(
    run_dir: Path,
    configuration: dict[str, object],
    key: str,
    seed: int,
    producing_commit: str,
) -> float:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    semantics = configuration["training_semantics"]
    if not isinstance(semantics, dict):
        raise ValueError("Stage-2 configuration has invalid training semantics")
    expected = {
        **semantics,
        "status": "completed",
        "seed": seed,
        "git_commit_sha": producing_commit,
        "feature_manifest_contract_version": configuration["feature_contract"],
        "split_boundaries": configuration["split_boundaries"],
        "context_ablation": get_context_ablation(key).metadata(),
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"Completed run {run_dir} has incompatible manifest field: {field}"
            )
    feature_identity = configuration["feature_store"]
    if not isinstance(feature_identity, dict):
        raise ValueError("Stage-2 configuration has invalid feature-store identity")
    manifest_store = Path(str(manifest.get("resolved_feature_store_path")))
    configured_store = Path(str(feature_identity["resolved_path"]))
    if not feature_stores_equivalent(manifest_store, configured_store):
        raise ValueError(f"Completed run {run_dir} uses a different feature store")
    if (
        manifest.get("global_context_source_hashes")
        != feature_identity["global_context_source_hashes"]
        or manifest.get("global_context_normalized_store_hashes")
        != feature_identity["global_context_normalized_store_hashes"]
        or manifest.get("resolved_source_paths") != feature_identity["canonical_inputs"]
    ):
        raise ValueError(f"Completed run {run_dir} has inconsistent feature identity")
    primary = _validate_validation_artifacts(run_dir)
    if not math.isclose(
        primary,
        float(manifest["best_validation_primary_score"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"Completed run {run_dir} has inconsistent primary IC")
    for field in ("best_epoch", "stopped_epoch", "successful_optimizer_updates"):
        if int(manifest.get(field, 0)) <= 0:
            raise ValueError(f"Completed run {run_dir} has invalid {field}")
    duration = float(manifest.get("training_duration_seconds", 0.0))
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"Completed run {run_dir} has invalid training duration")
    scheduler_steps = manifest.get("scheduler_steps")
    if (
        not isinstance(scheduler_steps, dict)
        or scheduler_steps.get("steps_per_epoch") != STAGE2_UPDATES_PER_EPOCH
    ):
        raise ValueError(f"Completed run {run_dir} has wrong scheduler steps")
    return primary


def _validated_stage1_adoptions(
    state_path: Path, configuration: dict[str, object]
) -> dict[str, tuple[dict[str, object], Path, float]]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("state_version") != 1 or state.get("sweep_name") != (
        "stage1_context_ablation_seed29"
    ):
        raise ValueError("Stage-1 state has an incompatible identity")
    if state.get("status") != "completed":
        raise ValueError("Stage-1 adoption requires a completed state")
    source_configuration = state.get("configuration")
    jobs = state.get("jobs")
    if not isinstance(source_configuration, dict) or not isinstance(jobs, list):
        raise ValueError("Stage-1 state is missing configuration or jobs")
    semantics = configuration["training_semantics"]
    if not isinstance(semantics, dict):
        raise ValueError("Stage-2 configuration has invalid training semantics")
    expected_configuration = {
        "git_commit_sha": STAGE1_PRODUCING_COMMIT,
        "feature_contract": configuration["feature_contract"],
        "local_context_symbols": configuration["local_context_symbols"],
        "global_context_symbols": configuration["global_context_symbols"],
        "model_name": semantics["model_name"],
        "tcn_settings": semantics["tcn_settings"],
        "parameter_count": semantics["parameter_count"],
        "peer_features": semantics["peer_features"],
        "optimizer_variant": semantics["optimizer_variant"],
        "objective": semantics["objective"],
        "sam": semantics["sam"],
        "global_context": semantics["global_context"],
        "seed": 29,
        "split_boundaries": configuration["split_boundaries"],
        "ablation_order": list(STAGE1_CONTEXT_ABLATION_ORDER),
    }
    for field, value in expected_configuration.items():
        if source_configuration.get(field) != value:
            raise ValueError(f"Stage-1 state has incompatible configuration: {field}")
    source_store = Path(str(source_configuration.get("resolved_feature_store_path")))
    current_identity = configuration["feature_store"]
    if not isinstance(current_identity, dict):
        raise ValueError("Stage-2 configuration has invalid feature-store identity")
    current_store = Path(str(current_identity["resolved_path"]))
    if not feature_stores_equivalent(source_store, current_store):
        raise ValueError("Stage-1 state identifies a different feature store")
    identities = tuple(
        (job.get("context_ablation"), job.get("seed"))
        for job in jobs
        if isinstance(job, dict)
    )
    if len(jobs) != 25 or identities != tuple(
        (key, 29) for key in STAGE1_CONTEXT_ABLATION_ORDER
    ):
        raise ValueError("Stage-1 state does not contain the canonical 25 jobs")
    if any(job.get("status") != "completed" for job in jobs):
        raise ValueError("Stage-1 state contains an incomplete job")
    run_dirs = tuple(Path(str(job.get("run_dir"))).resolve() for job in jobs)
    if len(set(run_dirs)) != 25:
        raise ValueError("Stage-1 state contains duplicate run directories")

    by_key = {str(job["context_ablation"]): job for job in jobs}
    adopted: dict[str, tuple[dict[str, object], Path, float]] = {}
    for key in ADOPTED_STAGE1_KEYS:
        job = by_key[key]
        run_dir = Path(str(job["run_dir"])).resolve()
        score = validate_stage2_completed_run(
            run_dir, configuration, key, 29, STAGE1_PRODUCING_COMMIT
        )
        recorded_score = float(job.get("primary_validation_ic"))
        if not math.isclose(score, recorded_score, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Stage-1 state score disagrees for {key}")
        adopted[key] = (job, run_dir, score)
    return adopted


def _new_state(
    configuration: dict[str, object], stage1_state_path: Path
) -> dict[str, object]:
    adopted = _validated_stage1_adoptions(stage1_state_path, configuration)
    now = datetime.now(timezone.utc).isoformat()
    jobs: list[dict[str, object]] = []
    for position, base in enumerate(stage2_jobs()):
        key = str(base["context_ablation"])
        seed = int(base["seed"])
        job = {
            **base,
            "status": "pending",
            "result_origin": None,
            "run_dir": None,
            "producing_git_commit_sha": None,
            "source_stage1_state": None,
            "source_stage1_job": None,
            "started_at_utc": None,
            "completed_at_utc": None,
            "primary_validation_ic": None,
            "error": None,
        }
        if seed == 29 and key in adopted:
            source_job, run_dir, score = adopted[key]
            job.update(
                {
                    "status": "completed",
                    "result_origin": "adopted_stage1",
                    "run_dir": str(run_dir),
                    "producing_git_commit_sha": STAGE1_PRODUCING_COMMIT,
                    "source_stage1_state": str(stage1_state_path),
                    "source_stage1_job": {
                        "position": STAGE1_CONTEXT_ABLATION_ORDER.index(key),
                        "context_ablation": key,
                        "seed": 29,
                        "run_dir": str(source_job["run_dir"]),
                    },
                    "completed_at_utc": now,
                    "primary_validation_ic": score,
                }
            )
        jobs.append(job)
    if sum(job["result_origin"] == "adopted_stage1" for job in jobs) != 5:
        raise RuntimeError("Fresh Stage-2 state must adopt exactly five jobs")
    return {
        "state_version": STATE_VERSION,
        "sweep_name": SWEEP_NAME,
        "status": "running",
        "configuration": configuration,
        "created_at_utc": now,
        "completed_at_utc": None,
        "jobs": jobs,
    }


def _configurations_match(
    stored: dict[str, object], current: dict[str, object]
) -> bool:
    stored_copy = dict(stored)
    current_copy = dict(current)
    stored_feature = dict(stored_copy.pop("feature_store", {}))
    current_feature = dict(current_copy.pop("feature_store", {}))
    stored_path = Path(str(stored_feature.pop("resolved_path", "")))
    current_path = Path(str(current_feature.pop("resolved_path", "")))
    return (
        stored_copy == current_copy
        and stored_feature == current_feature
        and feature_stores_equivalent(stored_path, current_path)
    )


def _load_state(
    path: Path,
    configuration: dict[str, object],
    stage1_state_path: Path,
) -> dict[str, object]:
    if not path.exists():
        return _new_state(configuration, stage1_state_path)
    state = json.loads(path.read_text(encoding="utf-8"))
    if (
        state.get("state_version") != STATE_VERSION
        or state.get("sweep_name") != SWEEP_NAME
    ):
        raise ValueError("Stage-2 state has an incompatible identity")
    stored_configuration = state.get("configuration")
    if not isinstance(stored_configuration, dict) or not _configurations_match(
        stored_configuration, configuration
    ):
        raise ValueError("Stage-2 state configuration does not match this invocation")
    jobs = state.get("jobs")
    expected = tuple((job["context_ablation"], job["seed"]) for job in stage2_jobs())
    actual = (
        tuple(
            (job.get("context_ablation"), job.get("seed"))
            for job in jobs
            if isinstance(job, dict)
        )
        if isinstance(jobs, list)
        else ()
    )
    if actual != expected:
        raise ValueError("Stage-2 state does not contain the canonical 18 jobs")
    adopted = _validated_stage1_adoptions(stage1_state_path, configuration)
    by_identity = {
        (str(job["context_ablation"]), int(job["seed"])): job for job in jobs
    }
    for key, (source_job, run_dir, score) in adopted.items():
        job = by_identity[(key, 29)]
        source = job.get("source_stage1_job")
        if (
            job.get("status") != "completed"
            or job.get("result_origin") != "adopted_stage1"
            or Path(str(job.get("run_dir"))).resolve() != run_dir
            or job.get("producing_git_commit_sha") != STAGE1_PRODUCING_COMMIT
            or job.get("source_stage1_state") != str(stage1_state_path)
            or not isinstance(source, dict)
            or source.get("context_ablation") != key
            or source.get("seed") != 29
            or Path(str(source.get("run_dir"))).resolve()
            != Path(str(source_job["run_dir"])).resolve()
            or not math.isclose(
                float(job.get("primary_validation_ic")),
                score,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"Stage-2 adopted provenance disagrees for {key}/29")
    return state


def _prepare(
    *, require_clean: bool, stage1_state_path: Path
) -> tuple[str, bool, Path, dict[str, object]]:
    commit, clean = _git_identity(require_clean=require_clean)
    feature_store = resolve_feature_store().resolve()
    validate_feature_store(feature_store)
    for key in STAGE2_CONTEXT_ABLATION_ORDER:
        resolve_context_ablation_for_store(feature_store, key)
    configuration = _configuration(commit, feature_store, stage1_state_path)
    return commit, clean, feature_store, configuration


def _production_run_directories() -> set[Path]:
    if not RUN_OUTPUT_BASE.is_dir():
        return set()
    return {
        path.resolve()
        for path in RUN_OUTPUT_BASE.iterdir()
        if path.is_dir() and path.name != "_ops"
    }


def _candidate_run_dirs(
    configuration: dict[str, object], key: str, seed: int
) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for run_dir in _production_run_directories():
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        ablation = manifest.get("context_ablation")
        if (
            manifest.get("git_commit_sha")
            == configuration["orchestrator_git_commit_sha"]
            and manifest.get("seed") == seed
            and isinstance(ablation, dict)
            and ablation.get("key") == key
        ):
            candidates.append(run_dir)
    return tuple(sorted(candidates))


def _complete_or_recover_job(
    job: dict[str, object], configuration: dict[str, object]
) -> bool:
    key = str(job["context_ablation"])
    seed = int(job["seed"])
    if job.get("status") == "completed":
        origin = job.get("result_origin")
        producing_commit = (
            STAGE1_PRODUCING_COMMIT
            if origin == "adopted_stage1"
            else configuration["orchestrator_git_commit_sha"]
            if origin == "trained_stage2"
            else None
        )
        if producing_commit is None:
            raise ValueError(f"Completed Stage-2 job has no valid origin: {key}/{seed}")
        score = validate_stage2_completed_run(
            Path(str(job["run_dir"])),
            configuration,
            key,
            seed,
            str(producing_commit),
        )
        if not math.isclose(
            score,
            float(job["primary_validation_ic"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Stage-2 state score disagrees for {key}/{seed}")
        return True

    candidates = set(_candidate_run_dirs(configuration, key, seed))
    if job.get("run_dir"):
        candidates.add(Path(str(job["run_dir"])).resolve())
    completed: list[tuple[Path, float]] = []
    for run_dir in sorted(candidates):
        try:
            score = validate_stage2_completed_run(
                run_dir,
                configuration,
                key,
                seed,
                str(configuration["orchestrator_git_commit_sha"]),
            )
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
            continue
        completed.append((run_dir, score))
    if len(completed) > 1:
        raise ValueError(f"Multiple completed runs match Stage-2 job {key}/{seed}")
    if not completed:
        return False
    run_dir, score = completed[0]
    job.update(
        {
            "status": "completed",
            "result_origin": "trained_stage2",
            "run_dir": str(run_dir),
            "producing_git_commit_sha": configuration["orchestrator_git_commit_sha"],
            "source_stage1_state": None,
            "source_stage1_job": None,
            "completed_at_utc": job.get("completed_at_utc")
            or datetime.now(timezone.utc).isoformat(),
            "primary_validation_ic": score,
            "error": None,
        }
    )
    return True


def _assert_invocation_identity(
    commit: str, feature_store: Path, configuration: dict[str, object]
) -> None:
    current_commit, _ = _git_identity(require_clean=True)
    current_store = resolve_feature_store().resolve()
    if current_commit != commit or not feature_stores_equivalent(
        current_store, feature_store
    ):
        raise RuntimeError("Git commit or canonical feature store changed mid-sweep")
    current_identity = _feature_store_identity(current_store)
    expected_identity = configuration["feature_store"]
    if not isinstance(expected_identity, dict) or (
        current_identity["manifest_sha256"] != expected_identity["manifest_sha256"]
    ):
        raise RuntimeError("Canonical feature-store identity changed mid-sweep")


def dry_run_payload(stage1_state_path: Path) -> dict[str, object]:
    commit, clean, feature_store, configuration = _prepare(
        require_clean=False, stage1_state_path=stage1_state_path
    )
    state = _new_state(configuration, stage1_state_path)
    jobs = state["jobs"]
    adopted = sum(job["result_origin"] == "adopted_stage1" for job in jobs)
    pending = sum(job["status"] == "pending" for job in jobs)
    return {
        "sweep_name": SWEEP_NAME,
        "dry_run": True,
        "worktree_clean": clean,
        "orchestrator_git_commit_sha": commit,
        "resolved_feature_store_path": str(feature_store),
        "logical_job_count": len(jobs),
        "adopted_completed_job_count": adopted,
        "pending_training_job_count": pending,
        "configuration": configuration,
        "jobs": jobs,
    }


def run_sweep(state_dir: Path, stage1_state_path: Path) -> Path:
    commit, _, feature_store, configuration = _prepare(
        require_clean=True, stage1_state_path=stage1_state_path
    )
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.json"
    with exclusive_process_lock(state_dir / "sweep.lock", SWEEP_NAME):
        if owner := active_lock_owner(PRODUCTION_TRAINING_LOCK):
            raise RuntimeError(f"Another production training run is active: {owner}")
        state = _load_state(state_path, configuration, stage1_state_path)
        _atomic_write_json(state_path, state)
        jobs = state["jobs"]
        for position, job in enumerate(jobs, start=1):
            if _complete_or_recover_job(job, configuration):
                _atomic_write_json(state_path, state)
                print(
                    f"[{position}/18] verified {job['context_ablation']} "
                    f"seed={job['seed']}: {job['run_dir']}",
                    flush=True,
                )
                continue
            _assert_invocation_identity(commit, feature_store, configuration)
            if owner := active_lock_owner(PRODUCTION_TRAINING_LOCK):
                raise RuntimeError(
                    f"Another production training run is active: {owner}"
                )
            before = _production_run_directories()
            job.update(
                {
                    "status": "running",
                    "result_origin": None,
                    "run_dir": None,
                    "producing_git_commit_sha": None,
                    "source_stage1_state": None,
                    "source_stage1_job": None,
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "completed_at_utc": None,
                    "primary_validation_ic": None,
                    "error": None,
                }
            )
            _atomic_write_json(state_path, state)
            print(
                f"[{position}/18] starting {job['context_ablation']} "
                f"seed={job['seed']}",
                flush=True,
            )
            try:
                result = subprocess.run(job["command"], cwd=_RESEARCH, check=False)
            except OSError as error:
                job.update({"status": "failed", "error": str(error)})
                _atomic_write_json(state_path, state)
                raise RuntimeError(
                    "Could not start Stage-2 training for "
                    f"{job['context_ablation']} seed={job['seed']}"
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
                    "Stage-2 training failed for "
                    f"{job['context_ablation']} seed={job['seed']}"
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
                score = validate_stage2_completed_run(
                    created[0],
                    configuration,
                    str(job["context_ablation"]),
                    int(job["seed"]),
                    commit,
                )
            except (
                FileNotFoundError,
                KeyError,
                OSError,
                TypeError,
                ValueError,
            ) as error:
                job.update(
                    {
                        "status": "failed",
                        "error": f"completed-run validation failed: {error}",
                    }
                )
                _atomic_write_json(state_path, state)
                raise RuntimeError(
                    "Stage-2 completed-run validation failed for "
                    f"{job['context_ablation']} seed={job['seed']}"
                ) from error
            job.update(
                {
                    "status": "completed",
                    "result_origin": "trained_stage2",
                    "producing_git_commit_sha": commit,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "primary_validation_ic": score,
                    "error": None,
                }
            )
            _atomic_write_json(state_path, state)
            print(
                f"[{position}/18] completed {job['context_ablation']} "
                f"seed={job['seed']} IC={score:.8f}",
                flush=True,
            )
        verified = sum(_complete_or_recover_job(job, configuration) for job in jobs)
        if verified != 18:
            raise RuntimeError("Stage-2 did not finish with exactly 18 verified jobs")
        state["status"] = "completed"
        state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(state_path, state)
    return state_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--stage1-state", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage1_state = args.stage1_state.resolve()
    if args.dry_run:
        print(
            json.dumps(dry_run_payload(stage1_state), indent=2, allow_nan=False),
            flush=True,
        )
        return
    state_path = run_sweep(args.state_dir.resolve(), stage1_state)
    print(f"Stage-2 sweep completed: {state_path}", flush=True)


if __name__ == "__main__":
    main()
