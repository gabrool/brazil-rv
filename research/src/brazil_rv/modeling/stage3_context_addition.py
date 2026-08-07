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
from types import MappingProxyType

from .context_ablation import get_context_ablation, resolve_context_ablation_for_store
from .contract import (
    FEATURE_CONTRACT_VERSION,
    GLOBAL_CONTEXT_SYMBOLS,
    LOCAL_CONTEXT_SYMBOLS,
    PROJECT_ROOT,
    RUN_OUTPUT_BASE,
    SplitBoundaries,
)
from .data import resolve_feature_store, validate_feature_store
from .process_lock import (
    PRODUCTION_TRAINING_LOCK,
    active_lock_owner,
    exclusive_process_lock,
)
from .stage2_context_ablation import (
    STAGE1_PRODUCING_COMMIT,
    STAGE2_CONTEXT_ABLATION_ORDER,
    STAGE2_SEEDS,
    STAGE2_UPDATES_PER_EPOCH,
    STATE_VERSION as STAGE2_STATE_VERSION,
    SWEEP_NAME as STAGE2_SWEEP_NAME,
    _feature_store_identity,
    _training_semantics,
    _validate_validation_artifacts,
    feature_stores_equivalent,
)

SWEEP_NAME = "stage3_context_addition_matched_seeds"
STATE_VERSION = 1
STAGE2_PRODUCING_COMMIT = "fb2d9787bb63d25fa57c8ce24d73d4d6038f5085"
PACKAGED_FEATURE_MANIFEST_SHA256 = (
    "a02cc58d6d91e5366356a3d97fa95e877b7951262153e45ea94afff6c5c3035"
)
STAGE3_LOGICAL_CONFIGURATION_ORDER = (
    "core",
    "core_plus_win",
    "core_plus_es",
    "core_plus_nq",
    "core_plus_cl",
    "core_plus_hg",
    "core_plus_6e",
    "core_plus_6m",
)
STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION = MappingProxyType(
    {
        "core": "drop_win_and_global_non_rates",
        "core_plus_win": "drop_global_non_rates",
        "core_plus_es": "drop_win_and_global_non_rates_except_es",
        "core_plus_nq": "drop_win_and_global_non_rates_except_nq",
        "core_plus_cl": "drop_win_and_global_non_rates_except_cl",
        "core_plus_hg": "drop_win_and_global_non_rates_except_hg",
        "core_plus_6e": "drop_win_and_global_non_rates_except_6e",
        "core_plus_6m": "drop_win_and_global_non_rates_except_6m",
    }
)
STAGE3_SEEDS = STAGE2_SEEDS
ADOPTED_STAGE2_LOGICAL_CONFIGURATION = "core_plus_win"
ADOPTED_STAGE2_CONTEXT_ABLATION = "drop_global_non_rates"
_REPOSITORY = PROJECT_ROOT / "quant" / "b3-quant"
_RESEARCH = _REPOSITORY / "research"


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
        raise RuntimeError("Stage-3 execution requires a clean Git worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, clean


def build_stage3_command(logical_configuration: str, seed: int) -> tuple[str, ...]:
    try:
        key = STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION[logical_configuration]
    except KeyError as error:
        raise ValueError("Unknown Stage-3 logical configuration") from error
    if seed not in STAGE3_SEEDS:
        raise ValueError("Stage-3 seed is outside the canonical matrix")
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
        str(seed),
        "--context-ablation",
        key,
    )


def stage3_jobs() -> tuple[dict[str, object], ...]:
    jobs = tuple(
        {
            "logical_configuration": logical,
            "context_ablation": STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION[logical],
            "context_ablation_metadata": get_context_ablation(
                STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION[logical]
            ).metadata(),
            "seed": seed,
            "command": list(build_stage3_command(logical, seed)),
        }
        for logical in STAGE3_LOGICAL_CONFIGURATION_ORDER
        for seed in STAGE3_SEEDS
    )
    identities = {
        (job["logical_configuration"], job["context_ablation"], job["seed"])
        for job in jobs
    }
    if len(jobs) != 24 or len(identities) != 24:
        raise RuntimeError("Stage-3 matrix must contain 24 unique jobs")
    return jobs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _feature_identities_equivalent(
    left: dict[str, object], right: dict[str, object]
) -> bool:
    left_copy = dict(left)
    right_copy = dict(right)
    left_path = Path(str(left_copy.pop("resolved_path", "")))
    right_path = Path(str(right_copy.pop("resolved_path", "")))
    if left_copy != right_copy:
        return False
    return feature_stores_equivalent(left_path, right_path) or left_copy.get(
        "manifest_sha256"
    ) == right_copy.get("manifest_sha256")


def _normalized_path_reference(value: object) -> str:
    return str(value).replace("\\", "/").rstrip("/").casefold()


def _feature_path_reference_matches(
    recorded: str,
    configuration: dict[str, object],
    producing_commit: str,
) -> bool:
    feature_identity = configuration.get("feature_store")
    if not isinstance(feature_identity, dict):
        return False
    accepted = [str(feature_identity.get("resolved_path", ""))]
    if producing_commit == STAGE2_PRODUCING_COMMIT:
        accepted.append(
            str(configuration.get("source_stage2_feature_store_resolved_path", ""))
        )
    normalized_recorded = _normalized_path_reference(recorded)
    if normalized_recorded and normalized_recorded in {
        _normalized_path_reference(path) for path in accepted if path
    }:
        return True
    recorded_path = Path(recorded)
    for accepted_path in (Path(path) for path in accepted if path):
        if feature_stores_equivalent(recorded_path, accepted_path):
            return True
    if recorded_path.is_dir():
        try:
            recorded_identity = _feature_store_identity(recorded_path)
        except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
            return False
        return _feature_identities_equivalent(recorded_identity, feature_identity)
    return False


def _reject_test_derived_metadata(payload: object, location: str) -> None:
    if isinstance(payload, dict):
        for raw_key, value in payload.items():
            key = str(raw_key).lower().replace("-", "_")
            if key not in {"test_start", "test_end"} and "test" in key.split("_"):
                raise ValueError(
                    f"{location} contains forbidden test-derived field: {raw_key}"
                )
            if (
                key in {"selection_split", "model_selection_split", "ranking_split"}
                and str(value).lower() == "test"
            ):
                raise ValueError(f"{location} uses the test split for selection")
            _reject_test_derived_metadata(value, location)
    elif isinstance(payload, list):
        for value in payload:
            _reject_test_derived_metadata(value, location)


def _configuration(
    commit: str, feature_store: Path, stage2_state_path: Path
) -> dict[str, object]:
    feature_identity = _feature_store_identity(feature_store)
    source_state = json.loads(stage2_state_path.read_text(encoding="utf-8"))
    source_configuration = source_state.get("configuration")
    source_feature_identity = (
        source_configuration.get("feature_store")
        if isinstance(source_configuration, dict)
        else None
    )
    if not isinstance(source_feature_identity, dict) or not isinstance(
        source_feature_identity.get("resolved_path"), str
    ):
        raise ValueError("Stage-2 source state is missing its feature-store path")
    return {
        "orchestrator_git_commit_sha": commit,
        "feature_store": feature_identity,
        "feature_contract": FEATURE_CONTRACT_VERSION,
        "local_context_symbols": list(LOCAL_CONTEXT_SYMBOLS),
        "global_context_symbols": list(GLOBAL_CONTEXT_SYMBOLS),
        "training_semantics": _training_semantics(),
        "split_boundaries": {
            key: str(value) for key, value in asdict(SplitBoundaries()).items()
        },
        "logical_configuration_order": list(STAGE3_LOGICAL_CONFIGURATION_ORDER),
        "context_ablation_by_logical_configuration": dict(
            STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION
        ),
        "context_ablation_metadata_by_logical_configuration": {
            logical: get_context_ablation(key).metadata()
            for logical, key in STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION.items()
        },
        "seeds": list(STAGE3_SEEDS),
        "logical_job_count": 24,
        "adopted_stage2_job_count": 3,
        "new_training_job_count": 21,
        "source_stage2_state": str(stage2_state_path),
        "source_stage2_state_sha256": _sha256(stage2_state_path),
        "source_stage2_feature_store_resolved_path": source_feature_identity[
            "resolved_path"
        ],
        "required_stage2_producing_commit": STAGE2_PRODUCING_COMMIT,
        "required_feature_manifest_sha256": PACKAGED_FEATURE_MANIFEST_SHA256,
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


def validate_stage3_completed_run(
    run_dir: Path,
    configuration: dict[str, object],
    key: str,
    seed: int,
    producing_commit: str,
) -> float:
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _reject_test_derived_metadata(manifest, f"run manifest {run_dir}")
    semantics = configuration.get("training_semantics")
    if not isinstance(semantics, dict):
        raise ValueError("Stage-3 configuration has invalid training semantics")
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
    feature_identity = configuration.get("feature_store")
    if not isinstance(feature_identity, dict):
        raise ValueError("Stage-3 configuration has invalid feature-store identity")
    recorded_feature_store = manifest.get("resolved_feature_store_path")
    if not isinstance(recorded_feature_store, str):
        raise ValueError(f"Completed run {run_dir} is missing its feature-store path")
    if not _feature_path_reference_matches(
        recorded_feature_store, configuration, producing_commit
    ):
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


def _production_run_directories() -> set[Path]:
    if not RUN_OUTPUT_BASE.is_dir():
        return set()
    return {
        path.resolve()
        for path in RUN_OUTPUT_BASE.iterdir()
        if path.is_dir() and path.name != "_ops"
    }


def _source_run_candidates(
    source_job: dict[str, object], seed: int
) -> tuple[Path, ...]:
    candidates = _production_run_directories()
    recorded = Path(str(source_job.get("run_dir")))
    if recorded.is_dir():
        candidates.add(recorded.resolve())
    compatible: list[Path] = []
    for run_dir in sorted(candidates):
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        ablation = manifest.get("context_ablation")
        if (
            manifest.get("git_commit_sha") == STAGE2_PRODUCING_COMMIT
            and manifest.get("seed") == seed
            and isinstance(ablation, dict)
            and ablation.get("key") == ADOPTED_STAGE2_CONTEXT_ABLATION
        ):
            compatible.append(run_dir)
    return tuple(compatible)


def _validate_stage2_configuration(
    source: dict[str, object], current: dict[str, object]
) -> None:
    expected = {
        "orchestrator_git_commit_sha": STAGE2_PRODUCING_COMMIT,
        "feature_contract": current["feature_contract"],
        "local_context_symbols": current["local_context_symbols"],
        "global_context_symbols": current["global_context_symbols"],
        "training_semantics": current["training_semantics"],
        "split_boundaries": current["split_boundaries"],
        "configuration_order": list(STAGE2_CONTEXT_ABLATION_ORDER),
        "seeds": list(STAGE2_SEEDS),
        "required_stage1_producing_commit": STAGE1_PRODUCING_COMMIT,
    }
    for field, value in expected.items():
        if source.get(field) != value:
            raise ValueError(f"Stage-2 state has incompatible configuration: {field}")
    if not isinstance(source.get("source_stage1_state"), str):
        raise ValueError("Stage-2 state is missing its Stage-1 provenance")
    source_identity = source.get("feature_store")
    current_identity = current.get("feature_store")
    if (
        not isinstance(source_identity, dict)
        or not isinstance(current_identity, dict)
        or not _feature_identities_equivalent(source_identity, current_identity)
    ):
        raise ValueError("Stage-2 state identifies a different feature store")
    if _normalized_path_reference(source_identity.get("resolved_path")) != (
        _normalized_path_reference(
            current.get("source_stage2_feature_store_resolved_path")
        )
    ):
        raise ValueError("Stage-2 source feature-store path provenance changed")


def _validated_stage2_adoptions(
    state_path: Path, configuration: dict[str, object]
) -> dict[int, tuple[dict[str, object], Path, float, str]]:
    raw = state_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != configuration["source_stage2_state_sha256"]:
        raise ValueError("Stage-2 source state changed during preflight")
    state = json.loads(raw)
    _reject_test_derived_metadata(state, "Stage-2 source state")
    if (
        state.get("state_version") != STAGE2_STATE_VERSION
        or state.get("sweep_name") != STAGE2_SWEEP_NAME
    ):
        raise ValueError("Stage-2 state has an incompatible identity")
    if state.get("status") != "completed":
        raise ValueError("Stage-3 adoption requires a completed Stage-2 state")
    source_configuration = state.get("configuration")
    jobs = state.get("jobs")
    if not isinstance(source_configuration, dict) or not isinstance(jobs, list):
        raise ValueError("Stage-2 state is missing configuration or jobs")
    _validate_stage2_configuration(source_configuration, configuration)
    expected = tuple(
        (key, seed) for key in STAGE2_CONTEXT_ABLATION_ORDER for seed in STAGE2_SEEDS
    )
    actual = tuple(
        (job.get("context_ablation"), job.get("seed"))
        for job in jobs
        if isinstance(job, dict)
    )
    if len(jobs) != 18 or actual != expected:
        raise ValueError("Stage-2 state does not contain the canonical 18 jobs")
    if any(job.get("status") != "completed" for job in jobs):
        raise ValueError("Stage-2 state contains an incomplete job")
    recorded_run_dirs = tuple(str(job.get("run_dir")) for job in jobs)
    if (
        any(not value or value == "None" for value in recorded_run_dirs)
        or len(set(recorded_run_dirs)) != 18
    ):
        raise ValueError("Stage-2 state contains missing or duplicate run directories")

    adopted: dict[int, tuple[dict[str, object], Path, float, str]] = {}
    by_identity = {
        (str(job["context_ablation"]), int(job["seed"])): job for job in jobs
    }
    for seed in STAGE3_SEEDS:
        source_job = by_identity[(ADOPTED_STAGE2_CONTEXT_ABLATION, seed)]
        if (
            source_job.get("result_origin") != "trained_stage2"
            or source_job.get("producing_git_commit_sha") != STAGE2_PRODUCING_COMMIT
            or source_job.get("source_stage1_state") is not None
            or source_job.get("source_stage1_job") is not None
        ):
            raise ValueError(f"Stage-2 source provenance is invalid for seed {seed}")
        completed: list[tuple[Path, float, str]] = []
        for run_dir in _source_run_candidates(source_job, seed):
            try:
                score = validate_stage3_completed_run(
                    run_dir,
                    configuration,
                    ADOPTED_STAGE2_CONTEXT_ABLATION,
                    seed,
                    STAGE2_PRODUCING_COMMIT,
                )
            except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
                continue
            completed.append((run_dir, score, _sha256(run_dir / "run_manifest.json")))
        if len(completed) != 1:
            raise ValueError(
                "Stage-3 preflight requires exactly one compatible completed "
                f"Stage-2 drop_global_non_rates run for seed {seed}; "
                f"found {len(completed)}"
            )
        run_dir, score, manifest_sha = completed[0]
        if not math.isclose(
            score,
            float(source_job.get("primary_validation_ic")),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Stage-2 source score disagrees for seed {seed}")
        adopted[seed] = (source_job, run_dir, score, manifest_sha)
    return adopted


def _new_state(
    configuration: dict[str, object],
    stage2_state_path: Path,
    adopted: dict[int, tuple[dict[str, object], Path, float, str]],
) -> dict[str, object]:
    now = datetime.now(timezone.utc).isoformat()
    jobs: list[dict[str, object]] = []
    for base in stage3_jobs():
        logical = str(base["logical_configuration"])
        seed = int(base["seed"])
        job = {
            **base,
            "status": "pending",
            "result_origin": None,
            "run_dir": None,
            "run_manifest_sha256": None,
            "producing_git_commit_sha": None,
            "source_stage2_state": None,
            "source_stage2_state_sha256": None,
            "source_stage2_job": None,
            "started_at_utc": None,
            "completed_at_utc": None,
            "primary_validation_ic": None,
            "recovery_count": 0,
            "last_recovery_at_utc": None,
            "error": None,
        }
        if logical == ADOPTED_STAGE2_LOGICAL_CONFIGURATION:
            source_job, run_dir, score, manifest_sha = adopted[seed]
            job.update(
                {
                    "status": "completed",
                    "result_origin": "adopted_stage2",
                    "run_dir": str(run_dir),
                    "run_manifest_sha256": manifest_sha,
                    "producing_git_commit_sha": STAGE2_PRODUCING_COMMIT,
                    "source_stage2_state": str(stage2_state_path),
                    "source_stage2_state_sha256": configuration[
                        "source_stage2_state_sha256"
                    ],
                    "source_stage2_job": {
                        "position": STAGE2_CONTEXT_ABLATION_ORDER.index(
                            ADOPTED_STAGE2_CONTEXT_ABLATION
                        )
                        * len(STAGE2_SEEDS)
                        + STAGE2_SEEDS.index(seed),
                        "logical_configuration": ADOPTED_STAGE2_CONTEXT_ABLATION,
                        "context_ablation": ADOPTED_STAGE2_CONTEXT_ABLATION,
                        "seed": seed,
                        "run_dir": str(source_job["run_dir"]),
                        "result_origin": source_job["result_origin"],
                        "producing_git_commit_sha": source_job[
                            "producing_git_commit_sha"
                        ],
                    },
                    "completed_at_utc": now,
                    "primary_validation_ic": score,
                }
            )
        jobs.append(job)
    if sum(job["result_origin"] == "adopted_stage2" for job in jobs) != 3:
        raise RuntimeError("Fresh Stage-3 state must adopt exactly three jobs")
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
    stored_feature = stored_copy.pop("feature_store", None)
    current_feature = current_copy.pop("feature_store", None)
    stored_copy.pop("source_stage2_state", None)
    current_copy.pop("source_stage2_state", None)
    return (
        stored_copy == current_copy
        and isinstance(stored_feature, dict)
        and isinstance(current_feature, dict)
        and _feature_identities_equivalent(stored_feature, current_feature)
    )


def _validate_adopted_job(
    job: dict[str, object],
    configuration: dict[str, object],
    source_job: dict[str, object],
    run_dir: Path,
    score: float,
    manifest_sha: str,
) -> None:
    seed = int(job["seed"])
    provenance = job.get("source_stage2_job")
    if (
        job.get("status") != "completed"
        or job.get("result_origin") != "adopted_stage2"
        or job.get("producing_git_commit_sha") != STAGE2_PRODUCING_COMMIT
        or job.get("source_stage2_state_sha256")
        != configuration["source_stage2_state_sha256"]
        or not isinstance(job.get("source_stage2_state"), str)
        or not isinstance(provenance, dict)
        or provenance.get("logical_configuration") != ADOPTED_STAGE2_CONTEXT_ABLATION
        or provenance.get("context_ablation") != ADOPTED_STAGE2_CONTEXT_ABLATION
        or provenance.get("seed") != seed
        or provenance.get("position")
        != STAGE2_CONTEXT_ABLATION_ORDER.index(ADOPTED_STAGE2_CONTEXT_ABLATION)
        * len(STAGE2_SEEDS)
        + STAGE2_SEEDS.index(seed)
        or provenance.get("run_dir") != source_job.get("run_dir")
        or provenance.get("result_origin") != "trained_stage2"
        or provenance.get("producing_git_commit_sha") != STAGE2_PRODUCING_COMMIT
        or job.get("run_manifest_sha256") != manifest_sha
        or not math.isclose(
            float(job.get("primary_validation_ic")),
            score,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"Stage-3 adopted provenance disagrees for seed {seed}")
    if _sha256(run_dir / "run_manifest.json") != manifest_sha:
        raise ValueError(f"Stage-3 adopted run manifest changed for seed {seed}")
    job["run_dir"] = str(run_dir)


def _load_state(
    path: Path,
    configuration: dict[str, object],
    stage2_state_path: Path,
    adopted: dict[int, tuple[dict[str, object], Path, float, str]],
) -> dict[str, object]:
    if not path.exists():
        return _new_state(configuration, stage2_state_path, adopted)
    state = json.loads(path.read_text(encoding="utf-8"))
    _reject_test_derived_metadata(state, "Stage-3 state")
    if (
        state.get("state_version") != STATE_VERSION
        or state.get("sweep_name") != SWEEP_NAME
    ):
        raise ValueError("Stage-3 state has an incompatible identity")
    if state.get("status") not in {"running", "completed"}:
        raise ValueError("Stage-3 state has an invalid status")
    stored_configuration = state.get("configuration")
    if not isinstance(stored_configuration, dict) or not _configurations_match(
        stored_configuration, configuration
    ):
        raise ValueError("Stage-3 state configuration does not match this invocation")
    jobs = state.get("jobs")
    expected = tuple(
        (
            job["logical_configuration"],
            job["context_ablation"],
            job["seed"],
        )
        for job in stage3_jobs()
    )
    actual = (
        tuple(
            (
                job.get("logical_configuration"),
                job.get("context_ablation"),
                job.get("seed"),
            )
            for job in jobs
            if isinstance(job, dict)
        )
        if isinstance(jobs, list)
        else ()
    )
    if actual != expected:
        raise ValueError("Stage-3 state does not contain the canonical 24 jobs")
    allowed_statuses = {"pending", "running", "failed", "completed"}
    for job in jobs:
        logical = str(job["logical_configuration"])
        key = str(job["context_ablation"])
        seed = int(job["seed"])
        recovery_count = job.get("recovery_count")
        if (
            job.get("status") not in allowed_statuses
            or job.get("context_ablation_metadata")
            != get_context_ablation(key).metadata()
            or job.get("command") != list(build_stage3_command(logical, seed))
            or not isinstance(recovery_count, int)
            or isinstance(recovery_count, bool)
            or recovery_count < 0
        ):
            raise ValueError(f"Stage-3 job metadata is malformed: {logical}")
        should_be_adopted = logical == ADOPTED_STAGE2_LOGICAL_CONFIGURATION
        if not should_be_adopted and job.get("result_origin") == "adopted_stage2":
            raise ValueError(f"Unexpected Stage-2 adoption: {logical}/{job['seed']}")
    if state["status"] == "completed" and any(
        job["status"] != "completed" for job in jobs
    ):
        raise ValueError("Completed Stage-3 state contains an incomplete job")
    by_identity = {
        (str(job["logical_configuration"]), int(job["seed"])): job for job in jobs
    }
    for seed, values in adopted.items():
        source_job, run_dir, score, manifest_sha = values
        _validate_adopted_job(
            by_identity[(ADOPTED_STAGE2_LOGICAL_CONFIGURATION, seed)],
            configuration,
            source_job,
            run_dir,
            score,
            manifest_sha,
        )
    return state


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


def _completed_job_artifacts(
    job: dict[str, object], configuration: dict[str, object]
) -> tuple[Path, float, str]:
    logical = str(job["logical_configuration"])
    key = str(job["context_ablation"])
    seed = int(job["seed"])
    origin = job.get("result_origin")
    if origin == "adopted_stage2":
        producing_commit = STAGE2_PRODUCING_COMMIT
        source_job = job.get("source_stage2_job")
        if not isinstance(source_job, dict):
            raise ValueError(
                f"Adopted Stage-3 job lacks source provenance: {logical}/{seed}"
            )
        candidates = set(_source_run_candidates(source_job, seed))
    elif origin == "trained_stage3":
        producing_commit = str(configuration["orchestrator_git_commit_sha"])
        candidates = set(_candidate_run_dirs(configuration, key, seed))
    else:
        raise ValueError(f"Completed Stage-3 job has no valid origin: {logical}/{seed}")
    recorded = Path(str(job.get("run_dir")))
    if (recorded / "run_manifest.json").is_file():
        candidates.add(recorded.resolve())
    completed: list[tuple[Path, float, str]] = []
    for run_dir in sorted(candidates):
        try:
            score = validate_stage3_completed_run(
                run_dir, configuration, key, seed, producing_commit
            )
            manifest_sha = _sha256(run_dir / "run_manifest.json")
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
            continue
        if manifest_sha == job.get("run_manifest_sha256"):
            completed.append((run_dir, score, manifest_sha))
    if len(completed) != 1:
        raise ValueError(
            "Completed Stage-3 job requires exactly one compatible immutable "
            f"run for {logical}/{seed}; found {len(completed)}"
        )
    run_dir, score, manifest_sha = completed[0]
    if not math.isclose(
        score,
        float(job.get("primary_validation_ic")),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"Stage-3 state score disagrees for {logical}/{seed}")
    return run_dir, score, manifest_sha


def _complete_or_recover_job(
    job: dict[str, object], configuration: dict[str, object]
) -> bool:
    logical = str(job["logical_configuration"])
    key = str(job["context_ablation"])
    seed = int(job["seed"])
    if job.get("status") == "completed":
        run_dir, _, _ = _completed_job_artifacts(job, configuration)
        job["run_dir"] = str(run_dir)
        return True

    candidates = set(_candidate_run_dirs(configuration, key, seed))
    if job.get("run_dir"):
        recorded = Path(str(job["run_dir"]))
        if recorded.is_dir():
            candidates.add(recorded.resolve())
    completed: list[tuple[Path, float, str]] = []
    for run_dir in sorted(candidates):
        try:
            score = validate_stage3_completed_run(
                run_dir,
                configuration,
                key,
                seed,
                str(configuration["orchestrator_git_commit_sha"]),
            )
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
            continue
        completed.append((run_dir, score, _sha256(run_dir / "run_manifest.json")))
    if len(completed) > 1:
        raise ValueError(f"Multiple completed runs match Stage-3 job {logical}/{seed}")
    if completed:
        run_dir, score, manifest_sha = completed[0]
        job.update(
            {
                "status": "completed",
                "result_origin": "trained_stage3",
                "run_dir": str(run_dir),
                "run_manifest_sha256": manifest_sha,
                "producing_git_commit_sha": configuration[
                    "orchestrator_git_commit_sha"
                ],
                "source_stage2_state": None,
                "source_stage2_state_sha256": None,
                "source_stage2_job": None,
                "completed_at_utc": job.get("completed_at_utc")
                or datetime.now(timezone.utc).isoformat(),
                "primary_validation_ic": score,
                "error": None,
            }
        )
        return True
    if job.get("status") == "running":
        job["recovery_count"] = int(job.get("recovery_count", 0)) + 1
        job["last_recovery_at_utc"] = datetime.now(timezone.utc).isoformat()
    return False


def _prepare(
    *, require_clean: bool, stage2_state_path: Path
) -> tuple[str, bool, Path, dict[str, object]]:
    commit, clean = _git_identity(require_clean=require_clean)
    feature_store = resolve_feature_store().resolve()
    validate_feature_store(feature_store)
    feature_identity = _feature_store_identity(feature_store)
    if feature_identity["manifest_sha256"] != PACKAGED_FEATURE_MANIFEST_SHA256:
        raise ValueError("Canonical feature store is not the packaged Stage-2 store")
    for key in STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION.values():
        resolve_context_ablation_for_store(feature_store, key)
    configuration = _configuration(commit, feature_store, stage2_state_path)
    return commit, clean, feature_store, configuration


def _assert_invocation_identity(
    commit: str, feature_store: Path, configuration: dict[str, object]
) -> None:
    current_commit, _ = _git_identity(require_clean=True)
    current_store = resolve_feature_store().resolve()
    current_identity = _feature_store_identity(current_store)
    expected_identity = configuration.get("feature_store")
    if (
        current_commit != commit
        or not isinstance(expected_identity, dict)
        or not _feature_identities_equivalent(current_identity, expected_identity)
        or current_identity["manifest_sha256"] != PACKAGED_FEATURE_MANIFEST_SHA256
        or not feature_stores_equivalent(current_store, feature_store)
    ):
        raise RuntimeError("Git commit or canonical feature store changed mid-sweep")


def dry_run_payload(stage2_state_path: Path) -> dict[str, object]:
    commit, clean, feature_store, configuration = _prepare(
        require_clean=False, stage2_state_path=stage2_state_path
    )
    adopted = _validated_stage2_adoptions(stage2_state_path, configuration)
    state = _new_state(configuration, stage2_state_path, adopted)
    jobs = state["jobs"]
    return {
        "sweep_name": SWEEP_NAME,
        "dry_run": True,
        "worktree_clean": clean,
        "orchestrator_git_commit_sha": commit,
        "resolved_feature_store_path": str(feature_store),
        "logical_job_count": len(jobs),
        "adopted_completed_job_count": sum(
            job["result_origin"] == "adopted_stage2" for job in jobs
        ),
        "pending_training_job_count": sum(job["status"] == "pending" for job in jobs),
        "configuration": configuration,
        "jobs": jobs,
    }


def run_sweep(state_dir: Path, stage2_state_path: Path) -> Path:
    commit, _, feature_store, configuration = _prepare(
        require_clean=True, stage2_state_path=stage2_state_path
    )
    adopted = _validated_stage2_adoptions(stage2_state_path, configuration)
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.json"
    with exclusive_process_lock(state_dir / "sweep.lock", SWEEP_NAME):
        if owner := active_lock_owner(PRODUCTION_TRAINING_LOCK):
            raise RuntimeError(f"Another production training run is active: {owner}")
        state = _load_state(state_path, configuration, stage2_state_path, adopted)
        _atomic_write_json(state_path, state)
        jobs = state["jobs"]
        for position, job in enumerate(jobs, start=1):
            if _complete_or_recover_job(job, configuration):
                _atomic_write_json(state_path, state)
                print(
                    f"[{position}/24] verified {job['logical_configuration']} "
                    f"seed={job['seed']}: {job['run_dir']}",
                    flush=True,
                )
                continue
            _atomic_write_json(state_path, state)
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
                    "run_manifest_sha256": None,
                    "producing_git_commit_sha": None,
                    "source_stage2_state": None,
                    "source_stage2_state_sha256": None,
                    "source_stage2_job": None,
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "completed_at_utc": None,
                    "primary_validation_ic": None,
                    "error": None,
                }
            )
            _atomic_write_json(state_path, state)
            print(
                f"[{position}/24] starting {job['logical_configuration']} "
                f"seed={job['seed']}",
                flush=True,
            )
            try:
                result = subprocess.run(job["command"], cwd=_RESEARCH, check=False)
            except OSError as error:
                job.update({"status": "failed", "error": str(error)})
                _atomic_write_json(state_path, state)
                raise RuntimeError(
                    "Could not start Stage-3 training for "
                    f"{job['logical_configuration']} seed={job['seed']}"
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
                    "Stage-3 training failed for "
                    f"{job['logical_configuration']} seed={job['seed']}"
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
                score = validate_stage3_completed_run(
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
                    "Stage-3 completed-run validation failed for "
                    f"{job['logical_configuration']} seed={job['seed']}"
                ) from error
            job.update(
                {
                    "status": "completed",
                    "result_origin": "trained_stage3",
                    "run_manifest_sha256": _sha256(created[0] / "run_manifest.json"),
                    "producing_git_commit_sha": commit,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "primary_validation_ic": score,
                    "error": None,
                }
            )
            _atomic_write_json(state_path, state)
            print(
                f"[{position}/24] completed {job['logical_configuration']} "
                f"seed={job['seed']} IC={score:.8f}",
                flush=True,
            )
        verified = sum(_complete_or_recover_job(job, configuration) for job in jobs)
        if (
            verified != 24
            or sum(job["result_origin"] == "adopted_stage2" for job in jobs) != 3
            or sum(job["result_origin"] == "trained_stage3" for job in jobs) != 21
        ):
            raise RuntimeError("Stage-3 did not finish with 3 adopted and 21 new jobs")
        state["status"] = "completed"
        state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(state_path, state)
    return state_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--stage2-state", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage2_state = args.stage2_state.resolve()
    if args.dry_run:
        print(
            json.dumps(dry_run_payload(stage2_state), indent=2, allow_nan=False),
            flush=True,
        )
        return
    state_path = run_sweep(args.state_dir.resolve(), stage2_state)
    print(f"Stage-3 sweep completed: {state_path}", flush=True)


if __name__ == "__main__":
    main()
