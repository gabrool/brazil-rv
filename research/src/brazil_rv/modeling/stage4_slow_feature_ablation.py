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

import torch

from brazil_rv.preprocessing.contract import SLOW_CHANNELS

from .analyze_stage3_context_addition import (
    _validate_configuration as _validate_stage3_configuration,
)
from .audit_slow_features import validate_training_slow_audit
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
from .feature_ablation import (
    get_feature_ablation,
    resolve_feature_ablation,
    resolve_feature_ablation_for_store,
)
from .process_lock import (
    PRODUCTION_TRAINING_LOCK,
    active_lock_owner,
    exclusive_process_lock,
)
from .stage2_context_ablation import (
    _feature_store_identity,
    _training_semantics,
)
from .stage3_context_addition import (
    PACKAGED_FEATURE_MANIFEST_SHA256,
    STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION,
    STAGE3_LOGICAL_CONFIGURATION_ORDER,
    STAGE3_SEEDS,
    STATE_VERSION as STAGE3_STATE_VERSION,
    SWEEP_NAME as STAGE3_SWEEP_NAME,
    _completed_job_artifacts as _stage3_completed_job_artifacts,
    _feature_identities_equivalent,
    _reject_test_derived_metadata,
    build_stage3_command,
    validate_stage3_completed_run,
)

SWEEP_NAME = "stage4_slow_low_prior_ablation_matched_seeds"
STATE_VERSION = 1
STAGE4_LOGICAL_CONFIGURATION_ORDER = ("full_slow", "drop_slow_low_prior")
STAGE4_FEATURE_ABLATION_BY_LOGICAL_CONFIGURATION = MappingProxyType(
    {"full_slow": "none", "drop_slow_low_prior": "drop_slow_low_prior"}
)
STAGE4_SEEDS = STAGE3_SEEDS
FROZEN_CONTEXT_ABLATION = "drop_win_and_global_non_rates"
ADOPTED_STAGE3_LOGICAL_CONFIGURATION = "core"
EXPECTED_RETAINED_CONTEXTS = (
    "WDO$",
    "DI1F27",
    "DI1F28",
    "DI1F29",
    "DI1F31",
    "DI1$N",
    "ZT.v.0",
    "ZN.v.0",
)
_REPOSITORY = PROJECT_ROOT / "quant" / "b3-quant"
_RESEARCH = _REPOSITORY / "research"
_REQUIRED_OUTPUTS = (
    "run_manifest.json",
    "best.pt",
    "final.pt",
    "history.csv",
    "validation_metrics.json",
    "validation_daily_metrics.parquet",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        raise RuntimeError("Stage-4 execution requires a clean Git worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, clean


def build_stage4_command(logical_configuration: str, seed: int) -> tuple[str, ...]:
    try:
        feature_key = STAGE4_FEATURE_ABLATION_BY_LOGICAL_CONFIGURATION[
            logical_configuration
        ]
    except KeyError as error:
        raise ValueError("Unknown Stage-4 logical configuration") from error
    if seed not in STAGE4_SEEDS:
        raise ValueError("Stage-4 seed is outside the canonical matrix")
    get_context_ablation(FROZEN_CONTEXT_ABLATION)
    get_feature_ablation(feature_key)
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
        "--context-ablation",
        FROZEN_CONTEXT_ABLATION,
        "--feature-ablation",
        feature_key,
        "--seed",
        str(seed),
    )


def _feature_ablation_metadata(key: str) -> dict[str, object]:
    return resolve_feature_ablation(
        get_feature_ablation(key), slow_features=SLOW_CHANNELS
    ).metadata()


def _job_specification(logical_configuration: str, seed: int) -> dict[str, object]:
    feature_key = STAGE4_FEATURE_ABLATION_BY_LOGICAL_CONFIGURATION[
        logical_configuration
    ]
    return {
        "logical_configuration": logical_configuration,
        "seed": seed,
        "context_ablation": get_context_ablation(FROZEN_CONTEXT_ABLATION).metadata(),
        "feature_ablation": _feature_ablation_metadata(feature_key),
        "command": list(build_stage4_command(logical_configuration, seed)),
    }


def stage4_jobs() -> tuple[dict[str, object], ...]:
    jobs: list[dict[str, object]] = []
    for logical in STAGE4_LOGICAL_CONFIGURATION_ORDER:
        feature_key = STAGE4_FEATURE_ABLATION_BY_LOGICAL_CONFIGURATION[logical]
        for seed in STAGE4_SEEDS:
            specification = _job_specification(logical, seed)
            serialized = json.dumps(
                specification, sort_keys=True, separators=(",", ":")
            )
            jobs.append(
                {
                    "logical_configuration": logical,
                    "context_ablation": FROZEN_CONTEXT_ABLATION,
                    "context_ablation_metadata": get_context_ablation(
                        FROZEN_CONTEXT_ABLATION
                    ).metadata(),
                    "feature_ablation": feature_key,
                    "feature_ablation_metadata": specification["feature_ablation"],
                    "seed": seed,
                    "command": list(build_stage4_command(logical, seed)),
                    "serialized_job_specification": serialized,
                    "job_specification_sha256": hashlib.sha256(
                        serialized.encode()
                    ).hexdigest(),
                }
            )
    identities = {
        (job["logical_configuration"], job["feature_ablation"], job["seed"])
        for job in jobs
    }
    if len(jobs) != 6 or len(identities) != 6:
        raise RuntimeError("Stage-4 matrix must contain six unique jobs")
    return tuple(jobs)


def _retained_context_symbols(store: Path) -> tuple[str, ...]:
    resolved = resolve_context_ablation_for_store(store, FROZEN_CONTEXT_ABLATION)
    retained_local = tuple(
        symbol
        for slot, symbol in enumerate(LOCAL_CONTEXT_SYMBOLS)
        if slot not in resolved.local_slots
    )
    retained_global = tuple(
        symbol
        for slot, symbol in enumerate(GLOBAL_CONTEXT_SYMBOLS)
        if slot not in resolved.global_slots
    )
    retained = (*retained_local, *retained_global)
    if retained != EXPECTED_RETAINED_CONTEXTS or resolved.equity_slow_indices != (20,):
        raise ValueError("Frozen context identity does not match the Stage-4 contract")
    return retained


def _artifact_hashes(run_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in _REQUIRED_OUTPUTS:
        path = run_dir / name
        if not path.is_file():
            raise ValueError(f"Completed run is missing required artifact: {path}")
        hashes[name] = _sha256(path)
    return hashes


def _validate_feature_identity(
    run_dir: Path,
    expected_metadata: dict[str, object],
    *,
    allow_legacy_none: bool,
) -> str:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    final = torch.load(run_dir / "final.pt", map_location="cpu", weights_only=False)
    if not isinstance(best, dict) or not isinstance(final, dict):
        raise ValueError("Completed run checkpoints are malformed")
    presence = tuple(
        "feature_ablation" in artifact for artifact in (manifest, best, final)
    )
    if not any(presence):
        if allow_legacy_none and expected_metadata.get("key") == "none":
            return "legacy_implicit_none"
        raise ValueError("Completed run is missing feature-ablation identity")
    if not all(presence):
        raise ValueError("Run/checkpoint feature-ablation presence disagrees")
    if any(
        artifact.get("feature_ablation") != expected_metadata
        for artifact in (manifest, best, final)
    ):
        raise ValueError("Run/checkpoint feature-ablation identity disagrees")
    return "explicit_registry_metadata"


def _completed_run_validation_configuration(
    configuration: dict[str, object], producing_commit: str
) -> dict[str, object]:
    if producing_commit != configuration.get("source_stage3_producing_commit"):
        return configuration
    source_path = configuration.get("source_stage3_feature_store_resolved_path")
    feature_identity = configuration.get("feature_store")
    if not isinstance(source_path, str) or not isinstance(feature_identity, dict):
        raise ValueError("Stage-3 portable feature-store provenance is incomplete")
    validation_configuration = dict(configuration)
    source_identity = dict(feature_identity)
    source_identity["resolved_path"] = source_path
    validation_configuration["feature_store"] = source_identity
    return validation_configuration


def validate_stage4_completed_run(
    run_dir: Path,
    configuration: dict[str, object],
    feature_key: str,
    seed: int,
    producing_commit: str,
    *,
    allow_legacy_none: bool = False,
) -> tuple[float, dict[str, str], str]:
    validation_configuration = _completed_run_validation_configuration(
        configuration, producing_commit
    )
    score = validate_stage3_completed_run(
        run_dir,
        validation_configuration,
        FROZEN_CONTEXT_ABLATION,
        seed,
        producing_commit,
    )
    metadata_by_key = configuration.get("feature_ablation_metadata_by_key")
    if not isinstance(metadata_by_key, dict):
        raise ValueError("Stage-4 configuration lacks feature-ablation metadata")
    expected = metadata_by_key.get(feature_key)
    if not isinstance(expected, dict):
        raise ValueError("Stage-4 configuration has an unknown feature ablation")
    source = _validate_feature_identity(
        run_dir, expected, allow_legacy_none=allow_legacy_none
    )
    return score, _artifact_hashes(run_dir), source


def _configuration(
    commit: str,
    feature_store: Path,
    stage3_state_path: Path,
    slow_audit_path: Path,
) -> dict[str, object]:
    feature_identity = _feature_store_identity(feature_store)
    stage3_state = json.loads(stage3_state_path.read_text(encoding="utf-8"))
    source_configuration = stage3_state.get("configuration")
    source_feature_identity = (
        source_configuration.get("feature_store")
        if isinstance(source_configuration, dict)
        else None
    )
    if not isinstance(source_feature_identity, dict):
        raise ValueError("Stage-3 source state lacks feature-store identity")
    feature_metadata = {
        key: resolve_feature_ablation_for_store(feature_store, key).metadata()
        for key in ("none", "drop_slow_low_prior")
    }
    audit = validate_training_slow_audit(slow_audit_path, feature_identity)
    return {
        "orchestrator_git_commit_sha": commit,
        "feature_store": feature_identity,
        "feature_contract": FEATURE_CONTRACT_VERSION,
        "local_context_symbols": list(LOCAL_CONTEXT_SYMBOLS),
        "global_context_symbols": list(GLOBAL_CONTEXT_SYMBOLS),
        "retained_context_symbols": list(_retained_context_symbols(feature_store)),
        "training_semantics": _training_semantics(),
        "split_boundaries": {
            key: str(value) for key, value in asdict(SplitBoundaries()).items()
        },
        "logical_configuration_order": list(STAGE4_LOGICAL_CONFIGURATION_ORDER),
        "feature_ablation_by_logical_configuration": dict(
            STAGE4_FEATURE_ABLATION_BY_LOGICAL_CONFIGURATION
        ),
        "feature_ablation_metadata_by_key": feature_metadata,
        "context_ablation": FROZEN_CONTEXT_ABLATION,
        "context_ablation_metadata": get_context_ablation(
            FROZEN_CONTEXT_ABLATION
        ).metadata(),
        "seeds": list(STAGE4_SEEDS),
        "logical_job_count": 6,
        "adopted_stage3_job_count": 3,
        "new_training_job_count": 3,
        "source_stage3_state": str(stage3_state_path),
        "source_stage3_state_sha256": _sha256(stage3_state_path),
        "source_stage3_feature_store_resolved_path": source_feature_identity.get(
            "resolved_path"
        ),
        "source_stage3_producing_commit": (
            source_configuration.get("orchestrator_git_commit_sha")
            if isinstance(source_configuration, dict)
            else None
        ),
        "required_feature_manifest_sha256": PACKAGED_FEATURE_MANIFEST_SHA256,
        "training_slow_audit": {
            "path": str(slow_audit_path),
            "sha256": _sha256(slow_audit_path),
            "audit_name": audit["audit_name"],
            "audit_version": audit["audit_version"],
        },
    }


def _validate_stage3_source_configuration(
    source: dict[str, object], current: dict[str, object]
) -> None:
    _validate_stage3_configuration(source)
    source_identity = source.get("feature_store")
    current_identity = current.get("feature_store")
    if (
        not isinstance(source_identity, dict)
        or not isinstance(current_identity, dict)
        or not _feature_identities_equivalent(source_identity, current_identity)
        or source_identity.get("manifest_sha256") != PACKAGED_FEATURE_MANIFEST_SHA256
    ):
        raise ValueError("Stage-3 source state identifies a different feature store")
    if source.get("orchestrator_git_commit_sha") != current.get(
        "source_stage3_producing_commit"
    ):
        raise ValueError("Stage-3 producing commit provenance changed")


def _validated_stage3_adoptions(
    state_path: Path, configuration: dict[str, object]
) -> dict[int, tuple[dict[str, object], Path, float, str, dict[str, str], str]]:
    raw = state_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != configuration["source_stage3_state_sha256"]:
        raise ValueError("Stage-3 source state changed during preflight")
    state = json.loads(raw)
    _reject_test_derived_metadata(state, "Stage-3 source state")
    if (
        state.get("state_version") != STAGE3_STATE_VERSION
        or state.get("sweep_name") != STAGE3_SWEEP_NAME
        or state.get("status") != "completed"
    ):
        raise ValueError("Stage-4 adoption requires a completed Stage-3 state")
    source_configuration = state.get("configuration")
    jobs = state.get("jobs")
    if not isinstance(source_configuration, dict) or not isinstance(jobs, list):
        raise ValueError("Stage-3 source state is missing configuration or jobs")
    _validate_stage3_source_configuration(source_configuration, configuration)
    expected = tuple(
        (
            logical,
            STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION[logical],
            seed,
        )
        for logical in STAGE3_LOGICAL_CONFIGURATION_ORDER
        for seed in STAGE3_SEEDS
    )
    actual = tuple(
        (
            job.get("logical_configuration"),
            job.get("context_ablation"),
            job.get("seed"),
        )
        for job in jobs
        if isinstance(job, dict)
    )
    if len(jobs) != 24 or actual != expected:
        raise ValueError("Stage-3 source state does not contain its canonical matrix")
    if any(job.get("status") != "completed" for job in jobs):
        raise ValueError("Stage-3 source state contains an incomplete job")

    by_identity = {
        (str(job["logical_configuration"]), int(job["seed"])): job for job in jobs
    }
    adopted = {}
    for seed in STAGE4_SEEDS:
        source_job = by_identity[(ADOPTED_STAGE3_LOGICAL_CONFIGURATION, seed)]
        if (
            source_job.get("context_ablation") != FROZEN_CONTEXT_ABLATION
            or source_job.get("result_origin") != "trained_stage3"
            or source_job.get("producing_git_commit_sha")
            != source_configuration["orchestrator_git_commit_sha"]
            or source_job.get("source_stage2_state") is not None
            or source_job.get("source_stage2_job") is not None
            or source_job.get("command")
            != list(build_stage3_command(ADOPTED_STAGE3_LOGICAL_CONFIGURATION, seed))
        ):
            raise ValueError(f"Stage-3 core provenance is invalid for seed {seed}")
        run_dir, source_score, source_manifest_sha = _stage3_completed_job_artifacts(
            source_job, source_configuration
        )
        if source_job.get("run_manifest_sha256") != source_manifest_sha:
            raise ValueError(f"Stage-3 core manifest hash disagrees for seed {seed}")
        score, output_hashes, identity_source = validate_stage4_completed_run(
            run_dir,
            configuration,
            "none",
            seed,
            str(source_configuration["orchestrator_git_commit_sha"]),
            allow_legacy_none=True,
        )
        if not math.isclose(score, source_score, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Stage-3 core validation score disagrees for seed {seed}")
        adopted[seed] = (
            source_job,
            run_dir,
            score,
            source_manifest_sha,
            output_hashes,
            identity_source,
        )
    return adopted


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


def _new_state(
    configuration: dict[str, object],
    stage3_state_path: Path,
    adopted: dict[int, tuple[dict[str, object], Path, float, str, dict[str, str], str]],
) -> dict[str, object]:
    now = datetime.now(timezone.utc).isoformat()
    jobs: list[dict[str, object]] = []
    for base in stage4_jobs():
        logical = str(base["logical_configuration"])
        seed = int(base["seed"])
        job = {
            **base,
            "status": "pending",
            "result_origin": None,
            "run_dir": None,
            "run_manifest_sha256": None,
            "output_sha256": None,
            "feature_ablation_identity_source": None,
            "producing_git_commit_sha": None,
            "source_stage3_state": None,
            "source_stage3_state_sha256": None,
            "source_stage3_job": None,
            "started_at_utc": None,
            "completed_at_utc": None,
            "primary_validation_ic": None,
            "recovery_count": 0,
            "last_recovery_at_utc": None,
            "error": None,
        }
        if logical == "full_slow":
            source_job, run_dir, score, manifest_sha, hashes, identity_source = adopted[
                seed
            ]
            job.update(
                {
                    "status": "completed",
                    "result_origin": "adopted_stage3",
                    "run_dir": str(run_dir),
                    "run_manifest_sha256": manifest_sha,
                    "output_sha256": dict(hashes),
                    "feature_ablation_identity_source": identity_source,
                    "producing_git_commit_sha": configuration[
                        "source_stage3_producing_commit"
                    ],
                    "source_stage3_state": str(stage3_state_path),
                    "source_stage3_state_sha256": configuration[
                        "source_stage3_state_sha256"
                    ],
                    "source_stage3_job": {
                        "position": STAGE3_LOGICAL_CONFIGURATION_ORDER.index("core")
                        * len(STAGE3_SEEDS)
                        + STAGE3_SEEDS.index(seed),
                        "logical_configuration": "core",
                        "context_ablation": FROZEN_CONTEXT_ABLATION,
                        "feature_ablation": "none",
                        "seed": seed,
                        "run_dir": str(source_job["run_dir"]),
                        "result_origin": source_job["result_origin"],
                        "producing_git_commit_sha": source_job[
                            "producing_git_commit_sha"
                        ],
                        "run_manifest_sha256": source_job["run_manifest_sha256"],
                    },
                    "completed_at_utc": now,
                    "primary_validation_ic": score,
                }
            )
        jobs.append(job)
    if sum(job["result_origin"] == "adopted_stage3" for job in jobs) != 3:
        raise RuntimeError("Fresh Stage-4 state must adopt exactly three controls")
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
    stored_copy.pop("source_stage3_state", None)
    current_copy.pop("source_stage3_state", None)
    stored_audit = stored_copy.get("training_slow_audit")
    current_audit = current_copy.get("training_slow_audit")
    if isinstance(stored_audit, dict) and isinstance(current_audit, dict):
        stored_audit = dict(stored_audit)
        current_audit = dict(current_audit)
        stored_audit.pop("path", None)
        current_audit.pop("path", None)
        stored_copy["training_slow_audit"] = stored_audit
        current_copy["training_slow_audit"] = current_audit
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
    output_hashes: dict[str, str],
    identity_source: str,
) -> None:
    seed = int(job["seed"])
    provenance = job.get("source_stage3_job")
    if (
        job.get("status") != "completed"
        or job.get("result_origin") != "adopted_stage3"
        or job.get("feature_ablation") != "none"
        or job.get("source_stage3_state_sha256")
        != configuration["source_stage3_state_sha256"]
        or not isinstance(job.get("source_stage3_state"), str)
        or not isinstance(provenance, dict)
        or provenance.get("logical_configuration") != "core"
        or provenance.get("context_ablation") != FROZEN_CONTEXT_ABLATION
        or provenance.get("feature_ablation") != "none"
        or provenance.get("seed") != seed
        or provenance.get("position")
        != STAGE3_LOGICAL_CONFIGURATION_ORDER.index("core") * len(STAGE3_SEEDS)
        + STAGE3_SEEDS.index(seed)
        or provenance.get("result_origin") != source_job.get("result_origin")
        or provenance.get("producing_git_commit_sha")
        != source_job.get("producing_git_commit_sha")
        or provenance.get("run_dir") != source_job.get("run_dir")
        or provenance.get("run_manifest_sha256")
        != source_job.get("run_manifest_sha256")
        or job.get("producing_git_commit_sha")
        != configuration["source_stage3_producing_commit"]
        or job.get("run_manifest_sha256") != manifest_sha
        or job.get("output_sha256") != output_hashes
        or job.get("feature_ablation_identity_source") != identity_source
        or not math.isclose(
            float(job.get("primary_validation_ic")),
            score,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"Stage-4 adopted provenance disagrees for seed {seed}")
    job["run_dir"] = str(run_dir)


def _load_state(
    path: Path,
    configuration: dict[str, object],
    stage3_state_path: Path,
    adopted: dict[int, tuple[dict[str, object], Path, float, str, dict[str, str], str]],
) -> dict[str, object]:
    if not path.exists():
        return _new_state(configuration, stage3_state_path, adopted)
    state = json.loads(path.read_text(encoding="utf-8"))
    _reject_test_derived_metadata(state, "Stage-4 state")
    if (
        state.get("state_version") != STATE_VERSION
        or state.get("sweep_name") != SWEEP_NAME
        or state.get("status") not in {"running", "completed"}
    ):
        raise ValueError("Stage-4 state has an incompatible identity or status")
    stored_configuration = state.get("configuration")
    if not isinstance(stored_configuration, dict) or not _configurations_match(
        stored_configuration, configuration
    ):
        raise ValueError("Stage-4 state configuration does not match this invocation")
    jobs = state.get("jobs")
    expected = tuple(
        (job["logical_configuration"], job["feature_ablation"], job["seed"])
        for job in stage4_jobs()
    )
    actual = (
        tuple(
            (
                job.get("logical_configuration"),
                job.get("feature_ablation"),
                job.get("seed"),
            )
            for job in jobs
            if isinstance(job, dict)
        )
        if isinstance(jobs, list)
        else ()
    )
    if actual != expected:
        raise ValueError("Stage-4 state does not contain the canonical six jobs")
    allowed_statuses = {"pending", "running", "failed", "completed"}
    for job, base in zip(jobs, stage4_jobs(), strict=True):
        recovery_count = job.get("recovery_count")
        immutable_fields = (
            "logical_configuration",
            "context_ablation",
            "context_ablation_metadata",
            "feature_ablation",
            "feature_ablation_metadata",
            "seed",
            "command",
            "serialized_job_specification",
            "job_specification_sha256",
        )
        if (
            job.get("status") not in allowed_statuses
            or any(job.get(field) != base[field] for field in immutable_fields)
            or not isinstance(recovery_count, int)
            or isinstance(recovery_count, bool)
            or recovery_count < 0
        ):
            raise ValueError("Stage-4 job metadata is malformed")
        if (
            job["logical_configuration"] != "full_slow"
            and job.get("result_origin") == "adopted_stage3"
        ):
            raise ValueError("Only full_slow controls may be adopted")
    if state["status"] == "completed" and any(
        job["status"] != "completed" for job in jobs
    ):
        raise ValueError("Completed Stage-4 state contains an incomplete job")
    by_identity = {
        (str(job["logical_configuration"]), int(job["seed"])): job for job in jobs
    }
    for seed, values in adopted.items():
        _validate_adopted_job(by_identity[("full_slow", seed)], configuration, *values)
    return state


def _production_run_directories() -> set[Path]:
    if not RUN_OUTPUT_BASE.is_dir():
        return set()
    return {
        path.resolve()
        for path in RUN_OUTPUT_BASE.iterdir()
        if path.is_dir() and path.name != "_ops"
    }


def _candidate_treatment_runs(
    configuration: dict[str, object], seed: int
) -> tuple[Path, ...]:
    expected = configuration["feature_ablation_metadata_by_key"]["drop_slow_low_prior"]
    candidates: list[Path] = []
    for run_dir in _production_run_directories():
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        context = manifest.get("context_ablation")
        if (
            manifest.get("git_commit_sha")
            == configuration["orchestrator_git_commit_sha"]
            and manifest.get("seed") == seed
            and isinstance(context, dict)
            and context.get("key") == FROZEN_CONTEXT_ABLATION
            and manifest.get("feature_ablation") == expected
        ):
            candidates.append(run_dir)
    return tuple(sorted(candidates))


def _valid_completed_treatment_candidates(
    job: dict[str, object], configuration: dict[str, object]
) -> tuple[tuple[Path, float, dict[str, str], str], ...]:
    seed = int(job["seed"])
    candidates = set(_candidate_treatment_runs(configuration, seed))
    if job.get("run_dir"):
        recorded = Path(str(job["run_dir"]))
        if recorded.is_dir():
            candidates.add(recorded.resolve())
    completed = []
    for run_dir in sorted(candidates):
        try:
            score, hashes, source = validate_stage4_completed_run(
                run_dir,
                configuration,
                "drop_slow_low_prior",
                seed,
                str(configuration["orchestrator_git_commit_sha"]),
            )
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
            continue
        completed.append((run_dir, score, hashes, source))
    return tuple(completed)


def _completed_job_artifacts(
    job: dict[str, object], configuration: dict[str, object]
) -> tuple[Path, float, str, dict[str, str], str]:
    logical = str(job["logical_configuration"])
    seed = int(job["seed"])
    if job.get("result_origin") == "adopted_stage3":
        run_dir = Path(str(job.get("run_dir")))
        score, hashes, source = validate_stage4_completed_run(
            run_dir,
            configuration,
            "none",
            seed,
            str(configuration["source_stage3_producing_commit"]),
            allow_legacy_none=True,
        )
    elif job.get("result_origin") == "trained_stage4":
        if (
            not isinstance(job.get("run_dir"), str)
            or job.get("producing_git_commit_sha")
            != configuration["orchestrator_git_commit_sha"]
        ):
            raise ValueError(
                f"Completed Stage-4 job lacks immutable provenance: {logical}/{seed}"
            )
        run_dir = Path(str(job["run_dir"]))
        score, hashes, source = validate_stage4_completed_run(
            run_dir,
            configuration,
            "drop_slow_low_prior",
            seed,
            str(configuration["orchestrator_git_commit_sha"]),
        )
    else:
        raise ValueError(f"Completed Stage-4 job has no valid origin: {logical}/{seed}")
    manifest_sha = hashes["run_manifest.json"]
    if (
        job.get("run_manifest_sha256") != manifest_sha
        or job.get("output_sha256") != hashes
        or job.get("feature_ablation_identity_source") != source
        or not math.isclose(
            score,
            float(job.get("primary_validation_ic")),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"Stage-4 state disagrees with artifacts: {logical}/{seed}")
    return run_dir, score, manifest_sha, hashes, source


def _complete_or_recover_job(
    job: dict[str, object], configuration: dict[str, object]
) -> bool:
    status = job.get("status")
    if status == "completed":
        _completed_job_artifacts(job, configuration)
        return True
    seed = int(job["seed"])
    completed = _valid_completed_treatment_candidates(job, configuration)
    if status == "pending":
        if completed:
            raise ValueError(
                "Unbound completed treatment artifact contamination for pending "
                f"Stage-4 seed {seed}: found {len(completed)} valid candidate(s)"
            )
        return False
    if status == "failed":
        if completed:
            raise ValueError(
                "Unbound completed treatment artifact exists for failed Stage-4 "
                f"seed {seed}; operator inspection required"
            )
        return False
    if status != "running":
        raise ValueError(f"Unsupported Stage-4 treatment status: {status}")
    if len(completed) > 1:
        raise ValueError(f"Multiple completed runs match Stage-4 treatment seed {seed}")
    job["recovery_count"] = int(job.get("recovery_count", 0)) + 1
    job["last_recovery_at_utc"] = datetime.now(timezone.utc).isoformat()
    if not completed:
        return False
    run_dir, score, hashes, source = completed[0]
    job.update(
        {
            "status": "completed",
            "result_origin": "trained_stage4",
            "run_dir": str(run_dir),
            "run_manifest_sha256": hashes["run_manifest.json"],
            "output_sha256": hashes,
            "feature_ablation_identity_source": source,
            "producing_git_commit_sha": configuration["orchestrator_git_commit_sha"],
            "source_stage3_state": None,
            "source_stage3_state_sha256": None,
            "source_stage3_job": None,
            "completed_at_utc": job.get("completed_at_utc")
            or datetime.now(timezone.utc).isoformat(),
            "primary_validation_ic": score,
            "error": None,
        }
    )
    return True


def _prepare(
    *,
    require_clean: bool,
    stage3_state_path: Path,
    slow_audit_path: Path,
) -> tuple[str, bool, Path, dict[str, object]]:
    commit, clean = _git_identity(require_clean=require_clean)
    feature_store = resolve_feature_store().resolve()
    validate_feature_store(feature_store)
    feature_identity = _feature_store_identity(feature_store)
    if feature_identity["manifest_sha256"] != PACKAGED_FEATURE_MANIFEST_SHA256:
        raise ValueError("Canonical feature store is not the frozen Stage-3 store")
    _retained_context_symbols(feature_store)
    resolve_feature_ablation_for_store(feature_store, "none")
    resolve_feature_ablation_for_store(feature_store, "drop_slow_low_prior")
    configuration = _configuration(
        commit, feature_store, stage3_state_path, slow_audit_path
    )
    return commit, clean, feature_store, configuration


def _assert_invocation_identity(
    commit: str,
    feature_store: Path,
    configuration: dict[str, object],
    stage3_state_path: Path,
    slow_audit_path: Path,
) -> None:
    current_commit, _ = _git_identity(require_clean=True)
    current_store = resolve_feature_store().resolve()
    current_identity = _feature_store_identity(current_store)
    expected_identity = configuration.get("feature_store")
    audit_identity = configuration.get("training_slow_audit")
    if (
        current_commit != commit
        or not isinstance(expected_identity, dict)
        or not _feature_identities_equivalent(current_identity, expected_identity)
        or current_identity["manifest_sha256"] != PACKAGED_FEATURE_MANIFEST_SHA256
        or _sha256(stage3_state_path) != configuration["source_stage3_state_sha256"]
        or not isinstance(audit_identity, dict)
        or _sha256(slow_audit_path) != audit_identity.get("sha256")
        or not feature_store.samefile(current_store)
    ):
        raise RuntimeError(
            "Git, source state, audit, or feature store changed mid-sweep"
        )


def dry_run_payload(
    stage3_state_path: Path, slow_audit_path: Path
) -> dict[str, object]:
    commit, clean, feature_store, configuration = _prepare(
        require_clean=False,
        stage3_state_path=stage3_state_path,
        slow_audit_path=slow_audit_path,
    )
    adopted = _validated_stage3_adoptions(stage3_state_path, configuration)
    state = _new_state(configuration, stage3_state_path, adopted)
    jobs = state["jobs"]
    for job in jobs:
        if job["status"] == "pending":
            _complete_or_recover_job(job, configuration)
    return {
        "sweep_name": SWEEP_NAME,
        "dry_run": True,
        "worktree_clean": clean,
        "orchestrator_git_commit_sha": commit,
        "resolved_feature_store_path": str(feature_store),
        "logical_job_count": len(jobs),
        "adopted_control_count": sum(
            job["result_origin"] == "adopted_stage3" for job in jobs
        ),
        "pending_training_job_count": sum(job["status"] == "pending" for job in jobs),
        "seeds": list(STAGE4_SEEDS),
        "control": "full_slow",
        "treatment": "drop_slow_low_prior",
        "context": FROZEN_CONTEXT_ABLATION,
        "test_metrics_accessed": False,
        "configuration": configuration,
        "jobs": jobs,
    }


def format_dry_run_preflight(payload: dict[str, object]) -> str:
    return "\n".join(
        (
            f"logical jobs: {payload['logical_job_count']}",
            f"adopted controls: {payload['adopted_control_count']}",
            f"new training jobs: {payload['pending_training_job_count']}",
            "seeds: 11,29,47",
            "control: full_slow",
            "treatment: drop_slow_low_prior",
            "context: drop_win_and_global_non_rates",
            "test metrics accessed: no",
        )
    )


def run_sweep(state_dir: Path, stage3_state_path: Path, slow_audit_path: Path) -> Path:
    commit, _, feature_store, configuration = _prepare(
        require_clean=True,
        stage3_state_path=stage3_state_path,
        slow_audit_path=slow_audit_path,
    )
    adopted = _validated_stage3_adoptions(stage3_state_path, configuration)
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.json"
    with exclusive_process_lock(state_dir / "sweep.lock", SWEEP_NAME):
        if owner := active_lock_owner(PRODUCTION_TRAINING_LOCK):
            raise RuntimeError(f"Another production training run is active: {owner}")
        state = _load_state(state_path, configuration, stage3_state_path, adopted)
        _atomic_write_json(state_path, state)
        jobs = state["jobs"]
        for position, job in enumerate(jobs, start=1):
            if _complete_or_recover_job(job, configuration):
                _atomic_write_json(state_path, state)
                print(
                    f"[{position}/6] verified {job['logical_configuration']} "
                    f"seed={job['seed']}: {job['run_dir']}",
                    flush=True,
                )
                continue
            _atomic_write_json(state_path, state)
            _assert_invocation_identity(
                commit,
                feature_store,
                configuration,
                stage3_state_path,
                slow_audit_path,
            )
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
                    "output_sha256": None,
                    "feature_ablation_identity_source": None,
                    "producing_git_commit_sha": None,
                    "source_stage3_state": None,
                    "source_stage3_state_sha256": None,
                    "source_stage3_job": None,
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "completed_at_utc": None,
                    "primary_validation_ic": None,
                    "error": None,
                }
            )
            _atomic_write_json(state_path, state)
            print(
                f"[{position}/6] starting {job['logical_configuration']} "
                f"seed={job['seed']}",
                flush=True,
            )
            try:
                result = subprocess.run(job["command"], cwd=_RESEARCH, check=False)
            except OSError as error:
                job.update({"status": "failed", "error": str(error)})
                _atomic_write_json(state_path, state)
                raise RuntimeError(
                    f"Could not start Stage-4 treatment seed={job['seed']}"
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
                raise RuntimeError(f"Stage-4 treatment failed for seed={job['seed']}")
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
                score, hashes, identity_source = validate_stage4_completed_run(
                    created[0],
                    configuration,
                    "drop_slow_low_prior",
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
                    f"Stage-4 completed-run validation failed for seed={job['seed']}"
                ) from error
            job.update(
                {
                    "status": "completed",
                    "result_origin": "trained_stage4",
                    "run_manifest_sha256": hashes["run_manifest.json"],
                    "output_sha256": hashes,
                    "feature_ablation_identity_source": identity_source,
                    "producing_git_commit_sha": commit,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "primary_validation_ic": score,
                    "error": None,
                }
            )
            _atomic_write_json(state_path, state)
            print(
                f"[{position}/6] completed drop_slow_low_prior "
                f"seed={job['seed']} IC={score:.8f}",
                flush=True,
            )
        verified = sum(_complete_or_recover_job(job, configuration) for job in jobs)
        if (
            verified != 6
            or sum(job["result_origin"] == "adopted_stage3" for job in jobs) != 3
            or sum(job["result_origin"] == "trained_stage4" for job in jobs) != 3
        ):
            raise RuntimeError(
                "Stage-4 did not finish with three adopted and three new jobs"
            )
        state["status"] = "completed"
        state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(state_path, state)
    return state_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--stage3-state", type=Path, required=True)
    parser.add_argument("--slow-audit", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage3_state = args.stage3_state.resolve()
    slow_audit = args.slow_audit.resolve()
    if args.dry_run:
        payload = dry_run_payload(stage3_state, slow_audit)
        print(format_dry_run_preflight(payload), flush=True)
        return
    state_path = run_sweep(args.state_dir.resolve(), stage3_state, slow_audit)
    print(f"Stage-4 sweep completed: {state_path}", flush=True)


if __name__ == "__main__":
    main()
