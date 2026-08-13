from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch

from .analyze_stage2_context_ablation import (
    _period_metrics,
    _read_validation_daily_metrics,
    _validate_metrics_json,
)
from .audit_realized_distributions import (
    AUDIT_JSON,
    AUDIT_NAME,
    AUDIT_VERSION,
    FEATURE_PARQUET,
    TARGET_PARQUET,
    run_realized_distribution_audit,
    validate_realized_distribution_audit,
)
from .context_ablation import resolve_context_ablation_for_store
from .context_routing_adaptive import (
    CONDITIONAL_ROUTING_RUN_COUNT_MAXIMUM,
    ISSUER_RUN_COUNT_MAXIMUM,
    ISSUER_RUN_COUNT_MINIMUM,
    MANDATORY_ROUTING_RUN_COUNT,
    ROUTING_RUN_COUNT_MAXIMUM,
    ROUTING_RUN_COUNT_MINIMUM,
    TOTAL_TRAINING_RUN_COUNT_MAXIMUM,
    TOTAL_TRAINING_RUN_COUNT_MINIMUM,
    joint_synthesis_gate,
    seed29_candidate_gate,
    select_candidate,
    three_seed_candidate_gate,
    within_source_combination_gate,
)
from .context_routing_artifacts import (
    create_validated_archive,
    publish_output_pointer,
    sha256_file,
    validate_archive,
    validate_archive_sha256,
    write_archive_sha256,
)
from .contract import (
    ALLOWED_SEEDS,
    CONTEXT_ROUTING_MODES,
    FEATURE_CONTRACT_VERSION,
    PROJECT_ROOT,
    RUN_OUTPUT_BASE,
    VALIDATION_END,
    VALIDATION_START,
    TCNArchitecture,
    TCNSettings,
    architecture_for_model,
    context_routing_metadata,
    expected_trainable_parameter_count,
    peer_feature_metadata,
)
from .data import resolve_feature_store, validate_feature_store
from .routing_identity_preflight import (
    PREFLIGHT_VERSION,
    build_routing_preflight_identity,
    run_routing_identity_preflight,
    validate_routing_identity_preflight,
)
from .engine import objective_metadata, sam_metadata, validate_runtime
from .evaluate import _validate_run_checkpoint_identity
from .feature_ablation import resolve_feature_ablation_for_store
from .process_lock import (
    PRODUCTION_TRAINING_LOCK,
    active_lock_owner,
    exclusive_process_lock,
)
from .stage2_context_ablation import _feature_store_identity
from .stage3_context_addition import (
    _feature_identities_equivalent,
    _reject_test_derived_metadata,
)

EXPERIMENT_NAME = "stage5_context_routing_adaptive_v1"
STATE_VERSION = 2
PEER_PRIMARY_STATE_POINTER = (
    RUN_OUTPUT_BASE / "_ops" / "peer_primary_matrix_current_path.txt"
)
OUTPUT_POINTER = (
    RUN_OUTPUT_BASE / "_ops" / "context_routing_sequence_outputs_current_path.txt"
)
FROZEN_CONTEXT_ABLATION = "drop_win_and_global_non_rates"
FROZEN_FEATURE_ABLATION = "none"
FROZEN_TCN_BASE = {
    "fusion": "context_pooled",
    "width": 64,
    "receptive_field": "full",
    "block": "swiglu",
}
ISSUER_SEED_ORDER = (29, 11, 47)
_REQUIRED_OUTPUTS = (
    "run_manifest.json",
    "best.pt",
    "final.pt",
    "history.csv",
    "validation_metrics.json",
    "validation_daily_metrics.parquet",
)
_REPOSITORY = PROJECT_ROOT / "quant" / "b3-quant"
_RESEARCH = _REPOSITORY / "research"
SUMMARY_JSON = "experiment_summary.json"
SUMMARY_MARKDOWN = "experiment_summary.md"
PLAN_JSON = "plan.json"
PAIRED_RESULTS_JSON = "paired_results.json"
PREFLIGHT_JSON = "routing_identity_preflight.json"
EXPERIMENT_LOG = "experiment.log"
COMPLETED_RUN_PATHS_JSON = "completed_run_paths.json"
ARTIFACT_HASHES_JSON = "artifact_hashes.json"
RUNBOOK = "operator_runbook.md"
ARCHIVE_NAME = "context_routing_sequence_outputs.tar.gz"
ARCHIVE_SHA256_NAME = f"{ARCHIVE_NAME}.sha256"


@dataclass(frozen=True)
class CompletedRun:
    run_dir: Path
    seed: int
    peer_features: str
    slow_routing: str
    macro_temporal_routing: str
    context_routing_experiment: str
    producing_git_commit_sha: str
    primary_ic: float
    metrics: dict[str, object]
    output_sha256: dict[str, str]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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
        raise RuntimeError("Stage-5 execution requires a clean Git worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, clean


def resolve_peer_primary_state(explicit: Path | None = None) -> Path:
    source = PEER_PRIMARY_STATE_POINTER if explicit is None else explicit
    source = source.expanduser().resolve()
    if source.suffix.lower() == ".txt":
        if not source.is_file():
            raise FileNotFoundError(
                f"Peer-primary state pointer does not exist: {source}"
            )
        target = Path(source.read_text(encoding="utf-8").strip()).expanduser()
        if not target.is_absolute():
            target = source.parent / target
        source = target.resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"Authoritative peer-primary state does not exist: {source}"
        )
    return source


def _settings(
    slow_routing: str = "late_only",
    macro_temporal_routing: str = "late_only",
    experiment: str = "legacy",
) -> TCNSettings:
    return TCNSettings(
        **FROZEN_TCN_BASE,
        slow_routing=slow_routing,
        macro_temporal_routing=macro_temporal_routing,
        context_routing_experiment=experiment,
    )


def build_training_command(
    *,
    seed: int,
    peer_features: str,
    slow_routing: str = "late_only",
    macro_temporal_routing: str = "late_only",
    context_routing_experiment: str = "legacy",
) -> tuple[str, ...]:
    if seed not in ALLOWED_SEEDS:
        raise ValueError("Training seed is outside the frozen experiment contract")
    settings = _settings(
        slow_routing, macro_temporal_routing, context_routing_experiment
    )
    architecture_for_model("tcn", settings)
    if peer_features not in {"selected", "selected_plus_issuer"}:
        raise ValueError("Stage-5 supports only selected peer representations")
    if context_routing_experiment == "factorial_v1" and peer_features != "selected":
        raise ValueError("Routing experiments must keep peer_features=selected")
    command = [
        sys.executable,
        "-m",
        "brazil_rv.modeling.train",
        "--model",
        "tcn",
        "--tcn-fusion",
        settings.fusion,
        "--tcn-width",
        str(settings.width),
        "--tcn-receptive-field",
        settings.receptive_field,
        "--tcn-block",
        settings.block,
        "--peer-features",
        peer_features,
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
        FROZEN_FEATURE_ABLATION,
        "--seed",
        str(seed),
    ]
    if context_routing_experiment == "factorial_v1":
        command.extend(
            (
                "--context-routing-experiment",
                context_routing_experiment,
                "--slow-routing",
                slow_routing,
                "--macro-temporal-routing",
                macro_temporal_routing,
            )
        )
    return tuple(command)


def routing_stage(slow_routing: str, macro_temporal_routing: str) -> str:
    if slow_routing == "late_only" and macro_temporal_routing == "late_only":
        raise ValueError("Adaptive routing never trains an all-off scaffold control")
    if slow_routing != "late_only" and macro_temporal_routing == "late_only":
        return "slow_only"
    if slow_routing == "late_only" and macro_temporal_routing != "late_only":
        return "macro_temporal_only"
    return "joint_synthesis"


def _finalize_job_specification(specification: dict[str, object]) -> dict[str, object]:
    serialized = json.dumps(specification, sort_keys=True, separators=(",", ":"))
    return {
        **specification,
        "serialized_job_specification": serialized,
        "job_specification_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
    }


def routing_job(
    slow_routing: str,
    macro_temporal_routing: str,
    seed: int,
    stage: str,
) -> dict[str, object]:
    if (
        slow_routing not in CONTEXT_ROUTING_MODES
        or macro_temporal_routing not in CONTEXT_ROUTING_MODES
        or (slow_routing == "late_only" and macro_temporal_routing == "late_only")
    ):
        raise ValueError("Adaptive routing jobs require at least one enabled route")
    specification = {
        "job_id": (
            f"routing:slow={slow_routing}:macro={macro_temporal_routing}:seed={seed}"
        ),
        "job_kind": "routing",
        "stage": stage,
        "seed": seed,
        "peer_features": "selected",
        "slow_routing": slow_routing,
        "macro_temporal_routing": macro_temporal_routing,
        "context_routing_experiment": "factorial_v1",
        "command": list(
            build_training_command(
                seed=seed,
                peer_features="selected",
                slow_routing=slow_routing,
                macro_temporal_routing=macro_temporal_routing,
                context_routing_experiment="factorial_v1",
            )
        ),
    }
    return _finalize_job_specification(specification)


def routing_jobs() -> tuple[dict[str, object], ...]:
    return (
        routing_job("early_concat", "late_only", 29, "mandatory_seed29_screen"),
        routing_job("film", "late_only", 29, "mandatory_seed29_screen"),
        routing_job("late_only", "early_concat", 29, "mandatory_seed29_screen"),
        routing_job("late_only", "film", 29, "mandatory_seed29_screen"),
    )


def issuer_job(seed: int) -> dict[str, object]:
    if seed not in ISSUER_SEED_ORDER:
        raise ValueError("Issuer job seed is outside the frozen contract")
    specification = {
        "job_id": f"issuer:seed={seed}",
        "job_kind": "issuer",
        "stage": "seed29_screen" if seed == 29 else "matched_seed_confirmation",
        "seed": seed,
        "peer_features": "selected_plus_issuer",
        "slow_routing": "late_only",
        "macro_temporal_routing": "late_only",
        "context_routing_experiment": "legacy",
        "command": list(
            build_training_command(seed=seed, peer_features="selected_plus_issuer")
        ),
    }
    return _finalize_job_specification(specification)


def issuer_jobs() -> tuple[dict[str, object], ...]:
    return tuple(issuer_job(seed) for seed in ISSUER_SEED_ORDER)


def _artifact_hashes(run_dir: Path) -> dict[str, str]:
    hashes = {}
    for name in _REQUIRED_OUTPUTS:
        path = run_dir / name
        if not path.is_file():
            raise ValueError(f"Completed run is missing required artifact: {path}")
        hashes[name] = _sha256(path)
    return hashes


def _manifest_peer_mode(manifest: dict[str, object]) -> str | None:
    peer = manifest.get("peer_features")
    return str(peer.get("mode")) if isinstance(peer, dict) else None


def validate_completed_run(
    run_dir: Path,
    feature_store: Path,
    *,
    seed: int,
    peer_features: str,
    slow_routing: str,
    macro_temporal_routing: str,
    context_routing_experiment: str,
    producing_commit: str | None = None,
) -> CompletedRun:
    run_dir = run_dir.resolve()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    settings = _settings(
        slow_routing, macro_temporal_routing, context_routing_experiment
    )
    architecture = architecture_for_model("tcn", settings)
    if not isinstance(architecture, TCNArchitecture):
        raise AssertionError("Frozen TCN settings did not resolve a TCN architecture")
    expected_context = resolve_context_ablation_for_store(
        feature_store, FROZEN_CONTEXT_ABLATION
    ).metadata()
    expected_feature = resolve_feature_ablation_for_store(
        feature_store, FROZEN_FEATURE_ABLATION
    ).metadata()
    recorded_settings = manifest.get("tcn_settings")
    try:
        normalized_settings = (
            TCNSettings(**recorded_settings)
            if isinstance(recorded_settings, dict)
            else None
        )
    except TypeError as error:
        raise ValueError("Completed run has invalid TCN settings") from error
    expected_commit = manifest.get("git_commit_sha")
    if (
        manifest.get("status") != "completed"
        or manifest.get("model_name") != "tcn"
        or manifest.get("model_family") != "tcn"
        or manifest.get("seed") != seed
        or normalized_settings != settings
        or manifest.get("global_context") != "enabled"
        or manifest.get("context_ablation") != expected_context
        or manifest.get("feature_ablation") != expected_feature
        or _manifest_peer_mode(manifest) != peer_features
        or manifest.get("peer_features")
        != peer_feature_metadata("tcn", architecture, peer_features)
        or manifest.get("objective") != objective_metadata("soft_spearman", 0.50)
        or manifest.get("optimizer_variant") != "sam_adamw"
        or manifest.get("sam") != sam_metadata("sam_adamw", 0.125)
        or manifest.get("parameter_count")
        != expected_trainable_parameter_count("tcn", architecture, peer_features)
        or not isinstance(expected_commit, str)
        or (producing_commit is not None and expected_commit != producing_commit)
    ):
        raise ValueError(
            f"Completed run violates the frozen Stage-5 identity: {run_dir}"
        )
    if context_routing_experiment == "factorial_v1":
        if manifest.get("context_routing") != context_routing_metadata(architecture):
            raise ValueError("Factorial run has invalid routing metadata")
    else:
        recorded_routing = manifest.get("context_routing")
        if (
            recorded_routing is not None
            and recorded_routing != context_routing_metadata(architecture)
        ):
            raise ValueError("Legacy run has invalid routing metadata")
    recorded_store = (
        Path(str(manifest["resolved_feature_store_path"])).expanduser().resolve()
    )
    try:
        same_store = recorded_store.samefile(feature_store)
    except OSError:
        same_store = recorded_store == feature_store
    if not same_store:
        raise ValueError("Completed run uses another resolved feature store")

    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    final = torch.load(run_dir / "final.pt", map_location="cpu", weights_only=False)
    if not isinstance(best, dict) or not isinstance(final, dict):
        raise ValueError("Completed run checkpoints are malformed")
    _validate_run_checkpoint_identity(manifest, best, feature_store, run_dir)
    _validate_run_checkpoint_identity(manifest, final, feature_store, run_dir)
    daily = _read_validation_daily_metrics(run_dir)
    metrics = _period_metrics(daily, VALIDATION_START, VALIDATION_END)
    _validate_metrics_json(run_dir, metrics)
    primary = float(metrics["primary_ic"])
    if not math.isclose(
        primary,
        float(manifest["best_validation_primary_score"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Completed run manifest and validation metrics disagree")
    return CompletedRun(
        run_dir=run_dir,
        seed=seed,
        peer_features=peer_features,
        slow_routing=slow_routing,
        macro_temporal_routing=macro_temporal_routing,
        context_routing_experiment=context_routing_experiment,
        producing_git_commit_sha=str(expected_commit),
        primary_ic=primary,
        metrics=metrics,
        output_sha256=_artifact_hashes(run_dir),
    )


def _source_incumbents(
    state_path: Path,
    feature_store: Path,
) -> dict[int, tuple[CompletedRun, dict[str, object]]]:
    raw = state_path.read_bytes()
    state = json.loads(raw)
    _reject_test_derived_metadata(state, "peer-primary source state")
    if state.get("status") != "completed" or not isinstance(state.get("jobs"), list):
        raise ValueError(
            "Peer-primary adoption requires a completed authoritative state"
        )
    candidates: dict[int, list[tuple[CompletedRun, dict[str, object]]]] = {
        seed: [] for seed in ALLOWED_SEEDS
    }
    for position, job in enumerate(state["jobs"]):
        if not isinstance(job, dict) or job.get("status") != "completed":
            continue
        run_value = job.get("run_dir")
        if not isinstance(run_value, str):
            continue
        run_dir = Path(run_value).expanduser().resolve()
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        seed = manifest.get("seed")
        if seed not in candidates or _manifest_peer_mode(manifest) != "selected":
            continue
        try:
            completed = validate_completed_run(
                run_dir,
                feature_store,
                seed=int(seed),
                peer_features="selected",
                slow_routing="late_only",
                macro_temporal_routing="late_only",
                context_routing_experiment="legacy",
            )
        except (KeyError, OSError, TypeError, ValueError):
            continue
        manifest_hash = completed.output_sha256["run_manifest.json"]
        if (
            job.get("run_manifest_sha256") is not None
            and job.get("run_manifest_sha256") != manifest_hash
        ):
            raise ValueError("Authoritative peer-primary job manifest hash disagrees")
        candidates[int(seed)].append(
            (
                completed,
                {
                    "position": position,
                    "run_dir": str(run_dir),
                    "run_manifest_sha256": manifest_hash,
                    "source_job_identity": {
                        key: job.get(key)
                        for key in (
                            "logical_configuration",
                            "configuration",
                            "seed",
                            "result_origin",
                            "producing_git_commit_sha",
                        )
                        if key in job
                    },
                },
            )
        )
    if any(len(values) != 1 for values in candidates.values()):
        counts = {seed: len(values) for seed, values in candidates.items()}
        raise ValueError(
            "Authoritative peer-primary state must identify exactly one frozen "
            f"selected incumbent per seed: {counts}"
        )
    return {seed: values[0] for seed, values in candidates.items()}


def issuer_seed29_gate(
    incumbent: CompletedRun,
    issuer: CompletedRun,
) -> dict[str, object]:
    if incumbent.seed != 29 or issuer.seed != 29:
        raise ValueError("Issuer first-stage gate requires the matched seed 29 pair")
    return seed29_candidate_gate(incumbent.metrics, issuer.metrics)


def issuer_three_seed_gate(
    incumbents: dict[int, CompletedRun],
    issuers: dict[int, CompletedRun],
) -> dict[str, object]:
    return three_seed_candidate_gate(
        {seed: run.metrics for seed, run in incumbents.items()},
        {seed: run.metrics for seed, run in issuers.items()},
    )


def routing_seed29_gate(
    incumbent: CompletedRun,
    candidate: CompletedRun,
) -> dict[str, object]:
    if incumbent.seed != 29 or candidate.seed != 29:
        raise ValueError("Routing first-stage gate requires a matched seed-29 pair")
    return seed29_candidate_gate(incumbent.metrics, candidate.metrics)


def routing_three_seed_gate(
    incumbents: dict[int, CompletedRun],
    candidates: dict[int, CompletedRun],
) -> dict[str, object]:
    return three_seed_candidate_gate(
        {seed: run.metrics for seed, run in incumbents.items()},
        {seed: run.metrics for seed, run in candidates.items()},
    )


def _selection_candidate(
    run: CompletedRun, gate: dict[str, object]
) -> dict[str, object]:
    return {
        "identity": (f"slow={run.slow_routing}|macro={run.macro_temporal_routing}"),
        "slow_routing": run.slow_routing,
        "macro_temporal_routing": run.macro_temporal_routing,
        "gate": gate,
    }


def _peer_primary_provenance(peer_primary_state: Path) -> dict[str, object]:
    provenance: dict[str, object] = {
        "resolved_state_path": str(peer_primary_state),
        "resolved_state_sha256": _sha256(peer_primary_state),
        "authoritative_pointer_path": str(PEER_PRIMARY_STATE_POINTER),
    }
    if PEER_PRIMARY_STATE_POINTER.is_file():
        pointer_target = resolve_peer_primary_state()
        provenance.update(
            {
                "authoritative_pointer_sha256": _sha256(PEER_PRIMARY_STATE_POINTER),
                "authoritative_pointer_target": str(pointer_target),
                "explicit_cli_override": not pointer_target.samefile(
                    peer_primary_state
                ),
            }
        )
    else:
        provenance.update(
            {
                "authoritative_pointer_sha256": None,
                "authoritative_pointer_target": None,
                "explicit_cli_override": True,
            }
        )
    return provenance


def _configuration(
    commit: str,
    feature_store: Path,
    peer_primary_state: Path,
) -> dict[str, object]:
    legacy = architecture_for_model("tcn", _settings())
    factorial = architecture_for_model(
        "tcn", _settings("late_only", "late_only", "factorial_v1")
    )
    return {
        "orchestrator_git_commit_sha": commit,
        "feature_store": _feature_store_identity(feature_store),
        "feature_contract": FEATURE_CONTRACT_VERSION,
        "source_peer_primary": _peer_primary_provenance(peer_primary_state),
        "context_ablation": resolve_context_ablation_for_store(
            feature_store, FROZEN_CONTEXT_ABLATION
        ).metadata(),
        "feature_ablation": resolve_feature_ablation_for_store(
            feature_store, FROZEN_FEATURE_ABLATION
        ).metadata(),
        "legacy_incumbent": {
            "tcn_settings": asdict(_settings()),
            "architecture": asdict(legacy),
            "peer_features": peer_feature_metadata("tcn", legacy, "selected"),
            "parameter_count": expected_trainable_parameter_count(
                "tcn", legacy, "selected"
            ),
        },
        "factorial_scaffold": {
            "schema": context_routing_metadata(factorial),
            "parameter_count": expected_trainable_parameter_count(
                "tcn", factorial, "selected"
            ),
            "state_dict_structure_equal_across_arms": True,
        },
        "seeds": list(ALLOWED_SEEDS),
        "identity_preflight": {
            "version": PREFLIGHT_VERSION,
            "execution_modes": ["eager", "compiled"],
            "deterministic_sam_steps": 3,
            "all_off_scaffold_control_training": False,
        },
        "issuer_run_count_minimum": ISSUER_RUN_COUNT_MINIMUM,
        "issuer_run_count_maximum": ISSUER_RUN_COUNT_MAXIMUM,
        "routing_run_count_minimum": ROUTING_RUN_COUNT_MINIMUM,
        "routing_run_count_maximum": ROUTING_RUN_COUNT_MAXIMUM,
        "total_training_run_count_minimum": TOTAL_TRAINING_RUN_COUNT_MINIMUM,
        "total_training_run_count_maximum": TOTAL_TRAINING_RUN_COUNT_MAXIMUM,
        "final_holdout_status": "sealed_not_accessed",
    }


def _new_job(base: dict[str, object]) -> dict[str, object]:
    return {
        **base,
        "status": "pending",
        "run_dir": None,
        "run_manifest_sha256": None,
        "output_sha256": None,
        "producing_git_commit_sha": None,
        "started_at_utc": None,
        "completed_at_utc": None,
        "primary_validation_ic": None,
        "context_routing": None,
        "recovery_count": 0,
        "last_recovery_at_utc": None,
        "error": None,
    }


def _new_state(
    configuration: dict[str, object],
    incumbents: dict[int, tuple[CompletedRun, dict[str, object]]],
    audit_path: Path,
    preflight_path: Path,
) -> dict[str, object]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "state_version": STATE_VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "status": "running",
        "configuration": configuration,
        "created_at_utc": now,
        "completed_at_utc": None,
        "routing_identity_preflight": {
            "path": str(preflight_path),
            "sha256": _sha256(preflight_path),
            "version": PREFLIGHT_VERSION,
            "status": "passed",
        },
        "realized_distribution_audit": {
            "path": str(audit_path.parent),
            "audit_json": str(audit_path),
            "sha256": _sha256(audit_path),
            "audit_name": AUDIT_NAME,
            "audit_version": AUDIT_VERSION,
        },
        "incumbent_runs": {
            str(seed): {
                "run_dir": str(run.run_dir),
                "primary_validation_ic": run.primary_ic,
                "producing_git_commit_sha": run.producing_git_commit_sha,
                "output_sha256": run.output_sha256,
                "source_provenance": provenance,
            }
            for seed, (run, provenance) in incumbents.items()
        },
        "issuer_jobs": [],
        "routing_jobs": [],
        "decisions": [],
        "issuer_screen": None,
        "routing_summary": None,
        "output_contract": {
            "archive": ARCHIVE_NAME,
            "archive_sha256": ARCHIVE_SHA256_NAME,
            "output_pointer": str(OUTPUT_POINTER),
        },
    }


def _configurations_match(
    stored: dict[str, object], current: dict[str, object]
) -> bool:
    left = dict(stored)
    right = dict(current)
    left_feature = left.pop("feature_store", None)
    right_feature = right.pop("feature_store", None)
    return (
        left == right
        and isinstance(left_feature, dict)
        and isinstance(right_feature, dict)
        and _feature_identities_equivalent(left_feature, right_feature)
    )


def _validate_dynamic_jobs(recorded: object, job_kind: str) -> list[dict[str, object]]:
    if not isinstance(recorded, list):
        raise ValueError("Adaptive job state must be a list")
    result: list[dict[str, object]] = []
    identities: set[str] = set()
    statuses = {"pending", "running", "failed", "completed"}
    immutable = (
        "job_id",
        "job_kind",
        "stage",
        "seed",
        "peer_features",
        "slow_routing",
        "macro_temporal_routing",
        "context_routing_experiment",
        "command",
        "serialized_job_specification",
        "job_specification_sha256",
    )
    for job in recorded:
        if not isinstance(job, dict) or job.get("job_kind") != job_kind:
            raise ValueError("Adaptive job state contains a malformed job")
        if job_kind == "issuer":
            expected = issuer_job(int(job["seed"]))
        else:
            expected = routing_job(
                str(job["slow_routing"]),
                str(job["macro_temporal_routing"]),
                int(job["seed"]),
                str(job["stage"]),
            )
        recovery = job.get("recovery_count")
        identity = str(job.get("job_id"))
        if (
            job.get("status") not in statuses
            or any(job.get(field) != expected[field] for field in immutable)
            or not isinstance(recovery, int)
            or isinstance(recovery, bool)
            or recovery < 0
            or identity in identities
        ):
            raise ValueError("Adaptive job identity drifted")
        identities.add(identity)
        result.append(job)
    return result


def _load_state(
    state_path: Path,
    configuration: dict[str, object],
    incumbents: dict[int, tuple[CompletedRun, dict[str, object]]],
    audit_path: Path,
    preflight_path: Path,
) -> dict[str, object]:
    if not state_path.exists():
        return _new_state(configuration, incumbents, audit_path, preflight_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    _reject_test_derived_metadata(state, "Stage-5 state")
    if (
        state.get("state_version") != STATE_VERSION
        or state.get("experiment_name") != EXPERIMENT_NAME
        or state.get("status") not in {"running", "completed"}
        or not isinstance(state.get("configuration"), dict)
        or not _configurations_match(state["configuration"], configuration)
    ):
        raise ValueError("Stage-5 state has an incompatible identity")
    _validate_dynamic_jobs(state.get("issuer_jobs"), "issuer")
    _validate_dynamic_jobs(state.get("routing_jobs"), "routing")
    decisions = state.get("decisions")
    if (
        not isinstance(decisions, list)
        or any(not isinstance(item, dict) for item in decisions)
        or len({item.get("decision_id") for item in decisions}) != len(decisions)
    ):
        raise ValueError("Stage-5 adaptive decisions are malformed or duplicated")
    audit = state.get("realized_distribution_audit")
    if (
        not isinstance(audit, dict)
        or audit.get("sha256") != _sha256(audit_path)
        or audit.get("audit_name") != AUDIT_NAME
        or audit.get("audit_version") != AUDIT_VERSION
    ):
        raise ValueError("Stage-5 audit provenance changed")
    preflight = state.get("routing_identity_preflight")
    if (
        not isinstance(preflight, dict)
        or preflight.get("sha256") != _sha256(preflight_path)
        or preflight.get("version") != PREFLIGHT_VERSION
        or preflight.get("status") != "passed"
    ):
        raise ValueError("Stage-5 routing identity preflight provenance changed")
    for seed, (run, provenance) in incumbents.items():
        recorded = state.get("incumbent_runs", {}).get(str(seed))
        expected = {
            "run_dir": str(run.run_dir),
            "primary_validation_ic": run.primary_ic,
            "producing_git_commit_sha": run.producing_git_commit_sha,
            "output_sha256": run.output_sha256,
            "source_provenance": provenance,
        }
        if recorded != expected:
            raise ValueError(f"Stage-5 incumbent provenance changed for seed {seed}")
    return state


def _ensure_job(
    state: dict[str, object], specification: dict[str, object]
) -> dict[str, object]:
    key = "issuer_jobs" if specification["job_kind"] == "issuer" else "routing_jobs"
    jobs = state[key]
    if not isinstance(jobs, list):
        raise ValueError("Adaptive job collection is malformed")
    matches = [job for job in jobs if job.get("job_id") == specification["job_id"]]
    if len(matches) > 1:
        raise ValueError("Adaptive job identity is duplicated")
    if matches:
        job = matches[0]
        for field, value in specification.items():
            if job.get(field) != value:
                raise ValueError("Persisted adaptive job specification drifted")
        return job
    job = _new_job(specification)
    jobs.append(job)
    return job


def _record_decision(
    state: dict[str, object], decision_id: str, payload: dict[str, object]
) -> dict[str, object]:
    record = {"decision_id": decision_id, **payload}
    decisions = state["decisions"]
    if not isinstance(decisions, list):
        raise ValueError("Adaptive decision state is malformed")
    matches = [item for item in decisions if item.get("decision_id") == decision_id]
    if len(matches) > 1:
        raise ValueError("Adaptive decision identity is duplicated")
    if matches:
        if matches[0] != record:
            raise ValueError(f"Adaptive decision changed on restart: {decision_id}")
        return matches[0]
    decisions.append(record)
    return record


def _plan_payload(state: dict[str, object]) -> dict[str, object]:
    return {
        "experiment_name": EXPERIMENT_NAME,
        "state_version": STATE_VERSION,
        "status": state["status"],
        "precommitted_adaptive_sequence": {
            "mandatory_seed29": [
                "slow_early_concat_only",
                "slow_film_only",
                "macro_temporal_early_concat_only",
                "macro_temporal_film_only",
            ],
            "within_source_combination": "only_if_both_individual_routes_pass",
            "joint_synthesis": "only_if_each_source_has_an_eligible_selection",
            "confirmation_seeds": [11, 47],
            "confirmation_scope": "single_selected_candidate_only",
            "all_off_scaffold_control_training": False,
        },
        "run_counts": {
            "issuer_minimum": ISSUER_RUN_COUNT_MINIMUM,
            "issuer_maximum": ISSUER_RUN_COUNT_MAXIMUM,
            "routing_mandatory": MANDATORY_ROUTING_RUN_COUNT,
            "routing_conditional_maximum": CONDITIONAL_ROUTING_RUN_COUNT_MAXIMUM,
            "routing_minimum": ROUTING_RUN_COUNT_MINIMUM,
            "routing_maximum": ROUTING_RUN_COUNT_MAXIMUM,
            "total_minimum": TOTAL_TRAINING_RUN_COUNT_MINIMUM,
            "total_maximum": TOTAL_TRAINING_RUN_COUNT_MAXIMUM,
        },
        "jobs": [*state["issuer_jobs"], *state["routing_jobs"]],
        "decisions": state["decisions"],
        "final_holdout_status": "sealed_not_accessed",
    }


def _paired_results_payload(state: dict[str, object]) -> dict[str, object]:
    routing_jobs = []
    for job in state["routing_jobs"]:
        routing_jobs.append(
            {
                key: job.get(key)
                for key in (
                    "job_id",
                    "stage",
                    "seed",
                    "slow_routing",
                    "macro_temporal_routing",
                    "status",
                    "run_dir",
                    "primary_validation_ic",
                    "output_sha256",
                    "context_routing",
                )
            }
        )
    return {
        "experiment_name": EXPERIMENT_NAME,
        "status": state["status"],
        "issuer_screen": state["issuer_screen"],
        "adaptive_decisions": state["decisions"],
        "routing_runs": routing_jobs,
        "routing_summary": state["routing_summary"],
        "final_holdout_status": "sealed_not_accessed",
        "transaction_cost_modeling": False,
    }


def _persist_state(state_path: Path, state: dict[str, object]) -> None:
    _atomic_write_json(state_path, state)
    _atomic_write_json(state_path.parent / PLAN_JSON, _plan_payload(state))
    _atomic_write_json(
        state_path.parent / PAIRED_RESULTS_JSON, _paired_results_payload(state)
    )


def _log(state_dir: Path, message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    with (state_dir / EXPERIMENT_LOG).open("a", encoding="utf-8") as output:
        output.write(f"{timestamp} {message}\n")
        output.flush()
        os.fsync(output.fileno())


def _production_run_directories() -> set[Path]:
    if not RUN_OUTPUT_BASE.is_dir():
        return set()
    return {
        path.resolve()
        for path in RUN_OUTPUT_BASE.iterdir()
        if path.is_dir() and path.name != "_ops"
    }


def _candidate_runs(
    job: dict[str, object],
    feature_store: Path,
    commit: str,
) -> tuple[CompletedRun, ...]:
    candidates = set()
    if isinstance(job.get("run_dir"), str):
        recorded = Path(str(job["run_dir"]))
        if recorded.is_dir():
            candidates.add(recorded.resolve())
    for run_dir in _production_run_directories():
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        settings = manifest.get("tcn_settings")
        if (
            manifest.get("git_commit_sha") == commit
            and manifest.get("seed") == job["seed"]
            and _manifest_peer_mode(manifest) == job["peer_features"]
            and isinstance(settings, dict)
            and settings.get("slow_routing", "late_only") == job["slow_routing"]
            and settings.get("macro_temporal_routing", "late_only")
            == job["macro_temporal_routing"]
            and settings.get("context_routing_experiment", "legacy")
            == job["context_routing_experiment"]
        ):
            candidates.add(run_dir)
    completed = []
    for run_dir in sorted(candidates):
        try:
            completed.append(
                validate_completed_run(
                    run_dir,
                    feature_store,
                    seed=int(job["seed"]),
                    peer_features=str(job["peer_features"]),
                    slow_routing=str(job["slow_routing"]),
                    macro_temporal_routing=str(job["macro_temporal_routing"]),
                    context_routing_experiment=str(job["context_routing_experiment"]),
                    producing_commit=commit,
                )
            )
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
            continue
    return tuple(completed)


def _bind_completed_job(
    job: dict[str, object],
    run: CompletedRun,
    commit: str,
) -> None:
    routing_metadata = None
    if job["job_kind"] == "routing":
        architecture = architecture_for_model(
            "tcn",
            _settings(
                str(job["slow_routing"]),
                str(job["macro_temporal_routing"]),
                "factorial_v1",
            ),
        )
        routing_metadata = context_routing_metadata(architecture)
    job.update(
        {
            "status": "completed",
            "run_dir": str(run.run_dir),
            "run_manifest_sha256": run.output_sha256["run_manifest.json"],
            "output_sha256": run.output_sha256,
            "producing_git_commit_sha": commit,
            "completed_at_utc": job.get("completed_at_utc")
            or datetime.now(timezone.utc).isoformat(),
            "primary_validation_ic": run.primary_ic,
            "context_routing": routing_metadata,
            "error": None,
        }
    )


def _recover_job(
    job: dict[str, object],
    feature_store: Path,
    commit: str,
) -> CompletedRun | None:
    candidates = _candidate_runs(job, feature_store, commit)
    status = job["status"]
    if status == "completed":
        if len(candidates) != 1:
            raise ValueError("Completed Stage-5 job has missing or ambiguous artifacts")
        run = candidates[0]
        if (
            job.get("run_manifest_sha256") != run.output_sha256["run_manifest.json"]
            or job.get("output_sha256") != run.output_sha256
            or job.get("producing_git_commit_sha") != commit
            or not math.isclose(
                float(job["primary_validation_ic"]),
                run.primary_ic,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("Completed Stage-5 job state disagrees with artifacts")
        return run
    if status == "pending" and candidates:
        raise ValueError("Unbound completed run contaminates a pending Stage-5 job")
    if status == "failed" and candidates:
        raise ValueError("Failed Stage-5 job has an unbound completed artifact")
    if status == "running":
        if len(candidates) > 1:
            raise ValueError("Multiple completed runs match a recovering Stage-5 job")
        job["recovery_count"] = int(job["recovery_count"]) + 1
        job["last_recovery_at_utc"] = datetime.now(timezone.utc).isoformat()
        if candidates:
            _bind_completed_job(job, candidates[0], commit)
            return candidates[0]
    return None


def _launch_job(
    job: dict[str, object],
    state: dict[str, object],
    state_path: Path,
    feature_store: Path,
    commit: str,
) -> CompletedRun:
    recovered = _recover_job(job, feature_store, commit)
    if recovered is not None:
        _persist_state(state_path, state)
        return recovered
    if owner := active_lock_owner(PRODUCTION_TRAINING_LOCK):
        raise RuntimeError(f"Another production training run is active: {owner}")
    before = _production_run_directories()
    job.update(
        {
            "status": "running",
            "run_dir": None,
            "run_manifest_sha256": None,
            "output_sha256": None,
            "producing_git_commit_sha": None,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_at_utc": None,
            "primary_validation_ic": None,
            "context_routing": None,
            "error": None,
        }
    )
    _persist_state(state_path, state)
    _log(state_path.parent, f"launch {job['job_id']}")
    try:
        result = subprocess.run(job["command"], cwd=_RESEARCH, check=False)
    except OSError as error:
        job.update({"status": "failed", "error": str(error)})
        _persist_state(state_path, state)
        raise RuntimeError("Could not start Stage-5 training") from error
    created = tuple(
        sorted(path for path in _production_run_directories() if path not in before)
    )
    if len(created) == 1:
        job["run_dir"] = str(created[0])
    if result.returncode != 0 or len(created) != 1:
        job.update(
            {
                "status": "failed",
                "error": (
                    f"training exited with code {result.returncode}; "
                    f"new run directory count={len(created)}"
                ),
            }
        )
        _persist_state(state_path, state)
        raise RuntimeError(str(job["error"]))
    run = validate_completed_run(
        created[0],
        feature_store,
        seed=int(job["seed"]),
        peer_features=str(job["peer_features"]),
        slow_routing=str(job["slow_routing"]),
        macro_temporal_routing=str(job["macro_temporal_routing"]),
        context_routing_experiment=str(job["context_routing_experiment"]),
        producing_commit=commit,
    )
    _bind_completed_job(job, run, commit)
    _persist_state(state_path, state)
    _log(state_path.parent, f"complete {job['job_id']} primary_ic={run.primary_ic:.8f}")
    return run


def _ensure_preflight(state_dir: Path, expected_identity: dict[str, object]) -> Path:
    path = state_dir / PREFLIGHT_JSON
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        _reject_test_derived_metadata(payload, "routing identity preflight")
        validate_routing_identity_preflight(
            payload,
            expected_identity,
            asdict(validate_runtime()),
        )
        return path
    run_routing_identity_preflight(path, expected_identity)
    return path


def _ensure_audit(
    state_dir: Path,
    feature_store: Path,
    feature_identity: dict[str, object],
) -> Path:
    final_dir = state_dir / "realized_distribution_audit"
    final = final_dir / AUDIT_JSON
    if final.is_file():
        validate_realized_distribution_audit(final, feature_identity)
        return final
    if final_dir.exists():
        raise ValueError(
            "Incomplete realized-distribution audit directory requires operator inspection"
        )
    partial = state_dir / f"realized_distribution_audit.partial.{uuid4().hex}"
    audit_path = run_realized_distribution_audit(feature_store, partial)
    validate_realized_distribution_audit(audit_path, feature_identity)
    os.replace(partial, final_dir)
    return final


def _assert_invocation_identity(
    commit: str,
    feature_store: Path,
    configuration: dict[str, object],
    peer_primary_state: Path,
    audit_path: Path,
    preflight_path: Path,
    state: dict[str, object],
) -> None:
    current_commit, _ = _git_identity(require_clean=True)
    current_store = resolve_feature_store().resolve()
    current_identity = _feature_store_identity(current_store)
    source = configuration["source_peer_primary"]
    if not isinstance(source, dict):
        raise ValueError("Peer-primary provenance is malformed")
    if (
        current_commit != commit
        or not _feature_identities_equivalent(
            current_identity, configuration["feature_store"]
        )
        or not feature_store.samefile(current_store)
        or _sha256(peer_primary_state) != source["resolved_state_sha256"]
        or _sha256(audit_path) != state["realized_distribution_audit"]["sha256"]
        or _sha256(preflight_path) != state["routing_identity_preflight"]["sha256"]
    ):
        raise RuntimeError(
            "Git, incumbent source, audit, preflight, or feature store changed mid-run"
        )


def _run_authorized_job(
    state: dict[str, object],
    state_path: Path,
    specification: dict[str, object],
    feature_store: Path,
    commit: str,
    peer_primary_state: Path,
    audit_path: Path,
    preflight_path: Path,
) -> CompletedRun:
    job = _ensure_job(state, specification)
    _persist_state(state_path, state)
    _assert_invocation_identity(
        commit,
        feature_store,
        state["configuration"],
        peer_primary_state,
        audit_path,
        preflight_path,
        state,
    )
    return _launch_job(job, state, state_path, feature_store, commit)


def _routing_key(run: CompletedRun) -> tuple[str, str, int]:
    return (run.slow_routing, run.macro_temporal_routing, run.seed)


def _candidate_for_selection(
    candidates: list[dict[str, object]], identity: str
) -> dict[str, object]:
    matches = [
        candidate for candidate in candidates if candidate["identity"] == identity
    ]
    if len(matches) != 1:
        raise ValueError(f"Selected adaptive candidate is unavailable: {identity}")
    return matches[0]


def _run_adaptive_sequence(
    state: dict[str, object],
    state_path: Path,
    feature_store: Path,
    commit: str,
    peer_primary_state: Path,
    audit_path: Path,
    preflight_path: Path,
    incumbents: dict[int, CompletedRun],
) -> None:
    issuer_runs: dict[int, CompletedRun] = {}
    issuer_runs[29] = _run_authorized_job(
        state,
        state_path,
        issuer_job(29),
        feature_store,
        commit,
        peer_primary_state,
        audit_path,
        preflight_path,
    )
    issuer_seed29 = issuer_seed29_gate(incumbents[29], issuer_runs[29])
    state["issuer_screen"] = {"seed29": issuer_seed29, "three_seed": None}
    _record_decision(
        state,
        "issuer_seed29_confirmation_eligibility",
        {
            "decision": "run" if issuer_seed29["passed"] else "skip",
            "reason": (
                "seed29_issuer_gate_passed"
                if issuer_seed29["passed"]
                else "seed29_issuer_gate_failed"
            ),
            "gate": issuer_seed29,
        },
    )
    if issuer_seed29["passed"]:
        for seed in (11, 47):
            issuer_runs[seed] = _run_authorized_job(
                state,
                state_path,
                issuer_job(seed),
                feature_store,
                commit,
                peer_primary_state,
                audit_path,
                preflight_path,
            )
        state["issuer_screen"]["three_seed"] = issuer_three_seed_gate(
            incumbents, issuer_runs
        )
    else:
        for seed in (11, 47):
            _record_decision(
                state,
                f"issuer_confirmation_seed_{seed}",
                {
                    "decision": "skip",
                    "reason": "seed29_issuer_gate_failed",
                    "seed": seed,
                },
            )
    _persist_state(state_path, state)

    routing_runs: dict[tuple[str, str, int], CompletedRun] = {}
    gates: dict[tuple[str, str], dict[str, object]] = {}
    for specification in routing_jobs():
        run = _run_authorized_job(
            state,
            state_path,
            specification,
            feature_store,
            commit,
            peer_primary_state,
            audit_path,
            preflight_path,
        )
        key = _routing_key(run)
        routing_runs[key] = run
        gate = routing_seed29_gate(incumbents[29], run)
        gates[key[:2]] = gate
        _record_decision(
            state,
            f"routing_seed29_gate:slow={key[0]}:macro={key[1]}",
            {
                "decision": "eligible" if gate["passed"] else "ineligible",
                "reason": "seed29_gate_passed"
                if gate["passed"]
                else "seed29_gate_failed",
                "slow_routing": key[0],
                "macro_temporal_routing": key[1],
                "gate": gate,
            },
        )
        _persist_state(state_path, state)

    slow_candidates = [
        _selection_candidate(
            routing_runs[(route, "late_only", 29)], gates[(route, "late_only")]
        )
        for route in ("early_concat", "film")
    ]
    slow_combination = within_source_combination_gate(
        gates[("early_concat", "late_only")], gates[("film", "late_only")]
    )
    _record_decision(state, "slow_within_source_combination", slow_combination)
    if slow_combination["should_run"]:
        run = _run_authorized_job(
            state,
            state_path,
            routing_job(
                "early_concat_film",
                "late_only",
                29,
                "conditional_within_source",
            ),
            feature_store,
            commit,
            peer_primary_state,
            audit_path,
            preflight_path,
        )
        routing_runs[_routing_key(run)] = run
        gate = routing_seed29_gate(incumbents[29], run)
        gates[(run.slow_routing, run.macro_temporal_routing)] = gate
        slow_candidates.append(_selection_candidate(run, gate))
        _record_decision(
            state,
            "routing_seed29_gate:slow=early_concat_film:macro=late_only",
            {
                "decision": "eligible" if gate["passed"] else "ineligible",
                "reason": "seed29_gate_passed"
                if gate["passed"]
                else "seed29_gate_failed",
                "slow_routing": run.slow_routing,
                "macro_temporal_routing": run.macro_temporal_routing,
                "gate": gate,
            },
        )
    slow_selection = select_candidate(slow_candidates)
    _record_decision(state, "slow_source_selection", slow_selection)
    _persist_state(state_path, state)

    macro_candidates = [
        _selection_candidate(
            routing_runs[("late_only", route, 29)], gates[("late_only", route)]
        )
        for route in ("early_concat", "film")
    ]
    macro_combination = within_source_combination_gate(
        gates[("late_only", "early_concat")], gates[("late_only", "film")]
    )
    _record_decision(
        state, "macro_temporal_within_source_combination", macro_combination
    )
    if macro_combination["should_run"]:
        run = _run_authorized_job(
            state,
            state_path,
            routing_job(
                "late_only",
                "early_concat_film",
                29,
                "conditional_within_source",
            ),
            feature_store,
            commit,
            peer_primary_state,
            audit_path,
            preflight_path,
        )
        routing_runs[_routing_key(run)] = run
        gate = routing_seed29_gate(incumbents[29], run)
        gates[(run.slow_routing, run.macro_temporal_routing)] = gate
        macro_candidates.append(_selection_candidate(run, gate))
        _record_decision(
            state,
            "routing_seed29_gate:slow=late_only:macro=early_concat_film",
            {
                "decision": "eligible" if gate["passed"] else "ineligible",
                "reason": "seed29_gate_passed"
                if gate["passed"]
                else "seed29_gate_failed",
                "slow_routing": run.slow_routing,
                "macro_temporal_routing": run.macro_temporal_routing,
                "gate": gate,
            },
        )
    macro_selection = select_candidate(macro_candidates)
    _record_decision(state, "macro_temporal_source_selection", macro_selection)
    _persist_state(state_path, state)

    joint_decision = joint_synthesis_gate(slow_selection, macro_selection)
    _record_decision(state, "joint_synthesis_eligibility", joint_decision)
    joint_candidate = None
    if joint_decision["should_run"]:
        selected_slow = slow_selection["selected"]
        selected_macro = macro_selection["selected"]
        run = _run_authorized_job(
            state,
            state_path,
            routing_job(
                str(selected_slow["slow_routing"]),
                str(selected_macro["macro_temporal_routing"]),
                29,
                "conditional_joint_synthesis",
            ),
            feature_store,
            commit,
            peer_primary_state,
            audit_path,
            preflight_path,
        )
        routing_runs[_routing_key(run)] = run
        gate = routing_seed29_gate(incumbents[29], run)
        gates[(run.slow_routing, run.macro_temporal_routing)] = gate
        joint_candidate = _selection_candidate(run, gate)
        _record_decision(
            state,
            f"routing_seed29_gate:slow={run.slow_routing}:macro={run.macro_temporal_routing}",
            {
                "decision": "eligible" if gate["passed"] else "ineligible",
                "reason": "seed29_gate_passed"
                if gate["passed"]
                else "seed29_gate_failed",
                "slow_routing": run.slow_routing,
                "macro_temporal_routing": run.macro_temporal_routing,
                "gate": gate,
            },
        )

    final_candidates = []
    for selection, candidates in (
        (slow_selection, slow_candidates),
        (macro_selection, macro_candidates),
    ):
        if selection["selected"] is not None:
            final_candidates.append(
                _candidate_for_selection(candidates, selection["selected"]["identity"])
            )
    if joint_candidate is not None:
        final_candidates.append(joint_candidate)
    final_selection = select_candidate(final_candidates)
    _record_decision(state, "final_routing_candidate_selection", final_selection)
    confirmation = None
    selected = final_selection["selected"]
    if selected is None:
        for seed in (11, 47):
            _record_decision(
                state,
                f"routing_confirmation_seed_{seed}",
                {
                    "decision": "skip",
                    "reason": "no_seed29_routing_candidate_eligible",
                    "seed": seed,
                },
            )
    else:
        selected_key = (
            str(selected["slow_routing"]),
            str(selected["macro_temporal_routing"]),
            29,
        )
        selected_runs = {29: routing_runs[selected_key]}
        for seed in (11, 47):
            selected_runs[seed] = _run_authorized_job(
                state,
                state_path,
                routing_job(
                    selected_key[0],
                    selected_key[1],
                    seed,
                    "selected_candidate_confirmation",
                ),
                feature_store,
                commit,
                peer_primary_state,
                audit_path,
                preflight_path,
            )
        confirmation = routing_three_seed_gate(incumbents, selected_runs)
        _record_decision(
            state,
            "selected_routing_candidate_confirmation",
            {
                "decision": "confirmed" if confirmation["passed"] else "not_confirmed",
                "reason": (
                    "three_seed_confirmation_passed"
                    if confirmation["passed"]
                    else "three_seed_confirmation_failed"
                ),
                "selected": selected,
                "gate": confirmation,
            },
        )

    state["routing_summary"] = {
        "sequence": "precommitted_adaptive_seed29_then_single_candidate_confirmation",
        "mandatory_seed29_run_count": MANDATORY_ROUTING_RUN_COUNT,
        "actual_routing_run_count": len(state["routing_jobs"]),
        "slow_combination_decision": slow_combination,
        "slow_selection": slow_selection,
        "macro_temporal_combination_decision": macro_combination,
        "macro_temporal_selection": macro_selection,
        "joint_synthesis_decision": joint_decision,
        "final_selection": final_selection,
        "three_seed_confirmation": confirmation,
        "all_off_scaffold_control_training": False,
        "final_holdout_status": "sealed_not_accessed",
        "transaction_cost_modeling": False,
    }
    _persist_state(state_path, state)


def _summary_payload(state: dict[str, object]) -> dict[str, object]:
    return {
        "experiment_name": EXPERIMENT_NAME,
        "state_version": STATE_VERSION,
        "status": state["status"],
        "created_at_utc": state["created_at_utc"],
        "completed_at_utc": state["completed_at_utc"],
        "configuration": state["configuration"],
        "routing_identity_preflight": state["routing_identity_preflight"],
        "realized_distribution_audit": state["realized_distribution_audit"],
        "issuer_screen": state["issuer_screen"],
        "routing": state["routing_summary"],
        "completed_job_counts": {
            "issuer": sum(job["status"] == "completed" for job in state["issuer_jobs"]),
            "routing": sum(
                job["status"] == "completed" for job in state["routing_jobs"]
            ),
        },
        "final_holdout_status": "sealed_not_accessed",
        "transaction_cost_modeling": False,
    }


def _summary_markdown(summary: dict[str, object]) -> str:
    routing = summary["routing"]
    selected = routing["final_selection"]["selected"]
    selected_text = "none" if selected is None else str(selected["identity"])
    confirmation = routing["three_seed_confirmation"]
    confirmed = (
        "not run" if confirmation is None else str(confirmation["passed"]).lower()
    )
    return (
        "# Context-routing adaptive sequence\n\n"
        f"- Status: {summary['status']}\n"
        f"- Issuer training runs: {summary['completed_job_counts']['issuer']}\n"
        f"- Routing training runs: {summary['completed_job_counts']['routing']}\n"
        f"- Selected routing candidate: {selected_text}\n"
        f"- Three-seed confirmation passed: {confirmed}\n"
        "- All-off scaffold control trained: no\n"
        "- Held-out test accessed: no\n"
    )


def _completed_run_paths_payload(state: dict[str, object]) -> dict[str, object]:
    return {
        "source_peer_primary": state["configuration"]["source_peer_primary"],
        "incumbents": state["incumbent_runs"],
        "completed_jobs": [
            {
                "job_id": job["job_id"],
                "run_dir": job["run_dir"],
                "producing_git_commit_sha": job["producing_git_commit_sha"],
                "output_sha256": job["output_sha256"],
                "context_routing": job["context_routing"],
            }
            for job in [*state["issuer_jobs"], *state["routing_jobs"]]
            if job["status"] == "completed"
        ],
        "routing_identity_preflight": state["routing_identity_preflight"],
        "realized_distribution_audit": state["realized_distribution_audit"],
        "final_holdout_status": "sealed_not_accessed",
    }


def _runbook_text(state_dir: Path, peer_primary_state: Path) -> str:
    module = "brazil_rv.modeling.stage5_context_routing"
    repository = shlex.quote(str(_RESEARCH))
    state = shlex.quote(str(state_dir))
    peer = shlex.quote(str(peer_primary_state))
    pointer = shlex.quote(str(OUTPUT_POINTER))
    common = (
        f"uv run --frozen python -m {module} --state-dir {state} "
        f"--peer-primary-state {peer}"
    )
    return (
        "# Operator runbook\n\n"
        "The primary commands below are for Ubuntu/bash on the single GH200 host. "
        "Launch and resume are intentionally identical. The held-out test remains sealed.\n\n"
        "## Ubuntu/bash\n\n"
        "```bash\n"
        "set -euo pipefail\n"
        f"cd {repository}\n\n"
        "# Validate invocation without running preflight, audit, or training.\n"
        f"{common} --dry-run\n\n"
        "# Launch.\n"
        f"{common}\n\n"
        "# Resume (the command is intentionally identical).\n"
        f"{common}\n\n"
        "# Status.\n"
        f"uv run --frozen python -m {module} --state-dir {state} --status\n\n"
        "# Inspect the atomic output pointer and archive members.\n"
        f"OUTPUT_POINTER={pointer}\n"
        'OUTPUT_DIR="$(<"$OUTPUT_POINTER")"\n'
        f'ARCHIVE="$OUTPUT_DIR/{ARCHIVE_NAME}"\n'
        "printf '%s\\n' \"$OUTPUT_DIR\"\n"
        'tar -tzf "$ARCHIVE"\n\n'
        "# Verify the archive SHA-256 sidecar.\n"
        f'(cd "$OUTPUT_DIR" && sha256sum --check {shlex.quote(ARCHIVE_SHA256_NAME)})\n\n'
        "# Retrieve the archive and sidecar into the current directory.\n"
        'cp -- "$ARCHIVE" "$ARCHIVE.sha256" .\n'
        "```\n\n"
        "## Windows/PowerShell retrieval example\n\n"
        "```powershell\n"
        f"$outputDir = Get-Content -LiteralPath '{OUTPUT_POINTER}'\n"
        f"$archive = Join-Path $outputDir '{ARCHIVE_NAME}'\n"
        "$digest = Get-FileHash -Algorithm SHA256 -LiteralPath $archive\n"
        "$digest\n"
        'Copy-Item -LiteralPath $archive, "$archive.sha256" -Destination .\n'
        "```\n"
    )


def _archive_members() -> tuple[str, ...]:
    return (
        PLAN_JSON,
        "state.json",
        PREFLIGHT_JSON,
        f"realized_distribution_audit/{AUDIT_JSON}",
        f"realized_distribution_audit/{FEATURE_PARQUET}",
        f"realized_distribution_audit/{TARGET_PARQUET}",
        PAIRED_RESULTS_JSON,
        SUMMARY_JSON,
        SUMMARY_MARKDOWN,
        EXPERIMENT_LOG,
        COMPLETED_RUN_PATHS_JSON,
        RUNBOOK,
        ARTIFACT_HASHES_JSON,
    )


def _expected_archive_hashes(state_dir: Path) -> dict[str, str]:
    return {
        member: sha256_file(state_dir.joinpath(*Path(member).parts))
        for member in _archive_members()
    }


def _finalize_outputs(state_dir: Path, state: dict[str, object]) -> None:
    archive_path = state_dir / ARCHIVE_NAME
    sidecar_path = state_dir / ARCHIVE_SHA256_NAME
    if archive_path.is_file():
        expected = _expected_archive_hashes(state_dir)
        validate_archive(archive_path, expected)
        if sidecar_path.exists():
            validate_archive_sha256(archive_path, sidecar_path)
        else:
            write_archive_sha256(archive_path, sidecar_path)
        publish_output_pointer(
            OUTPUT_POINTER, state_dir, archive_path, sidecar_path, expected
        )
        return

    summary = _summary_payload(state)
    _atomic_write_json(state_dir / SUMMARY_JSON, summary)
    (state_dir / SUMMARY_MARKDOWN).write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    _atomic_write_json(
        state_dir / COMPLETED_RUN_PATHS_JSON,
        _completed_run_paths_payload(state),
    )
    payload_members = tuple(
        member for member in _archive_members() if member != ARTIFACT_HASHES_JSON
    )
    payload_hashes = {
        member: sha256_file(state_dir.joinpath(*Path(member).parts))
        for member in payload_members
    }
    _atomic_write_json(
        state_dir / ARTIFACT_HASHES_JSON,
        {
            "experiment_name": EXPERIMENT_NAME,
            "source_provenance": state["configuration"]["source_peer_primary"],
            "payload_sha256": payload_hashes,
            "run_output_sha256": {
                job["job_id"]: job["output_sha256"]
                for job in [*state["issuer_jobs"], *state["routing_jobs"]]
                if job["status"] == "completed"
            },
        },
    )
    expected = create_validated_archive(state_dir, _archive_members(), archive_path)
    write_archive_sha256(archive_path, sidecar_path)
    validate_archive(archive_path, expected)
    validate_archive_sha256(archive_path, sidecar_path)
    publish_output_pointer(
        OUTPUT_POINTER, state_dir, archive_path, sidecar_path, expected
    )


def dry_run_payload(peer_primary_state: Path) -> dict[str, object]:
    commit, clean = _git_identity(require_clean=False)
    feature_store = resolve_feature_store().resolve()
    validate_feature_store(feature_store)
    configuration = _configuration(commit, feature_store, peer_primary_state)
    incumbents = _source_incumbents(peer_primary_state, feature_store)
    return {
        "experiment_name": EXPERIMENT_NAME,
        "dry_run": True,
        "worktree_clean": clean,
        "orchestrator_git_commit_sha": commit,
        "resolved_feature_store_path": str(feature_store),
        "source_peer_primary": configuration["source_peer_primary"],
        "incumbent_runs": {
            str(seed): str(values[0].run_dir) for seed, values in incumbents.items()
        },
        "routing_identity_preflight_will_run_first": True,
        "realized_distribution_audit_will_run_after_preflight": True,
        "issuer_runs": {
            "mandatory": ISSUER_RUN_COUNT_MINIMUM,
            "conditional": ISSUER_RUN_COUNT_MAXIMUM - ISSUER_RUN_COUNT_MINIMUM,
            "minimum": ISSUER_RUN_COUNT_MINIMUM,
            "maximum": ISSUER_RUN_COUNT_MAXIMUM,
        },
        "routing_runs": {
            "mandatory": MANDATORY_ROUTING_RUN_COUNT,
            "conditional_maximum": CONDITIONAL_ROUTING_RUN_COUNT_MAXIMUM,
            "minimum": ROUTING_RUN_COUNT_MINIMUM,
            "maximum": ROUTING_RUN_COUNT_MAXIMUM,
        },
        "total_training_runs": {
            "minimum": TOTAL_TRAINING_RUN_COUNT_MINIMUM,
            "maximum": TOTAL_TRAINING_RUN_COUNT_MAXIMUM,
        },
        "all_off_scaffold_control_training": False,
        "final_holdout_status": "sealed_not_accessed",
        "configuration": configuration,
    }


def format_dry_run(payload: dict[str, object]) -> str:
    issuer = payload["issuer_runs"]
    routing = payload["routing_runs"]
    total = payload["total_training_runs"]
    return "\n".join(
        (
            f"incumbent runs reused: {len(payload['incumbent_runs'])}",
            "routing identity preflight runs first: yes (eager + compiled, 3 SAM steps)",
            "realized-distribution audit follows preflight: yes",
            (
                "issuer runs: "
                f"mandatory={issuer['mandatory']} conditional={issuer['conditional']} "
                f"min={issuer['minimum']} max={issuer['maximum']}"
            ),
            (
                "routing runs: "
                f"mandatory={routing['mandatory']} "
                f"conditional_max={routing['conditional_maximum']} "
                f"min={routing['minimum']} max={routing['maximum']}"
            ),
            f"total training runs: min={total['minimum']} max={total['maximum']}",
            "all-off scaffold control training: no",
            "held-out test accessed: no",
        )
    )


def status_payload(state_dir: Path) -> dict[str, object]:
    state_path = state_dir.resolve() / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Experiment state does not exist: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return {
        "experiment_name": state.get("experiment_name"),
        "status": state.get("status"),
        "issuer_jobs": len(state.get("issuer_jobs", [])),
        "routing_jobs": len(state.get("routing_jobs", [])),
        "decisions": len(state.get("decisions", [])),
        "completed_at_utc": state.get("completed_at_utc"),
        "output_pointer": str(OUTPUT_POINTER),
    }


def run_experiment(state_dir: Path, peer_primary_state: Path) -> Path:
    commit, _ = _git_identity(require_clean=True)
    feature_store = resolve_feature_store().resolve()
    validate_feature_store(feature_store)
    feature_identity = _feature_store_identity(feature_store)
    configuration = _configuration(commit, feature_store, peer_primary_state)
    incumbents_with_provenance = _source_incumbents(peer_primary_state, feature_store)
    incumbents = {
        seed: values[0] for seed, values in incumbents_with_provenance.items()
    }
    state_dir = state_dir.resolve()
    if state_dir == feature_store or state_dir.is_relative_to(feature_store):
        raise ValueError(
            "Stage-5 state and audit artifacts must be outside the feature store"
        )
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.json"
    _atomic_write_text(
        state_dir / RUNBOOK, _runbook_text(state_dir, peer_primary_state)
    )
    expected_preflight_identity = build_routing_preflight_identity(commit)
    with exclusive_process_lock(state_dir / "experiment.lock", EXPERIMENT_NAME):
        log_path = state_dir / EXPERIMENT_LOG
        if not log_path.exists():
            log_path.touch()
        preflight_path = _ensure_preflight(state_dir, expected_preflight_identity)
        audit_path = _ensure_audit(state_dir, feature_store, feature_identity)
        state = _load_state(
            state_path,
            configuration,
            incumbents_with_provenance,
            audit_path,
            preflight_path,
        )
        _persist_state(state_path, state)
        if state["status"] == "completed":
            _finalize_outputs(state_dir, state)
            return state_path
        _run_adaptive_sequence(
            state,
            state_path,
            feature_store,
            commit,
            peer_primary_state,
            audit_path,
            preflight_path,
            incumbents,
        )
        state["status"] = "completed"
        state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _persist_state(state_path, state)
        _log(state_dir, "adaptive sequence completed; finalizing archive")
        _finalize_outputs(state_dir, state)
    return state_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--peer-primary-state", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--status", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.status:
        print(json.dumps(status_payload(args.state_dir), indent=2), flush=True)
        return
    peer_primary_state = resolve_peer_primary_state(args.peer_primary_state)
    if args.dry_run:
        print(format_dry_run(dry_run_payload(peer_primary_state)), flush=True)
        return
    state_path = run_experiment(args.state_dir.resolve(), peer_primary_state)
    print(f"Stage-5 context-routing experiment completed: {state_path}", flush=True)


if __name__ == "__main__":
    main()
