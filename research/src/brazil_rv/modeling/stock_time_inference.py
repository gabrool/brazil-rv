from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import torch

from .analyze_stage3_context_addition import _validate_configuration
from .context_ablation import get_context_ablation
from .contract import (
    FEATURE_STORE_POINTER,
    VALIDATION_END,
    VALIDATION_START,
)
from .data import _validate_sample_index, load_sample_index, select_sample_split
from .evaluate import (
    _normalize_feature_ablation_identity,
    _validate_run_checkpoint_identity,
)
from .stage2_context_ablation import (
    _feature_store_identity,
    feature_stores_equivalent,
)
from .stage3_context_addition import (
    ADOPTED_STAGE2_LOGICAL_CONFIGURATION,
    STAGE2_PRODUCING_COMMIT,
    STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION,
    STAGE3_LOGICAL_CONFIGURATION_ORDER,
    STAGE3_SEEDS,
    STATE_VERSION as STAGE3_STATE_VERSION,
    SWEEP_NAME as STAGE3_SWEEP_NAME,
    _completed_job_artifacts,
    _reject_test_derived_metadata,
    _validated_stage2_adoptions,
    build_stage3_command,
    validate_stage3_completed_run,
)
from .stock_time_cache import (
    INFERENCE_CODE_PATHS,
    METRIC_REPRODUCTION_ABSOLUTE_TOLERANCE,
    sha256,
)


@dataclass(frozen=True)
class Stage3AnalysisJob:
    position: int
    logical_configuration: str
    context_ablation: str
    seed: int
    run_dir: Path
    run_manifest_path: Path
    run_manifest_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    producing_git_commit_sha: str
    manifest: dict[str, object]


@dataclass(frozen=True)
class AnalysisInputs:
    state_path: Path
    state_sha256: str
    state: dict[str, object]
    configuration: dict[str, object]
    feature_store: Path
    feature_identity: dict[str, object]
    feature_manifest: dict[str, object]
    sample_index: pl.DataFrame
    validation_rows: pl.DataFrame
    jobs: tuple[Stage3AnalysisJob, ...]
    analyzer_git_commit_sha: str
    analyzer_worktree_clean: bool
    analyzer_source_sha256: str
    inference_code_sha256: dict[str, str] = field(default_factory=dict)


def inference_code_identity() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[4]
    return {
        relative: sha256(repository / relative) for relative in INFERENCE_CODE_PATHS
    }


def _git_identity() -> tuple[str, bool]:
    repository = Path(__file__).resolve().parents[4]
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, not bool(status)


def assert_repository_identity(inputs: AnalysisInputs) -> None:
    commit, clean = _git_identity()
    if not clean:
        raise RuntimeError("Non-dry analysis requires a clean worktree")
    if commit != inputs.analyzer_git_commit_sha:
        raise RuntimeError("Repository commit changed during analysis")
    if inference_code_identity() != inputs.inference_code_sha256:
        raise RuntimeError("Inference-affecting source changed during analysis")


def reject_test_derived_path(path: Path, location: str) -> None:
    normalized = str(path).replace("\\", "/").casefold()
    forbidden = (
        "/final_test/",
        "/final-test/",
        "/evaluations/test/",
        "/test_evaluation/",
        "/evaluation_test/",
    )
    if any(marker in f"/{normalized.strip('/')}/" for marker in forbidden):
        raise ValueError(f"{location} is test-derived: {path}")


def _resolve_state_feature_store(configuration: dict[str, object]) -> Path:
    identity = configuration.get("feature_store")
    if not isinstance(identity, dict) or not isinstance(
        identity.get("resolved_path"), str
    ):
        raise ValueError("Stage-3 state lacks a resolved feature-store identity")
    recorded = Path(str(identity["resolved_path"])).expanduser()
    if recorded.is_dir():
        return recorded.resolve()
    pointer = FEATURE_STORE_POINTER
    if not pointer.is_file():
        raise FileNotFoundError(f"Recorded feature store is unavailable: {recorded}")
    current = Path(pointer.read_text(encoding="utf-8").strip()).resolve()
    if not feature_stores_equivalent(recorded, current) and (
        _feature_store_identity(current).get("manifest_sha256")
        != identity.get("manifest_sha256")
    ):
        raise ValueError("Current canonical feature store differs from Stage-3")
    return current


def _validate_checkpoint_identity(
    run_dir: Path,
    manifest: dict[str, object],
    feature_store: Path,
) -> tuple[Path, str]:
    checkpoint_path = run_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _validate_run_checkpoint_identity(manifest, checkpoint, feature_store, run_dir)
    feature_identity = _normalize_feature_ablation_identity(
        manifest, checkpoint, run_dir=run_dir
    )
    if feature_identity.metadata["key"] != "none":
        raise ValueError("Stage-4 feature-ablation checkpoints are forbidden")
    del checkpoint
    return checkpoint_path, sha256(checkpoint_path)


def validate_analysis_inputs(stage3_state_path: Path, scope: str) -> AnalysisInputs:
    if scope not in {"core", "full-stage3"}:
        raise ValueError(f"Unknown analysis scope: {scope}")
    stage3_state_path = stage3_state_path.resolve()
    reject_test_derived_path(stage3_state_path, "Stage-3 state path")
    raw_state = stage3_state_path.read_bytes()
    state_sha = hashlib.sha256(raw_state).hexdigest()
    state = json.loads(raw_state)
    _reject_test_derived_metadata(state, "Stage-3 state")
    if (
        state.get("state_version") != STAGE3_STATE_VERSION
        or state.get("sweep_name") != STAGE3_SWEEP_NAME
        or state.get("status") != "completed"
    ):
        raise ValueError("Analyzer requires a completed canonical Stage-3 state")
    configuration = state.get("configuration")
    jobs = state.get("jobs")
    if not isinstance(configuration, dict) or not isinstance(jobs, list):
        raise ValueError("Stage-3 state lacks configuration or jobs")
    _validate_configuration(configuration)
    _validated_stage2_adoptions(
        Path(str(configuration["source_stage2_state"])), configuration
    )
    expected_order = tuple(
        (
            logical,
            STAGE3_ABLATION_BY_LOGICAL_CONFIGURATION[logical],
            seed,
        )
        for logical in STAGE3_LOGICAL_CONFIGURATION_ORDER
        for seed in STAGE3_SEEDS
    )
    actual_order = tuple(
        (
            job.get("logical_configuration"),
            job.get("context_ablation"),
            job.get("seed"),
        )
        for job in jobs
        if isinstance(job, dict)
    )
    if actual_order != expected_order or len(jobs) != 24:
        raise ValueError("Analyzer requires the exact ordered canonical 24-job matrix")
    if any(job.get("status") != "completed" for job in jobs):
        raise ValueError("Analyzer refuses a partially completed Stage-3 matrix")
    feature_store = _resolve_state_feature_store(configuration)
    sample_index = load_sample_index(feature_store)
    _validate_sample_index(sample_index)
    feature_identity = _feature_store_identity(feature_store)
    configured_identity = configuration["feature_store"]
    if not isinstance(configured_identity, dict) or (
        feature_identity["manifest_sha256"]
        != configured_identity.get("manifest_sha256")
    ):
        raise ValueError("Feature-store manifest identity differs from Stage-3")
    feature_manifest = json.loads(
        (feature_store / "manifest.json").read_text(encoding="utf-8")
    )
    validation_rows = select_sample_split(sample_index, "validation").sort("sample_id")
    if (
        validation_rows.get_column("trade_date").min() != VALIDATION_START
        or validation_rows.get_column("trade_date").max() != VALIDATION_END
    ):
        raise ValueError("Validation rows have the wrong boundaries")
    selected_logicals = (
        ("core",) if scope == "core" else STAGE3_LOGICAL_CONFIGURATION_ORDER
    )
    resolved: list[Stage3AnalysisJob] = []
    all_run_dirs: list[Path] = []
    for position, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise ValueError("Stage-3 job is malformed")
        logical = str(job["logical_configuration"])
        key = str(job["context_ablation"])
        seed = int(job["seed"])
        if job.get("context_ablation_metadata") != get_context_ablation(
            key
        ).metadata() or job.get("command") != list(build_stage3_command(logical, seed)):
            raise ValueError(f"Stage-3 job metadata is invalid: {logical}/{seed}")
        run_dir, score, manifest_sha = _completed_job_artifacts(job, configuration)
        all_run_dirs.append(run_dir.resolve())
        if logical not in selected_logicals:
            continue
        producing_commit = (
            STAGE2_PRODUCING_COMMIT
            if logical == ADOPTED_STAGE2_LOGICAL_CONFIGURATION
            else str(configuration["orchestrator_git_commit_sha"])
        )
        if job.get("producing_git_commit_sha") != producing_commit:
            raise ValueError(f"Stage-3 producing commit is invalid: {logical}/{seed}")
        validated_score = validate_stage3_completed_run(
            run_dir, configuration, key, seed, producing_commit
        )
        if not math.isclose(
            score,
            validated_score,
            rel_tol=0.0,
            abs_tol=METRIC_REPRODUCTION_ABSOLUTE_TOLERANCE,
        ):
            raise ValueError(f"Stage-3 score changed: {logical}/{seed}")
        manifest_path = run_dir / "run_manifest.json"
        if manifest_sha != job.get("run_manifest_sha256"):
            raise ValueError(f"Run-manifest hash changed: {logical}/{seed}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _reject_test_derived_metadata(manifest, f"run manifest {logical}/{seed}")
        manifest["run_dir"] = str(run_dir.resolve())
        checkpoint_path, checkpoint_sha = _validate_checkpoint_identity(
            run_dir, manifest, feature_store
        )
        resolved.append(
            Stage3AnalysisJob(
                position=position,
                logical_configuration=logical,
                context_ablation=key,
                seed=seed,
                run_dir=run_dir.resolve(),
                run_manifest_path=manifest_path.resolve(),
                run_manifest_sha256=manifest_sha,
                checkpoint_path=checkpoint_path.resolve(),
                checkpoint_sha256=checkpoint_sha,
                producing_git_commit_sha=producing_commit,
                manifest=manifest,
            )
        )
    if len(set(all_run_dirs)) != 24:
        raise ValueError("Stage-3 state contains duplicate run identities")
    expected_selected_count = 3 if scope == "core" else 24
    if len(resolved) != expected_selected_count:
        raise ValueError("Resolved inference matrix has the wrong size")
    if len({(job.checkpoint_path, job.checkpoint_sha256) for job in resolved}) != (
        expected_selected_count
    ):
        raise ValueError("Selected jobs contain duplicate checkpoint identities")
    commit, clean = _git_identity()
    reporting_source = Path(__file__).with_name("analyze_stock_time_attribution.py")
    return AnalysisInputs(
        state_path=stage3_state_path,
        state_sha256=state_sha,
        state=state,
        configuration=configuration,
        feature_store=feature_store,
        feature_identity=feature_identity,
        feature_manifest=feature_manifest,
        sample_index=sample_index,
        validation_rows=validation_rows,
        jobs=tuple(resolved),
        analyzer_git_commit_sha=commit,
        analyzer_worktree_clean=clean,
        analyzer_source_sha256=sha256(reporting_source),
        inference_code_sha256=inference_code_identity(),
    )
