from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import torch

from brazil_rv.preprocessing.contract import HORIZONS

from .context_ablation import get_context_ablation
from .contract import FEATURE_CONTRACT_VERSION, SplitBoundaries
from .engine import EvaluationObservations, validate_runtime
from .evaluate import collect_neural_evaluation
from .process_lock import ProcessLockLease
from .stage3_context_addition import _reject_test_derived_metadata

if TYPE_CHECKING:
    from .stock_time_inference import AnalysisInputs, Stage3AnalysisJob


ANALYSIS_NAME = "stock_time_attribution"
CACHE_VERSION = 4
METRIC_REPRODUCTION_ABSOLUTE_TOLERANCE = 1e-12
SHARED_ARRAY_NAMES = (
    "sample_id",
    "date_idx",
    "decision_idx",
    "targets",
    "raw_returns",
    "label_mask",
)
INFERENCE_CODE_PATHS = (
    "research/src/brazil_rv/preprocessing/contract.py",
    "research/src/brazil_rv/preprocessing/transforms.py",
    "research/src/brazil_rv/modeling/analyze_stage3_context_addition.py",
    "research/src/brazil_rv/modeling/contract.py",
    "research/src/brazil_rv/modeling/context_ablation.py",
    "research/src/brazil_rv/modeling/feature_ablation.py",
    "research/src/brazil_rv/modeling/data.py",
    "research/src/brazil_rv/modeling/layers.py",
    "research/src/brazil_rv/modeling/model.py",
    "research/src/brazil_rv/modeling/engine.py",
    "research/src/brazil_rv/modeling/metrics.py",
    "research/src/brazil_rv/modeling/evaluate.py",
    "research/src/brazil_rv/modeling/stage2_context_ablation.py",
    "research/src/brazil_rv/modeling/stage3_context_addition.py",
    "research/src/brazil_rv/modeling/stock_time_cache.py",
    "research/src/brazil_rv/modeling/stock_time_inference.py",
)


def default_cache_directory(output_dir: Path, analysis_name: str) -> Path:
    return output_dir.resolve().parent / f"_{analysis_name}_cache"


def prediction_cache_directory(
    cache_dir: Path,
    logical_configuration: str,
    seed: int,
) -> Path:
    return cache_dir / "predictions" / f"{logical_configuration}_seed{seed}"


def shared_validation_directory(cache_dir: Path) -> Path:
    return cache_dir / "shared_validation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
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


def atomic_write_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("wb") as destination:
            np.save(destination, values, allow_pickle=False)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def split_boundaries() -> dict[str, str]:
    return {key: str(value) for key, value in asdict(SplitBoundaries()).items()}


def job_cache_identity(
    inputs: AnalysisInputs, job: Stage3AnalysisJob
) -> dict[str, object]:
    sample_ids = (
        inputs.validation_rows.get_column("sample_id").to_numpy().astype(np.int64)
    )
    return {
        "cache_name": ANALYSIS_NAME,
        "cache_version": CACHE_VERSION,
        "split": "validation",
        "logical_configuration": job.logical_configuration,
        "context_ablation": job.context_ablation,
        "context_ablation_specification": get_context_ablation(
            job.context_ablation
        ).metadata(),
        "seed": job.seed,
        "run_manifest_sha256": job.run_manifest_sha256,
        "checkpoint_sha256": job.checkpoint_sha256,
        "producing_training_commit_sha": job.producing_git_commit_sha,
        "inference_configuration": {
            key: job.manifest.get(key)
            for key in (
                "model_name",
                "model_family",
                "tcn_settings",
                "architecture_constants",
                "global_context",
                "objective",
                "seed",
                "context_ablation",
                "feature_ablation",
                "compile",
            )
        },
        "inference_code_sha256": inputs.inference_code_sha256,
        "feature_manifest_sha256": inputs.feature_identity["manifest_sha256"],
        "feature_contract": FEATURE_CONTRACT_VERSION,
        "split_boundaries": split_boundaries(),
        "sample_count": int(sample_ids.size),
        "sample_id_sha256": _array_sha256(sample_ids),
        "prediction_shape": [int(sample_ids.size), 158, len(HORIZONS)],
        "prediction_dtype": "float32",
    }


def shared_cache_identity(inputs: AnalysisInputs) -> dict[str, object]:
    sample_ids = (
        inputs.validation_rows.get_column("sample_id").to_numpy().astype(np.int64)
    )
    return {
        "cache_name": ANALYSIS_NAME,
        "cache_version": CACHE_VERSION,
        "split": "validation",
        "feature_manifest_sha256": inputs.feature_identity["manifest_sha256"],
        "feature_contract": FEATURE_CONTRACT_VERSION,
        "split_boundaries": split_boundaries(),
        "inference_code_sha256": inputs.inference_code_sha256,
        "sample_count": int(sample_ids.size),
        "sample_id_sha256": _array_sha256(sample_ids),
    }


def cache_directory(cache_dir: Path, job: Stage3AnalysisJob) -> Path:
    return prediction_cache_directory(cache_dir, job.logical_configuration, job.seed)


def _cache_creation_provenance(
    inputs: AnalysisInputs, job: Stage3AnalysisJob | None = None
) -> dict[str, object]:
    provenance: dict[str, object] = {
        "stage3_state_path": str(inputs.state_path),
        "stage3_state_sha256": inputs.state_sha256,
        "analyzer_git_commit_sha": inputs.analyzer_git_commit_sha,
        "analyzer_worktree_clean": inputs.analyzer_worktree_clean,
        "analyzer_source_sha256": inputs.analyzer_source_sha256,
    }
    if job is not None:
        provenance.update(
            {
                "run_dir": str(job.run_dir),
                "run_manifest_path": str(job.run_manifest_path),
                "run_manifest_sha256": job.run_manifest_sha256,
                "checkpoint_path": str(job.checkpoint_path),
                "producing_git_commit_sha": job.producing_git_commit_sha,
            }
        )
    return provenance


def _validate_semantic_identity(
    recorded: object, expected: dict[str, object], location: str
) -> None:
    if not isinstance(recorded, dict):
        raise ValueError(f"{location} semantic identity is missing")
    if recorded.get("cache_name") == ANALYSIS_NAME and (
        recorded.get("cache_version") != CACHE_VERSION
    ):
        raise ValueError(
            f"{location} cache version mismatch: expected {CACHE_VERSION}, "
            f"found {recorded.get('cache_version')}"
        )
    code_hashes = recorded.get("inference_code_sha256")
    if not isinstance(code_hashes, dict) or set(code_hashes) != set(
        INFERENCE_CODE_PATHS
    ):
        raise ValueError(f"{location} semantic code-hash mapping is incomplete")
    if recorded != expected:
        raise ValueError(f"{location} semantic identity mismatch")


def _validate_creation_provenance(value: object, location: str) -> None:
    if (
        not isinstance(value, dict)
        or value.get("analyzer_worktree_clean") is not True
        or not isinstance(value.get("analyzer_source_sha256"), str)
        or not isinstance(value.get("analyzer_git_commit_sha"), str)
    ):
        raise ValueError(f"{location} creation provenance is invalid")


def validate_cache_manifest(
    manifest_path: Path, expected_identity: dict[str, object]
) -> tuple[Path, dict[str, object]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _reject_test_derived_metadata(manifest, f"prediction cache {manifest_path}")
    if manifest.get("status") != "completed":
        raise ValueError(f"Prediction cache is incomplete: {manifest_path}")
    _validate_semantic_identity(
        manifest.get("identity"), expected_identity, "Prediction cache"
    )
    _validate_creation_provenance(
        manifest.get("creation_provenance"), "Prediction cache"
    )
    prediction = manifest.get("prediction_file")
    if not isinstance(prediction, dict):
        raise ValueError(f"Prediction cache file metadata is missing: {manifest_path}")
    prediction_path = manifest_path.parent / str(prediction.get("name"))
    if not prediction_path.is_file() or sha256(prediction_path) != prediction.get(
        "sha256"
    ):
        raise ValueError(f"Prediction cache hash is invalid: {manifest_path}")
    array = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    if (
        list(array.shape) != expected_identity["prediction_shape"]
        or str(array.dtype) != expected_identity["prediction_dtype"]
    ):
        raise ValueError(f"Prediction cache shape or dtype is invalid: {manifest_path}")
    return prediction_path, manifest


def metric_reproduction_gate(
    run_dir: Path,
    recomputed_summary: dict[str, object],
    recomputed_daily_rows: list[dict[str, object]],
    *,
    tolerance: float = METRIC_REPRODUCTION_ABSOLUTE_TOLERANCE,
) -> dict[str, object]:
    recorded = json.loads(
        (run_dir / "validation_metrics.json").read_text(encoding="utf-8")
    )
    recorded_primary = float(recorded["primary_score"])
    recomputed_primary = float(recomputed_summary["primary_score"])
    primary_difference = abs(recomputed_primary - recorded_primary)
    recorded_horizons = {
        int(row["horizon_minutes"]): row for row in recorded["horizons"]
    }
    recomputed_horizons = {
        int(row["horizon_minutes"]): row for row in recomputed_summary["horizons"]
    }
    if recorded_horizons.keys() != recomputed_horizons.keys():
        raise ValueError("Recorded and recomputed metric horizons differ")
    horizon_differences = {
        f"{horizon}m": abs(
            float(recomputed_horizons[horizon]["mean_daily_spearman_ic"])
            - float(recorded_horizons[horizon]["mean_daily_spearman_ic"])
        )
        for horizon in recorded_horizons
    }
    recomputed_daily = pl.DataFrame(recomputed_daily_rows).sort(
        "date_idx", "horizon_minutes"
    )
    recorded_daily = pl.read_parquet(run_dir / "validation_daily_metrics.parquet").sort(
        "date_idx", "horizon_minutes"
    )
    if not np.array_equal(
        recomputed_daily.select("date_idx", "horizon_minutes").to_numpy(),
        recorded_daily.select("date_idx", "horizon_minutes").to_numpy(),
    ):
        raise ValueError("Recorded and recomputed daily metric rows are misaligned")
    daily_differences: dict[str, float] = {}
    for column in (
        "spearman_ic",
        "rank_target_pearson_ic",
        "top_return",
        "bottom_return",
        "top_minus_bottom",
        "long_only_top",
        "one_way_turnover",
    ):
        left = recomputed_daily.get_column(column).to_numpy()
        right = recorded_daily.get_column(column).to_numpy()
        if not np.array_equal(np.isnan(left), np.isnan(right)):
            raise ValueError(f"Daily metric finiteness changed: {column}")
        finite = np.isfinite(left) & np.isfinite(right)
        daily_differences[column] = (
            float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0
        )
    maximum_difference = max(
        [primary_difference, *horizon_differences.values(), *daily_differences.values()]
    )
    if maximum_difference > tolerance:
        raise ValueError(
            f"Fresh inference failed validation metric parity: {maximum_difference}"
        )
    return {
        "recomputed_primary_ic": recomputed_primary,
        "recorded_primary_ic": recorded_primary,
        "absolute_primary_difference": primary_difference,
        "horizon_absolute_differences": horizon_differences,
        "daily_metric_maximum_absolute_differences": daily_differences,
        "maximum_absolute_difference": maximum_difference,
        "absolute_tolerance": tolerance,
        "passed": True,
    }


def validate_observation_alignment(
    observations: EvaluationObservations, validation_rows: pl.DataFrame
) -> None:
    expected = (
        ("sample_id", observations.sample_id),
        ("date_idx", observations.date_idx),
        ("decision_idx", observations.decision_idx),
    )
    for name, actual in expected:
        reference = validation_rows.get_column(name).to_numpy().astype(np.int64)
        if actual.dtype != np.int64 or not np.array_equal(actual, reference):
            raise ValueError(f"Collected {name} is not aligned to validation rows")
    expected_shape = (validation_rows.height, 158, len(HORIZONS))
    arrays = (
        (observations.predictions, np.float32),
        (observations.targets, np.float32),
        (observations.raw_returns, np.float32),
        (observations.label_mask, np.bool_),
    )
    if any(
        array.shape != expected_shape or array.dtype != dtype for array, dtype in arrays
    ):
        raise ValueError(
            "Collected observation arrays violate the dense cache contract"
        )


def remove_recognized_partial_cache(directory: Path, expected_names: set[str]) -> None:
    files = list(directory.iterdir())
    allowed = expected_names | {f"{name}.tmp" for name in expected_names}
    unexpected = [path for path in files if path.is_dir() or path.name not in allowed]
    if unexpected:
        raise ValueError(
            f"Ambiguous incomplete cache contains unexpected entries: {unexpected}"
        )
    for path in files:
        path.unlink()


def write_or_validate_shared_cache(
    cache_root: Path,
    inputs: AnalysisInputs,
    observations: EvaluationObservations | None = None,
    production_lock: ProcessLockLease | None = None,
) -> tuple[Path, dict[str, np.ndarray]]:
    shared_dir = shared_validation_directory(cache_root)
    manifest_path = shared_dir / "manifest.json"
    expected_identity = shared_cache_identity(inputs)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _reject_test_derived_metadata(manifest, f"shared cache {manifest_path}")
        if manifest.get("status") != "completed":
            raise ValueError("Shared validation cache is incomplete")
        _validate_semantic_identity(
            manifest.get("identity"), expected_identity, "Shared validation cache"
        )
        _validate_creation_provenance(
            manifest.get("creation_provenance"), "Shared validation cache"
        )
        files = manifest.get("files")
        if not isinstance(files, dict) or set(files) != set(SHARED_ARRAY_NAMES):
            raise ValueError("Shared validation cache file matrix is invalid")
        arrays: dict[str, np.ndarray] = {}
        for name in SHARED_ARRAY_NAMES:
            metadata = files[name]
            if not isinstance(metadata, dict):
                raise ValueError("Shared validation cache metadata is malformed")
            path = shared_dir / str(metadata.get("name"))
            if not path.is_file() or sha256(path) != metadata.get("sha256"):
                raise ValueError(f"Shared validation cache hash is invalid: {name}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != metadata.get("shape") or str(
                array.dtype
            ) != metadata.get("dtype"):
                raise ValueError(f"Shared validation cache contract changed: {name}")
            arrays[name] = array
        if observations is not None:
            for name in SHARED_ARRAY_NAMES:
                if not np.array_equal(arrays[name], getattr(observations, name)):
                    raise ValueError(f"Inference shared array changed: {name}")
        return manifest_path, arrays
    if shared_dir.exists() and any(shared_dir.iterdir()):
        if observations is None:
            raise ValueError("Incomplete shared validation cache cannot be resumed")
        remove_recognized_partial_cache(
            shared_dir,
            {*(f"{name}.npy" for name in SHARED_ARRAY_NAMES), "manifest.json"},
        )
    if observations is None:
        raise FileNotFoundError("Shared validation cache has not been created")
    shared_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, object] = {}
    arrays = {}
    for name in SHARED_ARRAY_NAMES:
        values = np.ascontiguousarray(getattr(observations, name))
        path = shared_dir / f"{name}.npy"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite shared cache: {path}")
        if production_lock is not None:
            production_lock.assert_owned()
        atomic_write_npy(path, values)
        files[name] = {
            "name": path.name,
            "sha256": sha256(path),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
        }
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    if production_lock is not None:
        production_lock.assert_owned()
    atomic_write_json(
        manifest_path,
        {
            "status": "completed",
            "identity": expected_identity,
            "creation_provenance": _cache_creation_provenance(inputs),
            "files": files,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return manifest_path, arrays


def _write_prediction_cache(
    cache_root: Path,
    inputs: AnalysisInputs,
    job: Stage3AnalysisJob,
    observations: EvaluationObservations,
    metric_gate: dict[str, object],
    shared_manifest_path: Path,
    production_lock: ProcessLockLease | None = None,
) -> Path:
    job_cache_dir = cache_directory(cache_root, job)
    manifest_path = job_cache_dir / "manifest.json"
    if manifest_path.exists() or (
        job_cache_dir.exists() and any(job_cache_dir.iterdir())
    ):
        raise FileExistsError(
            f"Refusing to overwrite prediction cache: {job_cache_dir}"
        )
    job_cache_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = job_cache_dir / "predictions.npy"
    predictions = np.ascontiguousarray(observations.predictions, dtype=np.float32)
    if production_lock is not None:
        production_lock.assert_owned()
    atomic_write_npy(prediction_path, predictions)
    if production_lock is not None:
        production_lock.assert_owned()
    atomic_write_json(
        manifest_path,
        {
            "status": "completed",
            "identity": job_cache_identity(inputs, job),
            "creation_provenance": _cache_creation_provenance(inputs, job),
            "prediction_file": {
                "name": prediction_path.name,
                "sha256": sha256(prediction_path),
                "shape": list(predictions.shape),
                "dtype": str(predictions.dtype),
            },
            "shared_validation_manifest": {
                "relative_path": os.path.relpath(
                    shared_manifest_path, manifest_path.parent
                ),
                "sha256": sha256(shared_manifest_path),
            },
            "metric_reproduction_gate": metric_gate,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return manifest_path


def adopt_or_infer_caches(
    cache_root: Path,
    inputs: AnalysisInputs,
    state: dict[str, object],
    state_path: Path,
    production_lock: ProcessLockLease | None = None,
) -> tuple[dict[tuple[str, int], Path], dict[str, np.ndarray]]:
    cache_paths: dict[tuple[str, int], Path] = {}
    runtime_validated = False
    state_jobs = state["jobs"]
    if not isinstance(state_jobs, list):
        raise ValueError("Analysis state jobs are malformed")
    for job, state_job in zip(inputs.jobs, state_jobs, strict=True):
        if not isinstance(state_job, dict):
            raise ValueError("Analysis state job is malformed")
        job_dir = cache_directory(cache_root, job)
        manifest_path = job_dir / "manifest.json"
        expected_identity = job_cache_identity(inputs, job)
        if manifest_path.is_file():
            prediction_path, manifest = validate_cache_manifest(
                manifest_path, expected_identity
            )
            shared_metadata = manifest.get("shared_validation_manifest")
            if not isinstance(shared_metadata, dict) or not isinstance(
                shared_metadata.get("relative_path"), str
            ):
                raise ValueError(
                    f"Prediction cache shared provenance is invalid: {manifest_path}"
                )
            shared_path = (
                manifest_path.parent / str(shared_metadata["relative_path"])
            ).resolve()
            if (
                shared_path
                != (shared_validation_directory(cache_root) / "manifest.json").resolve()
                or not shared_path.is_file()
                or sha256(shared_path) != shared_metadata.get("sha256")
            ):
                raise ValueError(
                    f"Prediction cache shared provenance is invalid: {manifest_path}"
                )
            gate = manifest.get("metric_reproduction_gate")
            if not isinstance(gate, dict) or gate.get("passed") is not True:
                raise ValueError(
                    f"Prediction cache lacks metric parity: {manifest_path}"
                )
            state_job.update(
                {
                    "status": "completed",
                    "cache_manifest_path": str(manifest_path),
                    "cache_manifest_sha256": sha256(manifest_path),
                    "completed_at_utc": state_job.get("completed_at_utc")
                    or manifest.get("created_at_utc"),
                    "error": None,
                }
            )
            cache_paths[(job.logical_configuration, job.seed)] = prediction_path
            atomic_write_json(state_path, state)
            continue
        if job_dir.exists() and any(job_dir.iterdir()):
            if (
                state_job.get("status") == "completed"
                or state.get("status") == "completed"
            ):
                raise ValueError(
                    "Completed analysis state is missing a prediction cache"
                )
            remove_recognized_partial_cache(
                job_dir, {"predictions.npy", "manifest.json"}
            )
        if state.get("status") == "completed":
            raise ValueError("Completed analysis state is missing a prediction cache")
        if not runtime_validated:
            validate_runtime()
            torch.set_float32_matmul_precision("high")
            runtime_validated = True
        state_job.update(
            {
                "status": "running",
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "completed_at_utc": None,
                "cache_manifest_path": None,
                "cache_manifest_sha256": None,
                "error": None,
            }
        )
        atomic_write_json(state_path, state)
        try:
            evaluation = collect_neural_evaluation(
                job.manifest, inputs.feature_store, inputs.validation_rows
            )
            validate_observation_alignment(
                evaluation.observations, inputs.validation_rows
            )
            gate = metric_reproduction_gate(
                job.run_dir, evaluation.summary, evaluation.daily_rows
            )
            shared_manifest, _ = write_or_validate_shared_cache(
                cache_root, inputs, evaluation.observations, production_lock
            )
            manifest_path = _write_prediction_cache(
                cache_root,
                inputs,
                job,
                evaluation.observations,
                gate,
                shared_manifest,
                production_lock,
            )
            prediction_path, _ = validate_cache_manifest(
                manifest_path, expected_identity
            )
            if production_lock is not None:
                production_lock.assert_owned()
        except BaseException as error:
            state_job.update(
                {"status": "failed", "error": f"{type(error).__name__}: {error}"}
            )
            atomic_write_json(state_path, state)
            raise
        state_job.update(
            {
                "status": "completed",
                "cache_manifest_path": str(manifest_path),
                "cache_manifest_sha256": sha256(manifest_path),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": None,
            }
        )
        cache_paths[(job.logical_configuration, job.seed)] = prediction_path
        atomic_write_json(state_path, state)
    if production_lock is not None:
        production_lock.assert_owned()
    state["status"] = "inference_completed"
    atomic_write_json(state_path, state)
    _, shared = write_or_validate_shared_cache(
        cache_root, inputs, production_lock=production_lock
    )
    return cache_paths, shared
