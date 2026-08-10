from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from types import MappingProxyType
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
METRIC_REPRODUCTION_GATE_SCHEMA_VERSION = 1
METRIC_REPRODUCTION_PRIMARY_IC_ABSOLUTE_TOLERANCE = 1e-6
METRIC_REPRODUCTION_HORIZON_IC_ABSOLUTE_TOLERANCE = 1e-6
METRIC_REPRODUCTION_DAILY_IC_ABSOLUTE_TOLERANCE = 1e-4
METRIC_REPRODUCTION_ECONOMIC_RETURN_ABSOLUTE_TOLERANCE = 1e-12
METRIC_REPRODUCTION_TURNOVER_ABSOLUTE_TOLERANCE = 5e-3
METRIC_REPRODUCTION_DAILY_THRESHOLDS = MappingProxyType(
    {
        "spearman_ic": METRIC_REPRODUCTION_DAILY_IC_ABSOLUTE_TOLERANCE,
        "rank_target_pearson_ic": (METRIC_REPRODUCTION_DAILY_IC_ABSOLUTE_TOLERANCE),
        "top_return": METRIC_REPRODUCTION_ECONOMIC_RETURN_ABSOLUTE_TOLERANCE,
        "bottom_return": METRIC_REPRODUCTION_ECONOMIC_RETURN_ABSOLUTE_TOLERANCE,
        "top_minus_bottom": (METRIC_REPRODUCTION_ECONOMIC_RETURN_ABSOLUTE_TOLERANCE),
        "long_only_top": (METRIC_REPRODUCTION_ECONOMIC_RETURN_ABSOLUTE_TOLERANCE),
        "one_way_turnover": METRIC_REPRODUCTION_TURNOVER_ABSOLUTE_TOLERANCE,
    }
)
_DAILY_KEY_COLUMNS = ("date_idx", "horizon_minutes")
_RECORDED_DAILY_COLUMNS = (
    "trade_date",
    *_DAILY_KEY_COLUMNS,
    *METRIC_REPRODUCTION_DAILY_THRESHOLDS,
)
_RECOMPUTED_DAILY_COLUMNS = (
    *_DAILY_KEY_COLUMNS,
    *METRIC_REPRODUCTION_DAILY_THRESHOLDS,
)
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


def metric_reproduction_thresholds() -> dict[str, object]:
    return {
        "primary_ic_absolute_tolerance": (
            METRIC_REPRODUCTION_PRIMARY_IC_ABSOLUTE_TOLERANCE
        ),
        "horizon_mean_daily_spearman_ic_absolute_tolerance": (
            METRIC_REPRODUCTION_HORIZON_IC_ABSOLUTE_TOLERANCE
        ),
        "daily_metric_absolute_tolerances": dict(METRIC_REPRODUCTION_DAILY_THRESHOLDS),
    }


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
    validate_metric_reproduction_gate(manifest.get("metric_reproduction_gate"))
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


def _metric_gate_error(failures: list[dict[str, object]]) -> None:
    details = {
        "schema_version": METRIC_REPRODUCTION_GATE_SCHEMA_VERSION,
        "failures": failures,
    }
    raise ValueError(
        "Fresh inference failed validation metric parity:\n"
        + json.dumps(details, indent=2, sort_keys=True, allow_nan=False)
    )


def _finite_metric(
    value: object,
    location: str,
    failures: list[dict[str, object]],
) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        failures.append(
            {"check": "finite_metric", "location": location, "value": repr(value)}
        )
        return None
    if not math.isfinite(result):
        failures.append(
            {"check": "finite_metric", "location": location, "value": str(result)}
        )
        return None
    return result


def _summary_horizons(
    summary: dict[str, object],
    location: str,
    failures: list[dict[str, object]],
) -> dict[int, dict[str, object]] | None:
    raw_rows = summary.get("horizons")
    if not isinstance(raw_rows, list) or any(
        not isinstance(row, dict) for row in raw_rows
    ):
        failures.append({"check": "horizon_schema", "location": location})
        return None
    rows = tuple(row for row in raw_rows if isinstance(row, dict))
    horizon_values = tuple(row.get("horizon_minutes") for row in rows)
    if any(
        not isinstance(horizon, int) or isinstance(horizon, bool)
        for horizon in horizon_values
    ):
        failures.append({"check": "horizon_type", "location": location})
        return None
    if horizon_values != HORIZONS:
        failures.append(
            {
                "check": "required_horizon_order",
                "location": location,
                "expected": list(HORIZONS),
                "actual": list(horizon_values),
            }
        )
        return None
    for row in rows:
        _finite_metric(
            row.get("mean_daily_spearman_ic"),
            f"{location}.horizons.{row['horizon_minutes']}",
            failures,
        )
    return {int(row["horizon_minutes"]): row for row in rows}


def _daily_structure_failures(
    recorded: pl.DataFrame,
    recomputed: pl.DataFrame,
) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    for location, frame, expected_columns in (
        ("recorded_daily", recorded, _RECORDED_DAILY_COLUMNS),
        ("recomputed_daily", recomputed, _RECOMPUTED_DAILY_COLUMNS),
    ):
        if set(frame.columns) != set(expected_columns):
            failures.append(
                {
                    "check": "required_daily_columns",
                    "location": location,
                    "expected": list(expected_columns),
                    "actual": frame.columns,
                }
            )
    if failures:
        return failures
    if recorded.height != recomputed.height or recorded.height == 0:
        failures.append(
            {
                "check": "daily_row_count",
                "recorded": recorded.height,
                "recomputed": recomputed.height,
            }
        )
    for location, frame in (
        ("recorded_daily", recorded),
        ("recomputed_daily", recomputed),
    ):
        key_rows = frame.select(_DAILY_KEY_COLUMNS).rows()
        if key_rows != frame.sort(_DAILY_KEY_COLUMNS).select(_DAILY_KEY_COLUMNS).rows():
            failures.append({"check": "daily_row_order", "location": location})
        if len(key_rows) != len(set(key_rows)):
            failures.append({"check": "duplicate_daily_key", "location": location})
        if any(
            not isinstance(date_index, int)
            or isinstance(date_index, bool)
            or not isinstance(horizon, int)
            or isinstance(horizon, bool)
            for date_index, horizon in key_rows
        ):
            failures.append({"check": "daily_key_type", "location": location})
        if set(frame.get_column("horizon_minutes").to_list()) != set(HORIZONS):
            failures.append(
                {
                    "check": "required_daily_horizons",
                    "location": location,
                    "expected": list(HORIZONS),
                    "actual": sorted(
                        str(value)
                        for value in set(frame.get_column("horizon_minutes").to_list())
                    ),
                }
            )
    if (
        recorded.select(_DAILY_KEY_COLUMNS).rows()
        != recomputed.select(_DAILY_KEY_COLUMNS).rows()
    ):
        failures.append(
            {
                "check": "daily_key_alignment",
                "key_columns": list(_DAILY_KEY_COLUMNS),
            }
        )
    if recorded.get_column("trade_date").null_count() != 0:
        failures.append({"check": "recorded_trade_date_missing"})
    return failures


def _gate_number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Metric reproduction gate provenance is invalid: {location}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Metric reproduction gate provenance is invalid: {location}")
    return result


def _validate_gate_comparison(
    value: object,
    *,
    location: str,
    threshold: float,
    difference_field: str,
    extra_fields: set[str] | None = None,
    with_values: bool = True,
) -> bool:
    required = {
        difference_field,
        "threshold",
        "passed",
        *(extra_fields or set()),
    }
    if with_values:
        required.update(("recorded", "recomputed"))
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(
            f"Metric reproduction gate provenance is invalid: {location} schema"
        )
    difference = _gate_number(value[difference_field], f"{location}.{difference_field}")
    values_match = True
    if with_values:
        recorded = _gate_number(value["recorded"], f"{location}.recorded")
        recomputed = _gate_number(value["recomputed"], f"{location}.recomputed")
        values_match = difference == abs(recomputed - recorded)
    if value["threshold"] != threshold or not values_match:
        raise ValueError(
            f"Metric reproduction gate provenance is invalid: {location} values"
        )
    expected_pass = difference <= threshold
    if not isinstance(value["passed"], bool) or value["passed"] is not expected_pass:
        raise ValueError(
            f"Metric reproduction gate provenance is invalid: {location} pass status"
        )
    return expected_pass


def validate_metric_reproduction_gate(value: object) -> dict[str, object]:
    required = {
        "schema_version",
        "passed",
        "thresholds",
        "daily_rows",
        "primary_ic",
        "horizons",
        "daily_metrics",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Metric reproduction gate provenance is invalid: schema")
    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Metric reproduction gate provenance is invalid: JSON"
        ) from exc
    if (
        not isinstance(value["schema_version"], int)
        or isinstance(value["schema_version"], bool)
        or value["schema_version"] != METRIC_REPRODUCTION_GATE_SCHEMA_VERSION
    ):
        raise ValueError("Metric reproduction gate provenance is invalid: version")
    if value["thresholds"] != metric_reproduction_thresholds():
        raise ValueError("Metric reproduction gate provenance is invalid: thresholds")

    daily_rows = value["daily_rows"]
    if not isinstance(daily_rows, dict) or set(daily_rows) != {
        "recorded_count",
        "recomputed_count",
        "key_columns",
        "metric_columns",
    }:
        raise ValueError(
            "Metric reproduction gate provenance is invalid: daily row schema"
        )
    recorded_count = daily_rows["recorded_count"]
    recomputed_count = daily_rows["recomputed_count"]
    if (
        not isinstance(recorded_count, int)
        or isinstance(recorded_count, bool)
        or recorded_count <= 0
        or not isinstance(recomputed_count, int)
        or isinstance(recomputed_count, bool)
        or recomputed_count != recorded_count
        or daily_rows["key_columns"] != list(_DAILY_KEY_COLUMNS)
        or daily_rows["metric_columns"] != list(METRIC_REPRODUCTION_DAILY_THRESHOLDS)
    ):
        raise ValueError(
            "Metric reproduction gate provenance is invalid: daily row metadata"
        )

    passes = [
        _validate_gate_comparison(
            value["primary_ic"],
            location="primary_ic",
            threshold=METRIC_REPRODUCTION_PRIMARY_IC_ABSOLUTE_TOLERANCE,
            difference_field="absolute_difference",
        )
    ]
    horizons = value["horizons"]
    expected_horizon_keys = {f"{horizon}m" for horizon in HORIZONS}
    if not isinstance(horizons, dict) or set(horizons) != expected_horizon_keys:
        raise ValueError(
            "Metric reproduction gate provenance is invalid: horizon schema"
        )
    for horizon in HORIZONS:
        key = f"{horizon}m"
        row = horizons[key]
        passed = _validate_gate_comparison(
            row,
            location=f"horizons.{key}",
            threshold=METRIC_REPRODUCTION_HORIZON_IC_ABSOLUTE_TOLERANCE,
            difference_field="absolute_difference",
            extra_fields={"horizon_minutes"},
        )
        if not isinstance(row, dict) or row["horizon_minutes"] != horizon:
            raise ValueError(
                f"Metric reproduction gate provenance is invalid: horizons.{key}"
            )
        passes.append(passed)

    daily_metrics = value["daily_metrics"]
    if not isinstance(daily_metrics, dict) or set(daily_metrics) != set(
        METRIC_REPRODUCTION_DAILY_THRESHOLDS
    ):
        raise ValueError(
            "Metric reproduction gate provenance is invalid: daily metric schema"
        )
    for metric, threshold in METRIC_REPRODUCTION_DAILY_THRESHOLDS.items():
        row = daily_metrics[metric]
        passed = _validate_gate_comparison(
            row,
            location=f"daily_metrics.{metric}",
            threshold=threshold,
            difference_field="maximum_absolute_difference",
            extra_fields={"worst_row"},
            with_values=False,
        )
        if not isinstance(row, dict):
            raise ValueError(
                f"Metric reproduction gate provenance is invalid: {metric}"
            )
        worst = row["worst_row"]
        if not isinstance(worst, dict) or set(worst) != {
            "date_idx",
            "trade_date",
            "horizon_minutes",
            "recorded",
            "recomputed",
            "absolute_difference",
        }:
            raise ValueError(
                f"Metric reproduction gate provenance is invalid: {metric} worst row"
            )
        date_index = worst["date_idx"]
        horizon = worst["horizon_minutes"]
        trade_date = worst["trade_date"]
        if (
            not isinstance(date_index, int)
            or isinstance(date_index, bool)
            or not isinstance(trade_date, str)
            or not isinstance(horizon, int)
            or isinstance(horizon, bool)
            or horizon not in HORIZONS
        ):
            raise ValueError(
                f"Metric reproduction gate provenance is invalid: {metric} worst key"
            )
        try:
            date.fromisoformat(trade_date)
        except ValueError as exc:
            raise ValueError(
                f"Metric reproduction gate provenance is invalid: {metric} worst date"
            ) from exc
        worst_recorded = _gate_number(
            worst["recorded"], f"daily_metrics.{metric}.worst_row.recorded"
        )
        worst_recomputed = _gate_number(
            worst["recomputed"], f"daily_metrics.{metric}.worst_row.recomputed"
        )
        worst_difference = _gate_number(
            worst["absolute_difference"],
            f"daily_metrics.{metric}.worst_row.absolute_difference",
        )
        if (
            worst_difference != abs(worst_recomputed - worst_recorded)
            or worst_difference != row["maximum_absolute_difference"]
        ):
            raise ValueError(
                f"Metric reproduction gate provenance is invalid: {metric} worst row"
            )
        passes.append(passed)

    expected_pass = all(passes)
    if not isinstance(value["passed"], bool) or value["passed"] is not expected_pass:
        raise ValueError(
            "Metric reproduction gate provenance is invalid: overall pass status"
        )
    if value["passed"] is not True:
        raise ValueError("Prediction cache lacks metric parity")
    return value


def metric_reproduction_gate(
    run_dir: Path,
    recomputed_summary: dict[str, object],
    recomputed_daily_rows: list[dict[str, object]],
) -> dict[str, object]:
    recorded = json.loads(
        (run_dir / "validation_metrics.json").read_text(encoding="utf-8")
    )
    failures: list[dict[str, object]] = []
    if not isinstance(recorded, dict):
        _metric_gate_error([{"check": "recorded_summary_schema"}])
    recorded_primary = _finite_metric(
        recorded.get("primary_score"), "recorded.primary_score", failures
    )
    recomputed_primary = _finite_metric(
        recomputed_summary.get("primary_score"),
        "recomputed.primary_score",
        failures,
    )
    recorded_horizons = _summary_horizons(recorded, "recorded", failures)
    recomputed_horizons = _summary_horizons(recomputed_summary, "recomputed", failures)
    if failures:
        _metric_gate_error(failures)
    assert recorded_primary is not None
    assert recomputed_primary is not None
    assert recorded_horizons is not None
    assert recomputed_horizons is not None
    primary_difference = abs(recomputed_primary - recorded_primary)
    primary_result = {
        "recorded": recorded_primary,
        "recomputed": recomputed_primary,
        "absolute_difference": primary_difference,
        "threshold": METRIC_REPRODUCTION_PRIMARY_IC_ABSOLUTE_TOLERANCE,
        "passed": (
            primary_difference <= METRIC_REPRODUCTION_PRIMARY_IC_ABSOLUTE_TOLERANCE
        ),
    }
    horizon_results: dict[str, object] = {}
    for horizon in HORIZONS:
        recorded_value = float(recorded_horizons[horizon]["mean_daily_spearman_ic"])
        recomputed_value = float(recomputed_horizons[horizon]["mean_daily_spearman_ic"])
        difference = abs(recomputed_value - recorded_value)
        horizon_results[f"{horizon}m"] = {
            "horizon_minutes": horizon,
            "recorded": recorded_value,
            "recomputed": recomputed_value,
            "absolute_difference": difference,
            "threshold": METRIC_REPRODUCTION_HORIZON_IC_ABSOLUTE_TOLERANCE,
            "passed": (difference <= METRIC_REPRODUCTION_HORIZON_IC_ABSOLUTE_TOLERANCE),
        }

    recomputed_daily = pl.DataFrame(recomputed_daily_rows)
    recorded_daily = pl.read_parquet(run_dir / "validation_daily_metrics.parquet")
    structure_failures = _daily_structure_failures(recorded_daily, recomputed_daily)
    if structure_failures:
        _metric_gate_error(structure_failures)

    daily_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    failures = []
    for column in METRIC_REPRODUCTION_DAILY_THRESHOLDS:
        if (
            recorded_daily.get_column(column).null_count() != 0
            or recomputed_daily.get_column(column).null_count() != 0
        ):
            failures.append({"check": "daily_metric_null", "metric": column})
            continue
        try:
            recorded_values = np.asarray(
                recorded_daily.get_column(column).to_list(), dtype=np.float64
            )
            recomputed_values = np.asarray(
                recomputed_daily.get_column(column).to_list(), dtype=np.float64
            )
        except (TypeError, ValueError):
            failures.append({"check": "daily_metric_type", "metric": column})
            continue
        if np.isinf(recorded_values).any() or np.isinf(recomputed_values).any():
            failures.append({"check": "daily_metric_infinity", "metric": column})
            continue
        if not np.array_equal(np.isnan(recorded_values), np.isnan(recomputed_values)):
            failures.append({"check": "daily_metric_nan_pattern", "metric": column})
            continue
        if not np.isfinite(recorded_values).any():
            failures.append(
                {"check": "daily_metric_has_no_finite_value", "metric": column}
            )
            continue
        daily_arrays[column] = (recorded_values, recomputed_values)
    if failures:
        _metric_gate_error(failures)

    daily_results: dict[str, object] = {}
    trade_dates = recorded_daily.get_column("trade_date").to_list()
    date_indices = recorded_daily.get_column("date_idx").to_list()
    horizon_minutes = recorded_daily.get_column("horizon_minutes").to_list()
    for column, threshold in METRIC_REPRODUCTION_DAILY_THRESHOLDS.items():
        recorded_values, recomputed_values = daily_arrays[column]
        finite = np.isfinite(recorded_values)
        differences = np.abs(recomputed_values - recorded_values)
        finite_positions = np.flatnonzero(finite)
        worst_position = int(finite_positions[np.argmax(differences[finite_positions])])
        maximum_difference = float(differences[worst_position])
        trade_date = trade_dates[worst_position]
        trade_date_text = (
            trade_date.isoformat()
            if hasattr(trade_date, "isoformat")
            else str(trade_date)
        )
        daily_results[column] = {
            "maximum_absolute_difference": maximum_difference,
            "threshold": threshold,
            "passed": maximum_difference <= threshold,
            "worst_row": {
                "date_idx": int(date_indices[worst_position]),
                "trade_date": trade_date_text,
                "horizon_minutes": int(horizon_minutes[worst_position]),
                "recorded": float(recorded_values[worst_position]),
                "recomputed": float(recomputed_values[worst_position]),
                "absolute_difference": maximum_difference,
            },
        }

    failed_comparisons: list[dict[str, object]] = []
    if primary_result["passed"] is not True:
        failed_comparisons.append({"metric": "primary_ic", **primary_result})
    for key, result in horizon_results.items():
        if isinstance(result, dict) and result["passed"] is not True:
            failed_comparisons.append({"metric": key, **result})
    for metric, result in daily_results.items():
        if isinstance(result, dict) and result["passed"] is not True:
            worst_row = result["worst_row"]
            assert isinstance(worst_row, dict)
            failed_comparisons.append(
                {
                    "metric": metric,
                    "recorded": worst_row["recorded"],
                    "recomputed": worst_row["recomputed"],
                    "difference": result["maximum_absolute_difference"],
                    **result,
                }
            )
    if failed_comparisons:
        _metric_gate_error(failed_comparisons)

    gate = {
        "schema_version": METRIC_REPRODUCTION_GATE_SCHEMA_VERSION,
        "passed": True,
        "thresholds": metric_reproduction_thresholds(),
        "daily_rows": {
            "recorded_count": recorded_daily.height,
            "recomputed_count": recomputed_daily.height,
            "key_columns": list(_DAILY_KEY_COLUMNS),
            "metric_columns": list(METRIC_REPRODUCTION_DAILY_THRESHOLDS),
        },
        "primary_ic": primary_result,
        "horizons": horizon_results,
        "daily_metrics": daily_results,
    }
    return validate_metric_reproduction_gate(gate)


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
    validate_metric_reproduction_gate(metric_gate)
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
