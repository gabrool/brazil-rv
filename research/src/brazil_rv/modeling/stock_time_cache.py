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
METRIC_REPRODUCTION_GATE_SCHEMA_VERSION = 2
METRIC_REPRODUCTION_PRIMARY_IC_ABSOLUTE_TOLERANCE = 1e-6
METRIC_REPRODUCTION_HORIZON_IC_ABSOLUTE_TOLERANCE = 1e-6
METRIC_REPRODUCTION_DAILY_IC_ABSOLUTE_TOLERANCE = 1e-4
METRIC_REPRODUCTION_TURNOVER_ABSOLUTE_TOLERANCE = 5e-3
METRIC_REPRODUCTION_ECONOMIC_IDENTITY_ABSOLUTE_TOLERANCE = 1e-12
METRIC_REPRODUCTION_BLOCKING_DAILY_THRESHOLDS = MappingProxyType(
    {
        "spearman_ic": METRIC_REPRODUCTION_DAILY_IC_ABSOLUTE_TOLERANCE,
        "rank_target_pearson_ic": (METRIC_REPRODUCTION_DAILY_IC_ABSOLUTE_TOLERANCE),
        "one_way_turnover": METRIC_REPRODUCTION_TURNOVER_ABSOLUTE_TOLERANCE,
    }
)
METRIC_REPRODUCTION_ECONOMIC_METRICS = (
    "top_return",
    "bottom_return",
    "top_minus_bottom",
    "long_only_top",
)
METRIC_REPRODUCTION_ECONOMIC_SUMMARY_FIELDS = MappingProxyType(
    {metric: f"mean_{metric}" for metric in METRIC_REPRODUCTION_ECONOMIC_METRICS}
)
_DAILY_KEY_COLUMNS = ("date_idx", "horizon_minutes")
_DAILY_METRIC_COLUMNS = (
    "spearman_ic",
    "rank_target_pearson_ic",
    *METRIC_REPRODUCTION_ECONOMIC_METRICS,
    "one_way_turnover",
)
_RECORDED_DAILY_COLUMNS = (
    "trade_date",
    *_DAILY_KEY_COLUMNS,
    *_DAILY_METRIC_COLUMNS,
)
_RECOMPUTED_DAILY_COLUMNS = (*_DAILY_KEY_COLUMNS, *_DAILY_METRIC_COLUMNS)
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


def metric_reproduction_contract_metadata() -> dict[str, object]:
    return {
        "roles": {
            "primary_ic": "blocking",
            "horizon_mean_daily_spearman_ic": "blocking",
            "daily_spearman_ic": "blocking",
            "daily_rank_target_pearson_ic": "blocking",
            "daily_one_way_turnover": "blocking",
            "economic_returns": "diagnostic_only",
            "economic_identities": "blocking",
            "structural_checks": "blocking",
        },
        "blocking_thresholds": {
            "primary_ic_absolute_tolerance": (
                METRIC_REPRODUCTION_PRIMARY_IC_ABSOLUTE_TOLERANCE
            ),
            "horizon_mean_daily_spearman_ic_absolute_tolerance": (
                METRIC_REPRODUCTION_HORIZON_IC_ABSOLUTE_TOLERANCE
            ),
            "daily_metric_absolute_tolerances": dict(
                METRIC_REPRODUCTION_BLOCKING_DAILY_THRESHOLDS
            ),
            "economic_identity_absolute_tolerance": (
                METRIC_REPRODUCTION_ECONOMIC_IDENTITY_ABSOLUTE_TOLERANCE
            ),
        },
        "economic_diagnostic_metrics": list(METRIC_REPRODUCTION_ECONOMIC_METRICS),
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
        for field in (
            "mean_daily_spearman_ic",
            *METRIC_REPRODUCTION_ECONOMIC_SUMMARY_FIELDS.values(),
        ):
            _finite_metric(
                row.get(field),
                f"{location}.horizons.{row['horizon_minutes']}.{field}",
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


def _gate_difference(value: object, location: str) -> float:
    result = _gate_number(value, location)
    if result < 0.0:
        raise ValueError(f"Metric reproduction gate provenance is invalid: {location}")
    return result


def _validate_worst_key(
    value: object,
    *,
    location: str,
    value_fields: set[str],
) -> dict[str, object]:
    required = {
        "date_idx",
        "trade_date",
        "horizon_minutes",
        "absolute_difference",
        *value_fields,
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(
            f"Metric reproduction gate provenance is invalid: {location} schema"
        )
    date_index = value["date_idx"]
    horizon = value["horizon_minutes"]
    trade_date = value["trade_date"]
    if (
        not isinstance(date_index, int)
        or isinstance(date_index, bool)
        or not isinstance(trade_date, str)
        or not isinstance(horizon, int)
        or isinstance(horizon, bool)
        or horizon not in HORIZONS
    ):
        raise ValueError(
            f"Metric reproduction gate provenance is invalid: {location} key"
        )
    try:
        date.fromisoformat(trade_date)
    except ValueError as exc:
        raise ValueError(
            f"Metric reproduction gate provenance is invalid: {location} date"
        ) from exc
    return value


def _validate_blocking_comparison(
    value: object,
    *,
    location: str,
    threshold: float,
    difference_field: str,
    extra_fields: set[str] | None = None,
    with_values: bool = True,
) -> bool:
    required = {
        "role",
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
    if value["role"] != "blocking":
        raise ValueError(
            f"Metric reproduction gate provenance is invalid: {location} role"
        )
    stored_threshold = _gate_number(value["threshold"], f"{location}.threshold")
    difference = _gate_difference(
        value[difference_field], f"{location}.{difference_field}"
    )
    values_match = True
    if with_values:
        recorded = _gate_number(value["recorded"], f"{location}.recorded")
        recomputed = _gate_number(value["recomputed"], f"{location}.recomputed")
        values_match = difference == abs(recomputed - recorded)
    if stored_threshold != threshold or not values_match:
        raise ValueError(
            f"Metric reproduction gate provenance is invalid: {location} values"
        )
    expected_pass = difference <= threshold
    if not isinstance(value["passed"], bool) or value["passed"] is not expected_pass:
        raise ValueError(
            f"Metric reproduction gate provenance is invalid: {location} pass status"
        )
    return expected_pass


def _validate_blocking_daily_result(
    value: object,
    *,
    location: str,
    threshold: float,
) -> bool:
    passed = _validate_blocking_comparison(
        value,
        location=location,
        threshold=threshold,
        difference_field="maximum_absolute_difference",
        extra_fields={"worst_row"},
        with_values=False,
    )
    assert isinstance(value, dict)
    worst = _validate_worst_key(
        value["worst_row"],
        location=f"{location}.worst_row",
        value_fields={"recorded", "recomputed"},
    )
    recorded = _gate_number(worst["recorded"], f"{location}.worst_row.recorded")
    recomputed = _gate_number(worst["recomputed"], f"{location}.worst_row.recomputed")
    worst_difference = _gate_difference(
        worst["absolute_difference"],
        f"{location}.worst_row.absolute_difference",
    )
    maximum_difference = _gate_difference(
        value["maximum_absolute_difference"],
        f"{location}.maximum_absolute_difference",
    )
    if (
        worst_difference != abs(recomputed - recorded)
        or worst_difference != maximum_difference
    ):
        raise ValueError(
            f"Metric reproduction gate provenance is invalid: {location} worst row"
        )
    return passed


def _validate_economic_diagnostic(
    value: object,
    *,
    location: str,
) -> None:
    required = {
        "role",
        "maximum_absolute_difference",
        "mean_absolute_difference",
        "worst_row",
        "horizons",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(
            f"Metric reproduction gate provenance is invalid: {location} schema"
        )
    if value["role"] != "diagnostic_only":
        raise ValueError(
            f"Metric reproduction gate provenance is invalid: {location} role"
        )
    maximum = _gate_difference(
        value["maximum_absolute_difference"],
        f"{location}.maximum_absolute_difference",
    )
    mean = _gate_difference(
        value["mean_absolute_difference"],
        f"{location}.mean_absolute_difference",
    )
    if mean > maximum:
        raise ValueError(
            f"Metric reproduction gate provenance is invalid: {location} mean"
        )
    worst = _validate_worst_key(
        value["worst_row"],
        location=f"{location}.worst_row",
        value_fields={"recorded", "recomputed"},
    )
    recorded = _gate_number(worst["recorded"], f"{location}.worst_row.recorded")
    recomputed = _gate_number(worst["recomputed"], f"{location}.worst_row.recomputed")
    worst_difference = _gate_difference(
        worst["absolute_difference"],
        f"{location}.worst_row.absolute_difference",
    )
    if worst_difference != abs(recomputed - recorded) or worst_difference != maximum:
        raise ValueError(
            f"Metric reproduction gate provenance is invalid: {location} worst row"
        )
    horizons = value["horizons"]
    expected_horizon_keys = {f"{horizon}m" for horizon in HORIZONS}
    if not isinstance(horizons, dict) or set(horizons) != expected_horizon_keys:
        raise ValueError(
            f"Metric reproduction gate provenance is invalid: {location} horizons"
        )
    for horizon in HORIZONS:
        key = f"{horizon}m"
        result = horizons[key]
        if not isinstance(result, dict) or set(result) != {
            "horizon_minutes",
            "recorded_mean",
            "recomputed_mean",
            "absolute_difference",
        }:
            raise ValueError(
                f"Metric reproduction gate provenance is invalid: "
                f"{location}.horizons.{key} schema"
            )
        if result["horizon_minutes"] != horizon:
            raise ValueError(
                f"Metric reproduction gate provenance is invalid: "
                f"{location}.horizons.{key} key"
            )
        recorded_mean = _gate_number(
            result["recorded_mean"], f"{location}.horizons.{key}.recorded_mean"
        )
        recomputed_mean = _gate_number(
            result["recomputed_mean"],
            f"{location}.horizons.{key}.recomputed_mean",
        )
        difference = _gate_difference(
            result["absolute_difference"],
            f"{location}.horizons.{key}.absolute_difference",
        )
        if difference != abs(recomputed_mean - recorded_mean):
            raise ValueError(
                f"Metric reproduction gate provenance is invalid: "
                f"{location}.horizons.{key} values"
            )


def _validate_daily_identity_result(
    value: object,
    *,
    check: str,
    location: str,
    threshold: float,
) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "maximum_absolute_difference",
        "passed",
        "worst_row",
    }:
        raise ValueError(
            f"Metric reproduction gate provenance is invalid: {location} schema"
        )
    maximum = _gate_difference(
        value["maximum_absolute_difference"],
        f"{location}.maximum_absolute_difference",
    )
    expected_fields = (
        {"top_return", "long_only_top"}
        if check == "long_only_top_equals_top_return"
        else {"top_return", "bottom_return", "top_minus_bottom"}
    )
    worst = _validate_worst_key(
        value["worst_row"],
        location=f"{location}.worst_row",
        value_fields=expected_fields,
    )
    top = _gate_number(worst["top_return"], f"{location}.worst_row.top_return")
    if check == "long_only_top_equals_top_return":
        actual = _gate_number(
            worst["long_only_top"], f"{location}.worst_row.long_only_top"
        )
        expected = top
    else:
        bottom = _gate_number(
            worst["bottom_return"], f"{location}.worst_row.bottom_return"
        )
        actual = _gate_number(
            worst["top_minus_bottom"],
            f"{location}.worst_row.top_minus_bottom",
        )
        expected = top - bottom
    worst_difference = _gate_difference(
        worst["absolute_difference"],
        f"{location}.worst_row.absolute_difference",
    )
    if worst_difference != abs(actual - expected) or worst_difference != maximum:
        raise ValueError(
            f"Metric reproduction gate provenance is invalid: {location} worst row"
        )
    expected_pass = maximum <= threshold
    if not isinstance(value["passed"], bool) or value["passed"] is not expected_pass:
        raise ValueError(
            f"Metric reproduction gate provenance is invalid: {location} pass status"
        )
    return expected_pass


def _validate_summary_identity_result(
    value: object,
    *,
    check: str,
    location: str,
    threshold: float,
) -> bool:
    expected_fields = (
        {"mean_top_return", "mean_long_only_top"}
        if check == "long_only_top_equals_top_return"
        else {
            "mean_top_return",
            "mean_bottom_return",
            "mean_top_minus_bottom",
        }
    )
    required = {"absolute_difference", "passed", *expected_fields}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(
            f"Metric reproduction gate provenance is invalid: {location} schema"
        )
    top = _gate_number(value["mean_top_return"], f"{location}.mean_top_return")
    if check == "long_only_top_equals_top_return":
        actual = _gate_number(
            value["mean_long_only_top"], f"{location}.mean_long_only_top"
        )
        expected = top
    else:
        bottom = _gate_number(
            value["mean_bottom_return"], f"{location}.mean_bottom_return"
        )
        actual = _gate_number(
            value["mean_top_minus_bottom"],
            f"{location}.mean_top_minus_bottom",
        )
        expected = top - bottom
    difference = _gate_difference(
        value["absolute_difference"], f"{location}.absolute_difference"
    )
    if difference != abs(actual - expected):
        raise ValueError(
            f"Metric reproduction gate provenance is invalid: {location} values"
        )
    expected_pass = difference <= threshold
    if not isinstance(value["passed"], bool) or value["passed"] is not expected_pass:
        raise ValueError(
            f"Metric reproduction gate provenance is invalid: {location} pass status"
        )
    return expected_pass


def _validate_economic_identity_checks(value: object) -> bool:
    required = {"role", "threshold", "passed", "daily", "horizons"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(
            "Metric reproduction gate provenance is invalid: economic identity schema"
        )
    if value["role"] != "blocking":
        raise ValueError(
            "Metric reproduction gate provenance is invalid: economic identity role"
        )
    threshold = _gate_number(value["threshold"], "economic_identity_checks.threshold")
    if threshold != METRIC_REPRODUCTION_ECONOMIC_IDENTITY_ABSOLUTE_TOLERANCE:
        raise ValueError(
            "Metric reproduction gate provenance is invalid: "
            "economic identity threshold"
        )
    checks = (
        "long_only_top_equals_top_return",
        "top_minus_bottom_equals_top_return_minus_bottom_return",
    )
    passes: list[bool] = []
    daily = value["daily"]
    if not isinstance(daily, dict) or set(daily) != {"recorded", "recomputed"}:
        raise ValueError(
            "Metric reproduction gate provenance is invalid: economic daily schema"
        )
    for source in ("recorded", "recomputed"):
        source_results = daily[source]
        if not isinstance(source_results, dict) or set(source_results) != set(checks):
            raise ValueError(
                "Metric reproduction gate provenance is invalid: "
                f"economic daily {source} schema"
            )
        for check in checks:
            passes.append(
                _validate_daily_identity_result(
                    source_results[check],
                    check=check,
                    location=f"economic_identity_checks.daily.{source}.{check}",
                    threshold=threshold,
                )
            )
    horizons = value["horizons"]
    if not isinstance(horizons, dict) or set(horizons) != {
        "recorded",
        "recomputed",
    }:
        raise ValueError(
            "Metric reproduction gate provenance is invalid: economic horizon schema"
        )
    expected_horizon_keys = {f"{horizon}m" for horizon in HORIZONS}
    for source in ("recorded", "recomputed"):
        source_results = horizons[source]
        if not isinstance(source_results, dict) or set(source_results) != (
            expected_horizon_keys
        ):
            raise ValueError(
                "Metric reproduction gate provenance is invalid: "
                f"economic horizon {source} schema"
            )
        for horizon in HORIZONS:
            key = f"{horizon}m"
            result = source_results[key]
            if not isinstance(result, dict) or set(result) != {
                "horizon_minutes",
                *checks,
            }:
                raise ValueError(
                    "Metric reproduction gate provenance is invalid: "
                    f"economic horizon {source}.{key} schema"
                )
            if result["horizon_minutes"] != horizon:
                raise ValueError(
                    "Metric reproduction gate provenance is invalid: "
                    f"economic horizon {source}.{key} key"
                )
            for check in checks:
                passes.append(
                    _validate_summary_identity_result(
                        result[check],
                        check=check,
                        location=(
                            f"economic_identity_checks.horizons.{source}.{key}.{check}"
                        ),
                        threshold=threshold,
                    )
                )
    expected_pass = all(passes)
    if not isinstance(value["passed"], bool) or value["passed"] is not expected_pass:
        raise ValueError(
            "Metric reproduction gate provenance is invalid: "
            "economic identity pass status"
        )
    return expected_pass


def validate_metric_reproduction_gate(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Metric reproduction gate provenance is invalid: schema")
    version = value.get("schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != METRIC_REPRODUCTION_GATE_SCHEMA_VERSION
    ):
        raise ValueError("Metric reproduction gate provenance is invalid: version")
    required = {
        "schema_version",
        "passed",
        "contract",
        "structural_checks",
        "blocking_comparisons",
        "economic_diagnostics",
        "economic_identity_checks",
    }
    if set(value) != required:
        raise ValueError("Metric reproduction gate provenance is invalid: schema")
    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Metric reproduction gate provenance is invalid: JSON"
        ) from exc
    if value["contract"] != metric_reproduction_contract_metadata():
        raise ValueError("Metric reproduction gate provenance is invalid: contract")

    structural = value["structural_checks"]
    if not isinstance(structural, dict) or set(structural) != {
        "role",
        "passed",
        "recorded_count",
        "recomputed_count",
        "key_columns",
        "metric_columns",
        "horizons",
    }:
        raise ValueError(
            "Metric reproduction gate provenance is invalid: structural schema"
        )
    recorded_count = structural["recorded_count"]
    recomputed_count = structural["recomputed_count"]
    structural_pass = (
        structural["role"] == "blocking"
        and structural["passed"] is True
        and isinstance(recorded_count, int)
        and not isinstance(recorded_count, bool)
        and recorded_count > 0
        and isinstance(recomputed_count, int)
        and not isinstance(recomputed_count, bool)
        and recomputed_count == recorded_count
        and structural["key_columns"] == list(_DAILY_KEY_COLUMNS)
        and structural["metric_columns"] == list(_DAILY_METRIC_COLUMNS)
        and structural["horizons"] == list(HORIZONS)
    )
    if not structural_pass:
        raise ValueError(
            "Metric reproduction gate provenance is invalid: structural metadata"
        )

    blocking = value["blocking_comparisons"]
    if not isinstance(blocking, dict) or set(blocking) != {
        "primary_ic",
        "horizons",
        "daily_metrics",
    }:
        raise ValueError(
            "Metric reproduction gate provenance is invalid: blocking schema"
        )
    passes = [
        _validate_blocking_comparison(
            blocking["primary_ic"],
            location="blocking_comparisons.primary_ic",
            threshold=METRIC_REPRODUCTION_PRIMARY_IC_ABSOLUTE_TOLERANCE,
            difference_field="absolute_difference",
        )
    ]
    horizons = blocking["horizons"]
    expected_horizon_keys = {f"{horizon}m" for horizon in HORIZONS}
    if not isinstance(horizons, dict) or set(horizons) != expected_horizon_keys:
        raise ValueError(
            "Metric reproduction gate provenance is invalid: blocking horizons"
        )
    for horizon in HORIZONS:
        key = f"{horizon}m"
        row = horizons[key]
        passed = _validate_blocking_comparison(
            row,
            location=f"blocking_comparisons.horizons.{key}",
            threshold=METRIC_REPRODUCTION_HORIZON_IC_ABSOLUTE_TOLERANCE,
            difference_field="absolute_difference",
            extra_fields={"horizon_minutes"},
        )
        if not isinstance(row, dict) or row["horizon_minutes"] != horizon:
            raise ValueError(
                "Metric reproduction gate provenance is invalid: "
                f"blocking horizons.{key} key"
            )
        passes.append(passed)
    daily_metrics = blocking["daily_metrics"]
    if not isinstance(daily_metrics, dict) or set(daily_metrics) != set(
        METRIC_REPRODUCTION_BLOCKING_DAILY_THRESHOLDS
    ):
        raise ValueError(
            "Metric reproduction gate provenance is invalid: blocking daily schema"
        )
    for metric, threshold in METRIC_REPRODUCTION_BLOCKING_DAILY_THRESHOLDS.items():
        passes.append(
            _validate_blocking_daily_result(
                daily_metrics[metric],
                location=f"blocking_comparisons.daily_metrics.{metric}",
                threshold=threshold,
            )
        )

    diagnostics = value["economic_diagnostics"]
    if not isinstance(diagnostics, dict) or set(diagnostics) != set(
        METRIC_REPRODUCTION_ECONOMIC_METRICS
    ):
        raise ValueError(
            "Metric reproduction gate provenance is invalid: economic diagnostics"
        )
    for metric in METRIC_REPRODUCTION_ECONOMIC_METRICS:
        _validate_economic_diagnostic(
            diagnostics[metric], location=f"economic_diagnostics.{metric}"
        )

    identity_pass = _validate_economic_identity_checks(
        value["economic_identity_checks"]
    )
    expected_pass = structural_pass and identity_pass and all(passes)
    if not isinstance(value["passed"], bool) or value["passed"] is not expected_pass:
        raise ValueError(
            "Metric reproduction gate provenance is invalid: overall pass status"
        )
    if value["passed"] is not True:
        raise ValueError("Prediction cache lacks metric parity")
    return value


def _daily_economic_identity_results(
    source: str,
    arrays: dict[str, np.ndarray],
    trade_dates: list[object],
    date_indices: list[object],
    horizon_minutes: list[object],
    failures: list[dict[str, object]],
) -> dict[str, object]:
    results: dict[str, object] = {}
    definitions = (
        (
            "long_only_top_equals_top_return",
            ("top_return", "long_only_top"),
        ),
        (
            "top_minus_bottom_equals_top_return_minus_bottom_return",
            ("top_return", "bottom_return", "top_minus_bottom"),
        ),
    )
    for check, fields in definitions:
        values = tuple(arrays[field] for field in fields)
        if check == "long_only_top_equals_top_return":
            valid_pattern = np.array_equal(np.isnan(values[0]), np.isnan(values[1]))
            finite = np.isfinite(values[0]) & np.isfinite(values[1])
            differences = np.abs(values[1] - values[0])
        else:
            expected_nan = np.isnan(values[0]) | np.isnan(values[1])
            valid_pattern = np.array_equal(expected_nan, np.isnan(values[2]))
            finite = (
                np.isfinite(values[0]) & np.isfinite(values[1]) & np.isfinite(values[2])
            )
            differences = np.abs(values[2] - (values[0] - values[1]))
        if not valid_pattern:
            failures.append(
                {
                    "check": "economic_identity_nan_pattern",
                    "source": source,
                    "identity": check,
                }
            )
            continue
        finite_positions = np.flatnonzero(finite)
        if finite_positions.size == 0:
            failures.append(
                {
                    "check": "economic_identity_has_no_finite_value",
                    "source": source,
                    "identity": check,
                }
            )
            continue
        worst_position = int(finite_positions[np.argmax(differences[finite_positions])])
        maximum_difference = float(differences[worst_position])
        trade_date = trade_dates[worst_position]
        worst_row: dict[str, object] = {
            "date_idx": int(date_indices[worst_position]),
            "trade_date": (
                trade_date.isoformat()
                if hasattr(trade_date, "isoformat")
                else str(trade_date)
            ),
            "horizon_minutes": int(horizon_minutes[worst_position]),
            "absolute_difference": maximum_difference,
        }
        for field, field_values in zip(fields, values, strict=True):
            worst_row[field] = float(field_values[worst_position])
        passed = (
            maximum_difference
            <= METRIC_REPRODUCTION_ECONOMIC_IDENTITY_ABSOLUTE_TOLERANCE
        )
        results[check] = {
            "maximum_absolute_difference": maximum_difference,
            "passed": passed,
            "worst_row": worst_row,
        }
        if not passed:
            failures.append(
                {
                    "check": "economic_identity",
                    "source": source,
                    "scope": "daily",
                    "identity": check,
                    "difference": maximum_difference,
                    "threshold": (
                        METRIC_REPRODUCTION_ECONOMIC_IDENTITY_ABSOLUTE_TOLERANCE
                    ),
                    "worst_row": worst_row,
                }
            )
    return results


def _summary_economic_identity_results(
    source: str,
    horizons: dict[int, dict[str, object]],
    failures: list[dict[str, object]],
) -> dict[str, object]:
    results: dict[str, object] = {}
    for horizon in HORIZONS:
        row = horizons[horizon]
        top = float(row["mean_top_return"])
        bottom = float(row["mean_bottom_return"])
        spread = float(row["mean_top_minus_bottom"])
        long_only = float(row["mean_long_only_top"])
        long_difference = abs(long_only - top)
        spread_difference = abs(spread - (top - bottom))
        long_passed = (
            long_difference <= METRIC_REPRODUCTION_ECONOMIC_IDENTITY_ABSOLUTE_TOLERANCE
        )
        spread_passed = (
            spread_difference
            <= METRIC_REPRODUCTION_ECONOMIC_IDENTITY_ABSOLUTE_TOLERANCE
        )
        key = f"{horizon}m"
        results[key] = {
            "horizon_minutes": horizon,
            "long_only_top_equals_top_return": {
                "mean_top_return": top,
                "mean_long_only_top": long_only,
                "absolute_difference": long_difference,
                "passed": long_passed,
            },
            "top_minus_bottom_equals_top_return_minus_bottom_return": {
                "mean_top_return": top,
                "mean_bottom_return": bottom,
                "mean_top_minus_bottom": spread,
                "absolute_difference": spread_difference,
                "passed": spread_passed,
            },
        }
        for identity, difference, passed in (
            (
                "long_only_top_equals_top_return",
                long_difference,
                long_passed,
            ),
            (
                "top_minus_bottom_equals_top_return_minus_bottom_return",
                spread_difference,
                spread_passed,
            ),
        ):
            if not passed:
                failures.append(
                    {
                        "check": "economic_identity",
                        "source": source,
                        "scope": "horizon_summary",
                        "identity": identity,
                        "horizon_minutes": horizon,
                        "difference": difference,
                        "threshold": (
                            METRIC_REPRODUCTION_ECONOMIC_IDENTITY_ABSOLUTE_TOLERANCE
                        ),
                    }
                )
    return results


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
        "role": "blocking",
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
            "role": "blocking",
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
    for column in _DAILY_METRIC_COLUMNS:
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

    trade_dates = recorded_daily.get_column("trade_date").to_list()
    date_indices = recorded_daily.get_column("date_idx").to_list()
    horizon_minutes = recorded_daily.get_column("horizon_minutes").to_list()
    blocking_daily_results: dict[str, object] = {}
    for column, threshold in METRIC_REPRODUCTION_BLOCKING_DAILY_THRESHOLDS.items():
        recorded_values, recomputed_values = daily_arrays[column]
        finite = np.isfinite(recorded_values)
        differences = np.abs(recomputed_values - recorded_values)
        finite_positions = np.flatnonzero(finite)
        worst_position = int(finite_positions[np.argmax(differences[finite_positions])])
        maximum_difference = float(differences[worst_position])
        trade_date = trade_dates[worst_position]
        blocking_daily_results[column] = {
            "role": "blocking",
            "maximum_absolute_difference": maximum_difference,
            "threshold": threshold,
            "passed": maximum_difference <= threshold,
            "worst_row": {
                "date_idx": int(date_indices[worst_position]),
                "trade_date": (
                    trade_date.isoformat()
                    if hasattr(trade_date, "isoformat")
                    else str(trade_date)
                ),
                "horizon_minutes": int(horizon_minutes[worst_position]),
                "recorded": float(recorded_values[worst_position]),
                "recomputed": float(recomputed_values[worst_position]),
                "absolute_difference": maximum_difference,
            },
        }

    economic_diagnostics: dict[str, object] = {}
    for metric in METRIC_REPRODUCTION_ECONOMIC_METRICS:
        recorded_values, recomputed_values = daily_arrays[metric]
        finite_positions = np.flatnonzero(np.isfinite(recorded_values))
        differences = np.abs(recomputed_values - recorded_values)
        worst_position = int(finite_positions[np.argmax(differences[finite_positions])])
        maximum_difference = float(differences[worst_position])
        mean_difference = float(np.mean(differences[finite_positions]))
        trade_date = trade_dates[worst_position]
        summary_field = METRIC_REPRODUCTION_ECONOMIC_SUMMARY_FIELDS[metric]
        per_horizon: dict[str, object] = {}
        for horizon in HORIZONS:
            recorded_mean = float(recorded_horizons[horizon][summary_field])
            recomputed_mean = float(recomputed_horizons[horizon][summary_field])
            per_horizon[f"{horizon}m"] = {
                "horizon_minutes": horizon,
                "recorded_mean": recorded_mean,
                "recomputed_mean": recomputed_mean,
                "absolute_difference": abs(recomputed_mean - recorded_mean),
            }
        economic_diagnostics[metric] = {
            "role": "diagnostic_only",
            "maximum_absolute_difference": maximum_difference,
            "mean_absolute_difference": mean_difference,
            "worst_row": {
                "date_idx": int(date_indices[worst_position]),
                "trade_date": (
                    trade_date.isoformat()
                    if hasattr(trade_date, "isoformat")
                    else str(trade_date)
                ),
                "horizon_minutes": int(horizon_minutes[worst_position]),
                "recorded": float(recorded_values[worst_position]),
                "recomputed": float(recomputed_values[worst_position]),
                "absolute_difference": maximum_difference,
            },
            "horizons": per_horizon,
        }

    failures = []
    recorded_arrays = {
        metric: daily_arrays[metric][0]
        for metric in METRIC_REPRODUCTION_ECONOMIC_METRICS
    }
    recomputed_arrays = {
        metric: daily_arrays[metric][1]
        for metric in METRIC_REPRODUCTION_ECONOMIC_METRICS
    }
    daily_identity_results = {
        "recorded": _daily_economic_identity_results(
            "recorded",
            recorded_arrays,
            trade_dates,
            date_indices,
            horizon_minutes,
            failures,
        ),
        "recomputed": _daily_economic_identity_results(
            "recomputed",
            recomputed_arrays,
            trade_dates,
            date_indices,
            horizon_minutes,
            failures,
        ),
    }
    summary_identity_results = {
        "recorded": _summary_economic_identity_results(
            "recorded", recorded_horizons, failures
        ),
        "recomputed": _summary_economic_identity_results(
            "recomputed", recomputed_horizons, failures
        ),
    }

    if primary_result["passed"] is not True:
        failures.append({"metric": "primary_ic", **primary_result})
    for key, result in horizon_results.items():
        if isinstance(result, dict) and result["passed"] is not True:
            failures.append({"metric": key, **result})
    for metric, result in blocking_daily_results.items():
        if isinstance(result, dict) and result["passed"] is not True:
            worst_row = result["worst_row"]
            assert isinstance(worst_row, dict)
            failures.append(
                {
                    "metric": metric,
                    "recorded": worst_row["recorded"],
                    "recomputed": worst_row["recomputed"],
                    "difference": result["maximum_absolute_difference"],
                    **result,
                }
            )
    if failures:
        _metric_gate_error(failures)

    economic_identity_checks = {
        "role": "blocking",
        "threshold": METRIC_REPRODUCTION_ECONOMIC_IDENTITY_ABSOLUTE_TOLERANCE,
        "passed": True,
        "daily": daily_identity_results,
        "horizons": summary_identity_results,
    }
    gate = {
        "schema_version": METRIC_REPRODUCTION_GATE_SCHEMA_VERSION,
        "passed": True,
        "contract": metric_reproduction_contract_metadata(),
        "structural_checks": {
            "role": "blocking",
            "passed": True,
            "recorded_count": recorded_daily.height,
            "recomputed_count": recomputed_daily.height,
            "key_columns": list(_DAILY_KEY_COLUMNS),
            "metric_columns": list(_DAILY_METRIC_COLUMNS),
            "horizons": list(HORIZONS),
        },
        "blocking_comparisons": {
            "primary_ic": primary_result,
            "horizons": horizon_results,
            "daily_metrics": blocking_daily_results,
        },
        "economic_diagnostics": economic_diagnostics,
        "economic_identity_checks": economic_identity_checks,
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
