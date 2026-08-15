from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from brazil_rv.preprocessing.contract import CONTRACT_VERSION

from .contract import (
    EXPECTED_DECISIONS_PER_DATE,
    EXPECTED_SPLIT_DATE_COUNTS,
    EXPECTED_SPLIT_SAMPLE_COUNTS,
    HORIZONS,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
    TCN_READOUTS,
)
from .data import (
    feature_store_identity,
    int64_identity_sha256,
    sample_window_metadata,
)
from .evaluate import load_current_neural_run
from .horizon_diagnostics import (
    BOOTSTRAP_SEED,
    PERMUTATION_SEED,
    RIDGE_PENALTIES,
)
from .metrics import moving_block_bootstrap
from .model import count_trainable_parameters
from .stage_conclusions import (
    build_context_training_summary,
    build_hypothesis_summary,
    stage_summary_markdown,
)


CONTEXT_FAMILIES = ("wdo", "br_rates", "us_rates")
FROZEN_CANDIDATES = (
    *(f"block_{index}" for index in range(1, 7)),
    "uniform_mean",
    "concatenated",
    "final_post_fusion",
    "incumbent_predictions",
)
GRADIENT_GROUPS = (
    "input_projection",
    *(f"block_{index}" for index in range(1, 7)),
    "slow_projection",
    "peer_adapter",
    "shared_fusion",
    "prediction_head",
)
RESIDUAL_PROBES = ("slow", "selected_peer", "macro_interaction", "combined")
HORIZON_PAIRS = {(30, 60), (30, 120), (60, 120)}
COMPARISON_SEEDS = {
    **{f"horizon_multiscale_vs_final_seed{seed}": seed for seed in (11, 29, 47)},
    "shared_multiscale_vs_final_seed29": 29,
    "final_score_mlp_vs_final_seed29": 29,
    "horizon_multiscale_vs_shared_seed29": 29,
    "horizon_multiscale_vs_final_three_seed": 0,
}
TARGET_HORIZON_PAIRS = {
    (left, right) for left in HORIZONS for right in HORIZONS if left <= right
}
GATE_RUNS = {
    ("shared_multiscale_seed29", 29),
    *((f"horizon_multiscale_seed{seed}", seed) for seed in (11, 29, 47)),
    ("horizon_multiscale_three_seed", 0),
}


# JSON, CSV, and Parquet round trips preserve these metrics well below this bound.
SERIALIZATION_TOLERANCE = 1e-12
# Gate weights and entropies are emitted from float32 model tensors.
FLOAT32_DIAGNOSTIC_TOLERANCE = 1e-6


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _read_csv(path: Path) -> pl.DataFrame:
    try:
        return pl.read_csv(path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise ValueError(f"Invalid CSV artifact {path}: {error}") from error


def _read_parquet(path: Path) -> pl.DataFrame:
    try:
        return pl.read_parquet(path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise ValueError(f"Invalid Parquet artifact {path}: {error}") from error


def _require_keys(value: dict[str, Any], required: set[str], path: Path) -> None:
    missing = required - set(value)
    if missing:
        raise ValueError(f"Artifact {path} is missing keys: {sorted(missing)}")


def _require_columns(frame: pl.DataFrame, required: set[str], path: Path) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Artifact {path} is missing columns: {sorted(missing)}")
    if frame.is_empty():
        raise ValueError(f"Artifact is empty: {path}")


def _require_finite_columns(
    frame: pl.DataFrame, columns: tuple[str, ...], path: Path
) -> None:
    for column in columns:
        series = frame.get_column(column)
        if series.null_count():
            raise ValueError(f"Artifact {path} has null values in {column}")
        try:
            values = series.cast(pl.Float64, strict=True).to_numpy()
        except (TypeError, ValueError, pl.exceptions.PolarsError) as error:
            raise ValueError(
                f"Artifact {path} has non-numeric values in {column}"
            ) from error
        if not np.isfinite(values).all():
            raise ValueError(f"Artifact {path} has non-finite values in {column}")


def _require_equal(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(f"{name} differs: expected {expected!r}, found {actual!r}")


def _require_close(
    name: str,
    actual: object,
    expected: object,
    *,
    tolerance: float = SERIALIZATION_TOLERANCE,
) -> None:
    actual_value = _finite(actual, name)
    expected_value = _finite(expected, f"expected {name}")
    if not np.isclose(actual_value, expected_value, rtol=0.0, atol=tolerance):
        raise ValueError(
            f"{name} differs: expected {expected_value!r}, found {actual_value!r}"
        )


def _require_unique_keys(
    frame: pl.DataFrame, columns: tuple[str, ...], path: Path
) -> None:
    if frame.select(*columns).n_unique() != frame.height:
        raise ValueError(f"Artifact {path} has duplicate keys {columns}")


def _require_bounds(
    frame: pl.DataFrame,
    column: str,
    lower: float,
    upper: float,
    path: Path,
) -> None:
    _require_finite_columns(frame, (column,), path)
    values = frame.get_column(column).cast(pl.Float64)
    if ((values < lower) | (values > upper)).any():
        raise ValueError(f"Artifact {path} has {column} outside [{lower}, {upper}]")


def _require_positive_integers(
    frame: pl.DataFrame, columns: tuple[str, ...], path: Path
) -> None:
    for column in columns:
        series = frame.get_column(column)
        if series.null_count():
            raise ValueError(f"Artifact {path} has null values in {column}")
        values = series.cast(pl.Float64, strict=True).to_numpy()
        if (
            not np.isfinite(values).all()
            or (values <= 0).any()
            or not np.equal(values, np.floor(values)).all()
        ):
            raise ValueError(
                f"Artifact {path} has invalid positive integers in {column}"
            )


def _require_correlation_matrix(value: object, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    if not np.allclose(np.diag(matrix), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must have a unit diagonal")
    if (matrix < -1.0 - 1e-12).any() or (matrix > 1.0 + 1e-12).any():
        raise ValueError(f"{name} contains an invalid correlation")
    return matrix


def _finite(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not numeric: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} is non-finite")
    return result


def _finite_array(value: object, name: str, shape: tuple[int, ...]) -> None:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")


def validate_split_contract(rows: pl.DataFrame, split: str) -> dict[str, object]:
    specs = {
        "train": (TRAIN_START, TRAIN_END),
        "validation": (VALIDATION_START, VALIDATION_END),
    }
    if split not in specs:
        raise ValueError(f"Unsupported stage split: {split}")
    if rows.is_empty():
        raise ValueError(f"{split} rows are empty")
    dates = rows.get_column("trade_date")
    if dates.max() >= TEST_START:
        raise ValueError(f"{split} rows include the held-out test period")
    expected_start, expected_end = specs[split]
    _require_equal(f"{split} start", dates.min(), expected_start)
    _require_equal(f"{split} end", dates.max(), expected_end)
    _require_equal(
        f"{split} date count",
        dates.n_unique(),
        EXPECTED_SPLIT_DATE_COUNTS[split],
    )
    _require_equal(
        f"{split} sample count",
        rows.height,
        EXPECTED_SPLIT_SAMPLE_COUNTS[split],
    )
    if rows.get_column("sample_id").n_unique() != rows.height:
        raise ValueError(f"{split} sample identifiers are not unique")
    decisions = rows.group_by("trade_date").agg(pl.col("decision_idx").sort())
    expected = list(range(EXPECTED_DECISIONS_PER_DATE))
    if any(values.to_list() != expected for values in decisions["decision_idx"]):
        raise ValueError(f"{split} dates must contain decisions 0..54 exactly once")
    return sample_window_metadata(rows, split)


def validate_preflight(
    path: Path,
    store: Path,
    train_rows: pl.DataFrame,
    validation_rows: pl.DataFrame,
) -> None:
    summary = read_json_object(path)
    _require_keys(
        summary,
        {
            "feature_store",
            "test_accessed",
            "parameter_counts",
            "added_parameter_counts",
            "tap_receptive_field_minutes",
            "readout_modes",
            "split_contract",
        },
        path,
    )
    _require_equal(
        "feature-store identity",
        summary["feature_store"],
        feature_store_identity(store),
    )
    _require_equal(
        "feature-store contract",
        summary["feature_store"]["contract_version"],
        CONTRACT_VERSION,
    )
    _require_equal("preflight test access", summary["test_accessed"], False)
    _require_equal("preflight readouts", summary["readout_modes"], list(TCN_READOUTS))
    _require_equal(
        "tap receptive fields",
        summary["tap_receptive_field_minutes"],
        [15, 35, 75, 155, 315, 635],
    )
    counts = summary["parameter_counts"]
    additions = summary["added_parameter_counts"]
    _require_equal("parameter-count modes", set(counts), set(TCN_READOUTS))
    _require_equal(
        "added parameter counts",
        additions,
        {
            "final": 0,
            "shared_multiscale": 6,
            "horizon_multiscale": 18,
            "final_score_mlp": 17,
        },
    )
    _require_equal(
        "preflight split contract",
        summary["split_contract"],
        {
            "train": validate_split_contract(train_rows, "train"),
            "validation": validate_split_contract(validation_rows, "validation"),
        },
    )


def validate_target_basis(output_dir: Path) -> None:
    summary_path = output_dir / "target_basis_summary.json"
    summary = read_json_object(summary_path)
    _require_keys(
        summary,
        {
            "train_end",
            "date_count",
            "valid_counts_by_horizon",
            "coverage_fraction_by_horizon",
            "complete_case_count",
            "pooled_target_correlation",
            "complete_case_target_covariance",
            "complete_case_correlation_sensitivity",
            "eigenvalues",
            "eigenvectors_columns",
            "variance_shares",
            "fixed_basis_rows",
            "fixed_basis_variance",
            "raw_return_headline_correlation",
            "raw_complete_case_count",
        },
        summary_path,
    )
    _require_equal("target audit train end", summary["train_end"], str(TRAIN_END))
    _require_equal(
        "target audit date count",
        summary["date_count"],
        EXPECTED_SPLIT_DATE_COUNTS["train"],
    )
    pooled_correlation = _require_correlation_matrix(
        summary["pooled_target_correlation"], "pooled target correlation"
    )
    _require_correlation_matrix(
        summary["complete_case_correlation_sensitivity"],
        "complete-case target correlation",
    )
    _require_correlation_matrix(
        summary["raw_return_headline_correlation"], "raw-return correlation"
    )
    covariance = np.asarray(
        summary["complete_case_target_covariance"], dtype=np.float64
    )
    if (
        covariance.shape != (3, 3)
        or not np.isfinite(covariance).all()
        or not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12)
        or (np.diag(covariance) <= 0.0).any()
    ):
        raise ValueError("Complete-case covariance is invalid")
    eigenvalues = np.asarray(summary["eigenvalues"], dtype=np.float64)
    variance_shares = np.asarray(summary["variance_shares"], dtype=np.float64)
    fixed_variance = np.asarray(summary["fixed_basis_variance"], dtype=np.float64)
    for name, values in (
        ("eigenvalues", eigenvalues),
        ("variance shares", variance_shares),
        ("fixed-basis variance", fixed_variance),
    ):
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError(f"{name} must contain three finite values")
    if (eigenvalues < -1e-12).any() or np.any(np.diff(eigenvalues) > 1e-12):
        raise ValueError("Target eigenvalues are invalid")
    if (variance_shares < -1e-12).any() or not np.isclose(
        variance_shares.sum(), 1.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError("Target variance shares must be nonnegative and sum to one")
    if (fixed_variance < -1e-12).any():
        raise ValueError("Fixed-basis variance must be nonnegative")
    fixed_basis = np.asarray(summary["fixed_basis_rows"], dtype=np.float64)
    eigenvectors = np.asarray(summary["eigenvectors_columns"], dtype=np.float64)
    for name, matrix, product in (
        ("fixed basis", fixed_basis, fixed_basis @ fixed_basis.T),
        ("eigenvectors", eigenvectors, eigenvectors.T @ eigenvectors),
    ):
        if matrix.shape != (3, 3) or not np.allclose(
            product, np.eye(3), rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"{name} must be orthonormal")
    complete_scale = np.sqrt(np.diag(covariance))
    expected_complete_correlation = covariance / np.outer(
        complete_scale, complete_scale
    )
    if not np.allclose(
        summary["complete_case_correlation_sensitivity"],
        expected_complete_correlation,
        rtol=0.0,
        atol=SERIALIZATION_TOLERANCE,
    ):
        raise ValueError("Complete-case correlation disagrees with covariance")
    expected_eigenvalues = np.linalg.eigvalsh(pooled_correlation)[::-1]
    if not np.allclose(
        eigenvalues,
        expected_eigenvalues,
        rtol=0.0,
        atol=SERIALIZATION_TOLERANCE,
    ) or not np.allclose(
        pooled_correlation @ eigenvectors,
        eigenvectors * eigenvalues,
        rtol=0.0,
        atol=SERIALIZATION_TOLERANCE,
    ):
        raise ValueError("Target eigendecomposition disagrees with correlation")
    if not np.allclose(
        variance_shares,
        eigenvalues / eigenvalues.sum(),
        rtol=0.0,
        atol=SERIALIZATION_TOLERANCE,
    ):
        raise ValueError("Target variance shares disagree with eigenvalues")
    if not np.allclose(
        fixed_variance,
        np.diag(fixed_basis @ covariance @ fixed_basis.T),
        rtol=0.0,
        atol=SERIALIZATION_TOLERANCE,
    ):
        raise ValueError("Fixed-basis variance disagrees with covariance")
    for name in ("complete_case_count", "raw_complete_case_count"):
        if not isinstance(summary[name], int) or summary[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")

    pair_path = output_dir / "target_pairwise.csv"
    pairwise = _read_csv(pair_path)
    _require_columns(
        pairwise,
        {
            "scope",
            "left_horizon",
            "right_horizon",
            "valid_count",
            "covariance",
            "correlation",
        },
        pair_path,
    )
    _require_equal(
        "target pairwise row count", pairwise.height, 3 * len(TARGET_HORIZON_PAIRS)
    )
    _require_unique_keys(
        pairwise, ("scope", "left_horizon", "right_horizon"), pair_path
    )
    _require_finite_columns(pairwise, ("covariance",), pair_path)
    _require_bounds(pairwise, "correlation", -1.0, 1.0, pair_path)
    _require_positive_integers(pairwise, ("valid_count",), pair_path)
    for scope in ("pooled", "equal_date", "equal_decision"):
        selected = pairwise.filter(pl.col("scope") == scope)
        _require_equal(
            f"{scope} horizon pairs",
            set(selected.select("left_horizon", "right_horizon").iter_rows()),
            TARGET_HORIZON_PAIRS,
        )
        for horizon in HORIZONS:
            diagonal = selected.filter(
                (pl.col("left_horizon") == horizon)
                & (pl.col("right_horizon") == horizon)
            )["correlation"][0]
            _require_close(f"{scope} correlation diagonal {horizon}", diagonal, 1.0)

    pooled_rows = pairwise.filter(pl.col("scope") == "pooled")
    for row in pooled_rows.iter_rows(named=True):
        left = HORIZONS.index(int(row["left_horizon"]))
        right = HORIZONS.index(int(row["right_horizon"]))
        _require_close(
            f"pooled summary correlation {row['left_horizon']}/{row['right_horizon']}",
            pooled_correlation[left, right],
            row["correlation"],
        )
    counts = summary["valid_counts_by_horizon"]
    coverage = summary["coverage_fraction_by_horizon"]
    _require_equal("target count horizons", set(counts), {str(h) for h in HORIZONS})
    _require_equal(
        "target coverage horizons", set(coverage), {str(h) for h in HORIZONS}
    )
    for horizon in HORIZONS:
        diagonal = pooled_rows.filter(
            (pl.col("left_horizon") == horizon) & (pl.col("right_horizon") == horizon)
        )["valid_count"][0]
        _require_equal(f"target valid count {horizon}", counts[str(horizon)], diagonal)
        value = _finite(coverage[str(horizon)], f"target coverage {horizon}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Target coverage {horizon} is outside [0, 1]")

    date_path = output_dir / "target_basis_by_date.parquet"
    by_date = _read_parquet(date_path)
    _require_columns(
        by_date,
        {
            "trade_date",
            "left_horizon",
            "right_horizon",
            "valid_count",
            "covariance",
            "correlation",
        },
        date_path,
    )
    _require_equal(
        "target by-date rows",
        by_date.height,
        EXPECTED_SPLIT_DATE_COUNTS["train"] * len(TARGET_HORIZON_PAIRS),
    )
    _require_equal(
        "target by-date count",
        by_date["trade_date"].n_unique(),
        EXPECTED_SPLIT_DATE_COUNTS["train"],
    )
    _require_equal("target by-date start", by_date["trade_date"].min(), TRAIN_START)
    if by_date["trade_date"].max() > TRAIN_END:
        raise ValueError("Target audit contains post-training dates")
    _require_unique_keys(
        by_date, ("trade_date", "left_horizon", "right_horizon"), date_path
    )
    _require_positive_integers(by_date, ("valid_count",), date_path)
    _require_finite_columns(by_date, ("covariance",), date_path)
    _require_bounds(by_date, "correlation", -1.0, 1.0, date_path)

    decision_path = output_dir / "target_basis_by_decision.csv"
    by_decision = _read_csv(decision_path)
    _require_columns(
        by_decision,
        {
            "decision_idx",
            "left_horizon",
            "right_horizon",
            "valid_count",
            "covariance",
            "correlation",
        },
        decision_path,
    )
    _require_equal(
        "target decision rows",
        by_decision.height,
        EXPECTED_DECISIONS_PER_DATE * len(TARGET_HORIZON_PAIRS),
    )
    _require_equal(
        "target decisions",
        set(by_decision["decision_idx"]),
        set(range(EXPECTED_DECISIONS_PER_DATE)),
    )
    _require_unique_keys(
        by_decision,
        ("decision_idx", "left_horizon", "right_horizon"),
        decision_path,
    )
    _require_positive_integers(by_decision, ("valid_count",), decision_path)
    _require_finite_columns(by_decision, ("covariance",), decision_path)
    _require_bounds(by_decision, "correlation", -1.0, 1.0, decision_path)


def validate_frozen_probes(output_dir: Path) -> None:
    table_path = output_dir / "frozen_block_probes.csv"
    frame = _read_csv(table_path)
    _require_columns(
        frame,
        {
            "candidate",
            "horizon_minutes",
            "validation_ic",
            "selected_penalty",
            "feature_dim",
            "valid_count",
            "coefficient_norm",
        },
        table_path,
    )
    _require_equal("frozen candidates", set(frame["candidate"]), set(FROZEN_CANDIDATES))
    _require_equal("frozen row count", frame.height, len(FROZEN_CANDIDATES) * 4)
    _require_unique_keys(frame, ("candidate", "horizon_minutes"), table_path)
    _require_bounds(frame, "validation_ic", -1.0, 1.0, table_path)
    _require_positive_integers(frame, ("feature_dim",), table_path)
    for candidate in FROZEN_CANDIDATES:
        _require_equal(
            f"{candidate} horizons",
            set(frame.filter(pl.col("candidate") == candidate)["horizon_minutes"]),
            {*HORIZONS, 0},
        )
    fitted = frame.filter(
        (pl.col("candidate") != "incumbent_predictions")
        & pl.col("horizon_minutes").is_in(HORIZONS)
    )
    _require_finite_columns(
        fitted, ("selected_penalty", "valid_count", "coefficient_norm"), table_path
    )
    _require_positive_integers(fitted, ("valid_count",), table_path)
    if (fitted["coefficient_norm"] < 0.0).any() or not set(
        fitted["selected_penalty"]
    ).issubset(set(RIDGE_PENALTIES)):
        raise ValueError("Frozen-probe fit metadata is invalid")
    references = frame.filter(
        (pl.col("candidate") == "incumbent_predictions")
        | (pl.col("horizon_minutes") == 0)
    )
    for column in ("selected_penalty", "valid_count", "coefficient_norm"):
        if references[column].null_count() != references.height:
            raise ValueError(f"Frozen-probe {column} has invalid null semantics")

    summary_path = output_dir / "frozen_block_probe_summary.json"
    summary = read_json_object(summary_path)
    _require_keys(
        summary,
        {
            "probe_calibration_start",
            "ridge_penalties",
            "receptive_field_minutes",
            "best_tap_by_horizon",
            "shallower_30_deeper_120",
            "concatenation_beats_every_individual_tap_all_horizons",
            "concatenated_beats_final_post_fusion_by_horizon",
            "earlier_tap_beats_final_post_fusion_by_horizon",
        },
        summary_path,
    )
    calibration_start = date.fromisoformat(str(summary["probe_calibration_start"]))
    if not TRAIN_START <= calibration_start <= TRAIN_END:
        raise ValueError("Frozen-probe calibration boundary is outside training")
    _require_equal(
        "frozen ridge penalties", summary["ridge_penalties"], list(RIDGE_PENALTIES)
    )
    _require_equal(
        "frozen receptive fields",
        summary["receptive_field_minutes"],
        [15, 35, 75, 155, 315, 635],
    )
    scores = {
        (str(candidate), int(horizon)): float(value)
        for candidate, horizon, value in frame.select(
            "candidate", "horizon_minutes", "validation_ic"
        ).iter_rows()
    }
    best_taps = {
        str(minutes): max(
            (f"block_{index}" for index in range(1, 7)),
            key=lambda candidate: scores[candidate, minutes],
        )
        for minutes in HORIZONS
    }
    _require_equal("frozen best taps", summary["best_tap_by_horizon"], best_taps)
    _require_equal(
        "frozen depth ordering",
        summary["shallower_30_deeper_120"],
        int(best_taps["30"].split("_")[1]) < int(best_taps["120"].split("_")[1]),
    )
    _require_equal(
        "frozen concatenation versus individual taps",
        summary["concatenation_beats_every_individual_tap_all_horizons"],
        all(
            scores["concatenated", minutes]
            > max(scores[f"block_{index}", minutes] for index in range(1, 7))
            for minutes in HORIZONS
        ),
    )
    expected_concatenated = {
        str(minutes): scores["concatenated", minutes]
        > scores["final_post_fusion", minutes]
        for minutes in HORIZONS
    }
    expected_earlier = {
        str(minutes): max(scores[f"block_{index}", minutes] for index in range(1, 6))
        > scores["final_post_fusion", minutes]
        for minutes in HORIZONS
    }
    _require_equal(
        "frozen concatenated evidence",
        summary["concatenated_beats_final_post_fusion_by_horizon"],
        expected_concatenated,
    )
    _require_equal(
        "frozen earlier-tap evidence",
        summary["earlier_tap_beats_final_post_fusion_by_horizon"],
        expected_earlier,
    )


def validate_context_inference(output_dir: Path) -> None:
    modes = {
        "baseline",
        *(
            f"{family}_{suffix}"
            for family in CONTEXT_FAMILIES
            for suffix in ("masked", "permuted")
        ),
    }
    table_path = output_dir / "context_inference_probes.csv"
    frame = _read_csv(table_path)
    _require_columns(
        frame,
        {
            "mode",
            "horizon_minutes",
            "baseline_ic",
            "candidate_ic",
            "delta_ic",
            "delta_lower_95",
            "delta_upper_95",
        },
        table_path,
    )
    _require_equal("context inference modes", set(frame["mode"]), modes)
    _require_equal("context inference row count", frame.height, len(modes) * 4)
    _require_unique_keys(frame, ("mode", "horizon_minutes"), table_path)
    for column in ("baseline_ic", "candidate_ic"):
        _require_bounds(frame, column, -1.0, 1.0, table_path)
    for column in ("delta_ic", "delta_lower_95", "delta_upper_95"):
        _require_bounds(frame, column, -2.0, 2.0, table_path)
    for mode in modes:
        selected = frame.filter(pl.col("mode") == mode)
        _require_equal(
            f"{mode} horizons", set(selected["horizon_minutes"]), {*HORIZONS, 0}
        )
        for row in selected.iter_rows(named=True):
            _require_close(
                f"{mode} delta {row['horizon_minutes']}",
                row["delta_ic"],
                float(row["candidate_ic"]) - float(row["baseline_ic"]),
            )
    baseline_rows = frame.filter(pl.col("mode") == "baseline")
    for row in baseline_rows.iter_rows(named=True):
        _require_close(
            "baseline context candidate", row["candidate_ic"], row["baseline_ic"]
        )
        for name in ("delta_ic", "delta_lower_95", "delta_upper_95"):
            _require_equal(f"baseline context {name}", row[name], 0.0)

    manifest_path = output_dir / "context_permutation_manifest.json"
    manifest = read_json_object(manifest_path)
    _require_keys(
        manifest,
        {"seed", "mapping_sha256", "group_sizes", "self_map_count"},
        manifest_path,
    )
    _require_equal("context permutation seed", manifest["seed"], PERMUTATION_SEED)
    digest = manifest["mapping_sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Context permutation digest is invalid")
    try:
        int(digest, 16)
    except ValueError as error:
        raise ValueError("Context permutation digest is invalid") from error
    _require_equal("context permutation self maps", manifest["self_map_count"], 0)
    group_sizes = manifest["group_sizes"]
    if not isinstance(group_sizes, dict):
        raise ValueError("Context permutation group sizes must be an object")
    _require_equal(
        "context permutation quarters",
        set(group_sizes),
        {"2024Q3", "2024Q4", "2025Q1", "2025Q2"},
    )
    if any(not isinstance(size, int) or size < 2 for size in group_sizes.values()):
        raise ValueError("Context permutation quarters must contain at least two dates")
    _require_equal(
        "context permutation date count",
        sum(group_sizes.values()),
        EXPECTED_SPLIT_DATE_COUNTS["validation"],
    )

    summary_path = output_dir / "context_family_summary.json"
    summary = read_json_object(summary_path)
    _require_keys(summary, {"interpretation", "inference"}, summary_path)
    if (
        not isinstance(summary["interpretation"], str)
        or not summary["interpretation"].strip()
    ):
        raise ValueError("Context interpretation is empty")
    inference = summary["inference"]
    _require_equal("context summary modes", set(inference), modes)
    baseline_summary = inference["baseline"]
    baseline_horizons = {
        int(row["horizon_minutes"]): row["mean_daily_spearman_ic"]
        for row in baseline_summary["horizons"]
    }
    _require_equal(
        "context baseline summary horizons", set(baseline_horizons), set(HORIZONS)
    )
    for mode in modes:
        metrics = inference[mode]
        _require_keys(metrics, {"primary_score", "horizons"}, summary_path)
        candidate_horizons = {
            int(row["horizon_minutes"]): row["mean_daily_spearman_ic"]
            for row in metrics["horizons"]
        }
        _require_equal(
            f"context {mode} summary horizons", set(candidate_horizons), set(HORIZONS)
        )
        selected = frame.filter(pl.col("mode") == mode)
        for row in selected.iter_rows(named=True):
            horizon = int(row["horizon_minutes"])
            expected_baseline = (
                baseline_summary["primary_score"]
                if horizon == 0
                else baseline_horizons[horizon]
            )
            expected_candidate = (
                metrics["primary_score"]
                if horizon == 0
                else candidate_horizons[horizon]
            )
            _require_close(
                f"context {mode} baseline {horizon}",
                row["baseline_ic"],
                expected_baseline,
            )
            _require_close(
                f"context {mode} candidate {horizon}",
                row["candidate_ic"],
                expected_candidate,
            )


def validate_gradient_audit(output_dir: Path) -> None:
    table_path = output_dir / "horizon_gradient_audit.parquet"
    frame = _read_parquet(table_path)
    key_columns = (
        "date_idx",
        "decision_idx",
        "group",
        "left_horizon",
        "right_horizon",
    )
    _require_columns(
        frame,
        {
            *key_columns,
            "cosine",
            "undefined_reason",
            "left_gradient_norm",
            "right_gradient_norm",
        },
        table_path,
    )
    samples = frame.select("date_idx", "decision_idx").unique()
    _require_equal("gradient sample count", samples.height, 20)
    _require_equal(
        "gradient row count",
        frame.height,
        20 * len(GRADIENT_GROUPS) * len(HORIZON_PAIRS),
    )
    _require_unique_keys(frame, key_columns, table_path)
    _require_equal("gradient groups", set(frame["group"]), set(GRADIENT_GROUPS))
    _require_equal(
        "gradient horizon pairs",
        set(frame.select("left_horizon", "right_horizon").iter_rows()),
        HORIZON_PAIRS,
    )
    expected_cells = {
        (group, left, right)
        for group in GRADIENT_GROUPS
        for left, right in HORIZON_PAIRS
    }
    for date_idx, decision_idx in samples.iter_rows():
        selected = frame.filter(
            (pl.col("date_idx") == date_idx) & (pl.col("decision_idx") == decision_idx)
        )
        _require_equal(
            f"gradient cells for {date_idx}/{decision_idx}",
            set(selected.select("group", "left_horizon", "right_horizon").iter_rows()),
            expected_cells,
        )
    _require_finite_columns(
        frame, ("left_gradient_norm", "right_gradient_norm"), table_path
    )
    if (frame["left_gradient_norm"] < 0.0).any() or (
        frame["right_gradient_norm"] < 0.0
    ).any():
        raise ValueError("Gradient norms must be nonnegative")
    for row in frame.select(
        "cosine",
        "undefined_reason",
        "left_gradient_norm",
        "right_gradient_norm",
    ).iter_rows(named=True):
        cosine = row["cosine"]
        reason = row["undefined_reason"]
        if cosine is None:
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("Null gradient cosine requires an undefined reason")
            if (
                reason == "zero_norm"
                and row["left_gradient_norm"] != 0.0
                and row["right_gradient_norm"] != 0.0
            ):
                raise ValueError("zero_norm cosine has no zero gradient norm")
        else:
            value = _finite(cosine, "gradient cosine")
            if not -1.0 <= value <= 1.0:
                raise ValueError("Gradient cosine is outside [-1, 1]")
            if reason is not None:
                raise ValueError("Defined gradient cosine carries an undefined reason")

    summary_path = output_dir / "horizon_gradient_summary.json"
    summary = read_json_object(summary_path)
    _require_keys(
        summary,
        {
            "train_only",
            "train_end",
            "sample_count",
            "by_group_and_horizon_pair",
            "single_horizon_controls",
        },
        summary_path,
    )
    _require_equal("gradient train-only flag", summary["train_only"], True)
    _require_equal("gradient train end", summary["train_end"], str(TRAIN_END))
    _require_equal(
        "gradient summary sample count", summary["sample_count"], samples.height
    )
    summary_rows = summary["by_group_and_horizon_pair"]
    _require_equal(
        "gradient summary row count",
        len(summary_rows),
        len(GRADIENT_GROUPS) * len(HORIZON_PAIRS),
    )
    summary_keys = {
        (row["group"], int(row["left_horizon"]), int(row["right_horizon"]))
        for row in summary_rows
    }
    _require_equal("gradient summary cells", summary_keys, expected_cells)
    if len(summary_keys) != len(summary_rows):
        raise ValueError("Gradient summary has duplicate cells")
    for row in summary_rows:
        _require_keys(
            row,
            {
                "group",
                "left_horizon",
                "right_horizon",
                "mean_cosine",
                "median_cosine",
                "fraction_negative",
                "mean_left_gradient_norm",
                "mean_right_gradient_norm",
                "valid_samples",
            },
            summary_path,
        )
        selected = frame.filter(
            (pl.col("group") == row["group"])
            & (pl.col("left_horizon") == row["left_horizon"])
            & (pl.col("right_horizon") == row["right_horizon"])
        )
        defined = np.asarray(
            [value for value in selected["cosine"].to_list() if value is not None],
            dtype=np.float64,
        )
        _require_equal(
            "gradient valid sample count", row["valid_samples"], defined.size
        )
        _require_close(
            "gradient mean left norm",
            row["mean_left_gradient_norm"],
            selected["left_gradient_norm"].mean(),
        )
        _require_close(
            "gradient mean right norm",
            row["mean_right_gradient_norm"],
            selected["right_gradient_norm"].mean(),
        )
        if defined.size:
            _require_close("gradient mean cosine", row["mean_cosine"], defined.mean())
            _require_close(
                "gradient median cosine", row["median_cosine"], np.median(defined)
            )
            _require_close(
                "gradient negative fraction",
                row["fraction_negative"],
                np.mean(defined < 0.0),
            )
        elif any(
            row[name] is not None
            for name in ("mean_cosine", "median_cosine", "fraction_negative")
        ):
            raise ValueError(
                "Undefined gradient summary cell must use null cosine statistics"
            )


def validate_oof_plan(path: Path, expected: dict[str, object]) -> None:
    plan = read_json_object(path)
    _require_equal("OOF plan", plan, expected)
    _require_equal("OOF block ordering", list(plan["blocks"]), ["B0", "B1", "B2", "B3"])
    previous_end = None
    for block in plan["blocks"].values():
        if block["end"] > str(TRAIN_END):
            raise ValueError("OOF plan exceeds the training period")
        if previous_end is not None and previous_end >= block["start"]:
            raise ValueError("OOF blocks are not strictly chronological")
        previous_end = block["end"]


def validate_oof_residual(output_dir: Path) -> None:
    table_path = output_dir / "oof_residual_probes.csv"
    frame = _read_csv(table_path)
    numeric = (
        "residual_prediction_ic",
        "base_ic",
        "base_plus_correction_ic",
        "delta_from_base",
        "selected_penalty",
        "valid_count",
        "feature_dim",
    )
    _require_columns(
        frame,
        {"probe", "horizon_minutes", *numeric},
        table_path,
    )
    _require_equal("OOF residual probes", set(frame["probe"]), set(RESIDUAL_PROBES))
    _require_equal(
        "OOF residual row count", frame.height, len(RESIDUAL_PROBES) * len(HORIZONS)
    )
    _require_unique_keys(frame, ("probe", "horizon_minutes"), table_path)
    for probe in RESIDUAL_PROBES:
        _require_equal(
            f"{probe} OOF horizons",
            set(frame.filter(pl.col("probe") == probe)["horizon_minutes"]),
            set(HORIZONS),
        )
    for column in (
        "residual_prediction_ic",
        "base_ic",
        "base_plus_correction_ic",
    ):
        _require_bounds(frame, column, -1.0, 1.0, table_path)
    _require_bounds(frame, "delta_from_base", -2.0, 2.0, table_path)
    _require_finite_columns(frame, ("selected_penalty",), table_path)
    _require_positive_integers(frame, ("valid_count", "feature_dim"), table_path)
    if not set(frame["selected_penalty"]).issubset(set(RIDGE_PENALTIES)):
        raise ValueError("OOF residual probe selected an unknown ridge penalty")
    for row in frame.iter_rows(named=True):
        _require_close(
            f"OOF delta {row['probe']}/{row['horizon_minutes']}",
            row["delta_from_base"],
            float(row["base_plus_correction_ic"]) - float(row["base_ic"]),
        )
    for horizon in HORIZONS:
        base_values = frame.filter(pl.col("horizon_minutes") == horizon)["base_ic"]
        reference = base_values[0]
        for value in base_values:
            _require_close(f"OOF base IC {horizon}", value, reference)

    summary_path = output_dir / "oof_residual_probe_summary.json"
    summary = read_json_object(summary_path)
    _require_keys(
        summary,
        {
            "fit_block",
            "evaluation_block",
            "calibration_start",
            "sector_subsector_note",
            "results",
        },
        summary_path,
    )
    _require_equal("OOF residual fit block", summary["fit_block"], "B2")
    _require_equal("OOF residual evaluation block", summary["evaluation_block"], "B3")
    calibration_start = date.fromisoformat(str(summary["calibration_start"]))
    if not TRAIN_START <= calibration_start <= TRAIN_END:
        raise ValueError("OOF residual calibration boundary is outside training")
    if (
        not isinstance(summary["sector_subsector_note"], str)
        or not summary["sector_subsector_note"].strip()
    ):
        raise ValueError("OOF residual sector/subsector note is empty")
    result_rows = summary["results"]
    _require_equal("OOF residual summary rows", len(result_rows), frame.height)
    result_keys = {(row["probe"], int(row["horizon_minutes"])) for row in result_rows}
    _require_equal(
        "OOF residual summary keys",
        result_keys,
        set(frame.select("probe", "horizon_minutes").iter_rows()),
    )
    if len(result_keys) != len(result_rows):
        raise ValueError("OOF residual summary has duplicate results")
    table_rows = {
        (row["probe"], int(row["horizon_minutes"])): row
        for row in frame.iter_rows(named=True)
    }
    for result in result_rows:
        key = (result["probe"], int(result["horizon_minutes"]))
        table_row = table_rows[key]
        for name in numeric:
            _require_close(f"OOF summary {key} {name}", result[name], table_row[name])


def validate_completed_run(
    run_dir: Path,
    store: Path,
    expected: dict[str, Any],
) -> None:
    store_identity = feature_store_identity(store)
    _require_equal(
        "expected feature-store identity",
        expected["feature_store_identity"],
        store_identity,
    )
    _require_equal(
        "expected feature-store path",
        str(Path(expected["feature_store"]).resolve()),
        str(store.resolve()),
    )

    manifest_path = run_dir / "run_manifest.json"
    manifest = read_json_object(manifest_path)
    _require_keys(
        manifest,
        {
            "status",
            "repository_commit",
            "run_provenance",
            "feature_store",
            "feature_store_identity",
            "split",
            "seed",
            "global_context",
            "training_horizon",
            "selection_horizon",
            "context_family_ablation",
            "training",
            "model",
            "parameter_count",
            "objective",
            "optimizer",
            "sam",
            "best_epoch",
            "best_validation_score",
            "epochs_completed",
            "total_run_seconds",
        },
        manifest_path,
    )
    _require_equal("run status", manifest["status"], "completed")
    _require_equal("run provenance", manifest["run_provenance"], expected)
    _require_equal(
        "run repository commit",
        manifest["repository_commit"],
        expected["repository_commit"],
    )
    _require_equal(
        "run feature-store identity", manifest["feature_store_identity"], store_identity
    )
    _require_equal(
        "run feature-store path",
        str(Path(manifest["feature_store"]).resolve()),
        str(store.resolve()),
    )
    legacy_manifest = {
        "seed": expected["seed"],
        "global_context": expected["global_context"],
        "training_horizon": expected["training_horizon"],
        "selection_horizon": expected["selection_horizon"],
        "context_family_ablation": expected["context_family_ablation"],
        "parameter_count": expected["parameter_count"],
        "objective": expected["objective"],
        "optimizer": expected["optimizer"],
        "sam": expected["sam"],
        "model": expected["model"],
        "training": expected["training"],
    }
    for name, value in legacy_manifest.items():
        _require_equal(f"run {name}", manifest[name], value)
    split = manifest["split"]
    _require_equal(
        "run fit name", split.get("training"), expected["fit_window"]["name"]
    )
    _require_equal(
        "run selection name",
        split.get("selection"),
        expected["selection_window"]["name"],
    )
    _require_equal("run fit window", split.get("fit_window"), expected["fit_window"])
    _require_equal(
        "run selection window",
        split.get("selection_window"),
        expected["selection_window"],
    )
    _require_equal("run test access", split.get("test_accessed"), False)

    checkpoint_path = run_dir / "best_checkpoint.pt"
    try:
        raw_checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except (OSError, RuntimeError, EOFError) as error:
        raise ValueError(
            f"Invalid checkpoint artifact {checkpoint_path}: {error}"
        ) from error
    required_checkpoint = {
        "repository_commit",
        "run_provenance",
        "model_name",
        "architecture",
        "tcn_settings",
        "peer_features",
        "optimizer_variant",
        "objective",
        "sam",
        "seed",
        "epoch",
        "validation_score",
        "feature_store",
        "feature_store_identity",
        "global_context",
        "training_horizon",
        "selection_horizon",
        "context_family_ablation",
        "fit_window",
        "selection_window",
        "parameter_count",
        "model_state_dict",
    }
    missing = required_checkpoint - set(raw_checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing stage provenance: {sorted(missing)}")
    _require_equal(
        "checkpoint run provenance", raw_checkpoint["run_provenance"], expected
    )
    model_provenance = expected["model"]
    checkpoint_expectations = {
        "repository_commit": expected["repository_commit"],
        "model_name": model_provenance["model_name"],
        "architecture": model_provenance["architecture"],
        "tcn_settings": model_provenance["tcn_settings"],
        "peer_features": model_provenance["peer_features"],
        "optimizer_variant": expected["optimizer"],
        "objective": expected["objective"],
        "sam": expected["sam"],
        "seed": expected["seed"],
        "feature_store_identity": store_identity,
        "global_context": expected["global_context"],
        "training_horizon": expected["training_horizon"],
        "selection_horizon": expected["selection_horizon"],
        "context_family_ablation": expected["context_family_ablation"],
        "fit_window": expected["fit_window"],
        "selection_window": expected["selection_window"],
        "parameter_count": expected["parameter_count"],
    }
    for name, value in checkpoint_expectations.items():
        _require_equal(f"checkpoint {name}", raw_checkpoint[name], value)
    _require_equal(
        "checkpoint feature-store path",
        str(Path(raw_checkpoint["feature_store"]).resolve()),
        str(store.resolve()),
    )
    _require_equal(
        "checkpoint best epoch", raw_checkpoint["epoch"], manifest["best_epoch"]
    )
    _require_equal(
        "checkpoint validation score",
        raw_checkpoint["validation_score"],
        manifest["best_validation_score"],
    )
    try:
        model, restored, restored_store = load_current_neural_run(run_dir)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"Checkpoint strict reconstruction failed: {error}") from error
    _require_equal("restored feature store", restored_store.resolve(), store.resolve())
    _require_equal(
        "restored parameter count",
        count_trainable_parameters(model),
        expected["parameter_count"],
    )
    _require_equal(
        "restored training horizon",
        restored["training_horizon"],
        expected["training_horizon"],
    )

    metrics_path = run_dir / "validation_metrics.json"
    metrics = read_json_object(metrics_path)
    _require_keys(metrics, {"primary_score", "horizons"}, metrics_path)
    _finite(metrics["primary_score"], "validation primary score")
    metric_rows = metrics["horizons"]
    _require_equal("validation metric row count", len(metric_rows), len(HORIZONS))
    metric_horizons = [int(row["horizon_minutes"]) for row in metric_rows]
    _require_equal("validation metric horizons", set(metric_horizons), set(HORIZONS))
    if len(set(metric_horizons)) != len(metric_horizons):
        raise ValueError("Validation metrics contain duplicate horizons")
    metric_by_horizon = {
        int(row["horizon_minutes"]): _finite(
            row["mean_daily_spearman_ic"], "validation horizon IC"
        )
        for row in metric_rows
    }

    daily_path = run_dir / "validation_daily_metrics.parquet"
    daily = _read_parquet(daily_path)
    _require_columns(daily, {"date_idx", "horizon_minutes", "spearman_ic"}, daily_path)
    _require_equal(
        "validation daily horizons", set(daily["horizon_minutes"]), set(HORIZONS)
    )
    expected_date_count = int(expected["selection_window"]["date_count"])
    _require_equal(
        "validation daily rows", daily.height, expected_date_count * len(HORIZONS)
    )
    _require_unique_keys(daily, ("date_idx", "horizon_minutes"), daily_path)
    _require_bounds(daily, "spearman_ic", -1.0, 1.0, daily_path)
    recomputed: dict[int, float] = {}
    for horizon in HORIZONS:
        selected = daily.filter(pl.col("horizon_minutes") == horizon).sort("date_idx")
        _require_equal(
            f"validation daily date count {horizon}",
            selected.height,
            expected_date_count,
        )
        date_indices = selected["date_idx"].cast(pl.Int64).to_numpy()
        _require_equal(
            f"validation date identity {horizon}",
            int64_identity_sha256(date_indices),
            expected["selection_window"]["date_identity_sha256"],
        )
        recomputed[horizon] = float(selected["spearman_ic"].mean())
        _require_close(
            f"validation horizon IC {horizon}",
            metric_by_horizon[horizon],
            recomputed[horizon],
        )
    recomputed_primary = float(np.mean([recomputed[horizon] for horizon in HORIZONS]))
    _require_close(
        "validation aggregate primary score",
        metrics["primary_score"],
        recomputed_primary,
    )
    selected_score = (
        float(metrics["primary_score"])
        if expected["training_horizon"] == "all"
        else metric_by_horizon[int(expected["training_horizon"])]
    )
    _require_close(
        "selected validation score", manifest["best_validation_score"], selected_score
    )

    history_path = run_dir / "history.csv"
    history = _read_csv(history_path)
    _require_columns(
        history,
        {
            "epoch",
            "train_objective_loss",
            "validation_objective_loss",
            "validation_primary_ic",
            "optimizer_steps",
        },
        history_path,
    )
    _require_equal("history epoch count", history.height, manifest["epochs_completed"])
    _require_unique_keys(history, ("epoch",), history_path)
    _require_equal(
        "history epochs", set(history["epoch"]), set(range(1, history.height + 1))
    )
    if history.height > int(expected["training"]["maximum_epochs"]):
        raise ValueError("Training history exceeds maximum epochs")
    _require_finite_columns(
        history,
        (
            "train_objective_loss",
            "validation_objective_loss",
            "validation_primary_ic",
            "optimizer_steps",
        ),
        history_path,
    )
    _require_positive_integers(history, ("epoch", "optimizer_steps"), history_path)
    best = history.filter(pl.col("epoch") == manifest["best_epoch"])
    _require_equal("history best epoch occurrence", best.height, 1)
    _require_equal(
        "history best validation score",
        best["validation_primary_ic"][0],
        manifest["best_validation_score"],
    )
    duration = _finite(manifest["total_run_seconds"], "training duration")
    if duration < 0.0:
        raise ValueError("Training duration must be nonnegative")


def validate_consolidated(output_dir: Path, arm_names: tuple[str, ...]) -> None:
    run_matrix_path = output_dir / "run_matrix.csv"
    run_matrix = _read_csv(run_matrix_path)
    _require_columns(
        run_matrix,
        {
            "arm",
            "seed",
            "readout",
            "training_horizon",
            "context_family_ablation",
            "validation_ic",
            "ic_30",
            "ic_60",
            "ic_120",
            "best_epoch",
            "epochs_completed",
            "parameter_count",
            "training_duration_seconds",
        },
        run_matrix_path,
    )
    _require_equal("run-matrix arms", set(run_matrix["arm"]), set(arm_names))
    _require_equal("run-matrix row count", run_matrix.height, len(arm_names))
    _require_unique_keys(run_matrix, ("arm",), run_matrix_path)
    _require_finite_columns(
        run_matrix,
        (
            "validation_ic",
            "ic_30",
            "ic_60",
            "ic_120",
            "best_epoch",
            "epochs_completed",
            "parameter_count",
            "training_duration_seconds",
        ),
        run_matrix_path,
    )
    for column in ("validation_ic", "ic_30", "ic_60", "ic_120"):
        _require_bounds(run_matrix, column, -1.0, 1.0, run_matrix_path)
    _require_positive_integers(
        run_matrix,
        ("best_epoch", "epochs_completed", "parameter_count"),
        run_matrix_path,
    )
    if (run_matrix["training_duration_seconds"] < 0.0).any():
        raise ValueError("Run-matrix training durations must be nonnegative")

    comparison_path = output_dir / "multiscale_comparison.csv"
    comparisons = _read_csv(comparison_path)
    _require_columns(
        comparisons,
        {
            "comparison",
            "seed",
            "horizon_minutes",
            "control_ic",
            "candidate_ic",
            "seed_mean_delta",
            "seed_std_delta",
            "delta_ic",
            "delta_lower_95",
            "delta_upper_95",
        },
        comparison_path,
    )
    _require_equal(
        "multiscale comparisons", set(comparisons["comparison"]), set(COMPARISON_SEEDS)
    )
    _require_equal(
        "multiscale comparison row count",
        comparisons.height,
        len(COMPARISON_SEEDS) * (len(HORIZONS) + 1),
    )
    for name, seed in COMPARISON_SEEDS.items():
        selected = comparisons.filter(pl.col("comparison") == name)
        _require_equal(
            f"{name} horizons",
            set(selected["horizon_minutes"]),
            {*HORIZONS, 0},
        )
        _require_equal(f"{name} seed", set(selected["seed"]), {seed})
    _require_equal(
        "multiscale comparison keys",
        comparisons.select("comparison", "seed", "horizon_minutes").n_unique(),
        comparisons.height,
    )
    _require_finite_columns(
        comparisons,
        ("delta_ic", "delta_lower_95", "delta_upper_95"),
        comparison_path,
    )
    for column in ("delta_ic", "delta_lower_95", "delta_upper_95"):
        _require_bounds(comparisons, column, -2.0, 2.0, comparison_path)
    for row in comparisons.iter_rows(named=True):
        name = str(row["comparison"])
        if float(row["delta_lower_95"]) > float(row["delta_upper_95"]):
            raise ValueError(f"Comparison interval is reversed for {name}")
        if name == "horizon_multiscale_vs_final_three_seed":
            if row["control_ic"] is not None or row["candidate_ic"] is not None:
                raise ValueError("Three-seed comparison must not invent aggregate ICs")
            _finite(row["seed_mean_delta"], "three-seed mean delta")
            seed_std = _finite(row["seed_std_delta"], "three-seed delta std")
            if seed_std < 0.0:
                raise ValueError("Three-seed delta standard deviation is negative")
        else:
            control = _finite(row["control_ic"], "comparison control IC")
            candidate = _finite(row["candidate_ic"], "comparison candidate IC")
            if not -1.0 <= control <= 1.0 or not -1.0 <= candidate <= 1.0:
                raise ValueError(f"Comparison IC is outside [-1, 1] for {name}")
            _require_close(
                f"comparison delta identity {name}/{row['horizon_minutes']}",
                row["delta_ic"],
                candidate - control,
            )
            if row["seed_mean_delta"] is not None or row["seed_std_delta"] is not None:
                raise ValueError("Single-seed comparison has three-seed statistics")

    paired_path = output_dir / "multiscale_paired_daily.parquet"
    paired = _read_parquet(paired_path)
    _require_columns(
        paired,
        {"comparison", "seed", "date_idx", "horizon_minutes", "delta_ic"},
        paired_path,
    )
    _require_equal("paired horizons", set(paired["horizon_minutes"]), set(HORIZONS))
    _require_equal(
        "paired comparisons", set(paired["comparison"]), set(COMPARISON_SEEDS)
    )
    _require_equal(
        "paired keys",
        paired.select("comparison", "seed", "date_idx", "horizon_minutes").n_unique(),
        paired.height,
    )
    paired_group_sizes = []
    for name, seed in COMPARISON_SEEDS.items():
        selected = paired.filter(pl.col("comparison") == name)
        _require_equal(f"paired {name} seed", set(selected["seed"]), {seed})
        _require_equal(
            f"paired {name} horizons",
            set(selected["horizon_minutes"]),
            set(HORIZONS),
        )
        paired_group_sizes.extend(
            selected.group_by("horizon_minutes").len().get_column("len").to_list()
        )
    if not paired_group_sizes or len(set(paired_group_sizes)) != 1:
        raise ValueError("Paired comparison groups have inconsistent date counts")
    _require_finite_columns(paired, ("delta_ic",), paired_path)
    _require_bounds(paired, "delta_ic", -2.0, 2.0, paired_path)
    comparison_lookup = {
        (str(row["comparison"]), int(row["horizon_minutes"])): row
        for row in comparisons.iter_rows(named=True)
    }
    for name in COMPARISON_SEEDS:
        selected = paired.filter(pl.col("comparison") == name)
        for horizon in (*HORIZONS, 0):
            values = (
                selected.group_by("date_idx")
                .agg(pl.col("delta_ic").mean())
                .sort("date_idx")["delta_ic"]
                .to_numpy()
                if horizon == 0
                else selected.filter(pl.col("horizon_minutes") == horizon)
                .sort("date_idx")["delta_ic"]
                .to_numpy()
            )
            interval = moving_block_bootstrap(values, seed=BOOTSTRAP_SEED)
            row = comparison_lookup[(name, horizon)]
            for field, expected_value in (
                ("delta_ic", interval["estimate"][0]),
                ("delta_lower_95", interval["lower_95"][0]),
                ("delta_upper_95", interval["upper_95"][0]),
            ):
                _require_close(
                    f"paired comparison {name}/{horizon} {field}",
                    row[field],
                    expected_value,
                )

    three_seed_name = "horizon_multiscale_vs_final_three_seed"
    constituents = paired.filter(
        pl.col("comparison").is_in(
            [f"horizon_multiscale_vs_final_seed{seed}" for seed in (11, 29, 47)]
        )
    )
    expected_three_seed = {
        (int(date_idx), int(horizon)): float(delta)
        for date_idx, horizon, delta in constituents.group_by(
            "date_idx", "horizon_minutes"
        )
        .agg(pl.col("delta_ic").mean())
        .iter_rows()
    }
    actual_three_seed = paired.filter(pl.col("comparison") == three_seed_name)
    for date_idx, horizon, delta in actual_three_seed.select(
        "date_idx", "horizon_minutes", "delta_ic"
    ).iter_rows():
        _require_close(
            f"three-seed paired daily {date_idx}/{horizon}",
            delta,
            expected_three_seed[(int(date_idx), int(horizon))],
        )
    for horizon in (*HORIZONS, 0):
        seed_deltas = np.asarray(
            [
                comparison_lookup[(f"horizon_multiscale_vs_final_seed{seed}", horizon)][
                    "delta_ic"
                ]
                for seed in (11, 29, 47)
            ],
            dtype=np.float64,
        )
        row = comparison_lookup[(three_seed_name, horizon)]
        _require_close(
            f"three-seed mean delta {horizon}",
            row["seed_mean_delta"],
            seed_deltas.mean(),
        )
        _require_close(
            f"three-seed delta std {horizon}",
            row["seed_std_delta"],
            seed_deltas.std(ddof=1),
        )

    gates_path = output_dir / "multiscale_gate_weights.csv"
    gates = _read_csv(gates_path)
    _require_columns(
        gates,
        {
            "arm",
            "seed",
            "horizon_minutes",
            "block",
            "receptive_field_minutes",
            "weight",
            "weight_std",
            "entropy",
            "entropy_std",
        },
        gates_path,
    )
    _require_equal("gate blocks", set(gates["block"]), set(range(1, 7)))
    _require_equal("gate horizons", set(gates["horizon_minutes"]), set(HORIZONS))
    _require_equal(
        "gate runs",
        set(gates.select("arm", "seed").iter_rows()),
        GATE_RUNS,
    )
    _require_equal(
        "gate row count",
        gates.height,
        len(GATE_RUNS) * len(HORIZONS) * 6,
    )
    _require_unique_keys(gates, ("arm", "seed", "horizon_minutes", "block"), gates_path)
    _require_finite_columns(gates, ("weight", "entropy"), gates_path)
    _require_bounds(gates, "weight", 0.0, 1.0, gates_path)
    _require_bounds(
        gates,
        "entropy",
        0.0,
        math.log(6.0) + FLOAT32_DIAGNOSTIC_TOLERANCE,
        gates_path,
    )
    receptive_fields = {1: 15, 2: 35, 3: 75, 4: 155, 5: 315, 6: 635}
    for block, receptive_field in (
        gates.select("block", "receptive_field_minutes").unique().iter_rows()
    ):
        _require_equal(
            f"gate receptive field block {block}",
            receptive_field,
            receptive_fields[int(block)],
        )
    for key, group in gates.group_by("arm", "seed", "horizon_minutes"):
        if group.height != 6 or not np.isclose(group["weight"].sum(), 1.0, atol=1e-6):
            raise ValueError(f"Scale weights are invalid for {key}")
        entropy_values = group["entropy"].unique()
        if entropy_values.len() != 1:
            raise ValueError(f"Gate entropy is inconsistent for {key}")
        weights = group.sort("block")["weight"].to_numpy()
        positive_weights = weights[weights > 0.0]
        _require_close(
            f"gate entropy {key}",
            entropy_values[0],
            -(positive_weights * np.log(positive_weights)).sum(),
            tolerance=FLOAT32_DIAGNOSTIC_TOLERANCE,
        )
    summary_arm = "horizon_multiscale_three_seed"
    individual_gates = gates.filter(pl.col("arm") != summary_arm)
    summary_gates = gates.filter(pl.col("arm") == summary_arm)
    for column in ("weight_std", "entropy_std"):
        if individual_gates[column].is_not_null().any():
            raise ValueError(f"Individual gate rows have non-null {column}")
        if summary_gates[column].null_count():
            raise ValueError(f"Three-seed gate summary has null {column}")
        if (summary_gates[column] < 0.0).any():
            raise ValueError(f"Three-seed gate summary has negative {column}")
    constituent_gates = individual_gates.filter(
        pl.col("arm").str.starts_with("horizon_multiscale_seed")
    )
    _require_equal(
        "three-seed gate constituent runs",
        set(constituent_gates.select("arm", "seed").unique().iter_rows()),
        {(f"horizon_multiscale_seed{seed}", seed) for seed in (11, 29, 47)},
    )
    for row in summary_gates.iter_rows(named=True):
        selected = constituent_gates.filter(
            (pl.col("horizon_minutes") == row["horizon_minutes"])
            & (pl.col("block") == row["block"])
        )
        for field, statistic in (
            ("weight", selected["weight"].mean()),
            ("weight_std", selected["weight"].std()),
            ("entropy", selected["entropy"].mean()),
            ("entropy_std", selected["entropy"].std()),
        ):
            _require_close(
                f"three-seed gate {row['horizon_minutes']}/{row['block']} {field}",
                row[field],
                statistic,
            )

    context_path = output_dir / "audits" / "context" / "context_training_ablations.csv"
    context = _read_csv(context_path)
    _require_columns(
        context,
        {
            "context_family",
            "horizon_minutes",
            "control_ic",
            "candidate_ic",
            "delta_ic",
            "delta_lower_95",
            "delta_upper_95",
            "worst_horizon_delta",
            "best_epoch",
            "epochs_completed",
            "parameter_count",
            "training_duration_seconds",
        },
        context_path,
    )
    _require_equal(
        "context retraining families",
        set(context["context_family"]),
        set(CONTEXT_FAMILIES),
    )
    _require_equal(
        "context retraining row count", context.height, len(CONTEXT_FAMILIES) * 4
    )
    _require_unique_keys(context, ("context_family", "horizon_minutes"), context_path)
    for family in CONTEXT_FAMILIES:
        selected = context.filter(pl.col("context_family") == family)
        _require_equal(
            f"{family} retraining horizons",
            set(selected["horizon_minutes"]),
            {*HORIZONS, 0},
        )
    _require_finite_columns(
        context,
        (
            "control_ic",
            "candidate_ic",
            "delta_ic",
            "delta_lower_95",
            "delta_upper_95",
            "worst_horizon_delta",
            "best_epoch",
            "epochs_completed",
            "parameter_count",
            "training_duration_seconds",
        ),
        context_path,
    )
    for column in ("control_ic", "candidate_ic"):
        _require_bounds(context, column, -1.0, 1.0, context_path)
    for column in ("delta_ic", "delta_lower_95", "delta_upper_95"):
        _require_bounds(context, column, -2.0, 2.0, context_path)
    _require_positive_integers(
        context,
        ("best_epoch", "epochs_completed", "parameter_count"),
        context_path,
    )
    if (context["training_duration_seconds"] < 0.0).any():
        raise ValueError("Context retraining durations must be nonnegative")
    for family in CONTEXT_FAMILIES:
        selected = context.filter(pl.col("context_family") == family)
        worst = selected.filter(pl.col("horizon_minutes").is_in(HORIZONS))[
            "delta_ic"
        ].min()
        for row in selected.iter_rows(named=True):
            _require_close(
                f"context delta identity {family}/{row['horizon_minutes']}",
                row["delta_ic"],
                float(row["candidate_ic"]) - float(row["control_ic"]),
            )
            _require_close(
                f"context worst horizon {family}", row["worst_horizon_delta"], worst
            )
            if float(row["delta_lower_95"]) > float(row["delta_upper_95"]):
                raise ValueError(f"Context interval is reversed for {family}")

    controls_path = output_dir / "audits" / "gradient" / "single_horizon_controls.csv"
    controls = _read_csv(controls_path)
    _require_columns(
        controls,
        {
            "training_horizon",
            "paired_control_ic",
            "selected_horizon_ic",
            "delta_from_control",
        },
        controls_path,
    )
    _require_equal(
        "single-horizon controls",
        set(controls["training_horizon"].cast(pl.Utf8)),
        {str(value) for value in HORIZONS},
    )
    _require_equal("single-horizon control row count", controls.height, len(HORIZONS))
    _require_unique_keys(controls, ("training_horizon",), controls_path)
    _require_finite_columns(
        controls,
        (
            "paired_control_ic",
            "selected_horizon_ic",
            "delta_from_control",
        ),
        controls_path,
    )
    for column in ("paired_control_ic", "selected_horizon_ic"):
        _require_bounds(controls, column, -1.0, 1.0, controls_path)
    _require_bounds(controls, "delta_from_control", -2.0, 2.0, controls_path)
    for row in controls.iter_rows(named=True):
        _require_close(
            f"single-horizon delta {row['training_horizon']}",
            row["delta_from_control"],
            float(row["selected_horizon_ic"]) - float(row["paired_control_ic"]),
        )

    summary_path = output_dir / "stage_summary.json"
    summary = read_json_object(summary_path)
    _require_keys(
        summary,
        {
            "training_run_count",
            "test_accessed",
            "hypotheses",
            "bottleneck_interpretation",
            "promotion",
            "artifacts",
        },
        summary_path,
    )
    _require_equal(
        "stage training-run count", summary["training_run_count"], len(arm_names)
    )
    _require_equal("stage test access", summary["test_accessed"], False)
    frozen_summary = read_json_object(
        output_dir / "audits" / "frozen_block" / "frozen_block_probe_summary.json"
    )
    gradient_summary = read_json_object(
        output_dir / "audits" / "gradient" / "horizon_gradient_summary.json"
    )
    context_summary = read_json_object(
        output_dir / "audits" / "context" / "context_family_summary.json"
    )
    oof_summary = read_json_object(
        output_dir / "audits" / "oof" / "oof_residual_probe_summary.json"
    )
    target_summary = read_json_object(
        output_dir / "audits" / "target_basis" / "target_basis_summary.json"
    )
    comparison_rows = comparisons.to_dicts()
    context_rows = context.to_dicts()
    control_rows = controls.to_dicts()
    context_training = build_context_training_summary(context_rows)
    _require_equal(
        "context retraining summary",
        context_summary.get("retraining_ablation"),
        {
            "interpretation": "Retraining ablation measures replaceable usefulness.",
            "families": context_training,
        },
    )
    evidence = build_hypothesis_summary(
        comparison_rows,
        frozen_summary,
        gradient_summary,
        control_rows,
        context_summary,
        context_rows,
        oof_summary,
        target_summary,
    )
    expected_summary = {
        "training_run_count": len(arm_names),
        "test_accessed": False,
        "bootstrap": {"block_days": 5, "seed": BOOTSTRAP_SEED},
        "three_seed_horizon_multiscale": [
            row
            for row in comparison_rows
            if row["comparison"] == "horizon_multiscale_vs_final_three_seed"
        ],
        "context_training_ablations": context_training,
        **evidence,
        "promotion": "none",
    }
    _require_equal("stage summary reproducibility", summary, expected_summary)
    markdown = (output_dir / "stage_summary.md").read_text(encoding="utf-8")
    _require_equal(
        "stage Markdown reproducibility", markdown, stage_summary_markdown(summary)
    )
