from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from brazil_rv.modeling.contract import HORIZONS, TRAIN_END, TRAIN_START
from brazil_rv.modeling.horizon_diagnostics import FIXED_TARGET_BASIS
from brazil_rv.modeling.stage_conclusions import build_hypothesis_summary
from brazil_rv.modeling.stage_validation import (
    GRADIENT_GROUPS,
    HORIZON_PAIRS,
    _require_correlation_matrix,
    TARGET_HORIZON_PAIRS,
    read_json_object,
    validate_gradient_audit,
    validate_target_basis,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _gradient_artifacts(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(exist_ok=True)
    undefined_cell = (GRADIENT_GROUPS[0], *sorted(HORIZON_PAIRS)[0])
    rows: list[dict[str, object]] = []
    for sample in range(20):
        for group in GRADIENT_GROUPS:
            for left, right in sorted(HORIZON_PAIRS):
                undefined = (group, left, right) == undefined_cell
                rows.append(
                    {
                        "date_idx": sample,
                        "decision_idx": sample,
                        "group": group,
                        "left_horizon": left,
                        "right_horizon": right,
                        "cosine": None if undefined else 0.25,
                        "undefined_reason": "zero_norm" if undefined else None,
                        "left_gradient_norm": 0.0 if undefined else 1.0,
                        "right_gradient_norm": 0.0 if undefined else 2.0,
                    }
                )
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(
        output_dir / "horizon_gradient_audit.parquet"
    )
    summary_rows = []
    for group in GRADIENT_GROUPS:
        for left, right in sorted(HORIZON_PAIRS):
            undefined = (group, left, right) == undefined_cell
            summary_rows.append(
                {
                    "group": group,
                    "left_horizon": left,
                    "right_horizon": right,
                    "mean_cosine": None if undefined else 0.25,
                    "median_cosine": None if undefined else 0.25,
                    "fraction_negative": None if undefined else 0.0,
                    "mean_left_gradient_norm": 0.0 if undefined else 1.0,
                    "mean_right_gradient_norm": 0.0 if undefined else 2.0,
                    "valid_samples": 0 if undefined else 20,
                }
            )
    summary = {
        "train_only": True,
        "train_end": str(TRAIN_END),
        "sample_count": 20,
        "by_group_and_horizon_pair": summary_rows,
        "single_horizon_controls": [],
    }
    _write_json(output_dir / "horizon_gradient_summary.json", summary)
    return summary


def _conclusion_inputs(gradient_summary: dict[str, object]) -> dict[str, object]:
    def comparison(name: str, horizon: int, delta: float) -> dict[str, object]:
        return {
            "comparison": name,
            "horizon_minutes": horizon,
            "delta_ic": delta,
            "delta_lower_95": delta - 0.001,
            "delta_upper_95": delta + 0.001,
        }

    comparisons = [
        comparison("shared_multiscale_vs_final_seed29", 0, 0.001),
        comparison("horizon_multiscale_vs_shared_seed29", 0, 0.001),
        comparison("final_score_mlp_vs_final_seed29", 0, 0.001),
        comparison("horizon_multiscale_vs_final_seed29", 0, 0.001),
        *[
            comparison("horizon_multiscale_vs_final_three_seed", horizon, 0.001)
            for horizon in (*HORIZONS, 0)
        ],
    ]
    false_by_horizon = {str(horizon): False for horizon in HORIZONS}
    return {
        "comparisons": comparisons,
        "frozen_summary": {
            "best_tap_by_horizon": {str(horizon): "block_1" for horizon in HORIZONS},
            "earlier_tap_beats_final_post_fusion_by_horizon": false_by_horizon,
            "concatenated_beats_final_post_fusion_by_horizon": false_by_horizon,
        },
        "gradient_summary": gradient_summary,
        "single_horizon_rows": [
            {"training_horizon": str(horizon), "delta_from_control": 0.0}
            for horizon in HORIZONS
        ],
        "context_summary": {"inference": {"baseline": {}}},
        "context_rows": [
            {
                "context_family": family,
                "horizon_minutes": 0,
                "delta_ic": 0.0,
                "delta_lower_95": -0.001,
                "delta_upper_95": 0.001,
            }
            for family in ("wdo", "br_rates", "us_rates")
        ],
        "oof_summary": {"results": []},
        "target_summary": {
            "pooled_target_correlation": np.eye(3).tolist(),
            "eigenvalues": [1.0, 1.0, 1.0],
            "variance_shares": [1 / 3, 1 / 3, 1 / 3],
            "fixed_basis_variance": [1.0, 1.0, 1.0],
        },
    }


def test_valid_gradient_grid_with_zero_norm_cells_validates_and_summarizes(
    tmp_path: Path,
) -> None:
    summary = _gradient_artifacts(tmp_path)
    validate_gradient_audit(tmp_path)
    result = build_hypothesis_summary(**_conclusion_inputs(summary))
    assert (
        result["hypotheses"]["horizon_conflict"][
            "gradient_cells_without_defined_cosines"
        ]
        == 1
    )


@pytest.mark.parametrize("mutation", ("truncated", "duplicated"))
def test_gradient_grid_rejects_truncation_or_duplication(
    tmp_path: Path, mutation: str
) -> None:
    _gradient_artifacts(tmp_path)
    path = tmp_path / "horizon_gradient_audit.parquet"
    rows = pl.read_parquet(path).to_dicts()
    if mutation == "truncated":
        rows.pop()
    else:
        rows[-1] = rows[0].copy()
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(path)
    with pytest.raises(ValueError):
        validate_gradient_audit(tmp_path)


@pytest.mark.parametrize(
    ("cosine", "reason"),
    ((1.1, None), (0.25, "zero_norm"), (None, None)),
)
def test_gradient_grid_rejects_invalid_cosine_null_semantics(
    tmp_path: Path, cosine: float | None, reason: str | None
) -> None:
    _gradient_artifacts(tmp_path)
    path = tmp_path / "horizon_gradient_audit.parquet"
    rows = pl.read_parquet(path).to_dicts()
    rows[1]["cosine"] = cosine
    rows[1]["undefined_reason"] = reason
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(path)
    with pytest.raises(ValueError):
        validate_gradient_audit(tmp_path)


def test_gradient_summary_disagreement_is_rejected(tmp_path: Path) -> None:
    _gradient_artifacts(tmp_path)
    path = tmp_path / "horizon_gradient_summary.json"
    summary = read_json_object(path)
    summary["by_group_and_horizon_pair"][1]["mean_cosine"] = 0.5
    _write_json(path, summary)
    with pytest.raises(ValueError, match="gradient mean cosine differs"):
        validate_gradient_audit(tmp_path)


def _target_artifacts(output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True)
    pairs = sorted(TARGET_HORIZON_PAIRS)
    pairwise_rows = [
        {
            "scope": scope,
            "left_horizon": left,
            "right_horizon": right,
            "valid_count": 100,
            "covariance": 1.0 if left == right else 0.0,
            "correlation": 1.0 if left == right else 0.0,
        }
        for scope in ("pooled", "equal_date", "equal_decision")
        for left, right in pairs
    ]
    pl.DataFrame(pairwise_rows).write_csv(output_dir / "target_pairwise.csv")
    offsets = np.linspace(0, (TRAIN_END - TRAIN_START).days, 716, dtype=np.int64)
    dates = [TRAIN_START + timedelta(days=int(offset)) for offset in offsets]
    date_rows = [
        {
            "trade_date": trade_date,
            "left_horizon": left,
            "right_horizon": right,
            "valid_count": 10,
            "covariance": 1.0 if left == right else 0.0,
            "correlation": 1.0 if left == right else 0.0,
        }
        for trade_date in dates
        for left, right in pairs
    ]
    pl.DataFrame(date_rows).write_parquet(output_dir / "target_basis_by_date.parquet")
    decision_rows = [
        {
            "decision_idx": decision_idx,
            "left_horizon": left,
            "right_horizon": right,
            "valid_count": 10,
            "covariance": 1.0 if left == right else 0.0,
            "correlation": 1.0 if left == right else 0.0,
        }
        for decision_idx in range(55)
        for left, right in pairs
    ]
    pl.DataFrame(decision_rows).write_csv(output_dir / "target_basis_by_decision.csv")
    summary = {
        "train_end": str(TRAIN_END),
        "date_count": 716,
        "valid_counts_by_horizon": {str(horizon): 100 for horizon in HORIZONS},
        "coverage_fraction_by_horizon": {str(horizon): 0.5 for horizon in HORIZONS},
        "complete_case_count": 100,
        "pooled_target_correlation": np.eye(3).tolist(),
        "complete_case_target_covariance": np.eye(3).tolist(),
        "complete_case_correlation_sensitivity": np.eye(3).tolist(),
        "eigenvalues": [1.0, 1.0, 1.0],
        "eigenvectors_columns": np.eye(3).tolist(),
        "variance_shares": [1 / 3, 1 / 3, 1 / 3],
        "fixed_basis_rows": FIXED_TARGET_BASIS.tolist(),
        "fixed_basis_variance": [1.0, 1.0, 1.0],
        "raw_return_headline_correlation": np.eye(3).tolist(),
        "raw_complete_case_count": 100,
    }
    _write_json(output_dir / "target_basis_summary.json", summary)


def test_valid_target_basis_grid_validates(tmp_path: Path) -> None:
    _target_artifacts(tmp_path)
    validate_target_basis(tmp_path)


@pytest.mark.parametrize(
    "mutation",
    (
        "nan",
        "duplicate",
        "missing_pair",
        "invalid_correlation",
        "post_training_date",
        "summary_disagreement",
    ),
)
def test_target_basis_corruptions_are_rejected(tmp_path: Path, mutation: str) -> None:
    _target_artifacts(tmp_path)
    pair_path = tmp_path / "target_pairwise.csv"
    date_path = tmp_path / "target_basis_by_date.parquet"
    summary_path = tmp_path / "target_basis_summary.json"
    if mutation in {"nan", "duplicate", "missing_pair", "invalid_correlation"}:
        rows = pl.read_csv(pair_path).to_dicts()
        if mutation == "nan":
            rows[0]["correlation"] = float("nan")
        elif mutation == "duplicate":
            rows[-1] = rows[0].copy()
        elif mutation == "missing_pair":
            rows.pop()
        else:
            rows[0]["correlation"] = 1.1
        pl.DataFrame(rows).write_csv(pair_path)
    elif mutation == "post_training_date":
        rows = pl.read_parquet(date_path).to_dicts()
        rows[0]["trade_date"] = TRAIN_END + timedelta(days=1)
        pl.DataFrame(rows).write_parquet(date_path)
    else:
        summary = read_json_object(summary_path)
        summary["pooled_target_correlation"][0][1] = 0.1
        summary["pooled_target_correlation"][1][0] = 0.1
        _write_json(summary_path, summary)
    with pytest.raises(ValueError):
        validate_target_basis(tmp_path)


def test_correlation_matrix_accepts_tiny_upper_bound_roundoff() -> None:
    matrix = np.eye(3)
    matrix[0, 0] = 1.0000000000000002
    np.testing.assert_array_equal(
        _require_correlation_matrix(matrix, "correlation"), matrix
    )


def test_correlation_matrix_accepts_tiny_lower_bound_roundoff() -> None:
    matrix = np.eye(3)
    matrix[0, 1] = matrix[1, 0] = -1.0000000000000002
    np.testing.assert_array_equal(
        _require_correlation_matrix(matrix, "correlation"), matrix
    )


def test_correlation_matrix_rejects_material_bound_violation() -> None:
    matrix = np.eye(3)
    matrix[0, 1] = matrix[1, 0] = 1.000000001
    with pytest.raises(ValueError, match="invalid correlation"):
        _require_correlation_matrix(matrix, "correlation")
