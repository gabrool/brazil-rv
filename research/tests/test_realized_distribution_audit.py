from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from brazil_rv.modeling.audit_realized_distributions import (
    AUDIT_JSON,
    AUDIT_NAME,
    AUDIT_VERSION,
    FEATURE_PARQUET,
    FROZEN_CONTEXT_ABLATION,
    FROZEN_FEATURE_ABLATION,
    RETAINED_MACRO_SYMBOLS,
    TARGET_PARQUET,
    DeterministicSystematicSample,
    StreamingDistribution,
    TargetStructureDiagnostics,
    ViolationLog,
    _add_nonbinary_violation,
    _add_structural_violation,
    _global_consumed_mask,
    _warnings,
    validate_realized_distribution_audit,
)
from brazil_rv.modeling.contract import HORIZONS, TRAIN_END, TRAIN_START
from brazil_rv.preprocessing.contract import DECISION_GLOBAL_INDICES


def test_streaming_distribution_conditions_exact_statistics_on_validity() -> None:
    values = np.asarray([[-1.0, 0.0, 1.0, np.nan], [5.0, 6.0, 7.0, 8.0]])
    valid = np.asarray([[True, True, True, True], [False, False, False, False]])
    distribution = StreamingDistribution(sample_capacity=8)
    distribution.update(values, valid, (-1.0, 1.0))
    row = distribution.row(
        table="feature",
        family="synthetic",
        symbol=None,
        channel_index=0,
        channel_name="x",
        status="observed_numerical",
        clip_bounds=(-1.0, 1.0),
    )
    assert row["valid_count"] == 4
    assert row["finite_count"] == 3
    assert row["nonfinite_count"] == 1
    assert row["mean"] == 0.0
    assert row["standard_deviation"] == pytest.approx(math.sqrt(2.0 / 3.0))
    assert row["minimum"] == -1.0
    assert row["maximum"] == 1.0
    assert row["zero_fraction"] == pytest.approx(1.0 / 3.0)
    assert row["clipping_boundary_fraction"] == pytest.approx(2.0 / 3.0)


def test_deterministic_quantile_sample_is_chunk_invariant_and_bounded() -> None:
    values = np.linspace(-5.0, 5.0, 101)
    whole = DeterministicSystematicSample(capacity=9)
    whole.update(values)
    chunked = DeterministicSystematicSample(capacity=9)
    for chunk in np.array_split(values, 13):
        chunked.update(chunk)
    assert whole.seen == chunked.seen == values.size
    assert whole.values.size <= 9
    np.testing.assert_array_equal(whole.positions, chunked.positions)
    np.testing.assert_array_equal(whole.values, chunked.values)
    assert whole.stride == chunked.stride


def test_structural_mask_and_target_contract_diagnostics() -> None:
    violations = ViolationLog()
    _add_structural_violation(
        violations,
        np.asarray([0.0, 3.0, 0.0]),
        np.asarray([True, True, False]),
        "structural",
    )
    _add_nonbinary_violation(
        violations,
        np.asarray([0.0, 0.5, 1.0]),
        np.ones(3, dtype=bool),
        "mask",
    )
    targets = np.asarray(
        [
            [
                [[-0.75, -0.5, -1.0]],
                [[-0.25, -0.5, 0.0]],
                [[0.25, 0.5, 0.0]],
                [[0.75, 0.5, 1.0]],
            ]
        ],
        dtype=np.float32,
    ).reshape(1, 4, 1, 3)
    labels = np.ones_like(targets, dtype=bool)
    eligible = np.ones((1, 4, 1), dtype=bool)
    cross_mean = {h: StreamingDistribution(16) for h in HORIZONS}
    cross_std = {h: StreamingDistribution(16) for h in HORIZONS}
    diagnostics = TargetStructureDiagnostics()
    diagnostics.update(targets, labels, eligible, cross_mean, cross_std, violations)
    rows = diagnostics.rows()
    assert rows["30m"]["label_coverage"] == 1.0
    assert rows["30m"]["extreme_attainable_rank_fraction"] == 0.5
    assert rows["60m"]["cross_section_tie_prevalence"] == 1.0
    assert rows["120m"]["label_tie_prevalence"] == 0.5
    failures = {row["code"]: row["count"] for row in violations.rows()}
    assert failures == {
        "nonbinary_mask_channel": 1,
        "target_outside_open_interval": 2,
        "unexpected_nonzero_structural_channel": 1,
    }


def test_warning_policy_and_global_consumption_are_deterministic() -> None:
    rows = [
        {
            "status": "observed_numerical",
            "family": "synthetic",
            "symbol": None,
            "channel_name": "x",
            "clipping_boundary_fraction": 0.02,
            "zero_fraction": 0.995,
            "standard_deviation": 0.0,
            "finite_count": 100,
        }
    ]
    rows.append(
        {
            **rows[0],
            "channel_name": "large_scale",
            "clipping_boundary_fraction": 0.0,
            "zero_fraction": 0.0,
            "standard_deviation": 20.0,
        }
    )
    assert {warning["code"] for warning in _warnings(rows)} == {
        "material_clipping_boundary_fraction",
        "very_high_observed_zero_fraction",
        "constant_valid_channel",
        "large_valid_dispersion",
    }
    readiness = np.zeros((2, len(DECISION_GLOBAL_INDICES)), dtype=bool)
    readiness[0, 0] = True
    readiness[1, -1] = True
    consumed = _global_consumed_mask(readiness)
    first = DECISION_GLOBAL_INDICES[0]
    last = DECISION_GLOBAL_INDICES[-1]
    assert consumed[0, : first - 345].sum() == 0
    assert consumed[0, first - 345 : first].all()
    assert consumed[1, last - 345 : last].all()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_audit_artifact_validator_rejects_failures_and_tampering(
    tmp_path: Path,
) -> None:
    feature_path = tmp_path / FEATURE_PARQUET
    target_path = tmp_path / TARGET_PARQUET
    common = {
        "table": "feature",
        "family": "synthetic",
        "channel_name": "x",
        "status": "valid_numerical",
        "valid_count": 1,
        "finite_count": 1,
        "nonfinite_count": 0,
        "mean": 0.0,
        "standard_deviation": 0.0,
        "median": 0.0,
        "mad": 0.0,
        "minimum": 0.0,
        "maximum": 0.0,
        "zero_fraction": 1.0,
        "clipping_boundary_fraction": None,
        "quantile_sample_method": "deterministic_systematic_valid_ordinal_v1",
    }
    pl.DataFrame([common], infer_schema_length=None).write_parquet(feature_path)
    pl.DataFrame(
        [
            {
                **common,
                "table": "target",
                "family": "target",
                "channel_name": "target_30m",
            }
        ],
        infer_schema_length=None,
    ).write_parquet(target_path)
    audit = {
        "audit_name": AUDIT_NAME,
        "audit_version": AUDIT_VERSION,
        "training_only": True,
        "test_metrics_accessed": False,
        "retained_macro_symbols": list(RETAINED_MACRO_SYMBOLS),
        "applied_ablations": {
            "context": {"key": FROZEN_CONTEXT_ABLATION},
            "feature": {"key": FROZEN_FEATURE_ABLATION},
        },
        "split_identity": {
            "split": "train",
            "start": str(TRAIN_START),
            "end": str(TRAIN_END),
        },
        "feature_store": {"manifest_sha256": "synthetic"},
        "hard_failures": [],
        "warnings": [],
        "summary_verdict": "pass",
        "row_counts": {
            FEATURE_PARQUET: 1,
            TARGET_PARQUET: 1,
        },
        "output_sha256": {
            FEATURE_PARQUET: _sha256(feature_path),
            TARGET_PARQUET: _sha256(target_path),
        },
    }
    audit_path = tmp_path / AUDIT_JSON
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    assert validate_realized_distribution_audit(audit_path) == audit

    audit["hard_failures"] = [{"code": "synthetic_failure"}]
    audit["summary_verdict"] = "fail"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="hard failures"):
        validate_realized_distribution_audit(audit_path)
    validate_realized_distribution_audit(audit_path, require_pass=False)

    feature_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="output changed"):
        validate_realized_distribution_audit(audit_path, require_pass=False)
