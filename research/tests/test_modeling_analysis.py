from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from brazil_rv.modeling.analyze import (
    align_observations,
    compare_ensembles,
    compare_observation_ensembles,
)
from brazil_rv.modeling.engine import EvaluationObservations
from brazil_rv.modeling.metrics import rank_average_predictions


def _observations(predictions: np.ndarray) -> EvaluationObservations:
    samples, equities, horizons = predictions.shape
    targets = np.broadcast_to(
        np.linspace(-1.0, 1.0, equities, dtype=np.float32)[None, :, None],
        predictions.shape,
    ).copy()
    return EvaluationObservations(
        predictions=predictions.astype(np.float32),
        targets=targets,
        raw_returns=targets.copy(),
        label_mask=np.ones_like(targets, dtype=bool),
        sample_id=np.arange(samples, dtype=np.int64),
        date_idx=np.arange(samples, dtype=np.int64),
        decision_idx=np.zeros(samples, dtype=np.int64),
    )


def test_rank_ensemble_uniformly_averages_tie_aware_member_ranks() -> None:
    mask = np.ones((1, 3, 1), dtype=bool)
    left = np.asarray([[[3.0], [1.0], [2.0]]], dtype=np.float32)
    right = np.asarray([[[1.0], [3.0], [2.0]]], dtype=np.float32)
    np.testing.assert_array_equal(
        rank_average_predictions((left, right), mask),
        np.ones((1, 3, 1), dtype=np.float32),
    )


def test_alignment_reorders_by_sample_id_and_rejects_target_mismatch() -> None:
    baseline = _observations(np.ones((10, 32, 3), dtype=np.float32))
    reverse = np.arange(9, -1, -1)
    reordered = EvaluationObservations(
        **{
            name: getattr(baseline, name)[reverse]
            for name in EvaluationObservations.__dataclass_fields__
        }
    )
    aligned = align_observations({"left": baseline, "right": reordered})
    np.testing.assert_array_equal(aligned["left"].sample_id, aligned["right"].sample_id)
    broken_targets = reordered.targets.copy()
    broken_targets[0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="targets"):
        align_observations(
            {
                "left": baseline,
                "right": EvaluationObservations(
                    **{
                        **{
                            name: getattr(reordered, name)
                            for name in EvaluationObservations.__dataclass_fields__
                        },
                        "targets": broken_targets,
                    }
                ),
            }
        )


def _write_run(
    path: Path,
    *,
    seed: int,
    observations: EvaluationObservations,
) -> None:
    path.mkdir()
    (path / "validation_predictions").mkdir()
    (path / "run_manifest.json").write_text(
        json.dumps({"seed": seed}), encoding="utf-8"
    )
    np.savez(
        path / "validation_reference.npz",
        **{
            name: getattr(observations, name)
            for name in EvaluationObservations.__dataclass_fields__
            if name != "predictions"
        },
    )
    np.savez(
        path / "validation_predictions" / "epoch_20.npz",
        raw=observations.predictions,
    )


def test_comparison_reports_member_ensemble_bootstrap_and_guardrails(
    tmp_path: Path,
) -> None:
    base = _observations(np.zeros((10, 32, 3), dtype=np.float32))
    target = base.targets
    candidate_runs = []
    parent_runs = []
    for seed, noise in ((11, 0.001), (29, -0.001)):
        candidate = EvaluationObservations(
            **{**base.__dict__, "predictions": target + noise}
        )
        parent = EvaluationObservations(
            **{**base.__dict__, "predictions": -target + noise}
        )
        candidate_path = tmp_path / f"candidate_{seed}"
        parent_path = tmp_path / f"parent_{seed}"
        _write_run(candidate_path, seed=seed, observations=candidate)
        _write_run(parent_path, seed=seed, observations=parent)
        candidate_runs.append(candidate_path)
        parent_runs.append(parent_path)
    output = compare_ensembles(
        candidate_runs,
        parent_runs,
        candidate_rule="final_raw",
        parent_rule="final_raw",
        output_dir=tmp_path / "analysis",
    )
    report = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    assert report["candidate"]["learned_weights"] is False
    assert report["candidate"]["ensemble_ic"] > report["parent"]["ensemble_ic"]
    assert set(report["per_date_delta_bootstrap"]) == {"5", "10"}
    assert len(report["horizon_guardrails"]) == 3
    assert len(report["time_of_day_guardrails"]) == 1
    assert (output / "daily_delta.parquet").is_file()


def test_comparison_allows_different_candidate_and_parent_member_counts(
    tmp_path: Path,
) -> None:
    predictions = np.broadcast_to(
        np.linspace(-1.0, 1.0, 32, dtype=np.float32)[None, :, None],
        (10, 32, 3),
    ).copy()
    observations = _observations(predictions)
    output = compare_observation_ensembles(
        {"parent_11": observations, "variant_11": observations},
        {"parent_11": observations},
        candidate_rule="uniform_parent_plus_variant",
        parent_rule="parent_only",
        output_dir=tmp_path / "analysis",
    )
    report = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    assert len(report["candidate"]["member_ic"]) == 2
    assert len(report["parent"]["member_ic"]) == 1
