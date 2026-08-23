from __future__ import annotations

from pathlib import Path

import numpy as np

from brazil_rv.modeling.engine import EvaluationObservations
from brazil_rv.modeling.three_fold_sidecar_screen import (
    _gate,
    crossfit_patience_observations,
)


def _observations(predictions: np.ndarray) -> EvaluationObservations:
    samples, equities, _ = predictions.shape
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


def _write_trajectory(path: Path, raw_epochs: list[np.ndarray]) -> None:
    reference = _observations(raw_epochs[-1])
    path.mkdir()
    (path / "validation_predictions").mkdir()
    np.savez(
        path / "validation_reference.npz",
        **{
            name: getattr(reference, name)
            for name in EvaluationObservations.__dataclass_fields__
            if name != "predictions"
        },
    )
    for epoch, predictions in enumerate(raw_epochs, start=1):
        np.savez(
            path / "validation_predictions" / f"epoch_{epoch:02d}.npz",
            raw=predictions,
        )


def test_fold_c_patience_replay_is_out_of_half_for_105_dates(tmp_path: Path) -> None:
    target = _observations(np.zeros((105, 32, 3), dtype=np.float32)).targets
    odd_selection_good = target.copy()
    odd_selection_good[1::2] *= -1
    even_selection_good = -odd_selection_good
    trajectory = [
        odd_selection_good,
        even_selection_good,
        *([even_selection_good] * 18),
    ]
    run = tmp_path / "run"
    _write_trajectory(run, trajectory)

    observations, replays = crossfit_patience_observations(run)

    np.testing.assert_array_equal(observations.predictions, -target)
    assert [row["selected_epoch"] for row in replays] == [1, 2]

    for epoch in range(3, 20):
        (run / "validation_predictions" / f"epoch_{epoch:02d}.npz").unlink()
    retained, frozen = crossfit_patience_observations(run, replays)
    np.testing.assert_array_equal(retained.predictions, -target)
    assert frozen == replays


def test_three_fold_gate_requires_every_fold_nonnegative_and_mean_one_bp() -> None:
    assert _gate(
        {
            "fold_c": {"delta": 0.001},
            "fold_a": {"delta": 0.002},
            "fold_b": {"delta": 0.0},
        },
        "delta",
    )
    assert not _gate(
        {
            "fold_c": {"delta": 0.003},
            "fold_a": {"delta": 0.001},
            "fold_b": {"delta": -0.00001},
        },
        "delta",
    )
