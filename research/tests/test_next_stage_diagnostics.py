from __future__ import annotations

import numpy as np

from brazil_rv.modeling.metrics import primary_validation_score
from brazil_rv.modeling.next_stage_diagnostics import (
    crossfit_patience_observations,
)


def _write_trajectory(path) -> None:
    path.mkdir()
    prediction_dir = path / "validation_predictions"
    prediction_dir.mkdir()
    dates, equities = 102, 32
    targets = np.broadcast_to(
        np.linspace(-1.0, 1.0, equities, dtype=np.float32)[None, :, None],
        (dates, equities, 3),
    ).copy()
    odd_good = targets.copy()
    odd_good[1::2] *= -1
    even_good = -odd_good
    final_ema = np.roll(targets, 2, axis=1)
    np.savez(
        path / "validation_reference.npz",
        targets=targets,
        raw_returns=targets,
        label_mask=np.ones_like(targets, dtype=bool),
        sample_id=np.arange(dates, dtype=np.int64),
        date_idx=np.arange(dates, dtype=np.int64),
        decision_idx=np.zeros(dates, dtype=np.int64),
    )
    for epoch in range(1, 21):
        raw = odd_good if epoch == 1 else even_good
        np.savez(
            prediction_dir / f"epoch_{epoch:02d}.npz",
            raw=raw,
            ema_0995=final_ema,
        )


def test_r1_selects_patience_on_opposite_parity_before_blending(tmp_path) -> None:
    run = tmp_path / "run"
    _write_trajectory(run)
    parent, directions = crossfit_patience_observations(run)
    blended, blend_directions = crossfit_patience_observations(run, blend="final")
    assert [row["selected_epoch"] for row in directions] == [1, 2]
    assert blend_directions == directions
    parent_score = primary_validation_score(
        parent.predictions,
        parent.targets,
        parent.label_mask,
        parent.date_idx,
    )
    blended_score = primary_validation_score(
        blended.predictions,
        blended.targets,
        blended.label_mask,
        blended.date_idx,
    )
    assert parent_score == -1.0
    assert -1.0 < blended_score < 1.0
