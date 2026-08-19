from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from brazil_rv.modeling.trajectory import (
    load_frozen_selection,
    model_state_dicts_for_rule,
    predictions_for_rule,
)


def test_crossfit_selected_patience_rule_restores_raw_best_epoch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "validation_predictions").mkdir()
    (run_dir / "trajectory_diagnostics.json").write_text(
        json.dumps({"patience3": {"selected_epoch": 4, "stopped_epoch": 7}}),
        encoding="utf-8",
    )
    torch.save(
        {"epoch": 4, "model_state_dict": {"weight": torch.tensor([4.0])}},
        run_dir / "checkpoints" / "epoch_04.pt",
    )
    expected_predictions = np.asarray([[[4.0]]], dtype=np.float32)
    np.savez(
        run_dir / "validation_predictions" / "epoch_04.npz",
        raw=expected_predictions,
    )
    selection_path = tmp_path / "trajectory_selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "schema": "TRAJECTORY_SELECTION",
                "selected_rule": "patience3_raw",
            }
        ),
        encoding="utf-8",
    )

    assert load_frozen_selection(selection_path)["selected_rule"] == "patience3_raw"
    states = model_state_dicts_for_rule(run_dir, "patience3_raw")
    torch.testing.assert_close(states[0]["weight"], torch.tensor([4.0]))
    np.testing.assert_array_equal(
        predictions_for_rule(run_dir, "patience3_raw"), expected_predictions
    )
