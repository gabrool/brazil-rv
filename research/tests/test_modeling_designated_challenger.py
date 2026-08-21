from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import brazil_rv.modeling.analyze as analyze_module
import brazil_rv.modeling.designated_challenger as challenger_module
from brazil_rv.modeling.designated_challenger import (
    DESIGNATED_CHALLENGER_NAME,
    _crossfit_parent_observations,
    challenger_contract,
    compare_discovery_screen,
)
from brazil_rv.modeling.engine import EvaluationObservations


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


def test_designated_challenger_contract_freezes_recipe_and_selection_role() -> None:
    contract = challenger_contract()

    assert contract["name"] == DESIGNATED_CHALLENGER_NAME
    assert contract["retention_comparator"] == "canonical_parent_only"
    assert contract["beats_either_allowed"] is False
    assert contract["members"]["parent"]["rule"].endswith("patience3_raw")
    residual = contract["members"]["residual_auxiliary"]
    assert residual["target"] == "win_wdo_di_level_residual_rank"
    assert residual["weight"] == 0.5
    assert residual["readout"] == "final_ema_0995"
    assert contract["learned_weights"] is False


def test_parent_patience_predictions_are_crossfit_out_of_half(tmp_path: Path) -> None:
    target = _observations(np.zeros((102, 32, 3), dtype=np.float32)).targets
    first_parity_good = target.copy()
    first_parity_good[1::2] *= -1
    second_parity_good = -first_parity_good
    raw_epochs = [first_parity_good, second_parity_good, *([second_parity_good] * 18)]
    run = tmp_path / "run"
    _write_trajectory(run, raw_epochs)

    observations, directions = _crossfit_parent_observations(run)

    np.testing.assert_array_equal(observations.predictions, -target)
    assert [row["selected_epoch"] for row in directions] == [1, 2]


def test_discovery_screen_reports_both_comparators_but_selects_on_parent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(analyze_module, "BOOTSTRAP_REPLICATIONS", 20)
    target = _observations(np.zeros((10, 32, 3), dtype=np.float32)).targets
    candidate = _observations(target)
    parent_predictions = target.copy()
    parent_predictions[:, [0, 1]] = parent_predictions[:, [1, 0]]
    parent = _observations(parent_predictions)
    challenger = _observations(target)
    monkeypatch.setattr(
        challenger_module,
        "load_designated_challenger_members",
        lambda _fold, run_root: {
            **{f"parent_seed_{seed}": parent for seed in (11, 29, 47)},
            **{f"residual_seed_{seed}": challenger for seed in (11, 29, 47)},
        },
    )

    output = compare_discovery_screen(
        {"candidate": candidate},
        {f"seed_{seed}": parent for seed in (11, 29, 47)},
        fold="fold_a",
        candidate_rule="candidate_rule",
        parent_rule="patience3_raw",
        output_dir=tmp_path / "screen",
        run_root=tmp_path,
    )

    summary = json.loads((output / "screen_summary.json").read_text(encoding="utf-8"))
    assert summary["candidate_minus_canonical_ic"] > 0.0
    assert (
        0.0
        < summary["candidate_minus_challenger_ic"]
        < summary["candidate_minus_canonical_ic"]
    )
    assert summary["selection_contract"] == {
        "retention_comparator": "canonical_parent_only",
        "challenger_role": "informational_only",
        "beats_either_allowed": False,
    }
    assert (output / "vs_canonical" / "analysis.json").is_file()
    assert (output / "vs_designated_challenger" / "analysis.json").is_file()
