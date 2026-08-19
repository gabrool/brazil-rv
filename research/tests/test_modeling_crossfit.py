from __future__ import annotations

import numpy as np

from brazil_rv.modeling.crossfit import (
    CANDIDATE_RULES,
    CENTERED_WEIGHT_RULE,
    TrajectoryMember,
    centered_epoch_window,
    crossfit_fold,
)
from brazil_rv.modeling.engine import EvaluationObservations


def _reference(date_count: int = 10) -> EvaluationObservations:
    equities = 32
    horizons = 3
    targets = np.broadcast_to(
        np.linspace(-1.0, 1.0, equities, dtype=np.float32)[None, :, None],
        (date_count, equities, horizons),
    ).copy()
    return EvaluationObservations(
        predictions=np.empty_like(targets),
        targets=targets,
        raw_returns=targets.copy(),
        label_mask=np.ones_like(targets, dtype=bool),
        sample_id=np.arange(date_count, dtype=np.int64),
        date_idx=np.arange(date_count, dtype=np.int64),
        decision_idx=np.zeros(date_count, dtype=np.int64),
    )


def test_patience_checkpoint_is_selected_on_opposite_date_parity() -> None:
    reference = _reference()
    target = reference.targets
    odd_good = target.copy()
    odd_good[1::2] *= -1
    even_good = -odd_good
    raw_epochs = [odd_good, even_good, *([even_good] * 18)]
    ema_epochs = [target.copy() for _ in range(20)]
    member = TrajectoryMember(
        name="seed_11",
        reference=reference,
        fixed_predictions={
            rule: target.copy()
            for rule in CANDIDATE_RULES
            if not rule.startswith("patience")
        },
        epoch_predictions={"raw": raw_epochs, "ema_0995": ema_epochs},
        parity_predictions={CENTERED_WEIGHT_RULE: {"odd": odd_good, "even": even_good}},
    )

    report = crossfit_fold([member])

    raw = report["rules"]["patience3_raw"]
    ema = report["rules"]["patience3_ema_0995"]
    centered = report["rules"][CENTERED_WEIGHT_RULE]
    assert raw["ensemble_crossfit_ic"] == -1.0
    assert ema["ensemble_crossfit_ic"] == 1.0
    assert centered["ensemble_crossfit_ic"] == -1.0
    directions = report["rule_selection_crossfit"]["directions"]
    assert (
        directions[0]["rules"]["patience3_raw"]["member_patience_replay"]["seed_11"][
            "selected_epoch"
        ]
        == 1
    )
    assert (
        directions[1]["rules"]["patience3_raw"]["member_patience_replay"]["seed_11"][
            "selected_epoch"
        ]
        == 2
    )
    assert directions[0]["rules"][CENTERED_WEIGHT_RULE]["member_patience_replay"][
        "seed_11"
    ]["averaged_epochs"] == [1, 2, 3, 4, 5]


def test_centered_epoch_window_is_five_checkpoints_and_shifts_at_boundaries() -> None:
    assert centered_epoch_window(1) == (1, 2, 3, 4, 5)
    assert centered_epoch_window(8) == (6, 7, 8, 9, 10)
    assert centered_epoch_window(20) == (16, 17, 18, 19, 20)
