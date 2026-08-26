from __future__ import annotations

import numpy as np

from brazil_rv.modeling.data import _append_nextgen_target
from brazil_rv.modeling.contract import EQUITY_COUNT
from brazil_rv.modeling.engine import EvaluationObservations, validation_primary_metric
from brazil_rv.modeling.metrics import primary_validation_score
from brazil_rv.preprocessing.contract import (
    DECISION_EQUITY_INDICES,
    EQUITY_SESSION_MINUTES,
    MIN_ACTIVE_EQUITIES,
)
from brazil_rv.preprocessing.nextgen_targets import (
    center_leg_cross_section,
    exact_leg_returns,
    mutation_causality_audit,
)


def test_exact_leg_returns_use_two_disjoint_exact_15_minute_windows() -> None:
    grid = np.zeros((1, EQUITY_SESSION_MINUTES, 5), dtype=np.float64)
    observed = np.zeros(grid.shape[:2], dtype=bool)
    entry = DECISION_EQUITY_INDICES[0]
    grid[0, entry, 0] = 100.0
    grid[0, entry + 14, 3] = 101.0
    grid[0, entry + 15, 0] = 102.0
    grid[0, entry + 29, 3] = 104.0
    observed[0, [entry, entry + 14, entry + 15, entry + 29]] = True

    values, mask = exact_leg_returns(grid, observed)

    assert mask[0, 0].all()
    np.testing.assert_allclose(
        values[0, 0], [np.log(101.0 / 100.0), np.log(104.0 / 102.0)]
    )
    assert all(mutation_causality_audit().values())


def test_leg_target_construction_centers_each_leg_independently() -> None:
    count = MIN_ACTIVE_EQUITIES + 2
    raw = np.zeros((count, 1, 2), dtype=np.float32)
    raw[:, 0, 0] = np.arange(count, dtype=np.float32)
    raw[:, 0, 1] = np.arange(count, 0, -1, dtype=np.float32)
    mask = np.ones_like(raw, dtype=bool)
    sigma = np.linspace(0.5, 1.5, count)

    masked, targets, label_mask, medians = center_leg_cross_section(raw, mask, sigma)

    assert label_mask.all()
    np.testing.assert_array_equal(masked, raw)
    np.testing.assert_allclose(targets[:, 0, 0].mean(), 0.0, atol=1e-7)
    np.testing.assert_allclose(targets[:, 0, 1].mean(), 0.0, atol=1e-7)
    np.testing.assert_allclose(medians[0], np.median(raw, axis=0)[0])


def test_fourth_head_is_appended_but_excluded_from_primary_metric() -> None:
    batch_size = 4
    equity_count = EQUITY_COUNT
    common = {
        "targets": np.zeros((batch_size, equity_count, 3), dtype=np.float32),
        "label_mask": np.ones((batch_size, equity_count, 3), dtype=bool),
        "raw_returns": np.zeros((batch_size, equity_count, 3), dtype=np.float32),
    }
    arrays = {
        "leg_targets.npy": np.zeros((2, equity_count, 2, 2), dtype=np.float32),
        "leg_label_mask.npy": np.ones((2, equity_count, 2, 2), dtype=bool),
        "leg_raw_returns.npy": np.zeros((2, equity_count, 2, 2), dtype=np.float32),
    }
    arrays["leg_targets.npy"][..., 0] = 0.25
    arrays["leg_raw_returns.npy"][..., 0] = 0.5
    _append_nextgen_target(
        common,
        arrays,
        np.asarray([0, 0, 1, 1]),
        np.asarray([0, 1, 0, 1]),
        np.ones((batch_size, equity_count), dtype=bool),
        batch_size,
    )
    assert common["targets"].shape[-1] == 4
    np.testing.assert_allclose(common["targets"][..., 3], 0.25)

    generator = np.random.default_rng(48)
    targets = generator.normal(size=(batch_size, equity_count, 4)).astype(np.float32)
    predictions = targets.copy()
    predictions[..., 3] *= -1.0
    mask = np.ones_like(targets, dtype=bool)
    observations = EvaluationObservations(
        predictions=predictions,
        targets=targets,
        raw_returns=np.zeros_like(targets),
        label_mask=mask,
        sample_id=np.arange(batch_size, dtype=np.int64),
        date_idx=np.asarray([0, 0, 1, 1], dtype=np.int64),
        decision_idx=np.arange(batch_size, dtype=np.int64),
    )
    expected = primary_validation_score(
        predictions[..., :3], targets[..., :3], mask[..., :3], observations.date_idx
    )
    assert validation_primary_metric(observations) == expected
