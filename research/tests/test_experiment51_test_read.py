from __future__ import annotations

from datetime import date

import numpy as np

from brazil_rv.modeling.engine import EvaluationObservations
from brazil_rv.modeling.experiment51_test_read import (
    difficulty_context,
    interpretation_band,
    paired_h2_minus_h1,
)


def test_interpretation_band_boundaries_are_exact() -> None:
    assert interpretation_band(0.040) == "A"
    assert interpretation_band(0.039999) == "B"
    assert interpretation_band(0.035) == "B"
    assert interpretation_band(0.034999) == "C"
    assert interpretation_band(0.030) == "C"
    assert interpretation_band(0.029999) == "D"


def test_paired_staleness_is_h2_minus_h1_and_excludes_date_259() -> None:
    daily = np.concatenate(
        (
            np.full(129, 0.04),
            np.full(129, 0.03),
            np.asarray([50_000.0]),
        )
    )

    paired = paired_h2_minus_h1(daily)

    assert paired.shape == (129,)
    assert np.allclose(paired, -0.01)


def test_difficulty_context_uses_only_raw_labels_and_masks() -> None:
    raw = np.asarray(
        [
            [[0.0], [0.1], [0.2]],
            [[0.1], [0.2], [0.3]],
        ],
        dtype=np.float32,
    )
    observations = EvaluationObservations(
        predictions=np.full_like(raw, 50_000.0),
        targets=np.full_like(raw, -50_000.0),
        raw_returns=raw,
        label_mask=np.ones_like(raw, dtype=bool),
        sample_id=np.asarray([1, 2]),
        date_idx=np.asarray([10, 10]),
        decision_idx=np.asarray([0, 1]),
    )

    result = difficulty_context(observations, {10: date(2025, 7, 7)})

    assert len(result) == 1
    assert result[0]["quarter"] == "2025Q3"
    assert np.isclose(result[0]["active_universe_size"], 3.0)
    assert np.isclose(result[0]["cross_sectional_dispersion"], np.std(raw[0, :, 0]))
    assert np.isclose(result[0]["per_name_vol_level"], np.std([0.0, 0.1], ddof=1))
