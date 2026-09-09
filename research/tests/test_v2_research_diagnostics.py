from dataclasses import replace

import numpy as np
import pytest

from brazil_rv.v2.evaluate import _spearman
from brazil_rv.v2.normalization import average_ranks
from brazil_rv.v2.research_diagnostics import (
    momentum_diagnostics,
    pooled_momentum_diagnostics,
    rank_residual_ic,
)
from brazil_rv.v2.research_rounds import _single_spread_series
from test_v2_evaluate import _fixture


def test_rank_residual_ic_matches_daily_ols_and_does_not_invent_signal():
    rng = np.random.default_rng(31)
    momentum = rng.normal(size=60)
    score = momentum + rng.normal(size=60)
    target = rng.normal(size=60)
    selected = np.ones(60, dtype=bool)
    selected[:5] = False
    x = np.column_stack((np.ones(55), average_ranks(momentum[selected])))
    y = average_ranks(score[selected])
    residual = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    expected = _spearman(residual, target[selected], np.ones(55, dtype=bool))
    assert rank_residual_ic(score, momentum, target, selected) == pytest.approx(
        expected
    )
    assert np.isnan(rank_residual_ic(momentum, momentum, target, selected))
    assert np.isnan(rank_residual_ic(score, momentum, target, np.arange(60) < 19))


def test_spread_attribution_keeps_original_book_tails_and_missing_support():
    inputs = _fixture()
    momentum = np.broadcast_to(np.arange(60)[None, :, None], inputs.scores.shape)
    mask = np.ones_like(momentum, dtype=bool)
    mask[:, :10] = False
    mask[3] = False
    result = momentum_diagnostics(inputs, momentum, mask)
    contributions = np.asarray(
        [
            [
                row[f"spread_contribution_{group}_bps_per_session"]
                for group in ("extreme_momentum", "middle_momentum", "unknown_momentum")
            ]
            for row in result["daily"]
        ],
        dtype=float,
    )
    np.testing.assert_allclose(
        contributions.sum(axis=1), _single_spread_series(inputs), equal_nan=True
    )
    assert contributions[3, 0] == contributions[3, 1] == 0
    assert contributions[3, 2] == pytest.approx(_single_spread_series(inputs)[3])
    assert result["summary"]["primary_ic"]["defined_sessions"] == 15
    assert (
        result["summary"]["primary_ic_on_momentum_supported_population"][
            "defined_sessions"
        ]
        == 14
    )
    assert result["selection_weight"] == 0
    assert result["extreme_momentum_spread_share"] == pytest.approx(
        np.nansum(contributions[:, 0]) / np.nansum(contributions)
    )


def test_diagnostics_are_daily_causal_and_pooled_by_session():
    inputs = _fixture()
    momentum = np.broadcast_to(np.arange(60)[None, :, None], inputs.scores.shape).copy()
    mask = np.ones_like(momentum, dtype=bool)
    first = momentum_diagnostics(inputs, momentum, mask)
    changed_scores = inputs.scores.copy()
    changed_scores[8:] *= -1
    momentum[8:] *= -1
    changed = momentum_diagnostics(
        replace(inputs, scores=changed_scores), momentum, mask
    )
    assert first["daily"][:8] == changed["daily"][:8]
    second = {**changed, "daily": changed["daily"][:5]}
    pooled = pooled_momentum_diagnostics({"F1": first, "F2": second})
    key = "composite_rank_correlation_with_momentum"
    values = np.asarray(
        [row[key] for row in first["daily"] + second["daily"]], dtype=float
    )
    assert pooled["summary"][key]["mean"] == pytest.approx(np.nanmean(values))
    assert pooled["summary"][key]["sd"] == pytest.approx(np.nanstd(values, ddof=1))
    assert pooled["summary"][key]["possible_sessions"] == 30
