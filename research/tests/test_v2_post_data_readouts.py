from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from brazil_rv.v2 import research_rounds as rr
from brazil_rv.v2.execution_policy import ExecutionPolicy, traded_signal
from brazil_rv.v2.post_data_readouts import (
    alignment,
    require_full_primary_scores,
    score_source,
)


def test_attention_radius_readouts_preserve_original_score_sources():
    choices = {
        "recipes": {
            name: "asam20" for name in ("TE_slow", "TL_slow", "TE_family", "TE_all")
        }
    }
    for name in choices["recipes"]:
        assert score_source(name, choices) == (name, "asam20")
        assert score_source(f"{name}_asam50", choices) == (name, "asam50")
    assert score_source("S0", choices) == ("S0", "incumbent")


def test_longer_blocks_keep_fold_boundaries_and_missing_dates():
    # Each fold is constant, so crossing the boundary is unnecessary and the
    # weighted mean is known exactly under either block choice.
    values = [np.ones(120), np.full(120, 3.0)]
    for block in (20, 60):
        result = rr._folded_bootstrap(values, replications=50, block_length=block)
        assert result["estimate"] == result["lower_95"] == result["upper_95"] == 2
        assert result["block_length_sessions"] == block
        assert result["fold_boundary_preserved"]
    values[0][:] = np.nan
    result = rr._folded_bootstrap(values, replications=50, block_length=60)
    assert result["estimate"] == 3
    assert result["possible_observations"] == 240
    assert result["finite_observations"] == 120


@pytest.mark.parametrize("registered_policy", [False, True])
def test_alignment_compares_heads_and_target_views_on_shared_names(registered_policy):
    shape = (2, 32, 5)
    values = np.broadcast_to(np.arange(32)[None, :, None], shape).astype(float).copy()
    masks = np.ones(shape, bool)
    masks[0, 0, 4] = False
    inputs = SimpleNamespace(
        dates=[date(2020, 1, 1) + timedelta(days=i) for i in range(2)],
        active=np.ones(shape[:2], bool),
        scores=values,
        score_mask=np.ones(shape, bool),
        neutral_midrank_targets=values,
        neutral_target_mask=masks,
        scaled_midrank_targets=values,
        scaled_target_mask=masks,
        shareholder_midrank_targets=values,
        shareholder_target_mask=masks,
        price_midrank_targets=values,
        price_target_mask=masks,
        execution_policy=ExecutionPolicy(theta=0.25, horizons=(3, 5, 10))
        if registered_policy
        else None,
    )
    if registered_policy:
        inputs.score_mask[..., :2] = False
        inputs.scores[..., :2] = np.nan
    result = alignment(inputs)
    assert result["populations"]["common_scored_outcomes"] == [31, 32]
    assert result["populations"]["same_target_view_outcomes"] == [31, 32]
    assert result["populations"]["individual_head_outcomes"][0] == [32, 32, 31]
    if registered_policy:
        _, valid = traded_signal(inputs, inputs.execution_policy)
        assert result["populations"]["score_available_to_book"] == valid.sum(1).tolist()
    for name, series in result["series"].items():
        if name == "composite_persistence_1":
            assert series == [None, 1.0]
        elif name == "composite_persistence_5":
            assert series == [None, None]
        else:
            np.testing.assert_allclose(series, [1, 1])


def test_primary_population_check_allows_absent_untrained_heads_only():
    active = np.ones((2, 32), bool)
    mask = np.ones((2, 32, 5), bool)
    mask[..., :2] = False
    require_full_primary_scores(mask, active)
    mask[0, 0, 2] = False
    with pytest.raises(ValueError, match="PIT-active"):
        require_full_primary_scores(mask, active)
