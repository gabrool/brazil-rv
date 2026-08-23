from __future__ import annotations

import numpy as np
import torch

from brazil_rv.modeling.engine import collect_equity_input_ablation_predictions
from brazil_rv.modeling.feature_removal import (
    _components,
    _definition,
    _historical_single_fold,
    _preview_passes,
    _rank_standardize,
)


def test_rank_standardize_is_tie_aware() -> None:
    values = np.asarray([[1.0, 3.0], [1.0, 2.0], [2.0, 1.0]])
    ranked = _rank_standardize(values)
    assert ranked[0, 0] == ranked[1, 0]
    assert np.isclose(np.square(ranked).sum(axis=0), 1.0).all()


def test_components_use_fixed_absolute_threshold() -> None:
    matrix = np.eye(4)
    matrix[0, 1] = matrix[1, 0] = 0.81
    matrix[1, 2] = matrix[2, 1] = -0.90
    matrix[2, 3] = matrix[3, 2] = 0.79
    assert _components(matrix) == [[0, 1, 2], [3]]


def test_definition_separates_dynamic_and_slow_fields() -> None:
    assert _definition(["slow_3", "dynamic_2", "dynamic_0"]) == ((0, 2), (3,))


def test_preview_gate_requires_mean_and_each_fold() -> None:
    passing = {
        "fold_c": {"parent_minus_ablated_ic": 0.0002},
        "fold_a": {"parent_minus_ablated_ic": 0.0004},
        "fold_b": {"parent_minus_ablated_ic": -0.0001},
    }
    failing = {**passing, "fold_b": {"parent_minus_ablated_ic": 0.0006}}
    assert _preview_passes(passing)
    assert not _preview_passes(failing)


def test_historical_p0_single_is_normalized_without_recomputing() -> None:
    historical = {
        "parent_minus_zeroed_ic": 0.001,
        "per_horizon_parent_minus_zeroed_ic": {"15": 0.002, "30": 0.0},
        "date_count": 102,
        "moving_block_bootstrap": {
            "5": {"lower": -0.1, "upper": 0.1},
            "10": {"lower": -0.2, "upper": 0.2},
        },
    }
    normalized = _historical_single_fold(historical)
    assert normalized["parent_minus_ablated_ic"] == 0.001
    assert normalized["per_horizon_parent_minus_ablated_ic"]["15"] == 0.002
    assert normalized["block10_interval"] == historical["moving_block_bootstrap"]["10"]
    assert normalized["source"] == "imported_unchanged_from_experiment_39_p0_3"


class _SumModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(
        self,
        patches: torch.Tensor,
        history_patch_mask: torch.Tensor,
        instrument_mask: torch.Tensor,
        slow_features: torch.Tensor,
        state_position: torch.Tensor,
    ) -> torch.Tensor:
        del history_patch_mask, instrument_mask, state_position
        values = patches[:, :158].sum(dim=(-2, -1)) + slow_features[:, :158].sum(-1)
        return values.unsqueeze(-1).repeat(1, 1, 3) + self.anchor


def test_group_ablation_batches_variants_without_touching_metadata() -> None:
    batch_size = 2
    patches = torch.ones((batch_size, 173, 2, 52), dtype=torch.float32)
    slow = torch.ones((batch_size, 173, 32), dtype=torch.float32)
    batch = {
        "patches": patches,
        "history_patch_mask": torch.ones((batch_size, 173, 2), dtype=torch.bool),
        "instrument_mask": torch.ones((batch_size, 173), dtype=torch.bool),
        "slow_features": slow,
        "state_position": torch.ones(batch_size, dtype=torch.int64),
        "targets": torch.zeros((batch_size, 158, 3), dtype=torch.float32),
        "label_mask": torch.ones((batch_size, 158, 3), dtype=torch.bool),
        "raw_returns": torch.zeros((batch_size, 158, 3), dtype=torch.float32),
        "sample_id": torch.tensor([1, 0]),
        "date_idx": torch.tensor([4, 3]),
        "decision_idx": torch.tensor([2, 1]),
        "sample_valid_mask": torch.ones(batch_size, dtype=torch.bool),
    }
    reference, predictions = collect_equity_input_ablation_predictions(
        _SumModel(),
        [batch],
        {
            "dynamic": ((0,), ()),
            "slow": ((), (1,)),
            "both": ((0,), (1,)),
        },
        variants_per_forward=2,
    )
    assert reference.sample_id.tolist() == [0, 1]
    baseline = 2 * 52 + 32
    assert np.all(predictions["dynamic"] == baseline - 4)
    assert np.all(predictions["slow"] == baseline - 1)
    assert np.all(predictions["both"] == baseline - 5)
