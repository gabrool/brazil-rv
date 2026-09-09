from dataclasses import replace

import pytest
import torch
from torch import nn

from brazil_rv.v2.branch_diagnostics import gate_activations
from brazil_rv.v2.config import ModelConfig
from brazil_rv.v2.model import DailyMultiHorizonModel
from brazil_rv.v2.train import sam_accumulated_step


def test_gradient_logging_preserves_sam_update_and_rng() -> None:
    torch.manual_seed(19)
    reference = nn.Sequential(nn.Linear(3, 4), nn.Dropout(0.25), nn.Linear(4, 1))
    logged = nn.Sequential(nn.Linear(3, 4), nn.Dropout(0.25), nn.Linear(4, 1))
    logged.load_state_dict(reference.state_dict())
    x = torch.randn(8, 3)
    rng = torch.get_rng_state()
    result = None
    for model, record in ((reference, False), (logged, True)):
        torch.set_rng_state(rng)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        result = sam_accumulated_step(
            model,
            optimizer,
            (lambda: model(x).square().mean(),),
            record_branch_gradients=record,
        )
        if not record:
            expected_rng = torch.get_rng_state()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert all(
        torch.equal(a, b)
        for a, b in zip(reference.parameters(), logged.parameters(), strict=True)
    )
    assert set(result.branch_gradient_norms) == {"0", "2"}
    assert all(value > 0 for value in result.branch_gradient_norms.values())


@pytest.mark.parametrize("disabled", [False, True])
def test_gate_diagnostics_counts_states_and_preserves_model_rng(disabled: bool) -> None:
    config = ModelConfig(slow_feature_count=3, compile_forward=False, dropout=0.1)
    model = DailyMultiHorizonModel(replace(config, disable_fast_stream=disabled)).eval()
    slow = torch.randn(1, 3, 60, 3)
    current = torch.randn(1, 3, config.current_feature_count)
    batch = {
        "slow_features": slow,
        "slow_feature_mask": torch.ones_like(slow, dtype=torch.bool),
        "slow_history_mask": torch.ones(1, 3, 60, dtype=torch.bool),
        "slow_feature_age_sessions": torch.zeros_like(slow),
        "current_features": current,
        "current_feature_mask": torch.ones_like(current, dtype=torch.bool),
        "current_feature_age_sessions": torch.zeros_like(current),
        "active_mask": torch.tensor([[True, True, False]]),
        "fast_present": torch.tensor([[True, False, False]]),
        "fast_patch_values": torch.randn(1, 1, 5, 7),
        "fast_patch_valid": torch.ones(1, 1, 5, 7, dtype=torch.bool),
        "fast_patch_mask": torch.ones(1, 1, 5, dtype=torch.bool),
        "fast_name_index": torch.tensor([[0]]),
        "fast_state_position": torch.tensor([[5]]),
    }
    state = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state()
    report = gate_activations(model, [batch], torch.device("cpu"))
    assert torch.equal(rng, torch.get_rng_state())
    assert all(
        torch.equal(value, state[name]) for name, value in model.state_dict().items()
    )
    assert not model.training
    assert report["archive_fast_present_counts"] == [1, 1]
    assert report["effective_fast_present_counts"] == ([2, 0] if disabled else [1, 1])
    for gate in report["gates"].values():
        assert 0 < gate["0"]["mean"] < 1
        assert (gate["1"]["channel_observations"] == 0) == disabled
