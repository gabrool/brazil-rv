from dataclasses import replace

import pytest
import torch

from brazil_rv.v2.config import ModelConfig
from brazil_rv.v2.model import DailyMultiHorizonModel
from brazil_rv.v2.train import _model_forward, _to_device


def _models_and_batch():
    config = ModelConfig(
        slow_feature_count=2,
        current_feature_count=0,
        disable_fast_stream=True,
        slow_lookback=20,
        dropout=0.2,
        compile_forward=False,
    )
    torch.manual_seed(19)
    parent = DailyMultiHorizonModel(config)
    rng = torch.random.get_rng_state().clone()
    torch.manual_seed(19)
    child = DailyMultiHorizonModel(
        replace(config, sidecar_feature_counts=(("events", 2),))
    )
    assert torch.equal(torch.random.get_rng_state(), rng)
    for key, value in parent.state_dict().items():
        assert torch.equal(child.state_dict()[key], value), key
    slow = torch.randn(2, 3, 20, 2)
    batch = dict(
        slow_features=slow,
        slow_feature_mask=torch.ones_like(slow, dtype=torch.bool),
        slow_history_mask=torch.ones(2, 3, 20, dtype=torch.bool),
        slow_feature_age_sessions=torch.zeros_like(slow),
        active_mask=torch.ones(2, 3, dtype=torch.bool),
        targets=torch.zeros(2, 3, 5),
        target_mask=torch.ones(2, 3, 5, dtype=torch.bool),
    )
    return parent, child, batch


@pytest.mark.parametrize("training", [False, True])
def test_all_invalid_families_preserve_s0_forward_rng_and_parent_gradients(training):
    parent, child, batch = _models_and_batch()
    parent.train(training)
    child.train(training)
    extended = dict(
        batch,
        sidecar_events_values=torch.full((2, 3, 2), float("nan")),
        sidecar_events_valid=torch.zeros(2, 3, 2, dtype=torch.bool),
        sidecar_events_age_sessions=torch.full((2, 3, 2), -1.0),
    )
    child_batch = _to_device(extended, torch.device("cpu"))
    torch.manual_seed(42)
    expected = _model_forward(parent, batch)
    expected.square().sum().backward()
    rng = torch.random.get_rng_state().clone()
    torch.manual_seed(42)
    actual = _model_forward(child, child_batch)
    actual.square().sum().backward()
    assert torch.equal(actual, expected)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert torch.count_nonzero(child.sidecar_projections["events"].grad) == 0
    child_parameters = dict(child.named_parameters())
    for name, parameter in parent.named_parameters():
        assert torch.equal(child_parameters[name].grad, parameter.grad), name


def test_invalid_names_receive_no_residual_or_sidecar_gradient_from_valid_peers():
    parent, child, batch = _models_and_batch()
    parent.eval()
    child.eval()
    with torch.no_grad():
        child.sidecar_projections["events"].fill_(0.025)
    valid = torch.zeros(2, 3, 2, dtype=torch.bool)
    valid[:, 0, 0] = True
    values = torch.randn(2, 3, 2, requires_grad=True)
    extended = dict(
        batch,
        sidecar_events_values=values,
        sidecar_events_valid=valid,
        sidecar_events_age_sessions=torch.where(valid, 0.0, -1.0),
    )
    expected = _model_forward(parent, batch)
    actual = _model_forward(child, extended)
    assert torch.equal(actual[:, 1:], expected[:, 1:])
    assert not torch.equal(actual[:, 0], expected[:, 0])
    actual[:, 1:].sum().backward(retain_graph=True)
    assert torch.count_nonzero(child.sidecar_projections["events"].grad) == 0
    child.zero_grad(set_to_none=True)
    actual[:, 0].sum().backward()
    assert torch.count_nonzero(child.sidecar_projections["events"].grad) > 0
    assert torch.count_nonzero(values.grad[~valid]) == 0


def test_zero_projection_accepts_valid_information_without_changing_initial_parent():
    parent, child, batch = _models_and_batch()
    parent.eval()
    child.eval()
    extended = dict(
        batch,
        sidecar_events_values=torch.ones(2, 3, 2),
        sidecar_events_valid=torch.ones(2, 3, 2, dtype=torch.bool),
        sidecar_events_age_sessions=torch.zeros(2, 3, 2),
    )
    expected = _model_forward(parent, batch)
    actual = _model_forward(child, extended)
    assert torch.equal(actual, expected)
    actual.square().sum().backward()
    assert torch.count_nonzero(child.sidecar_projections["events"].grad) > 0
