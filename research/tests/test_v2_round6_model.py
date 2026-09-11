from dataclasses import replace

import torch

from brazil_rv.v2.model import DailyMultiHorizonModel
from brazil_rv.v2.train import (
    _model_forward,
    _train_parser,
    build_optimizer,
    stage_p_model_config,
)
from test_v2_residual_sidecars import _models_and_batch


def test_mlp_uses_only_latest_permitted_slow_row_and_masks():
    parent, _, batch = _models_and_batch()
    model = DailyMultiHorizonModel(replace(parent.config, slow_encoder_kind="mlp"))
    model.eval()
    assert not any(isinstance(module, torch.nn.GRU) for module in model.modules())
    expected = _model_forward(model, batch)
    changed = {key: value.clone() for key, value in batch.items()}
    changed["slow_features"][..., :-1, :] = 10000
    changed["slow_feature_age_sessions"][..., :-1, :] = 10000
    changed["slow_feature_mask"][..., :-1, :] = False
    assert torch.equal(_model_forward(model, changed), expected)
    changed["slow_features"][..., -1, 0] += 10
    assert not torch.equal(_model_forward(model, changed), expected)


def test_mlp_has_no_gradient_from_older_rows_or_invalid_latest_payloads():
    parent, _, batch = _models_and_batch()
    model = DailyMultiHorizonModel(replace(parent.config, slow_encoder_kind="mlp"))
    model.eval()
    values = batch["slow_features"].clone().requires_grad_()
    batch["slow_features"] = values
    batch["slow_feature_mask"][..., -1, 0] = False
    _model_forward(model, batch).square().sum().backward()
    assert torch.count_nonzero(values.grad[..., :-1, :]) == 0
    assert torch.count_nonzero(values.grad[..., -1, 0]) == 0
    assert torch.count_nonzero(values.grad[..., -1, 1]) > 0


def test_multiplier_changes_only_transferred_parameters_and_p_is_uniform():
    parent, _, _ = _models_and_batch()
    first_name = next(iter(dict(parent.named_parameters())))
    parent.pretrained_parameter_names = frozenset([first_name])
    low = build_optimizer(parent, pretrained_lr_multiplier=0.3)
    high = build_optimizer(parent, pretrained_lr_multiplier=1.0)
    for a, b in zip(low.param_groups, high.param_groups, strict=True):
        assert a["params"] == b["params"]
        assert b["lr"] == 0.0003
        assert a["lr"] == (0.0003 * 0.3 if a["pretrained"] else 0.0003)
    assert (
        stage_p_model_config(
            replace(parent.config, time_decay_half_life_sessions=756.0)
        ).time_decay_half_life_sessions
        is None
    )
    options = {action.dest for action in _train_parser()._actions}
    assert {
        "slow_encoder_kind",
        "pretrained_lr_multiplier",
        "time_decay_half_life",
    } <= options
