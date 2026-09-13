import copy

import pytest
import torch
from torch import nn

from brazil_rv.v2.round7_training import (
    optimizer_step,
    recipe_optimizer,
    sam_perturbations,
)


def test_asam_uses_squared_metric_and_keeps_zero_weights_live():
    model = nn.Linear(3, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.0, 0.2, -0.6]]))
        model.bias.zero_()
    model.weight.grad = torch.tensor([[1.0, -2.0, 3.0]])
    model.bias.grad = torch.tensor([0.5])
    optimizer = recipe_optimizer(model, cuda=False)
    used, perturbation, norm = sam_perturbations(optimizer, 0.5, adaptive=True)
    metric = torch.tensor([0.01, 0.21, 0.61, 1.0])
    gradient = torch.tensor([1.0, -2.0, 3.0, 0.5])
    expected = 0.5 * metric.square() * gradient / (metric * gradient).norm()
    actual = torch.cat([p.flatten() for p in perturbation])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close((actual / metric).norm(), torch.tensor(0.5))
    assert actual[0] != 0
    for parameter in used:
        parameter.grad.zero_()
    _, zero, _ = sam_perturbations(optimizer, 0.5, adaptive=True)
    assert all(torch.isfinite(v).all() and not v.any() for v in zero)


def test_optimizer_routes_gru_biases_norms_and_transferred_parameters():
    model = nn.Sequential(nn.GRU(4, 4), nn.LayerNorm(4), nn.Linear(4, 1))
    transferred = {"0.weight_ih_l0", "0.bias_ih_l0"}
    optimizer = recipe_optimizer(
        model,
        cuda=False,
        learning_rate=1e-4,
        transferred=transferred,
        transferred_multiplier=0.3,
    )
    groups = {id(p): group for group in optimizer.param_groups for p in group["params"]}
    for name, parameter in model.named_parameters():
        group = groups[id(parameter)]
        excluded = "bias" in name or name.startswith("1.")
        assert group["weight_decay"] == (0 if excluded else 0.01)
        assert group["adaptive"] is not excluded
        assert group["lr"] == pytest.approx(3e-5 if name in transferred else 1e-4)


def test_unexposed_family_encoders_are_not_lr_suppressed():
    from types import SimpleNamespace
    import numpy as np
    from brazil_rv.v2.characteristic_model import (
        CharacteristicConfig,
        CharacteristicModel,
    )
    from brazil_rv.v2.round7_training import (
        unexposed_families,
        transferred_parameter_names,
    )

    class Store:
        def read(self, name, rows):
            if name == "active":
                return np.ones((len(rows), 2), bool)
            # Known ages count as exposure even without a usable value.
            return np.full((len(rows), 2, 1), 2 if "events" in name else -1)

    data = SimpleNamespace(store=Store(), date_indices=np.arange(3))
    preparation = SimpleNamespace(
        families={n: SimpleNamespace(support=(0,)) for n in ("lending", "events")}
    )
    cold = unexposed_families(data, preparation)
    assert cold == ["lending"]
    model = CharacteristicModel(
        CharacteristicConfig(family_counts=(("lending", 1), ("events", 1)))
    )
    transferred = transferred_parameter_names(model, cold)
    optimizer = recipe_optimizer(
        model,
        cuda=False,
        learning_rate=1e-4,
        transferred=transferred,
        transferred_multiplier=0.3,
    )
    groups = {id(p): g for g in optimizer.param_groups for p in g["params"]}
    assert all(
        groups[id(p)]["lr"] == 1e-4 for p in model.families["lending"].parameters()
    )
    assert all(
        groups[id(p)]["lr"] == pytest.approx(3e-5)
        for p in model.families["events"].parameters()
    )


@pytest.mark.parametrize("adaptive", [False, True])
def test_probe_restores_state_optimizer_and_rng(adaptive):
    from brazil_rv.v2.characteristic_model import (
        CharacteristicConfig,
        CharacteristicModel,
    )
    from brazil_rv.v2.training_diagnostics import probe

    torch.manual_seed(18)
    model = CharacteristicModel(
        CharacteristicConfig(
            hidden_width=8,
            width=16,
            inner_width=16,
            blocks=1,
            lookback=60,
            peer_timing="early",
            temporal_encoder="attention",
        )
    )
    optimizer = recipe_optimizer(model, cuda=False)
    batch = {
        "slow_features": torch.randn(2, 24, 60, 32),
        "slow_feature_mask": torch.ones(2, 24, 60, 32, dtype=torch.bool),
        "slow_history_mask": torch.ones(2, 24, 60, dtype=torch.bool),
        "slow_feature_age_sessions": torch.zeros(2, 24, 60, 32),
        "active_mask": torch.ones(2, 24, dtype=torch.bool),
        "targets": torch.rand(2, 24, 5),
        "target_mask": torch.ones(2, 24, 5, dtype=torch.bool),
    }
    # Populate real Adam moments before checking that the probe leaves them intact.
    from brazil_rv.v2.round7_training import TrainingObjective

    objective = TrainingObjective(
        model,
        characteristic=True,
        head_indices=[2, 3, 4],
        loss_kind="soft_spearman",
        cuda=False,
    )
    optimizer_step(model, optimizer, lambda: objective(batch), 0.125, adaptive=adaptive)
    state, original_optimizer = (
        copy.deepcopy(model.state_dict()),
        copy.deepcopy(optimizer.state_dict()),
    )
    rng = torch.get_rng_state().clone()
    result = probe(
        model,
        optimizer,
        batch,
        characteristic=True,
        horizons=(3, 5, 10),
        rho=0.125,
        adaptive=adaptive,
        eta=0.01,
    )
    assert "temporal_peer" in result["modules"] and result["peer_bypass"]
    assert result["sam_score_movement_over_clean_std"] > 0
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    for name, value in state.items():
        torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
    for key, value in original_optimizer["state"].items():
        for field, tensor in value.items():
            torch.testing.assert_close(
                optimizer.state_dict()["state"][key][field], tensor, atol=0, rtol=0
            )


def test_asam_exact_restore_when_second_pass_fails():
    model = nn.Linear(3, 1)
    optimizer = recipe_optimizer(model, cuda=False)
    state = copy.deepcopy(model.state_dict())
    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second pass")
        return model(torch.ones(2, 3)).square().mean()

    with pytest.raises(RuntimeError, match="second pass"):
        optimizer_step(model, optimizer, closure, 0.5, adaptive=True)
    for name, value in state.items():
        torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
