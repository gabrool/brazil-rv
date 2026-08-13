from __future__ import annotations

import torch

from brazil_rv.modeling.baselines import SharedCausalTCN, apply_context_film
from brazil_rv.modeling.context_routing import align_macro_histories
from brazil_rv.modeling.contract import (
    ABSOLUTE_PATCH_COUNT,
    CONTEXT_COUNT,
    PATCH_INPUT_WIDTH,
    PEER_STATE_WIDTH,
    SLOW_FEATURE_COUNT,
    TCNSettings,
    resolve_tcn_architecture,
)


def _inputs(equities: int = 4) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    instruments = equities + CONTEXT_COUNT
    patches = (
        torch.randn(
            1, instruments, ABSOLUTE_PATCH_COUNT, PATCH_INPUT_WIDTH, generator=generator
        )
        * 0.01
    )
    history = torch.ones(1, instruments, ABSOLUTE_PATCH_COUNT, dtype=torch.bool)
    mask = torch.ones(1, instruments, dtype=torch.bool)
    return {
        "patches": patches,
        "history_patch_mask": history,
        "instrument_mask": mask,
        "slow_features": torch.randn(
            1, instruments, SLOW_FEATURE_COUNT, generator=generator
        )
        * 0.01,
        "state_position": torch.tensor([32]),
        "peer_state": torch.randn(1, equities, PEER_STATE_WIDTH, generator=generator),
    }


def _model(slow: str = "late_only", macro: str = "late_only") -> SharedCausalTCN:
    architecture = resolve_tcn_architecture(
        TCNSettings(slow_routing=slow, macro_temporal_routing=macro)
    )
    return SharedCausalTCN(architecture, "selected", equity_count=4)


def test_current_incumbent_forward_is_finite_and_deterministic() -> None:
    torch.manual_seed(29)
    model = _model().eval()
    inputs = _inputs()
    first = model(**inputs)
    second = model(**inputs)
    assert first.shape == (1, 4, 3)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert model.routing is None


def test_isolated_slow_film_is_zero_initialized_and_incumbent_neutral() -> None:
    torch.manual_seed(29)
    incumbent = _model().eval()
    torch.manual_seed(29)
    film = _model("film").eval()
    common = {
        name: value
        for name, value in film.state_dict().items()
        if not name.startswith("routing.")
    }
    for name, value in incumbent.state_dict().items():
        torch.testing.assert_close(common[name], value, atol=0, rtol=0)
    routing = film.routing
    assert routing is not None
    assert routing.slow_film is not None
    assert routing.macro_film is None
    assert routing.slow_early_input_adapter is None
    for layer in routing.slow_film:
        assert not layer.weight.any()
    inputs = _inputs()
    torch.testing.assert_close(incumbent(**inputs), film(**inputs), atol=0, rtol=0)


def test_every_current_routing_mode_builds_without_historical_scaffold() -> None:
    for route in ("early_concat", "film", "early_concat_film"):
        slow = _model(route, "late_only")
        macro = _model("late_only", route)
        assert slow.routing is not None
        assert macro.routing is not None
    assert "context_routing_experiment" not in TCNSettings.__dataclass_fields__


def test_film_combines_gamma_before_tanh() -> None:
    hidden = torch.tensor([[[1.0, 2.0]]])
    slow_gamma = torch.tensor([[[0.3, -0.2]]])
    macro_gamma = torch.tensor([[[-0.1, 0.4]]])
    beta = torch.tensor([[[0.5, -0.25]]])
    expected = (1 + torch.tanh(slow_gamma + macro_gamma)) * hidden + beta
    torch.testing.assert_close(
        apply_context_film(hidden, slow_gamma + macro_gamma, beta),
        expected,
        atol=0,
        rtol=0,
    )


def test_macro_history_alignment_is_causal() -> None:
    local = torch.arange(6 * 69, dtype=torch.float32).reshape(1, 6, 69, 1)
    global_ = (1_000 + torch.arange(2 * 69, dtype=torch.float32)).reshape(1, 2, 69, 1)
    state = torch.tensor([20])
    local_mask = torch.ones(1, 6, 69, dtype=torch.bool)
    global_mask = torch.ones(1, 2, 69, dtype=torch.bool)
    values, valid = align_macro_histories(
        local, global_, state, local_mask, global_mask
    )
    assert valid[:, :, :20].all()
    assert not valid[:, :, 20:].any()
    assert values[0, 6, 0, 0] == global_[0, 0, 49, 0]
    assert not values[:, :, 20:].any()
