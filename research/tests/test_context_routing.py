from __future__ import annotations

from datetime import datetime, timezone
from itertools import product

import torch

from brazil_rv.modeling import train
from brazil_rv.modeling.context_routing import align_macro_histories
from brazil_rv.modeling.contract import (
    ABSOLUTE_PATCH_COUNT,
    CONTEXT_ROUTING_MODES,
    EQUITY_COUNT,
    INSTRUMENT_COUNT,
    LOCAL_CONTEXT_COUNT,
    PATCH_INPUT_WIDTH,
    PEER_STATE_WIDTH,
    SLOW_FEATURE_COUNT,
    TCNSettings,
    context_routing_metadata,
    context_routing_parameter_count,
    expected_trainable_parameter_count,
    resolve_tcn_architecture,
)
from brazil_rv.modeling.model import build_neural_model, count_trainable_parameters


def _settings(
    slow: str = "late_only",
    macro: str = "late_only",
    experiment: str = "legacy",
) -> TCNSettings:
    return TCNSettings("context_pooled", 64, "full", "swiglu", slow, macro, experiment)


def _inputs() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    patches = 0.01 * torch.randn(
        1,
        INSTRUMENT_COUNT,
        ABSOLUTE_PATCH_COUNT,
        PATCH_INPUT_WIDTH,
        generator=generator,
    )
    history = torch.zeros(1, INSTRUMENT_COUNT, ABSOLUTE_PATCH_COUNT, dtype=torch.bool)
    history[:, :4, 12:15] = True
    history[:, EQUITY_COUNT : EQUITY_COUNT + LOCAL_CONTEXT_COUNT, :15] = True
    history[:, EQUITY_COUNT + LOCAL_CONTEXT_COUNT :] = True
    instrument = torch.zeros(1, INSTRUMENT_COUNT, dtype=torch.bool)
    instrument[:, :4] = True
    instrument[:, EQUITY_COUNT:] = True
    slow = 0.01 * torch.randn(
        1, INSTRUMENT_COUNT, SLOW_FEATURE_COUNT, generator=generator
    )
    peer = 0.01 * torch.randn(1, EQUITY_COUNT, PEER_STATE_WIDTH, generator=generator)
    return (
        {
            "patches": patches,
            "history_patch_mask": history,
            "instrument_mask": instrument,
            "slow_features": slow,
            "state_position": torch.tensor([15]),
        },
        peer,
    )


def _forward(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    peer: torch.Tensor,
) -> torch.Tensor:
    return model(
        inputs["patches"],
        inputs["history_patch_mask"],
        inputs["instrument_mask"],
        inputs["slow_features"],
        inputs["state_position"],
        peer,
    )


def test_macro_alignment_is_causal_right_aligned_and_mask_preserving() -> None:
    states = torch.tensor([0, 1, 15, ABSOLUTE_PATCH_COUNT])
    batch = states.numel()
    position = torch.arange(ABSOLUTE_PATCH_COUNT, dtype=torch.float32)
    local = torch.stack(
        [
            torch.stack([1000 * row + 100 * source + position for source in range(6)])
            for row in range(batch)
        ]
    )[..., None]
    global_values = torch.stack(
        [
            torch.stack(
                [10000 + 1000 * row + 100 * source + position for source in range(2)]
            )
            for row in range(batch)
        ]
    )[..., None]
    local_mask = torch.ones(batch, 6, ABSOLUTE_PATCH_COUNT, dtype=torch.bool)
    global_mask = torch.ones(batch, 2, ABSOLUTE_PATCH_COUNT, dtype=torch.bool)
    local_mask[2, 1] = False
    global_mask[2, 0, 68] = False

    aligned, valid = align_macro_histories(
        local, global_values, states, local_mask, global_mask
    )

    assert aligned.shape == (batch, 8, ABSOLUTE_PATCH_COUNT, 1)
    for row, state in enumerate(states.tolist()):
        assert not valid[row, :, state:].any()
        assert not aligned[row, :, state:].any()
        if state == 0:
            assert not valid[row].any()
            continue
        torch.testing.assert_close(aligned[row, 0, :state, 0], local[row, 0, :state, 0])
        source_positions = torch.arange(69 - state, 69)
        torch.testing.assert_close(
            aligned[row, 6, :state, 0],
            global_values[row, 0, source_positions, 0]
            * valid[row, 6, :state].to(global_values.dtype),
        )
        assert aligned[row, 7, state - 1, 0] == global_values[row, 1, 68, 0]
    assert not valid[2, 1].any()
    assert not valid[2, 6, 14]
    assert aligned[2, 6, 14, 0] == 0


def test_factorial_scaffold_preserves_initialization_predictions_and_counts() -> None:
    legacy_architecture = resolve_tcn_architecture(_settings())
    torch.manual_seed(123)
    legacy = build_neural_model("tcn", legacy_architecture, "selected").eval()
    legacy_rng = torch.get_rng_state().clone()
    incumbent = {name: value.clone() for name, value in legacy.state_dict().items()}
    inputs, peer = _inputs()
    with torch.no_grad():
        expected = _forward(legacy, inputs, peer)

    routing_reference: dict[str, torch.Tensor] | None = None
    counts = set()
    structures = set()
    for slow, macro in product(CONTEXT_ROUTING_MODES, repeat=2):
        architecture = resolve_tcn_architecture(_settings(slow, macro, "factorial_v1"))
        torch.manual_seed(123)
        model = build_neural_model("tcn", architecture, "selected").eval()
        assert torch.equal(torch.get_rng_state(), legacy_rng)
        state = model.state_dict()
        for name, expected_tensor in incumbent.items():
            assert torch.equal(state[name], expected_tensor), name
        routing = {
            name: value for name, value in state.items() if name.startswith("routing.")
        }
        if routing_reference is None:
            routing_reference = {name: value.clone() for name, value in routing.items()}
        else:
            assert routing.keys() == routing_reference.keys()
            assert all(
                torch.equal(value, routing_reference[name])
                for name, value in routing.items()
            )
        structures.add(tuple(state))
        counts.add(count_trainable_parameters(model))
        assert count_trainable_parameters(model) == expected_trainable_parameter_count(
            "tcn", architecture, "selected"
        )
        with torch.no_grad():
            assert torch.equal(_forward(model, inputs, peer), expected)

    assert len(structures) == 1
    assert counts == {518_659}
    assert context_routing_parameter_count(architecture) == 240_640
    assert routing_reference is not None
    assert routing_reference


def test_all_off_scaffold_has_no_gradient_or_adamw_effect() -> None:
    legacy_architecture = resolve_tcn_architecture(_settings())
    factorial_architecture = resolve_tcn_architecture(
        _settings("late_only", "late_only", "factorial_v1")
    )
    torch.manual_seed(9)
    legacy = build_neural_model("tcn", legacy_architecture, "selected").eval()
    torch.manual_seed(9)
    factorial = build_neural_model("tcn", factorial_architecture, "selected").eval()
    inputs, peer = _inputs()
    optimizers = (
        torch.optim.AdamW(legacy.parameters(), lr=1e-4),
        torch.optim.AdamW(factorial.parameters(), lr=1e-4),
    )
    gradient_norms = []
    for model, optimizer in zip((legacy, factorial), optimizers, strict=True):
        optimizer.zero_grad()
        predictions = _forward(model, inputs, peer)
        predictions[:, :4].square().mean().backward()
        squared_norm = torch.zeros((), dtype=torch.float64)
        for parameter in model.parameters():
            if parameter.grad is not None:
                squared_norm += parameter.grad.detach().double().square().sum()
        gradient_norms.append(squared_norm.sqrt())
        optimizer.step()

    assert factorial.routing is not None
    assert all(parameter.grad is None for parameter in factorial.routing.parameters())
    assert torch.equal(gradient_norms[0], gradient_norms[1])
    for name, value in legacy.state_dict().items():
        assert torch.equal(factorial.state_dict()[name], value), name


def test_routing_metadata_cli_and_run_name_are_explicit() -> None:
    settings = _settings("early_concat_film", "film", "factorial_v1")
    architecture = resolve_tcn_architecture(settings)
    metadata = context_routing_metadata(architecture)
    assert metadata["ordered_source_symbols"] == [
        "WDO$",
        "DI1F27",
        "DI1F28",
        "DI1F29",
        "DI1F31",
        "DI1$N",
        "ZT.v.0",
        "ZN.v.0",
    ]
    assert metadata["macro_patch_source_width"] == 80
    assert metadata["enabled_routes"] == {
        "slow_early_concat": True,
        "slow_film": True,
        "macro_temporal_early_concat": False,
        "macro_temporal_film": True,
    }
    name = train._run_directory_name(
        "tcn",
        settings,
        "sam_adamw",
        "soft_spearman",
        0.50,
        0.125,
        "enabled",
        29,
        datetime(2026, 8, 13, tzinfo=timezone.utc),
        "drop_win_and_global_non_rates",
        "none",
        "selected",
    )
    assert "routing-factorial_v1_slow-early_concat_film_macro-film" in name
