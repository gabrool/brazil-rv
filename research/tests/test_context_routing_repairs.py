from __future__ import annotations

import torch

from brazil_rv.modeling.baselines import apply_context_film
from brazil_rv.modeling.contract import (
    ABSOLUTE_PATCH_COUNT,
    CONTEXT_ROUTING_MACRO_EARLY_SOURCE_WIDTH,
    CONTEXT_ROUTING_SOURCE_COUNT,
    EQUITY_COUNT,
    INSTRUMENT_COUNT,
    LOCAL_CONTEXT_COUNT,
    PATCH_INPUT_WIDTH,
    SLOW_FEATURE_COUNT,
    TCNSettings,
    context_routing_active_parameter_count,
    context_routing_metadata,
    context_routing_parameter_count,
    resolve_tcn_architecture,
)
from brazil_rv.modeling.model import build_neural_model
from brazil_rv.preprocessing.contract import GLOBAL_SLOW_CHANNELS


def _architecture(slow: str, macro: str):
    return resolve_tcn_architecture(
        TCNSettings(
            "context_pooled",
            64,
            "full",
            "swiglu",
            slow,
            macro,
            "factorial_v1",
        )
    )


def test_slow_condition_excludes_only_the_zt_zn_b3_close_channel() -> None:
    architecture = _architecture("early_concat", "late_only")
    torch.manual_seed(17)
    model = build_neural_model("tcn", architecture, "selected").eval()
    generator = torch.Generator().manual_seed(5)
    slow = torch.randn(1, INSTRUMENT_COUNT, SLOW_FEATURE_COUNT, generator=generator)
    instrument_mask = torch.ones(1, INSTRUMENT_COUNT, dtype=torch.bool)
    baseline = model._slow_condition(slow, instrument_mask)

    global_start = EQUITY_COUNT + LOCAL_CONTEXT_COUNT + 2
    excluded = GLOBAL_SLOW_CHANNELS.index(
        "previous_b3_close_to_decision_return_normalized"
    )
    excluded_changed = slow.clone()
    excluded_changed[:, global_start : global_start + 2, excluded] += 10_000.0
    torch.testing.assert_close(
        model._slow_condition(excluded_changed, instrument_mask),
        baseline,
        rtol=0.0,
        atol=0.0,
    )

    valid_changed = slow.clone()
    valid_changed[:, global_start : global_start + 2, 0] += 10_000.0
    assert not torch.equal(
        model._slow_condition(valid_changed, instrument_mask), baseline
    )


def test_film_uses_tanh_of_combined_gamma_exactly() -> None:
    hidden = torch.tensor([[-2.0, 3.0], [0.5, -0.25]])
    slow_gamma = torch.tensor([[0.2, -0.3], [2.0, -4.0]])
    macro_gamma = torch.tensor([[0.4, 0.5], [-1.0, 3.0]])
    slow_beta = torch.tensor([[0.1, -0.2], [0.3, 0.4]])
    macro_beta = torch.tensor([[-0.5, 0.6], [0.7, -0.8]])
    gamma_total = slow_gamma + macro_gamma
    beta_total = slow_beta + macro_beta
    expected = (1.0 + torch.tanh(gamma_total)) * hidden + beta_total
    actual = apply_context_film(hidden, gamma_total, beta_total)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert not torch.equal(actual, (1.0 + gamma_total) * hidden + beta_total)


def test_macro_early_availability_bits_make_unavailable_sources_neutral() -> None:
    architecture = _architecture("late_only", "early_concat")
    torch.manual_seed(23)
    model = build_neural_model("tcn", architecture, "selected").eval()
    generator = torch.Generator().manual_seed(9)
    patches = torch.randn(
        1,
        INSTRUMENT_COUNT,
        ABSOLUTE_PATCH_COUNT,
        PATCH_INPUT_WIDTH,
        generator=generator,
    )
    history = torch.zeros(1, INSTRUMENT_COUNT, ABSOLUTE_PATCH_COUNT, dtype=torch.bool)
    state_position = torch.tensor([15])
    unavailable = model._macro_raw_input(patches, history, state_position)
    assert unavailable.shape[-1] == (
        CONTEXT_ROUTING_SOURCE_COUNT * CONTEXT_ROUTING_MACRO_EARLY_SOURCE_WIDTH
    )
    assert torch.count_nonzero(unavailable) == 0

    changed = patches.clone()
    changed[:, EQUITY_COUNT + 1] += 100_000.0
    torch.testing.assert_close(
        model._macro_raw_input(changed, history, state_position),
        unavailable,
        rtol=0.0,
        atol=0.0,
    )
    assert model.routing is not None
    with torch.no_grad():
        model.routing.macro_early_output_projection.weight.fill_(0.25)
    torch.testing.assert_close(
        model._macro_early_input(unavailable),
        torch.zeros(1, ABSOLUTE_PATCH_COUNT, architecture.width),
        rtol=0.0,
        atol=0.0,
    )

    history[:, EQUITY_COUNT + 1, :15] = True
    available = model._macro_raw_input(patches, history, state_position)
    reshaped = available.reshape(
        1,
        ABSOLUTE_PATCH_COUNT,
        CONTEXT_ROUTING_SOURCE_COUNT,
        CONTEXT_ROUTING_MACRO_EARLY_SOURCE_WIDTH,
    )
    assert torch.equal(reshaped[0, :15, 0, -1], torch.ones(15, dtype=reshaped.dtype))
    assert torch.count_nonzero(reshaped[0, 15:, 0]) == 0


def test_active_and_total_routing_parameter_counts_are_distinct_and_recorded() -> None:
    all_off = _architecture("late_only", "late_only")
    slow_early = _architecture("early_concat", "late_only")
    macro_film = _architecture("late_only", "film")
    all_active = _architecture("early_concat_film", "early_concat_film")
    totals = {
        context_routing_parameter_count(architecture)
        for architecture in (all_off, slow_early, macro_film, all_active)
    }
    assert totals == {240_640}
    assert context_routing_active_parameter_count(all_off) == 0
    assert 0 < context_routing_active_parameter_count(slow_early) < 240_640
    assert 0 < context_routing_active_parameter_count(macro_film) < 240_640
    assert context_routing_active_parameter_count(all_active) == 240_640

    metadata = context_routing_metadata(slow_early)
    assert metadata["routing_scaffold_parameter_count"] == 240_640
    assert metadata["active_routing_parameter_count"] == (
        context_routing_active_parameter_count(slow_early)
    )
    assert metadata["active_components"] == ["slow_early_concat"]
