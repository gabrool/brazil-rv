from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from brazil_rv.modeling.contract import CONTEXT_COUNT, TCNArchitecture
from brazil_rv.modeling.model import SharedCausalTCN
from brazil_rv.preprocessing.contract import (
    DECISION_EQUITY_INDICES,
    EQUITY_SESSION_MINUTES,
    MIN_ACTIVE_EQUITIES,
)
from brazil_rv.preprocessing.to_close_targets import (
    center_to_close_cross_section,
    exact_to_close_returns,
    mutation_audit,
)


def _architecture(output_horizons: int) -> TCNArchitecture:
    return TCNArchitecture(
        patch_input_width=2,
        width=8,
        swiglu_hidden_width=4,
        residual_blocks=1,
        dilations=(1,),
        slow_width=2,
        fusion_width=16,
        dropout=0.0,
        output_horizons=output_horizons,
    )


def _inputs(equity_count: int, state_position: int = 15) -> tuple[torch.Tensor, ...]:
    instruments = equity_count + CONTEXT_COUNT
    return (
        torch.randn(1, instruments, 69, 2),
        torch.ones(1, instruments, 69, dtype=torch.bool),
        torch.ones(1, instruments, dtype=torch.bool),
        torch.randn(1, instruments, 2),
        torch.tensor([state_position]),
    )


def test_to_close_target_uses_exact_entry_and_final_mark() -> None:
    grid = np.zeros((1, EQUITY_SESSION_MINUTES, 5), dtype=np.float64)
    observed = np.zeros(grid.shape[:2], dtype=bool)
    entry = DECISION_EQUITY_INDICES[0]
    grid[0, entry, 0] = 100.0
    grid[0, -1, 3] = 103.0
    observed[0, [entry, EQUITY_SESSION_MINUTES - 1]] = True

    values, mask = exact_to_close_returns(grid, observed)

    assert mask[0, 0]
    np.testing.assert_allclose(values[0, 0], np.log(103.0 / 100.0))
    assert all(mutation_audit().values())


def test_to_close_target_centers_and_scales_each_decision() -> None:
    count = MIN_ACTIVE_EQUITIES + 3
    raw = np.tile(np.arange(count, dtype=np.float32)[:, None], (1, 55))
    mask = np.ones_like(raw, dtype=bool)
    sigma = np.linspace(0.5, 1.5, count)

    masked, targets, label_mask, medians = center_to_close_cross_section(
        raw, mask, sigma
    )

    assert label_mask.all()
    np.testing.assert_array_equal(masked, raw)
    np.testing.assert_allclose(targets.mean(axis=0), 0.0, atol=1e-7)
    np.testing.assert_allclose(medians, np.median(raw, axis=0))


def test_zero_initialized_to_close_head_preserves_parent_and_rng() -> None:
    torch.manual_seed(55)
    parent = SharedCausalTCN(architecture=_architecture(3), equity_count=4)
    parent_rng = torch.get_rng_state().clone()
    torch.manual_seed(55)
    candidate = SharedCausalTCN(
        architecture=replace(_architecture(3), output_horizons=4),
        equity_count=4,
        to_close_head=True,
    )
    candidate_rng = torch.get_rng_state().clone()
    assert torch.equal(parent_rng, candidate_rng)
    for name, value in parent.state_dict().items():
        torch.testing.assert_close(value, candidate.state_dict()[name], rtol=0, atol=0)

    torch.manual_seed(9)
    inputs = _inputs(4)
    parent.eval()
    candidate.eval()
    parent_scores = parent(*inputs)
    candidate_scores = candidate(*inputs)
    torch.testing.assert_close(candidate_scores[..., :3], parent_scores, rtol=0, atol=0)
    torch.testing.assert_close(
        candidate_scores[..., 3], torch.zeros_like(candidate_scores[..., 3])
    )


def test_to_close_basis_uses_remaining_session_minutes() -> None:
    model = SharedCausalTCN(
        architecture=_architecture(4), equity_count=4, to_close_head=True
    )
    assert model.to_close_readouts is not None
    with torch.no_grad():
        model.to_close_readouts.weight.zero_()
        model.to_close_readouts.bias.fill_(1.0)
    model.eval()
    torch.manual_seed(4)
    early = model(*_inputs(4, 15))[..., 3]
    torch.manual_seed(4)
    late = model(*_inputs(4, 69))[..., 3]
    early_h = 390.0 / 405.0
    late_h = 120.0 / 405.0
    torch.testing.assert_close(
        early, torch.full_like(early, 1.0 + early_h + np.sqrt(early_h))
    )
    torch.testing.assert_close(
        late, torch.full_like(late, 1.0 + late_h + np.sqrt(late_h))
    )
