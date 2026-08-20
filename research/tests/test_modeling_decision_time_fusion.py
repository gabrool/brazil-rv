from __future__ import annotations

import torch

from brazil_rv.modeling.contract import CONTEXT_COUNT, TCNArchitecture
from brazil_rv.modeling.engine import soft_spearman_loss
from brazil_rv.modeling.model import (
    DECISION_TIME_FUSION_VARIANT,
    SharedCausalTCN,
)


def _architecture() -> TCNArchitecture:
    return TCNArchitecture(
        patch_input_width=10,
        width=8,
        swiglu_hidden_width=4,
        residual_blocks=6,
        kernel_size=3,
        dilations=(1, 2, 4, 8, 16, 32),
        slow_width=32,
        fusion_states=CONTEXT_COUNT + 3,
        fusion_width=16,
        dropout=0.0,
        output_horizons=3,
    )


def _inputs() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(5)
    instruments = 4 + CONTEXT_COUNT
    patches = torch.randn(2, instruments, 69, 10, generator=generator)
    history = torch.zeros(2, instruments, 69, dtype=torch.bool)
    history[:, :, :24] = True
    history[:, 4 + 7 :, :] = True
    instrument_mask = torch.ones(2, instruments, dtype=torch.bool)
    instrument_mask[:, 3] = False
    slow = torch.randn(2, instruments, 32, generator=generator)
    positions = torch.tensor([20, 21])
    return patches, history, instrument_mask, slow, positions


def test_decision_time_fusion_starts_as_parent_without_shifting_rng() -> None:
    architecture = _architecture()
    torch.manual_seed(17)
    parent = SharedCausalTCN(architecture=architecture, equity_count=4).eval()
    parent_rng = torch.random.get_rng_state()
    torch.manual_seed(17)
    candidate = SharedCausalTCN(
        architecture=architecture,
        equity_count=4,
        variant=DECISION_TIME_FUSION_VARIANT,
    ).eval()
    candidate_rng = torch.random.get_rng_state()
    for name, value in parent.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[name], value, rtol=0, atol=0)
    assert torch.count_nonzero(candidate.decision_time_embedding[0].weight)
    assert not torch.count_nonzero(candidate.decision_time_fusion_adapter.weight)
    torch.testing.assert_close(
        candidate(*_inputs()), parent(*_inputs()), rtol=0, atol=0
    )
    torch.testing.assert_close(candidate_rng, parent_rng, rtol=0, atol=0)


def test_decision_time_fusion_path_wakes_within_ten_rank_steps() -> None:
    torch.manual_seed(19)
    model = SharedCausalTCN(
        architecture=_architecture(),
        equity_count=4,
        variant=DECISION_TIME_FUSION_VARIANT,
    ).eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("decision_time_"))
    embedding_at_start = model.decision_time_embedding[0].weight.detach().clone()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.02,
        weight_decay=0.0,
    )
    inputs = _inputs()
    rank = torch.linspace(-1.0, 1.0, 4)[None, :, None]
    targets = rank.expand(2, -1, 3).clone()
    targets[1] *= -1
    label_mask = inputs[2][:, :4, None].expand(-1, -1, 3)
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        loss = soft_spearman_loss(model(*inputs), targets, label_mask)
        loss.backward()
        optimizer.step()
    assert torch.linalg.vector_norm(model.decision_time_fusion_adapter.weight) > 0
    assert not torch.equal(
        model.decision_time_embedding[0].weight,
        embedding_at_start,
    )
