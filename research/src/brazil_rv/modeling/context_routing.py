from __future__ import annotations

import torch

from .contract import (
    ABSOLUTE_PATCH_COUNT,
    CONTEXT_ROUTING_GLOBAL_SOURCE_COUNT,
    CONTEXT_ROUTING_LOCAL_SOURCE_COUNT,
)


def causal_patch_mask(state_position: torch.Tensor) -> torch.Tensor:
    """Return the absolute B3 patch positions available before each decision."""
    if state_position.ndim != 1:
        raise ValueError("state_position must have shape [batch]")
    positions = torch.arange(
        ABSOLUTE_PATCH_COUNT, device=state_position.device, dtype=state_position.dtype
    )
    return positions[None, :] < state_position[:, None]


def align_macro_histories(
    local_values: torch.Tensor,
    global_values: torch.Tensor,
    state_position: torch.Tensor,
    local_mask: torch.Tensor,
    global_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align six absolute-grid local and two rolling global histories causally.

    Local source patch t maps directly to absolute target patch t. For ZT and ZN,
    source patch (69 - state_position) + t maps to target patch t. Every returned
    source is masked at and after the decision state.
    """
    if local_values.ndim < 3 or global_values.ndim != local_values.ndim:
        raise ValueError("Macro histories must have shape [batch, source, patch, ...]")
    if (
        local_values.shape[0] != global_values.shape[0]
        or local_values.shape[2:] != global_values.shape[2:]
        or local_values.shape[1] != CONTEXT_ROUTING_LOCAL_SOURCE_COUNT
        or global_values.shape[1] != CONTEXT_ROUTING_GLOBAL_SOURCE_COUNT
        or local_values.shape[2] != ABSOLUTE_PATCH_COUNT
    ):
        raise ValueError("Macro history value axes do not match the routing contract")
    if (
        local_mask.shape != local_values.shape[:3]
        or global_mask.shape != global_values.shape[:3]
        or local_mask.dtype != torch.bool
        or global_mask.dtype != torch.bool
    ):
        raise ValueError("Macro history masks do not match their value axes")
    if state_position.shape != (local_values.shape[0],):
        raise ValueError("state_position does not match the macro-history batch")

    causal = causal_patch_mask(state_position)
    target = torch.arange(
        ABSOLUTE_PATCH_COUNT,
        device=state_position.device,
        dtype=state_position.dtype,
    )
    source = (ABSOLUTE_PATCH_COUNT - state_position[:, None] + target[None, :]).clamp(
        0, ABSOLUTE_PATCH_COUNT - 1
    )
    index_shape = (
        local_values.shape[0],
        CONTEXT_ROUTING_GLOBAL_SOURCE_COUNT,
        ABSOLUTE_PATCH_COUNT,
        *local_values.shape[3:],
    )
    gather_index = source.view(
        local_values.shape[0],
        1,
        ABSOLUTE_PATCH_COUNT,
        *([1] * (local_values.ndim - 3)),
    ).expand(index_shape)
    aligned_global = torch.gather(global_values, 2, gather_index)
    aligned_global_mask = torch.gather(
        global_mask,
        2,
        source[:, None, :].expand(-1, CONTEXT_ROUTING_GLOBAL_SOURCE_COUNT, -1),
    )
    local_valid = local_mask & causal[:, None, :]
    global_valid = aligned_global_mask & causal[:, None, :]
    values = torch.cat((local_values, aligned_global), dim=1)
    valid = torch.cat((local_valid, global_valid), dim=1)
    expanded_valid = valid[(...,) + (None,) * (values.ndim - 3)]
    return values * expanded_valid.to(values.dtype), valid
