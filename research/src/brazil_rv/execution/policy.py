from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .config import ExecutionConfig


class BandPolicy(nn.Module):
    """Cross-sectional rank policy with refresh-time no-trade bands."""

    def __init__(self, config: ExecutionConfig) -> None:
        super().__init__()
        self.config = config
        blend = torch.tensor(config.horizon_blend, dtype=torch.float64)
        self.register_buffer("blend", blend / blend.sum())

    def _candidate(self, ranks: Tensor, tradeable: Tensor) -> tuple[Tensor, Tensor]:
        if not ranks.is_floating_point():
            raise ValueError("Ranks must be floating point")
        if ranks.ndim != 3 or ranks.shape[-1] != self.blend.numel():
            raise ValueError("Ranks must be [day, name, configured horizon]")
        if tradeable.shape != ranks.shape[:2]:
            raise ValueError("Tradeable mask does not align with ranks")

        mask = tradeable.to(dtype=ranks.dtype)
        blend = self.blend.to(ranks)
        relevant = tradeable[..., None] & (blend != 0)
        if (relevant & ~torch.isfinite(ranks)).any():
            raise ValueError("Tradeable ranks with nonzero blend weight must be finite")
        safe_ranks = torch.where(relevant, ranks, torch.zeros_like(ranks))
        blended = torch.sum(safe_ranks * blend, dim=-1)
        count = mask.sum(dim=-1, keepdim=True)
        mean = (blended * mask).sum(dim=-1, keepdim=True) / count.clamp_min(1.0)
        centered = (blended - mean) * mask
        absolute = centered.abs().sum(dim=-1, keepdim=True)
        valid = (count.squeeze(-1) >= 2) & (absolute.squeeze(-1) > 0.0)
        target = (
            self.config.gross_target
            * centered
            / absolute.clamp_min(torch.finfo(ranks.dtype).tiny)
        )
        return torch.where(valid[:, None], target, torch.zeros_like(target)), valid

    def step(
        self,
        ranks: Tensor,
        refresh: Tensor,
        current_weights: Tensor,
        sigma: Tensor,
        previous_target: Tensor,
        initialized: Tensor,
        tradeable_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return the target for one simulator minute.

        ``refresh`` may mark an entire cross-section (``[day]``) or individual
        names (``[day, name]``). A day builds its first target at its first valid
        refresh. Later non-refreshed names retain ``previous_target``.
        """

        if ranks.ndim != 3:
            raise ValueError("Ranks must be [day, name, horizon]")
        shape = ranks.shape[:2]
        if any(
            value.shape != shape for value in (current_weights, sigma, previous_target)
        ):
            raise ValueError("Policy state does not align with rank names")
        if initialized.shape != shape[:1]:
            raise ValueError("Initialized state must have one value per day")
        if refresh.shape == shape[:1]:
            refresh_names = refresh.bool()[:, None].expand(shape)
        elif refresh.shape == shape:
            refresh_names = refresh.bool()
        else:
            raise ValueError("Refresh must be [day] or [day, name]")
        tradeable = (
            torch.ones(shape, dtype=torch.bool, device=ranks.device)
            if tradeable_mask is None
            else tradeable_mask.bool()
        )
        if tradeable.shape != shape:
            raise ValueError("Tradeable mask does not align with ranks")

        candidate, candidate_valid = self._candidate(ranks, tradeable)
        refreshed_day = refresh_names.any(dim=-1)
        build = refreshed_day & ~initialized.bool() & candidate_valid
        output = previous_target * tradeable

        if self.config.band == 0.0:
            crosses_band = tradeable
        elif math.isinf(self.config.band):
            crosses_band = torch.zeros_like(tradeable)
        else:
            crosses_band = (
                (candidate - current_weights).abs() > self.config.band * sigma
            ) & tradeable
        update = (
            refresh_names
            & initialized.bool()[:, None]
            & candidate_valid[:, None]
            & crosses_band
        )
        output = torch.where(update, candidate, output)
        output = torch.where(build[:, None], candidate, output)
        return output, initialized.bool() | build
