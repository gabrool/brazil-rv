from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .config import ExecutionConfig
from .constraints import project_weights_bounded
from .features import (
    DEFAULT_POLICY_HORIZON_NAMES,
    PolicyState,
    policy_state_feature_names,
    policy_state_feature_width,
    policy_state_schema_sha256,
)


class BandPolicy(nn.Module):
    """Cross-sectional rank policy with refresh-time no-trade bands."""

    projection_mode = "exact"
    requires_policy_state = False

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
        cap_weights: Tensor | None = None,
        full_spread: Tensor | None = None,
        policy_state: PolicyState | None = None,
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

        del cap_weights, full_spread, policy_state
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


class ConcentratedPolicy(BandPolicy):
    """Equal-weight rank-tail policy with deterministic cap completion."""

    def __init__(self, config: ExecutionConfig) -> None:
        if config.concentration_k is None:
            raise ValueError("ConcentratedPolicy requires concentration_k")
        super().__init__(config)
        self.last_selection_extended_count: Tensor | None = None

    def _selection(
        self,
        blended: Tensor,
        tradeable: Tensor,
        cap_weights: Tensor,
        previous_target: Tensor,
        refresh_day: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        days, names = blended.shape
        positive = torch.zeros_like(tradeable)
        negative = torch.zeros_like(tradeable)
        extended = torch.zeros(days, dtype=torch.int64, device=blended.device)
        valid = torch.zeros(days, dtype=torch.bool, device=blended.device)
        k = int(self.config.concentration_k or 0)
        side_target = self.config.gross_target / 2.0
        capacity_tolerance = 32 * torch.finfo(blended.dtype).eps * max(1.0, side_target)

        for day in torch.nonzero(refresh_day, as_tuple=False).flatten().tolist():
            eligible = torch.nonzero(tradeable[day], as_tuple=False).flatten()
            if eligible.numel() < 2 * k:
                continue
            values = blended[day, eligible]
            ascending = eligible[torch.argsort(values, stable=True)]
            descending = eligible[torch.argsort(values, descending=True, stable=True)]
            pos_order = descending.tolist()
            neg_order = ascending.tolist()
            pos_names = pos_order[:k]
            neg_names = neg_order[:k]
            if set(pos_names) & set(neg_names):
                continue
            used = set(pos_names) | set(neg_names)
            pos_capacity = float(cap_weights[day, pos_names].sum())
            neg_capacity = float(cap_weights[day, neg_names].sum())
            pos_cursor = k
            neg_cursor = k
            extra = 0
            while (
                pos_capacity + capacity_tolerance < side_target
                or neg_capacity + capacity_tolerance < side_target
            ):
                progressed = False
                if pos_capacity + capacity_tolerance < side_target:
                    while pos_cursor < len(pos_order) and pos_order[pos_cursor] in used:
                        pos_cursor += 1
                    if pos_cursor < len(pos_order):
                        name = pos_order[pos_cursor]
                        pos_cursor += 1
                        pos_names.append(name)
                        used.add(name)
                        pos_capacity += float(cap_weights[day, name])
                        extra += 1
                        progressed = True
                if neg_capacity + capacity_tolerance < side_target:
                    while neg_cursor < len(neg_order) and neg_order[neg_cursor] in used:
                        neg_cursor += 1
                    if neg_cursor < len(neg_order):
                        name = neg_order[neg_cursor]
                        neg_cursor += 1
                        neg_names.append(name)
                        used.add(name)
                        neg_capacity += float(cap_weights[day, name])
                        extra += 1
                        progressed = True
                if not progressed:
                    break
            if (
                pos_capacity + capacity_tolerance < side_target
                or neg_capacity + capacity_tolerance < side_target
            ):
                continue

            exit_count = max(math.ceil(1.5 * k), len(pos_names), len(neg_names))
            pos_exit = set(pos_order[:exit_count])
            neg_exit = set(neg_order[:exit_count])
            pos_names.extend(
                name
                for name in torch.nonzero(previous_target[day] > 0, as_tuple=False)
                .flatten()
                .tolist()
                if name in pos_exit and name not in used
            )
            used.update(pos_names)
            neg_names.extend(
                name
                for name in torch.nonzero(previous_target[day] < 0, as_tuple=False)
                .flatten()
                .tolist()
                if name in neg_exit and name not in used
            )
            positive[day, pos_names] = True
            negative[day, neg_names] = True
            extended[day] = extra
            valid[day] = True
        return positive, negative, extended, valid

    def step(
        self,
        ranks: Tensor,
        refresh: Tensor,
        current_weights: Tensor,
        sigma: Tensor,
        previous_target: Tensor,
        initialized: Tensor,
        tradeable_mask: Tensor | None = None,
        cap_weights: Tensor | None = None,
        full_spread: Tensor | None = None,
        policy_state: PolicyState | None = None,
    ) -> tuple[Tensor, Tensor]:
        del policy_state
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
        if cap_weights is None or cap_weights.shape != shape:
            raise ValueError("ConcentratedPolicy requires aligned cap weights")
        if full_spread is None or full_spread.shape != shape:
            raise ValueError("ConcentratedPolicy requires aligned full spreads")

        blend = self.blend.to(ranks)
        relevant = tradeable[..., None] & (blend != 0)
        if (relevant & ~torch.isfinite(ranks)).any():
            raise ValueError("Tradeable ranks with nonzero blend weight must be finite")
        safe_ranks = torch.where(relevant, ranks, torch.zeros_like(ranks))
        blended = torch.sum(safe_ranks * blend, dim=-1)
        mask = tradeable.to(ranks.dtype)
        count = mask.sum(dim=-1, keepdim=True)
        centered = blended - (blended * mask).sum(
            dim=-1, keepdim=True
        ) / count.clamp_min(1)
        refreshed_day = refresh_names.any(dim=-1)
        positive, negative, extended, candidate_valid = self._selection(
            centered, tradeable, cap_weights, previous_target, refreshed_day
        )
        self.last_selection_extended_count = extended
        unit = self.config.gross_target / (2.0 * int(self.config.concentration_k or 1))
        candidate = unit * (positive.to(ranks.dtype) - negative.to(ranks.dtype))
        build = refreshed_day & ~initialized.bool() & candidate_valid
        output = previous_target * tradeable
        if self.config.band == 0.0 and self.config.cost_band_scale == 0.0:
            crosses_band = tradeable
        elif math.isinf(self.config.band):
            crosses_band = torch.zeros_like(tradeable)
        else:
            band_threshold = (
                self.config.band * sigma + self.config.cost_band_scale * full_spread
            )
            crosses_band = (candidate - current_weights).abs() > band_threshold
        update = (
            refresh_names
            & initialized.bool()[:, None]
            & candidate_valid[:, None]
            & crosses_band
        )
        output = torch.where(update, candidate, output)
        output = torch.where(build[:, None], candidate, output)
        return output, initialized.bool() | build


class NeuralPolicy(nn.Module):
    """Shared per-name policy with bounded neutral portfolio output."""

    projection_mode = "bounded"
    requires_policy_state = True

    def __init__(
        self,
        config: ExecutionConfig,
        *,
        horizon_count: int | None = None,
        horizon_names: tuple[str, ...] | None = None,
        widths: tuple[int, ...] = (64, 64),
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.config = config
        self.horizon_count = (
            len(config.horizon_blend) if horizon_count is None else horizon_count
        )
        if self.horizon_count <= 0:
            raise ValueError("NeuralPolicy requires at least one horizon")
        if self.horizon_count != len(config.horizon_blend):
            raise ValueError(
                "NeuralPolicy horizon count differs from the execution blend"
            )
        if horizon_names is None:
            if self.horizon_count != len(DEFAULT_POLICY_HORIZON_NAMES):
                raise ValueError("Noncanonical policies require ordered horizon names")
            horizon_names = DEFAULT_POLICY_HORIZON_NAMES
        policy_state_feature_names(self.horizon_count, horizon_names)
        self.horizon_names = horizon_names
        if not widths or any(width <= 0 for width in widths):
            raise ValueError("NeuralPolicy widths must be positive")
        self.widths = widths
        self.seed = int(seed)
        input_width = policy_state_feature_width(self.horizon_count)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            layers: list[nn.Module] = [nn.LayerNorm(input_width)]
            previous = input_width
            for width in widths:
                layers.extend((nn.Linear(previous, width), nn.SiLU()))
                previous = width
            self.trunk = nn.Sequential(*layers)
            self.output = nn.Linear(previous, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    @property
    def contract_metadata(self) -> dict[str, object]:
        return {
            "schema": "BRAZIL_RV_NEURAL_POLICY_V1",
            "horizon_names": list(self.horizon_names),
            "policy_state_schema_sha256": policy_state_schema_sha256(
                self.horizon_names
            ),
            "feature_names": list(
                policy_state_feature_names(self.horizon_count, self.horizon_names)
            ),
            "widths": list(self.widths),
            "seed": self.seed,
        }

    def step(
        self,
        ranks: Tensor,
        refresh: Tensor,
        current_weights: Tensor,
        sigma: Tensor,
        previous_target: Tensor,
        initialized: Tensor,
        tradeable_mask: Tensor | None = None,
        cap_weights: Tensor | None = None,
        full_spread: Tensor | None = None,
        policy_state: PolicyState | None = None,
    ) -> tuple[Tensor, Tensor]:
        del current_weights, sigma, previous_target, full_spread
        if policy_state is None:
            raise ValueError("NeuralPolicy requires PolicyState")
        if ranks.ndim != 3 or ranks.shape[-1] != self.horizon_count:
            raise ValueError("Ranks differ from the configured policy horizons")
        if tradeable_mask is None or cap_weights is None:
            raise ValueError("NeuralPolicy requires tradeability and cap weights")
        if not torch.equal(policy_state.tradeable_mask, tradeable_mask.bool()):
            raise ValueError("PolicyState tradeability differs from the step mask")
        if policy_state.cap_weights.shape != cap_weights.shape or not torch.equal(
            policy_state.cap_weights[tradeable_mask], cap_weights[tradeable_mask]
        ):
            raise ValueError("PolicyState caps differ from the step caps")
        if policy_state.horizon_names != self.horizon_names:
            raise ValueError("PolicyState ordered horizons differ from the policy")
        if policy_state.schema_sha256 != policy_state_schema_sha256(self.horizon_names):
            raise ValueError("PolicyState schema hash differs from the policy")
        features = policy_state.features()
        expected = (*ranks.shape[:2], policy_state_feature_width(self.horizon_count))
        if features.shape != expected:
            raise ValueError("PolicyState feature axes differ from ranks")
        raw = self.output(self.trunk(features)).squeeze(-1)
        target = project_weights_bounded(
            raw,
            tradeable_mask,
            cap_weights,
            self.config.gross_target,
        )
        if refresh.shape == ranks.shape[:1]:
            refreshed_day = refresh.bool() & tradeable_mask.any(dim=-1)
        elif refresh.shape == ranks.shape[:2]:
            refreshed_day = (refresh.bool() & tradeable_mask).any(dim=-1)
        else:
            raise ValueError("Refresh must be [day] or [day,name]")
        active = initialized.bool() | refreshed_day
        return torch.where(active[:, None], target, torch.zeros_like(target)), active
