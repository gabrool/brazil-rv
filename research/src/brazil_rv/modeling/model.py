from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from .contract import (
    CONTEXT_COUNT,
    EQUITY_COUNT,
    EXPECTED_DECISIONS_PER_DATE,
    LOCAL_CONTEXT_COUNT,
    STATE_TOKEN_SLOT,
    TARGETED_FUSION_GATE_BIAS,
    TCN_ARCHITECTURE,
    TCNArchitecture,
)
from .layers import CausalTCNResidualBlock, MuonLinear

PARENT_MODEL_VARIANT = "parent"
PHASE_A_MODEL_VARIANTS = (
    "decision_time",
    "temporal_stats",
    "multi_depth_stats",
    "cross_section_max_min",
    "learned_set_pool",
    "conditional_bucket_means",
)
MODEL_VARIANTS = (PARENT_MODEL_VARIANT, *PHASE_A_MODEL_VARIANTS)
BETA_BUCKET_SLOW_INDEX = 21  # beta_to_WDO; beta_to_WIN remains neutralized.
VOLATILITY_BUCKET_SLOW_INDEX = 19  # realized_vol_cross_section_rank
BUCKET_Z_THRESHOLD = 0.5

_MODEL_VARIANT_DETAILS: dict[str, dict[str, object]] = {
    PARENT_MODEL_VARIANT: {
        "name": PARENT_MODEL_VARIANT,
        "zero_initialized_residual_adapter": False,
    },
    "decision_time": {
        "name": "decision_time",
        "inputs": "sin_cos_decision_phase",
        "period_decisions": EXPECTED_DECISIONS_PER_DATE,
        "adapter": "linear_2_to_64",
        "zero_initialized_residual_adapter": True,
    },
    "temporal_stats": {
        "name": "temporal_stats",
        "inputs": ["masked_causal_mean", "masked_causal_std"],
        "source_blocks": [6],
        "adapter": "linear_128_to_64",
        "zero_initialized_residual_adapter": True,
    },
    "multi_depth_stats": {
        "name": "multi_depth_stats",
        "inputs": ["masked_causal_mean", "masked_causal_std"],
        "source_blocks": [2, 4, 6],
        "adapter": "linear_384_to_64",
        "zero_initialized_residual_adapter": True,
    },
    "cross_section_max_min": {
        "name": "cross_section_max_min",
        "inputs": ["masked_equity_max", "masked_equity_min"],
        "injection": "residual_into_existing_mean_dispersion",
        "adapter": "linear_128_to_128",
        "zero_initialized_residual_adapter": True,
    },
    "learned_set_pool": {
        "name": "learned_set_pool",
        "phi_width": 16,
        "pool": "masked_mean",
        "injection": "residual_into_existing_mean_dispersion",
        "adapter": "linear_16_to_128",
        "zero_initialized_residual_adapter": True,
    },
    "conditional_bucket_means": {
        "name": "conditional_bucket_means",
        "conditioning": [
            "beta_to_WDO",
            "realized_vol_cross_section_rank",
        ],
        "standardization": "within_sample_active_equities",
        "bucket_z_boundaries": [-BUCKET_Z_THRESHOLD, BUCKET_Z_THRESHOLD],
        "means_per_conditioner": 3,
        "injection": "residual_into_existing_mean_dispersion",
        "adapter": "linear_384_to_128",
        "zero_initialized_residual_adapter": True,
    },
}


def model_variant_metadata(variant: str) -> dict[str, object]:
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model variant: {variant}")
    return deepcopy(_MODEL_VARIANT_DETAILS[variant])


class SharedCausalTCN(nn.Module):
    model_name = "tcn"

    def __init__(
        self,
        *,
        architecture: TCNArchitecture = TCN_ARCHITECTURE,
        equity_count: int = EQUITY_COUNT,
        variant: str = PARENT_MODEL_VARIANT,
    ) -> None:
        super().__init__()
        model_variant_metadata(variant)
        self.architecture = architecture
        self.equity_count = equity_count
        self.instrument_count = equity_count + CONTEXT_COUNT
        self.variant = variant
        width = architecture.width
        self.input_projection = nn.Linear(
            architecture.patch_input_width, width, bias=False
        )
        self.blocks = nn.ModuleList(
            [
                CausalTCNResidualBlock(
                    width,
                    architecture.kernel_size,
                    dilation,
                    architecture.dropout,
                    architecture.swiglu_hidden_width,
                )
                for dilation in architecture.dilations
            ]
        )
        self.slow_projection = nn.Linear(architecture.slow_width, width, bias=False)
        self.state_norm = nn.LayerNorm(width)
        self.fusion_input = nn.Linear(
            architecture.fusion_states * width,
            architecture.fusion_width,
            bias=True,
        )
        self.fusion_output = MuonLinear(architecture.fusion_width, width, bias=False)
        self.fusion_gate = nn.Linear(2 * width, width, bias=True)
        self.fusion_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(architecture.dropout)
        self.prediction_head = nn.Linear(width, architecture.output_horizons, bias=True)
        self.apply(self._initialize_module)
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.constant_(self.fusion_gate.bias, TARGETED_FUSION_GATE_BIAS)
        self._add_variant_modules(width)
        if hasattr(self, "set_phi"):
            self.set_phi.apply(self._initialize_module)
        for adapter in self._variant_adapters():
            nn.init.zeros_(adapter.weight)

    def _add_variant_modules(self, width: int) -> None:
        if self.variant == "decision_time":
            self.decision_time_adapter = nn.Linear(2, width, bias=False)
        elif self.variant == "temporal_stats":
            self.temporal_stats_adapter = nn.Linear(2 * width, width, bias=False)
        elif self.variant == "multi_depth_stats":
            self.multi_depth_adapter = nn.Linear(6 * width, width, bias=False)
        elif self.variant == "cross_section_max_min":
            self.cross_section_moment_adapter = nn.Linear(
                2 * width, 2 * width, bias=False
            )
        elif self.variant == "learned_set_pool":
            self.set_phi = nn.Sequential(
                nn.Linear(width, 16),
                nn.GELU(),
                nn.Linear(16, 16),
            )
            self.set_pool_adapter = nn.Linear(16, 2 * width, bias=False)
        elif self.variant == "conditional_bucket_means":
            self.conditional_pool_adapter = nn.Linear(6 * width, 2 * width, bias=False)

    def _variant_adapters(self) -> tuple[nn.Linear, ...]:
        names = (
            "decision_time_adapter",
            "temporal_stats_adapter",
            "multi_depth_adapter",
            "cross_section_moment_adapter",
            "set_pool_adapter",
            "conditional_pool_adapter",
        )
        return tuple(
            module
            for name in names
            if isinstance((module := getattr(self, name, None)), nn.Linear)
        )

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _stream_states(self, hidden: torch.Tensor, batch_size: int) -> torch.Tensor:
        return hidden.reshape(
            batch_size,
            self.instrument_count,
            self.architecture.width,
            hidden.shape[-1],
        ).permute(0, 1, 3, 2)

    def _gather_hidden_states(
        self,
        hidden: torch.Tensor,
        batch_size: int,
        state_position: torch.Tensor,
    ) -> torch.Tensor:
        streams = self._stream_states(hidden, batch_size)
        positions = state_position[:, None].expand(-1, self.instrument_count)
        global_slots = (
            torch.arange(self.instrument_count, device=hidden.device)
            >= self.equity_count + LOCAL_CONTEXT_COUNT
        )
        positions = torch.where(global_slots[None], STATE_TOKEN_SLOT, positions)
        index = (positions - 1)[..., None, None].expand(
            -1, -1, 1, self.architecture.width
        )
        return streams.gather(2, index).squeeze(2)

    def _masked_temporal_stats(
        self,
        hidden: torch.Tensor,
        history_patch_mask: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        streams = self._stream_states(hidden, batch_size)
        weight = history_patch_mask[..., None].to(streams.dtype)
        count = weight.sum(dim=2).clamp_min(1.0)
        mean = (streams * weight).sum(dim=2) / count
        second = (streams.square() * weight).sum(dim=2) / count
        standard_deviation = torch.sqrt(torch.clamp(second - mean.square(), min=1e-6))
        return torch.cat((mean, standard_deviation), dim=-1)

    def _instrument_states(
        self,
        patches: torch.Tensor,
        history_patch_mask: torch.Tensor,
        slow_features: torch.Tensor,
        state_position: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = patches.shape[0]
        masked = patches * history_patch_mask[..., None].to(patches.dtype)
        hidden = (
            self.input_projection(masked)
            .permute(0, 1, 3, 2)
            .reshape(
                batch_size * self.instrument_count,
                self.architecture.width,
                masked.shape[2],
            )
        )
        depth_stats = []
        for block_index, block in enumerate(self.blocks, start=1):
            hidden = block(hidden)
            if self.variant == "multi_depth_stats" and block_index in (2, 4, 6):
                depth_stats.append(
                    self._masked_temporal_stats(hidden, history_patch_mask, batch_size)
                )
        raw = self._gather_hidden_states(hidden, batch_size, state_position)
        if self.variant == "temporal_stats":
            raw = raw + self.temporal_stats_adapter(
                self._masked_temporal_stats(hidden, history_patch_mask, batch_size)
            )
        elif self.variant == "multi_depth_stats":
            raw = raw + self.multi_depth_adapter(torch.cat(depth_stats, dim=-1))
        states = self.state_norm(raw + self.slow_projection(slow_features))
        if self.variant == "decision_time":
            start = STATE_TOKEN_SLOT - EXPECTED_DECISIONS_PER_DATE + 1
            decision = (state_position - start).to(states.dtype)
            angle = decision * (2.0 * torch.pi / EXPECTED_DECISIONS_PER_DATE)
            phase = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
            states = states + self.decision_time_adapter(phase)[:, None]
        return states

    @staticmethod
    def _masked_cross_section_stats(
        states: torch.Tensor, equity_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = equity_mask[..., None].to(states.dtype)
        count = weight.sum(dim=1).clamp_min(1.0)
        mean = (states * weight).sum(dim=1) / count
        second = (states.square() * weight).sum(dim=1) / count
        dispersion = torch.sqrt(torch.clamp(second - mean.square(), min=1e-6))
        return mean, dispersion

    @staticmethod
    def _masked_extrema(
        states: torch.Tensor, equity_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = equity_mask[..., None]
        maximum = states.masked_fill(~valid, -torch.inf).amax(dim=1)
        minimum = states.masked_fill(~valid, torch.inf).amin(dim=1)
        has_equity = equity_mask.any(dim=1, keepdim=True)
        maximum = torch.where(has_equity, maximum, torch.zeros_like(maximum))
        minimum = torch.where(has_equity, minimum, torch.zeros_like(minimum))
        return maximum, minimum

    @staticmethod
    def _conditional_means(
        states: torch.Tensor,
        values: torch.Tensor,
        equity_mask: torch.Tensor,
    ) -> torch.Tensor:
        weight = equity_mask.to(states.dtype)
        count = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (values * weight).sum(dim=1, keepdim=True) / count
        centered = (values - mean) * weight
        variance = centered.square().sum(dim=1, keepdim=True) / count
        standardized = centered / torch.sqrt(variance + 1e-6)
        memberships = (
            standardized < -BUCKET_Z_THRESHOLD,
            (standardized >= -BUCKET_Z_THRESHOLD)
            & (standardized <= BUCKET_Z_THRESHOLD),
            standardized > BUCKET_Z_THRESHOLD,
        )
        pooled = []
        for membership in memberships:
            bucket = membership & equity_mask
            bucket_weight = bucket[..., None].to(states.dtype)
            bucket_count = bucket_weight.sum(dim=1).clamp_min(1.0)
            pooled.append((states * bucket_weight).sum(dim=1) / bucket_count)
        return torch.cat(pooled, dim=-1)

    def _fuse(
        self,
        equity_states: torch.Tensor,
        context_states: torch.Tensor,
        equity_mask: torch.Tensor,
        context_mask: torch.Tensor,
        equity_slow: torch.Tensor,
    ) -> torch.Tensor:
        context = (
            context_states * context_mask[..., None].to(context_states.dtype)
        ).reshape(context_states.shape[0], CONTEXT_COUNT * self.architecture.width)
        mean, dispersion = self._masked_cross_section_stats(equity_states, equity_mask)
        market_moments = torch.cat((mean, dispersion), dim=-1)
        if self.variant == "cross_section_max_min":
            maximum, minimum = self._masked_extrema(equity_states, equity_mask)
            market_moments = market_moments + self.cross_section_moment_adapter(
                torch.cat((maximum, minimum), dim=-1)
            )
        elif self.variant == "learned_set_pool":
            transformed = self.set_phi(equity_states)
            weight = equity_mask[..., None].to(transformed.dtype)
            pooled = (transformed * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(
                1.0
            )
            market_moments = market_moments + self.set_pool_adapter(pooled)
        elif self.variant == "conditional_bucket_means":
            beta = equity_slow[..., BETA_BUCKET_SLOW_INDEX]
            volatility = equity_slow[..., VOLATILITY_BUCKET_SLOW_INDEX]
            conditional = torch.cat(
                (
                    self._conditional_means(equity_states, beta, equity_mask),
                    self._conditional_means(equity_states, volatility, equity_mask),
                ),
                dim=-1,
            )
            market_moments = market_moments + self.conditional_pool_adapter(conditional)
        shared = torch.cat((context, market_moments), dim=-1)
        shared = shared[:, None].expand(-1, self.equity_count, -1)
        inputs = torch.cat((equity_states, shared), dim=-1)
        fused = self.fusion_output(F.gelu(self.fusion_input(inputs)))
        fused = self.dropout(F.gelu(fused))
        gate = torch.sigmoid(
            self.fusion_gate(torch.cat((equity_states, fused), dim=-1))
        )
        return self.fusion_norm(equity_states + gate * fused)

    def forward(
        self,
        patches: torch.Tensor,
        history_patch_mask: torch.Tensor,
        instrument_mask: torch.Tensor,
        slow_features: torch.Tensor,
        state_position: torch.Tensor,
    ) -> torch.Tensor:
        states = self._instrument_states(
            patches, history_patch_mask, slow_features, state_position
        )
        equity_mask = instrument_mask[:, : self.equity_count]
        equity_states = states[:, : self.equity_count]
        fused = self._fuse(
            equity_states,
            states[:, self.equity_count :],
            equity_mask,
            instrument_mask[:, self.equity_count :],
            slow_features[:, : self.equity_count],
        )
        predictions = self.prediction_head(fused)
        return predictions * equity_mask[..., None].to(predictions.dtype)


def build_model(variant: str = PARENT_MODEL_VARIANT) -> SharedCausalTCN:
    return SharedCausalTCN(variant=variant)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
