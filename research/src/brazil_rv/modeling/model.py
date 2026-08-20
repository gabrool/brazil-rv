from __future__ import annotations

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
DECISION_TIME_FUSION_VARIANT = "decision_time_fusion"
MODEL_VARIANTS = (PARENT_MODEL_VARIANT, DECISION_TIME_FUSION_VARIANT)


def model_variant_metadata(variant: str) -> dict[str, object]:
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model variant: {variant}")
    return {
        "name": variant,
        "decision_embedding_width": 16
        if variant == DECISION_TIME_FUSION_VARIANT
        else None,
        "injection": (
            "zero_started_residual_into_shared_mean_dispersion_fusion"
            if variant == DECISION_TIME_FUSION_VARIANT
            else None
        ),
        "zero_initialized_residual_adapter": variant != PARENT_MODEL_VARIANT,
        "rng_stream_preserved": variant != PARENT_MODEL_VARIANT,
    }


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
        if variant == DECISION_TIME_FUSION_VARIANT:
            with torch.random.fork_rng(devices=[]):
                self.decision_time_embedding = nn.Sequential(
                    nn.Linear(2, 16),
                    nn.GELU(),
                    nn.Linear(16, 16),
                )
                self.decision_time_fusion_adapter = nn.Linear(16, 2 * width, bias=False)
                self.decision_time_embedding.apply(self._initialize_module)
                self.decision_time_fusion_adapter.apply(self._initialize_module)
                nn.init.zeros_(self.decision_time_fusion_adapter.weight)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _gather_hidden_states(
        self,
        hidden: torch.Tensor,
        batch_size: int,
        state_position: torch.Tensor,
    ) -> torch.Tensor:
        width = self.architecture.width
        streams = hidden.reshape(
            batch_size, self.instrument_count, width, hidden.shape[-1]
        ).permute(0, 1, 3, 2)
        positions = state_position[:, None].expand(-1, self.instrument_count)
        global_slots = (
            torch.arange(self.instrument_count, device=hidden.device)
            >= self.equity_count + LOCAL_CONTEXT_COUNT
        )
        positions = torch.where(global_slots[None], STATE_TOKEN_SLOT, positions)
        index = (positions - 1)[..., None, None].expand(-1, -1, 1, width)
        return streams.gather(2, index).squeeze(2)

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
        for block in self.blocks:
            hidden = block(hidden)
        raw = self._gather_hidden_states(hidden, batch_size, state_position)
        return self.state_norm(raw + self.slow_projection(slow_features))

    @staticmethod
    def _decision_phase(state_position: torch.Tensor) -> torch.Tensor:
        start = STATE_TOKEN_SLOT - EXPECTED_DECISIONS_PER_DATE + 1
        decision = (state_position - start).to(torch.float32)
        angle = decision * (2.0 * torch.pi / EXPECTED_DECISIONS_PER_DATE)
        return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)

    def _fuse(
        self,
        equity_states: torch.Tensor,
        context_states: torch.Tensor,
        equity_mask: torch.Tensor,
        context_mask: torch.Tensor,
        state_position: torch.Tensor,
    ) -> torch.Tensor:
        context = (
            context_states * context_mask[..., None].to(context_states.dtype)
        ).reshape(context_states.shape[0], CONTEXT_COUNT * self.architecture.width)
        weight = equity_mask[..., None].to(equity_states.dtype)
        count = weight.sum(dim=1).clamp_min(1.0)
        mean = (equity_states * weight).sum(dim=1) / count
        second = (equity_states.square() * weight).sum(dim=1) / count
        dispersion = torch.sqrt(torch.clamp(second - mean.square(), min=1e-6))
        market_moments = torch.cat((mean, dispersion), dim=-1)
        if self.variant == DECISION_TIME_FUSION_VARIANT:
            decision_context = self.decision_time_embedding(
                self._decision_phase(state_position).to(equity_states.dtype)
            )
            market_moments = market_moments + self.decision_time_fusion_adapter(
                decision_context
            )
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
            state_position,
        )
        predictions = self.prediction_head(fused)
        return predictions * equity_mask[..., None].to(predictions.dtype)


def build_model(variant: str = PARENT_MODEL_VARIANT) -> SharedCausalTCN:
    return SharedCausalTCN(variant=variant)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
