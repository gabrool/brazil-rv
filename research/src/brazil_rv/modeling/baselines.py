from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .contract import (
    CONTEXT_COUNT,
    EQUITY_COUNT,
    HORIZON_COUNT,
    INSTRUMENT_COUNT,
    MLP_DEPTH,
    MLP_SWIGLU_WIDTH,
    MLP_WIDTH,
    PATCH_INPUT_WIDTH,
    RESIDUAL_DROPOUT,
    RMS_NORM_EPS,
    SLOW_FEATURE_COUNT,
    TABULAR_FEATURE_COUNT,
    TARGETED_FUSION_GATE_BIAS,
    TCN_DILATIONS,
    TCN_FUSION_WIDTH,
    TCN_KERNEL_SIZE,
    TCN_WIDTH,
)
from .layers import CausalTCNResidualBlock, MuonLinear, SwiGLU


class SharedCausalTCN(nn.Module):
    model_name = "tcn"

    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(PATCH_INPUT_WIDTH, TCN_WIDTH, bias=False)
        self.blocks = nn.ModuleList(
            [
                CausalTCNResidualBlock(
                    TCN_WIDTH,
                    TCN_KERNEL_SIZE,
                    dilation,
                    RESIDUAL_DROPOUT,
                )
                for dilation in TCN_DILATIONS
            ]
        )
        self.slow_projection = nn.Linear(SLOW_FEATURE_COUNT, TCN_WIDTH, bias=False)
        self.state_norm = nn.LayerNorm(TCN_WIDTH)
        self.fusion_input = nn.Linear(9 * TCN_WIDTH, TCN_FUSION_WIDTH, bias=True)
        self.fusion_output = MuonLinear(TCN_FUSION_WIDTH, TCN_WIDTH, bias=False)
        self.fusion_gate = nn.Linear(2 * TCN_WIDTH, TCN_WIDTH, bias=True)
        self.fusion_norm = nn.LayerNorm(TCN_WIDTH)
        self.dropout = nn.Dropout(RESIDUAL_DROPOUT)
        self.prediction_head = nn.Linear(TCN_WIDTH, HORIZON_COUNT, bias=True)
        self.apply(self._initialize_module)
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.constant_(self.fusion_gate.bias, TARGETED_FUSION_GATE_BIAS)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _instrument_states(
        self,
        patches: torch.Tensor,
        history_patch_mask: torch.Tensor,
        slow_features: torch.Tensor,
        state_position: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = patches.shape[0]
        masked = patches * history_patch_mask[..., None].to(patches.dtype)
        hidden = self.input_projection(masked).reshape(
            batch_size * INSTRUMENT_COUNT, masked.shape[2], TCN_WIDTH
        )
        hidden = hidden.transpose(1, 2)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = hidden.transpose(1, 2).reshape(
            batch_size, INSTRUMENT_COUNT, masked.shape[2], TCN_WIDTH
        )
        gather_index = (
            (state_position - 1)
            .view(batch_size, 1, 1, 1)
            .expand(-1, INSTRUMENT_COUNT, 1, TCN_WIDTH)
        )
        state = hidden.gather(2, gather_index).squeeze(2)
        return self.state_norm(state + self.slow_projection(slow_features))

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
        equity_states = states[:, :EQUITY_COUNT]
        context_states = states[:, EQUITY_COUNT:]
        equity_mask = instrument_mask[:, :EQUITY_COUNT]
        weight = equity_mask[..., None].to(states.dtype)
        count = weight.sum(dim=1).clamp_min(1.0)
        mean = (equity_states * weight).sum(dim=1) / count
        second_moment = (equity_states.square() * weight).sum(dim=1) / count
        dispersion = torch.sqrt(torch.clamp(second_moment - mean.square(), min=1e-6))
        shared = torch.cat(
            (
                context_states.reshape(states.shape[0], CONTEXT_COUNT * TCN_WIDTH),
                mean,
                dispersion,
            ),
            dim=-1,
        )
        shared = shared[:, None].expand(-1, EQUITY_COUNT, -1)
        fusion_input = torch.cat((equity_states, shared), dim=-1)
        fused = self.fusion_output(F.gelu(self.fusion_input(fusion_input)))
        fused = self.dropout(F.gelu(fused))
        gate = torch.sigmoid(
            self.fusion_gate(torch.cat((equity_states, fused), dim=-1))
        )
        equity_states = self.fusion_norm(equity_states + gate * fused)
        predictions = self.prediction_head(equity_states)
        return predictions * equity_mask[..., None].to(predictions.dtype)


class _MLPResidualBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(MLP_WIDTH, eps=RMS_NORM_EPS)
        self.feedforward = SwiGLU(MLP_WIDTH, MLP_SWIGLU_WIDTH)
        self.dropout = nn.Dropout(RESIDUAL_DROPOUT)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.dropout(self.feedforward(self.norm(inputs)))


class ResidualTabularMLP(nn.Module):
    model_name = "mlp"

    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(TABULAR_FEATURE_COUNT, MLP_WIDTH, bias=True)
        self.blocks = nn.ModuleList([_MLPResidualBlock() for _ in range(MLP_DEPTH)])
        self.final_norm = nn.RMSNorm(MLP_WIDTH, eps=RMS_NORM_EPS)
        self.prediction_head = nn.Linear(MLP_WIDTH, HORIZON_COUNT, bias=True)
        self.apply(self._initialize_module)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self, tabular_features: torch.Tensor, equity_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.input_projection(tabular_features)
        for block in self.blocks:
            hidden = block(hidden)
        predictions = self.prediction_head(self.final_norm(hidden))
        return predictions * equity_mask[..., None].to(predictions.dtype)
