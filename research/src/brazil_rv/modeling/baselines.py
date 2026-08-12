from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .contract import (
    CONTEXT_COUNT,
    EQUITY_COUNT,
    HORIZON_COUNT,
    INSTRUMENT_COUNT,
    LOCAL_CONTEXT_COUNT,
    MLP_DEPTH,
    MLP_SWIGLU_WIDTH,
    MLP_WIDTH,
    PEER_STATE_WIDTH,
    RESIDUAL_DROPOUT,
    RMS_NORM_EPS,
    TABULAR_FEATURE_COUNT,
    TARGETED_FUSION_GATE_BIAS,
    STATE_TOKEN_SLOT,
    TCNArchitecture,
    validate_peer_feature_mode,
)
from .layers import CausalTCNResidualBlock, MuonLinear, SwiGLU


class SharedCausalTCN(nn.Module):
    model_name = "tcn"

    def __init__(
        self, architecture: TCNArchitecture, peer_features: str = "none"
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.peer_features = validate_peer_feature_mode(self.model_name, peer_features)
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
                    architecture.block,
                    architecture.swiglu_hidden_width,
                )
                for dilation in architecture.dilations
            ]
        )
        self.slow_projection = nn.Linear(architecture.slow_width, width, bias=False)
        self.state_norm = nn.LayerNorm(width)
        if architecture.fusion_mode != "none":
            self.fusion_input = nn.Linear(
                architecture.fusion_states * width,
                architecture.fusion_width,
                bias=True,
            )
            self.fusion_output = MuonLinear(
                architecture.fusion_width, width, bias=False
            )
            self.fusion_gate = nn.Linear(2 * width, width, bias=True)
            self.fusion_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(architecture.dropout)
        self.prediction_head = nn.Linear(width, architecture.output_horizons, bias=True)
        self.apply(self._initialize_module)
        if architecture.fusion_mode != "none":
            nn.init.zeros_(self.fusion_gate.weight)
            nn.init.constant_(self.fusion_gate.bias, TARGETED_FUSION_GATE_BIAS)
        self.peer_adapter: nn.Linear | None = None
        if self.peer_features != "none":
            with torch.random.fork_rng(devices=[]):
                self.peer_adapter = nn.Linear(PEER_STATE_WIDTH, width, bias=False)
                nn.init.zeros_(self.peer_adapter.weight)

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
        architecture = self.architecture
        batch_size = patches.shape[0]
        instrument_count = (
            INSTRUMENT_COUNT
            if architecture.fusion_mode in ("context_only", "context_pooled")
            else EQUITY_COUNT
        )
        patches = patches[:, :instrument_count]
        history_patch_mask = history_patch_mask[:, :instrument_count]
        slow_features = slow_features[:, :instrument_count]
        masked = patches * history_patch_mask[..., None].to(patches.dtype)
        hidden = self.input_projection(masked).reshape(
            batch_size * instrument_count, masked.shape[2], architecture.width
        )
        hidden = hidden.transpose(1, 2)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = hidden.transpose(1, 2).reshape(
            batch_size, instrument_count, masked.shape[2], architecture.width
        )
        gather_positions = state_position[:, None].expand(-1, instrument_count)
        if instrument_count == INSTRUMENT_COUNT:
            global_slots = (
                torch.arange(instrument_count, device=patches.device)
                >= EQUITY_COUNT + LOCAL_CONTEXT_COUNT
            )
            gather_positions = torch.where(
                global_slots[None], STATE_TOKEN_SLOT, gather_positions
            )
        gather_index = (gather_positions - 1)[..., None, None].expand(
            -1, -1, 1, architecture.width
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
        peer_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        states = self._instrument_states(
            patches, history_patch_mask, slow_features, state_position
        )
        equity_states = states[:, :EQUITY_COUNT]
        equity_mask = instrument_mask[:, :EQUITY_COUNT]
        if self.peer_adapter is None:
            if peer_state is not None:
                raise ValueError("Peer state is forbidden when peer features are none")
        else:
            if peer_state is None:
                raise ValueError("Peer state is required for peer-enabled TCN")
            if peer_state.shape != (
                equity_states.shape[0],
                EQUITY_COUNT,
                PEER_STATE_WIDTH,
            ):
                raise ValueError("Peer state has the wrong shape")
            equity_states = equity_states + self.peer_adapter(peer_state)
        fusion_mode = self.architecture.fusion_mode
        if fusion_mode != "none":
            shared_parts: list[torch.Tensor] = []
            if fusion_mode in ("context_only", "context_pooled"):
                shared_parts.append(
                    (
                        states[:, EQUITY_COUNT:]
                        * instrument_mask[:, EQUITY_COUNT:, None].to(states.dtype)
                    ).reshape(states.shape[0], CONTEXT_COUNT * self.architecture.width)
                )
            if fusion_mode in ("pooled_market", "context_pooled"):
                weight = equity_mask[..., None].to(states.dtype)
                count = weight.sum(dim=1).clamp_min(1.0)
                mean = (equity_states * weight).sum(dim=1) / count
                second_moment = (equity_states.square() * weight).sum(dim=1) / count
                dispersion = torch.sqrt(
                    torch.clamp(second_moment - mean.square(), min=1e-6)
                )
                shared_parts.extend((mean, dispersion))
            shared = torch.cat(shared_parts, dim=-1)
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
