from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .contract import (
    CONTEXT_COUNT,
    EQUITY_COUNT,
    LOCAL_CONTEXT_COUNT,
    TARGETED_FUSION_GATE_BIAS,
    TCN_ARCHITECTURE,
    TCNArchitecture,
)
from .layers import CausalTCNResidualBlock, MuonLinear


class SharedCausalTCN(nn.Module):
    model_name = "tcn"

    def __init__(
        self,
        *,
        architecture: TCNArchitecture = TCN_ARCHITECTURE,
        equity_count: int = EQUITY_COUNT,
        sidecar_feature_count: int | None = None,
        single_horizon_index: int | None = None,
        to_close_head: bool = False,
    ) -> None:
        super().__init__()
        if sidecar_feature_count is not None and sidecar_feature_count <= 0:
            raise ValueError("sidecar_feature_count must be positive")
        self.architecture = architecture
        self.equity_count = equity_count
        self.instrument_count = equity_count + CONTEXT_COUNT
        self.sidecar_feature_count = sidecar_feature_count
        self.to_close_head = bool(to_close_head)
        if self.to_close_head and architecture.output_horizons != 4:
            raise ValueError("The horizon-conditioned to-close head requires 4 outputs")
        if self.to_close_head and single_horizon_index is not None:
            raise ValueError("The to-close head cannot be combined with head masking")
        if single_horizon_index is not None and single_horizon_index not in (0, 1, 2):
            raise ValueError("single_horizon_index must be 0, 1, or 2")
        self.single_horizon_index = single_horizon_index
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
        incumbent_horizons = 3 if self.to_close_head else architecture.output_horizons
        self.prediction_head = nn.Linear(
            width,
            1 if single_horizon_index is not None else incumbent_horizons,
            bias=True,
        )
        to_close_readouts: nn.Linear | None = None
        if self.to_close_head:
            rng_state = torch.get_rng_state()
            try:
                to_close_readouts = nn.Linear(width, 3, bias=True)
            finally:
                torch.set_rng_state(rng_state)
        self.to_close_readouts = None
        self.apply(self._initialize_module)
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.constant_(self.fusion_gate.bias, TARGETED_FUSION_GATE_BIAS)
        self.sidecar_adapter: nn.Linear | None = None
        if sidecar_feature_count is not None:
            rng_state = torch.get_rng_state()
            try:
                self.sidecar_adapter = nn.Linear(
                    2 * sidecar_feature_count, width, bias=False
                )
            finally:
                torch.set_rng_state(rng_state)
            nn.init.zeros_(self.sidecar_adapter.weight)
        if to_close_readouts is not None:
            rng_state = torch.get_rng_state()
            try:
                self._initialize_module(to_close_readouts)
            finally:
                torch.set_rng_state(rng_state)
            nn.init.zeros_(to_close_readouts.weight)
            nn.init.zeros_(to_close_readouts.bias)
            self.to_close_readouts = to_close_readouts

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
        positions = torch.where(global_slots[None], hidden.shape[-1], positions)
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

    def _fuse(
        self,
        equity_states: torch.Tensor,
        context_states: torch.Tensor,
        equity_mask: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        context = (
            context_states * context_mask[..., None].to(context_states.dtype)
        ).reshape(context_states.shape[0], CONTEXT_COUNT * self.architecture.width)
        weight = equity_mask[..., None].to(equity_states.dtype)
        count = weight.sum(dim=1).clamp_min(1.0)
        mean = (equity_states * weight).sum(dim=1) / count
        second = (equity_states.square() * weight).sum(dim=1) / count
        dispersion = torch.sqrt(torch.clamp(second - mean.square(), min=1e-6))
        shared = torch.cat((context, mean, dispersion), dim=-1)
        shared = shared[:, None].expand(-1, self.equity_count, -1)
        inputs = torch.cat((equity_states, shared), dim=-1)
        fused = self.fusion_output(F.gelu(self.fusion_input(inputs)))
        fused = self.dropout(F.gelu(fused))
        gate = torch.sigmoid(
            self.fusion_gate(torch.cat((equity_states, fused), dim=-1))
        )
        return self.fusion_norm(equity_states + gate * fused)

    def _inject_sidecar(
        self,
        equity_states: torch.Tensor,
        sidecar_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.sidecar_adapter is None:
            if sidecar_features is not None:
                raise ValueError("Parent model received unexpected sidecar features")
            return equity_states
        if sidecar_features is None:
            raise ValueError("Sidecar model requires sidecar features")
        if sidecar_features.shape[:2] != equity_states.shape[:2]:
            raise ValueError("Sidecar batch/equity axes are misaligned")
        return equity_states + self.sidecar_adapter(sidecar_features)

    def forward(
        self,
        patches: torch.Tensor,
        history_patch_mask: torch.Tensor,
        instrument_mask: torch.Tensor,
        slow_features: torch.Tensor,
        state_position: torch.Tensor,
        sidecar_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        states = self._instrument_states(
            patches, history_patch_mask, slow_features, state_position
        )
        equity_mask = instrument_mask[:, : self.equity_count]
        equity_states = self._inject_sidecar(
            states[:, : self.equity_count], sidecar_features
        )
        fused = self._fuse(
            equity_states,
            states[:, self.equity_count :],
            equity_mask,
            instrument_mask[:, self.equity_count :],
        )
        predictions = self.prediction_head(fused)
        if self.to_close_readouts is not None:
            # Under the five-minute contract state_position is 15 + decision_idx;
            # the standard equity entry is 5*state_position - 60.
            remaining = (465 - 5 * state_position).to(fused.dtype)
            h = remaining / 405
            basis = torch.stack((torch.ones_like(h), h, torch.sqrt(h)), dim=-1)
            to_close = (self.to_close_readouts(fused) * basis[:, None]).sum(
                dim=-1, keepdim=True
            )
            predictions = torch.cat((predictions, to_close), dim=-1)
        if self.single_horizon_index is not None:
            horizon_mask = F.one_hot(
                torch.tensor(self.single_horizon_index, device=predictions.device),
                num_classes=self.architecture.output_horizons,
            ).to(predictions.dtype)
            predictions = predictions * horizon_mask
        return predictions * equity_mask[..., None].to(predictions.dtype)


def build_model(
    sidecar_feature_count: int | None = None,
    single_horizon_index: int | None = None,
    architecture: TCNArchitecture = TCN_ARCHITECTURE,
    to_close_head: bool = False,
) -> SharedCausalTCN:
    return SharedCausalTCN(
        architecture=architecture,
        sidecar_feature_count=sidecar_feature_count,
        single_horizon_index=single_horizon_index,
        to_close_head=to_close_head,
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
