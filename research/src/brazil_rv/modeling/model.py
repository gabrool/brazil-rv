from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from .contract import (
    CONTEXT_COUNT,
    EQUITY_COUNT,
    LOCAL_CONTEXT_COUNT,
    STATE_TOKEN_SLOT,
    TARGETED_FUSION_GATE_BIAS,
    TCN_ARCHITECTURE,
    TCNArchitecture,
)
from .layers import CausalTCNResidualBlock, MuonLinear

PARENT_MODEL_VARIANT = "parent"
COMPRESSED_GLOBAL_RISK_VARIANT = "compressed_global_risk"
FACTOR_MIXER_K4_VARIANT = "factor_mixer_k4"
FACTOR_MIXER_K8_VARIANT = "factor_mixer_k8"
SET_POOL_FACTOR_MIXER_VARIANT = "set_pool_factor_mixer_k4"
DI_TILT_EXPOSURE_VARIANT = "di_tilt_exposure"
CAPACITY_96_VARIANT = "capacity_96"
COMPETITIVE_FEATURE_GATE_VARIANT = "competitive_feature_gate"
MODEL_VARIANTS = (
    PARENT_MODEL_VARIANT,
    COMPRESSED_GLOBAL_RISK_VARIANT,
    FACTOR_MIXER_K4_VARIANT,
    FACTOR_MIXER_K8_VARIANT,
    SET_POOL_FACTOR_MIXER_VARIANT,
    DI_TILT_EXPOSURE_VARIANT,
    CAPACITY_96_VARIANT,
    COMPETITIVE_FEATURE_GATE_VARIANT,
)
MARKET_STATE_WIDTH = 4
SET_POOL_WIDTH = 16
FEATURE_GATE_TEMPERATURE = 2.0


def architecture_for_variant(variant: str) -> TCNArchitecture:
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model variant: {variant}")
    if variant == CAPACITY_96_VARIANT:
        return replace(TCN_ARCHITECTURE, width=96, fusion_width=192)
    return TCN_ARCHITECTURE


def model_variant_metadata(variant: str) -> dict[str, object]:
    architecture_for_variant(variant)
    details: dict[str, object] = {
        "name": variant,
        "zero_initialized_final_projection_only": variant
        not in (PARENT_MODEL_VARIANT, CAPACITY_96_VARIANT),
        "rng_stream_preserved_during_adapter_construction": variant
        not in (PARENT_MODEL_VARIANT, CAPACITY_96_VARIANT),
    }
    if variant == COMPRESSED_GLOBAL_RISK_VARIANT:
        details.update(
            inputs=(
                "ES_return_30m_normalized",
                "ES_realized_vol_30m_log_ratio",
                "HG_return_30m_normalized",
                "6M_return_30m_normalized",
            ),
            injection="shared_fusion_hidden_residual",
        )
    elif variant in (FACTOR_MIXER_K4_VARIANT, FACTOR_MIXER_K8_VARIANT):
        details.update(
            factors=4 if variant == FACTOR_MIXER_K4_VARIANT else 8,
            source="masked_temporal_mean_plus_slow_projection",
            injection="residual_into_fast_final_equity_state",
        )
    elif variant == SET_POOL_FACTOR_MIXER_VARIANT:
        details.update(
            factors=4,
            set_pool_width=SET_POOL_WIDTH,
            source="masked_temporal_mean_plus_slow_projection",
            injection="factor_residual_plus_shared_set_pool_residual",
        )
    elif variant == DI_TILT_EXPOSURE_VARIANT:
        details.update(
            input="causal_per_equity_beta_to_DI_tilt_sidecar",
            injection="one_zero_initialized_slow_input_row",
        )
    elif variant == CAPACITY_96_VARIANT:
        details.update(width=96, adamw_weight_decay=0.02)
    elif variant == COMPETITIVE_FEATURE_GATE_VARIANT:
        details.update(
            conditioning="compressed_global_risk_state",
            temperature=FEATURE_GATE_TEMPERATURE,
            gate="softmax_competitive_multiplicative_dynamic_feature_gate",
        )
    return details

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
        parent_modules = frozenset(self._modules)
        self._add_variant_modules(width, parent_modules)

    def _add_variant_modules(
        self, width: int, parent_modules: frozenset[str]
    ) -> None:
        if self.variant in (PARENT_MODEL_VARIANT, CAPACITY_96_VARIANT):
            return
        final_projections: list[nn.Linear] = []
        with torch.random.fork_rng(devices=[]):
            if self.variant == COMPRESSED_GLOBAL_RISK_VARIANT:
                self.market_state_encoder = nn.Sequential(
                    nn.Linear(MARKET_STATE_WIDTH, 16), nn.GELU()
                )
                self.market_state_adapter = nn.Linear(
                    16, self.architecture.fusion_width, bias=False
                )
                final_projections.append(self.market_state_adapter)
            elif self.variant in (
                FACTOR_MIXER_K4_VARIANT,
                FACTOR_MIXER_K8_VARIANT,
                SET_POOL_FACTOR_MIXER_VARIANT,
            ):
                factors = 8 if self.variant == FACTOR_MIXER_K8_VARIANT else 4
                self.factor_queries = nn.Parameter(torch.empty(factors, width))
                nn.init.normal_(self.factor_queries, mean=0.0, std=0.02)
                self.factor_loadings = nn.Linear(width, factors, bias=True)
                self.factor_output = nn.Linear(width, width, bias=False)
                final_projections.append(self.factor_output)
                if self.variant == SET_POOL_FACTOR_MIXER_VARIANT:
                    self.set_phi = nn.Sequential(
                        nn.Linear(width, SET_POOL_WIDTH),
                        nn.GELU(),
                        nn.Linear(SET_POOL_WIDTH, SET_POOL_WIDTH),
                    )
                    self.set_pool_adapter = nn.Linear(
                        SET_POOL_WIDTH, 2 * width, bias=False
                    )
                    final_projections.append(self.set_pool_adapter)
            elif self.variant == DI_TILT_EXPOSURE_VARIANT:
                self.di_tilt_projection = nn.Linear(1, width, bias=False)
                final_projections.append(self.di_tilt_projection)
            elif self.variant == COMPETITIVE_FEATURE_GATE_VARIANT:
                self.feature_gate_encoder = nn.Sequential(
                    nn.Linear(MARKET_STATE_WIDTH, 16), nn.GELU()
                )
                self.feature_gate_output = nn.Linear(
                    16, self.architecture.patch_input_width // 5, bias=False
                )
                final_projections.append(self.feature_gate_output)
            else:
                raise AssertionError(self.variant)
            for name, module in self.named_children():
                if name not in parent_modules:
                    module.apply(self._initialize_module)
            for projection in final_projections:
                nn.init.zeros_(projection.weight)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _stream_states(
        self, hidden: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
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

    def _masked_temporal_mean(
        self,
        hidden: torch.Tensor,
        history_patch_mask: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        streams = self._stream_states(hidden, batch_size)
        weight = history_patch_mask[..., None].to(streams.dtype)
        return (streams * weight).sum(dim=2) / weight.sum(dim=2).clamp_min(1.0)

    def _instrument_states(
        self,
        patches: torch.Tensor,
        history_patch_mask: torch.Tensor,
        slow_features: torch.Tensor,
        state_position: torch.Tensor,
        tilt_exposure: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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
        slow = self.slow_projection(slow_features)
        if self.variant == DI_TILT_EXPOSURE_VARIANT:
            slow = slow.clone()
            slow[:, : self.equity_count] = slow[:, : self.equity_count] + (
                self.di_tilt_projection(tilt_exposure[..., None])
            )
        states = self.state_norm(raw + slow)
        if self.variant in (
            FACTOR_MIXER_K4_VARIANT,
            FACTOR_MIXER_K8_VARIANT,
            SET_POOL_FACTOR_MIXER_VARIANT,
        ):
            smoothed = self.state_norm(
                self._masked_temporal_mean(hidden, history_patch_mask, batch_size) + slow
            )
            return states, smoothed
        return states, None

    def _factor_mix(
        self, smoothed: torch.Tensor, equity_mask: torch.Tensor
    ) -> torch.Tensor:
        equity = smoothed[:, : self.equity_count]
        scores = torch.einsum("bew,kw->bek", equity, self.factor_queries)
        scores = scores / self.architecture.width**0.5
        attention = torch.softmax(
            scores.transpose(1, 2).masked_fill(~equity_mask[:, None], -torch.inf),
            dim=-1,
        )
        factors = torch.einsum("bke,bew->bkw", attention, equity)
        loadings = torch.softmax(self.factor_loadings(equity), dim=-1)
        mixed = torch.einsum("bek,bkw->bew", loadings, factors)
        return self.factor_output(mixed)

    def _fuse(
        self,
        equity_states: torch.Tensor,
        context_states: torch.Tensor,
        equity_mask: torch.Tensor,
        context_mask: torch.Tensor,
        market_state: torch.Tensor,
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
        if self.variant == SET_POOL_FACTOR_MIXER_VARIANT:
            transformed = self.set_phi(equity_states)
            transformed_weight = equity_mask[..., None].to(transformed.dtype)
            pooled = (transformed * transformed_weight).sum(
                dim=1
            ) / transformed_weight.sum(dim=1).clamp_min(1.0)
            market_moments = market_moments + self.set_pool_adapter(pooled)
        shared = torch.cat((context, market_moments), dim=-1)
        shared = shared[:, None].expand(-1, self.equity_count, -1)
        inputs = torch.cat((equity_states, shared), dim=-1)
        fusion_hidden = self.fusion_input(inputs)
        if self.variant == COMPRESSED_GLOBAL_RISK_VARIANT:
            market = self.market_state_adapter(self.market_state_encoder(market_state))
            fusion_hidden = fusion_hidden + market[:, None]
        fused = self.fusion_output(F.gelu(fusion_hidden))
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
        market_state: torch.Tensor | None = None,
        tilt_exposure: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if market_state is None:
            market_state = patches.new_zeros((patches.shape[0], MARKET_STATE_WIDTH))
        if tilt_exposure is None:
            tilt_exposure = patches.new_zeros((patches.shape[0], self.equity_count))
        if self.variant == COMPETITIVE_FEATURE_GATE_VARIANT:
            logits = self.feature_gate_output(self.feature_gate_encoder(market_state))
            gate = logits.shape[-1] * torch.softmax(
                FEATURE_GATE_TEMPERATURE * logits, dim=-1
            )
            patch_gate = gate[:, None, None, None, :].expand(
                -1, self.instrument_count, patches.shape[2], 5, -1
            )
            patches = patches.reshape(
                patches.shape[0],
                self.instrument_count,
                patches.shape[2],
                5,
                -1,
            )
            patches = (patches * patch_gate).flatten(-2)
        states, smoothed = self._instrument_states(
            patches,
            history_patch_mask,
            slow_features,
            state_position,
            tilt_exposure,
        )
        equity_mask = instrument_mask[:, : self.equity_count]
        equity_states = states[:, : self.equity_count]
        if smoothed is not None:
            equity_states = equity_states + self._factor_mix(smoothed, equity_mask)
        fused = self._fuse(
            equity_states,
            states[:, self.equity_count :],
            equity_mask,
            instrument_mask[:, self.equity_count :],
            market_state,
        )
        predictions = self.prediction_head(fused)
        return predictions * equity_mask[..., None].to(predictions.dtype)


def build_model(variant: str = PARENT_MODEL_VARIANT) -> SharedCausalTCN:
    return SharedCausalTCN(
        architecture=architecture_for_variant(variant), variant=variant
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
