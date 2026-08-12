from __future__ import annotations

import torch
from torch import nn

from .contract import (
    ABSOLUTE_PATCH_COUNT,
    EQUITY_COUNT,
    FAMILY_COUNT,
    LOCAL_CONTEXT_COUNT,
    INSTRUMENT_COUNT,
    INSTRUMENT_FAMILY_IDS,
    NEURAL_MODELS,
    PATCH_INPUT_WIDTH,
    POOLED_INDUCING_TOKEN_COUNT,
    SLOW_FEATURE_COUNT,
    STATE_TOKEN_SLOT,
    TARGETED_FUSION_GATE_BIAS,
    TEMPORAL_TOKEN_COUNT,
    TRANSFORMER_MODELS,
    TCNArchitecture,
    TransformerArchitecture,
    validate_peer_feature_mode,
    architecture_for_model,
)
from .layers import (
    PooledMarketMemory,
    RotaryEmbedding,
    TargetedFusionBlock,
    TransformerStack,
)


class TargetedCrossAssetTransformer(nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        architecture = architecture_for_model(model_name)
        if not isinstance(architecture, TransformerArchitecture):
            raise ValueError(f"{model_name} is not a transformer setting")
        if architecture.d_model != architecture.attention_heads * architecture.head_dim:
            raise ValueError("Transformer attention dimensions are inconsistent")
        self.model_name = model_name
        self.d_model = architecture.d_model
        self.patch_projection = nn.Linear(
            PATCH_INPUT_WIDTH, architecture.d_model, bias=False
        )
        self.slow_projection = nn.Linear(
            SLOW_FEATURE_COUNT, architecture.d_model, bias=False
        )
        self.absolute_time_embedding = nn.Embedding(
            TEMPORAL_TOKEN_COUNT, architecture.d_model
        )
        self.family_embedding = nn.Embedding(FAMILY_COUNT, architecture.d_model)
        self.state_token = nn.Parameter(torch.empty(architecture.d_model))
        self.input_dropout = nn.Dropout(architecture.input_dropout)

        rope = RotaryEmbedding(
            architecture.head_dim, TEMPORAL_TOKEN_COUNT, architecture.rope_base
        )
        self.temporal_encoder = TransformerStack(
            architecture.temporal_depth,
            architecture.d_model,
            architecture.attention_heads,
            architecture.swiglu_width,
            architecture.rms_norm_eps,
            architecture.qk_norm_eps,
            architecture.residual_dropout,
            rope=rope,
        )
        self.instrument_norm = nn.RMSNorm(
            architecture.d_model, eps=architecture.rms_norm_eps
        )
        self.pooled_memory = (
            PooledMarketMemory(
                architecture.d_model,
                architecture.attention_heads,
                architecture.swiglu_width,
                architecture.rms_norm_eps,
                architecture.qk_norm_eps,
                architecture.residual_dropout,
                POOLED_INDUCING_TOKEN_COUNT,
            )
            if architecture.pooled_memory_tokens
            else None
        )
        self.targeted_fusion = (
            TargetedFusionBlock(
                architecture.d_model,
                architecture.attention_heads,
                architecture.swiglu_width,
                architecture.rms_norm_eps,
                architecture.qk_norm_eps,
                architecture.residual_dropout,
                TARGETED_FUSION_GATE_BIAS,
            )
            if architecture.fusion_blocks
            else None
        )
        self.prediction_head = nn.Linear(
            architecture.d_model, architecture.output_horizons, bias=True
        )
        self.register_buffer(
            "instrument_family_ids",
            torch.tensor(INSTRUMENT_FAMILY_IDS, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "patch_position_ids",
            torch.arange(ABSOLUTE_PATCH_COUNT, dtype=torch.long),
            persistent=False,
        )
        self.apply(self._initialize_module)
        nn.init.normal_(self.state_token, mean=0.0, std=0.02)
        if self.pooled_memory is not None:
            nn.init.normal_(self.pooled_memory.inducing_tokens, mean=0.0, std=0.02)
        if self.targeted_fusion is not None:
            nn.init.zeros_(self.targeted_fusion.gate.weight)
            nn.init.constant_(self.targeted_fusion.gate.bias, TARGETED_FUSION_GATE_BIAS)
        self.temporal_encoder.scale_residual_weights()

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)

    def _temporal_states(
        self,
        patches: torch.Tensor,
        history_patch_mask: torch.Tensor,
        slow_features: torch.Tensor,
        state_position: torch.Tensor,
        instrument_count: int,
    ) -> torch.Tensor:
        batch_size = patches.shape[0]
        patches = patches[:, :instrument_count]
        history_patch_mask = history_patch_mask[:, :instrument_count]
        slow_features = slow_features[:, :instrument_count]
        family_ids = self.instrument_family_ids[:instrument_count]
        family = self.family_embedding(family_ids)[None, :, None, :]
        absolute_time = self.absolute_time_embedding(self.patch_position_ids)[
            None, None, :, :
        ]
        patch_tokens = self.patch_projection(patches) + absolute_time + family
        state_positions = state_position[:, None].expand(-1, instrument_count)
        if instrument_count == INSTRUMENT_COUNT:
            global_slots = (
                torch.arange(instrument_count, device=patches.device)
                >= EQUITY_COUNT + LOCAL_CONTEXT_COUNT
            )
            state_positions = torch.where(
                global_slots[None], STATE_TOKEN_SLOT, state_positions
            )
        state_time = self.absolute_time_embedding(state_positions)
        state_tokens = (
            self.state_token[None, None, :]
            + state_time
            + self.family_embedding(family_ids)[None, :, :]
        ).unsqueeze(2)
        tokens = self.input_dropout(torch.cat((patch_tokens, state_tokens), dim=2))
        state_mask = torch.ones(
            (batch_size, instrument_count, 1),
            dtype=torch.bool,
            device=history_patch_mask.device,
        )
        temporal_mask = torch.cat((history_patch_mask, state_mask), dim=2).reshape(
            batch_size * instrument_count, TEMPORAL_TOKEN_COUNT
        )
        patch_positions = self.patch_position_ids.view(1, 1, -1).expand(
            batch_size, instrument_count, -1
        )
        position_ids = torch.cat(
            (patch_positions, state_positions[..., None]), dim=2
        ).reshape(batch_size * instrument_count, TEMPORAL_TOKEN_COUNT)
        encoded = self.temporal_encoder(
            tokens.reshape(
                batch_size * instrument_count,
                TEMPORAL_TOKEN_COUNT,
                self.d_model,
            ),
            temporal_mask,
            position_ids,
        )
        states = encoded[:, -1].reshape(batch_size, instrument_count, self.d_model)
        return self.instrument_norm(states + self.slow_projection(slow_features))

    def forward(
        self,
        patches: torch.Tensor,
        history_patch_mask: torch.Tensor,
        instrument_mask: torch.Tensor,
        slow_features: torch.Tensor,
        state_position: torch.Tensor,
    ) -> torch.Tensor:
        instrument_count = (
            EQUITY_COUNT
            if self.model_name in ("temporal_only", "pooled_market")
            else INSTRUMENT_COUNT
        )
        states = self._temporal_states(
            patches,
            history_patch_mask,
            slow_features,
            state_position,
            instrument_count,
        )
        equity_states = states[:, :EQUITY_COUNT]
        equity_mask = instrument_mask[:, :EQUITY_COUNT]
        if self.targeted_fusion is not None:
            memory_parts: list[torch.Tensor] = []
            mask_parts: list[torch.Tensor] = []
            if self.model_name in ("context_only", "context_pooled"):
                memory_parts.append(states[:, EQUITY_COUNT:])
                mask_parts.append(instrument_mask[:, EQUITY_COUNT:])
            if self.pooled_memory is not None:
                pooled = self.pooled_memory(equity_states, equity_mask)
                memory_parts.append(pooled)
                mask_parts.append(
                    torch.ones(pooled.shape[:2], dtype=torch.bool, device=pooled.device)
                )
            memory = torch.cat(memory_parts, dim=1)
            memory_mask = torch.cat(mask_parts, dim=1)
            equity_states = self.targeted_fusion(
                equity_states, memory, memory_mask, equity_mask
            )
        predictions = self.prediction_head(equity_states)
        return predictions * equity_mask[..., None].to(predictions.dtype)


def build_neural_model(
    model_name: str,
    tcn_architecture: TCNArchitecture | None = None,
    peer_features: str = "none",
) -> nn.Module:
    if model_name not in NEURAL_MODELS:
        raise ValueError(f"Unknown neural model: {model_name}")
    peer_features = validate_peer_feature_mode(model_name, peer_features)
    if model_name in TRANSFORMER_MODELS:
        if tcn_architecture is not None:
            raise ValueError(f"TCN architecture is forbidden for model {model_name}")
        return TargetedCrossAssetTransformer(model_name)
    from .baselines import ResidualTabularMLP, SharedCausalTCN

    if model_name == "tcn":
        if tcn_architecture is None:
            raise ValueError("TCN architecture is required for model tcn")
        return SharedCausalTCN(tcn_architecture, peer_features)
    if tcn_architecture is not None:
        raise ValueError("TCN architecture is forbidden for model mlp")
    return ResidualTabularMLP()


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
