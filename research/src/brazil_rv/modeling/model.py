from __future__ import annotations

import torch
from torch import nn

from .contract import (
    ABSOLUTE_PATCH_COUNT,
    ATTENTION_HEADS,
    CROSS_ASSET_DEPTH,
    D_MODEL,
    EQUITY_COUNT,
    FAMILY_COUNT,
    HEAD_DIM,
    HORIZON_COUNT,
    INPUT_DROPOUT,
    INSTRUMENT_FAMILY_IDS,
    INSTRUMENT_COUNT,
    MODEL_VARIANTS,
    PATCH_INPUT_WIDTH,
    QK_NORM_EPS,
    RESIDUAL_DROPOUT,
    RMS_NORM_EPS,
    ROPE_BASE,
    SWIGLU_WIDTH,
    TEMPORAL_DEPTH,
    TEMPORAL_TOKEN_COUNT,
)
from .layers import RotaryEmbedding, TransformerStack


class CrossAssetPatchITransformerV1(nn.Module):
    def __init__(self, variant: str = "full") -> None:
        super().__init__()
        if variant not in MODEL_VARIANTS:
            raise ValueError(f"Unknown model variant: {variant}")
        if D_MODEL != ATTENTION_HEADS * HEAD_DIM:
            raise ValueError("Fixed attention dimensions are inconsistent")
        self.variant = variant
        self.patch_projection = nn.Linear(PATCH_INPUT_WIDTH, D_MODEL, bias=False)
        self.slow_projection = nn.Linear(3, D_MODEL, bias=False)
        self.absolute_time_embedding = nn.Embedding(TEMPORAL_TOKEN_COUNT, D_MODEL)
        self.family_embedding = nn.Embedding(FAMILY_COUNT, D_MODEL)
        self.state_token = nn.Parameter(torch.empty(D_MODEL))
        self.input_dropout = nn.Dropout(INPUT_DROPOUT)

        rope = RotaryEmbedding(HEAD_DIM, TEMPORAL_TOKEN_COUNT, ROPE_BASE)
        self.temporal_encoder = TransformerStack(
            TEMPORAL_DEPTH,
            D_MODEL,
            ATTENTION_HEADS,
            SWIGLU_WIDTH,
            RMS_NORM_EPS,
            QK_NORM_EPS,
            RESIDUAL_DROPOUT,
            rope=rope,
        )
        self.instrument_norm = nn.RMSNorm(D_MODEL, eps=RMS_NORM_EPS)
        self.cross_asset_encoder = (
            TransformerStack(
                CROSS_ASSET_DEPTH,
                D_MODEL,
                ATTENTION_HEADS,
                SWIGLU_WIDTH,
                RMS_NORM_EPS,
                QK_NORM_EPS,
                RESIDUAL_DROPOUT,
                rope=None,
            )
            if variant == "full"
            else None
        )
        self.prediction_head = nn.Linear(D_MODEL, HORIZON_COUNT, bias=True)
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
        self.temporal_encoder.scale_residual_weights()
        if self.cross_asset_encoder is not None:
            self.cross_asset_encoder.scale_residual_weights()

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
        state_time = self.absolute_time_embedding(state_position)[:, None, :]
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
        state_positions = state_position.view(batch_size, 1, 1).expand(
            -1, instrument_count, -1
        )
        position_ids = torch.cat((patch_positions, state_positions), dim=2).reshape(
            batch_size * instrument_count, TEMPORAL_TOKEN_COUNT
        )
        encoded = self.temporal_encoder(
            tokens.reshape(
                batch_size * instrument_count,
                TEMPORAL_TOKEN_COUNT,
                D_MODEL,
            ),
            temporal_mask,
            position_ids,
        )
        states = encoded[:, -1].reshape(batch_size, instrument_count, D_MODEL)
        return self.instrument_norm(states + self.slow_projection(slow_features))

    def forward(
        self,
        patches: torch.Tensor,
        history_patch_mask: torch.Tensor,
        instrument_mask: torch.Tensor,
        slow_features: torch.Tensor,
        state_position: torch.Tensor,
    ) -> torch.Tensor:
        if self.variant == "temporal_only":
            equity_states = self._temporal_states(
                patches,
                history_patch_mask,
                slow_features,
                state_position,
                EQUITY_COUNT,
            )
        else:
            instrument_states = self._temporal_states(
                patches,
                history_patch_mask,
                slow_features,
                state_position,
                INSTRUMENT_COUNT,
            )
            if self.cross_asset_encoder is None:
                raise RuntimeError("Full model is missing its cross-asset encoder")
            instrument_states = self.cross_asset_encoder(
                instrument_states, instrument_mask
            )
            equity_states = instrument_states[:, :EQUITY_COUNT]
        predictions = self.prediction_head(equity_states)
        return predictions * instrument_mask[:, :EQUITY_COUNT, None].to(
            predictions.dtype
        )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
