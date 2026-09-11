from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

from brazil_rv.modeling.contract import TCN_ARCHITECTURE
from brazil_rv.modeling.layers import CausalTCNResidualBlock, SwiGLU

from .config import ModelConfig
from .contract import (
    FAST_REAL_PATCHES,
    TARGETED_FUSION_GATE_BIAS,
    V1_STORE_V2_ZERO_SLOW_FIELDS,
)

_LEGACY_FAST_PATCH_WIDTH = TCN_ARCHITECTURE.patch_input_width
_NATIVE_FAST_CHANNELS = 7
_NATIVE_FAST_INPUT_WIDTH = 2 * _NATIVE_FAST_CHANNELS
_FAST_HIDDEN_WIDTH = TCN_ARCHITECTURE.width
_V1_EQUITY_PREFIX_PATCHES = 12
_V1_ABSOLUTE_STATE_POSITION = _V1_EQUITY_PREFIX_PATCHES + FAST_REAL_PATCHES
_FEATURE_AGE_CAP_SESSIONS = 252.0
_FEATURE_AGE_LOG_DENOMINATOR = math.log1p(_FEATURE_AGE_CAP_SESSIONS)


def _bounded_feature_age(
    age_sessions: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bounded age and its independent known/left-censored indicator."""

    torch._assert_async(
        torch.all(torch.isfinite(age_sessions)),
        "feature ages must be finite",
    )
    age_known = age_sessions >= 0.0
    torch._assert_async(
        torch.all((age_sessions >= -1.0) & (~valid | age_known)),
        "feature ages must be at least -1 and known for every valid feature",
    )
    bounded = (
        torch.log1p(age_sessions.clamp(min=0.0, max=_FEATURE_AGE_CAP_SESSIONS))
        / _FEATURE_AGE_LOG_DENOMINATOR
    )
    return torch.where(age_known, bounded, torch.zeros_like(bounded)), age_known


class FastTCNEncoder(nn.Module):
    """Encode completed within-day patches, with an isolated legacy adapter."""

    def __init__(self, *, legacy_v1_context: bool = False) -> None:
        super().__init__()
        self.legacy_v1_context = legacy_v1_context
        self.input_projection = nn.Linear(
            (
                _LEGACY_FAST_PATCH_WIDTH
                if legacy_v1_context
                else _NATIVE_FAST_INPUT_WIDTH
            ),
            _FAST_HIDDEN_WIDTH,
            bias=False,
        )
        self.blocks = nn.ModuleList(
            CausalTCNResidualBlock(
                _FAST_HIDDEN_WIDTH,
                TCN_ARCHITECTURE.kernel_size,
                dilation,
                TCN_ARCHITECTURE.dropout,
                TCN_ARCHITECTURE.swiglu_hidden_width,
            )
            for dilation in TCN_ARCHITECTURE.dilations
        )
        self.slow_projection = (
            nn.Linear(TCN_ARCHITECTURE.slow_width, _FAST_HIDDEN_WIDTH, bias=False)
            if legacy_v1_context
            else None
        )
        self.state_norm = nn.LayerNorm(_FAST_HIDDEN_WIDTH)
        keep = torch.ones(TCN_ARCHITECTURE.slow_width, dtype=torch.float32)
        if legacy_v1_context:
            keep[list(V1_STORE_V2_ZERO_SLOW_FIELDS)] = 0.0
        self.register_buffer("slow_keep_mask", keep, persistent=False)
        self.apply(_initialize_module)

    def forward(
        self,
        patches: torch.Tensor,
        patch_mask: torch.Tensor,
        v1_equity_slow: torch.Tensor | None = None,
        state_position: torch.Tensor | None = None,
        *,
        patch_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if patches.ndim != 4:
            raise ValueError("patches must have shape [batch, name, patch, field]")
        if patch_mask.shape != patches.shape[:-1]:
            raise ValueError("patch_mask is misaligned with patches")
        batch_size, name_count, patch_count, _ = patches.shape
        if self.legacy_v1_context:
            if patches.shape[-1] != _LEGACY_FAST_PATCH_WIDTH:
                raise ValueError("legacy fast patches must have width 130")
            if patch_count != FAST_REAL_PATCHES:
                raise ValueError("the legacy fast stream must have 69 real patches")
            if patch_valid is not None and patch_valid.shape != patches.shape:
                raise ValueError("legacy fast validity is misaligned with patches")
            if v1_equity_slow is None or v1_equity_slow.shape != (
                batch_size,
                name_count,
                TCN_ARCHITECTURE.slow_width,
            ):
                raise ValueError("legacy v1 context requires shape [batch, name, 32]")
            prefix = patches.new_zeros(
                batch_size,
                name_count,
                _V1_EQUITY_PREFIX_PATCHES,
                _LEGACY_FAST_PATCH_WIDTH,
            )
            prefix_mask = patch_mask.new_zeros(
                batch_size, name_count, _V1_EQUITY_PREFIX_PATCHES
            )
            absolute_patches = torch.cat((prefix, patches), dim=2)
            absolute_mask = torch.cat((prefix_mask, patch_mask), dim=2)
        else:
            if patches.shape[-1] != _NATIVE_FAST_CHANNELS:
                raise ValueError("native fast patches must have seven channels")
            if not 0 < patch_count <= FAST_REAL_PATCHES:
                raise ValueError("native fast patch count is outside the frozen limit")
            if patch_valid is None or patch_valid.shape != patches.shape:
                raise ValueError("native fast patches require per-channel validity")
            absolute_patches = patches
            absolute_mask = patch_mask
        if self.legacy_v1_context:
            masked = torch.where(
                absolute_mask[..., None],
                absolute_patches,
                torch.zeros_like(absolute_patches),
            )
        else:
            assert patch_valid is not None
            effective_valid = patch_valid.bool() & patch_mask[..., None].bool()
            clean = torch.where(effective_valid, patches, torch.zeros_like(patches))
            torch._assert_async(
                torch.all(torch.isfinite(clean)),
                "native fast values marked valid must be finite",
            )
            masked = torch.cat((clean, effective_valid.to(clean.dtype)), dim=-1)
        absolute_patch_count = masked.shape[2]
        hidden = (
            self.input_projection(masked)
            .permute(0, 1, 3, 2)
            .reshape(
                batch_size * name_count,
                _FAST_HIDDEN_WIDTH,
                absolute_patch_count,
            )
        )
        for block in self.blocks:
            hidden = block(hidden)
        sequence = hidden.reshape(
            batch_size, name_count, _FAST_HIDDEN_WIDTH, absolute_patch_count
        ).permute(0, 1, 3, 2)
        if state_position is None:
            last = torch.full(
                (batch_size, name_count),
                absolute_patch_count - 1,
                device=patches.device,
                dtype=torch.long,
            )
        else:
            last = state_position.to(device=patches.device, dtype=torch.long) - 1
            if last.ndim == 1:
                last = last[:, None].expand(-1, name_count)
            if last.shape != (batch_size, name_count):
                raise ValueError(
                    "state_position must have shape [batch] or [batch, name]"
                )
        if self.legacy_v1_context:
            torch._assert_async(
                torch.all(
                    (last == absolute_patch_count - 1)
                    | ((last == -1) & ~absolute_mask.bool().any(dim=-1))
                ),
                "legacy fast_state_position must identify absolute position 81",
            )
        else:
            expected_position = patch_mask.long().sum(dim=-1)
            expected_mask = (
                torch.arange(patch_count, device=patches.device)[None, None, :]
                < expected_position[..., None]
            )
            torch._assert_async(
                torch.all(
                    (last + 1 == expected_position)
                    & ((last >= 0) | (expected_position == 0))
                    & torch.all(patch_mask.bool() == expected_mask, dim=-1)
                ),
                "native fast_state_position must identify its last present patch",
            )
        index = last.clamp_min(0)[..., None, None].expand(-1, -1, 1, _FAST_HIDDEN_WIDTH)
        raw = sequence.gather(2, index).squeeze(2)
        if self.legacy_v1_context:
            assert v1_equity_slow is not None
            assert self.slow_projection is not None
            slow_keep = self.slow_keep_mask.to(dtype=torch.bool)
            neutralized_slow = torch.where(
                slow_keep,
                v1_equity_slow,
                torch.zeros_like(v1_equity_slow),
            )
            raw = raw + self.slow_projection(neutralized_slow)
        return self.state_norm(raw)


class VectorSwiGLUResidualBlock(nn.Module):
    def __init__(self, width: int, hidden_width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.swiglu = SwiGLU(width, hidden_width)
        self.dropout = nn.Dropout(dropout)
        self.apply(_initialize_module)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.dropout(self.swiglu(self.norm(inputs)))


class DailyMultiHorizonModel(nn.Module):
    """Shared, embedding-free daily model for five horizons and to-close."""

    horizon_count = 5
    output_count = 6

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.pretrained_parameter_names: frozenset[str] = frozenset()
        self.fast_checkpoint_sha256: str | None = None
        self.pretrain_checkpoint_sha256: str | None = None
        legacy_fast = config.fast_encoder_mode == "legacy_v1_contaminated"
        self.fast_initialization_provenance: dict[str, object] = {
            "mode": (
                "legacy_v1_architecture_uninitialized" if legacy_fast else "fresh"
            ),
            "contaminated": legacy_fast,
            "explicitly_allowed": config.allow_contaminated_v1_initialization,
            "checkpoint_sha256": None,
        }
        self.slow_input_projection = nn.Linear(
            4 * config.slow_feature_count, config.hidden_width
        )
        self.slow_input_norm = nn.LayerNorm(config.hidden_width)
        if config.slow_encoder_kind == "mlp":
            self.slow_encoder = nn.Sequential(
                *(
                    VectorSwiGLUResidualBlock(
                        config.hidden_width, config.trunk_swiglu_hidden, config.dropout
                    )
                    for _ in range(2)
                )
            )
        else:
            self.slow_encoder = nn.GRU(
                config.hidden_width,
                config.hidden_width,
                num_layers=config.gru_layers,
                batch_first=True,
                dropout=config.dropout if config.gru_layers == 2 else 0.0,
            )
        if config.current_feature_count:
            self.current_input_projection = nn.Linear(
                4 * config.current_feature_count, config.hidden_width
            )
            self.current_input_norm = nn.LayerNorm(config.hidden_width)
            self.fast_encoder = FastTCNEncoder(legacy_v1_context=legacy_fast)
            self.absent_state = nn.Parameter(torch.zeros(_FAST_HIDDEN_WIDTH))
            self.fast_gate = nn.Linear(
                config.hidden_width + _FAST_HIDDEN_WIDTH, _FAST_HIDDEN_WIDTH
            )
        self.pool_gate = nn.Linear(
            config.hidden_width + 2 * config.hidden_width,
            2 * config.hidden_width,
        )
        fusion_input_width = (
            4 * config.hidden_width + _FAST_HIDDEN_WIDTH + 1
            if config.current_feature_count
            else 3 * config.hidden_width
        )
        fusion_input_width += config.common_state_feature_count
        self.fusion_projection = nn.Linear(fusion_input_width, config.fusion_width)
        self.trunk = nn.Sequential(
            *(
                VectorSwiGLUResidualBlock(
                    config.fusion_width,
                    config.trunk_swiglu_hidden,
                    config.dropout,
                )
                for _ in range(config.trunk_blocks)
            )
        )
        self.heads = nn.ModuleDict(
            {
                **{
                    f"d{horizon}": nn.Linear(config.fusion_width, 1)
                    for horizon in (1, 2, 3, 5, 10)
                },
                "to_close": nn.Linear(config.fusion_width, 1),
            }
        )
        self.apply(_initialize_module)
        if config.current_feature_count:
            nn.init.zeros_(self.fast_gate.weight)
            nn.init.constant_(self.fast_gate.bias, TARGETED_FUSION_GATE_BIAS)
        nn.init.zeros_(self.pool_gate.weight)
        nn.init.constant_(self.pool_gate.bias, TARGETED_FUSION_GATE_BIAS)
        nn.init.zeros_(self.heads["to_close"].weight)
        nn.init.zeros_(self.heads["to_close"].bias)
        if config.fast_pretrained_checkpoint is not None:
            load_v1_fast_encoder(
                self,
                config.fast_pretrained_checkpoint,
                expected_sha256=config.fast_pretrained_sha256,
            )
        # Construct after the parent, using zeros directly: no RNG consumption,
        # so both its initialization and subsequent dropout stream stay exact.
        self.sidecar_projections = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(config.fusion_width, 3 * count))
                for name, count in config.sidecar_feature_counts
            }
        )

    def _slow_states(
        self,
        slow_features: torch.Tensor,
        slow_feature_mask: torch.Tensor,
        slow_history_mask: torch.Tensor,
        slow_feature_age_sessions: torch.Tensor,
    ) -> torch.Tensor:
        if slow_features.ndim != 4:
            raise ValueError("slow_features must have shape [batch, name, date, field]")
        if slow_history_mask.shape != slow_features.shape[:-1]:
            raise ValueError("slow_history_mask is misaligned with slow_features")
        if slow_feature_mask.shape != slow_features.shape:
            raise ValueError("slow_feature_mask is misaligned with slow_features")
        if slow_feature_age_sessions.shape != slow_features.shape:
            raise ValueError("slow feature ages are misaligned with slow_features")
        if slow_features.shape[-1] != self.config.slow_feature_count:
            raise ValueError("slow feature width differs from the model configuration")
        batch_size, name_count, lookback, _ = slow_features.shape
        if lookback != self.config.slow_lookback:
            raise ValueError("slow lookback differs from the model configuration")
        feature_valid = slow_feature_mask.bool()
        valid = slow_history_mask.bool()
        torch._assert_async(
            torch.all(~feature_valid | valid[..., None]),
            "slow features cannot be valid in left-padding rows",
        )
        torch._assert_async(
            torch.all(~valid[..., :-1] | valid[..., 1:]),
            "slow history must be a left-padded calendar suffix",
        )
        if self.config.slow_encoder_kind == "mlp":
            # E8 sees only the final permitted slow row (t-1), including that
            # row's masks and ages. Never substitute an older observed row.
            slow_features = slow_features[..., -1:, :]
            feature_valid = feature_valid[..., -1:, :]
            slow_feature_age_sessions = slow_feature_age_sessions[..., -1:, :]
            valid = valid[..., -1:]
        clean = torch.where(
            feature_valid, slow_features, torch.zeros_like(slow_features)
        )
        bounded_age, age_known = _bounded_feature_age(
            slow_feature_age_sessions.to(dtype=clean.dtype), feature_valid
        )
        projected = self.slow_input_norm(
            self.slow_input_projection(
                torch.cat(
                    (
                        clean,
                        feature_valid.to(clean.dtype),
                        bounded_age,
                        age_known.to(clean.dtype),
                    ),
                    dim=-1,
                )
            )
        )
        projected = torch.where(
            valid[..., None], projected, torch.zeros_like(projected)
        )
        if self.config.slow_encoder_kind == "mlp":
            state = self.slow_encoder(projected[..., 0, :])
            return torch.where(valid[..., -1, None], state, torch.zeros_like(state))
        flat = projected.reshape(batch_size * name_count, lookback, -1)
        lengths = valid.reshape(batch_size * name_count, lookback).sum(dim=1)
        has_history = lengths > 0
        # Move each real calendar suffix to the front without compressing
        # missing sessions inside it.
        positions = torch.arange(lookback, device=slow_features.device)[None, :]
        source = lookback - lengths[:, None] + positions
        source = source.clamp(min=0, max=lookback - 1)
        packed_inputs = flat.gather(1, source[..., None].expand(-1, -1, flat.shape[-1]))
        packed_inputs = torch.where(
            positions[..., None] < lengths[:, None, None],
            packed_inputs,
            torch.zeros_like(packed_inputs),
        )
        # The real suffix is now right-padded.  Gather the output at its final
        # real calendar step, before any padded zero can advance the recurrent
        # state.  This is equivalent to a packed GRU while remaining friendly
        # to full-graph compilation.
        sequence, _ = self.slow_encoder(packed_inputs)
        last = (lengths - 1).clamp_min(0)
        state = sequence.gather(
            1,
            last[:, None, None].expand(-1, 1, sequence.shape[-1]),
        )[:, 0].reshape(batch_size, name_count, self.config.hidden_width)
        has_history = has_history.reshape(batch_size, name_count)
        # A name can enter today's strictly prior-session universe before it
        # has a rank-normalized row in the t-1 slow window.  The GRU state of
        # that genuinely empty sequence is its fixed zero initial state.  The
        # learned absent_state remains reserved for the optional fast stream.
        return torch.where(has_history[..., None], state, torch.zeros_like(state))

    def _current_states(
        self,
        current_features: torch.Tensor | None = None,
        current_feature_mask: torch.Tensor | None = None,
        current_feature_age_sessions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            current_features is None
            or current_feature_mask is None
            or current_feature_age_sessions is None
        ):
            raise ValueError("the current branch requires values, validity and ages")
        if current_features.ndim != 3:
            raise ValueError("current_features must have shape [batch, name, field]")
        if current_feature_mask.shape != current_features.shape:
            raise ValueError("current_feature_mask is misaligned with current_features")
        if current_feature_age_sessions.shape != current_features.shape:
            raise ValueError(
                "current feature ages are misaligned with current_features"
            )
        if current_features.shape[-1] != self.config.current_feature_count:
            raise ValueError(
                "current feature width differs from the model configuration"
            )
        valid = current_feature_mask.bool()
        clean = torch.where(valid, current_features, torch.zeros_like(current_features))
        bounded_age, age_known = _bounded_feature_age(
            current_feature_age_sessions.to(dtype=clean.dtype), valid
        )
        return self.current_input_norm(
            self.current_input_projection(
                torch.cat(
                    (
                        clean,
                        valid.to(clean.dtype),
                        bounded_age,
                        age_known.to(clean.dtype),
                    ),
                    dim=-1,
                )
            )
        )

    @staticmethod
    def _flags(
        reference: torch.Tensor,
        value: torch.Tensor | None,
        default: float,
    ) -> torch.Tensor:
        if value is None:
            return reference.new_full(reference.shape[:2], default)
        if value.ndim == 1:
            value = value[:, None].expand(-1, reference.shape[1])
        if value.shape != reference.shape[:2]:
            raise ValueError("sample flags must have shape [batch] or [batch, name]")
        return value.to(device=reference.device, dtype=reference.dtype)

    def _fast_states(
        self,
        slow: torch.Tensor,
        present: torch.Tensor,
        fast_patch_values: torch.Tensor | None,
        fast_patch_valid: torch.Tensor | None,
        fast_patch_mask: torch.Tensor | None,
        fast_name_index: torch.Tensor | None,
        fast_state_position: torch.Tensor | None,
        v1_equity_slow: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, name_count = slow.shape[:2]
        absent = self.absent_state.view(1, 1, -1).expand(batch_size, name_count, -1)
        if fast_patch_values is None:
            if (
                fast_patch_valid is not None
                or fast_patch_mask is not None
                or fast_name_index is not None
                or fast_state_position is not None
                or v1_equity_slow is not None
            ):
                raise ValueError("fast metadata was supplied without fast values")
            torch._assert_async(
                torch.all(~present.bool()),
                "fast_present cannot be true without the fast stream",
            )
            return absent
        if fast_patch_values.ndim != 4:
            raise ValueError(
                "fast values must have shape [batch, fast, patch, channel]"
            )
        if fast_patch_values.shape[0] != batch_size:
            raise ValueError("fast values are misaligned with the model batch")
        fast_count, patch_count = fast_patch_values.shape[1:3]
        if (
            fast_patch_valid is None
            or fast_patch_valid.shape != fast_patch_values.shape
            or fast_patch_mask is None
            or fast_patch_mask.shape != (batch_size, fast_count, patch_count)
            or fast_name_index is None
            or fast_name_index.shape != (batch_size, fast_count)
            or fast_state_position is None
            or fast_state_position.shape != (batch_size, fast_count)
        ):
            raise ValueError("compact fast values and metadata are misaligned")
        if self.fast_encoder.legacy_v1_context:
            if v1_equity_slow is None or v1_equity_slow.shape != (
                batch_size,
                fast_count,
                TCN_ARCHITECTURE.slow_width,
            ):
                raise ValueError("legacy fast slots require compact v1 equity slow")
        elif v1_equity_slow is not None and v1_equity_slow.shape != (
            batch_size,
            fast_count,
            TCN_ARCHITECTURE.slow_width,
        ):
            raise ValueError("compact v1 equity slow is misaligned")

        real_slot = fast_name_index >= 0
        if fast_count == 0:
            torch._assert_async(
                torch.all(~present.bool()),
                "fast_present cannot be true when every compact slot is padding",
            )
            return absent
        selected_names = fast_name_index.clamp_min(0).long()
        torch._assert_async(
            torch.all((~real_slot) | (selected_names < name_count)),
            "compact fast name index is outside the broad store axis",
        )
        occupancy = torch.zeros(
            batch_size,
            name_count,
            dtype=torch.long,
            device=fast_name_index.device,
        ).scatter_add(
            1,
            selected_names,
            real_slot.long(),
        )
        torch._assert_async(
            torch.all(occupancy <= 1),
            "compact fast name indices must be unique within each date",
        )
        derived_present = occupancy.reshape(batch_size, name_count).bool()
        torch._assert_async(
            torch.all(derived_present == present.bool()),
            "fast_present disagrees with compact fast name indices",
        )
        torch._assert_async(
            torch.all(
                real_slot
                | (
                    ~fast_patch_mask.bool().any(dim=-1)
                    & ~fast_patch_valid.bool().any(dim=(-1, -2))
                    & (fast_state_position == 0)
                )
            ),
            "padded compact fast slots must contain only zero metadata",
        )
        encoded = self.fast_encoder(
            fast_patch_values.reshape(batch_size * fast_count, 1, patch_count, -1),
            fast_patch_mask.reshape(batch_size * fast_count, 1, patch_count),
            (
                None
                if v1_equity_slow is None
                else v1_equity_slow.reshape(
                    batch_size * fast_count, 1, TCN_ARCHITECTURE.slow_width
                )
            ),
            fast_state_position.reshape(batch_size * fast_count),
            patch_valid=fast_patch_valid.reshape(
                batch_size * fast_count, 1, patch_count, -1
            ),
        ).reshape(batch_size, fast_count, _FAST_HIDDEN_WIDTH)
        encoded = torch.where(real_slot[..., None], encoded, torch.zeros_like(encoded))
        updates = torch.zeros_like(absent).scatter_add(
            1,
            selected_names[..., None].expand(-1, -1, _FAST_HIDDEN_WIDTH),
            encoded,
        )
        return torch.where(derived_present[..., None], updates, absent)

    def _legacy_dense_fast_states(
        self,
        slow: torch.Tensor,
        present: torch.Tensor,
        fast_patches: torch.Tensor,
        fast_patch_mask: torch.Tensor | None,
        fast_state_position: torch.Tensor | None,
        v1_equity_slow: torch.Tensor | None,
    ) -> torch.Tensor:
        """Diagnostic-only adapter for the superseded dense v1 input layout."""

        if not self.fast_encoder.legacy_v1_context:
            raise ValueError("dense fast patches are accepted only in legacy v1 mode")
        batch_size, name_count = slow.shape[:2]
        if (
            fast_patches.shape
            != (
                batch_size,
                name_count,
                FAST_REAL_PATCHES,
                _LEGACY_FAST_PATCH_WIDTH,
            )
            or fast_patch_mask is None
            or fast_patch_mask.shape != fast_patches.shape[:-1]
            or v1_equity_slow is None
            or v1_equity_slow.shape
            != (batch_size, name_count, TCN_ARCHITECTURE.slow_width)
        ):
            raise ValueError("dense legacy fast inputs are misaligned")
        if fast_state_position is None:
            positions = fast_patches.new_full(
                (batch_size, name_count),
                _V1_ABSOLUTE_STATE_POSITION,
                dtype=torch.long,
            )
        else:
            positions = fast_state_position.to(device=fast_patches.device).long()
            if positions.ndim == 1:
                positions = positions[:, None].expand(-1, name_count)
            if positions.shape != (batch_size, name_count):
                raise ValueError("dense legacy state positions are misaligned")
        compact_names = torch.arange(
            name_count, device=fast_patches.device, dtype=torch.long
        )[None, :].expand(batch_size, -1)
        compact_names = torch.where(
            present.bool(), compact_names, torch.full_like(compact_names, -1)
        )
        compact_mask = fast_patch_mask & present.bool()[..., None]
        patch_valid = compact_mask[..., None].expand_as(fast_patches)
        positions = torch.where(present.bool(), positions, torch.zeros_like(positions))
        compact_slow = torch.where(
            present.bool()[..., None], v1_equity_slow, torch.zeros_like(v1_equity_slow)
        )
        return self._fast_states(
            slow,
            present,
            fast_patches,
            patch_valid,
            compact_mask,
            compact_names,
            positions,
            compact_slow,
        )

    def forward(
        self,
        slow_features: torch.Tensor,
        slow_feature_mask: torch.Tensor,
        slow_history_mask: torch.Tensor,
        active_mask: torch.Tensor,
        fast_patches: torch.Tensor | None = None,
        fast_patch_mask: torch.Tensor | None = None,
        fast_present: torch.Tensor | None = None,
        fast_state_position: torch.Tensor | None = None,
        v1_equity_slow: torch.Tensor | None = None,
        *,
        current_features: torch.Tensor | None = None,
        current_feature_mask: torch.Tensor | None = None,
        slow_feature_age_sessions: torch.Tensor,
        current_feature_age_sessions: torch.Tensor | None = None,
        common_state_features: torch.Tensor | None = None,
        fast_patch_values: torch.Tensor | None = None,
        fast_patch_valid: torch.Tensor | None = None,
        fast_name_index: torch.Tensor | None = None,
        sidecars: Mapping[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor:
        if active_mask.shape != slow_features.shape[:2]:
            raise ValueError("active_mask is misaligned with the model rows")
        if self.config.disable_fast_stream:
            fast_patches = fast_patch_values = fast_patch_valid = None
            fast_patch_mask = fast_name_index = fast_state_position = None
            v1_equity_slow = None
            fast_present = torch.zeros_like(active_mask, dtype=slow_features.dtype)
        if fast_patches is not None and fast_patch_values is not None:
            raise ValueError(
                "native compact and dense legacy fast inputs cannot be mixed"
            )
        if fast_patch_values is not None:
            if (
                fast_patch_values.ndim != 4
                or fast_patch_values.shape[0] != slow_features.shape[0]
                or fast_name_index is None
                or fast_name_index.shape != fast_patch_values.shape[:2]
            ):
                raise ValueError("compact fast values and name indices are misaligned")
            real = fast_name_index >= 0
            names = fast_name_index.clamp_min(0).long()
            torch._assert_async(
                torch.all((~real) | (names < slow_features.shape[1])),
                "compact fast name index is outside the broad store axis",
            )
            derived_present = (
                torch.zeros(
                    slow_features.shape[:2],
                    dtype=torch.long,
                    device=slow_features.device,
                ).scatter_add(1, names, real.long())
                if names.shape[1]
                else torch.zeros(
                    slow_features.shape[:2],
                    dtype=torch.long,
                    device=slow_features.device,
                )
            )
            torch._assert_async(
                torch.all(derived_present <= 1),
                "compact fast name indices must be unique within each date",
            )
            present = (
                derived_present.to(slow_features.dtype)
                if fast_present is None
                else self._flags(slow_features, fast_present, 0.0)
            )
        else:
            inferred_present = 0.0 if fast_patches is None else 1.0
            present = self._flags(slow_features, fast_present, inferred_present)
        slow = self._slow_states(
            slow_features,
            slow_feature_mask,
            slow_history_mask,
            slow_feature_age_sessions,
        )
        if self.config.current_feature_count:
            current = self._current_states(
                current_features, current_feature_mask, current_feature_age_sessions
            )
            if current.shape[:2] != slow.shape[:2]:
                raise ValueError("current and slow feature axes are misaligned")
        if self.config.current_feature_count and fast_patches is not None:
            fast = self._legacy_dense_fast_states(
                slow,
                present,
                fast_patches,
                fast_patch_mask,
                fast_state_position,
                v1_equity_slow,
            )
        elif self.config.current_feature_count:
            fast = self._fast_states(
                slow,
                present,
                fast_patch_values,
                fast_patch_valid,
                fast_patch_mask,
                fast_name_index,
                fast_state_position,
                v1_equity_slow,
            )

        weights = active_mask.bool()[..., None]
        count = weights.sum(dim=1).clamp_min(1)
        mean = torch.where(weights, slow, torch.zeros_like(slow)).sum(dim=1) / count
        second = (
            torch.where(weights, slow.square(), torch.zeros_like(slow)).sum(dim=1)
            / count
        )
        dispersion = torch.sqrt(torch.clamp(second - mean.square(), min=1e-6))
        pooled = torch.cat((mean, dispersion), dim=-1)
        pooled = pooled[:, None].expand(-1, slow.shape[1], -1)

        gated_pool = (
            torch.sigmoid(self.pool_gate(torch.cat((slow, pooled), dim=-1))) * pooled
        )
        if self.config.current_feature_count:
            gated_fast = (
                torch.sigmoid(self.fast_gate(torch.cat((slow, fast), dim=-1))) * fast
            )
            fused = torch.cat(
                (slow, current, gated_fast, gated_pool, present[..., None]), dim=-1
            )
        else:
            fused = torch.cat((slow, gated_pool), dim=-1)
        if self.config.common_state_feature_count:
            if common_state_features is None or common_state_features.shape != (
                slow.shape[0],
                self.config.common_state_feature_count,
            ):
                raise ValueError(
                    "common-state fields must align with the batch date axis"
                )
            common = common_state_features[:, None, :].expand(-1, slow.shape[1], -1)
            fused = torch.cat((fused, common), dim=-1)
        hidden = self.fusion_projection(fused)
        supplied = {} if sidecars is None else sidecars
        if supplied.keys() != self.sidecar_projections.keys():
            raise ValueError("sidecar tensors differ from the configured families")
        for name, projection in self.sidecar_projections.items():
            values, valid, age = supplied[name]
            if (
                values.shape != valid.shape
                or values.shape != age.shape
                or values.shape[:2] != hidden.shape[:2]
                or values.shape[-1] * 3 != projection.shape[1]
            ):
                raise ValueError(
                    f"{name} sidecar values, masks and ages are misaligned"
                )
            valid = valid.bool() & active_mask.bool()[..., None]
            clean = torch.where(valid, values, torch.zeros_like(values))
            bounded_age, _ = _bounded_feature_age(age, valid)
            bounded_age = torch.where(valid, bounded_age, torch.zeros_like(bounded_age))
            inputs = torch.cat((clean, valid.to(clean.dtype), bounded_age), dim=-1)
            residual = torch.nn.functional.linear(inputs, projection)
            # Gate after cross-sectional pooling: an invalid name's own parent
            # representation cannot change through another name's sidecar.
            hidden = torch.where(
                valid.any(dim=-1, keepdim=True), hidden + residual, hidden
            )
        hidden = self.trunk(hidden)
        predictions = torch.cat(
            tuple(
                self.heads[name](hidden)
                for name in (*[f"d{x}" for x in (1, 2, 3, 5, 10)], "to_close")
            ),
            dim=-1,
        )
        return torch.where(
            active_mask.bool()[..., None], predictions, torch.zeros_like(predictions)
        )


def _initialize_module(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def count_non_fast_parameters(model: DailyMultiHorizonModel) -> int:
    fast_ids = (
        {id(parameter) for parameter in model.fast_encoder.parameters()}
        if model.config.current_feature_count
        else set()
    )
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in fast_ids
    )


def load_v1_fast_encoder(
    model: DailyMultiHorizonModel,
    checkpoint_path: Path,
    *,
    expected_sha256: str | None = None,
) -> frozenset[str]:
    if model.config.fast_encoder_mode != "legacy_v1_contaminated":
        raise ValueError("v1 checkpoints require the contaminated legacy fast mode")
    if not model.config.allow_contaminated_v1_initialization:
        raise ValueError("v1 initialization was not explicitly allowed")
    if expected_sha256 is None:
        raise ValueError("v1 fast initialization requires an expected SHA-256")
    payload_bytes = checkpoint_path.read_bytes()
    actual_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError("v1 checkpoint SHA-256 differs from the frozen manifest")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, Mapping):
        raise ValueError("v1 checkpoint does not contain a model state dictionary")
    source = {
        str(name).removeprefix("_orig_mod.").removeprefix("module."): value
        for name, value in payload.items()
        if isinstance(value, torch.Tensor)
    }
    target = model.fast_encoder.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    for name, expected in target.items():
        candidates = (name, f"fast_encoder.{name}")
        match = next((source[key] for key in candidates if key in source), None)
        if match is None:
            raise ValueError(f"v1 checkpoint is missing fast encoder tensor {name}")
        if match.shape != expected.shape:
            raise ValueError(f"v1 fast encoder tensor {name} has the wrong shape")
        loaded[name] = match
    model.fast_encoder.load_state_dict(loaded, strict=True)
    initialized = frozenset(
        f"fast_encoder.{name}" for name, _ in model.fast_encoder.named_parameters()
    )
    model.pretrained_parameter_names |= initialized
    model.fast_checkpoint_sha256 = actual_sha256
    model.fast_initialization_provenance = {
        "mode": "contaminated_v1_checkpoint",
        "contaminated": True,
        "explicitly_allowed": True,
        "checkpoint_sha256": actual_sha256,
    }
    return initialized
