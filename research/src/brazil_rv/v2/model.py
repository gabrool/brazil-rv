from __future__ import annotations

import hashlib
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

_FAST_PATCH_WIDTH = TCN_ARCHITECTURE.patch_input_width
_FAST_HIDDEN_WIDTH = TCN_ARCHITECTURE.width
_V1_EQUITY_PREFIX_PATCHES = 12
_V1_ABSOLUTE_STATE_POSITION = _V1_EQUITY_PREFIX_PATCHES + FAST_REAL_PATCHES


class FastTCNEncoder(nn.Module):
    """The exact deployed store-v2 v1 per-equity state at the 15:45 cutoff."""

    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(
            _FAST_PATCH_WIDTH, _FAST_HIDDEN_WIDTH, bias=False
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
        self.slow_projection = nn.Linear(
            TCN_ARCHITECTURE.slow_width, _FAST_HIDDEN_WIDTH, bias=False
        )
        self.state_norm = nn.LayerNorm(_FAST_HIDDEN_WIDTH)
        keep = torch.ones(TCN_ARCHITECTURE.slow_width, dtype=torch.float32)
        keep[list(V1_STORE_V2_ZERO_SLOW_FIELDS)] = 0.0
        self.register_buffer("slow_keep_mask", keep, persistent=False)
        self.apply(_initialize_module)

    def forward(
        self,
        patches: torch.Tensor,
        patch_mask: torch.Tensor,
        v1_equity_slow: torch.Tensor,
        state_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if patches.ndim != 4:
            raise ValueError("patches must have shape [batch, name, patch, field]")
        if patch_mask.shape != patches.shape[:-1]:
            raise ValueError("patch_mask is misaligned with patches")
        if patches.shape[-1] != _FAST_PATCH_WIDTH:
            raise ValueError("patch width differs from the frozen fast encoder")
        batch_size, name_count, patch_count, _ = patches.shape
        if patch_count != FAST_REAL_PATCHES:
            raise ValueError("the fast stream must end at synthetic cutoff index 345")
        if v1_equity_slow.shape != (
            batch_size,
            name_count,
            TCN_ARCHITECTURE.slow_width,
        ):
            raise ValueError(
                "v1_equity_slow must have shape [batch, name, 32]"
            )
        prefix = patches.new_zeros(
            batch_size,
            name_count,
            _V1_EQUITY_PREFIX_PATCHES,
            _FAST_PATCH_WIDTH,
        )
        prefix_mask = patch_mask.new_zeros(
            batch_size, name_count, _V1_EQUITY_PREFIX_PATCHES
        )
        absolute_patches = torch.cat((prefix, patches), dim=2)
        absolute_mask = torch.cat((prefix_mask, patch_mask), dim=2)
        masked = absolute_patches * absolute_mask[..., None].to(patches.dtype)
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
                _V1_ABSOLUTE_STATE_POSITION - 1,
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
        torch._assert_async(
            torch.all(last == _V1_ABSOLUTE_STATE_POSITION - 1),
            "fast_state_position must identify absolute v1 position 81",
        )
        index = last[..., None, None].expand(-1, -1, 1, _FAST_HIDDEN_WIDTH)
        raw = sequence.gather(2, index).squeeze(2)
        neutralized_slow = v1_equity_slow * self.slow_keep_mask.to(
            dtype=v1_equity_slow.dtype
        )
        return self.state_norm(raw + self.slow_projection(neutralized_slow))


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
        self.slow_input_norm = nn.LayerNorm(config.slow_feature_count)
        self.slow_encoder = nn.GRU(
            config.slow_feature_count + 2,
            config.hidden_width,
            num_layers=config.gru_layers,
            batch_first=True,
            dropout=config.dropout if config.gru_layers == 2 else 0.0,
        )
        if config.hidden_width != _FAST_HIDDEN_WIDTH:
            raise ValueError("The starter model requires width 64 for v1 fast transfer")
        self.fast_encoder = FastTCNEncoder()
        self.absent_state = nn.Parameter(torch.zeros(_FAST_HIDDEN_WIDTH))
        self.fast_gate = nn.Linear(
            config.hidden_width + _FAST_HIDDEN_WIDTH,
            _FAST_HIDDEN_WIDTH,
        )
        self.pool_gate = nn.Linear(
            config.hidden_width + 2 * config.hidden_width,
            2 * config.hidden_width,
        )
        fusion_input_width = (
            config.hidden_width + _FAST_HIDDEN_WIDTH + 2 * config.hidden_width + 2
        )
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
        fast_ids = {id(parameter) for parameter in self.fast_encoder.parameters()}
        non_fast_count = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in fast_ids
        )
        if non_fast_count > 150_000:
            raise ValueError("starter-model non-fast parameter count exceeds 150k")

    def _slow_states(
        self,
        slow_features: torch.Tensor,
        slow_history_mask: torch.Tensor,
        active_mask: torch.Tensor,
        fast_present: torch.Tensor,
        days_since_last_slow_row: torch.Tensor,
    ) -> torch.Tensor:
        if slow_features.ndim != 4:
            raise ValueError("slow_features must have shape [batch, name, date, field]")
        if slow_history_mask.shape != slow_features.shape[:-1]:
            raise ValueError("slow_history_mask is misaligned with slow_features")
        if slow_features.shape[-1] != self.config.slow_feature_count:
            raise ValueError("slow feature width differs from the model configuration")
        batch_size, name_count, lookback, _ = slow_features.shape
        if lookback != self.config.slow_lookback:
            raise ValueError("slow lookback differs from the model configuration")
        valid = slow_history_mask.bool()
        clean = torch.where(
            valid[..., None], slow_features, torch.zeros_like(slow_features)
        )
        normalized = self.slow_input_norm(clean)
        flags = torch.stack(
            (
                fast_present.to(normalized.dtype),
                days_since_last_slow_row.to(normalized.dtype),
            ),
            dim=-1,
        )
        flags = flags[:, :, None].expand(-1, -1, lookback, -1)
        inputs = torch.cat((normalized, flags), dim=-1)
        inputs = torch.where(valid[..., None], inputs, torch.zeros_like(inputs))
        sequence, _ = self.slow_encoder(
            inputs.reshape(batch_size * name_count, lookback, -1)
        )
        sequence = sequence.reshape(
            batch_size, name_count, lookback, self.config.hidden_width
        )
        positions = torch.arange(lookback, device=slow_features.device)
        last = torch.where(valid, positions, -1).amax(dim=-1)
        torch._assert_async(
            torch.all((last >= 0) | ~active_mask.bool()),
            "Every active model row needs at least one slow observation",
        )
        last = last.clamp_min(0)
        index = last[..., None, None].expand(-1, -1, 1, self.config.hidden_width)
        return sequence.gather(2, index).squeeze(2)

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
        fast_patches: torch.Tensor | None,
        fast_patch_mask: torch.Tensor | None,
        fast_state_position: torch.Tensor | None,
        v1_equity_slow: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, name_count = slow.shape[:2]
        absent = self.absent_state.view(1, 1, -1).expand_as(slow)
        if fast_patches is None:
            if (
                fast_patch_mask is not None
                or fast_state_position is not None
                or v1_equity_slow is not None
            ):
                raise ValueError("fast metadata was supplied without fast patches")
            torch._assert_async(
                torch.all(~present.bool()),
                "fast_present cannot be true without the v1 fast stream",
            )
            return absent

        if fast_patches.ndim != 4:
            raise ValueError("patches must have shape [batch, name, patch, field]")
        if fast_patches.shape[:2] != (batch_size, name_count):
            raise ValueError("fast patches are misaligned with the model rows")
        if fast_patch_mask is None:
            raise ValueError("fast_patch_mask is required with fast patches")
        if fast_patch_mask.shape != fast_patches.shape[:-1]:
            raise ValueError("patch_mask is misaligned with patches")
        if fast_patches.shape[2:] != (FAST_REAL_PATCHES, _FAST_PATCH_WIDTH):
            if fast_patches.shape[-1] != _FAST_PATCH_WIDTH:
                raise ValueError("patch width differs from the frozen fast encoder")
            raise ValueError("the fast stream must end at synthetic cutoff index 345")
        if v1_equity_slow is None:
            raise ValueError("v1_equity_slow is required with fast patches")
        if v1_equity_slow.shape != (
            batch_size,
            name_count,
            TCN_ARCHITECTURE.slow_width,
        ):
            raise ValueError("v1_equity_slow must have shape [batch, name, 32]")

        dense_state_position: torch.Tensor | None = None
        if fast_state_position is not None:
            dense_state_position = fast_state_position.to(
                device=fast_patches.device, dtype=torch.long
            )
            if dense_state_position.ndim == 1:
                dense_state_position = dense_state_position[:, None].expand(
                    -1, name_count
                )
            if dense_state_position.shape != (batch_size, name_count):
                raise ValueError(
                    "state_position must have shape [batch] or [batch, name]"
                )
            torch._assert_async(
                torch.all(dense_state_position == _V1_ABSOLUTE_STATE_POSITION),
                "fast_state_position must identify absolute v1 position 81",
            )

        # The fast stream is available for only a small, dynamic PIT subset of
        # the dense daily panel.  Flatten names into independent one-name rows
        # so the TCN retains its exact per-instrument computation while avoiding
        # activations for absent names.
        flat_present = present.bool().reshape(-1)
        present_indices = torch.nonzero(flat_present, as_tuple=False).squeeze(1)
        selected_patches = fast_patches.reshape(
            batch_size * name_count, FAST_REAL_PATCHES, _FAST_PATCH_WIDTH
        ).index_select(0, present_indices)
        selected_mask = fast_patch_mask.reshape(
            batch_size * name_count, FAST_REAL_PATCHES
        ).index_select(0, present_indices)
        selected_slow = v1_equity_slow.reshape(
            batch_size * name_count, TCN_ARCHITECTURE.slow_width
        ).index_select(0, present_indices)
        selected_position = (
            None
            if dense_state_position is None
            else dense_state_position.reshape(-1).index_select(0, present_indices)
        )
        encoded = self.fast_encoder(
            selected_patches[:, None],
            selected_mask[:, None],
            selected_slow[:, None],
            selected_position,
        ).squeeze(1)
        flat_fast = absent.reshape(batch_size * name_count, _FAST_HIDDEN_WIDTH)
        return flat_fast.index_copy(0, present_indices, encoded).reshape_as(absent)

    def forward(
        self,
        slow_features: torch.Tensor,
        slow_history_mask: torch.Tensor,
        active_mask: torch.Tensor,
        fast_patches: torch.Tensor | None = None,
        fast_patch_mask: torch.Tensor | None = None,
        fast_present: torch.Tensor | None = None,
        days_since_last_slow_row: torch.Tensor | None = None,
        fast_state_position: torch.Tensor | None = None,
        v1_equity_slow: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if active_mask.shape != slow_features.shape[:2]:
            raise ValueError("active_mask is misaligned with the model rows")
        inferred_present = 0.0 if fast_patches is None else 1.0
        present = self._flags(slow_features, fast_present, inferred_present)
        days = self._flags(slow_features, days_since_last_slow_row, 0.0)
        slow = self._slow_states(
            slow_features, slow_history_mask, active_mask, present, days
        )
        fast = self._fast_states(
            slow,
            present,
            fast_patches,
            fast_patch_mask,
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

        gated_fast = (
            torch.sigmoid(self.fast_gate(torch.cat((slow, fast), dim=-1))) * fast
        )
        gated_pool = (
            torch.sigmoid(self.pool_gate(torch.cat((slow, pooled), dim=-1))) * pooled
        )
        fused = torch.cat(
            (slow, gated_fast, gated_pool, present[..., None], days[..., None]), dim=-1
        )
        hidden = self.trunk(self.fusion_projection(fused))
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
    fast_ids = {id(parameter) for parameter in model.fast_encoder.parameters()}
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
    return initialized
