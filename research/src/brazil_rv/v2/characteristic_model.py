"""Round-7 characteristic model: shared encoders, exact context, optional ensemble."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .contract import TARGETED_FUSION_GATE_BIAS
from .model import _bounded_feature_age, _initialize_module, encode_slow_history


@dataclass(frozen=True)
class CharacteristicConfig:
    family_counts: tuple[tuple[str, int], ...] = ()
    common_field_count: int = 0
    temporal: bool = True
    context: str = "pool"
    members: int = 1
    horizons: tuple[int, ...] = (3, 5, 10)
    slow_feature_count: int = 32
    lookback: int = 60
    hidden_width: int = 64
    width: int = 256
    inner_width: int = 256
    blocks: int = 3
    dropout: float = 0.1


def encode_values(values, valid, ages):
    """Three channels; unknown age remains distinct from an observed zero age."""
    clean = torch.where(valid, values, torch.zeros_like(values))
    age, known = _bounded_feature_age(ages, valid)
    age = torch.where(known, age, -torch.ones_like(age))
    return torch.cat((clean, valid.to(clean.dtype), age.to(clean.dtype)), dim=-1)


class EnsembleLinear(nn.Module):
    """One matrix with per-member input/output multipliers, no matrix copies."""

    def __init__(self, input_width: int, output_width: int, members: int):
        super().__init__()
        self.linear = nn.Linear(input_width, output_width)
        self.r = nn.Parameter(torch.empty(members, input_width))
        self.s = nn.Parameter(torch.empty(members, output_width))
        with torch.no_grad():
            self.r.bernoulli_(0.5).mul_(2).sub_(1)
            self.s.bernoulli_(0.5).mul_(2).sub_(1)

    def forward(self, values):
        return self.linear(values * self.r) * self.s


class CharacteristicBlock(nn.Module):
    def __init__(self, config: CharacteristicConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.width)
        if config.members == 1:
            self.up = nn.Linear(config.width, 2 * config.inner_width)
            self.down = nn.Linear(config.inner_width, config.width)
        else:
            self.up = EnsembleLinear(
                config.width, 2 * config.inner_width, config.members
            )
            self.down = EnsembleLinear(config.inner_width, config.width, config.members)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, values):
        left, gate = self.up(self.norm(values)).chunk(2, dim=-1)
        return values + self.dropout(self.down(left * F.silu(gate)))


class CrossStockContext(nn.Module):
    def __init__(self, width: int, kind: str, dropout: float):
        super().__init__()
        self.kind = kind
        self.dropout = dropout
        if kind == "attention":
            self.qkv = nn.Linear(width, 3 * 4 * 32)
            self.output = nn.Linear(4 * 32, width)
            self.norm = nn.LayerNorm(width)
        elif kind == "pool":
            self.gate = nn.Linear(3 * width, 2 * width)
            self.output = nn.Linear(3 * width, width)
        else:
            raise ValueError("context must be pool or attention")

    def forward(self, values, active):
        if self.kind == "attention":
            b, n, _ = values.shape
            qkv = self.qkv(values).reshape(b, n, 3, 4, 32).permute(2, 0, 3, 1, 4)
            context = (
                F.scaled_dot_product_attention(
                    qkv[0],
                    qkv[1],
                    qkv[2],
                    attn_mask=active[:, None, None, :],
                    dropout_p=self.dropout if self.training else 0.0,
                )
                .transpose(1, 2)
                .reshape(b, n, 128)
            )
            result = self.norm(values + self.output(context))
        else:
            # Reductions stay FP32 under BF16, as in the sealed S0 path.
            population = values.float()
            mask = active[..., None]
            count = mask.sum(dim=1).clamp_min(1)
            mean = torch.where(mask, population, 0.0).sum(dim=1) / count
            second = torch.where(mask, population.square(), 0.0).sum(dim=1) / count
            dispersion = (second - mean.square()).clamp_min(1e-6).sqrt()
            pooled = torch.cat((mean, dispersion), dim=-1)[:, None].expand(
                -1, values.shape[1], -1
            )
            joint = torch.cat((values, pooled), dim=-1)
            gated = torch.sigmoid(self.gate(joint)) * pooled
            result = self.output(torch.cat((values, gated), dim=-1))
        return torch.where(active[..., None], result, 0.0)


class CharacteristicModel(nn.Module):
    """Return [date, name, member, head]; encoders execute once per stock/date."""

    def __init__(self, config: CharacteristicConfig):
        super().__init__()
        self.config = config
        if config.temporal:
            self.temporal_config = ModelConfig(
                slow_feature_count=config.slow_feature_count,
                current_feature_count=0,
                disable_fast_stream=True,
                slow_lookback=config.lookback,
                hidden_width=config.hidden_width,
            )
            self.slow_input_projection = nn.Linear(
                4 * config.slow_feature_count, config.hidden_width
            )
            self.slow_input_norm = nn.LayerNorm(config.hidden_width)
            self.slow_encoder = nn.GRU(
                config.hidden_width, config.hidden_width, batch_first=True
            )
        self.core = nn.Sequential(
            nn.Linear(3 * config.slow_feature_count, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
        )
        self.families = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(3 * size, 64),
                    nn.GELU(),
                    nn.Linear(64, 32),
                    nn.LayerNorm(32),
                )
                for name, size in config.family_counts
            }
        )
        joint_width = (
            (config.hidden_width if config.temporal else 0)
            + 128
            + 33 * len(config.family_counts)
        )
        self.joint = nn.Sequential(
            nn.Linear(joint_width, config.width), nn.LayerNorm(config.width)
        )
        self.film = (
            nn.Sequential(
                nn.Linear(3 * config.common_field_count, 64),
                nn.GELU(),
                nn.Linear(64, 2 * config.width),
            )
            if config.common_field_count
            else None
        )
        self.context = CrossStockContext(config.width, config.context, config.dropout)
        self.trunk = nn.Sequential(
            *(CharacteristicBlock(config) for _ in range(config.blocks))
        )
        self.head = nn.Linear(config.width, len(config.horizons))
        self.apply(_initialize_module)
        if self.film is not None:
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)
        if config.context == "pool":
            nn.init.zeros_(self.context.gate.weight)
            nn.init.constant_(self.context.gate.bias, TARGETED_FUSION_GATE_BIAS)

    def forward(
        self,
        slow_features,
        slow_feature_mask,
        slow_history_mask,
        active_mask,
        *,
        slow_feature_age_sessions,
        sidecars=None,
        common_state=None,
    ):
        active = active_mask.bool()
        parts = []
        if self.config.temporal:
            parts.append(
                encode_slow_history(
                    slow_features,
                    slow_feature_mask,
                    slow_history_mask,
                    slow_feature_age_sessions,
                    config=self.temporal_config,
                    input_projection=self.slow_input_projection,
                    input_norm=self.slow_input_norm,
                    encoder=self.slow_encoder,
                )
            )
        # The final permitted decision row is already aligned by the store;
        # no consumer-side extra lag, and no substitution of an older valid row.
        core_valid = slow_feature_mask[..., -1, :].bool() & active[..., None]
        parts.append(
            self.core(
                encode_values(
                    slow_features[..., -1, :],
                    core_valid,
                    slow_feature_age_sessions[..., -1, :],
                )
            )
        )
        supplied = {} if sidecars is None else sidecars
        for name, encoder in self.families.items():
            values, valid, age = supplied[name]
            valid = valid.bool() & active[..., None]
            present = valid.any(dim=-1, keepdim=True)
            encoded = encoder(encode_values(values, valid, age))
            parts.extend(
                (torch.where(present, encoded, 0.0), present.to(encoded.dtype))
            )
        state = self.joint(torch.cat(parts, dim=-1))
        if self.film is not None:
            common_values, common_valid, common_ages = common_state
            gamma, beta = self.film(
                encode_values(common_values, common_valid, common_ages)
            ).chunk(2, dim=-1)
            state = state * (1 + gamma[:, None]) + beta[:, None]
        state = self.context(state, active)
        # Only the trunk is ensembled; no repeated 60-session GRU computation.
        state = state.unsqueeze(2).expand(-1, -1, self.config.members, -1)
        scores = self.head(self.trunk(state))
        return torch.where(active[..., None, None], scores, 0.0)
