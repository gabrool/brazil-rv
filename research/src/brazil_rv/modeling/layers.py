from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class MuonLinear(nn.Linear):
    pass


def qk_rms_normalize(tensor: torch.Tensor, eps: float) -> torch.Tensor:
    scale = torch.rsqrt(tensor.float().square().mean(dim=-1, keepdim=True) + eps).to(
        tensor.dtype
    )
    return tensor * scale


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_positions: int, base: float) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        inverse_frequency = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_positions, dtype=torch.float32)
        angles = torch.outer(positions, inverse_frequency)
        self.register_buffer(
            "cosine", angles.cos().repeat_interleave(2, dim=-1), persistent=False
        )
        self.register_buffer(
            "sine", angles.sin().repeat_interleave(2, dim=-1), persistent=False
        )

    @staticmethod
    def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
        pairs = tensor.reshape(*tensor.shape[:-1], -1, 2)
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1)
        return rotated.flatten(-2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cosine = self.cosine[position_ids].unsqueeze(1).to(query.dtype)
        sine = self.sine[position_ids].unsqueeze(1).to(query.dtype)
        return (
            query * cosine + self._rotate_half(query) * sine,
            key * cosine + self._rotate_half(key) * sine,
        )


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        qk_norm_eps: float,
        *,
        rope: RotaryEmbedding | None,
    ) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by attention heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.qk_norm_eps = qk_norm_eps
        self.rope = rope
        self.query = MuonLinear(d_model, d_model, bias=False)
        self.key = MuonLinear(d_model, d_model, bias=False)
        self.value = MuonLinear(d_model, d_model, bias=False)
        self.output = MuonLinear(d_model, d_model, bias=False)

    def forward(
        self,
        inputs: torch.Tensor,
        key_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, d_model = inputs.shape

        def project(layer: MuonLinear) -> torch.Tensor:
            return (
                layer(inputs)
                .reshape(
                    batch_size,
                    sequence_length,
                    self.heads,
                    self.head_dim,
                )
                .transpose(1, 2)
            )

        query = qk_rms_normalize(project(self.query), self.qk_norm_eps)
        key = qk_rms_normalize(project(self.key), self.qk_norm_eps)
        value = project(self.value)
        if self.rope is not None:
            if position_ids is None:
                raise ValueError("Temporal attention requires RoPE position IDs")
            query, key = self.rope(query, key, position_ids)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=key_mask[:, None, None, :],
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=False,
        )
        return self.output(
            attended.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, sequence_length, d_model)
        )


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_width: int) -> None:
        super().__init__()
        self.gate = MuonLinear(d_model, hidden_width, bias=False)
        self.up = MuonLinear(d_model, hidden_width, bias=False)
        self.down = MuonLinear(hidden_width, d_model, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(inputs)) * self.up(inputs))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        hidden_width: int,
        norm_eps: float,
        qk_norm_eps: float,
        residual_dropout: float,
        *,
        rope: RotaryEmbedding | None,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.attention = MultiHeadAttention(d_model, heads, qk_norm_eps, rope=rope)
        self.feedforward_norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.feedforward = SwiGLU(d_model, hidden_width)
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        key_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs = inputs + self.residual_dropout(
            self.attention(self.attention_norm(inputs), key_mask, position_ids)
        )
        return inputs + self.residual_dropout(
            self.feedforward(self.feedforward_norm(inputs))
        )


class TransformerStack(nn.Module):
    def __init__(
        self,
        depth: int,
        d_model: int,
        heads: int,
        hidden_width: int,
        norm_eps: float,
        qk_norm_eps: float,
        residual_dropout: float,
        *,
        rope: RotaryEmbedding | None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    heads,
                    hidden_width,
                    norm_eps,
                    qk_norm_eps,
                    residual_dropout,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.RMSNorm(d_model, eps=norm_eps)

    def scale_residual_weights(self) -> None:
        scale = 1.0 / math.sqrt(2.0 * len(self.blocks))
        with torch.no_grad():
            for block in self.blocks:
                block.attention.output.weight.mul_(scale)
                block.feedforward.down.weight.mul_(scale)

    def forward(
        self,
        inputs: torch.Tensor,
        key_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            inputs = block(inputs, key_mask, position_ids)
        return self.final_norm(inputs)
