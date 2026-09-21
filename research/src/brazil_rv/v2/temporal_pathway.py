"""Matched early/late peer interaction on dated historical representations."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class HistoricalAttentionBlock(nn.Module):
    """One masked temporal attention and feed-forward residual block."""

    def __init__(self, width, dropout):
        super().__init__()
        self.attention = PeerAttention(width, dropout)
        self.norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
            nn.Dropout(dropout),
        )

    def forward(self, values, valid):
        shape = values.shape
        values = self.attention(
            values.reshape(-1, shape[-2], shape[-1]), valid.reshape(-1, shape[-2])
        )
        values = values + self.ffn(self.norm(values))
        return torch.where(valid[..., None], values.reshape(shape), 0.0)


class HistoricalAttention(HistoricalAttentionBlock):
    """Bidirectional within the available window, never across decision dates."""

    def __init__(self, width, lookback, dropout, layers=1):
        super().__init__(width, dropout)
        positions = torch.arange(lookback)[:, None]
        frequencies = torch.exp(torch.arange(0, width, 2) * (-math.log(10000) / width))
        encoding = torch.zeros(lookback, width)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("positions", encoding)
        self.additional_layers = nn.ModuleList(
            HistoricalAttentionBlock(width, dropout) for _ in range(layers - 1)
        )

    def forward(self, values, valid):
        values = super().forward(values + self.positions, valid)
        for layer in self.additional_layers:
            values = layer(values, valid)
        return values


class PeerAttention(nn.Module):
    """Shared exact attention; padding never acts as a key or output query."""

    def __init__(self, width, dropout):
        super().__init__()
        self.dropout = dropout
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.output_norm = nn.LayerNorm(width)

    def forward(self, values, valid):
        batch, names, width = values.shape
        q, k, v = (
            self.qkv(self.norm(values))
            .reshape(batch, names, 3, 4, width // 4)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        result = (
            F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=valid[:, None, None, :],
                dropout_p=self.dropout if self.training else 0.0,
            )
            .transpose(1, 2)
            .reshape(batch, names, width)
        )
        result = self.output_norm(values + self.output(result))
        return torch.where(valid[..., None], result, 0.0)

    def initialize_attention(self):
        # The global .02 initializer makes this width-64 residual attention
        # route almost uniform and very weak. Width-scaled Xavier matrices
        # restore trainable peer routing in the controlled unseen-date test.
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.output.weight)


class TemporalPeerPathway(nn.Module):
    """Matched pooling with early, late or no additional peer interaction."""

    def __init__(self, width, dropout, timing):
        super().__init__()
        self.timing = timing
        if timing != "pool":
            self.peer = PeerAttention(width, dropout)
        self.query = nn.Linear(width, width, bias=False)

    def pool(self, sequence, valid):
        # Finite all-masked rows yield exactly zero and finite gradients.
        logits = (
            sequence.float() * self.query(sequence[..., -1, :]).float()[..., None, :]
        ).sum(-1)
        weights = torch.softmax(logits.masked_fill(~valid, -1e9), dim=-1)
        weights = weights * valid
        return (weights[..., None] * sequence.float()).sum(-2).to(sequence.dtype)

    def forward(self, sequence, history, active):
        batch, names, steps, width = sequence.shape
        valid = history & active[..., None]
        sequence = torch.where(valid[..., None], sequence, 0.0)
        if self.timing == "early":
            sequence = (
                self.peer(
                    sequence.transpose(1, 2).reshape(batch * steps, names, width),
                    valid.transpose(1, 2).reshape(batch * steps, names),
                )
                .reshape(batch, steps, names, width)
                .transpose(1, 2)
            )
            return self.pool(sequence, valid)
        pooled = self.pool(sequence, valid)
        return pooled if self.timing == "pool" else self.peer(pooled, valid.any(-1))
