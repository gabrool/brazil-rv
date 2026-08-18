from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MuonLinear(nn.Linear):
    pass


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden_width: int) -> None:
        super().__init__()
        self.gate = MuonLinear(width, hidden_width, bias=False)
        self.up = MuonLinear(width, hidden_width, bias=False)
        self.down = MuonLinear(hidden_width, width, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(inputs)) * self.up(inputs))


class CausalTCNResidualBlock(nn.Module):
    def __init__(
        self,
        width: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        hidden_width: int,
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.convolution = nn.Conv1d(
            width,
            width,
            kernel_size,
            dilation=dilation,
            padding=0,
            bias=True,
        )
        self.norm = nn.LayerNorm(width)
        self.swiglu = SwiGLU(width, hidden_width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.convolution(F.pad(inputs, (self.left_padding, 0)))
        hidden = self.norm(hidden.transpose(1, 2))
        hidden = self.dropout(self.swiglu(hidden))
        return inputs + hidden.transpose(1, 2)
