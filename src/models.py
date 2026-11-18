"""DeepLOB (Zhang et al., 2018) and a CNN-Transformer variant with the same CNN front-end.

Both models map a window of LOB snapshots, shape (B, 1, T=100, 40), to logits over
(up, stationary, down). They differ only in the temporal block applied to the (B, T, 96)
sequence produced by the shared front-end: an LSTM(64) versus a Transformer encoder.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _conv(c_in: int, c_out: int, kernel: tuple[int, int], stride: tuple[int, int] = (1, 1)):
    """Conv2d + LeakyReLU(0.01). Temporal kernels (k, 1) use 'same' padding along time,
    with the extra row at the end for even k (as in the original Keras implementation)."""
    k = kernel[0]
    pad = nn.ZeroPad2d((0, 0, (k - 1) // 2, k // 2)) if k > 1 else nn.Identity()
    return nn.Sequential(pad, nn.Conv2d(c_in, c_out, kernel, stride), nn.LeakyReLU(0.01))


class ConvFrontEnd(nn.Module):
    """DeepLOB convolutional blocks and inception module: (B, 1, T, 40) -> (B, T, 96)."""

    def __init__(self, filters: int = 16, inception: int = 32) -> None:
        super().__init__()
        f = filters
        self.conv = nn.Sequential(
            # Block 1: (1x2, stride 1x2) merges price and volume at each level/side: 40 -> 20.
            _conv(1, f, (1, 2), (1, 2)), _conv(f, f, (4, 1)), _conv(f, f, (4, 1)),
            # Block 2: merges ask and bid at each level: 20 -> 10.
            _conv(f, f, (1, 2), (1, 2)), _conv(f, f, (4, 1)), _conv(f, f, (4, 1)),
            # Block 3: integrates the 10 levels: 10 -> 1.
            _conv(f, f, (1, 10)), _conv(f, f, (4, 1)), _conv(f, f, (4, 1)),
        )  # fmt: skip
        self.branch3 = nn.Sequential(
            _conv(f, inception, (1, 1)), _conv(inception, inception, (3, 1))
        )
        self.branch5 = nn.Sequential(
            _conv(f, inception, (1, 1)), _conv(inception, inception, (5, 1))
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=1, padding=(1, 0)), _conv(f, inception, (1, 1))
        )
        self.out_dim = 3 * inception

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)  # (B, 16, T, 1)
        x = torch.cat([self.branch3(x), self.branch5(x), self.branch_pool(x)], dim=1)
        return x.squeeze(-1).transpose(1, 2)  # (B, T, 96)


class DeepLOB(nn.Module):
    """CNN front-end -> LSTM(64) -> linear classifier on the last hidden state (~61k params)."""

    def __init__(self, hidden: int = 64, n_classes: int = 3) -> None:
        super().__init__()
        self.cnn = ConvFrontEnd()
        self.lstm = nn.LSTM(self.cnn.out_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(self.cnn(x))
        return self.head(h[:, -1])  # logits; softmax is applied inside the loss


def sinusoidal_positions(length: int, dim: int) -> torch.Tensor:
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017), shape (length, dim)."""
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    freq = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe


class CNNTransformer(nn.Module):
    """CNN front-end -> linear projection + sinusoidal positions -> Transformer encoder.

    The defaults (d_model 64, 4 heads, 1 post-LN layer, FFN 128) give ~59k parameters,
    matching DeepLOB's ~61k, so the comparison isolates the temporal block, not capacity.
    Attention is bidirectional (no causal mask): the whole window precedes the label, so
    there is no look-ahead. As in DeepLOB, the output at the last time step is classified.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 1,
        ff_dim: int = 128,
        dropout: float = 0.1,
        window: int = 100,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self.cnn = ConvFrontEnd()
        self.proj = nn.Linear(self.cnn.out_dim, d_model)
        self.register_buffer("pos", sinusoidal_positions(window, d_model), persistent=False)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(self.cnn(x)) + self.pos[: x.shape[2]]
        return self.head(self.encoder(z)[:, -1])


MODELS = {"deeplob": DeepLOB, "transformer": CNNTransformer}


def count_params(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
