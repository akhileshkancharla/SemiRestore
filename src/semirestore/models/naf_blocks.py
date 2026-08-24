"""Activation-free NAF building blocks for efficient image restoration.

Adapted from ``src/semirestore/models/naf_blocks.py`` at upstream training
revision ``d037473ddf4a3cd20eb3fef933991cd66749f4f2``. Packaging and public
exports were adapted for this repository; parameter names and numerical
operations are preserved for checkpoint compatibility.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    """Apply per-pixel channel layer normalization to an NCHW tensor."""

    def __init__(self, channels: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.channels = channels
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.channels:
            raise ValueError(
                f"LayerNorm2d expected N,{self.channels},H,W; got {tuple(inputs.shape)}"
            )
        normalized = F.layer_norm(
            inputs.permute(0, 2, 3, 1),
            (self.channels,),
            self.weight,
            self.bias,
            self.epsilon,
        )
        return normalized.permute(0, 3, 1, 2).contiguous()


class SimpleGate(nn.Module):
    """Split channels evenly and multiply the two halves elementwise."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[1] % 2:
            raise ValueError("SimpleGate requires an even channel count")
        first, second = inputs.chunk(2, dim=1)
        return first * second


class NAFBlock(nn.Module):
    """NAFNet-style residual block with zero-initialized residual scales."""

    def __init__(
        self,
        channels: int,
        *,
        depthwise_expand: int = 2,
        ffn_expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        depthwise_channels = channels * depthwise_expand
        ffn_channels = channels * ffn_expand
        if channels < 1 or depthwise_channels % 2 or ffn_channels % 2:
            raise ValueError("NAFBlock expansion channels must be positive and even")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, depthwise_channels, 1)
        self.depthwise = nn.Conv2d(
            depthwise_channels,
            depthwise_channels,
            3,
            padding=1,
            groups=depthwise_channels,
        )
        self.gate1 = SimpleGate()
        gated_channels = depthwise_channels // 2
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(gated_channels, gated_channels, 1),
        )
        self.conv2 = nn.Conv2d(gated_channels, channels, 1)
        self.dropout1 = nn.Dropout(dropout) if dropout else nn.Identity()

        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, ffn_channels, 1)
        self.gate2 = SimpleGate()
        self.conv4 = nn.Conv2d(ffn_channels // 2, channels, 1)
        self.dropout2 = nn.Dropout(dropout) if dropout else nn.Identity()

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.conv1(self.norm1(inputs))
        features = self.gate1(self.depthwise(features))
        features = features * self.channel_attention(features)
        features = self.dropout1(self.conv2(features))
        first_residual = inputs + features * self.beta

        features = self.conv3(self.norm2(first_residual))
        features = self.dropout2(self.conv4(self.gate2(features)))
        return first_residual + features * self.gamma


__all__ = ["LayerNorm2d", "NAFBlock", "SimpleGate"]
