"""Statistics-conditioned NAF encoder-decoder with a conservative 2x SR head.

Adapted from ``src/semirestore/models/naf_sr.py`` at upstream training revision
``d037473ddf4a3cd20eb3fef933991cd66749f4f2``. Documentation and package
exports were adapted; module names, parameter shapes, conditioning order, and
forward operations are preserved for checkpoint compatibility.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .naf_blocks import NAFBlock


class ConditioningStatisticsError(ValueError):
    """Raised when an explicit global-conditioning override is invalid."""


def compute_conditioning_statistics(inputs: torch.Tensor) -> torch.Tensor:
    """Return batch mean/std/min/max in the checkpoint-compatible order."""

    if not isinstance(inputs, torch.Tensor) or inputs.ndim != 4:
        raise ConditioningStatisticsError("Conditioning input must be a rank-4 tensor")
    if not torch.is_floating_point(inputs) or inputs.is_complex():
        raise ConditioningStatisticsError("Conditioning input must be floating point")
    if inputs.numel() == 0 or not bool(torch.isfinite(inputs).all().item()):
        raise ConditioningStatisticsError("Conditioning input must be nonempty and finite")
    flattened = inputs.flatten(2)
    return torch.cat(
        (
            flattened.mean(2),
            flattened.std(2, unbiased=False),
            flattened.amin(2),
            flattened.amax(2),
        ),
        dim=1,
    )


def validate_conditioning_statistics(
    statistics: object,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Validate an explicit mean/std/min/max override for one input batch."""

    if not isinstance(statistics, torch.Tensor):
        raise ConditioningStatisticsError("Conditioning statistics must be a PyTorch tensor")
    expected_shape = (inputs.shape[0], 4)
    if statistics.layout != torch.strided or tuple(statistics.shape) != expected_shape:
        raise ConditioningStatisticsError(
            f"Conditioning statistics must have dense shape {expected_shape}"
        )
    if statistics.dtype != inputs.dtype:
        raise ConditioningStatisticsError("Conditioning statistics dtype must match the input")
    if statistics.device != inputs.device:
        raise ConditioningStatisticsError("Conditioning statistics device must match the input")
    if not bool(torch.isfinite(statistics).all().item()):
        raise ConditioningStatisticsError("Conditioning statistics must be finite")
    if bool((statistics[:, 1] < 0).any().item()):
        raise ConditioningStatisticsError("Conditioning standard deviation cannot be negative")
    return statistics


def _block_stack(channels: int, count: int, dropout: float) -> nn.Sequential:
    if count < 1:
        raise ValueError("Every NAF stage needs at least one block")
    return nn.Sequential(*(NAFBlock(channels, dropout=dropout) for _ in range(count)))


class NAFSR(nn.Module):
    """Restore one-channel images with optional internal statistic conditioning."""

    scale: int = 2

    def __init__(
        self,
        *,
        width: int = 48,
        encoder_blocks: Sequence[int] = (2, 2, 4),
        middle_blocks: int = 6,
        decoder_blocks: Sequence[int] = (2, 2, 2),
        dropout: float = 0.0,
        statistics_conditioning: bool = False,
        conditioning_hidden: int = 64,
    ) -> None:
        super().__init__()
        encoder_counts = tuple(int(value) for value in encoder_blocks)
        decoder_counts = tuple(int(value) for value in decoder_blocks)
        if width < 4:
            raise ValueError("NAFSR width must be at least 4")
        if not encoder_counts or len(encoder_counts) != len(decoder_counts):
            raise ValueError("encoder_blocks and decoder_blocks need the same non-zero length")
        if middle_blocks < 1:
            raise ValueError("NAFSR needs at least one middle block")

        self.width = width
        self.encoder_counts = encoder_counts
        self.middle_blocks = middle_blocks
        self.decoder_counts = decoder_counts
        self.dropout = dropout
        self.statistics_conditioning = statistics_conditioning
        self.conditioning_hidden = conditioning_hidden
        self.padder_size = 2 ** len(encoder_counts)

        self.intro = nn.Conv2d(1, width, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        channels = width
        for count in encoder_counts:
            self.encoders.append(_block_stack(channels, count, dropout))
            self.downsamples.append(nn.Conv2d(channels, channels * 2, 2, stride=2))
            channels *= 2

        self.middle = _block_stack(channels, middle_blocks, dropout)
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for count in decoder_counts:
            self.upsamples.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            channels //= 2
            self.decoders.append(_block_stack(channels, count, dropout))

        stage_channels = [
            *(width * (2**index) for index in range(len(encoder_counts))),
            width * (2 ** len(encoder_counts)),
            *(width * (2**index) for index in reversed(range(len(decoder_counts)))),
        ]
        self.conditioning_channels = tuple(stage_channels)
        if statistics_conditioning:
            if conditioning_hidden < 4:
                raise ValueError("conditioning_hidden must be at least 4")
            self.conditioner = nn.Sequential(
                nn.Linear(4, conditioning_hidden),
                nn.GELU(),
                nn.Linear(conditioning_hidden, 2 * sum(stage_channels)),
            )
            nn.init.zeros_(self.conditioner[-1].weight)
            nn.init.zeros_(self.conditioner[-1].bias)
        else:
            self.conditioner = None

        self.sr_head = nn.Sequential(
            nn.Conv2d(width, self.scale * self.scale, 3, padding=1),
            nn.PixelShuffle(self.scale),
        )
        final = self.sr_head[0]
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final.bias)

    def model_config(self) -> dict[str, object]:
        """Return the architecture fields required to reconstruct this model."""

        config: dict[str, object] = {
            "width": self.width,
            "encoder_blocks": list(self.encoder_counts),
            "middle_blocks": self.middle_blocks,
            "decoder_blocks": list(self.decoder_counts),
            "dropout": self.dropout,
        }
        if self.statistics_conditioning:
            config["statistics_conditioning"] = True
            config["conditioning_hidden"] = self.conditioning_hidden
        return config

    def _conditioning(
        self,
        inputs: torch.Tensor,
        statistics_override: torch.Tensor | None = None,
    ) -> list[torch.Tensor] | None:
        if self.conditioner is None:
            if statistics_override is not None:
                raise ConditioningStatisticsError(
                    "Conditioning statistics cannot be used by an unconditioned model"
                )
            return None
        statistics = (
            compute_conditioning_statistics(inputs)
            if statistics_override is None
            else validate_conditioning_statistics(statistics_override, inputs)
        )
        parameters = self.conditioner(statistics)
        return list(
            torch.split(parameters, [2 * value for value in self.conditioning_channels], dim=1)
        )

    @staticmethod
    def _apply_condition(features: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        scale, shift = parameters.chunk(2, dim=1)
        return features * (1.0 + 0.1 * torch.tanh(scale)[:, :, None, None]) + (
            0.1 * shift[:, :, None, None]
        )

    def _pad(self, inputs: torch.Tensor) -> torch.Tensor:
        height, width = inputs.shape[-2:]
        pad_height = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_width = (self.padder_size - width % self.padder_size) % self.padder_size
        return F.pad(inputs, (0, pad_width, 0, pad_height), mode="replicate")

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        conditioning_statistics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError(
                f"NAFSR expects NCHW input with one channel; got {tuple(inputs.shape)}"
            )
        height, width = inputs.shape[-2:]
        conditioning = self._conditioning(inputs, conditioning_statistics)
        condition_index = 0
        features = self.intro(self._pad(inputs))
        skips: list[torch.Tensor] = []
        for encoder, downsample in zip(self.encoders, self.downsamples, strict=True):
            if conditioning is not None:
                features = self._apply_condition(features, conditioning[condition_index])
                condition_index += 1
            features = encoder(features)
            skips.append(features)
            features = downsample(features)

        if conditioning is not None:
            features = self._apply_condition(features, conditioning[condition_index])
            condition_index += 1
        features = self.middle(features)
        for upsample, decoder, skip in zip(
            self.upsamples, self.decoders, reversed(skips), strict=True
        ):
            features = upsample(features) + skip
            if conditioning is not None:
                features = self._apply_condition(features, conditioning[condition_index])
                condition_index += 1
            features = decoder(features)

        learned_residual = self.sr_head(features)[..., : height * 2, : width * 2]
        bicubic = F.interpolate(
            inputs,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        return bicubic + learned_residual


__all__ = [
    "ConditioningStatisticsError",
    "NAFSR",
    "compute_conditioning_statistics",
    "validate_conditioning_statistics",
]
