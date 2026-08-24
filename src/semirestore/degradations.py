"""Reproducible raw-domain SEM degradation adapted from historical training."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from .data import DatasetValidationError

DEGRADATION_VERSION = "semirestore-historical-degradation-v1"
_OPERATIONS = ("blur", "gaussian", "speckle", "downsample")
_DOWNSAMPLE_MODES = frozenset({"area", "bicubic"})


@dataclass(frozen=True, slots=True)
class ParameterRange:
    """Inclusive finite sampling range."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.low) or not math.isfinite(self.high) or self.high < self.low:
            raise DatasetValidationError("Degradation ranges must be finite with high >= low")


@dataclass(frozen=True, slots=True)
class DegradationConfig:
    """Validated historical degradation composition."""

    blur_sigma: ParameterRange = ParameterRange(0.0, 0.0)
    gaussian_noise_std: ParameterRange = ParameterRange(0.0, 0.0)
    speckle_std: ParameterRange = ParameterRange(0.0, 0.0)
    additive_bias: ParameterRange = ParameterRange(0.0, 0.0)
    downsample_modes: tuple[str, ...] = ("area", "bicubic")
    operation_order: tuple[str, ...] = _OPERATIONS
    randomize_order: bool = True
    scale: int = 2

    def __post_init__(self) -> None:
        nonnegative = (self.blur_sigma, self.gaussian_noise_std, self.speckle_std)
        if any(value.low < 0 for value in nonnegative):
            raise DatasetValidationError("Blur and noise ranges cannot be negative")
        if not self.downsample_modes or any(
            mode not in _DOWNSAMPLE_MODES for mode in self.downsample_modes
        ):
            raise DatasetValidationError("downsample_modes must contain only area and/or bicubic")
        if set(self.operation_order) != set(_OPERATIONS) or len(self.operation_order) != 4:
            raise DatasetValidationError(
                "operation_order must contain blur, gaussian, speckle, and downsample exactly once"
            )
        if type(self.randomize_order) is not bool:
            raise DatasetValidationError("randomize_order must be a boolean")
        if self.scale != 2:
            raise DatasetValidationError("Historical degradation uses an exact scale of 2")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> DegradationConfig:
        """Load the historical schema without accepting silent extra settings."""

        allowed = {
            "schema_version",
            "blur_sigma",
            "gaussian_noise_std",
            "speckle_std",
            "additive_bias",
            "downsample_modes",
            "operation_order",
            "output_clipped",
            "scale",
        }
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise DatasetValidationError(f"Unknown degradation configuration fields: {unknown}")
        if values.get("schema_version") != 1:
            raise DatasetValidationError("Unsupported degradation configuration schema")
        if values.get("output_clipped", False) is not False:
            raise DatasetValidationError("Historical raw degradation must not clip its output")

        order_value = values.get("operation_order", "uniform_random_per_sample")
        if order_value == "uniform_random_per_sample":
            operation_order = _OPERATIONS
            randomize_order = True
        elif isinstance(order_value, Sequence) and not isinstance(order_value, (str, bytes)):
            operation_order = tuple(str(item) for item in order_value)
            randomize_order = False
        else:
            raise DatasetValidationError(
                "operation_order must be 'uniform_random_per_sample' or an explicit sequence"
            )
        modes = values.get("downsample_modes")
        if not isinstance(modes, Sequence) or isinstance(modes, (str, bytes)):
            raise DatasetValidationError("downsample_modes must be a sequence")
        return cls(
            blur_sigma=_mapping_range(values, "blur_sigma"),
            gaussian_noise_std=_mapping_range(values, "gaussian_noise_std"),
            speckle_std=_mapping_range(values, "speckle_std"),
            additive_bias=_mapping_range(values, "additive_bias", allow_negative=True),
            downsample_modes=tuple(str(mode) for mode in modes),
            operation_order=operation_order,
            randomize_order=randomize_order,
            scale=values.get("scale", 2),
        )


@dataclass(frozen=True, slots=True)
class DegradationResult:
    """Degraded tensor plus fully serializable sampled parameters."""

    tensor: torch.Tensor
    metadata: dict[str, Any]


def _mapping_range(
    values: Mapping[str, Any], name: str, *, allow_negative: bool = False
) -> ParameterRange:
    item = values.get(name)
    if not isinstance(item, Mapping):
        raise DatasetValidationError(f"Degradation configuration is missing range {name!r}")
    low, high = item.get("low"), item.get("high")
    if (
        isinstance(low, bool)
        or isinstance(high, bool)
        or not isinstance(low, (int, float))
        or not isinstance(high, (int, float))
    ):
        raise DatasetValidationError(f"Degradation range {name!r} must be numeric")
    result = ParameterRange(float(low), float(high))
    if not allow_negative and result.low < 0:
        raise DatasetValidationError(f"Degradation range {name!r} cannot be negative")
    return result


def derive_degradation_seed(base_seed: int, sample_id: str, *, epoch: int = 0) -> int:
    """Derive a stable seed independent of worker count, batching, and sample order."""

    if type(base_seed) is not int or base_seed < 0:
        raise DatasetValidationError("base_seed must be a non-negative integer")
    if type(epoch) is not int or epoch < 0:
        raise DatasetValidationError("epoch must be a non-negative integer")
    if not isinstance(sample_id, str) or not sample_id:
        raise DatasetValidationError("sample_id must be a non-empty string")
    payload = f"{DEGRADATION_VERSION}\0{base_seed}\0{epoch}\0{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _uniform(value: ParameterRange, generator: torch.Generator) -> float:
    if value.low == value.high:
        return value.low
    fraction = float(torch.rand((), generator=generator).item())
    return value.low + fraction * (value.high - value.low)


def _gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    # This threshold, radius rule, and reflective boundary are historical behavior.
    if sigma < 0.05:
        return image
    radius = max(1, min(4, int(round(3 * sigma))))
    if image.shape[-2] <= radius or image.shape[-1] <= radius:
        raise DatasetValidationError(
            f"Reflective blur radius {radius} requires both dimensions to exceed the radius"
        )
    positions = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel_1d = torch.exp(-(positions.square()) / (2 * sigma * sigma))
    kernel_1d /= kernel_1d.sum()
    kernel = torch.outer(kernel_1d, kernel_1d)[None, None]
    padded = F.pad(image[None], (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(padded, kernel)[0]


def _noise_like(image: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    noise = torch.randn(image.shape, generator=generator, dtype=torch.float32, device="cpu")
    return noise.to(device=image.device, dtype=image.dtype)


def degrade_sem_image(
    target: torch.Tensor,
    config: DegradationConfig,
    *,
    sample_id: str,
    base_seed: int,
    epoch: int = 0,
) -> DegradationResult:
    """Apply the seeded historical pipeline to a one-channel CHW HR tensor."""

    if not isinstance(target, torch.Tensor):
        raise DatasetValidationError("Degradation target must be a torch.Tensor")
    if target.ndim != 3 or target.shape[0] != 1:
        raise DatasetValidationError(
            f"Degradation target must be one-channel CHW; got shape {tuple(target.shape)}"
        )
    if not target.dtype.is_floating_point:
        raise DatasetValidationError("Degradation target must use a floating-point dtype")
    if target.shape[-2] < 2 or target.shape[-1] < 2:
        raise DatasetValidationError("Degradation target spatial dimensions must be at least 2")
    if target.shape[-2] % config.scale or target.shape[-1] % config.scale:
        raise DatasetValidationError("Degradation target dimensions must be divisible by 2")
    if not bool(torch.isfinite(target).all().item()):
        raise DatasetValidationError("Degradation target contains NaN or infinity")

    seed = derive_degradation_seed(base_seed, sample_id, epoch=epoch)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    blur_sigma = _uniform(config.blur_sigma, generator)
    gaussian_std = _uniform(config.gaussian_noise_std, generator)
    speckle_std = _uniform(config.speckle_std, generator)
    additive_bias = _uniform(config.additive_bias, generator)
    mode_index = int(torch.randint(len(config.downsample_modes), (), generator=generator).item())
    downsample_mode = config.downsample_modes[mode_index]
    if config.randomize_order:
        permutation = torch.randperm(len(config.operation_order), generator=generator).tolist()
        operation_order = tuple(config.operation_order[index] for index in permutation)
    else:
        operation_order = config.operation_order

    image = target.clone()
    for operation in operation_order:
        if operation == "blur":
            image = _gaussian_blur(image, blur_sigma)
        elif operation == "gaussian":
            image = image + _noise_like(image, generator) * gaussian_std + additive_bias
        elif operation == "speckle":
            image = image + image * _noise_like(image, generator) * speckle_std
        else:
            size = (target.shape[-2] // config.scale, target.shape[-1] // config.scale)
            if downsample_mode == "bicubic":
                image = F.interpolate(
                    image[None],
                    size=size,
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )[0]
            else:
                image = F.interpolate(image[None], size=size, mode="area")[0]

    result = image.contiguous()
    metadata: dict[str, Any] = {
        "degradation_version": DEGRADATION_VERSION,
        "sample_id": sample_id,
        "base_seed": base_seed,
        "epoch": epoch,
        "derived_seed": seed,
        "scale": config.scale,
        "blur_sigma": blur_sigma,
        "gaussian_noise_std": gaussian_std,
        "speckle_std": speckle_std,
        "additive_bias": additive_bias,
        "downsample_mode": downsample_mode,
        "operation_order": list(operation_order),
        "blur_kernel": "gaussian_radius_min4_round_3sigma",
        "blur_boundary": "reflect",
        "bicubic_align_corners": False,
        "bicubic_antialias": True,
        "output_clipped": False,
        "input_shape": list(target.shape),
        "output_shape": list(result.shape),
    }
    return DegradationResult(result, metadata)


__all__ = [
    "DEGRADATION_VERSION",
    "DegradationConfig",
    "DegradationResult",
    "ParameterRange",
    "degrade_sem_image",
    "derive_degradation_seed",
]
