"""Explicit-range full-reference restoration metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch.nn import functional as F

RangePolicy = Literal["reject", "clip"]

SSIM_WINDOW_SIZE = 11
SSIM_SIGMA = 1.5
SSIM_K1 = 0.01
SSIM_K2 = 0.03


@dataclass(frozen=True, slots=True)
class ImageReferenceMetrics:
    """Full-reference scores for one image."""

    sample_id: str
    psnr_db: float
    ssim: float


@dataclass(frozen=True, slots=True)
class ReferenceMetricSummary:
    """Per-image scores and deterministic aggregate means."""

    per_image: tuple[ImageReferenceMetrics, ...]
    mean_psnr_db: float
    mean_ssim: float
    data_range: float
    data_min: float
    range_policy: RangePolicy

    def as_dict(self) -> dict[str, Any]:
        """Return strict-JSON-friendly primitive values."""

        return {
            "per_image": [
                {
                    "sample_id": item.sample_id,
                    "psnr_db": _serial_number(item.psnr_db),
                    "ssim": _serial_number(item.ssim),
                }
                for item in self.per_image
            ],
            "aggregate": {
                "image_count": len(self.per_image),
                "mean_psnr_db": _serial_number(self.mean_psnr_db),
                "mean_ssim": _serial_number(self.mean_ssim),
            },
            "policy": {
                "metric_kind": "full_reference",
                "data_range": self.data_range,
                "data_min": self.data_min,
                "range_policy": self.range_policy,
                "psnr_zero_error": "positive_infinity",
                "ssim_window_size": SSIM_WINDOW_SIZE,
                "ssim_sigma": SSIM_SIGMA,
                "ssim_gaussian_weights": True,
                "ssim_sample_covariance": False,
            },
        }


def _serial_number(value: float) -> float | str:
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    if math.isnan(value):
        return "NaN"
    return value


def _validate_range(data_range: float, data_min: float, range_policy: str) -> RangePolicy:
    if (
        isinstance(data_range, bool)
        or not isinstance(data_range, (int, float))
        or not math.isfinite(data_range)
        or data_range <= 0
    ):
        raise ValueError("data_range must be finite and positive")
    if (
        isinstance(data_min, bool)
        or not isinstance(data_min, (int, float))
        or not math.isfinite(data_min)
    ):
        raise ValueError("data_min must be finite")
    if range_policy not in ("reject", "clip"):
        raise ValueError("range_policy must be 'reject' or 'clip'")
    return range_policy  # type: ignore[return-value]


def _as_nchw(tensor: torch.Tensor, *, label: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if tensor.ndim == 2:
        tensor = tensor[None, None]
    elif tensor.ndim == 3:
        tensor = tensor[None] if tensor.shape[0] == 1 else tensor[:, None]
    if tensor.ndim != 4 or tensor.shape[1] != 1:
        raise ValueError(
            f"{label} must be grayscale HW, 1HW, NHW, or N1HW; got {tuple(tensor.shape)}"
        )
    if tensor.numel() == 0:
        raise ValueError(f"{label} must not be empty")
    if not tensor.dtype.is_floating_point:
        raise ValueError(f"{label} must use a floating-point dtype")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{label} contains NaN or infinity")
    return tensor


def _validated_pair(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float,
    data_min: float,
    range_policy: str,
) -> tuple[torch.Tensor, torch.Tensor, RangePolicy]:
    policy = _validate_range(data_range, data_min, range_policy)
    prediction_nchw = _as_nchw(prediction, label="prediction")
    target_nchw = _as_nchw(target, label="target")
    if prediction_nchw.shape != target_nchw.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: {tuple(prediction_nchw.shape)} vs "
            f"{tuple(target_nchw.shape)}"
        )
    if prediction_nchw.device != target_nchw.device:
        raise ValueError("Prediction and target must be on the same device")
    lower = float(data_min)
    upper = lower + float(data_range)
    if policy == "reject":
        for label, tensor in (("prediction", prediction_nchw), ("target", target_nchw)):
            minimum = float(tensor.min().item())
            maximum = float(tensor.max().item())
            if minimum < lower or maximum > upper:
                raise ValueError(
                    f"{label} values [{minimum}, {maximum}] exceed explicit range "
                    f"[{lower}, {upper}]"
                )
    else:
        prediction_nchw = prediction_nchw.clamp(lower, upper)
        target_nchw = target_nchw.clamp(lower, upper)
    return prediction_nchw.to(torch.float64), target_nchw.to(torch.float64), policy


def peak_signal_to_noise_ratio(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float,
    data_min: float = 0.0,
    range_policy: RangePolicy = "reject",
) -> torch.Tensor:
    """Return one PSNR value per image; exact matches yield positive infinity."""

    prediction_nchw, target_nchw, _ = _validated_pair(
        prediction,
        target,
        data_range=data_range,
        data_min=data_min,
        range_policy=range_policy,
    )
    mse = (prediction_nchw - target_nchw).square().flatten(1).mean(1)
    scale = torch.tensor(float(data_range) ** 2, dtype=mse.dtype, device=mse.device)
    return torch.where(
        mse == 0,
        torch.full_like(mse, torch.inf),
        10.0 * torch.log10(scale / mse),
    )


def _ssim_kernel(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = SSIM_WINDOW_SIZE // 2
    positions = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-(positions.square()) / (2 * SSIM_SIGMA**2))
    kernel_1d /= kernel_1d.sum()
    return torch.outer(kernel_1d, kernel_1d)[None, None]


def structural_similarity_index(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float,
    data_min: float = 0.0,
    range_policy: RangePolicy = "reject",
) -> torch.Tensor:
    """Return Gaussian-window SSIM per image using the historical policy."""

    prediction_nchw, target_nchw, _ = _validated_pair(
        prediction,
        target,
        data_range=data_range,
        data_min=data_min,
        range_policy=range_policy,
    )
    height, width = prediction_nchw.shape[-2:]
    if min(height, width) < SSIM_WINDOW_SIZE:
        raise ValueError(
            f"SSIM images must be at least {SSIM_WINDOW_SIZE}x{SSIM_WINDOW_SIZE}; "
            f"got {height}x{width}"
        )
    kernel = _ssim_kernel(device=prediction_nchw.device, dtype=prediction_nchw.dtype)
    mean_prediction = F.conv2d(prediction_nchw, kernel)
    mean_target = F.conv2d(target_nchw, kernel)
    mean_prediction_sq = mean_prediction.square()
    mean_target_sq = mean_target.square()
    mean_cross = mean_prediction * mean_target
    variance_prediction = F.conv2d(prediction_nchw.square(), kernel) - mean_prediction_sq
    variance_target = F.conv2d(target_nchw.square(), kernel) - mean_target_sq
    covariance = F.conv2d(prediction_nchw * target_nchw, kernel) - mean_cross
    c1 = (SSIM_K1 * float(data_range)) ** 2
    c2 = (SSIM_K2 * float(data_range)) ** 2
    score_map = ((2 * mean_cross + c1) * (2 * covariance + c2)) / (
        (mean_prediction_sq + mean_target_sq + c1)
        * (variance_prediction + variance_target + c2)
    )
    return score_map.flatten(1).mean(1)


def compute_reference_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float,
    data_min: float = 0.0,
    range_policy: RangePolicy = "reject",
    sample_ids: list[str] | tuple[str, ...] | None = None,
) -> ReferenceMetricSummary:
    """Compute per-image and aggregate PSNR/SSIM for paired reference images."""

    psnr_values = peak_signal_to_noise_ratio(
        prediction,
        target,
        data_range=data_range,
        data_min=data_min,
        range_policy=range_policy,
    )
    ssim_values = structural_similarity_index(
        prediction,
        target,
        data_range=data_range,
        data_min=data_min,
        range_policy=range_policy,
    )
    count = int(psnr_values.numel())
    if sample_ids is None:
        ids = tuple(str(index) for index in range(count))
    else:
        ids = tuple(sample_ids)
        if len(ids) != count or any(not isinstance(item, str) or not item for item in ids):
            raise ValueError("sample_ids must contain one non-empty string per image")
    psnr_cpu = psnr_values.detach().cpu().tolist()
    ssim_cpu = ssim_values.detach().cpu().tolist()
    per_image = tuple(
        ImageReferenceMetrics(sample_id, float(psnr), float(ssim))
        for sample_id, psnr, ssim in zip(ids, psnr_cpu, ssim_cpu, strict=True)
    )
    return ReferenceMetricSummary(
        per_image=per_image,
        mean_psnr_db=float(psnr_values.mean().item()),
        mean_ssim=float(ssim_values.mean().item()),
        data_range=float(data_range),
        data_min=float(data_min),
        range_policy=range_policy,
    )


__all__ = [
    "ImageReferenceMetrics",
    "ReferenceMetricSummary",
    "compute_reference_metrics",
    "peak_signal_to_noise_ratio",
    "structural_similarity_index",
]
