"""Deterministic no-reference intensity measurements for normalized SEM images."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

INTENSITY_DIAGNOSTIC_VERSION = "semirestore-intensity-v1"
ENTROPY_BINS = 256
LOWER_SATURATION_THRESHOLD = 1.0 / 255.0
UPPER_SATURATION_THRESHOLD = 254.0 / 255.0


class IntensityDiagnosticError(ValueError):
    """Raised when an image cannot be measured under the normalized contract."""


@dataclass(frozen=True, slots=True)
class IntensityMeasurement:
    """One raw measurement with units and a bounded interpretation."""

    value: float
    units: str
    expected_range: tuple[float, float]
    interpretation: str
    qualitative_label: str | None = None
    display_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "units": self.units,
            "expected_range": list(self.expected_range),
            "interpretation": self.interpretation,
            "qualitative_label": self.qualitative_label,
            "display_score": self.display_score,
        }


@dataclass(frozen=True, slots=True)
class IntensityDiagnostics:
    """Serialization-friendly intensity profile and explicit heuristic warnings."""

    measurements: dict[str, IntensityMeasurement]
    qualitative_label: str
    triggered_rules: tuple[str, ...]
    warnings: tuple[str, ...]
    diagnostic_version: str = INTENSITY_DIAGNOSTIC_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "measurements": {
                name: measurement.to_dict()
                for name, measurement in self.measurements.items()
            },
            "qualitative_label": self.qualitative_label,
            "triggered_rules": list(self.triggered_rules),
            "warnings": list(self.warnings),
            "diagnostic_version": self.diagnostic_version,
            "metric_kind": "no_reference_descriptive",
            "is_accuracy_or_confidence": False,
        }


def _normalized_array(image: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        if image.layout != torch.strided:
            raise IntensityDiagnosticError("Intensity diagnostics require a dense tensor")
        if image.requires_grad:
            image = image.detach()
        array = image.cpu().numpy()
    elif isinstance(image, np.ndarray):
        array = image
    else:
        raise IntensityDiagnosticError("Intensity diagnostics require an array or tensor")
    if array.ndim != 2:
        raise IntensityDiagnosticError(
            f"Intensity diagnostics require one 2D grayscale image; got shape {array.shape}"
        )
    if array.size == 0:
        raise IntensityDiagnosticError("Intensity diagnostics require a non-empty image")
    if (
        np.issubdtype(array.dtype, np.bool_)
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise IntensityDiagnosticError("Intensity diagnostics require real numeric values")
    converted = np.array(array, dtype=np.float64, copy=True, order="C")
    if not np.isfinite(converted).all():
        raise IntensityDiagnosticError("Intensity diagnostics require finite values")
    minimum, maximum = float(converted.min()), float(converted.max())
    if minimum < 0.0 or maximum > 1.0:
        raise IntensityDiagnosticError(
            f"Intensity values [{minimum}, {maximum}] exceed the explicit normalized range [0, 1]"
        )
    return converted


def _entropy(array: np.ndarray) -> float:
    counts, _ = np.histogram(array, bins=ENTROPY_BINS, range=(0.0, 1.0))
    probabilities = counts[counts > 0].astype(np.float64) / array.size
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _mean_label(value: float) -> str:
    if value < 0.15:
        return "dark"
    if value > 0.85:
        return "bright"
    return "mid-range"


def _contrast_label(value: float) -> str:
    if value < 0.10:
        return "low"
    if value > 0.60:
        return "broad"
    return "moderate"


def analyze_intensity(image: np.ndarray | torch.Tensor) -> IntensityDiagnostics:
    """Measure one canonical ``[0,1]`` image without modifying it."""

    array = _normalized_array(image)
    minimum = float(array.min())
    maximum = float(array.max())
    mean = float(array.mean(dtype=np.float64))
    standard_deviation = float(array.std(dtype=np.float64))
    dynamic_range = maximum - minimum
    percentile_05, percentile_95 = np.quantile(array, (0.05, 0.95))
    contrast_proxy = float(percentile_95 - percentile_05)
    entropy = _entropy(array)
    lower_saturation = float(np.mean(array <= LOWER_SATURATION_THRESHOLD))
    upper_saturation = float(np.mean(array >= UPPER_SATURATION_THRESHOLD))

    measurements = {
        "mean": IntensityMeasurement(
            mean,
            "normalized_intensity",
            (0.0, 1.0),
            "Arithmetic mean brightness; it does not measure restoration quality.",
            _mean_label(mean),
        ),
        "standard_deviation": IntensityMeasurement(
            standard_deviation,
            "normalized_intensity",
            (0.0, 0.5),
            "Population RMS contrast around the image mean.",
        ),
        "minimum": IntensityMeasurement(
            minimum,
            "normalized_intensity",
            (0.0, 1.0),
            "Lowest observed normalized intensity.",
        ),
        "maximum": IntensityMeasurement(
            maximum,
            "normalized_intensity",
            (0.0, 1.0),
            "Highest observed normalized intensity.",
        ),
        "dynamic_range": IntensityMeasurement(
            dynamic_range,
            "normalized_intensity",
            (0.0, 1.0),
            "Observed maximum minus minimum; sensitive to individual extrema.",
        ),
        "contrast_proxy": IntensityMeasurement(
            contrast_proxy,
            "normalized_intensity_p95_minus_p05",
            (0.0, 1.0),
            "Robust 95th-minus-5th percentile intensity span.",
            _contrast_label(contrast_proxy),
        ),
        "entropy": IntensityMeasurement(
            entropy,
            "bits_per_pixel_histogram",
            (0.0, math.log2(ENTROPY_BINS)),
            "Shannon entropy of 256 fixed-width normalized-intensity bins.",
        ),
        "lower_saturation_fraction": IntensityMeasurement(
            lower_saturation,
            "fraction_of_pixels",
            (0.0, 1.0),
            f"Fraction at or below {LOWER_SATURATION_THRESHOLD:.8f}.",
        ),
        "upper_saturation_fraction": IntensityMeasurement(
            upper_saturation,
            "fraction_of_pixels",
            (0.0, 1.0),
            f"Fraction at or above {UPPER_SATURATION_THRESHOLD:.8f}.",
        ),
    }

    rules: list[str] = []
    warnings = [
        "No-reference intensity diagnostics cannot establish reconstruction "
        "correctness or confidence."
    ]
    if dynamic_range <= 1e-8:
        label = "constant"
        rules.append("dynamic_range <= 1e-8")
    elif mean < 0.10:
        label = "dark"
        rules.append("mean < 0.10")
    elif mean > 0.90:
        label = "bright"
        rules.append("mean > 0.90")
    elif contrast_proxy < 0.10:
        label = "low_contrast"
        rules.append("p95_minus_p05 < 0.10")
    else:
        label = "nominal_intensity_profile"
        rules.append("no intensity caution threshold triggered")
    if lower_saturation > 0.01:
        warnings.append("More than 1% of pixels are near the lower intensity boundary.")
        rules.append("lower_saturation_fraction > 0.01")
    if upper_saturation > 0.01:
        warnings.append("More than 1% of pixels are near the upper intensity boundary.")
        rules.append("upper_saturation_fraction > 0.01")
    return IntensityDiagnostics(measurements, label, tuple(rules), tuple(warnings))


__all__ = [
    "ENTROPY_BINS",
    "INTENSITY_DIAGNOSTIC_VERSION",
    "IntensityDiagnosticError",
    "IntensityDiagnostics",
    "IntensityMeasurement",
    "analyze_intensity",
]
