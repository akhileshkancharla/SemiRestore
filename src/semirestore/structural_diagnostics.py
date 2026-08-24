"""Deterministic rule-based structural suitability analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from torch.nn import functional as F

STRUCTURAL_DIAGNOSTIC_VERSION = "semirestore-structural-v1"
STRUCTURAL_RULESET_VERSION = "semirestore-suitability-rules-v1"
EDGE_MAGNITUDE_THRESHOLD = 0.10
FLAT_DYNAMIC_RANGE_THRESHOLD = 0.02
FLAT_GRADIENT_ENERGY_THRESHOLD = 1e-4
BLUR_LAPLACIAN_VARIANCE_THRESHOLD = 0.002
BLUR_GRADIENT_MAGNITUDE_THRESHOLD = 0.03
NOISE_WARNING_THRESHOLD = 0.08
TEXTURE_EDGE_DENSITY_THRESHOLD = 0.35
TEXTURE_NOISE_AMBIGUITY_THRESHOLD = 0.04

Recommendation = Literal["bypass", "warn", "restore"]


class StructuralDiagnosticError(ValueError):
    """Raised when structural diagnostics cannot safely measure an image."""


@dataclass(frozen=True, slots=True)
class StructuralMeasurement:
    value: float
    units: str
    interpretation: str
    resolution_sensitive: bool
    qualitative_label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "units": self.units,
            "interpretation": self.interpretation,
            "resolution_sensitive": self.resolution_sensitive,
            "qualitative_label": self.qualitative_label,
        }


@dataclass(frozen=True, slots=True)
class StructuralDiagnostics:
    measurements: dict[str, StructuralMeasurement]
    recommendation: Recommendation
    triggered_rules: tuple[str, ...]
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    conventions: dict[str, str]
    diagnostic_version: str = STRUCTURAL_DIAGNOSTIC_VERSION
    ruleset_version: str = STRUCTURAL_RULESET_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "measurements": {
                name: measurement.to_dict()
                for name, measurement in self.measurements.items()
            },
            "recommendation": self.recommendation,
            "triggered_rules": list(self.triggered_rules),
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "conventions": dict(self.conventions),
            "diagnostic_version": self.diagnostic_version,
            "ruleset_version": self.ruleset_version,
            "recommendation_kind": "rule_based_advisory",
            "is_probability_accuracy_or_confidence": False,
        }


def _normalized_tensor(image: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        value = image.detach()
        if value.layout != torch.strided:
            raise StructuralDiagnosticError("Structural diagnostics require a dense tensor")
    elif isinstance(image, np.ndarray):
        if (
            np.issubdtype(image.dtype, np.bool_)
            or not np.issubdtype(image.dtype, np.number)
            or np.issubdtype(image.dtype, np.complexfloating)
        ):
            raise StructuralDiagnosticError("Structural diagnostics require real numeric values")
        value = torch.from_numpy(np.array(image, dtype=np.float64, copy=True, order="C"))
    else:
        raise StructuralDiagnosticError("Structural diagnostics require an array or tensor")
    if value.ndim != 2:
        raise StructuralDiagnosticError(
            f"Structural diagnostics require one 2D grayscale image; got {tuple(value.shape)}"
        )
    if value.shape[0] < 3 or value.shape[1] < 3:
        raise StructuralDiagnosticError("Structural diagnostics require at least a 3x3 image")
    if not value.dtype.is_floating_point:
        value = value.to(torch.float64)
    value = value.to(device="cpu", dtype=torch.float64).clone().contiguous()
    if not bool(torch.isfinite(value).all().item()):
        raise StructuralDiagnosticError("Structural diagnostics require finite values")
    minimum, maximum = float(value.min().item()), float(value.max().item())
    if minimum < 0.0 or maximum > 1.0:
        raise StructuralDiagnosticError(
            f"Structural values [{minimum}, {maximum}] exceed normalized range [0, 1]"
        )
    return value[None, None]


def _reflect_convolution(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    padded = F.pad(image, (1, 1, 1, 1), mode="reflect")
    return F.conv2d(padded, kernel[None, None])


def _label(value: float, low: float, high: float) -> str:
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "moderate"


def analyze_structure(image: np.ndarray | torch.Tensor) -> StructuralDiagnostics:
    """Measure fixed-resolution structure and apply explainable advisory rules."""

    tensor = _normalized_tensor(image)
    dtype = tensor.dtype
    laplacian_kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=dtype
    )
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=dtype
    ) / 8.0
    sobel_y = sobel_x.transpose(0, 1)
    gaussian = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=dtype
    ) / 16.0
    noise_kernel = torch.tensor(
        [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]], dtype=dtype
    )

    laplacian = _reflect_convolution(tensor, laplacian_kernel)
    gradient_x = _reflect_convolution(tensor, sobel_x)
    gradient_y = _reflect_convolution(tensor, sobel_y)
    gradient_magnitude = torch.sqrt(gradient_x.square() + gradient_y.square())
    low_pass = _reflect_convolution(tensor, gaussian)
    high_frequency = tensor - low_pass
    noise_response = F.conv2d(tensor, noise_kernel[None, None])

    laplacian_variance = float(laplacian.var(unbiased=False).item())
    mean_gradient_magnitude = float(gradient_magnitude.mean().item())
    gradient_energy = float(gradient_magnitude.square().mean().item())
    edge_density = float((gradient_magnitude >= EDGE_MAGNITUDE_THRESHOLD).double().mean().item())
    high_frequency_energy = float(high_frequency.square().mean().item())
    approximate_noise = float(
        (np.sqrt(np.pi / 2.0) / 6.0) * noise_response.abs().mean().item()
    )
    dynamic_range = float((tensor.max() - tensor.min()).item())
    blur_indicator = float(
        laplacian_variance < BLUR_LAPLACIAN_VARIANCE_THRESHOLD
        and mean_gradient_magnitude < BLUR_GRADIENT_MAGNITUDE_THRESHOLD
    )

    measurements = {
        "laplacian_sharpness_variance": StructuralMeasurement(
            laplacian_variance,
            "normalized_intensity_squared",
            "Population variance of the reflected 4-neighbor Laplacian response.",
            True,
            _label(laplacian_variance, 0.002, 0.03),
        ),
        "mean_gradient_magnitude": StructuralMeasurement(
            mean_gradient_magnitude,
            "normalized_intensity_per_pixel",
            "Mean magnitude of reflected Sobel gradients scaled by 1/8.",
            True,
            _label(mean_gradient_magnitude, 0.03, 0.15),
        ),
        "gradient_energy": StructuralMeasurement(
            gradient_energy,
            "normalized_intensity_squared_per_pixel",
            "Mean squared Sobel-gradient magnitude.",
            True,
        ),
        "edge_density": StructuralMeasurement(
            edge_density,
            "fraction_of_pixels",
            f"Fraction with Sobel magnitude at least {EDGE_MAGNITUDE_THRESHOLD}.",
            True,
        ),
        "high_frequency_energy": StructuralMeasurement(
            high_frequency_energy,
            "normalized_intensity_squared",
            "Mean squared residual from a reflected 3x3 Gaussian low-pass.",
            True,
        ),
        "approximate_noise_sigma": StructuralMeasurement(
            approximate_noise,
            "normalized_intensity",
            "Immerkaer-style valid-support estimate; edges and texture can bias it upward.",
            True,
        ),
        "blur_indicator": StructuralMeasurement(
            blur_indicator,
            "binary_rule",
            "One only when Laplacian variance and mean gradient are both below v1 thresholds.",
            True,
            "triggered" if blur_indicator else "not_triggered",
        ),
    }

    rules: list[str] = []
    reasons: list[str] = []
    warnings = [
        "Structural diagnostics are resolution-sensitive and must not be compared across "
        "different sampling scales without qualification.",
        "No-reference structure heuristics cannot prove reconstruction correctness or confidence.",
        "High-frequency energy can represent useful detail, noise, or both.",
    ]
    if (
        dynamic_range <= FLAT_DYNAMIC_RANGE_THRESHOLD
        and gradient_energy <= FLAT_GRADIENT_ENERGY_THRESHOLD
    ):
        recommendation: Recommendation = "bypass"
        rules.append("flat_content_v1")
        reasons.append(
            "Dynamic range and gradient energy are both below flat-content thresholds; "
            "restoration cannot recover evidenced detail from this advisory alone."
        )
    elif approximate_noise >= NOISE_WARNING_THRESHOLD:
        recommendation = "warn"
        rules.append("high_approximate_noise_v1")
        reasons.append(
            "The approximate noise estimate exceeds the warning threshold and may confound detail."
        )
    elif (
        edge_density >= TEXTURE_EDGE_DENSITY_THRESHOLD
        and approximate_noise >= TEXTURE_NOISE_AMBIGUITY_THRESHOLD
    ):
        recommendation = "warn"
        rules.append("texture_noise_ambiguity_v1")
        reasons.append(
            "Dense edges coexist with elevated high-frequency noise; useful texture and noise "
            "cannot be separated reliably without a reference."
        )
    elif bool(blur_indicator):
        recommendation = "restore"
        rules.append("low_sharpness_restore_candidate_v1")
        reasons.append(
            "Low Laplacian variance and gradient magnitude indicate a possible blur-restoration "
            "candidate, subject to model-domain limitations."
        )
    else:
        recommendation = "restore"
        rules.append("no_structural_bypass_or_warning_rule_v1")
        reasons.append("No structural bypass or warning threshold was triggered.")
    if approximate_noise >= TEXTURE_NOISE_AMBIGUITY_THRESHOLD:
        warnings.append(
            "The approximate noise estimate may include true SEM texture and edge structure."
        )
    return StructuralDiagnostics(
        measurements=measurements,
        recommendation=recommendation,
        triggered_rules=tuple(rules),
        reasons=tuple(reasons),
        warnings=tuple(warnings),
        conventions={
            "laplacian": "4-neighbor 3x3, reflect boundary",
            "gradient": "Sobel 3x3 divided by 8, reflect boundary",
            "high_frequency": "input minus 3x3 binomial Gaussian, reflect boundary",
            "noise": "Immerkaer 3x3 response, valid support, sqrt(pi/2)/6 mean absolute",
            "edge_threshold": str(EDGE_MAGNITUDE_THRESHOLD),
        },
    )


__all__ = [
    "STRUCTURAL_DIAGNOSTIC_VERSION",
    "STRUCTURAL_RULESET_VERSION",
    "StructuralDiagnosticError",
    "StructuralDiagnostics",
    "StructuralMeasurement",
    "analyze_structure",
]
