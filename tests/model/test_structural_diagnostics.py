from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from semirestore.structural_diagnostics import (
    STRUCTURAL_DIAGNOSTIC_VERSION,
    STRUCTURAL_RULESET_VERSION,
    StructuralDiagnosticError,
    analyze_structure,
)


def _value(result: object, name: str) -> float:
    return result.measurements[name].value  # type: ignore[attr-defined,no-any-return]


def _edge(size: int = 32) -> np.ndarray:
    image = np.zeros((size, size), dtype=np.float64)
    image[:, size // 2 :] = 1.0
    return image


def _blur(image: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    tensor = torch.from_numpy(image)[None, None]
    radius = kernel_size // 2
    padded = F.pad(tensor, (radius, radius, radius, radius), mode="reflect")
    return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)[0, 0].numpy()


def test_flat_image_has_zero_structure_and_bypass_advisory() -> None:
    result = analyze_structure(np.full((16, 16), 0.5, dtype=np.float32))

    assert _value(result, "laplacian_sharpness_variance") == 0.0
    assert _value(result, "mean_gradient_magnitude") == 0.0
    assert _value(result, "gradient_energy") == 0.0
    assert _value(result, "edge_density") == 0.0
    assert _value(result, "high_frequency_energy") == 0.0
    assert _value(result, "approximate_noise_sigma") == 0.0
    assert result.recommendation == "bypass"
    assert result.triggered_rules == ("flat_content_v1",)


def test_sharp_edge_has_positive_laplacian_gradient_and_edges() -> None:
    result = analyze_structure(_edge())

    assert _value(result, "laplacian_sharpness_variance") > 0.0
    assert _value(result, "mean_gradient_magnitude") > 0.0
    assert _value(result, "gradient_energy") > 0.0
    assert _value(result, "edge_density") > 0.0
    assert result.recommendation in ("restore", "warn")


def test_blur_reduces_sharpness_and_high_frequency_energy() -> None:
    sharp = _edge()
    blurred = _blur(sharp)

    sharp_result = analyze_structure(sharp)
    blurred_result = analyze_structure(blurred)

    assert _value(blurred_result, "laplacian_sharpness_variance") < _value(
        sharp_result, "laplacian_sharpness_variance"
    )
    assert _value(blurred_result, "high_frequency_energy") < _value(
        sharp_result, "high_frequency_energy"
    )


def test_noisy_image_triggers_explainable_warning() -> None:
    generator = np.random.default_rng(4)
    image = np.clip(0.5 + generator.normal(0.0, 0.25, size=(64, 64)), 0.0, 1.0)

    result = analyze_structure(image)

    assert _value(result, "approximate_noise_sigma") >= 0.08
    assert result.recommendation == "warn"
    assert result.triggered_rules == ("high_approximate_noise_v1",)
    assert "noise estimate" in result.reasons[0]


def test_textured_checkerboard_reports_high_frequency_limitations() -> None:
    coordinates = np.indices((32, 32)).sum(axis=0)
    image = (coordinates % 2).astype(np.float32)

    result = analyze_structure(image)

    assert _value(result, "high_frequency_energy") > 0.0
    assert any("useful detail, noise, or both" in warning for warning in result.warnings)
    assert result.to_dict()["is_probability_accuracy_or_confidence"] is False


def test_possible_blur_rule_is_advisory_not_probability() -> None:
    image = np.tile(np.linspace(0.2, 0.8, 64), (64, 1))
    result = analyze_structure(image)

    assert _value(result, "blur_indicator") == 1.0
    assert result.recommendation == "restore"
    assert result.triggered_rules == ("low_sharpness_restore_candidate_v1",)
    assert result.to_dict()["recommendation_kind"] == "rule_based_advisory"


def test_kernel_and_boundary_conventions_are_returned() -> None:
    result = analyze_structure(_edge(16))

    assert "4-neighbor" in result.conventions["laplacian"]
    assert "reflect boundary" in result.conventions["gradient"]
    assert "valid support" in result.conventions["noise"]
    assert result.conventions["edge_threshold"] == "0.1"


def test_measurements_explicitly_mark_resolution_sensitivity() -> None:
    result = analyze_structure(_edge())

    assert all(item.resolution_sensitive for item in result.measurements.values())
    assert any("resolution-sensitive" in warning for warning in result.warnings)


def test_resampling_changes_resolution_sensitive_measurements() -> None:
    original = _edge(16)
    upsampled = np.repeat(np.repeat(original, 2, axis=0), 2, axis=1)

    first = analyze_structure(original)
    second = analyze_structure(upsampled)

    assert _value(first, "edge_density") != _value(second, "edge_density")


def test_diagnostics_are_deterministic_and_do_not_mutate_input() -> None:
    generator = np.random.default_rng(9)
    image = generator.random((23, 19), dtype=np.float32)
    original = image.copy()

    first = analyze_structure(image)
    second = analyze_structure(image)

    assert first == second
    np.testing.assert_array_equal(image, original)


def test_tensor_input_is_supported_without_autograd_side_effects() -> None:
    image = torch.linspace(0.0, 1.0, 256).reshape(16, 16).requires_grad_()

    result = analyze_structure(image)

    assert _value(result, "mean_gradient_magnitude") > 0.0
    assert image.grad is None


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((3, 3, 1), dtype=np.float32),
        np.zeros((3, 3), dtype=np.bool_),
        np.zeros((3, 3), dtype=np.complex64),
        np.full((3, 3), np.nan, dtype=np.float32),
        np.full((3, 3), np.inf, dtype=np.float32),
        np.full((3, 3), -0.1, dtype=np.float32),
        np.full((3, 3), 1.1, dtype=np.float32),
    ],
)
def test_invalid_inputs_are_rejected(image: np.ndarray) -> None:
    with pytest.raises(StructuralDiagnosticError):
        analyze_structure(image)


def test_serialization_is_strict_json_safe_and_versioned() -> None:
    payload = analyze_structure(_edge()).to_dict()

    rendered = json.dumps(payload, allow_nan=False, sort_keys=True)

    assert STRUCTURAL_DIAGNOSTIC_VERSION in rendered
    assert STRUCTURAL_RULESET_VERSION in rendered
    assert payload["diagnostic_version"] == STRUCTURAL_DIAGNOSTIC_VERSION
    assert payload["ruleset_version"] == STRUCTURAL_RULESET_VERSION
