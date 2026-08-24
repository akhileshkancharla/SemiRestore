from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from semirestore.intensity_diagnostics import (
    INTENSITY_DIAGNOSTIC_VERSION,
    IntensityDiagnosticError,
    analyze_intensity,
)


def _value(result: object, name: str) -> float:
    return result.measurements[name].value  # type: ignore[attr-defined,no-any-return]


def test_numerical_statistics_are_correct() -> None:
    image = np.array([[0.0, 0.25], [0.75, 1.0]], dtype=np.float32)

    result = analyze_intensity(image)

    assert _value(result, "mean") == pytest.approx(0.5)
    assert _value(result, "standard_deviation") == pytest.approx(np.std(image))
    assert _value(result, "minimum") == 0.0
    assert _value(result, "maximum") == 1.0
    assert _value(result, "dynamic_range") == 1.0
    assert _value(result, "contrast_proxy") == pytest.approx(0.925)


def test_entropy_uses_fixed_256_bin_shannon_definition() -> None:
    image = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    result = analyze_intensity(image)

    assert _value(result, "entropy") == pytest.approx(1.0)
    assert result.measurements["entropy"].units == "bits_per_pixel_histogram"
    assert result.measurements["entropy"].expected_range == (0.0, 8.0)


def test_constant_image_has_zero_spread_and_explicit_rule() -> None:
    result = analyze_intensity(np.full((8, 8), 0.5, dtype=np.float32))

    assert result.qualitative_label == "constant"
    assert _value(result, "standard_deviation") == 0.0
    assert _value(result, "dynamic_range") == 0.0
    assert _value(result, "entropy") == 0.0
    assert "dynamic_range <= 1e-8" in result.triggered_rules


def test_dark_nonconstant_image_is_labeled_by_documented_threshold() -> None:
    image = np.linspace(0.02, 0.12, 100, dtype=np.float32).reshape(10, 10)

    result = analyze_intensity(image)

    assert result.qualitative_label == "dark"
    assert "mean < 0.10" in result.triggered_rules


def test_bright_nonconstant_image_is_labeled_by_documented_threshold() -> None:
    image = np.linspace(0.88, 0.98, 100, dtype=np.float32).reshape(10, 10)

    result = analyze_intensity(image)

    assert result.qualitative_label == "bright"
    assert "mean > 0.90" in result.triggered_rules


def test_low_contrast_image_is_labeled_without_accuracy_claim() -> None:
    image = np.linspace(0.47, 0.53, 100, dtype=np.float32).reshape(10, 10)

    result = analyze_intensity(image)
    payload = result.to_dict()

    assert result.qualitative_label == "low_contrast"
    assert payload["is_accuracy_or_confidence"] is False
    assert payload["metric_kind"] == "no_reference_descriptive"


def test_normal_synthetic_gradient_has_nominal_profile() -> None:
    image = np.linspace(0.05, 0.95, 256, dtype=np.float32).reshape(16, 16)

    result = analyze_intensity(image)

    assert result.qualitative_label == "nominal_intensity_profile"
    assert result.triggered_rules[0] == "no intensity caution threshold triggered"


def test_lower_and_upper_saturation_are_measured_and_warned() -> None:
    image = np.full((10, 10), 0.5, dtype=np.float32)
    image[0, :5] = 0.0
    image[-1, :5] = 1.0

    result = analyze_intensity(image)

    assert _value(result, "lower_saturation_fraction") == pytest.approx(0.05)
    assert _value(result, "upper_saturation_fraction") == pytest.approx(0.05)
    assert any("lower intensity boundary" in warning for warning in result.warnings)
    assert any("upper intensity boundary" in warning for warning in result.warnings)


def test_measurements_have_units_ranges_interpretations_and_no_display_score() -> None:
    result = analyze_intensity(np.linspace(0.1, 0.9, 64).reshape(8, 8))

    for measurement in result.measurements.values():
        assert measurement.units
        assert measurement.expected_range[0] <= measurement.value
        assert measurement.value <= measurement.expected_range[1]
        assert measurement.interpretation
        assert measurement.display_score is None


def test_diagnostics_are_deterministic_and_do_not_mutate_caller() -> None:
    generator = np.random.default_rng(7)
    image = generator.random((12, 13), dtype=np.float32)
    original = image.copy()

    first = analyze_intensity(image)
    second = analyze_intensity(image)

    assert first == second
    np.testing.assert_array_equal(image, original)


def test_tensor_input_is_supported_without_gradient_side_effects() -> None:
    image = torch.linspace(0.0, 1.0, 121).reshape(11, 11).requires_grad_()

    result = analyze_intensity(image)

    assert _value(result, "mean") == pytest.approx(0.5)
    assert image.grad is None


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((1, 1, 2), dtype=np.float32),
        np.array([], dtype=np.float32).reshape(0, 1),
        np.zeros((2, 2), dtype=np.bool_),
        np.zeros((2, 2), dtype=np.complex64),
        np.full((2, 2), np.nan, dtype=np.float32),
        np.full((2, 2), np.inf, dtype=np.float32),
        np.full((2, 2), -0.01, dtype=np.float32),
        np.full((2, 2), 1.01, dtype=np.float32),
    ],
)
def test_invalid_inputs_are_rejected(image: np.ndarray) -> None:
    with pytest.raises(IntensityDiagnosticError):
        analyze_intensity(image)


def test_serialization_is_strict_json_safe_and_versioned() -> None:
    result = analyze_intensity(np.linspace(0.0, 1.0, 64).reshape(8, 8))
    payload = result.to_dict()

    rendered = json.dumps(payload, allow_nan=False, sort_keys=True)

    assert INTENSITY_DIAGNOSTIC_VERSION in rendered
    assert "confidence" in rendered
    assert payload["diagnostic_version"] == INTENSITY_DIAGNOSTIC_VERSION
