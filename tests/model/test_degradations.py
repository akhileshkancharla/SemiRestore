from __future__ import annotations

import json

import pytest
import torch
from torch.nn import functional as F

from semirestore.data import DatasetValidationError
from semirestore.degradations import (
    DEGRADATION_VERSION,
    DegradationConfig,
    ParameterRange,
    degrade_sem_image,
    derive_degradation_seed,
)


def _target() -> torch.Tensor:
    return torch.linspace(-0.25, 1.25, 80, dtype=torch.float32).reshape(1, 8, 10)


def _config(**overrides: object) -> DegradationConfig:
    values: dict[str, object] = {
        "blur_sigma": ParameterRange(0.4, 0.9),
        "gaussian_noise_std": ParameterRange(0.01, 0.04),
        "speckle_std": ParameterRange(0.01, 0.03),
        "additive_bias": ParameterRange(-0.02, 0.02),
        "downsample_modes": ("area", "bicubic"),
        "operation_order": ("blur", "gaussian", "speckle", "downsample"),
        "randomize_order": True,
    }
    values.update(overrides)
    return DegradationConfig(**values)


def test_identical_seed_and_identity_produce_identical_output_and_metadata() -> None:
    first = degrade_sem_image(_target(), _config(), sample_id="sample-a", base_seed=2026)
    second = degrade_sem_image(_target(), _config(), sample_id="sample-a", base_seed=2026)

    torch.testing.assert_close(first.tensor, second.tensor, rtol=0, atol=0)
    assert first.metadata == second.metadata


def test_different_sample_ids_produce_expected_noise_variation() -> None:
    first = degrade_sem_image(_target(), _config(), sample_id="sample-a", base_seed=2026)
    second = degrade_sem_image(_target(), _config(), sample_id="sample-b", base_seed=2026)

    assert not torch.equal(first.tensor, second.tensor)
    assert first.metadata["derived_seed"] != second.metadata["derived_seed"]


def test_explicit_operation_order_is_honored() -> None:
    order = ("downsample", "gaussian", "speckle", "blur")
    result = degrade_sem_image(
        _target(),
        _config(operation_order=order, randomize_order=False),
        sample_id="ordered",
        base_seed=1,
    )

    assert result.metadata["operation_order"] == list(order)


def test_downsample_only_matches_historical_area_interpolation() -> None:
    config = _config(
        blur_sigma=ParameterRange(0.0, 0.0),
        gaussian_noise_std=ParameterRange(0.0, 0.0),
        speckle_std=ParameterRange(0.0, 0.0),
        additive_bias=ParameterRange(0.0, 0.0),
        downsample_modes=("area",),
        randomize_order=False,
    )
    target = _target()

    result = degrade_sem_image(target, config, sample_id="area", base_seed=5)

    expected = F.interpolate(target[None], size=(4, 5), mode="area")[0]
    torch.testing.assert_close(result.tensor, expected, rtol=0, atol=0)
    assert result.tensor.shape == (1, 4, 5)


def test_blur_only_before_downsample_changes_impulse_response() -> None:
    target = torch.zeros((1, 10, 10), dtype=torch.float32)
    target[:, 5, 5] = 1.0
    config = _config(
        blur_sigma=ParameterRange(1.0, 1.0),
        gaussian_noise_std=ParameterRange(0.0, 0.0),
        speckle_std=ParameterRange(0.0, 0.0),
        additive_bias=ParameterRange(0.0, 0.0),
        downsample_modes=("area",),
        randomize_order=False,
    )

    result = degrade_sem_image(target, config, sample_id="blur", base_seed=5)
    plain = F.interpolate(target[None], size=(5, 5), mode="area")[0]

    assert not torch.equal(result.tensor, plain)
    assert result.metadata["blur_sigma"] == 1.0


def test_noise_only_is_applied_without_clipping() -> None:
    target = torch.ones((1, 8, 10), dtype=torch.float32)
    config = _config(
        blur_sigma=ParameterRange(0.0, 0.0),
        gaussian_noise_std=ParameterRange(0.5, 0.5),
        speckle_std=ParameterRange(0.0, 0.0),
        additive_bias=ParameterRange(0.4, 0.4),
        downsample_modes=("area",),
        operation_order=("downsample", "gaussian", "speckle", "blur"),
        randomize_order=False,
    )

    result = degrade_sem_image(target, config, sample_id="noise", base_seed=17)

    assert torch.any(result.tensor > 1.0)
    assert result.metadata["output_clipped"] is False


def test_combined_degradation_has_exact_two_times_dimensions() -> None:
    result = degrade_sem_image(_target(), _config(), sample_id="combined", base_seed=2026)

    assert result.tensor.shape == (1, 4, 5)
    assert result.metadata["input_shape"] == [1, 8, 10]
    assert result.metadata["output_shape"] == [1, 4, 5]


def test_raw_out_of_range_values_remain_out_of_range() -> None:
    target = torch.tensor(
        [[[-4.0, -4.0, 3.0, 3.0], [-4.0, -4.0, 3.0, 3.0]]], dtype=torch.float32
    )
    config = _config(
        blur_sigma=ParameterRange(0.0, 0.0),
        gaussian_noise_std=ParameterRange(0.0, 0.0),
        speckle_std=ParameterRange(0.0, 0.0),
        additive_bias=ParameterRange(0.0, 0.0),
        downsample_modes=("area",),
        randomize_order=False,
    )

    result = degrade_sem_image(target, config, sample_id="raw", base_seed=0)

    assert float(result.tensor.min()) == -4.0
    assert float(result.tensor.max()) == 3.0


def test_metadata_is_complete_reproducible_and_json_serializable() -> None:
    result = degrade_sem_image(_target(), _config(), sample_id="meta", base_seed=9, epoch=3)

    rendered = json.dumps(result.metadata, sort_keys=True)

    assert DEGRADATION_VERSION in rendered
    assert result.metadata["sample_id"] == "meta"
    assert result.metadata["epoch"] == 3
    assert set(result.metadata["operation_order"]) == {
        "blur",
        "gaussian",
        "speckle",
        "downsample",
    }


def test_caller_target_is_not_mutated() -> None:
    target = _target()
    original = target.clone()

    degrade_sem_image(target, _config(), sample_id="immutable", base_seed=3)

    torch.testing.assert_close(target, original, rtol=0, atol=0)


def test_sample_seed_is_independent_of_processing_order() -> None:
    config = _config()
    forward = {
        sample_id: degrade_sem_image(
            _target(), config, sample_id=sample_id, base_seed=44, epoch=2
        ).tensor
        for sample_id in ("a", "b", "c")
    }
    reverse = {
        sample_id: degrade_sem_image(
            _target(), config, sample_id=sample_id, base_seed=44, epoch=2
        ).tensor
        for sample_id in ("c", "b", "a")
    }

    for sample_id in forward:
        torch.testing.assert_close(forward[sample_id], reverse[sample_id], rtol=0, atol=0)


def test_epoch_is_part_of_seed_derivation() -> None:
    assert derive_degradation_seed(7, "sample", epoch=0) != derive_degradation_seed(
        7, "sample", epoch=1
    )


def test_historical_mapping_is_loaded_without_changing_semantics() -> None:
    config = DegradationConfig.from_mapping(
        {
            "schema_version": 1,
            "blur_sigma": {"low": 0.1, "high": 1.0},
            "gaussian_noise_std": {"low": 0.0, "high": 0.1},
            "speckle_std": {"low": 0.0, "high": 0.2},
            "additive_bias": {"low": -0.05, "high": 0.05},
            "downsample_modes": ["area", "bicubic"],
            "operation_order": "uniform_random_per_sample",
            "output_clipped": False,
        }
    )

    assert config.randomize_order is True
    assert config.downsample_modes == ("area", "bicubic")
    assert config.additive_bias.low == -0.05


@pytest.mark.parametrize(
    "config",
    [
        lambda: DegradationConfig(blur_sigma=ParameterRange(-0.1, 0.2)),
        lambda: DegradationConfig(downsample_modes=("nearest",)),
        lambda: DegradationConfig(operation_order=("blur", "blur", "speckle", "downsample")),
        lambda: DegradationConfig(scale=3),
    ],
)
def test_invalid_configuration_is_rejected(config: object) -> None:
    with pytest.raises(DatasetValidationError):
        config()  # type: ignore[operator]


@pytest.mark.parametrize(
    "target",
    [
        torch.zeros((2, 8, 8)),
        torch.zeros((1, 7, 8)),
        torch.zeros((1, 8, 8), dtype=torch.int32),
        torch.full((1, 8, 8), float("nan")),
    ],
)
def test_invalid_target_is_rejected(target: torch.Tensor) -> None:
    with pytest.raises(DatasetValidationError):
        degrade_sem_image(target, _config(), sample_id="bad", base_seed=1)
