from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from PIL import Image

from semirestore import postprocessing


def _result(values: list[list[float]]) -> postprocessing.PostprocessingResult:
    tensor = torch.tensor([[values]], dtype=torch.float32)
    return postprocessing.postprocess_restoration(tensor)


def test_valid_cpu_float32_output_returns_canonical_array() -> None:
    tensor = torch.tensor([[[[0.0, 0.5], [0.75, 1.0]]]], dtype=torch.float32)

    result = postprocessing.postprocess_restoration(
        tensor,
        original_width=1,
        original_height=1,
    )

    assert result.image.shape == (2, 2)
    assert result.image.dtype == np.float32
    assert result.image.flags.c_contiguous
    np.testing.assert_array_equal(result.image, tensor.numpy()[0, 0])
    assert result.restored_width == 2
    assert result.restored_height == 2


def test_valid_cpu_float64_output_converts_to_float32() -> None:
    tensor = torch.tensor([[[[0.125, 0.875]]]], dtype=torch.float64)

    result = postprocessing.postprocess_restoration(tensor)

    assert result.image.dtype == np.float32
    assert result.source_tensor_dtype == "torch.float64"
    np.testing.assert_array_equal(result.image, np.array([[0.125, 0.875]], dtype=np.float32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_output_is_detached_and_transferred_to_cpu() -> None:
    tensor = torch.tensor([[[[0.25, 0.75]]]], device="cuda", requires_grad=True)

    result = postprocessing.postprocess_restoration(tensor)

    assert result.source_tensor_device.startswith("cuda")
    assert isinstance(result.image, np.ndarray)
    assert result.image.dtype == np.float32


def test_expected_two_x_dimensions_are_enforced_and_recorded() -> None:
    result = postprocessing.postprocess_restoration(
        torch.zeros((1, 1, 6, 10)),
        original_width=5,
        original_height=3,
    )

    assert result.original_width == 5
    assert result.original_height == 3
    assert result.scale_factor == 2
    assert result.metadata()["restored_width"] == 10


def test_values_below_and_above_range_are_clipped_with_statistics() -> None:
    tensor = torch.tensor([[[[-0.25, 0.0], [1.0, 1.5]]]])

    result = postprocessing.postprocess_restoration(tensor)

    np.testing.assert_array_equal(result.image, np.array([[0.0, 0.0], [1.0, 1.0]]))
    assert result.clipping.raw_minimum == pytest.approx(-0.25)
    assert result.clipping.raw_maximum == pytest.approx(1.5)
    assert result.clipping.clipped_minimum == 0.0
    assert result.clipping.clipped_maximum == 1.0
    assert result.clipping.values_below_zero == 1
    assert result.clipping.fraction_below_zero == pytest.approx(0.25)
    assert result.clipping.values_above_one == 1
    assert result.clipping.fraction_above_one == pytest.approx(0.25)
    assert result.clipping.total_values == 4
    assert result.clipping.clipping_occurred is True
    assert "Clipped 2 of 4" in result.warnings[0]


def test_no_clipping_case_is_recorded() -> None:
    result = _result([[0.0, 0.5, 1.0]])

    assert result.clipping.clipping_occurred is False
    assert result.clipping.values_below_zero == 0
    assert result.clipping.values_above_one == 0
    assert result.warnings == ()


def test_caller_tensor_remains_unchanged() -> None:
    tensor = torch.tensor([[[[-1.0, 2.0]]]])
    original = tensor.clone()

    result = postprocessing.postprocess_restoration(tensor)
    result.image.fill(0.5)

    torch.testing.assert_close(tensor, original, atol=0, rtol=0)


def test_gradient_bearing_tensor_is_safely_detached() -> None:
    tensor = torch.tensor([[[[0.25, 0.75]]]], requires_grad=True)

    result = postprocessing.postprocess_restoration(tensor)

    assert tensor.requires_grad
    assert isinstance(result.image, np.ndarray)
    np.testing.assert_array_equal(result.image, np.array([[0.25, 0.75]], dtype=np.float32))


def test_metadata_is_serialization_friendly() -> None:
    result = postprocessing.postprocess_restoration(
        torch.tensor([[[[-0.5, 1.5], [0.0, 1.0]]]], dtype=torch.float64),
        original_width=1,
        original_height=1,
    )

    assert result.metadata() == {
        "raw_minimum": -0.5,
        "raw_maximum": 1.5,
        "clipped_minimum": 0.0,
        "clipped_maximum": 1.0,
        "values_below_zero": 1,
        "fraction_below_zero": 0.25,
        "values_above_one": 1,
        "fraction_above_one": 0.25,
        "total_values": 4,
        "clipping_occurred": True,
        "original_width": 1,
        "original_height": 1,
        "restored_width": 2,
        "restored_height": 2,
        "scale_factor": 2,
        "source_tensor_dtype": "torch.float64",
        "source_tensor_device": "cpu",
        "postprocessing_version": "semirestore-postprocessing-v1",
        "warnings": ["Clipped 2 of 4 values to [0, 1] (1 below zero, 1 above one)."],
    }


def test_non_tensor_is_rejected() -> None:
    with pytest.raises(postprocessing.UnsupportedOutputError, match="PyTorch tensor"):
        postprocessing.postprocess_restoration(np.zeros((1, 1, 2, 2), dtype=np.float32))


@pytest.mark.parametrize(
    "tensor",
    [
        torch.zeros((2, 2)),
        torch.zeros((1, 2, 2)),
        torch.zeros((1, 1, 1, 2, 2)),
    ],
)
def test_invalid_rank_is_rejected(tensor: torch.Tensor) -> None:
    with pytest.raises(postprocessing.OutputValidationError, match="rank 4"):
        postprocessing.postprocess_restoration(tensor)


def test_invalid_batch_size_is_rejected() -> None:
    with pytest.raises(postprocessing.OutputValidationError, match="batch size"):
        postprocessing.postprocess_restoration(torch.zeros((2, 1, 2, 2)))


def test_invalid_channel_count_is_rejected() -> None:
    with pytest.raises(postprocessing.OutputValidationError, match="channel count"):
        postprocessing.postprocess_restoration(torch.zeros((1, 2, 2, 2)))


@pytest.mark.parametrize("shape", [(1, 1, 0, 2), (1, 1, 2, 0)])
def test_zero_spatial_dimension_is_rejected(shape: tuple[int, ...]) -> None:
    with pytest.raises(postprocessing.OutputValidationError, match="nonzero"):
        postprocessing.postprocess_restoration(torch.empty(shape))


@pytest.mark.parametrize(
    "tensor",
    [
        torch.zeros((1, 1, 2, 2), dtype=torch.int32),
        torch.zeros((1, 1, 2, 2), dtype=torch.bool),
        torch.zeros((1, 1, 2, 2), dtype=torch.complex64),
    ],
)
def test_integer_boolean_and_complex_tensors_are_rejected(tensor: torch.Tensor) -> None:
    with pytest.raises(postprocessing.UnsupportedOutputError, match="Unsupported tensor dtype"):
        postprocessing.postprocess_restoration(tensor)


def test_sparse_tensor_is_rejected() -> None:
    tensor = torch.sparse_coo_tensor(
        torch.tensor([[0], [0], [0], [0]]),
        torch.tensor([0.5]),
        size=(1, 1, 2, 2),
        check_invariants=False,
    )

    with pytest.raises(postprocessing.UnsupportedOutputError, match="layout"):
        postprocessing.postprocess_restoration(tensor)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_values_are_rejected(value: float) -> None:
    tensor = torch.tensor([[[[value]]]])

    with pytest.raises(postprocessing.OutputValidationError, match="NaN or infinity"):
        postprocessing.postprocess_restoration(tensor)


def test_incorrect_expected_scale_is_rejected_without_resizing() -> None:
    tensor = torch.zeros((1, 1, 5, 8))

    with pytest.raises(postprocessing.OutputValidationError, match="required 2x scale"):
        postprocessing.postprocess_restoration(
            tensor,
            original_width=4,
            original_height=3,
        )


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (None, 1),
        (1, None),
        (0, 1),
        (1, -1),
        (1.0, 1),
        (True, 1),
    ],
)
def test_invalid_original_dimensions_are_rejected(
    width: object,
    height: object,
) -> None:
    with pytest.raises(postprocessing.OutputValidationError, match="Original width and height"):
        postprocessing.postprocess_restoration(
            torch.zeros((1, 1, 2, 2)),
            original_width=width,  # type: ignore[arg-type]
            original_height=height,  # type: ignore[arg-type]
        )


def test_uint8_quantization_is_deterministic_round_half_up() -> None:
    result = _result([[0.0, 0.5, 1.0]])

    quantized = result.quantize(bit_depth=8)

    np.testing.assert_array_equal(quantized, np.array([[0, 128, 255]], dtype=np.uint8))
    assert quantized.flags.c_contiguous


def test_uint16_quantization_is_deterministic_round_half_up() -> None:
    result = _result([[0.0, 0.5, 1.0]])

    quantized = result.quantize(bit_depth=16)

    np.testing.assert_array_equal(quantized, np.array([[0, 32768, 65535]], dtype=np.uint16))
    assert quantized.flags.c_contiguous


@pytest.mark.parametrize(
    ("bit_depth", "expected_mode", "expected_dtype"),
    [(8, "L", np.uint8), (16, "I;16", np.uint16)],
)
def test_png_round_trip_preserves_quantized_values_dimensions_and_mode(
    bit_depth: int,
    expected_mode: str,
    expected_dtype: np.dtype[object],
) -> None:
    result = _result([[0.0, 0.1, 0.5], [0.9, 1.0, 0.25]])
    expected = result.quantize(bit_depth=bit_depth)

    encoded = result.encode(encoding="png", bit_depth=bit_depth)

    assert encoded.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(io.BytesIO(encoded)) as decoded:
        actual = np.array(decoded)
        assert decoded.size == (3, 2)
        assert decoded.mode == expected_mode
    assert actual.dtype == expected_dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("bit_depth", [1, 12, 32, 8.0, True])
def test_invalid_bit_depth_is_rejected(bit_depth: object) -> None:
    with pytest.raises(postprocessing.UnsupportedOutputError, match="exactly 8 or 16"):
        _result([[0.5]]).encode(bit_depth=bit_depth)  # type: ignore[arg-type]


@pytest.mark.parametrize("encoding", ["jpeg", "tiff", "", 1])
def test_invalid_or_lossy_encoding_is_rejected(encoding: object) -> None:
    with pytest.raises(postprocessing.UnsupportedOutputError, match="Only lossless PNG"):
        _result([[0.5]]).encode(encoding=encoding)  # type: ignore[arg-type]
