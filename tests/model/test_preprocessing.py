from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from semirestore import preprocessing


def _png_bytes(array: np.ndarray, mode: str = "L") -> bytes:
    output = io.BytesIO()
    Image.fromarray(array, mode=mode).save(output, format="PNG")
    return output.getvalue()


def test_uint8_2d_input_is_scaled_to_canonical_tensor() -> None:
    image = np.array([[0, 127], [128, 255]], dtype=np.uint8)

    result = preprocessing.preprocess_sem_image(image)

    expected = torch.tensor([[[[0.0, 127 / 255], [128 / 255, 1.0]]]], dtype=torch.float32)
    torch.testing.assert_close(result.tensor, expected)
    assert result.normalization == "uint8_divide_255"


def test_uint16_input_uses_full_dtype_range() -> None:
    image = np.array([[0, 32768, 65535]], dtype=np.uint16)

    result = preprocessing.preprocess_sem_image(image)

    expected = torch.tensor([[[[0.0, 32768 / 65535, 1.0]]]], dtype=torch.float32)
    torch.testing.assert_close(result.tensor, expected)
    assert result.normalization == "uint16_divide_65535"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_normalized_float_input_is_not_rescaled(dtype: np.dtype[object]) -> None:
    image = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=dtype)

    result = preprocessing.preprocess_sem_image(image)

    torch.testing.assert_close(
        result.tensor,
        torch.tensor([[[[0.0, 0.25], [0.5, 1.0]]]], dtype=torch.float32),
    )


def test_singleton_channel_is_squeezed_and_recorded() -> None:
    image = np.arange(6, dtype=np.uint8).reshape(2, 3, 1)

    result = preprocessing.preprocess_sem_image(image)

    assert result.tensor.shape == (1, 1, 2, 3)
    assert result.channel_conversion == "squeezed_singleton_channel"


def test_supported_pil_grayscale_input() -> None:
    image = Image.fromarray(np.array([[0, 255]], dtype=np.uint8), mode="L")

    result = preprocessing.preprocess_sem_image(image)

    assert result.source_type == "pil"
    assert result.source_mode == "L"
    assert result.original_dtype == "uint8"
    torch.testing.assert_close(result.tensor, torch.tensor([[[[0.0, 1.0]]]]))


def test_encoded_grayscale_image_bytes() -> None:
    content = _png_bytes(np.array([[0, 64], [128, 255]], dtype=np.uint8))

    result = preprocessing.preprocess_sem_image(content)

    assert result.source_type == "bytes"
    assert result.source_mode == "L"
    assert result.tensor.shape == (1, 1, 2, 2)


def test_valid_image_path(tmp_path: Path) -> None:
    path = tmp_path / "sem.png"
    path.write_bytes(_png_bytes(np.array([[0, 255]], dtype=np.uint8)))

    result = preprocessing.preprocess_sem_image(path)

    assert result.source_type == "path"
    assert result.original_width == 2
    assert result.original_height == 1


def test_tensor_contract_is_deterministic_contiguous_cpu_float32() -> None:
    image = np.arange(24, dtype=np.uint8).reshape(4, 6)[:, ::2]

    first = preprocessing.preprocess_sem_image(image)
    second = preprocessing.preprocess_sem_image(image)

    assert first.tensor.shape == (1, 1, 4, 3)
    assert first.tensor.dtype == torch.float32
    assert first.tensor.device.type == "cpu"
    assert first.tensor.is_contiguous()
    assert float(first.tensor.min()) >= 0.0
    assert float(first.tensor.max()) <= 1.0
    torch.testing.assert_close(first.tensor, second.tensor, atol=0, rtol=0)


def test_source_metadata_is_serialization_friendly() -> None:
    image = np.array([[0.25, 0.75]], dtype=np.float64)

    metadata = preprocessing.preprocess_sem_image(image).metadata()

    assert metadata == {
        "original_width": 2,
        "original_height": 1,
        "original_dtype": "float64",
        "source_type": "numpy",
        "source_mode": None,
        "original_intensity_min": 0.25,
        "original_intensity_max": 0.75,
        "normalization": "float64_to_float32_identity_0_1",
        "channel_conversion": "none",
        "warnings": [],
        "preprocessing_version": "semirestore-preprocessing-v1",
    }


def test_constant_image_is_valid_with_warning() -> None:
    result = preprocessing.preprocess_sem_image(np.full((2, 3), 7, dtype=np.uint8))

    assert result.tensor.shape == (1, 1, 2, 3)
    assert result.warnings == ("Input image is constant; dynamic range is zero.",)


def test_identical_rgb_channels_are_collapsed_and_recorded() -> None:
    channel = np.array([[0, 255]], dtype=np.uint8)
    image = np.stack((channel, channel, channel), axis=-1)

    result = preprocessing.preprocess_sem_image(image)

    assert result.channel_conversion == "collapsed_identical_rgb"
    assert "collapsed" in result.warnings[0]
    assert result.tensor.shape == (1, 1, 1, 2)


def test_input_array_is_not_modified() -> None:
    image = np.array([[0.0, 0.5], [0.75, 1.0]], dtype=np.float32)
    original = image.copy()

    result = preprocessing.preprocess_sem_image(image)
    result.tensor.zero_()

    np.testing.assert_array_equal(image, original)


def test_missing_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(preprocessing.ImageValidationError, match="does not exist"):
        preprocessing.preprocess_sem_image(tmp_path / "missing.png")


def test_directory_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(preprocessing.ImageValidationError, match="not a regular file"):
        preprocessing.preprocess_sem_image(tmp_path)


def test_empty_bytes_are_rejected() -> None:
    with pytest.raises(preprocessing.ImageDecodeError, match="empty"):
        preprocessing.preprocess_sem_image(b"")


def test_corrupt_encoded_bytes_are_rejected() -> None:
    with pytest.raises(preprocessing.ImageDecodeError, match="not a decodable"):
        preprocessing.preprocess_sem_image(b"not-an-image")


@pytest.mark.parametrize(
    "image",
    [
        np.empty((0, 2), dtype=np.uint8),
        np.empty((2, 0), dtype=np.uint8),
    ],
)
def test_empty_array_is_rejected(image: np.ndarray) -> None:
    with pytest.raises(preprocessing.ImageValidationError, match="empty"):
        preprocessing.preprocess_sem_image(image)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_float_is_rejected(value: float) -> None:
    with pytest.raises(preprocessing.ImageValidationError, match="NaN or infinity"):
        preprocessing.preprocess_sem_image(np.array([[value]], dtype=np.float32))


def test_negative_float_is_rejected() -> None:
    with pytest.raises(preprocessing.ImageValidationError, match=r"within \[0, 1\]"):
        preprocessing.preprocess_sem_image(np.array([[-0.01, 0.5]], dtype=np.float32))


def test_float_above_one_is_rejected_without_reinterpretation() -> None:
    with pytest.raises(preprocessing.ImageValidationError, match=r"within \[0, 1\]"):
        preprocessing.preprocess_sem_image(np.array([[0.0, 255.0]], dtype=np.float32))


@pytest.mark.parametrize(
    "image",
    [
        np.array([[True]], dtype=np.bool_),
        np.array([[1]], dtype=np.int16),
        np.array([[1]], dtype=np.uint32),
        np.array([[0.5]], dtype=np.float16),
        np.array([[1 + 2j]], dtype=np.complex64),
        np.array([[object()]], dtype=object),
    ],
)
def test_unsupported_dtype_is_rejected(image: np.ndarray) -> None:
    with pytest.raises(preprocessing.UnsupportedInputError, match="Unsupported input dtype"):
        preprocessing.preprocess_sem_image(image)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((3,), dtype=np.uint8),
        np.zeros((1, 1, 1, 1), dtype=np.uint8),
        np.zeros((2, 2, 2), dtype=np.uint8),
    ],
)
def test_unsupported_dimensions_or_channel_layout_are_rejected(image: np.ndarray) -> None:
    with pytest.raises(preprocessing.UnsupportedInputError):
        preprocessing.preprocess_sem_image(image)


def test_nonidentical_rgb_is_rejected() -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[..., 1] = 1

    with pytest.raises(preprocessing.UnsupportedInputError, match="Non-identical RGB"):
        preprocessing.preprocess_sem_image(image)


@pytest.mark.parametrize(
    ("shape", "limits", "message"),
    [
        ((2, 4), preprocessing.PreprocessingLimits(max_width=3), "maximum width"),
        ((4, 2), preprocessing.PreprocessingLimits(max_height=3), "maximum height"),
        ((4, 4), preprocessing.PreprocessingLimits(max_pixels=15), "maximum pixel count"),
    ],
)
def test_dimension_and_pixel_limits_are_enforced(
    shape: tuple[int, int],
    limits: preprocessing.PreprocessingLimits,
    message: str,
) -> None:
    with pytest.raises(preprocessing.ImageResourceError, match=message):
        preprocessing.preprocess_sem_image(np.zeros(shape, dtype=np.uint8), limits=limits)


def test_encoded_byte_limit_is_enforced_before_decode() -> None:
    limits = preprocessing.PreprocessingLimits(max_encoded_bytes=4)

    with pytest.raises(preprocessing.ImageResourceError, match="Encoded image size"):
        preprocessing.preprocess_sem_image(b"12345", limits=limits)


def test_decoded_dimension_limit_remains_a_resource_error() -> None:
    content = _png_bytes(np.zeros((2, 4), dtype=np.uint8))
    limits = preprocessing.PreprocessingLimits(max_width=3)

    with pytest.raises(preprocessing.ImageResourceError, match="maximum width"):
        preprocessing.preprocess_sem_image(content, limits=limits)


def test_unsupported_pil_mode_is_rejected() -> None:
    image = Image.new("P", (2, 2))

    with pytest.raises(preprocessing.UnsupportedInputError, match="Unsupported PIL mode"):
        preprocessing.preprocess_sem_image(image)


def test_pillow_decompression_bomb_is_reported_as_resource_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def bomb(_stream: object) -> Image.Image:
        raise Image.DecompressionBombError("synthetic bomb")

    monkeypatch.setattr(preprocessing.Image, "open", bomb)

    with pytest.raises(preprocessing.ImageResourceError, match="decompression-bomb"):
        preprocessing.preprocess_sem_image(b"encoded")
