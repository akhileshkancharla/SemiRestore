"""Validated CPU preprocessing for single-image SEM restoration inputs."""

from __future__ import annotations

import io
import stat
import warnings as warning_control
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

PREPROCESSING_VERSION = "semirestore-preprocessing-v1"
SUPPORTED_PIL_MODES = frozenset({"L", "I;16", "I;16L", "I;16B", "F", "RGB"})


class PreprocessingError(ValueError):
    """Base class for scientific input-validation failures."""


class UnsupportedInputError(PreprocessingError):
    """Raised when an input type, layout, mode, or dtype is unsupported."""


class ImageDecodeError(PreprocessingError):
    """Raised when encoded image input cannot be decoded safely."""


class ImageValidationError(PreprocessingError):
    """Raised when decoded intensity or shape values violate the contract."""


class ImageResourceError(PreprocessingError):
    """Raised when input dimensions or encoded size exceed configured limits."""


@dataclass(frozen=True, slots=True)
class PreprocessingLimits:
    """Explicit resource limits applied before allocating the model tensor."""

    max_width: int = 8192
    max_height: int = 8192
    max_pixels: int = 16_777_216
    max_encoded_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        values = (self.max_width, self.max_height, self.max_pixels, self.max_encoded_bytes)
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError("All preprocessing resource limits must be positive integers")


DEFAULT_LIMITS = PreprocessingLimits()


@dataclass(frozen=True, slots=True)
class PreprocessingResult:
    """Canonical model tensor plus serialization-friendly source metadata."""

    tensor: torch.Tensor
    original_width: int
    original_height: int
    original_dtype: str
    source_type: str
    source_mode: str | None
    original_intensity_min: float
    original_intensity_max: float
    normalization: str
    channel_conversion: str
    warnings: tuple[str, ...]
    preprocessing_version: str = PREPROCESSING_VERSION

    def metadata(self) -> dict[str, object]:
        """Return preprocessing metadata without tensor or imaging-library objects."""

        return {
            "original_width": self.original_width,
            "original_height": self.original_height,
            "original_dtype": self.original_dtype,
            "source_type": self.source_type,
            "source_mode": self.source_mode,
            "original_intensity_min": self.original_intensity_min,
            "original_intensity_max": self.original_intensity_max,
            "normalization": self.normalization,
            "channel_conversion": self.channel_conversion,
            "warnings": list(self.warnings),
            "preprocessing_version": self.preprocessing_version,
        }


ImageInput: TypeAlias = Path | bytes | np.ndarray | Image.Image


def _validate_dimensions(width: int, height: int, limits: PreprocessingLimits) -> None:
    if width < 1 or height < 1:
        raise ImageValidationError(f"Input image is empty: width={width}, height={height}")
    if width > limits.max_width:
        raise ImageResourceError(
            f"Input width {width} exceeds maximum width {limits.max_width}"
        )
    if height > limits.max_height:
        raise ImageResourceError(
            f"Input height {height} exceeds maximum height {limits.max_height}"
        )
    pixels = width * height
    if pixels > limits.max_pixels:
        raise ImageResourceError(
            f"Input pixel count {pixels} exceeds maximum pixel count {limits.max_pixels}"
        )


def _read_path(path: Path, limits: PreprocessingLimits) -> bytes:
    try:
        path_stat = path.stat()
    except FileNotFoundError as error:
        raise ImageValidationError(f"Input image path does not exist: {path}") from error
    except OSError as error:
        raise ImageValidationError(f"Could not inspect input image path: {path}") from error
    if not stat.S_ISREG(path_stat.st_mode):
        raise ImageValidationError(f"Input image path is not a regular file: {path}")
    if path_stat.st_size > limits.max_encoded_bytes:
        raise ImageResourceError(
            f"Encoded image size {path_stat.st_size} exceeds maximum "
            f"{limits.max_encoded_bytes} bytes"
        )
    try:
        with path.open("rb") as handle:
            content = handle.read(limits.max_encoded_bytes + 1)
    except OSError as error:
        raise ImageDecodeError(f"Could not read encoded image path: {path}") from error
    if len(content) > limits.max_encoded_bytes:
        raise ImageResourceError(
            f"Encoded image exceeds maximum {limits.max_encoded_bytes} bytes"
        )
    return content


def _validate_pil_mode(mode: str) -> None:
    if mode not in SUPPORTED_PIL_MODES:
        raise UnsupportedInputError(
            f"Unsupported PIL mode {mode!r}; expected grayscale L/I;16/F or identical RGB"
        )


def _pil_to_array(
    image: Image.Image,
    *,
    limits: PreprocessingLimits,
) -> tuple[np.ndarray, str]:
    mode = image.mode
    _validate_pil_mode(mode)
    _validate_dimensions(image.width, image.height, limits)
    try:
        image.load()
        array = np.array(image, copy=True)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ImageResourceError("Pillow rejected a decompression-bomb-sized image") from error
    except (OSError, ValueError) as error:
        raise ImageDecodeError("Could not decode image pixels") from error
    return array, mode


def _decode_encoded_image(
    content: bytes,
    *,
    limits: PreprocessingLimits,
) -> tuple[np.ndarray, str]:
    if not content:
        raise ImageDecodeError("Encoded image bytes are empty")
    if len(content) > limits.max_encoded_bytes:
        raise ImageResourceError(
            f"Encoded image size {len(content)} exceeds maximum {limits.max_encoded_bytes} bytes"
        )
    try:
        with warning_control.catch_warnings():
            warning_control.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as image:
                return _pil_to_array(image, limits=limits)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ImageResourceError("Pillow rejected a decompression-bomb-sized image") from error
    except PreprocessingError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        raise ImageDecodeError("Encoded bytes are not a decodable supported image") from error


def _validate_array_layout(array: np.ndarray, limits: PreprocessingLimits) -> tuple[int, int]:
    if array.ndim not in (2, 3):
        raise UnsupportedInputError(
            f"Expected a 2D grayscale or HxWx1/identical-RGB array; got shape {array.shape}"
        )
    if array.size == 0:
        raise ImageValidationError(f"Input image array is empty: shape {array.shape}")
    height, width = int(array.shape[0]), int(array.shape[1])
    _validate_dimensions(width, height, limits)
    if array.ndim == 3 and array.shape[2] not in (1, 3):
        raise UnsupportedInputError(
            f"Unsupported channel layout {array.shape}; expected HxWx1 or identical HxWx3"
        )
    return width, height


def _dtype_normalization(dtype: np.dtype[object]) -> str:
    if dtype.kind == "u" and dtype.itemsize == 1:
        return "uint8_divide_255"
    if dtype.kind == "u" and dtype.itemsize == 2:
        return "uint16_divide_65535"
    if dtype.kind == "f" and dtype.itemsize == 4:
        return "float32_identity_0_1"
    if dtype.kind == "f" and dtype.itemsize == 8:
        return "float64_to_float32_identity_0_1"
    raise UnsupportedInputError(
        f"Unsupported input dtype {dtype}; expected uint8, uint16, float32, or float64"
    )


def _collapse_channels(array: np.ndarray) -> tuple[np.ndarray, str, tuple[str, ...]]:
    if array.ndim == 2:
        return array, "none", ()
    if array.shape[2] == 1:
        return array[..., 0], "squeezed_singleton_channel", ()
    first = array[..., 0]
    if not (np.array_equal(first, array[..., 1]) and np.array_equal(first, array[..., 2])):
        raise UnsupportedInputError(
            "Non-identical RGB channels are not accepted as scientific grayscale input"
        )
    return (
        first,
        "collapsed_identical_rgb",
        ("Identical RGB channels were collapsed to one grayscale channel.",),
    )


def _normalize_array(
    array: np.ndarray,
    normalization: str,
) -> np.ndarray:
    if normalization == "uint8_divide_255":
        return array.astype(np.float32, copy=True) / np.float32(255.0)
    if normalization == "uint16_divide_65535":
        return array.astype(np.float32, copy=True) / np.float32(65535.0)
    return array.astype(np.float32, copy=True)


def preprocess_sem_image(
    image: ImageInput,
    *,
    limits: PreprocessingLimits = DEFAULT_LIMITS,
) -> PreprocessingResult:
    """Validate one SEM image and return contiguous CPU float32 NCHW input.

    Unsigned integers use their full dtype range. Floating-point arrays must
    already be finite and within ``[0, 1]``; no clipping or per-image min-max
    normalization is performed.
    """

    source_mode: str | None = None
    if isinstance(image, Path):
        source_type = "path"
        array, source_mode = _decode_encoded_image(
            _read_path(image, limits),
            limits=limits,
        )
    elif isinstance(image, bytes):
        source_type = "bytes"
        array, source_mode = _decode_encoded_image(image, limits=limits)
    elif isinstance(image, np.ndarray):
        source_type = "numpy"
        array = image
    elif isinstance(image, Image.Image):
        source_type = "pil"
        array, source_mode = _pil_to_array(image, limits=limits)
    else:
        raise UnsupportedInputError(
            "Unsupported input type; expected pathlib.Path, bytes, NumPy array, or PIL Image"
        )

    width, height = _validate_array_layout(array, limits)
    normalization = _dtype_normalization(array.dtype)
    if not np.isfinite(array).all():
        raise ImageValidationError("Input image contains NaN or infinity")
    grayscale, channel_conversion, conversion_warnings = _collapse_channels(array)
    original_min = float(grayscale.min())
    original_max = float(grayscale.max())
    if array.dtype.kind == "f" and (original_min < 0.0 or original_max > 1.0):
        raise ImageValidationError(
            f"Floating-point input must already be within [0, 1]; "
            f"received range [{original_min}, {original_max}]"
        )

    normalized = np.ascontiguousarray(_normalize_array(grayscale, normalization))
    if not np.isfinite(normalized).all() or normalized.min() < 0.0 or normalized.max() > 1.0:
        raise ImageValidationError("Normalized input is not finite within [0, 1]")
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).contiguous()
    result_warnings = list(conversion_warnings)
    if original_min == original_max:
        result_warnings.append("Input image is constant; dynamic range is zero.")

    return PreprocessingResult(
        tensor=tensor,
        original_width=width,
        original_height=height,
        original_dtype=str(array.dtype),
        source_type=source_type,
        source_mode=source_mode,
        original_intensity_min=original_min,
        original_intensity_max=original_max,
        normalization=normalization,
        channel_conversion=channel_conversion,
        warnings=tuple(result_warnings),
    )


__all__ = [
    "DEFAULT_LIMITS",
    "ImageDecodeError",
    "ImageResourceError",
    "ImageValidationError",
    "PREPROCESSING_VERSION",
    "PreprocessingError",
    "PreprocessingLimits",
    "PreprocessingResult",
    "UnsupportedInputError",
    "preprocess_sem_image",
]
