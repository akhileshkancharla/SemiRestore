"""Validated postprocessing for one raw SemiRestore model output."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

POSTPROCESSING_VERSION = "semirestore-postprocessing-v1"
RESTORATION_SCALE_FACTOR = 2
SUPPORTED_TENSOR_DTYPES = frozenset(
    {torch.float16, torch.bfloat16, torch.float32, torch.float64}
)
SUPPORTED_PNG_BIT_DEPTHS = frozenset({8, 16})


class PostprocessingError(ValueError):
    """Base class for restoration-output postprocessing failures."""


class UnsupportedOutputError(PostprocessingError):
    """Raised when an output representation or tensor type is unsupported."""


class OutputValidationError(PostprocessingError):
    """Raised when a raw output violates the scientific output contract."""


class OutputEncodingError(PostprocessingError):
    """Raised when lossless output encoding fails."""


@dataclass(frozen=True, slots=True)
class ClippingMetadata:
    """Trace of the explicit clamp from raw model values to ``[0, 1]``."""

    raw_minimum: float
    raw_maximum: float
    clipped_minimum: float
    clipped_maximum: float
    values_below_zero: int
    fraction_below_zero: float
    values_above_one: int
    fraction_above_one: float
    total_values: int
    clipping_occurred: bool


@dataclass(frozen=True, slots=True)
class PostprocessingResult:
    """Canonical restored image plus traceable source and clipping metadata."""

    image: np.ndarray
    restored_width: int
    restored_height: int
    clipping: ClippingMetadata
    original_width: int | None
    original_height: int | None
    scale_factor: int
    source_tensor_dtype: str
    source_tensor_device: str
    warnings: tuple[str, ...]
    postprocessing_version: str = POSTPROCESSING_VERSION

    def metadata(self) -> dict[str, object]:
        """Return serialization-friendly metadata without tensor or image objects."""

        return {
            "raw_minimum": self.clipping.raw_minimum,
            "raw_maximum": self.clipping.raw_maximum,
            "clipped_minimum": self.clipping.clipped_minimum,
            "clipped_maximum": self.clipping.clipped_maximum,
            "values_below_zero": self.clipping.values_below_zero,
            "fraction_below_zero": self.clipping.fraction_below_zero,
            "values_above_one": self.clipping.values_above_one,
            "fraction_above_one": self.clipping.fraction_above_one,
            "total_values": self.clipping.total_values,
            "clipping_occurred": self.clipping.clipping_occurred,
            "original_width": self.original_width,
            "original_height": self.original_height,
            "restored_width": self.restored_width,
            "restored_height": self.restored_height,
            "scale_factor": self.scale_factor,
            "source_tensor_dtype": self.source_tensor_dtype,
            "source_tensor_device": self.source_tensor_device,
            "postprocessing_version": self.postprocessing_version,
            "warnings": list(self.warnings),
        }

    def quantize(self, *, bit_depth: int) -> np.ndarray:
        """Return a deterministic uint8 or uint16 grayscale representation."""

        return quantize_restoration(self, bit_depth=bit_depth)

    def encode(self, *, encoding: str = "png", bit_depth: int = 16) -> bytes:
        """Encode the restored image losslessly in memory."""

        return encode_restoration(self, encoding=encoding, bit_depth=bit_depth)


def _validate_original_dimensions(
    original_width: int | None,
    original_height: int | None,
) -> tuple[int, int] | None:
    if original_width is None and original_height is None:
        return None
    if original_width is None or original_height is None:
        raise OutputValidationError(
            "Original width and height must either both be supplied or both be omitted"
        )
    if type(original_width) is not int or type(original_height) is not int:
        raise OutputValidationError("Original width and height must be positive integers")
    if original_width < 1 or original_height < 1:
        raise OutputValidationError("Original width and height must be positive integers")
    return original_width, original_height


def validate_raw_restoration_output(output: object) -> torch.Tensor:
    """Validate a raw dense floating-point restoration tensor without clipping it."""

    if not isinstance(output, torch.Tensor):
        raise UnsupportedOutputError("Raw model output must be a PyTorch tensor")
    if output.layout != torch.strided:
        raise UnsupportedOutputError(
            f"Unsupported tensor layout {output.layout}; expected a dense strided tensor"
        )
    if output.ndim != 4:
        raise OutputValidationError(
            f"Raw model output must have rank 4 and shape (1, 1, H, W); got {tuple(output.shape)}"
        )
    if output.shape[0] != 1:
        raise OutputValidationError(
            f"Raw model output batch size must be 1; received {output.shape[0]}"
        )
    if output.shape[1] != 1:
        raise OutputValidationError(
            f"Raw model output channel count must be 1; received {output.shape[1]}"
        )
    if output.shape[2] < 1 or output.shape[3] < 1:
        raise OutputValidationError(
            f"Raw model output spatial dimensions must be nonzero; got {tuple(output.shape[2:])}"
        )
    if output.dtype not in SUPPORTED_TENSOR_DTYPES:
        raise UnsupportedOutputError(
            f"Unsupported tensor dtype {output.dtype}; expected float16, bfloat16, "
            "float32, or float64"
        )
    if not bool(torch.isfinite(output).all().item()):
        raise OutputValidationError("Raw model output contains NaN or infinity")
    return output


def postprocess_restoration(
    output: object,
    *,
    original_width: int | None = None,
    original_height: int | None = None,
) -> PostprocessingResult:
    """Validate and clip one raw NAF-SR output without mutating it.

    When original dimensions are supplied, the raw output must be exactly two
    times their width and height. The returned image is a contiguous 2D CPU
    float32 NumPy array in ``[0, 1]``.
    """

    dimensions = _validate_original_dimensions(original_width, original_height)
    tensor = validate_raw_restoration_output(output)
    restored_height = int(tensor.shape[2])
    restored_width = int(tensor.shape[3])
    if dimensions is not None:
        expected_width = dimensions[0] * RESTORATION_SCALE_FACTOR
        expected_height = dimensions[1] * RESTORATION_SCALE_FACTOR
        if restored_width != expected_width or restored_height != expected_height:
            raise OutputValidationError(
                "Raw model output dimensions do not match the required 2x scale: "
                f"received {restored_width}x{restored_height}, expected "
                f"{expected_width}x{expected_height}"
            )

    detached = tensor.detach()
    raw_minimum = float(detached.amin().item())
    raw_maximum = float(detached.amax().item())
    values_below_zero = int((detached < 0).sum().item())
    values_above_one = int((detached > 1).sum().item())
    total_values = detached.numel()
    clipped_tensor = detached.clamp(0.0, 1.0).to(device="cpu", dtype=torch.float32)
    image = np.ascontiguousarray(clipped_tensor[0, 0].numpy())
    clipped_minimum = float(image.min())
    clipped_maximum = float(image.max())
    clipping_occurred = values_below_zero > 0 or values_above_one > 0
    warnings: tuple[str, ...] = ()
    if clipping_occurred:
        clipped_count = values_below_zero + values_above_one
        warnings = (
            f"Clipped {clipped_count} of {total_values} values to [0, 1] "
            f"({values_below_zero} below zero, {values_above_one} above one).",
        )

    clipping = ClippingMetadata(
        raw_minimum=raw_minimum,
        raw_maximum=raw_maximum,
        clipped_minimum=clipped_minimum,
        clipped_maximum=clipped_maximum,
        values_below_zero=values_below_zero,
        fraction_below_zero=values_below_zero / total_values,
        values_above_one=values_above_one,
        fraction_above_one=values_above_one / total_values,
        total_values=total_values,
        clipping_occurred=clipping_occurred,
    )
    return PostprocessingResult(
        image=image,
        restored_width=restored_width,
        restored_height=restored_height,
        clipping=clipping,
        original_width=original_width,
        original_height=original_height,
        scale_factor=RESTORATION_SCALE_FACTOR,
        source_tensor_dtype=str(tensor.dtype),
        source_tensor_device=str(tensor.device),
        warnings=warnings,
    )


def _validate_result_image(result: PostprocessingResult) -> np.ndarray:
    if not isinstance(result, PostprocessingResult):
        raise UnsupportedOutputError("Encoding requires a PostprocessingResult")
    image = result.image
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 2
        or image.dtype != np.float32
        or image.shape != (result.restored_height, result.restored_width)
        or not image.flags.c_contiguous
    ):
        raise OutputValidationError(
            "Postprocessed image must remain a contiguous 2D float32 array with recorded dimensions"
        )
    if image.size == 0 or not np.isfinite(image).all():
        raise OutputValidationError("Postprocessed image must be nonempty and finite")
    if float(image.min()) < 0.0 or float(image.max()) > 1.0:
        raise OutputValidationError("Postprocessed image values must remain within [0, 1]")
    return image


def quantize_restoration(result: PostprocessingResult, *, bit_depth: int) -> np.ndarray:
    """Quantize with deterministic round-half-up mapping over the full integer range."""

    if type(bit_depth) is not int or bit_depth not in SUPPORTED_PNG_BIT_DEPTHS:
        raise UnsupportedOutputError("PNG bit depth must be exactly 8 or 16")
    image = _validate_result_image(result)
    maximum = 255 if bit_depth == 8 else 65535
    dtype = np.uint8 if bit_depth == 8 else np.uint16
    quantized = np.floor(image.astype(np.float64) * maximum + 0.5).astype(dtype)
    return np.ascontiguousarray(quantized)


def encode_restoration(
    result: PostprocessingResult,
    *,
    encoding: str = "png",
    bit_depth: int = 16,
) -> bytes:
    """Return a single-channel lossless PNG without writing to disk."""

    if not isinstance(encoding, str) or encoding.lower() != "png":
        raise UnsupportedOutputError("Only lossless PNG encoding is supported")
    quantized = quantize_restoration(result, bit_depth=bit_depth)
    expected_mode = "L" if bit_depth == 8 else "I;16"
    encoded = io.BytesIO()
    try:
        image = Image.fromarray(quantized)
        if image.mode != expected_mode:
            raise OutputEncodingError(
                f"Pillow produced mode {image.mode!r}; expected {expected_mode!r}"
            )
        image.save(encoded, format="PNG")
    except OutputEncodingError:
        raise
    except (OSError, ValueError) as error:
        raise OutputEncodingError("Could not encode the restored image as PNG") from error
    return encoded.getvalue()


__all__ = [
    "ClippingMetadata",
    "OutputEncodingError",
    "OutputValidationError",
    "POSTPROCESSING_VERSION",
    "PostprocessingError",
    "PostprocessingResult",
    "RESTORATION_SCALE_FACTOR",
    "UnsupportedOutputError",
    "encode_restoration",
    "postprocess_restoration",
    "quantize_restoration",
    "validate_raw_restoration_output",
]
