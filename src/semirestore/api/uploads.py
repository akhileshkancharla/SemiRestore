"""Bounded, in-memory transport validation for uploaded SEM images."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from io import BytesIO
from typing import Literal, Protocol, cast

from PIL import Image, UnidentifiedImageError

from semirestore.api.errors import (
    APIError,
    EmptyUploadError,
    ImageDimensionsExceededError,
    InvalidImageError,
    UnsupportedMediaTypeError,
    UploadTooLargeError,
)
from semirestore.platform import RuntimeSettings

DetectedImageFormat = Literal["PNG", "JPEG", "TIFF"]

_FORMAT_MEDIA_TYPES: dict[str, str] = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "TIFF": "image/tiff",
}


class UploadSource(Protocol):
    """Minimal asynchronous upload surface required by the validator."""

    filename: str | None
    content_type: str | None

    async def read(self, size: int = -1) -> bytes:
        """Read at most ``size`` encoded bytes."""

    async def close(self) -> None:
        """Release framework-managed upload resources."""


@dataclass(frozen=True, slots=True)
class ValidatedUpload:
    """Immutable transport-safe image data for a later restoration endpoint."""

    encoded_bytes: bytes
    media_type: str
    detected_format: DetectedImageFormat
    width: int
    height: int


def _supported_media_types(settings: RuntimeSettings) -> tuple[str, ...]:
    supported = set(_FORMAT_MEDIA_TYPES.values())
    return tuple(
        media_type for media_type in settings.allowed_media_types if media_type in supported
    )


def _media_type_details(settings: RuntimeSettings) -> dict[str, list[str]]:
    return {"supported_media_types": list(_supported_media_types(settings))}


def _dimension_details(settings: RuntimeSettings) -> dict[str, int]:
    return {
        "maximum_width": settings.max_decoded_image_width,
        "maximum_height": settings.max_decoded_image_height,
        "maximum_pixels": settings.max_decoded_pixel_count,
    }


def _validate_dimensions(width: int, height: int, settings: RuntimeSettings) -> None:
    if (
        width > settings.max_decoded_image_width
        or height > settings.max_decoded_image_height
        or width * height > settings.max_decoded_pixel_count
    ):
        raise ImageDimensionsExceededError(details=_dimension_details(settings))


def _inspect_image(
    encoded_bytes: bytes,
    declared_media_type: str,
    settings: RuntimeSettings,
) -> tuple[DetectedImageFormat, str, int, int]:
    with Image.open(BytesIO(encoded_bytes)) as image:
        detected_format = image.format
        canonical_media_type = _FORMAT_MEDIA_TYPES.get(detected_format or "")
        if canonical_media_type is None:
            raise UnsupportedMediaTypeError(details=_media_type_details(settings))
        if canonical_media_type != declared_media_type:
            raise UnsupportedMediaTypeError(details=_media_type_details(settings))

        width, height = image.size
        _validate_dimensions(width, height, settings)
        if getattr(image, "n_frames", 1) != 1:
            raise InvalidImageError()
        image.verify()

    return cast(DetectedImageFormat, detected_format), canonical_media_type, width, height


def _load_verified_image(encoded_bytes: bytes) -> None:
    """Force decoding after metadata and structure validation has succeeded."""
    with Image.open(BytesIO(encoded_bytes)) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise InvalidImageError()
        image.load()


def _decode_and_validate(
    encoded_bytes: bytes,
    declared_media_type: str,
    settings: RuntimeSettings,
) -> tuple[DetectedImageFormat, str, int, int]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            metadata = _inspect_image(encoded_bytes, declared_media_type, settings)
            _load_verified_image(encoded_bytes)
            return metadata
    except APIError:
        raise
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
        raise ImageDimensionsExceededError(details=_dimension_details(settings)) from error
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        raise InvalidImageError() from error


async def validate_upload(
    upload: UploadSource | None,
    settings: RuntimeSettings,
) -> ValidatedUpload:
    """Validate and close one upload without persisting or preprocessing it."""
    if upload is None:
        raise EmptyUploadError()

    try:
        declared_media_type = (upload.content_type or "").partition(";")[0].strip().lower()
        if declared_media_type not in _supported_media_types(settings):
            raise UnsupportedMediaTypeError(details=_media_type_details(settings))

        encoded_bytes = await upload.read(settings.max_encoded_upload_bytes + 1)
        if not encoded_bytes:
            raise EmptyUploadError()
        if len(encoded_bytes) > settings.max_encoded_upload_bytes:
            raise UploadTooLargeError(
                details={"maximum_bytes": settings.max_encoded_upload_bytes}
            )

        detected_format, media_type, width, height = _decode_and_validate(
            encoded_bytes,
            declared_media_type,
            settings,
        )
        return ValidatedUpload(
            encoded_bytes=encoded_bytes,
            media_type=media_type,
            detected_format=detected_format,
            width=width,
            height=height,
        )
    finally:
        try:
            await upload.close()
        except Exception:
            pass
