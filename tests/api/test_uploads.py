from __future__ import annotations

import asyncio
import warnings
from dataclasses import fields
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from semirestore.api import uploads as upload_module
from semirestore.api.errors import (
    EmptyUploadError,
    ImageDimensionsExceededError,
    InvalidImageError,
    UnsupportedMediaTypeError,
    UploadTooLargeError,
)
from semirestore.api.uploads import ValidatedUpload, validate_upload
from semirestore.platform import RuntimeSettings


class MemoryUpload:
    def __init__(
        self,
        data: bytes,
        content_type: str | None,
        filename: str | None = "image.sem",
    ) -> None:
        self.data = data
        self.content_type = content_type
        self.filename = filename
        self.read_sizes: list[int] = []
        self.closed = False

    async def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return self.data if size < 0 else self.data[:size]

    async def close(self) -> None:
        self.closed = True


def make_image(
    image_format: str,
    *,
    size: tuple[int, int] = (4, 3),
    frames: int = 1,
) -> bytes:
    output = BytesIO()
    images = [Image.new("L", size, color=index) for index in range(frames)]
    images[0].save(
        output,
        format=image_format,
        save_all=frames > 1,
        append_images=images[1:],
    )
    return output.getvalue()


def validate(source: MemoryUpload | None, settings: RuntimeSettings | None = None) -> Any:
    return asyncio.run(validate_upload(source, settings or RuntimeSettings()))


@pytest.mark.parametrize(
    ("image_format", "media_type"),
    [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("TIFF", "image/tiff")],
)
def test_valid_single_frame_formats(image_format: str, media_type: str) -> None:
    source = MemoryUpload(make_image(image_format), media_type)

    result = validate(source)

    assert result.detected_format == image_format
    assert result.media_type == media_type
    assert (result.width, result.height) == (4, 3)
    assert result.encoded_bytes == source.data
    assert source.closed is True


def test_missing_and_empty_uploads_are_rejected() -> None:
    with pytest.raises(EmptyUploadError):
        validate(None)

    source = MemoryUpload(b"", "image/png")
    with pytest.raises(EmptyUploadError):
        validate(source)
    assert source.closed is True


def test_encoded_input_exactly_at_limit_is_accepted() -> None:
    encoded = make_image("PNG")
    settings = RuntimeSettings(max_encoded_upload_bytes=len(encoded))
    source = MemoryUpload(encoded, "image/png")

    result = validate(source, settings)

    assert result.encoded_bytes == encoded
    assert source.read_sizes == [len(encoded) + 1]


def test_encoded_input_one_byte_above_limit_is_rejected_after_bounded_read() -> None:
    encoded = make_image("PNG")
    maximum = len(encoded) - 1
    source = MemoryUpload(encoded, "image/png")

    with pytest.raises(UploadTooLargeError) as caught:
        validate(source, RuntimeSettings(max_encoded_upload_bytes=maximum))

    assert caught.value.details == {"maximum_bytes": maximum}
    assert source.read_sizes == [maximum + 1]
    assert source.closed is True


@pytest.mark.parametrize(
    "encoded",
    [b"not an image", make_image("PNG")[:-12]],
    ids=["malformed", "truncated"],
)
def test_malformed_or_truncated_content_is_rejected_without_decoder_details(
    encoded: bytes,
) -> None:
    source = MemoryUpload(encoded, "image/png", filename="C:/secret/checkpoint.pt")

    with pytest.raises(InvalidImageError) as caught:
        validate(source)

    assert caught.value.details is None
    assert "checkpoint" not in str(caught.value)
    assert source.closed is True


def test_declared_png_containing_jpeg_is_rejected() -> None:
    source = MemoryUpload(make_image("JPEG"), "image/png")

    with pytest.raises(UnsupportedMediaTypeError):
        validate(source)


@pytest.mark.parametrize("media_type", ["image/gif", "application/octet-stream", "image/svg+xml"])
def test_unsupported_declared_media_type_is_rejected_without_reading(
    media_type: str,
) -> None:
    source = MemoryUpload(make_image("PNG"), media_type)

    with pytest.raises(UnsupportedMediaTypeError) as caught:
        validate(source)

    assert caught.value.details == {
        "supported_media_types": ["image/png", "image/jpeg", "image/tiff"]
    }
    assert source.read_sizes == []
    assert source.closed is True


def test_filename_extension_does_not_override_detected_format() -> None:
    source = MemoryUpload(make_image("JPEG"), "image/jpeg", filename="misleading.png")

    result = validate(source)

    assert result.detected_format == "JPEG"
    assert result.media_type == "image/jpeg"


@pytest.mark.parametrize(
    ("size", "setting_overrides"),
    [
        ((5, 2), {"max_decoded_image_width": 4}),
        ((2, 5), {"max_decoded_image_height": 4}),
        ((4, 4), {"max_decoded_pixel_count": 15}),
    ],
    ids=["width", "height", "pixels"],
)
def test_decoded_dimension_limits_are_enforced(
    size: tuple[int, int],
    setting_overrides: dict[str, int],
) -> None:
    settings = RuntimeSettings(**setting_overrides)
    source = MemoryUpload(make_image("PNG", size=size), "image/png")

    with pytest.raises(ImageDimensionsExceededError) as caught:
        validate(source, settings)

    assert caught.value.details == {
        "maximum_width": settings.max_decoded_image_width,
        "maximum_height": settings.max_decoded_image_height,
        "maximum_pixels": settings.max_decoded_pixel_count,
    }


def test_decompression_bomb_warning_is_local_and_mapped_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_open = upload_module.Image.open

    def warning_open(*args: Any, **kwargs: Any) -> Any:
        warnings.warn("private decoder detail", Image.DecompressionBombWarning, stacklevel=2)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(upload_module.Image, "open", warning_open)
    source = MemoryUpload(make_image("PNG"), "image/png")

    with pytest.raises(ImageDimensionsExceededError) as caught:
        validate(source)

    assert "private decoder detail" not in str(caught.value)


def test_decompression_bomb_error_is_mapped_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    def error_open(*_args: Any, **_kwargs: Any) -> Any:
        raise Image.DecompressionBombError("private decoder detail")

    monkeypatch.setattr(upload_module.Image, "open", error_open)
    source = MemoryUpload(make_image("PNG"), "image/png")

    with pytest.raises(ImageDimensionsExceededError) as caught:
        validate(source)

    assert "private decoder detail" not in str(caught.value)


def test_multi_frame_tiff_is_rejected() -> None:
    source = MemoryUpload(make_image("TIFF", frames=2), "image/tiff")

    with pytest.raises(InvalidImageError):
        validate(source)


def test_path_like_control_character_filename_is_never_retained_or_persisted(
    tmp_path: Path,
) -> None:
    filename = f"{tmp_path}/../secret\x00\nimage.png"
    source = MemoryUpload(make_image("PNG"), "image/png", filename=filename)

    result = validate(source)

    assert isinstance(result, ValidatedUpload)
    assert "filename" not in {field.name for field in fields(result)}
    assert list(tmp_path.iterdir()) == []


def test_validated_upload_contains_no_decoder_or_open_file_handle() -> None:
    result = validate(MemoryUpload(make_image("PNG"), "image/png"))

    assert isinstance(result, ValidatedUpload)
    assert not hasattr(result, "image")
    assert not hasattr(result, "file")
    assert not hasattr(result, "__dict__")
