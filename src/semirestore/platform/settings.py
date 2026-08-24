"""Typed runtime configuration for the SemiRestore service platform."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MEDIA_TYPES = ("image/png", "image/jpeg", "image/tiff")


class RuntimeSettings(BaseSettings):
    """Platform settings loaded from ``SEMIRESTORE_`` environment variables.

    Paths are passed to the future model-service boundary without opening,
    resolving, or validating model-owned files.
    """

    model_config = SettingsConfigDict(
        env_prefix="SEMIRESTORE_",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
    )

    environment: str = Field(default="development", min_length=1, max_length=64)
    host: str = Field(default="127.0.0.1", min_length=1, max_length=253)
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    json_logging: bool = True

    max_encoded_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    max_decoded_image_width: int = Field(default=16_384, ge=1)
    max_decoded_image_height: int = Field(default=16_384, ge=1)
    max_decoded_pixel_count: int = Field(default=100_000_000, ge=1)
    allowed_media_types: tuple[str, ...] = DEFAULT_MEDIA_TYPES

    inference_concurrency_limit: int = Field(default=1, ge=1)
    concurrency_acquisition_timeout_seconds: float = Field(default=1.0, gt=0)
    inference_timeout_seconds: float = Field(default=120.0, gt=0)

    model_config_path: Path | None = None
    checkpoint_path: Path | None = None
    device_preference: Literal["auto", "cpu", "cuda"] = "auto"
    enable_fake_model_service: bool = False

    @field_validator("environment", "host")
    @classmethod
    def strip_nonempty_text(cls, value: str) -> str:
        """Reject values that contain only whitespace."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        """Accept conventional case-insensitive log-level values."""
        return value.upper() if isinstance(value, str) else value

    @field_validator("allowed_media_types")
    @classmethod
    def validate_media_types(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Normalize media types and reject empty or duplicate entries."""
        normalized = tuple(value.strip().lower() for value in values)
        if not normalized or any(not value for value in normalized):
            raise ValueError("must contain at least one non-blank media type")
        if len(set(normalized)) != len(normalized):
            raise ValueError("must not contain duplicate media types")
        return normalized

    @field_validator(
        "concurrency_acquisition_timeout_seconds",
        "inference_timeout_seconds",
    )
    @classmethod
    def validate_finite_timeouts(cls, value: float) -> float:
        """Reject infinite timeout values in addition to non-positive values."""
        if not math.isfinite(value):
            raise ValueError("timeout must be finite")
        return value
