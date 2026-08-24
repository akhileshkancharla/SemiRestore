"""Stable response schemas shared by SemiRestore API endpoints."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from semirestore.platform import ModelServiceState


class ResponseModel(BaseModel):
    """Strict base for public API response contracts."""

    model_config = ConfigDict(extra="forbid")


class LiveResponse(ResponseModel):
    """API process liveness response."""

    status: Literal["alive"] = "alive"


class ReadyResponse(ResponseModel):
    """Restoration-work readiness response."""

    ready: bool
    state: ModelServiceState
    unavailable_reason: str | None = None


class ModelHealthResponse(ReadyResponse):
    """Safe model-service health and provenance response."""

    device: str | None = None
    model_version: str | None = None
    checkpoint_checksum: str | None = None


class VersionResponse(ResponseModel):
    """Stable application version response."""

    application: Literal["semirestore"] = "semirestore"
    version: str = Field(min_length=1)


class ErrorCode(StrEnum):
    """Stable machine-readable API error categories."""

    INVALID_REQUEST = "invalid_request"
    EMPTY_UPLOAD = "empty_upload"
    UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"
    UPLOAD_TOO_LARGE = "upload_too_large"
    INVALID_IMAGE = "invalid_image"
    IMAGE_DIMENSIONS_EXCEEDED = "image_dimensions_exceeded"
    MODEL_UNAVAILABLE = "model_unavailable"
    INFERENCE_BUSY = "inference_busy"
    INFERENCE_TIMEOUT = "inference_timeout"
    RESTORATION_FAILED = "restoration_failed"
    INTERNAL_ERROR = "internal_error"


class ErrorBody(ResponseModel):
    """Error information nested inside every API error response."""

    code: ErrorCode
    message: str = Field(min_length=1, max_length=256)
    details: dict[str, JsonValue] | None = None
    request_id: str | None = Field(default=None, min_length=1, max_length=64)


class ErrorResponse(ResponseModel):
    """Reusable envelope for all API errors."""

    error: ErrorBody


class RestoredImageResponse(ResponseModel):
    """Base64-encoded restored image payload."""

    encoding: Literal["base64"] = "base64"
    media_type: Literal["image/png", "image/jpeg", "image/tiff"]
    content: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class RestoreInputResponse(ResponseModel):
    """Transport metadata for the validated input image."""

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    media_type: Literal["image/png", "image/jpeg", "image/tiff"]


class InferenceResponse(ResponseModel):
    """Available inference execution metadata."""

    latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    device: str | None = None


class ModelIdentityResponse(ResponseModel):
    """Available model identity metadata."""

    version: str | None = None
    checkpoint_checksum: str | None = None


class RestoreResponse(ResponseModel):
    """Complete restoration response with encoded image and metadata."""

    image: RestoredImageResponse
    input: RestoreInputResponse
    inference: InferenceResponse
    model: ModelIdentityResponse
    diagnostics: dict[str, JsonValue] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()
