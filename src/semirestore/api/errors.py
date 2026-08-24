"""Stable API exceptions and safe FastAPI exception handlers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import JsonValue
from starlette.exceptions import HTTPException as StarletteHTTPException

from semirestore.api.schemas import ErrorBody, ErrorCode, ErrorResponse
from semirestore.platform import (
    ModelServiceInferenceError,
    ModelServiceInitializationError,
    ModelServiceUnavailableError,
)


class APIError(Exception):
    """Base for deliberately safe errors raised by platform code."""

    code: ClassVar[ErrorCode]
    status_code: ClassVar[int]
    public_message: ClassVar[str]

    def __init__(self, *, details: Mapping[str, JsonValue] | None = None) -> None:
        super().__init__(self.public_message)
        self.details = dict(details) if details is not None else None


class InvalidRequestError(APIError):
    code = ErrorCode.INVALID_REQUEST
    status_code = status.HTTP_400_BAD_REQUEST
    public_message = "The request is invalid."


class EmptyUploadError(APIError):
    code = ErrorCode.EMPTY_UPLOAD
    status_code = status.HTTP_400_BAD_REQUEST
    public_message = "An image file is required."


class UnsupportedMediaTypeError(APIError):
    code = ErrorCode.UNSUPPORTED_MEDIA_TYPE
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    public_message = "The uploaded media type is not supported."


class UploadTooLargeError(APIError):
    code = ErrorCode.UPLOAD_TOO_LARGE
    status_code = status.HTTP_413_CONTENT_TOO_LARGE
    public_message = "The uploaded file exceeds the configured size limit."


class InvalidImageError(APIError):
    code = ErrorCode.INVALID_IMAGE
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    public_message = "The uploaded file is not a valid supported image."


class ImageDimensionsExceededError(APIError):
    code = ErrorCode.IMAGE_DIMENSIONS_EXCEEDED
    status_code = status.HTTP_413_CONTENT_TOO_LARGE
    public_message = "The decoded image exceeds the configured dimension limits."


class InferenceBusyError(APIError):
    code = ErrorCode.INFERENCE_BUSY
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    public_message = "The restoration service is busy. Try again later."


class InferenceTimeoutError(APIError):
    code = ErrorCode.INFERENCE_TIMEOUT
    status_code = status.HTTP_504_GATEWAY_TIMEOUT
    public_message = "The restoration request timed out."


class RestorationFailedError(APIError):
    code = ErrorCode.RESTORATION_FAILED
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    public_message = "The restoration request failed."


def _request_id(request: Request) -> str | None:
    """Read a future request ID without generating or trusting client input."""
    request_id = getattr(request.state, "request_id", None)
    if isinstance(request_id, str) and 1 <= len(request_id) <= 128:
        return request_id
    return None


def _error_response(
    request: Request,
    *,
    status_code: int,
    code: ErrorCode,
    message: str,
    details: dict[str, JsonValue] | None = None,
) -> JSONResponse:
    envelope = ErrorResponse(
        error=ErrorBody(
            code=code,
            message=message,
            details=details,
            request_id=_request_id(request),
        )
    )
    return JSONResponse(status_code=status_code, content=envelope.model_dump(mode="json"))


async def api_error_handler(request: Request, error: APIError) -> JSONResponse:
    """Map a safe platform exception without inspecting its string form."""
    return _error_response(
        request,
        status_code=error.status_code,
        code=error.code,
        message=error.public_message,
        details=error.details,
    )


async def model_unavailable_handler(
    request: Request,
    _error: ModelServiceInitializationError | ModelServiceUnavailableError,
) -> JSONResponse:
    """Map model availability failures to a stable, non-leaking response."""
    return _error_response(
        request,
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        code=ErrorCode.MODEL_UNAVAILABLE,
        message="The model service is unavailable.",
    )


async def model_inference_handler(
    request: Request,
    _error: ModelServiceInferenceError,
) -> JSONResponse:
    """Map model inference failures without exposing implementation details."""
    return _error_response(
        request,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code=ErrorCode.RESTORATION_FAILED,
        message=RestorationFailedError.public_message,
    )


async def request_validation_handler(
    request: Request,
    error: RequestValidationError,
) -> JSONResponse:
    """Return validation locations and types while omitting inputs and context."""
    issues = [
        {
            "location": [str(part) for part in issue.get("loc", ())],
            "type": str(issue.get("type", "validation_error")),
        }
        for issue in error.errors()
    ]
    return _error_response(
        request,
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        code=ErrorCode.INVALID_REQUEST,
        message="Request validation failed.",
        details={"issues": issues},
    )


async def http_error_handler(request: Request, error: StarletteHTTPException) -> JSONResponse:
    """Replace framework HTTP error details with stable public messages."""
    if error.status_code == status.HTTP_404_NOT_FOUND:
        message = "The requested resource was not found."
    elif error.status_code == status.HTTP_405_METHOD_NOT_ALLOWED:
        message = "The request method is not allowed."
    else:
        message = "The request is invalid."
    if error.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
        return _error_response(
            request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code=ErrorCode.INTERNAL_ERROR,
            message="An internal server error occurred.",
        )
    return _error_response(
        request,
        status_code=error.status_code,
        code=ErrorCode.INVALID_REQUEST,
        message=message,
    )


async def internal_error_handler(request: Request, _error: Exception) -> JSONResponse:
    """Suppress all details from unexpected failures."""
    return _error_response(
        request,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code=ErrorCode.INTERNAL_ERROR,
        message="An internal server error occurred.",
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Install the stable exception mapping on an application instance."""
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(ModelServiceInitializationError, model_unavailable_handler)
    app.add_exception_handler(ModelServiceUnavailableError, model_unavailable_handler)
    app.add_exception_handler(ModelServiceInferenceError, model_inference_handler)
    app.add_exception_handler(RequestValidationError, request_validation_handler)
    app.add_exception_handler(StarletteHTTPException, http_error_handler)
    app.add_exception_handler(Exception, internal_error_handler)
