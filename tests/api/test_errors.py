from __future__ import annotations

from typing import Any

import pytest
from fastapi import Query, Request
from fastapi.testclient import TestClient

from semirestore.api import create_app
from semirestore.api.errors import (
    APIError,
    EmptyUploadError,
    ImageDimensionsExceededError,
    InferenceBusyError,
    InferenceTimeoutError,
    InvalidImageError,
    InvalidRequestError,
    RestorationFailedError,
    UnsupportedMediaTypeError,
    UploadTooLargeError,
)
from semirestore.platform import (
    ModelServiceInferenceError,
    ModelServiceInitializationError,
    ModelServiceUnavailableError,
)


@pytest.mark.parametrize(
    ("error_type", "expected_status", "expected_code"),
    [
        (InvalidRequestError, 400, "invalid_request"),
        (EmptyUploadError, 400, "empty_upload"),
        (UnsupportedMediaTypeError, 415, "unsupported_media_type"),
        (UploadTooLargeError, 413, "upload_too_large"),
        (InvalidImageError, 422, "invalid_image"),
        (ImageDimensionsExceededError, 413, "image_dimensions_exceeded"),
        (InferenceBusyError, 503, "inference_busy"),
        (InferenceTimeoutError, 504, "inference_timeout"),
        (RestorationFailedError, 500, "restoration_failed"),
    ],
)
def test_platform_errors_map_to_stable_status_and_safe_details(
    error_type: type[APIError],
    expected_status: int,
    expected_code: str,
) -> None:
    app = create_app()

    @app.get("/_test/error")
    def raise_error() -> None:
        raise error_type(details={"limit": 10})

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/_test/error")

    assert response.status_code == expected_status
    assert response.json()["error"]["code"] == expected_code
    assert response.json()["error"]["details"] == {"limit": 10}
    assert response.json()["error"]["request_id"] == response.headers["x-request-id"]


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code"),
    [
        (
            ModelServiceInitializationError("C:/secret/checkpoint.pt could not load"),
            503,
            "model_unavailable",
        ),
        (
            ModelServiceUnavailableError("token=secret"),
            503,
            "model_unavailable",
        ),
        (
            ModelServiceInferenceError("tensor dump and C:/private/model.pt"),
            500,
            "restoration_failed",
        ),
    ],
)
def test_model_service_errors_suppress_internal_messages(
    error: Exception,
    expected_status: int,
    expected_code: str,
) -> None:
    app = create_app()

    @app.get("/_test/model-error")
    def raise_error() -> None:
        raise error

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/_test/model-error")

    assert response.status_code == expected_status
    assert response.json()["error"]["code"] == expected_code
    assert response.json()["error"]["details"] is None
    assert response.json()["error"]["request_id"] == response.headers["x-request-id"]
    assert "checkpoint.pt" not in response.text
    assert "token" not in response.text
    assert "tensor" not in response.text
    assert "C:/" not in response.text


def test_request_validation_error_omits_input_and_validation_context() -> None:
    app = create_app()

    @app.get("/_test/validated")
    def validated(value: int = Query(ge=1)) -> dict[str, int]:
        return {"value": value}

    unsafe_input = "C:/secret/checkpoint.pt?token=secret"
    with TestClient(app) as client:
        response = client.get("/_test/validated", params={"value": unsafe_input})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    issues = response.json()["error"]["details"]["issues"]
    assert issues == [{"location": ["query", "value"], "type": "int_parsing"}]
    assert unsafe_input not in response.text
    assert "checkpoint.pt" not in response.text
    assert "token" not in response.text


def test_unexpected_error_becomes_generic_internal_error() -> None:
    app = create_app()

    @app.get("/_test/internal-error")
    def raise_error() -> None:
        raise RuntimeError("Traceback: C:/secret/checkpoint.pt token=secret tensor=[1, 2]")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/_test/internal-error")

    assert response.status_code == 500
    assert response.json()["error"] == {
        "code": "internal_error",
        "message": "An internal server error occurred.",
        "details": None,
        "request_id": response.headers["x-request-id"],
    }
    for unsafe_text in ("Traceback", "C:/", "checkpoint.pt", "token", "tensor"):
        assert unsafe_text not in response.text


def test_framework_http_error_uses_error_envelope() -> None:
    app = create_app()

    with TestClient(app) as client:
        response = client.get("/does-not-exist")

    assert response.status_code == 404
    assert response.json()["error"] == {
        "code": "invalid_request",
        "message": "The requested resource was not found.",
        "details": None,
        "request_id": response.headers["x-request-id"],
    }


def test_handler_includes_validated_client_request_id() -> None:
    app = create_app()

    @app.get("/_test/request-id")
    def raise_error(request: Request) -> None:
        assert request.state.request_id == "existing-request-id"
        raise InvalidRequestError()

    with TestClient(app) as client:
        response = client.get(
            "/_test/request-id",
            headers={"x-request-id": "existing-request-id"},
        )

    assert response.json()["error"]["request_id"] == "existing-request-id"
    assert response.headers["x-request-id"] == "existing-request-id"


def test_existing_health_endpoint_contract_remains_compatible(
    fake_model_service: Any,
) -> None:
    app = create_app(model_service_factory=lambda _: fake_model_service)

    with TestClient(app) as client:
        response = client.get("/health/model")

    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert response.json()["state"] == "ready"
