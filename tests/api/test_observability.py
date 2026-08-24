from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from io import BytesIO, StringIO
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from PIL import Image

from semirestore.api import create_app
from semirestore.api.errors import InferenceBusyError, InferenceTimeoutError
from semirestore.api.metrics import PlatformMetrics
from semirestore.api.observability import (
    LOGGER_NAME,
    RequestObservabilityMiddleware,
    configure_application_logging,
    observe_inference,
    select_request_id,
)
from semirestore.api.schemas import ErrorBody
from semirestore.platform import (
    ModelServiceInferenceError,
    ModelServiceUnavailableError,
    RuntimeSettings,
)

GENERATED_ID = re.compile(r"[0-9a-f]{32}\Z")


def png_bytes() -> bytes:
    output = BytesIO()
    Image.new("L", (3, 2), color=32).save(output, format="PNG")
    return output.getvalue()


def upload(filename: str = "input.png") -> dict[str, tuple[str, bytes, str]]:
    return {"image": (filename, png_bytes(), "image/png")}


def log_stream(app: Any) -> StringIO:
    stream = StringIO()
    configure_application_logging(app.state.runtime.settings, stream=stream)
    return stream


def events(stream: StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line]


def test_generated_request_id_is_returned_on_health_response() -> None:
    app = create_app()

    with TestClient(app) as client:
        response = client.get("/health/live")

    request_id = response.headers["x-request-id"]
    assert GENERATED_ID.fullmatch(request_id) is not None


def test_valid_client_request_id_is_preserved_on_success_and_error() -> None:
    app = create_app()
    request_id = "edge-1.Trace_2026:abc"

    with TestClient(app) as client:
        success = client.get("/version", headers={"x-request-id": request_id})
        error = client.get("/missing", headers={"x-request-id": request_id})

    assert success.headers["x-request-id"] == request_id
    assert error.headers["x-request-id"] == request_id
    assert error.json()["error"]["request_id"] == request_id


def test_maximum_length_client_request_id_is_preserved() -> None:
    app = create_app()
    request_id = "a" * 64

    with TestClient(app) as client:
        response = client.get("/health/live", headers={"x-request-id": request_id})

    assert response.headers["x-request-id"] == request_id


@pytest.mark.parametrize(
    "unsafe_id",
    ["contains space", "-starts-with-symbol", "x" * 65],
)
def test_unsafe_client_request_id_is_replaced(unsafe_id: str) -> None:
    app = create_app()

    with TestClient(app) as client:
        response = client.get("/health/live", headers={"x-request-id": unsafe_id})

    assert GENERATED_ID.fullmatch(response.headers["x-request-id"]) is not None
    assert response.headers["x-request-id"] != unsafe_id


def test_control_characters_and_ambiguous_headers_are_replaced() -> None:
    newline_id = select_request_id([(b"x-request-id", b"unsafe\nvalue")])
    non_ascii_id = select_request_id([(b"x-request-id", b"non-ascii-\xff")])
    duplicate_id = select_request_id(
        [(b"x-request-id", b"first"), (b"x-request-id", b"second")]
    )

    assert GENERATED_ID.fullmatch(newline_id) is not None
    assert GENERATED_ID.fullmatch(non_ascii_id) is not None
    assert GENERATED_ID.fullmatch(duplicate_id) is not None


def test_response_header_covers_restore_validation_404_405_and_internal_errors(
    fake_model_service: Any,
) -> None:
    app = create_app(model_service_factory=lambda _: fake_model_service)

    @app.get("/_test/crash")
    def crash() -> None:
        raise RuntimeError("C:/secret/checkpoint.pt token=private")

    with TestClient(app, raise_server_exceptions=False) as client:
        responses = [
            client.post("/api/v1/restore", files=upload()),
            client.post("/api/v1/restore", files=None),
            client.get("/not-found"),
            client.post("/health/live"),
            client.get("/_test/crash"),
        ]

    assert [response.status_code for response in responses] == [200, 400, 404, 405, 500]
    for response in responses:
        assert GENERATED_ID.fullmatch(response.headers["x-request-id"]) is not None
        if response.status_code >= 400:
            assert response.json()["error"]["request_id"] == response.headers["x-request-id"]


def test_json_completion_event_has_stable_fields_and_uses_route_template() -> None:
    settings = RuntimeSettings(environment="staging", json_logging=True)
    app = create_app(settings=settings)
    stream = log_stream(app)

    with TestClient(app) as client:
        response = client.get(
            "/health/live?token=never-log-this",
            headers={"x-request-id": "trace-123"},
        )

    records = events(stream)
    assert len(records) == 1
    record = records[0]
    assert set(record) == {
        "timestamp",
        "level",
        "event",
        "environment",
        "request_id",
        "method",
        "route",
        "status",
        "status_class",
        "duration_ms",
        "inference_duration_ms",
        "outcome",
        "error_code",
        "model_readiness",
    }
    assert record["event"] == "http_request_completed"
    assert record["environment"] == "staging"
    assert record["request_id"] == response.headers["x-request-id"] == "trace-123"
    assert record["method"] == "GET"
    assert record["route"] == "/health/live"
    assert record["status"] == 200
    assert record["status_class"] == "2xx"
    assert record["level"] == "INFO"
    assert isinstance(record["duration_ms"], float)
    assert math.isfinite(record["duration_ms"])
    assert record["duration_ms"] >= 0
    assert "never-log-this" not in stream.getvalue()


def test_completion_logs_use_safe_unmatched_route_error_code_and_severity() -> None:
    app = create_app()

    @app.get("/_test/crash")
    def crash() -> None:
        raise RuntimeError("Traceback C:/secret/checkpoint.pt token=secret tensor=[1]")

    stream = log_stream(app)
    with TestClient(app, raise_server_exceptions=False) as client:
        missing = client.get("/private/checkpoint.pt?token=secret")
        crashed = client.get("/_test/crash")

    records = [record for record in events(stream) if record["event"] == "http_request_completed"]
    assert len(records) == 2
    assert records[0]["route"] == "<unmatched>"
    assert records[0]["level"] == "WARNING"
    assert records[0]["error_code"] == "invalid_request"
    assert records[1]["route"] == "/_test/crash"
    assert records[1]["level"] == "ERROR"
    assert records[1]["error_code"] == "internal_error"
    assert missing.status_code == 404
    assert crashed.status_code == 500
    for unsafe in ("private", "checkpoint.pt", "token", "secret", "tensor", "Traceback"):
        assert unsafe not in stream.getvalue()


def test_restore_logs_one_safe_request_event_and_one_inference_event(
    fake_model_service: Any,
) -> None:
    app = create_app(model_service_factory=lambda _: fake_model_service)
    stream = log_stream(app)
    unsafe_filename = "C:/secret/checkpoint.pt-token=private.png"

    with TestClient(app) as client:
        response = client.post("/api/v1/restore", files=upload(unsafe_filename))

    records = events(stream)
    completions = [record for record in records if record["event"] == "http_request_completed"]
    inferences = [record for record in records if record["event"] == "inference_completed"]
    assert response.status_code == 200
    assert len(completions) == 1
    assert len(inferences) == 1
    assert completions[0]["request_id"] == inferences[0]["request_id"]
    assert inferences[0]["route"] == "/api/v1/restore"
    assert inferences[0]["outcome"] == "success"
    assert inferences[0]["error_code"] is None
    assert inferences[0]["model_readiness"] == "ready"
    assert isinstance(inferences[0]["inference_duration_ms"], float)
    assert inferences[0]["inference_duration_ms"] >= 0
    unsafe_values = (
        unsafe_filename,
        "checkpoint.pt",
        "private",
        response.json()["image"]["content"],
    )
    for unsafe in unsafe_values:
        assert unsafe not in stream.getvalue()


def make_observation_request(settings: RuntimeSettings) -> Request:
    app = SimpleNamespace(
        state=SimpleNamespace(
            runtime=SimpleNamespace(
                settings=settings,
                metrics=PlatformMetrics(inference_capacity=1),
            )
        ),
    )
    route = SimpleNamespace(path="/api/v1/restore")
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/v1/restore",
            "raw_path": b"/api/v1/restore",
            "query_string": b"",
            "headers": [],
            "client": None,
            "server": None,
            "state": {"request_id": "observed-request"},
            "app": app,
            "route": route,
        }
    )


@pytest.mark.parametrize(
    ("error", "outcome", "error_code", "level", "readiness"),
    [
        (InferenceBusyError(), "busy", "inference_busy", "WARNING", "ready"),
        (InferenceTimeoutError(), "timeout", "inference_timeout", "WARNING", "ready"),
        (
            ModelServiceUnavailableError("C:/private/model.pt"),
            "unavailable",
            "model_unavailable",
            "WARNING",
            "unavailable",
        ),
        (
            ModelServiceInferenceError("tensor secret"),
            "failed",
            "restoration_failed",
            "ERROR",
            "ready",
        ),
        (RuntimeError("token=secret"), "failed", "internal_error", "ERROR", "ready"),
    ],
)
def test_inference_outcomes_are_stable_and_do_not_leak_exceptions(
    error: Exception,
    outcome: str,
    error_code: str,
    level: str,
    readiness: str,
) -> None:
    settings = RuntimeSettings()
    stream = StringIO()
    configure_application_logging(settings, stream=stream)
    request = make_observation_request(settings)

    async def fail() -> None:
        raise error

    with pytest.raises(type(error)):
        asyncio.run(observe_inference(request, fail))

    record = events(stream)[0]
    assert record["event"] == "inference_completed"
    assert record["outcome"] == outcome
    assert record["error_code"] == error_code
    assert record["level"] == level
    assert record["model_readiness"] == readiness
    assert isinstance(record["inference_duration_ms"], float)
    assert record["inference_duration_ms"] >= 0
    for unsafe in (str(error), "C:/", "model.pt", "tensor", "secret", "token"):
        if unsafe:
            assert unsafe not in stream.getvalue()


def test_request_cancellation_propagates_without_internal_or_completion_event() -> None:
    settings = RuntimeSettings()
    stream = StringIO()
    configure_application_logging(settings, stream=stream)

    async def cancelled_app(scope: Any, receive: Any, send: Any) -> None:
        raise asyncio.CancelledError

    middleware = RequestObservabilityMiddleware(
        cancelled_app,
        environment="development",
        metrics=PlatformMetrics(inference_capacity=1),
    )
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/cancel",
        "headers": [],
        "state": {},
    }

    async def receive() -> dict[str, object]:
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        raise AssertionError(f"unexpected response: {message['type']}")

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(middleware(scope, receive, send))  # type: ignore[arg-type]

    records = events(stream)
    assert [record["event"] for record in records] == ["http_request_cancelled"]
    assert records[0]["outcome"] == "cancelled"
    assert records[0]["error_code"] is None


def test_inference_cancellation_propagates_and_is_not_logged_as_completion() -> None:
    settings = RuntimeSettings()
    stream = StringIO()
    configure_application_logging(settings, stream=stream)
    request = make_observation_request(settings)

    async def cancel() -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(observe_inference(request, cancel))

    records = events(stream)
    assert [record["event"] for record in records] == ["inference_cancelled"]
    assert records[0]["outcome"] == "cancelled"
    assert records[0]["error_code"] is None


def test_logger_configuration_is_idempotent_and_does_not_modify_root() -> None:
    root = logging.getLogger()
    root_handlers = tuple(root.handlers)
    logger = logging.getLogger(LOGGER_NAME)
    stream = StringIO()

    first = configure_application_logging(RuntimeSettings(), stream=stream)
    second = configure_application_logging(RuntimeSettings(), stream=stream)
    owned = [
        handler
        for handler in logger.handlers
        if getattr(handler, "_semirestore_owned_handler", False)
    ]

    assert first is second is logger
    assert len(owned) == 1
    assert logger.propagate is False
    assert tuple(root.handlers) == root_handlers


def test_logger_records_can_be_captured_by_pytest(caplog: pytest.LogCaptureFixture) -> None:
    app = create_app()
    logger = logging.getLogger(LOGGER_NAME)
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME), TestClient(app) as client:
            response = client.get("/health/live")
    finally:
        logger.removeHandler(caplog.handler)

    captured = [
        record
        for record in caplog.records
        if getattr(record, "semirestore_event", None) == "http_request_completed"
    ]
    assert response.status_code == 200
    assert len(captured) == 1


def test_repeated_application_creation_does_not_duplicate_request_events() -> None:
    create_app()
    create_app()
    app = create_app()
    stream = log_stream(app)

    with TestClient(app) as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    assert [record["event"] for record in events(stream)] == ["http_request_completed"]


def test_log_level_setting_filters_lower_severity_events() -> None:
    settings = RuntimeSettings(log_level="WARNING")
    app = create_app(settings=settings)
    stream = log_stream(app)

    with TestClient(app) as client:
        live = client.get("/health/live")
        missing = client.get("/missing")

    records = events(stream)
    assert live.status_code == 200
    assert missing.status_code == 404
    assert [record["level"] for record in records] == ["WARNING"]
    assert records[0]["status"] == 404


def test_human_logging_format_remains_safe_and_readable() -> None:
    settings = RuntimeSettings(json_logging=False, environment="local")
    app = create_app(settings=settings)
    stream = log_stream(app)

    with TestClient(app) as client:
        response = client.get("/health/live", headers={"x-request-id": "human-1"})

    line = stream.getvalue().strip()
    assert response.status_code == 200
    assert " INFO event=http_request_completed" in line
    assert 'environment="local"' in line
    assert 'request_id="human-1"' in line
    assert 'route="/health/live"' in line
    assert "status=200" in line
    assert not line.startswith("{")


def test_openapi_contract_remains_available_with_observability() -> None:
    app = create_app()
    schema = app.openapi()

    assert "/api/v1/restore" in schema["paths"]
    assert "/health/live" in schema["paths"]
    request_id_schema = ErrorBody.model_json_schema()["properties"]["request_id"]
    assert request_id_schema["anyOf"][0]["maxLength"] == 64
