from __future__ import annotations

import asyncio
import math
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from PIL import Image
from prometheus_client import generate_latest
from prometheus_client.parser import text_string_to_metric_families

from semirestore.api import create_app
from semirestore.api.metrics import PlatformMetrics
from semirestore.api.observability import (
    RequestObservabilityMiddleware,
    observe_inference,
    observe_restoration,
)
from semirestore.platform import ModelServiceInferenceError, RuntimeSettings


def png_bytes() -> bytes:
    output = BytesIO()
    Image.new("L", (3, 2), color=32).save(output, format="PNG")
    return output.getvalue()


def files(filename: str = "input.png") -> dict[str, tuple[str, bytes, str]]:
    return {"image": (filename, png_bytes(), "image/png")}


def sample(
    metrics: PlatformMetrics,
    name: str,
    labels: dict[str, str] | None = None,
) -> float | None:
    return metrics.registry.get_sample_value(name, labels)


def exposition(metrics: PlatformMetrics) -> str:
    return generate_latest(metrics.registry).decode("utf-8")


def test_metrics_endpoint_is_available_without_model_and_uses_isolated_registry() -> None:
    app = create_app()
    registry = app.state.runtime.metrics.registry

    with TestClient(app) as client:
        response = client.get("/metrics")
        assert app.state.runtime.metrics.registry is registry

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain; version=")
    assert "charset=utf-8" in response.headers["content-type"]
    assert response.headers["cache-control"] == "no-store"
    assert "/metrics" not in app.openapi()["paths"]


def test_expected_metric_samples_are_exposed_after_successful_restoration(
    fake_model_service: Any,
) -> None:
    app = create_app(model_service_factory=lambda _: fake_model_service)

    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.post("/api/v1/restore", files=files()).status_code == 200
        response = client.get("/metrics")

    parsed_samples = {
        metric_sample.name
        for family in text_string_to_metric_families(response.text)
        for metric_sample in family.samples
    }
    assert {
        "semirestore_http_requests_total",
        "semirestore_http_request_duration_seconds_count",
        "semirestore_restoration_requests_total",
        "semirestore_inference_duration_seconds_count",
        "semirestore_inference_active",
        "semirestore_inference_waiting",
        "semirestore_inference_capacity",
        "semirestore_inference_busy_total",
        "semirestore_inference_timeouts_total",
    } <= parsed_samples


def test_multiple_applications_do_not_collide_or_share_values() -> None:
    first_app = create_app()
    second_app = create_app()
    first_metrics = first_app.state.runtime.metrics
    second_metrics = second_app.state.runtime.metrics

    assert first_metrics.registry is not second_metrics.registry
    with TestClient(first_app) as client:
        assert client.get("/health/live").status_code == 200

    labels = {"method": "GET", "route": "/health/live", "status_class": "2xx"}
    assert sample(first_metrics, "semirestore_http_requests_total", labels) == 1
    assert sample(second_metrics, "semirestore_http_requests_total", labels) is None


def test_http_metrics_count_once_and_bound_unmatched_paths_methods_and_queries() -> None:
    app = create_app()
    metrics = app.state.runtime.metrics
    raw_path = "/private/checkpoint-secret.pt"

    with TestClient(app) as client:
        live = client.get("/health/live?token=never-export")
        missing = client.get(f"{raw_path}?token=never-export")
        unusual = client.request("BREW", "/health/live")
        first_scrape = client.get("/metrics")
        second_scrape = client.get("/metrics")

    assert live.status_code == 200
    assert missing.status_code == 404
    assert unusual.status_code == 405
    assert first_scrape.status_code == second_scrape.status_code == 200
    assert sample(
        metrics,
        "semirestore_http_requests_total",
        {"method": "GET", "route": "/health/live", "status_class": "2xx"},
    ) == 1
    assert sample(
        metrics,
        "semirestore_http_requests_total",
        {"method": "GET", "route": "<unmatched>", "status_class": "4xx"},
    ) == 1
    assert sample(
        metrics,
        "semirestore_http_requests_total",
        {"method": "OTHER", "route": "/health/live", "status_class": "4xx"},
    ) == 1
    assert sample(
        metrics,
        "semirestore_http_request_duration_seconds_count",
        {"method": "GET", "route": "/health/live", "status_class": "2xx"},
    ) == 1
    assert sample(
        metrics,
        "semirestore_http_requests_total",
        {"method": "GET", "route": "/metrics", "status_class": "2xx"},
    ) is None
    exported = second_scrape.text
    assert raw_path not in exported
    assert "never-export" not in exported


def test_metrics_exclude_request_ids_filenames_exception_text_and_model_metadata(
    fake_model_service_factory: Any,
) -> None:
    unsafe_exception = "C:/private/checkpoint-secret.pt token=never-export tensor=[1]"
    service = fake_model_service_factory(
        inference_error=ModelServiceInferenceError(unsafe_exception)
    )
    app = create_app(model_service_factory=lambda _: service)
    unsafe_filename = "C:/private/checkpoint-secret.pt-token=never-export.png"
    request_id = "sensitive-request-id"

    with TestClient(app) as client:
        failed = client.post(
            "/api/v1/restore?query_secret=never-export",
            files=files(unsafe_filename),
            headers={"x-request-id": request_id},
        )
        service.inference_error = None
        success = client.post("/api/v1/restore", files=files(unsafe_filename))
        response = client.get("/metrics")

    assert failed.status_code == 500
    assert success.status_code == 200
    for unsafe in (
        request_id,
        unsafe_filename,
        unsafe_exception,
        "never-export",
        "checkpoint-secret.pt",
        "tensor=[1]",
        success.json()["image"]["content"],
        "test-checksum",
        fake_model_service_factory().synthetic_output_bytes.hex(),
    ):
        assert unsafe not in response.text


@pytest.mark.parametrize(
    ("service_error", "expected_code"),
    [
        (ModelServiceInferenceError("known private failure"), "restoration_failed"),
        (RuntimeError("C:/private/checkpoint.pt token=secret"), "internal_error"),
    ],
)
def test_failed_restoration_outcomes_are_bounded(
    fake_model_service_factory: Any,
    service_error: Exception,
    expected_code: str,
) -> None:
    service = fake_model_service_factory(inference_error=service_error)
    app = create_app(model_service_factory=lambda _: service)
    metrics = app.state.runtime.metrics

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/v1/restore", files=files())

    assert response.json()["error"]["code"] == expected_code
    assert sample(
        metrics,
        "semirestore_restoration_requests_total",
        {"outcome": "failed"},
    ) == 1
    assert sample(
        metrics,
        "semirestore_inference_duration_seconds_count",
        {"outcome": "failed"},
    ) == 1
    assert sample(
        metrics,
        "semirestore_http_requests_total",
        {"method": "POST", "route": "/api/v1/restore", "status_class": "5xx"},
    ) == 1


def test_success_and_unavailable_restoration_outcomes(
    fake_model_service: Any,
) -> None:
    ready_app = create_app(model_service_factory=lambda _: fake_model_service)
    unavailable_app = create_app()

    with TestClient(ready_app) as client:
        success = client.post("/api/v1/restore", files=files())
    with TestClient(unavailable_app) as client:
        unavailable = client.post("/api/v1/restore", files=files())

    assert success.status_code == 200
    assert unavailable.status_code == 503
    assert sample(
        ready_app.state.runtime.metrics,
        "semirestore_restoration_requests_total",
        {"outcome": "success"},
    ) == 1
    assert sample(
        unavailable_app.state.runtime.metrics,
        "semirestore_restoration_requests_total",
        {"outcome": "unavailable"},
    ) == 1
    assert sample(
        unavailable_app.state.runtime.metrics,
        "semirestore_inference_duration_seconds_count",
        {"outcome": "unavailable"},
    ) == 1


def test_busy_and_timeout_outcomes_remain_distinct(
    fake_model_service_factory: Any,
) -> None:
    release_restore = Event()
    busy_service = fake_model_service_factory(restore_release_event=release_restore)
    busy_settings = RuntimeSettings(
        concurrency_acquisition_timeout_seconds=0.02,
        inference_timeout_seconds=1.0,
    )
    busy_app = create_app(
        settings=busy_settings,
        model_service_factory=lambda _: busy_service,
    )

    with TestClient(busy_app) as client, ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(client.post, "/api/v1/restore", files=files())
        assert busy_service.restoration_started_event.wait(timeout=1)
        try:
            busy = client.post("/api/v1/restore", files=files())
        finally:
            release_restore.set()
        assert first.result(timeout=1).status_code == 200

    timeout_service = fake_model_service_factory(inference_delay_seconds=0.1)
    timeout_app = create_app(
        settings=RuntimeSettings(inference_timeout_seconds=0.01),
        model_service_factory=lambda _: timeout_service,
    )
    with TestClient(timeout_app) as client:
        timed_out = client.post("/api/v1/restore", files=files())

    assert busy.status_code == 503
    assert timed_out.status_code == 504
    assert sample(
        busy_app.state.runtime.metrics,
        "semirestore_restoration_requests_total",
        {"outcome": "busy"},
    ) == 1
    assert sample(
        busy_app.state.runtime.metrics,
        "semirestore_inference_busy_total",
    ) == 1
    assert sample(
        timeout_app.state.runtime.metrics,
        "semirestore_restoration_requests_total",
        {"outcome": "timeout"},
    ) == 1
    assert sample(
        timeout_app.state.runtime.metrics,
        "semirestore_inference_timeouts_total",
    ) == 1
    assert sample(
        timeout_app.state.runtime.metrics,
        "semirestore_inference_active",
    ) == 0


def make_request(metrics: PlatformMetrics) -> Request:
    runtime = SimpleNamespace(
        settings=RuntimeSettings(),
        metrics=metrics,
    )
    app = SimpleNamespace(state=SimpleNamespace(runtime=runtime))
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
            "state": {"request_id": "cancelled-request"},
            "app": app,
            "route": SimpleNamespace(path="/api/v1/restore"),
        }
    )


def test_cancelled_restoration_and_inference_use_bounded_outcomes() -> None:
    metrics = PlatformMetrics(inference_capacity=1)
    request = make_request(metrics)

    async def cancel() -> None:
        raise asyncio.CancelledError

    async def scenario() -> None:
        await observe_restoration(request, lambda: observe_inference(request, cancel))

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(scenario())

    assert sample(
        metrics,
        "semirestore_restoration_requests_total",
        {"outcome": "cancelled"},
    ) == 1
    assert sample(
        metrics,
        "semirestore_inference_duration_seconds_count",
        {"outcome": "cancelled"},
    ) == 1


def test_cancelled_http_request_uses_bounded_status_without_becoming_500() -> None:
    metrics = PlatformMetrics(inference_capacity=1)

    async def cancelled_app(scope: Any, receive: Any, send: Any) -> None:
        raise asyncio.CancelledError

    middleware = RequestObservabilityMiddleware(
        cancelled_app,
        environment="test",
        metrics=metrics,
    )
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/restore",
        "headers": [],
        "state": {},
        "route": SimpleNamespace(path="/api/v1/restore"),
    }

    async def receive() -> dict[str, object]:
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        raise AssertionError(f"unexpected response: {message['type']}")

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(middleware(scope, receive, send))  # type: ignore[arg-type]

    assert sample(
        metrics,
        "semirestore_http_requests_total",
        {
            "method": "POST",
            "route": "/api/v1/restore",
            "status_class": "cancelled",
        },
    ) == 1
    assert sample(
        metrics,
        "semirestore_http_requests_total",
        {"method": "POST", "route": "/api/v1/restore", "status_class": "5xx"},
    ) is None
def test_upload_validation_failure_never_enters_inference_metrics(
    fake_model_service: Any,
) -> None:
    app = create_app(model_service_factory=lambda _: fake_model_service)
    metrics = app.state.runtime.metrics

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/restore",
            files={"image": ("unsafe.png", b"not an image", "image/png")},
        )

    assert response.status_code == 422
    assert sample(metrics, "semirestore_inference_active") == 0
    assert sample(metrics, "semirestore_inference_waiting") == 0
    assert sample(
        metrics,
        "semirestore_restoration_requests_total",
        {"outcome": "failed"},
    ) is None


def test_metric_histogram_observations_are_finite_and_non_negative(
    fake_model_service: Any,
) -> None:
    app = create_app(model_service_factory=lambda _: fake_model_service)
    metrics = app.state.runtime.metrics

    with TestClient(app) as client:
        client.post("/api/v1/restore", files=files())

    request_sum = sample(
        metrics,
        "semirestore_http_request_duration_seconds_sum",
        {"method": "POST", "route": "/api/v1/restore", "status_class": "2xx"},
    )
    inference_sum = sample(
        metrics,
        "semirestore_inference_duration_seconds_sum",
        {"outcome": "success"},
    )
    assert request_sum is not None and math.isfinite(request_sum) and request_sum >= 0
    assert inference_sum is not None and math.isfinite(inference_sum) and inference_sum >= 0


def test_metrics_do_not_change_health_contract(fake_model_service: Any) -> None:
    app = create_app(model_service_factory=lambda _: fake_model_service)

    with TestClient(app) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")

    assert live.status_code == 200
    assert live.json() == {"status": "alive"}
    assert ready.status_code == 200
    assert ready.json()["ready"] is True


def test_exposition_contains_no_process_or_readiness_collectors() -> None:
    app = create_app()
    exported = exposition(app.state.runtime.metrics)

    assert "process_" not in exported
    assert "python_" not in exported
    assert "model_readiness" not in exported
