from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from threading import Event
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image

from semirestore.api import create_app
from semirestore.platform import ModelServiceInferenceError, RuntimeSettings


def png_bytes() -> bytes:
    output = BytesIO()
    Image.new("L", (3, 2), color=32).save(output, format="PNG")
    return output.getvalue()


def files() -> dict[str, tuple[str, bytes, str]]:
    return {"image": ("input.png", png_bytes(), "image/png")}


def test_full_capacity_keeps_health_responsive_and_rejects_waiter(
    fake_model_service_factory: Any,
) -> None:
    release_restore = Event()
    service = fake_model_service_factory(restore_release_event=release_restore)
    settings = RuntimeSettings(
        inference_concurrency_limit=1,
        concurrency_acquisition_timeout_seconds=0.02,
        inference_timeout_seconds=1.0,
    )
    app = create_app(settings=settings, model_service_factory=lambda _: service)

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as executor:
        first_future = executor.submit(client.post, "/api/v1/restore", files=files())
        assert service.restoration_started_event.wait(timeout=1)
        try:
            live = client.get("/health/live")
            ready = client.get("/health/ready")
            invalid = client.post(
                "/api/v1/restore",
                files={"image": ("invalid.png", b"not an image", "image/png")},
            )
            busy = client.post("/api/v1/restore", files=files())

            assert live.status_code == 200
            assert ready.status_code == 200
            assert invalid.status_code == 422
            assert invalid.json()["error"]["code"] == "invalid_image"
            assert busy.status_code == 503
            assert busy.json()["error"]["code"] == "inference_busy"
            assert service.restoration_calls == 1
        finally:
            release_restore.set()

        assert first_future.result(timeout=1).status_code == 200


def test_execution_timeout_releases_slot_for_later_request(
    fake_model_service_factory: Any,
) -> None:
    service = fake_model_service_factory(inference_delay_seconds=0.1)
    settings = RuntimeSettings(
        inference_concurrency_limit=1,
        concurrency_acquisition_timeout_seconds=0.1,
        inference_timeout_seconds=0.01,
    )
    app = create_app(settings=settings, model_service_factory=lambda _: service)

    with TestClient(app) as client:
        timed_out = client.post("/api/v1/restore", files=files())
        service.inference_delay_seconds = None
        later = client.post("/api/v1/restore", files=files())

    assert timed_out.status_code == 504
    assert timed_out.json()["error"]["code"] == "inference_timeout"
    assert service.restoration_cancelled_event.is_set()
    assert later.status_code == 200
    assert service.restoration_calls == 2


def test_known_model_failure_releases_slot_for_later_request(
    fake_model_service_factory: Any,
) -> None:
    service = fake_model_service_factory(
        inference_error=ModelServiceInferenceError("synthetic model failure")
    )
    app = create_app(model_service_factory=lambda _: service)

    with TestClient(app) as client:
        failed = client.post("/api/v1/restore", files=files())
        service.inference_error = None
        later = client.post("/api/v1/restore", files=files())

    assert failed.status_code == 500
    assert failed.json()["error"]["code"] == "restoration_failed"
    assert later.status_code == 200
    assert service.restoration_calls == 2


def test_unexpected_model_failure_releases_slot_and_remains_sanitized(
    fake_model_service_factory: Any,
) -> None:
    service = fake_model_service_factory(
        inference_error=RuntimeError("C:/secret/checkpoint.pt token=secret")
    )
    app = create_app(model_service_factory=lambda _: service)

    with TestClient(app, raise_server_exceptions=False) as client:
        failed = client.post("/api/v1/restore", files=files())
        service.inference_error = None
        later = client.post("/api/v1/restore", files=files())

    assert failed.status_code == 500
    assert failed.json()["error"]["code"] == "internal_error"
    assert "checkpoint.pt" not in failed.text
    assert "token" not in failed.text
    assert later.status_code == 200


def test_controller_is_created_once_and_reused_without_changing_response(
    fake_model_service: Any,
) -> None:
    app = create_app(model_service_factory=lambda _: fake_model_service)

    with TestClient(app) as client:
        controller = app.state.runtime.inference_gate
        first = client.post("/api/v1/restore", files=files())
        second = client.post("/api/v1/restore", files=files())

        assert controller is not None
        assert app.state.runtime.inference_gate is controller

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json().keys() == {
        "image",
        "input",
        "inference",
        "model",
        "diagnostics",
        "warnings",
    }
    assert fake_model_service.restoration_calls == 2
