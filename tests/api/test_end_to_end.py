from __future__ import annotations

import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO, StringIO
from pathlib import Path
from threading import Event
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image
from prometheus_client.parser import text_string_to_metric_families

from semirestore.api import create_app
from semirestore.api.observability import configure_application_logging
from semirestore.platform import ModelService, RuntimeSettings

REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}\Z")


def image_bytes(image_format: str, size: tuple[int, int] = (5, 4)) -> bytes:
    output = BytesIO()
    Image.new("L", size, color=96).save(output, format=image_format)
    return output.getvalue()


def upload(
    encoded: bytes,
    media_type: str,
    filename: str = "input.sem",
) -> dict[str, tuple[str, bytes, str]]:
    return {"image": (filename, encoded, media_type)}


def capture_logs(app: Any) -> StringIO:
    stream = StringIO()
    configure_application_logging(app.state.runtime.settings, stream=stream)
    return stream


def log_events(stream: StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line]


def test_unconfigured_production_application_is_live_safe_and_unready(
    tmp_path: Path,
) -> None:
    private_checkpoint = tmp_path / "private" / "checkpoint.pt"
    settings = RuntimeSettings(
        environment="production",
        checkpoint_path=private_checkpoint,
    )
    app = create_app(settings=settings)
    stream = capture_logs(app)

    assert settings.enable_fake_model_service is False
    assert app.state.runtime.model_service is None

    with TestClient(app) as client:
        assert app.state.runtime.startup_complete is True
        assert app.state.runtime.model_service is None

        live = client.get("/health/live")
        ready = client.get("/health/ready")
        model = client.get("/health/model")
        version = client.get("/version")
        metrics = client.get("/metrics")
        restoration = client.post(
            "/api/v1/restore",
            files=upload(image_bytes("PNG"), "image/png", str(private_checkpoint)),
            headers={"x-request-id": "production-unavailable"},
        )

    assert live.status_code == 200
    assert live.json() == {"status": "alive"}
    assert ready.status_code == 503
    assert ready.json() == {
        "ready": False,
        "state": "unavailable",
        "unavailable_reason": "model service adapter is not configured",
    }
    assert model.status_code == 200
    assert model.json() == {
        "ready": False,
        "state": "unavailable",
        "unavailable_reason": "model service adapter is not configured",
        "device": None,
        "model_version": None,
        "checkpoint_checksum": None,
    }
    assert version.status_code == 200
    assert version.json()["application"] == "semirestore"
    assert version.json()["version"]
    assert metrics.status_code == 200
    assert "semirestore_inference_capacity" in metrics.text
    assert restoration.status_code == 503
    assert restoration.headers["x-request-id"] == "production-unavailable"
    assert restoration.json()["error"] == {
        "code": "model_unavailable",
        "message": "The model service is unavailable.",
        "details": None,
        "request_id": "production-unavailable",
    }

    for response in (live, ready, model, version, metrics, restoration):
        assert REQUEST_ID.fullmatch(response.headers["x-request-id"])

    assert app.state.runtime.startup_complete is False
    assert app.state.runtime.model_service is None
    assert app.state.runtime.inference_gate is None
    assert not private_checkpoint.exists()
    for unsafe in (str(tmp_path), "checkpoint.pt", "private"):
        assert unsafe not in stream.getvalue()

    fake_flag_app = create_app(
        settings=RuntimeSettings(
            environment="production",
            enable_fake_model_service=True,
        )
    )
    with TestClient(fake_flag_app) as client:
        flag_ready = client.get("/health/ready")
        assert fake_flag_app.state.runtime.model_service is None
    assert flag_ready.status_code == 503
    assert flag_ready.json()["state"] == "unavailable"


def test_injected_service_restores_all_formats_reuses_lifecycle_and_is_private(
    tmp_path: Path,
    fake_model_service: Any,
) -> None:
    factory_calls = 0

    def factory(_: RuntimeSettings) -> ModelService:
        nonlocal factory_calls
        factory_calls += 1
        return fake_model_service

    app = create_app(model_service_factory=factory)
    stream = capture_logs(app)
    inputs = (
        ("PNG", "image/png"),
        ("JPEG", "image/jpeg"),
        ("TIFF", "image/tiff"),
    )
    response_ids: list[str] = []
    response_payloads: list[dict[str, Any]] = []

    with TestClient(app) as client:
        service_reference = app.state.runtime.model_service
        gate_reference = app.state.runtime.inference_gate
        assert service_reference is fake_model_service
        assert gate_reference is not None

        for index, (image_format, media_type) in enumerate(inputs):
            request_id = f"format-{index}"
            unsafe_filename = str(tmp_path / "secret" / f"checkpoint-{index}.pt")
            response = client.post(
                "/api/v1/restore?token=never-log",
                files=upload(image_bytes(image_format), media_type, unsafe_filename),
                headers={"x-request-id": request_id},
            )

            assert response.status_code == 200
            assert response.headers["x-request-id"] == request_id
            assert app.state.runtime.model_service is service_reference
            assert app.state.runtime.inference_gate is gate_reference
            response_ids.append(request_id)
            response_payloads.append(response.json())

        metrics = client.get("/metrics")

    assert factory_calls == 1
    assert fake_model_service.startup_calls == 1
    assert fake_model_service.restoration_calls == 3
    assert fake_model_service.shutdown_calls == 1
    assert app.state.runtime.model_service is None
    assert app.state.runtime.inference_gate is None

    for payload, (_, input_media_type) in zip(response_payloads, inputs, strict=True):
        assert base64.b64decode(payload["image"]["content"]) == (
            fake_model_service.synthetic_output_bytes
        )
        assert payload["image"]["media_type"] == "image/png"
        assert (payload["image"]["width"], payload["image"]["height"]) == (2, 2)
        assert payload["input"] == {
            "width": 5,
            "height": 4,
            "media_type": input_media_type,
        }
        assert payload["diagnostics"]["synthetic"] is True
        assert payload["warnings"] == [
            "Synthetic test output; no real restoration was performed."
        ]
        assert payload["model"] == {
            "version": "synthetic-test-model",
            "checkpoint_checksum": f"sha256:{'a' * 64}",
        }

    assert [item["detected_format"] for item in fake_model_service.restoration_inputs] == [
        "PNG",
        "JPEG",
        "TIFF",
    ]
    assert list(tmp_path.rglob("*")) == []
    assert metrics.status_code == 200
    assert (
        app.state.runtime.metrics.registry.get_sample_value(
            "semirestore_restoration_requests_total",
            {"outcome": "success"},
        )
        == 3
    )

    metric_samples = [
        sample
        for family in text_string_to_metric_families(metrics.text)
        for sample in family.samples
    ]
    metric_labels = {
        label
        for sample in metric_samples
        for label in sample.labels
    }
    assert metric_labels <= {"method", "route", "status_class", "outcome", "le"}
    assert {sample.labels["method"] for sample in metric_samples if "method" in sample.labels} <= {
        "POST"
    }
    assert {sample.labels["route"] for sample in metric_samples if "route" in sample.labels} <= {
        "/api/v1/restore"
    }
    assert {
        sample.labels["status_class"]
        for sample in metric_samples
        if "status_class" in sample.labels
    } <= {"2xx"}
    assert {
        sample.labels["outcome"]
        for sample in metric_samples
        if "outcome" in sample.labels
    } <= {"success"}
    for request_id in response_ids:
        assert request_id not in metrics.text
    log_text = stream.getvalue()
    for unsafe in (
        str(tmp_path),
        "checkpoint-0.pt",
        "never-log",
        *(payload["image"]["content"] for payload in response_payloads),
    ):
        assert unsafe not in log_text
    for request_id in response_ids:
        assert request_id in log_text
    assert all(
        event["route"] in {"/api/v1/restore", "/metrics"}
        for event in log_events(stream)
    )


def test_injected_service_upload_limit_returns_correlated_error(
    fake_model_service: Any,
) -> None:
    encoded = image_bytes("PNG")
    settings = RuntimeSettings(max_encoded_upload_bytes=len(encoded) - 1)
    app = create_app(settings=settings, model_service_factory=lambda _: fake_model_service)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/restore",
            files=upload(encoded, "image/png"),
            headers={"x-request-id": "upload-limited"},
        )

    assert response.status_code == 413
    assert response.headers["x-request-id"] == "upload-limited"
    assert response.json()["error"]["code"] == "upload_too_large"
    assert response.json()["error"]["request_id"] == "upload-limited"
    assert fake_model_service.restoration_calls == 0


def test_injected_service_capacity_and_timeout_are_enforced(
    fake_model_service_factory: Any,
) -> None:
    release_restore = Event()
    busy_service = fake_model_service_factory(restore_release_event=release_restore)
    busy_settings = RuntimeSettings(
        inference_concurrency_limit=1,
        concurrency_acquisition_timeout_seconds=0.02,
        inference_timeout_seconds=1.0,
    )
    busy_app = create_app(
        settings=busy_settings,
        model_service_factory=lambda _: busy_service,
    )

    with TestClient(busy_app) as client, ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(
            client.post,
            "/api/v1/restore",
            files=upload(image_bytes("PNG"), "image/png"),
        )
        assert busy_service.restoration_started_event.wait(timeout=1)
        try:
            busy = client.post(
                "/api/v1/restore",
                files=upload(image_bytes("PNG"), "image/png"),
                headers={"x-request-id": "capacity-busy"},
            )
        finally:
            release_restore.set()
        assert first.result(timeout=1).status_code == 200

    assert busy.status_code == 503
    assert busy.headers["x-request-id"] == "capacity-busy"
    assert busy.json()["error"]["code"] == "inference_busy"
    assert busy.json()["error"]["request_id"] == "capacity-busy"
    assert busy_service.maximum_active_restorations == 1
    assert (
        busy_app.state.runtime.metrics.registry.get_sample_value(
            "semirestore_inference_busy_total"
        )
        == 1
    )

    timeout_service = fake_model_service_factory(inference_delay_seconds=0.1)
    timeout_settings = RuntimeSettings(
        inference_concurrency_limit=1,
        concurrency_acquisition_timeout_seconds=0.1,
        inference_timeout_seconds=0.01,
    )
    timeout_app = create_app(
        settings=timeout_settings,
        model_service_factory=lambda _: timeout_service,
    )

    with TestClient(timeout_app) as client:
        timed_out = client.post(
            "/api/v1/restore",
            files=upload(image_bytes("PNG"), "image/png"),
            headers={"x-request-id": "execution-timeout"},
        )

    assert timed_out.status_code == 504
    assert timed_out.headers["x-request-id"] == "execution-timeout"
    assert timed_out.json()["error"]["code"] == "inference_timeout"
    assert timed_out.json()["error"]["request_id"] == "execution-timeout"
    assert timeout_service.restoration_cancelled_event.is_set()
    assert (
        timeout_app.state.runtime.metrics.registry.get_sample_value(
            "semirestore_inference_timeouts_total"
        )
        == 1
    )


def test_openapi_exposes_complete_public_contract_and_stable_errors() -> None:
    schema = create_app().openapi()
    paths = schema["paths"]
    operation = paths["/api/v1/restore"]["post"]

    assert {
        "/health/live",
        "/health/ready",
        "/health/model",
        "/version",
        "/api/v1/restore",
    } <= paths.keys()
    assert "/metrics" not in paths
    assert "multipart/form-data" in operation["requestBody"]["content"]
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/RestoreResponse"
    }
    for status_code in ("400", "413", "415", "422", "500", "503", "504"):
        assert operation["responses"][status_code]["content"]["application/json"][
            "schema"
        ] == {"$ref": "#/components/schemas/ErrorResponse"}
    assert {"ErrorResponse", "ErrorBody", "ErrorCode"} <= schema["components"][
        "schemas"
    ].keys()
