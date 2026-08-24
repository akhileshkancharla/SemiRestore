from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from semirestore.api import create_app
from semirestore.platform import (
    ModelHealth,
    ModelService,
    ModelServiceInferenceError,
    ModelServiceState,
    RestorationResult,
    RuntimeSettings,
)


def make_image(image_format: str, size: tuple[int, int] = (4, 3)) -> bytes:
    output = BytesIO()
    Image.new("L", size, color=64).save(output, format=image_format)
    return output.getvalue()


def upload_file(
    encoded: bytes,
    media_type: str,
    filename: str = "input.sem",
) -> dict[str, tuple[str, bytes, str]]:
    return {"image": (filename, encoded, media_type)}


def test_successful_png_restoration_serializes_complete_contract(
    fake_model_service: Any,
) -> None:
    input_bytes = make_image("PNG")
    app = create_app(model_service_factory=lambda _: fake_model_service)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/restore",
            files=upload_file(input_bytes, "image/png", "input.png"),
        )

    assert response.status_code == 200
    body = response.json()
    assert base64.b64decode(body["image"]["content"]) == fake_model_service.synthetic_output_bytes
    assert body["image"] == {
        "encoding": "base64",
        "media_type": "image/png",
        "content": body["image"]["content"],
        "width": 2,
        "height": 2,
    }
    assert body["input"] == {"width": 4, "height": 3, "media_type": "image/png"}
    assert body["inference"] == {
        "latency_ms": 12.5,
        "device": "test-device",
        "phase_latency_ms": {"restoration_total": 12.5},
    }
    assert body["model"] == {
        "name": "synthetic-naf-sr",
        "version": "synthetic-test-model",
        "training_revision": "synthetic-revision",
        "checkpoint_checksum": f"sha256:{'a' * 64}",
    }
    assert body["diagnostics"] == {"synthetic": True, "source_format": "PNG"}
    assert body["warnings"] == ["Synthetic test output; no real restoration was performed."]
    assert fake_model_service.restoration_calls == 1


@pytest.mark.parametrize(
    ("image_format", "media_type"),
    [("JPEG", "image/jpeg"), ("TIFF", "image/tiff")],
)
def test_supported_non_png_inputs_are_restored(
    fake_model_service: Any,
    image_format: str,
    media_type: str,
) -> None:
    app = create_app(model_service_factory=lambda _: fake_model_service)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/restore",
            files=upload_file(make_image(image_format), media_type),
        )

    assert response.status_code == 200
    assert response.json()["input"]["media_type"] == media_type
    assert fake_model_service.restoration_inputs[0]["detected_format"] == image_format


def test_lifespan_service_is_reused_across_restoration_requests(
    fake_model_service: ModelService,
) -> None:
    factory_calls = 0

    def factory(_: RuntimeSettings) -> ModelService:
        nonlocal factory_calls
        factory_calls += 1
        return fake_model_service

    app = create_app(model_service_factory=factory)
    files = upload_file(make_image("PNG"), "image/png")

    with TestClient(app) as client:
        first = client.post("/api/v1/restore", files=files)
        second = client.post("/api/v1/restore", files=files)

    assert first.status_code == 200
    assert second.status_code == 200
    assert factory_calls == 1
    assert fake_model_service.startup_calls == 1  # type: ignore[attr-defined]
    assert fake_model_service.restoration_calls == 2  # type: ignore[attr-defined]
    assert fake_model_service.shutdown_calls == 1  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("files", "expected_status", "expected_code"),
    [
        (None, 400, "empty_upload"),
        (upload_file(b"", "image/png"), 400, "empty_upload"),
        (upload_file(b"<svg/>", "image/svg+xml"), 415, "unsupported_media_type"),
        (upload_file(b"not an image", "image/png"), 422, "invalid_image"),
    ],
)
def test_upload_failures_use_stable_error_contract(
    fake_model_service: Any,
    files: dict[str, tuple[str, bytes, str]] | None,
    expected_status: int,
    expected_code: str,
) -> None:
    app = create_app(model_service_factory=lambda _: fake_model_service)

    with TestClient(app) as client:
        response = client.post("/api/v1/restore", files=files)

    assert response.status_code == expected_status
    assert response.json()["error"]["code"] == expected_code
    assert fake_model_service.restoration_calls == 0


def test_oversized_upload_is_rejected_before_inference(fake_model_service: Any) -> None:
    encoded = make_image("PNG")
    settings = RuntimeSettings(max_encoded_upload_bytes=len(encoded) - 1)
    app = create_app(settings=settings, model_service_factory=lambda _: fake_model_service)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/restore",
            files=upload_file(encoded, "image/png"),
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "upload_too_large"
    assert fake_model_service.restoration_calls == 0


def test_excessive_decoded_dimensions_are_rejected(fake_model_service: Any) -> None:
    settings = RuntimeSettings(max_decoded_image_width=3)
    app = create_app(settings=settings, model_service_factory=lambda _: fake_model_service)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/restore",
            files=upload_file(make_image("PNG", size=(4, 2)), "image/png"),
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "image_dimensions_exceeded"
    assert fake_model_service.restoration_calls == 0


def test_missing_checkpoint_application_has_no_fake_fallback(tmp_path: Path) -> None:
    app = create_app(settings=RuntimeSettings(checkpoint_path=tmp_path / "missing.pt"))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/restore",
            files=upload_file(make_image("PNG"), "image/png"),
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"


def test_unready_service_is_not_called(fake_model_service_factory: Any) -> None:
    service = fake_model_service_factory(
        health=ModelHealth(
            state=ModelServiceState.UNAVAILABLE,
            ready=False,
            unavailable_reason="synthetic test service is unready",
        )
    )
    app = create_app(model_service_factory=lambda _: service)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/restore",
            files=upload_file(make_image("PNG"), "image/png"),
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"
    assert service.restoration_calls == 0


def test_model_inference_failure_does_not_leak_exception(fake_model_service_factory: Any) -> None:
    service = fake_model_service_factory(
        inference_error=ModelServiceInferenceError(
            "C:/secret/checkpoint.pt token=secret tensor=[1, 2]"
        )
    )
    app = create_app(model_service_factory=lambda _: service)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/restore",
            files=upload_file(make_image("PNG"), "image/png"),
        )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "restoration_failed"
    for unsafe_text in ("C:/", "checkpoint.pt", "token", "tensor", "Traceback"):
        assert unsafe_text not in response.text


def test_invalid_service_result_is_mapped_to_restoration_failed(
    fake_model_service_factory: Any,
) -> None:
    invalid_result = object.__new__(RestorationResult)
    service = fake_model_service_factory(restoration_result=invalid_result)
    app = create_app(model_service_factory=lambda _: service)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/restore",
            files=upload_file(make_image("PNG"), "image/png"),
        )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "restoration_failed"
    assert response.json()["error"]["details"] is None


def test_unavailable_optional_result_metadata_remains_explicitly_empty(
    fake_model_service_factory: Any,
) -> None:
    result = RestorationResult(
        restored_image_bytes=make_image("PNG", size=(2, 2)),
        restored_media_type="image/png",
        restored_width=2,
        restored_height=2,
        original_width=4,
        original_height=3,
    )
    service = fake_model_service_factory(restoration_result=result)
    app = create_app(model_service_factory=lambda _: service)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/restore",
            files=upload_file(make_image("PNG"), "image/png"),
        )

    assert response.status_code == 200
    assert response.json()["inference"] == {
        "latency_ms": None,
        "device": None,
        "phase_latency_ms": {},
    }
    assert response.json()["model"] == {
        "name": None,
        "version": None,
        "training_revision": None,
        "checkpoint_checksum": None,
    }
    assert response.json()["diagnostics"] == {}
    assert response.json()["warnings"] == []


def test_malformed_multipart_uses_invalid_request_envelope(fake_model_service: Any) -> None:
    app = create_app(model_service_factory=lambda _: fake_model_service)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/restore",
            content=b"--broken\r\ninvalid multipart header\r\n\r\npayload",
            headers={"content-type": "multipart/form-data; boundary=broken"},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
    assert "multipart header" not in response.text


def test_images_are_not_persisted(tmp_path: Path, fake_model_service: Any) -> None:
    app = create_app(model_service_factory=lambda _: fake_model_service)
    path_like_name = str(tmp_path / "nested" / "input.png")

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/restore",
            files=upload_file(make_image("PNG"), "image/png", path_like_name),
        )

    assert response.status_code == 200
    assert list(tmp_path.iterdir()) == []


def test_openapi_describes_multipart_input_and_typed_response() -> None:
    app = create_app()

    schema = app.openapi()
    operation = schema["paths"]["/api/v1/restore"]["post"]

    assert "multipart/form-data" in operation["requestBody"]["content"]
    success_schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert success_schema == {"$ref": "#/components/schemas/RestoreResponse"}
    analysis_operation = schema["paths"]["/api/v1/analyze"]["post"]
    combined_operation = schema["paths"]["/api/v1/restore-and-analyze"]["post"]
    assert "multipart/form-data" in analysis_operation["requestBody"]["content"]
    assert analysis_operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/AnalyzeResponse"}
    assert combined_operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/RestoreResponse"}


def test_health_endpoints_remain_compatible(fake_model_service: Any) -> None:
    app = create_app(model_service_factory=lambda _: fake_model_service)

    with TestClient(app) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")

    assert live.status_code == 200
    assert ready.status_code == 200
