from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable

import httpx
import pytest
from fastapi.testclient import TestClient

from semirestore.api import create_app
from semirestore.model_manager import DEFAULT_CHECKPOINT_PATH
from semirestore.platform import RuntimeSettings
from semirestore.platform.load_testing import synthetic_grayscale_png
from semirestore.platform.smoke_testing import (
    SmokeOperation,
    SmokeTestConfig,
    SmokeTestError,
    SmokeTestUnavailableError,
    report_payload,
    run_smoke_test,
    validate_operation_payload,
)


def _restoration_document(width: int, height: int) -> dict[str, object]:
    restored = synthetic_grayscale_png(width * 2, height * 2)
    return {
        "image": {
            "encoding": "base64",
            "media_type": "image/png",
            "content": base64.b64encode(restored).decode("ascii"),
            "width": width * 2,
            "height": height * 2,
        },
        "input": {"width": width, "height": height, "media_type": "image/png"},
        "inference": {
            "latency_ms": 12.5,
            "device": "cpu",
            "phase_latency_ms": {"restoration_total": 12.5},
        },
        "model": {
            "name": "test-model",
            "version": "test-version",
            "training_revision": "test-revision",
            "checkpoint_checksum": f"sha256:{'a' * 64}",
        },
        "diagnostics": {"input": {"mean": 0.5}, "restored": {"mean": 0.6}},
        "warnings": ["Synthetic response."],
    }


def _analysis_document(width: int, height: int) -> dict[str, object]:
    return {
        "input": {"width": width, "height": height, "media_type": "image/png"},
        "analysis": {"latency_ms": 1.25},
        "diagnostics": {"input": {"mean": 0.5}},
        "suitability": {
            "recommendation": "warn",
            "reasons": ["Synthetic response."],
            "advisory_not_probability": True,
        },
        "warnings": ["Synthetic response."],
    }


def _transport(
    operation: SmokeOperation,
    document: dict[str, object],
    observed_paths: list[str],
) -> httpx.MockTransport:
    operation_path = f"/api/v1/{operation.value}"

    def handler(request: httpx.Request) -> httpx.Response:
        observed_paths.append(request.url.path)
        if request.url.path == "/health/live":
            return httpx.Response(200, json={"status": "alive"})
        if request.url.path == "/health/ready":
            return httpx.Response(
                200,
                json={"ready": True, "state": "ready", "unavailable_reason": None},
            )
        if request.url.path == "/health/model":
            return httpx.Response(
                200,
                json={
                    "ready": True,
                    "state": "ready",
                    "device": "cpu",
                    "model_version": "test-version",
                    "checkpoint_checksum": f"sha256:{'a' * 64}",
                    "unavailable_reason": None,
                },
            )
        if request.url.path == operation_path:
            assert request.method == "POST"
            assert b'name="image"' in request.content
            assert b"synthetic-smoke.png" in request.content
            return httpx.Response(200, json=document)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


@pytest.mark.parametrize(
    ("operation", "document_factory"),
    [
        (SmokeOperation.ANALYZE, _analysis_document),
        (SmokeOperation.RESTORE, _restoration_document),
        (SmokeOperation.RESTORE_AND_ANALYZE, _restoration_document),
    ],
)
def test_smoke_sequence_validates_each_operation_contract(
    operation: SmokeOperation,
    document_factory: Callable[[int, int], dict[str, object]],
) -> None:
    observed_paths: list[str] = []
    config = SmokeTestConfig(operation=operation, width=8, height=6)

    report = asyncio.run(
        run_smoke_test(
            config,
            transport=_transport(operation, document_factory(8, 6), observed_paths),
        )
    )

    assert observed_paths == [
        "/health/live",
        "/health/ready",
        "/health/model",
        f"/api/v1/{operation.value}",
    ]
    assert report.response.input_width == 8
    assert report.response.input_height == 6
    assert report.response.warning_count == 1
    assert report.checks[-2:] == ("multipart-upload", "contract")
    if operation is SmokeOperation.ANALYZE:
        assert report.response.suitability_recommendation == "warn"
        assert report.response.restored_media_type is None
    else:
        assert report.response.restored_media_type == "image/png"
        assert (report.response.restored_width, report.response.restored_height) == (16, 12)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"base_url": "http://user:secret@example.test"}, "embedded credentials"),
        ({"base_url": "file:///tmp/service"}, r"HTTP\(S\) URL"),
        ({"timeout_seconds": float("inf")}, "finite and positive"),
        ({"width": 0}, "width"),
        ({"height": 5000}, "height"),
    ],
)
def test_smoke_configuration_rejects_unsafe_or_unbounded_values(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SmokeTestConfig(**kwargs)  # type: ignore[arg-type]


def test_unready_smoke_failure_suppresses_server_details() -> None:
    unsafe = r"C:\models\private.pt secret-token"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health/live":
            return httpx.Response(200, json={"status": "alive"})
        return httpx.Response(
            503,
            json={"ready": False, "state": "unavailable", "unavailable_reason": unsafe},
        )

    with pytest.raises(SmokeTestUnavailableError) as captured:
        asyncio.run(run_smoke_test(SmokeTestConfig(), transport=httpx.MockTransport(handler)))

    assert unsafe not in str(captured.value)
    assert "private.pt" not in str(captured.value)


def test_operation_failure_suppresses_error_envelope_and_body() -> None:
    unsafe = "/srv/checkpoints/private.pt traceback secret-token"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health/live":
            return httpx.Response(200, json={"status": "alive"})
        if request.url.path == "/health/ready":
            return httpx.Response(200, json={"ready": True, "state": "ready"})
        if request.url.path == "/health/model":
            return httpx.Response(200, json={"ready": True, "state": "ready"})
        return httpx.Response(
            500,
            json={"error": {"code": "internal_error", "message": unsafe}},
        )

    with pytest.raises(SmokeTestError) as captured:
        asyncio.run(run_smoke_test(SmokeTestConfig(), transport=httpx.MockTransport(handler)))

    assert unsafe not in str(captured.value)
    assert "private.pt" not in str(captured.value)


def test_restoration_contract_rejects_invalid_or_mismatched_png() -> None:
    document = _restoration_document(8, 6)
    image = document["image"]
    assert isinstance(image, dict)
    image["content"] = base64.b64encode(b"not-png").decode("ascii")

    with pytest.raises(SmokeTestError, match="valid PNG"):
        validate_operation_payload(
            SmokeOperation.RESTORE,
            document,
            input_width=8,
            input_height=6,
        )


def test_smoke_report_contains_metadata_but_no_image_content() -> None:
    observed_paths: list[str] = []
    report = asyncio.run(
        run_smoke_test(
            SmokeTestConfig(operation=SmokeOperation.RESTORE, width=4, height=3),
            transport=_transport(
                SmokeOperation.RESTORE,
                _restoration_document(4, 3),
                observed_paths,
            ),
        )
    )

    payload = report_payload(report)
    serialized = str(payload)
    assert payload["schema_version"] == 1
    assert payload["operation"] == "restore"
    assert "content" not in serialized
    assert "synthetic-smoke.png" not in serialized
    assert "checkpoint" not in serialized


def test_validator_accepts_response_from_actual_fastapi_contract(
    fake_model_service_factory: type[object],
) -> None:
    app = create_app(model_service_factory=lambda _: fake_model_service_factory())  # type: ignore[operator]
    image = synthetic_grayscale_png(1, 1)

    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 200
        response = client.post(
            "/api/v1/restore-and-analyze",
            files={"image": ("synthetic-smoke.png", image, "image/png")},
        )

    assert response.status_code == 200
    summary = validate_operation_payload(
        SmokeOperation.RESTORE_AND_ANALYZE,
        response.json(),
        input_width=1,
        input_height=1,
    )
    assert summary.restored_media_type == "image/png"
    assert summary.diagnostic_sections == ("source_format", "synthetic")


@pytest.mark.local_checkpoint
def test_optional_real_checkpoint_smoke() -> None:
    if not DEFAULT_CHECKPOINT_PATH.is_file():
        pytest.skip("verified ignored runtime checkpoint is unavailable")

    app = create_app(
        settings=RuntimeSettings(
            checkpoint_path=DEFAULT_CHECKPOINT_PATH,
            device_preference="cpu",
        )
    )
    image = synthetic_grayscale_png(32, 32)
    with TestClient(app) as client:
        ready = client.get("/health/ready")
        assert ready.status_code == 200, "verified checkpoint did not make the adapter ready"
        response = client.post(
            "/api/v1/restore-and-analyze",
            files={"image": ("synthetic-smoke.png", image, "image/png")},
        )

    assert response.status_code == 200
    validate_operation_payload(
        SmokeOperation.RESTORE_AND_ANALYZE,
        response.json(),
        input_width=32,
        input_height=32,
    )
