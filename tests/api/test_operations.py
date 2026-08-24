from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from semirestore import __version__
from semirestore.api import create_app
from semirestore.platform import ModelHealth, ModelServiceState


def test_ready_service_exposes_operational_health_and_version(
    fake_model_service: Any,
) -> None:
    app = create_app(model_service_factory=lambda _: fake_model_service)

    with TestClient(app) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")
        model = client.get("/health/model")
        version = client.get("/version")

    assert live.status_code == 200
    assert live.json() == {"status": "alive"}
    assert ready.status_code == 200
    assert ready.json() == {
        "ready": True,
        "state": "ready",
        "unavailable_reason": None,
    }
    assert model.status_code == 200
    assert model.json() == {
        "state": "ready",
        "ready": True,
        "device": "test-device",
        "model_version": "synthetic-test-model",
        "checkpoint_checksum": "test-checksum",
        "unavailable_reason": None,
    }
    assert version.status_code == 200
    assert version.json() == {"application": "semirestore", "version": __version__}


def test_unready_model_returns_non_success_readiness(
    fake_model_service_factory: Any,
) -> None:
    service = fake_model_service_factory(
        health=ModelHealth(
            state=ModelServiceState.UNAVAILABLE,
            ready=False,
            device="cpu",
            model_version="model-v1",
            unavailable_reason="checkpoint is not installed",
        )
    )
    app = create_app(model_service_factory=lambda _: service)

    with TestClient(app) as client:
        ready = client.get("/health/ready")
        model = client.get("/health/model")

    assert ready.status_code == 503
    assert ready.json() == {
        "ready": False,
        "state": "unavailable",
        "unavailable_reason": "checkpoint is not installed",
    }
    assert model.status_code == 200
    assert model.json()["ready"] is False
    assert model.json()["unavailable_reason"] == "checkpoint is not installed"


def test_liveness_survives_model_startup_failure_without_leaking_error(
    fake_model_service_factory: Any,
) -> None:
    sensitive_error = RuntimeError("C:/secret/checkpoint.pt failed with token=secret")
    service = fake_model_service_factory(startup_error=sensitive_error)
    app = create_app(model_service_factory=lambda _: service)

    with TestClient(app) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")
        model = client.get("/health/model")

    assert live.status_code == 200
    assert live.json() == {"status": "alive"}
    assert ready.status_code == 503
    assert model.status_code == 200
    assert model.json() == {
        "state": "unavailable",
        "ready": False,
        "device": None,
        "model_version": None,
        "checkpoint_checksum": None,
        "unavailable_reason": "model service failed to initialize",
    }
    assert "checkpoint.pt" not in model.text
    assert "token" not in model.text


def test_liveness_does_not_call_model_health(fake_model_service: Any) -> None:
    def fail_health() -> ModelHealth:
        raise AssertionError("liveness consulted model health")

    fake_model_service.health = fail_health
    app = create_app(model_service_factory=lambda _: fake_model_service)

    with TestClient(app) as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_unconfigured_application_is_live_but_not_ready() -> None:
    app = create_app()

    with TestClient(app) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")

    assert live.status_code == 200
    assert ready.status_code == 503
    assert ready.json()["unavailable_reason"] == "model service adapter is not configured"
