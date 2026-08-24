from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends
from fastapi.testclient import TestClient

from semirestore.api import create_app
from semirestore.api.dependencies import get_model_service
from semirestore.platform import ModelService, ModelServiceState, RuntimeSettings


def test_lifespan_initializes_reuses_and_shuts_down_supplied_service(
    fake_model_service: Any,
) -> None:
    factory_calls = 0

    def factory(_: RuntimeSettings) -> ModelService:
        nonlocal factory_calls
        factory_calls += 1
        return fake_model_service

    app = create_app(model_service_factory=factory)

    @app.get("/_test/service-id")
    def service_id(
        service: Annotated[ModelService, Depends(get_model_service)],
    ) -> dict[str, int]:
        return {"id": id(service)}

    with TestClient(app) as client:
        first = client.get("/_test/service-id")
        second = client.get("/_test/service-id")

        assert factory_calls == 1
        assert fake_model_service.startup_calls == 1
        assert first.json()["id"] == id(fake_model_service)
        assert second.json()["id"] == id(fake_model_service)

    assert fake_model_service.shutdown_calls == 1


def test_lifespan_without_factory_stays_explicitly_unconfigured() -> None:
    app = create_app()

    with TestClient(app):
        health = app.state.runtime.model_health()

        assert health.state is ModelServiceState.UNAVAILABLE
        assert health.ready is False
        assert health.unavailable_reason == "model service adapter is not configured"


def test_startup_failure_is_contained_and_partial_service_is_closed(
    fake_model_service_factory: Any,
) -> None:
    service = fake_model_service_factory(startup_error=RuntimeError("sensitive startup detail"))
    app = create_app(model_service_factory=lambda _: service)

    with TestClient(app):
        health = app.state.runtime.model_health()

        assert health.ready is False
        assert health.unavailable_reason == "model service failed to initialize"
        assert "sensitive" not in health.unavailable_reason

    assert service.startup_calls == 1
    assert service.shutdown_calls == 1


def test_shutdown_failure_does_not_prevent_application_shutdown(
    fake_model_service_factory: Any,
) -> None:
    service = fake_model_service_factory(shutdown_error=RuntimeError("shutdown failed"))
    app = create_app(model_service_factory=lambda _: service)

    with TestClient(app):
        assert app.state.runtime.model_service is service

    assert service.shutdown_calls == 1
    assert app.state.runtime.model_service is None
