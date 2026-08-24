from __future__ import annotations

import pytest

from semirestore.platform import (
    ModelHealth,
    ModelService,
    ModelServiceError,
    ModelServiceInferenceError,
    ModelServiceInitializationError,
    ModelServiceState,
    ModelServiceUnavailableError,
)


def test_explicit_fake_satisfies_runtime_protocol(fake_model_service: ModelService) -> None:
    assert isinstance(fake_model_service, ModelService)


def test_ready_health_contains_only_boundary_metadata() -> None:
    health = ModelHealth(
        state=ModelServiceState.READY,
        ready=True,
        device="cpu",
        model_version="model-v1",
        checkpoint_checksum="sha256:test",
    )

    assert health.ready is True
    assert health.unavailable_reason is None


def test_unavailable_health_requires_safe_reason() -> None:
    health = ModelHealth(
        state=ModelServiceState.UNAVAILABLE,
        ready=False,
        unavailable_reason="checkpoint is not installed",
    )

    assert health.ready is False
    assert health.unavailable_reason == "checkpoint is not installed"


@pytest.mark.parametrize(
    "health",
    [
        ModelHealth,
    ],
)
def test_health_rejects_inconsistent_readiness(health: type[ModelHealth]) -> None:
    with pytest.raises(ValueError, match="ready must agree"):
        health(state=ModelServiceState.READY, ready=False, unavailable_reason="not ready")

    with pytest.raises(ValueError, match="cannot have an unavailable reason"):
        health(
            state=ModelServiceState.READY,
            ready=True,
            unavailable_reason="must not be present",
        )

    with pytest.raises(ValueError, match="must have an unavailable reason"):
        health(state=ModelServiceState.STARTING, ready=False)


@pytest.mark.parametrize(
    "exception_type",
    [
        ModelServiceInitializationError,
        ModelServiceUnavailableError,
        ModelServiceInferenceError,
    ],
)
def test_service_exceptions_share_platform_base(
    exception_type: type[ModelServiceError],
) -> None:
    assert issubclass(exception_type, ModelServiceError)
