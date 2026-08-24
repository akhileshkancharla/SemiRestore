"""Operational health and version endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from semirestore import __version__
from semirestore.api.application import ApplicationRuntime
from semirestore.api.dependencies import get_runtime
from semirestore.platform import ModelHealth

router = APIRouter(tags=["operations"])

RuntimeDependency = Annotated[ApplicationRuntime, Depends(get_runtime)]


def _model_health_content(health: ModelHealth) -> dict[str, str | bool | None]:
    """Serialize only model metadata approved by the platform protocol."""
    return {
        "state": health.state.value,
        "ready": health.ready,
        "device": health.device,
        "model_version": health.model_version,
        "checkpoint_checksum": health.checkpoint_checksum,
        "unavailable_reason": health.unavailable_reason,
    }


@router.get("/health/live")
def live() -> dict[str, str]:
    """Report API process liveness without consulting the model service."""
    return {"status": "alive"}


@router.get("/health/ready")
def ready(runtime: RuntimeDependency) -> JSONResponse:
    """Report whether this process can currently accept restoration work."""
    health = runtime.model_health()
    response_status = status.HTTP_200_OK if health.ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(
        status_code=response_status,
        content={
            "ready": health.ready,
            "state": health.state.value,
            "unavailable_reason": health.unavailable_reason,
        },
    )


@router.get("/health/model")
def model_health(runtime: RuntimeDependency) -> dict[str, str | bool | None]:
    """Return safe model-service readiness and provenance metadata."""
    return _model_health_content(runtime.model_health())


@router.get("/version")
def version() -> dict[str, str]:
    """Return stable package version metadata."""
    return {"application": "semirestore", "version": __version__}
