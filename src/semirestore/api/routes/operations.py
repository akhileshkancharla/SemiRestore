"""Operational health and version endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Response, status

from semirestore import __version__
from semirestore.api.application import ApplicationRuntime
from semirestore.api.dependencies import get_runtime
from semirestore.api.schemas import (
    LiveResponse,
    ModelHealthResponse,
    ReadyResponse,
    VersionResponse,
)

router = APIRouter(tags=["operations"])

RuntimeDependency = Annotated[ApplicationRuntime, Depends(get_runtime)]


@router.get("/health/live", response_model=LiveResponse)
def live() -> LiveResponse:
    """Report API process liveness without consulting the model service."""
    return LiveResponse()


@router.get(
    "/health/ready",
    response_model=ReadyResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadyResponse}},
)
def ready(runtime: RuntimeDependency, response: Response) -> ReadyResponse:
    """Report whether this process can currently accept restoration work."""
    health = runtime.model_health()
    if not health.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyResponse(
        ready=health.ready,
        state=health.state,
        unavailable_reason=health.unavailable_reason,
    )


@router.get("/health/model", response_model=ModelHealthResponse)
def model_health(runtime: RuntimeDependency) -> ModelHealthResponse:
    """Return safe model-service readiness and provenance metadata."""
    health = runtime.model_health()
    return ModelHealthResponse(
        state=health.state,
        ready=health.ready,
        device=health.device,
        model_version=health.model_version,
        checkpoint_checksum=health.checkpoint_checksum,
        unavailable_reason=health.unavailable_reason,
    )


@router.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    """Return stable package version metadata."""
    return VersionResponse(version=__version__)
