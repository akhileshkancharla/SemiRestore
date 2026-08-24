"""Prometheus-compatible metrics exposition endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from semirestore.api.application import ApplicationRuntime
from semirestore.api.dependencies import get_runtime

router = APIRouter(tags=["operations"])

RuntimeDependency = Annotated[ApplicationRuntime, Depends(get_runtime)]


@router.get("/metrics", include_in_schema=False)
def metrics(runtime: RuntimeDependency) -> Response:
    """Expose only this application's isolated SemiRestore registry."""
    return Response(
        content=generate_latest(runtime.metrics.registry),
        media_type=CONTENT_TYPE_LATEST,
        headers={"Cache-Control": "no-store"},
    )
