"""FastAPI dependencies for process-local platform state."""

from __future__ import annotations

from fastapi import Request

from semirestore.api.application import ApplicationRuntime
from semirestore.platform import ModelService, ModelServiceUnavailableError, RuntimeSettings


def get_runtime(request: Request) -> ApplicationRuntime:
    """Return the runtime state belonging to the current application."""
    return request.app.state.runtime


def get_settings(request: Request) -> RuntimeSettings:
    """Return the configuration used by the current application."""
    return get_runtime(request).settings


def get_model_service(request: Request) -> ModelService:
    """Return the single startup-initialized model service."""
    service = get_runtime(request).model_service
    if service is None:
        raise ModelServiceUnavailableError("model service is unavailable")
    return service
