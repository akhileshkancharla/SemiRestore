"""FastAPI dependencies for process-local platform state."""

from __future__ import annotations

from fastapi import Request

from semirestore.api.application import ApplicationRuntime
from semirestore.api.concurrency import InferenceGate
from semirestore.platform import ModelService, ModelServiceUnavailableError, RuntimeSettings


def get_runtime(request: Request) -> ApplicationRuntime:
    """Return the runtime state belonging to the current application."""
    return request.app.state.runtime


def get_settings(request: Request) -> RuntimeSettings:
    """Return the configuration used by the current application."""
    return get_runtime(request).settings


def get_model_service(request: Request) -> ModelService:
    """Return the single startup-initialized model service."""
    return require_model_service(get_runtime(request))


def require_model_service(runtime: ApplicationRuntime) -> ModelService:
    """Resolve the lifespan-owned model service from known runtime state."""
    service = runtime.model_service
    if service is None:
        raise ModelServiceUnavailableError("model service is unavailable")
    return service


def require_inference_gate(runtime: ApplicationRuntime) -> InferenceGate:
    """Resolve the lifespan-owned inference controller."""
    gate = runtime.inference_gate
    if gate is None:
        raise RuntimeError("inference controller is unavailable")
    return gate
