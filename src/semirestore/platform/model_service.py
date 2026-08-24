"""Platform-owned boundary for the future SemiRestore model service."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


class ModelServiceState(StrEnum):
    """Lifecycle states that are safe to expose through platform health APIs."""

    STARTING = "starting"
    READY = "ready"
    UNAVAILABLE = "unavailable"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class ModelHealth:
    """Safe model metadata returned across the platform boundary."""

    state: ModelServiceState
    ready: bool
    device: str | None = None
    model_version: str | None = None
    checkpoint_checksum: str | None = None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.ready != (self.state is ModelServiceState.READY):
            raise ValueError("ready must agree with the model service state")
        if self.ready and self.unavailable_reason is not None:
            raise ValueError("a ready model service cannot have an unavailable reason")
        if not self.ready and not self.unavailable_reason:
            raise ValueError("an unready model service must have an unavailable reason")


@runtime_checkable
class ModelService(Protocol):
    """Lifecycle and health surface required by the API platform.

    Restoration is intentionally absent until its cross-track result contract
    is available. Implementations own model and checkpoint behavior.
    """

    async def startup(self) -> None:
        """Initialize long-lived model resources once."""

    async def shutdown(self) -> None:
        """Release resources initialized at startup."""

    def health(self) -> ModelHealth:
        """Return current safe readiness and model metadata."""


class ModelServiceError(RuntimeError):
    """Base class for failures crossing the model-service boundary."""


class ModelServiceInitializationError(ModelServiceError):
    """The model service could not initialize its long-lived resources."""


class ModelServiceUnavailableError(ModelServiceError):
    """The model service cannot currently accept restoration work."""


class ModelServiceInferenceError(ModelServiceError):
    """A restoration operation failed inside the model-service boundary."""
