"""Thread-safe process-local lifecycle for the verified SemiRestore model."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from torch import nn

from .checkpoints import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_METADATA_PATH,
    PROJECT_ROOT,
    CheckpointCompatibilityError,
    CheckpointError,
    CheckpointMetadataError,
    CheckpointStructureError,
    CheckpointVerificationError,
    DeviceSelectionError,
    LoadedCheckpoint,
    load_conditioned_checkpoint,
)

DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "model" / "semirestore_conditioned.pt"


class ModelManagerState(StrEnum):
    """Explicit states for one process-local model lifecycle."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    CLOSED = "closed"


class ModelManagerError(RuntimeError):
    """Base class for lifecycle failures."""


class ModelNotReadyError(ModelManagerError):
    """Raised when model access is attempted outside the ready state."""


class ModelManagerLoadError(ModelManagerError):
    """Safe categorized loading failure with no checkpoint exception details."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(f"Model loading failed ({category})")


class ModelManagerClosedError(ModelManagerError):
    """Raised when an operation is forbidden after permanent closure."""


class ModelManagerStateError(ModelManagerError):
    """Raised when a lifecycle transition is invalid."""


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Tensor-free identity retained independently of the loaded model reference."""

    model_name: str
    architecture: str
    model_version: str | None
    training_revision: str | None
    resolved_device: str
    parameter_count: int
    checkpoint_path: str
    checkpoint_sha256: str
    scale_factor: int


@dataclass(frozen=True, slots=True)
class ModelManagerStatus:
    """Serialization-friendly snapshot of lifecycle and verified model identity."""

    state: ModelManagerState
    ready: bool
    model_name: str | None
    architecture: str | None
    model_version: str | None
    training_revision: str | None
    resolved_device: str | None
    parameter_count: int | None
    checkpoint_path: str
    checkpoint_sha256: str | None
    scale_factor: int | None
    last_loading_error_category: str | None
    retry_permitted: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible status mapping without model objects."""

        return {
            "state": self.state.value,
            "ready": self.ready,
            "model_name": self.model_name,
            "architecture": self.architecture,
            "model_version": self.model_version,
            "training_revision": self.training_revision,
            "resolved_device": self.resolved_device,
            "parameter_count": self.parameter_count,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "scale_factor": self.scale_factor,
            "last_loading_error_category": self.last_loading_error_category,
            "retry_permitted": self.retry_permitted,
        }


CheckpointLoader = Callable[..., LoadedCheckpoint]


def _public_checkpoint_path(path: Path) -> str:
    """Return a stable runtime identifier without exposing external absolute paths."""

    try:
        return path.resolve(strict=False).relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _failure_category(error: Exception) -> str:
    if isinstance(error, CheckpointMetadataError):
        return "checkpoint_metadata"
    if isinstance(error, CheckpointVerificationError):
        return "checkpoint_verification"
    if isinstance(error, CheckpointStructureError):
        return "checkpoint_structure"
    if isinstance(error, CheckpointCompatibilityError):
        return "checkpoint_compatibility"
    if isinstance(error, DeviceSelectionError):
        return "device_selection"
    if isinstance(error, CheckpointError):
        return "checkpoint_loading"
    return "unexpected_loading_error"


def _prepare_loaded_checkpoint(loaded: LoadedCheckpoint) -> tuple[nn.Module, ModelIdentity]:
    if not isinstance(loaded, LoadedCheckpoint) or not isinstance(loaded.model, nn.Module):
        raise TypeError("Loader returned an invalid checkpoint result")
    model = loaded.model
    model.eval()
    model.requires_grad_(False)
    actual_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if actual_parameter_count != loaded.parameter_count:
        raise ValueError("Loader returned inconsistent parameter metadata")
    scale_factor = getattr(model, "scale", None)
    if type(scale_factor) is not int or scale_factor < 1:
        raise ValueError("Loaded model has no valid integer scale factor")
    identity = ModelIdentity(
        model_name=loaded.model_name,
        architecture=loaded.architecture,
        model_version=loaded.model_version,
        training_revision=loaded.training_revision,
        resolved_device=str(loaded.device),
        parameter_count=loaded.parameter_count,
        checkpoint_path=_public_checkpoint_path(loaded.checkpoint_path),
        checkpoint_sha256=loaded.checkpoint_sha256,
        scale_factor=scale_factor,
    )
    return model, identity


class ModelManager:
    """Own one verified model instance for the lifetime of an application process."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        metadata_path: str | Path = DEFAULT_METADATA_PATH,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        device: str = "auto",
        loader: CheckpointLoader = load_conditioned_checkpoint,
    ) -> None:
        self._checkpoint_path = Path(checkpoint_path)
        self._metadata_path = Path(metadata_path)
        self._config_path = Path(config_path)
        self._device = device
        self._loader = loader
        self._condition = threading.Condition()
        self._state = ModelManagerState.UNLOADED
        self._model: nn.Module | None = None
        self._identity: ModelIdentity | None = None
        self._last_loading_error_category: str | None = None

    @property
    def state(self) -> ModelManagerState:
        """Return the current lifecycle state without triggering loading."""

        with self._condition:
            return self._state

    @property
    def is_ready(self) -> bool:
        """Report readiness without triggering loading."""

        with self._condition:
            return self._state is ModelManagerState.READY

    @property
    def model(self) -> nn.Module:
        """Return the persistent model only while the manager is ready."""

        with self._condition:
            if self._state is ModelManagerState.CLOSED:
                raise ModelManagerClosedError("Model manager is permanently closed")
            if self._state is not ModelManagerState.READY or self._model is None:
                raise ModelNotReadyError(
                    f"Model is unavailable while manager state is {self._state.value!r}"
                )
            return self._model

    def load(self) -> nn.Module:
        """Load once, or reuse the same ready instance for every successful caller."""

        with self._condition:
            while self._state is ModelManagerState.LOADING:
                self._condition.wait()
            if self._state is ModelManagerState.READY:
                if self._model is None:
                    raise ModelManagerStateError("Ready manager has no model")
                return self._model
            if self._state is ModelManagerState.CLOSED:
                raise ModelManagerClosedError("Model manager is permanently closed")
            if self._state is ModelManagerState.FAILED:
                category = self._last_loading_error_category or "checkpoint_loading"
                raise ModelManagerLoadError(category)
            self._state = ModelManagerState.LOADING

        try:
            loaded = self._loader(
                checkpoint_path=self._checkpoint_path,
                metadata_path=self._metadata_path,
                config_path=self._config_path,
                device=self._device,
            )
            model, identity = _prepare_loaded_checkpoint(loaded)
        except Exception as error:
            category = _failure_category(error)
            with self._condition:
                if self._state is ModelManagerState.CLOSED:
                    self._condition.notify_all()
                    raise ModelManagerClosedError(
                        "Model manager was closed while loading"
                    ) from None
                self._state = ModelManagerState.FAILED
                self._last_loading_error_category = category
                self._condition.notify_all()
            raise ModelManagerLoadError(category) from None

        with self._condition:
            if self._state is ModelManagerState.CLOSED:
                self._condition.notify_all()
                raise ModelManagerClosedError("Model manager was closed while loading")
            self._model = model
            self._identity = identity
            self._last_loading_error_category = None
            self._state = ModelManagerState.READY
            self._condition.notify_all()
            return model

    def reset_failure(self) -> None:
        """Permit an explicit retry by returning a failed manager to unloaded."""

        with self._condition:
            if self._state is ModelManagerState.CLOSED:
                raise ModelManagerClosedError("Model manager is permanently closed")
            if self._state is ModelManagerState.LOADING:
                raise ModelManagerStateError("Cannot reset while model loading is active")
            if self._state is ModelManagerState.READY:
                raise ModelManagerStateError("Cannot reset a ready model manager")
            if self._state is ModelManagerState.FAILED:
                self._state = ModelManagerState.UNLOADED
                self._last_loading_error_category = None

    def close(self) -> None:
        """Permanently close the manager and release its model reference."""

        with self._condition:
            self._model = None
            self._state = ModelManagerState.CLOSED
            self._last_loading_error_category = None
            self._condition.notify_all()

    def status(self) -> ModelManagerStatus:
        """Return an immutable, tensor-free snapshot without triggering loading."""

        with self._condition:
            identity = self._identity
            state = self._state
            return ModelManagerStatus(
                state=state,
                ready=state is ModelManagerState.READY,
                model_name=None if identity is None else identity.model_name,
                architecture=None if identity is None else identity.architecture,
                model_version=None if identity is None else identity.model_version,
                training_revision=None if identity is None else identity.training_revision,
                resolved_device=None if identity is None else identity.resolved_device,
                parameter_count=None if identity is None else identity.parameter_count,
                checkpoint_path=(
                    _public_checkpoint_path(self._checkpoint_path)
                    if identity is None
                    else identity.checkpoint_path
                ),
                checkpoint_sha256=None if identity is None else identity.checkpoint_sha256,
                scale_factor=None if identity is None else identity.scale_factor,
                last_loading_error_category=self._last_loading_error_category,
                retry_permitted=state is ModelManagerState.FAILED,
            )


__all__ = [
    "DEFAULT_CHECKPOINT_PATH",
    "ModelIdentity",
    "ModelManager",
    "ModelManagerClosedError",
    "ModelManagerError",
    "ModelManagerLoadError",
    "ModelManagerState",
    "ModelManagerStateError",
    "ModelManagerStatus",
    "ModelNotReadyError",
]
