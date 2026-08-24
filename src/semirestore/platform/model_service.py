"""Platform-owned boundary for the future SemiRestore model service."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from pydantic import JsonValue

if TYPE_CHECKING:
    from semirestore.api.uploads import ValidatedUpload

_SUPPORTED_OUTPUT_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/tiff"})
_MAX_DIAGNOSTICS_BYTES = 65_536
_MAX_IDENTITY_LENGTH = 256
_MAX_WARNING_LENGTH = 512


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


def _validate_safe_text(value: str | None, field_name: str) -> None:
    if value is None:
        return
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > _MAX_IDENTITY_LENGTH
        or not value.isprintable()
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"{field_name} must be a safe public identifier")


@dataclass(frozen=True, slots=True)
class RestorationResult:
    """Serializable, model-independent result returned to the API platform."""

    restored_image_bytes: bytes
    restored_media_type: str
    restored_width: int
    restored_height: int
    original_width: int
    original_height: int
    inference_latency_ms: float | None = None
    device: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    training_revision: str | None = None
    checkpoint_checksum: str | None = None
    phase_latency_ms: Mapping[str, float] = field(default_factory=dict)
    diagnostics: Mapping[str, JsonValue] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.restored_image_bytes, bytes) or not self.restored_image_bytes:
            raise ValueError("restored image bytes must be non-empty bytes")
        if self.restored_media_type not in _SUPPORTED_OUTPUT_MEDIA_TYPES:
            raise ValueError("restored media type is unsupported")
        for field_name in (
            "restored_width",
            "restored_height",
            "original_width",
            "original_height",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.inference_latency_ms is not None:
            latency = self.inference_latency_ms
            if (
                not isinstance(latency, (int, float))
                or isinstance(latency, bool)
                or not math.isfinite(latency)
                or latency < 0
            ):
                raise ValueError("inference latency must be finite and non-negative")
            object.__setattr__(self, "inference_latency_ms", float(latency))

        _validate_safe_text(self.device, "device")
        _validate_safe_text(self.model_name, "model name")
        _validate_safe_text(self.model_version, "model version")
        _validate_safe_text(self.training_revision, "training revision")
        _validate_safe_text(self.checkpoint_checksum, "checkpoint checksum")

        phases: dict[str, float] = {}
        if not isinstance(self.phase_latency_ms, Mapping):
            raise ValueError("phase latency must be a mapping")
        for name, latency in self.phase_latency_ms.items():
            if (
                not isinstance(name, str)
                or not name
                or len(name) > 64
                or not name.isidentifier()
                or not isinstance(latency, (int, float))
                or isinstance(latency, bool)
                or not math.isfinite(latency)
                or latency < 0
            ):
                raise ValueError("phase latency must contain safe names and finite values")
            phases[name] = float(latency)
        object.__setattr__(self, "phase_latency_ms", MappingProxyType(phases))

        if not isinstance(self.diagnostics, Mapping):
            raise ValueError("diagnostics must be a JSON-compatible mapping")
        try:
            serialized_diagnostics = json.dumps(
                dict(self.diagnostics),
                allow_nan=False,
                separators=(",", ":"),
            )
            decoded_diagnostics = json.loads(serialized_diagnostics)
        except (TypeError, ValueError) as error:
            raise ValueError("diagnostics must contain only JSON-compatible values") from error
        if len(serialized_diagnostics.encode("utf-8")) > _MAX_DIAGNOSTICS_BYTES:
            raise ValueError("diagnostics exceed the platform result limit")
        object.__setattr__(self, "diagnostics", MappingProxyType(decoded_diagnostics))

        if not isinstance(self.warnings, tuple):
            raise ValueError("warnings must be an immutable tuple")
        for warning in self.warnings:
            if (
                not isinstance(warning, str)
                or not warning
                or warning != warning.strip()
                or len(warning) > _MAX_WARNING_LENGTH
                or not warning.isprintable()
                or "/" in warning
                or "\\" in warning
            ):
                raise ValueError("warnings must contain only safe public text")


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Serializable model diagnostic result without image or tensor objects."""

    original_width: int
    original_height: int
    diagnostics: Mapping[str, JsonValue]
    suitability_recommendation: Literal["restore", "warn", "bypass"]
    suitability_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    analysis_latency_ms: float

    def __post_init__(self) -> None:
        for field_name in ("original_width", "original_height"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if (
            not isinstance(self.analysis_latency_ms, (int, float))
            or isinstance(self.analysis_latency_ms, bool)
            or not math.isfinite(self.analysis_latency_ms)
            or self.analysis_latency_ms < 0
        ):
            raise ValueError("analysis latency must be finite and non-negative")
        object.__setattr__(self, "analysis_latency_ms", float(self.analysis_latency_ms))
        try:
            serialized = json.dumps(dict(self.diagnostics), allow_nan=False, separators=(",", ":"))
            decoded = json.loads(serialized)
        except (TypeError, ValueError) as error:
            raise ValueError("diagnostics must contain only JSON-compatible values") from error
        if len(serialized.encode("utf-8")) > _MAX_DIAGNOSTICS_BYTES:
            raise ValueError("diagnostics exceed the platform result limit")
        object.__setattr__(self, "diagnostics", MappingProxyType(decoded))
        for values, field_name in (
            (self.suitability_reasons, "suitability reasons"),
            (self.warnings, "warnings"),
        ):
            if not isinstance(values, tuple):
                raise ValueError(f"{field_name} must be an immutable tuple")
            for value in values:
                if (
                    not isinstance(value, str)
                    or not value
                    or value != value.strip()
                    or len(value) > _MAX_WARNING_LENGTH
                    or not value.isprintable()
                    or "/" in value
                    or "\\" in value
                ):
                    raise ValueError(f"{field_name} must contain only safe public text")


@runtime_checkable
class ModelService(Protocol):
    """Lifecycle, health, and restoration surface required by the API platform."""

    async def startup(self) -> None:
        """Initialize long-lived model resources once."""

    async def shutdown(self) -> None:
        """Release resources initialized at startup."""

    def health(self) -> ModelHealth:
        """Return current safe readiness and model metadata."""

    async def analyze(self, upload: ValidatedUpload) -> AnalysisResult:
        """Analyze one transport-validated image with model-owned diagnostics."""

    async def restore(self, upload: ValidatedUpload) -> RestorationResult:
        """Restore one transport-validated image using long-lived resources."""

    async def restore_and_analyze(self, upload: ValidatedUpload) -> RestorationResult:
        """Restore and diagnose one image using the retained pipeline."""


class ModelServiceError(RuntimeError):
    """Base class for failures crossing the model-service boundary."""


class ModelServiceInitializationError(ModelServiceError):
    """The model service could not initialize its long-lived resources."""


class ModelServiceUnavailableError(ModelServiceError):
    """The model service cannot currently accept restoration work."""


class ModelServiceInferenceError(ModelServiceError):
    """A restoration operation failed inside the model-service boundary."""
