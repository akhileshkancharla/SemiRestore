"""Production adapter for the model-owned SemiRestore pipeline."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from threading import RLock
from typing import TYPE_CHECKING, Any, cast

from semirestore.intensity_diagnostics import IntensityDiagnosticError, analyze_intensity
from semirestore.model_manager import ModelManagerClosedError, ModelNotReadyError
from semirestore.pipeline import PipelineError, SemiRestorePipeline
from semirestore.platform.model_service import (
    AnalysisResult,
    ModelHealth,
    ModelServiceInferenceError,
    ModelServiceInitializationError,
    ModelServiceState,
    ModelServiceUnavailableError,
    RestorationResult,
)
from semirestore.platform.settings import RuntimeSettings
from semirestore.preprocessing import PreprocessingError, preprocess_sem_image
from semirestore.restoration_service import ManagerNotReadyError, RestorationServiceError
from semirestore.structural_diagnostics import StructuralDiagnosticError, analyze_structure

if TYPE_CHECKING:
    from semirestore.api.uploads import ValidatedUpload

PipelineFactory = Callable[..., SemiRestorePipeline]

_KNOWN_INFERENCE_ERRORS = (
    IntensityDiagnosticError,
    PipelineError,
    PreprocessingError,
    RestorationServiceError,
    StructuralDiagnosticError,
    TypeError,
    ValueError,
)
_UNAVAILABLE_ERRORS = (
    ManagerNotReadyError,
    ModelManagerClosedError,
    ModelNotReadyError,
)


class SemiRestoreModelService:
    """Adapt one long-lived ``SemiRestorePipeline`` to the platform protocol."""

    def __init__(
        self,
        settings: RuntimeSettings,
        *,
        pipeline_factory: PipelineFactory = SemiRestorePipeline.from_config,
    ) -> None:
        self._settings = settings
        self._pipeline_factory = pipeline_factory
        self._pipeline: SemiRestorePipeline | None = None
        self._state = ModelServiceState.STARTING
        self._unavailable_reason = "model service startup is in progress"
        self._lock = RLock()

    def _factory_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {"device": self._settings.device_preference}
        if self._settings.checkpoint_path is not None:
            kwargs["checkpoint_path"] = self._settings.checkpoint_path
        if self._settings.model_metadata_path is not None:
            kwargs["metadata_path"] = self._settings.model_metadata_path
        if self._settings.model_config_path is not None:
            kwargs["model_config_path"] = self._settings.model_config_path
        return kwargs

    async def startup(self) -> None:
        """Verify and load one real pipeline outside the event-loop thread."""
        with self._lock:
            if self._pipeline is not None:
                return
            if self._state is ModelServiceState.STOPPED:
                raise ModelServiceInitializationError(
                    "model service cannot restart after shutdown"
                )
            self._state = ModelServiceState.STARTING
            self._unavailable_reason = "model service startup is in progress"

        pipeline: SemiRestorePipeline | None = None
        try:
            pipeline = await asyncio.to_thread(
                self._pipeline_factory,
                **self._factory_kwargs(),
            )
            status = pipeline.status()
            if not status.ready:
                raise ModelServiceInitializationError("model pipeline is not ready")
        except asyncio.CancelledError:
            with self._lock:
                self._state = ModelServiceState.UNAVAILABLE
                self._unavailable_reason = "model service initialization was cancelled"
            raise
        except Exception:
            if pipeline is not None:
                try:
                    await asyncio.to_thread(pipeline.close)
                except Exception:
                    pass
            with self._lock:
                self._state = ModelServiceState.UNAVAILABLE
                self._unavailable_reason = "model service failed to initialize"
            raise ModelServiceInitializationError(
                "model service failed to initialize"
            ) from None

        with self._lock:
            self._pipeline = pipeline
            self._state = ModelServiceState.READY
            self._unavailable_reason = ""

    async def shutdown(self) -> None:
        """Detach and close the retained pipeline once."""
        with self._lock:
            pipeline = self._pipeline
            self._pipeline = None
            self._state = ModelServiceState.STOPPED
            self._unavailable_reason = "model service has stopped"
        if pipeline is not None:
            await asyncio.to_thread(pipeline.close)

    def health(self) -> ModelHealth:
        """Map cached manager status without loading or running inference."""
        with self._lock:
            pipeline = self._pipeline
            state = self._state
            unavailable_reason = self._unavailable_reason
        if pipeline is None:
            return ModelHealth(
                state=state,
                ready=False,
                unavailable_reason=unavailable_reason,
            )
        try:
            status = pipeline.status()
        except Exception:
            return ModelHealth(
                state=ModelServiceState.UNAVAILABLE,
                ready=False,
                unavailable_reason="model health check failed",
            )
        if not status.ready:
            return ModelHealth(
                state=ModelServiceState.UNAVAILABLE,
                ready=False,
                unavailable_reason="model pipeline is unavailable",
            )
        return ModelHealth(
            state=ModelServiceState.READY,
            ready=True,
            device=status.resolved_device,
            model_version=status.model_version,
            checkpoint_checksum=status.checkpoint_sha256,
        )

    def _ready_pipeline(self) -> SemiRestorePipeline:
        with self._lock:
            pipeline = self._pipeline
            state = self._state
        if pipeline is None or state is not ModelServiceState.READY:
            raise ModelServiceUnavailableError("model service is unavailable")
        return pipeline

    @staticmethod
    def _analyze_sync(
        pipeline: SemiRestorePipeline,
        upload: ValidatedUpload,
    ) -> AnalysisResult:
        started = time.perf_counter()
        preprocessed = preprocess_sem_image(
            upload.encoded_bytes,
            limits=pipeline.preprocessing_limits,
        )
        canonical = preprocessed.tensor[0, 0]
        intensity = analyze_intensity(canonical)
        structure = analyze_structure(canonical)
        warnings = tuple(dict.fromkeys((*intensity.warnings, *structure.warnings)))
        return AnalysisResult(
            original_width=preprocessed.original_width,
            original_height=preprocessed.original_height,
            diagnostics={
                "preprocessing": preprocessed.metadata(),
                "intensity": intensity.to_dict(),
                "structure": structure.to_dict(),
            },
            suitability_recommendation=structure.recommendation,
            suitability_reasons=structure.reasons,
            warnings=warnings,
            analysis_latency_ms=max(0.0, (time.perf_counter() - started) * 1000.0),
        )

    async def analyze(self, upload: ValidatedUpload) -> AnalysisResult:
        """Run public model-owned diagnostics outside the event-loop thread."""
        pipeline = self._ready_pipeline()
        try:
            return await asyncio.to_thread(self._analyze_sync, pipeline, upload)
        except asyncio.CancelledError:
            raise
        except _UNAVAILABLE_ERRORS:
            raise ModelServiceUnavailableError("model service is unavailable") from None
        except Exception:
            raise ModelServiceInferenceError("model analysis failed") from None

    async def restore(self, upload: ValidatedUpload) -> RestorationResult:
        """Restore through the pipeline's complete scientific result boundary."""
        return await self.restore_and_analyze(upload)

    async def restore_and_analyze(self, upload: ValidatedUpload) -> RestorationResult:
        """Run the blocking complete pipeline in a worker thread and map its result."""
        pipeline = self._ready_pipeline()
        try:
            model_result = await asyncio.to_thread(
                pipeline.restore_and_analyze,
                upload.encoded_bytes,
            )
            projection = model_result.platform_projection()
            return RestorationResult(**cast(dict[str, Any], projection))
        except asyncio.CancelledError:
            raise
        except _UNAVAILABLE_ERRORS:
            raise ModelServiceUnavailableError("model service is unavailable") from None
        except _KNOWN_INFERENCE_ERRORS:
            raise ModelServiceInferenceError("model inference failed") from None
        except Exception:
            raise ModelServiceInferenceError("model inference failed") from None


def create_model_service(settings: RuntimeSettings) -> SemiRestoreModelService:
    """Build the production model adapter without loading its checkpoint yet."""
    return SemiRestoreModelService(settings)


__all__ = ["PipelineFactory", "SemiRestoreModelService", "create_model_service"]
