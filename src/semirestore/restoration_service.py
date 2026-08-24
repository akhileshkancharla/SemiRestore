"""Synchronous single-image restoration using one ready persistent model."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import torch

from .model_manager import (
    ModelManager,
    ModelManagerClosedError,
    ModelManagerState,
    ModelManagerStatus,
    ModelNotReadyError,
)
from .postprocessing import (
    OutputValidationError,
    PostprocessingError,
    PostprocessingResult,
    UnsupportedOutputError,
    postprocess_restoration,
)
from .preprocessing import (
    DEFAULT_LIMITS,
    ImageDecodeError,
    ImageInput,
    ImageResourceError,
    ImageValidationError,
    PreprocessingError,
    PreprocessingLimits,
    PreprocessingResult,
    UnsupportedInputError,
    preprocess_sem_image,
)
from .spatial import SpatialPlan, SpatialPlanningError, create_spatial_plan

DEFAULT_DIRECT_INFERENCE_MAX_PIXELS = 512 * 512
PNG_MEDIA_TYPE = "image/png"
SIGNIFICANT_PADDING_OVERHEAD_FRACTION = 0.25
SUPPORTED_OUTPUT_BIT_DEPTHS = frozenset({8, 16})


class RestorationErrorCategory(StrEnum):
    """Stable failure categories safe to map across a platform boundary."""

    MANAGER_NOT_READY = "manager_not_ready"
    INVALID_INPUT = "invalid_input"
    RESOURCE_LIMIT = "resource_limit"
    UNSUPPORTED_OUTPUT = "unsupported_output"
    PREPROCESSING = "preprocessing_failure"
    DEVICE_TRANSFER = "device_transfer_failure"
    MODEL_INFERENCE = "model_inference_failure"
    INVALID_MODEL_OUTPUT = "invalid_model_output"
    POSTPROCESSING = "postprocessing_failure"
    SPATIAL_PLAN = "spatial_plan_failure"


class RestorationServiceError(RuntimeError):
    """Base safe error returned by the synchronous restoration boundary."""

    category: RestorationErrorCategory

    def __init__(self, category: RestorationErrorCategory, message: str) -> None:
        self.category = category
        super().__init__(message)


class ManagerNotReadyError(RestorationServiceError):
    def __init__(self) -> None:
        super().__init__(
            RestorationErrorCategory.MANAGER_NOT_READY,
            "The restoration model is not ready",
        )


class InvalidRestorationInputError(RestorationServiceError):
    def __init__(self) -> None:
        super().__init__(
            RestorationErrorCategory.INVALID_INPUT,
            "The restoration input is invalid or unsupported",
        )


class RestorationResourceLimitError(RestorationServiceError):
    def __init__(self, message: str = "The restoration input exceeds a resource limit") -> None:
        super().__init__(RestorationErrorCategory.RESOURCE_LIMIT, message)


class UnsupportedRestorationOutputError(RestorationServiceError):
    def __init__(self) -> None:
        super().__init__(
            RestorationErrorCategory.UNSUPPORTED_OUTPUT,
            "Restoration output must be an 8-bit or 16-bit PNG",
        )


class RestorationPreprocessingError(RestorationServiceError):
    def __init__(self) -> None:
        super().__init__(
            RestorationErrorCategory.PREPROCESSING,
            "Restoration preprocessing failed",
        )


class RestorationDeviceTransferError(RestorationServiceError):
    def __init__(self) -> None:
        super().__init__(
            RestorationErrorCategory.DEVICE_TRANSFER,
            "Could not prepare the restoration input on the model device",
        )


class RestorationInferenceError(RestorationServiceError):
    def __init__(self) -> None:
        super().__init__(
            RestorationErrorCategory.MODEL_INFERENCE,
            "Model inference failed",
        )


class InvalidModelOutputError(RestorationServiceError):
    def __init__(self) -> None:
        super().__init__(
            RestorationErrorCategory.INVALID_MODEL_OUTPUT,
            "The model produced an invalid restoration output",
        )


class RestorationPostprocessingError(RestorationServiceError):
    def __init__(self) -> None:
        super().__init__(
            RestorationErrorCategory.POSTPROCESSING,
            "Restoration postprocessing failed",
        )


class RestorationSpatialPlanError(RestorationServiceError):
    def __init__(self) -> None:
        super().__init__(
            RestorationErrorCategory.SPATIAL_PLAN,
            "The loaded model has an invalid spatial contract",
        )


@dataclass(frozen=True, slots=True)
class SingleImageRestorationResult:
    """Restored scientific image, lossless payload, provenance, and timings."""

    restored_image: np.ndarray
    png_bytes: bytes
    media_type: str
    png_bit_depth: int
    original_width: int
    original_height: int
    restored_width: int
    restored_height: int
    scale_factor: int
    spatial_plan: SpatialPlan
    preprocessing_metadata: dict[str, object]
    postprocessing_metadata: dict[str, object]
    preprocessing_latency_ms: float
    device_transfer_latency_ms: float
    inference_wait_latency_ms: float
    model_inference_latency_ms: float
    postprocessing_latency_ms: float
    total_latency_ms: float
    resolved_device: str
    model_name: str
    model_version: str | None
    training_revision: str | None
    checkpoint_sha256: str
    warnings: tuple[str, ...]

    def metadata(self) -> dict[str, object]:
        """Return JSON-compatible metadata without image arrays or encoded bytes."""

        return {
            "media_type": self.media_type,
            "png_bit_depth": self.png_bit_depth,
            "png_size_bytes": len(self.png_bytes),
            "original_width": self.original_width,
            "original_height": self.original_height,
            "restored_width": self.restored_width,
            "restored_height": self.restored_height,
            "scale_factor": self.scale_factor,
            "spatial_plan": self.spatial_plan.to_dict(),
            "preprocessing": dict(self.preprocessing_metadata),
            "postprocessing": dict(self.postprocessing_metadata),
            "latency_ms": {
                "preprocessing": self.preprocessing_latency_ms,
                "device_transfer": self.device_transfer_latency_ms,
                "inference_wait": self.inference_wait_latency_ms,
                "model_inference": self.model_inference_latency_ms,
                "postprocessing": self.postprocessing_latency_ms,
                "total": self.total_latency_ms,
            },
            "resolved_device": self.resolved_device,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "training_revision": self.training_revision,
            "checkpoint_sha256": self.checkpoint_sha256,
            "warnings": list(self.warnings),
        }


Preprocessor = Callable[..., PreprocessingResult]
Postprocessor = Callable[..., PostprocessingResult]
Clock = Callable[[], float]


def _elapsed_ms(started: float, finished: float) -> float:
    return max(0.0, (finished - started) * 1000.0)


def _synchronize_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _spatial_warnings(plan: SpatialPlan) -> tuple[str, ...]:
    if plan.padding_overhead_fraction < SIGNIFICANT_PADDING_OVERHEAD_FRACTION:
        return ()
    percentage = plan.padding_overhead_fraction * 100.0
    return (
        f"Internal alignment padding adds {plan.padding_overhead_pixels} compute pixels "
        f"({percentage:.1f}% over the unpadded input).",
    )


class SingleImageRestorationService:
    """Restore one image at a time with a ready, process-persistent model."""

    def __init__(
        self,
        manager: ModelManager,
        *,
        preprocessing_limits: PreprocessingLimits = DEFAULT_LIMITS,
        max_direct_input_pixels: int = DEFAULT_DIRECT_INFERENCE_MAX_PIXELS,
        preprocessor: Preprocessor = preprocess_sem_image,
        postprocessor: Postprocessor = postprocess_restoration,
        clock: Clock = time.perf_counter,
    ) -> None:
        if not isinstance(manager, ModelManager):
            raise TypeError("manager must be a ModelManager")
        if not isinstance(preprocessing_limits, PreprocessingLimits):
            raise TypeError("preprocessing_limits must be PreprocessingLimits")
        if type(max_direct_input_pixels) is not int or max_direct_input_pixels < 1:
            raise ValueError("max_direct_input_pixels must be a positive integer")
        if max_direct_input_pixels > preprocessing_limits.max_pixels:
            raise ValueError(
                "max_direct_input_pixels cannot exceed the preprocessing pixel limit"
            )
        self._manager = manager
        self._preprocessing_limits = preprocessing_limits
        self._max_direct_input_pixels = max_direct_input_pixels
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._clock = clock
        self._inference_lock = threading.Lock()

    @property
    def max_direct_input_pixels(self) -> int:
        return self._max_direct_input_pixels

    def _ready_model_and_status(self) -> tuple[torch.nn.Module, ModelManagerStatus]:
        status = self._manager.status()
        if status.state is not ModelManagerState.READY or not status.ready:
            raise ManagerNotReadyError()
        try:
            model = self._manager.model
        except (ModelNotReadyError, ModelManagerClosedError):
            raise ManagerNotReadyError() from None
        required_identity = (
            status.resolved_device,
            status.model_name,
            status.checkpoint_sha256,
            status.scale_factor,
        )
        if any(value is None for value in required_identity):
            raise ManagerNotReadyError()
        return model, status

    def restore(
        self,
        image: ImageInput,
        *,
        output_bit_depth: int = 16,
    ) -> SingleImageRestorationResult:
        """Run one validated FP32 restoration without loading or persisting data."""

        if type(output_bit_depth) is not int or output_bit_depth not in SUPPORTED_OUTPUT_BIT_DEPTHS:
            raise UnsupportedRestorationOutputError()
        model, status = self._ready_model_and_status()
        total_started = self._clock()

        preprocessing_started = self._clock()
        try:
            preprocessed = self._preprocessor(image, limits=self._preprocessing_limits)
        except ImageResourceError:
            raise RestorationResourceLimitError() from None
        except (UnsupportedInputError, ImageDecodeError, ImageValidationError):
            raise InvalidRestorationInputError() from None
        except PreprocessingError:
            raise RestorationPreprocessingError() from None
        except Exception:
            raise RestorationPreprocessingError() from None
        preprocessing_finished = self._clock()

        try:
            spatial_plan = create_spatial_plan(
                original_width=preprocessed.original_width,
                original_height=preprocessed.original_height,
                alignment=getattr(model, "padder_size", None),
                scale_factor=status.scale_factor,
            )
        except SpatialPlanningError:
            raise RestorationSpatialPlanError() from None
        if spatial_plan.padded_input_pixels > self._max_direct_input_pixels:
            raise RestorationResourceLimitError(
                f"Aligned input requires {spatial_plan.padded_input_pixels} compute pixels, "
                f"exceeding the direct-inference limit of {self._max_direct_input_pixels}; "
                "use tiled inference for larger images"
            )

        try:
            device = torch.device(status.resolved_device)
            _synchronize_cuda(device)
            transfer_started = self._clock()
            model_input = preprocessed.tensor.to(device=device, dtype=torch.float32)
            _synchronize_cuda(device)
            transfer_finished = self._clock()
        except Exception:
            raise RestorationDeviceTransferError() from None

        wait_started = self._clock()
        with self._inference_lock:
            wait_finished = self._clock()
            try:
                _synchronize_cuda(device)
                inference_started = self._clock()
                with torch.inference_mode():
                    raw_output = model(model_input)
                _synchronize_cuda(device)
                inference_finished = self._clock()
            except Exception:
                raise RestorationInferenceError() from None

        postprocessing_started = self._clock()
        try:
            postprocessed = self._postprocessor(
                raw_output,
                original_width=preprocessed.original_width,
                original_height=preprocessed.original_height,
            )
        except (OutputValidationError, UnsupportedOutputError):
            raise InvalidModelOutputError() from None
        except PostprocessingError:
            raise RestorationPostprocessingError() from None
        except Exception:
            raise RestorationPostprocessingError() from None
        if (
            postprocessed.restored_width != spatial_plan.final_restored_width
            or postprocessed.restored_height != spatial_plan.final_restored_height
        ):
            raise InvalidModelOutputError()
        try:
            png_bytes = postprocessed.encode(encoding="png", bit_depth=output_bit_depth)
        except PostprocessingError:
            raise RestorationPostprocessingError() from None
        except Exception:
            raise RestorationPostprocessingError() from None
        postprocessing_finished = self._clock()
        total_finished = self._clock()

        return SingleImageRestorationResult(
            restored_image=postprocessed.image,
            png_bytes=png_bytes,
            media_type=PNG_MEDIA_TYPE,
            png_bit_depth=output_bit_depth,
            original_width=preprocessed.original_width,
            original_height=preprocessed.original_height,
            restored_width=postprocessed.restored_width,
            restored_height=postprocessed.restored_height,
            scale_factor=postprocessed.scale_factor,
            spatial_plan=spatial_plan,
            preprocessing_metadata=preprocessed.metadata(),
            postprocessing_metadata=postprocessed.metadata(),
            preprocessing_latency_ms=_elapsed_ms(
                preprocessing_started,
                preprocessing_finished,
            ),
            device_transfer_latency_ms=_elapsed_ms(transfer_started, transfer_finished),
            inference_wait_latency_ms=_elapsed_ms(wait_started, wait_finished),
            model_inference_latency_ms=_elapsed_ms(inference_started, inference_finished),
            postprocessing_latency_ms=_elapsed_ms(
                postprocessing_started,
                postprocessing_finished,
            ),
            total_latency_ms=_elapsed_ms(total_started, total_finished),
            resolved_device=status.resolved_device,
            model_name=status.model_name,
            model_version=status.model_version,
            training_revision=status.training_revision,
            checkpoint_sha256=status.checkpoint_sha256,
            warnings=(
                preprocessed.warnings
                + _spatial_warnings(spatial_plan)
                + postprocessed.warnings
            ),
        )


__all__ = [
    "DEFAULT_DIRECT_INFERENCE_MAX_PIXELS",
    "InvalidModelOutputError",
    "InvalidRestorationInputError",
    "ManagerNotReadyError",
    "PNG_MEDIA_TYPE",
    "RestorationDeviceTransferError",
    "RestorationErrorCategory",
    "RestorationInferenceError",
    "RestorationPostprocessingError",
    "RestorationPreprocessingError",
    "RestorationResourceLimitError",
    "RestorationServiceError",
    "RestorationSpatialPlanError",
    "SIGNIFICANT_PADDING_OVERHEAD_FRACTION",
    "SingleImageRestorationResult",
    "SingleImageRestorationService",
    "UnsupportedRestorationOutputError",
]
