"""Unified preprocessing, diagnostics, restoration, and assurance pipeline."""

from __future__ import annotations

import base64
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .checkpoints import DEFAULT_CONFIG_PATH, DEFAULT_METADATA_PATH
from .intensity_diagnostics import IntensityDiagnostics, analyze_intensity
from .model_manager import DEFAULT_CHECKPOINT_PATH, ModelManager, ModelManagerStatus
from .preprocessing import (
    DEFAULT_LIMITS,
    ImageInput,
    PreprocessingLimits,
    PreprocessingResult,
    preprocess_sem_image,
)
from .restoration_service import (
    DEFAULT_DIRECT_INFERENCE_MAX_PIXELS,
    SingleImageRestorationResult,
    SingleImageRestorationService,
)
from .structural_diagnostics import StructuralDiagnostics, analyze_structure
from .tiling import DEFAULT_MAX_TILE_COUNT

PIPELINE_VERSION = "semirestore-pipeline-v1"
PNG_MEDIA_TYPE = "image/png"
InferenceMode = Literal["direct", "tiled"]


class PipelineError(RuntimeError):
    """Raised when the unified model pipeline cannot complete safely."""


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Explicit inference and output policy for one persistent pipeline."""

    mode: InferenceMode = "direct"
    output_bit_depth: int = 16
    tile_size: int = 256
    overlap: int = 32
    max_tile_count: int = DEFAULT_MAX_TILE_COUNT
    max_padded_pixels_per_tile: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("direct", "tiled"):
            raise ValueError("Pipeline mode must be 'direct' or 'tiled'")
        if self.output_bit_depth not in (8, 16):
            raise ValueError("Pipeline output bit depth must be 8 or 16")
        for name, value, minimum in (
            ("tile_size", self.tile_size, 2),
            ("max_tile_count", self.max_tile_count, 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if type(self.overlap) is not int or not 0 <= self.overlap < self.tile_size:
            raise ValueError("overlap must be an integer in [0, tile_size)")
        if self.max_padded_pixels_per_tile is not None and (
            type(self.max_padded_pixels_per_tile) is not int
            or self.max_padded_pixels_per_tile < 1
        ):
            raise ValueError("max_padded_pixels_per_tile must be positive when provided")


DEFAULT_PIPELINE_CONFIG = PipelineConfig()


@dataclass(frozen=True, slots=True)
class RestorationResult:
    """Final model-facing result with an explicit JSON projection."""

    restored_image: np.ndarray
    png_bytes: bytes
    media_type: str
    png_bit_depth: int
    original_width: int
    original_height: int
    restored_width: int
    restored_height: int
    input_diagnostics: dict[str, Any]
    restored_diagnostics: dict[str, Any]
    suitability_recommendation: str
    suitability_reasons: tuple[str, ...]
    restoration_metadata: dict[str, Any]
    spatial_metadata: dict[str, Any]
    tile_metadata: dict[str, Any] | None
    clipping_metadata: dict[str, Any]
    quality_indicators: dict[str, Any]
    model_name: str
    model_version: str | None
    checkpoint_sha256: str
    training_revision: str | None
    resolved_device: str
    timing_ms: dict[str, float]
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]
    pipeline_version: str = PIPELINE_VERSION

    def __post_init__(self) -> None:
        if self.media_type != PNG_MEDIA_TYPE or not self.png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("RestorationResult requires lossless PNG output")
        if self.restored_image.shape != (self.restored_height, self.restored_width):
            raise ValueError("Restored array dimensions do not match result metadata")
        if self.restored_image.dtype != np.float32 or not np.isfinite(self.restored_image).all():
            raise ValueError("Restored array must be finite float32")
        if float(self.restored_image.min()) < 0.0 or float(self.restored_image.max()) > 1.0:
            raise ValueError("Restored array must be in [0, 1]")
        for value in self.timing_ms.values():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Pipeline timings must be finite and non-negative")
        json.dumps(self._metadata_dict(), allow_nan=False)

    def _metadata_dict(self) -> dict[str, Any]:
        return {
            "pipeline_version": self.pipeline_version,
            "media_type": self.media_type,
            "png_bit_depth": self.png_bit_depth,
            "png_size_bytes": len(self.png_bytes),
            "dimensions": {
                "original_width": self.original_width,
                "original_height": self.original_height,
                "restored_width": self.restored_width,
                "restored_height": self.restored_height,
            },
            "input_diagnostics": self.input_diagnostics,
            "restored_diagnostics": self.restored_diagnostics,
            "suitability": {
                "recommendation": self.suitability_recommendation,
                "reasons": list(self.suitability_reasons),
                "advisory_not_probability": True,
            },
            "restoration": self.restoration_metadata,
            "spatial": self.spatial_metadata,
            "tiles": self.tile_metadata,
            "clipping": self.clipping_metadata,
            "quality_indicators": self.quality_indicators,
            "model": {
                "name": self.model_name,
                "version": self.model_version,
                "checkpoint_sha256": self.checkpoint_sha256,
                "training_revision": self.training_revision,
                "device": self.resolved_device,
            },
            "timing_ms": self.timing_ms,
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
        }

    def to_dict(self, *, include_png_base64: bool = True) -> dict[str, Any]:
        """Return a strict JSON-compatible projection, optionally including PNG bytes."""

        payload = self._metadata_dict()
        payload["restored_output"] = {
            "encoding": "base64" if include_png_base64 else "omitted",
            "media_type": self.media_type,
            "content": (
                base64.b64encode(self.png_bytes).decode("ascii")
                if include_png_base64
                else None
            ),
        }
        json.dumps(payload, allow_nan=False)
        return payload

    def metadata(self) -> dict[str, Any]:
        """Return metadata without embedding image bytes."""

        return self.to_dict(include_png_base64=False)

    def platform_projection(self) -> dict[str, Any]:
        """Return fields that map directly into the platform RestorationResult."""

        diagnostics = {
            "pipeline_version": self.pipeline_version,
            "input": self.input_diagnostics,
            "suitability": {
                "recommendation": self.suitability_recommendation,
                "reasons": list(self.suitability_reasons),
                "advisory_not_probability": True,
            },
            "quality_indicators": self.quality_indicators,
            "clipping": self.clipping_metadata,
            "timing_ms": self.timing_ms,
            "limitations": list(self.limitations),
        }
        return {
            "restored_image_bytes": self.png_bytes,
            "restored_media_type": self.media_type,
            "restored_width": self.restored_width,
            "restored_height": self.restored_height,
            "original_width": self.original_width,
            "original_height": self.original_height,
            "inference_latency_ms": self.timing_ms["restoration_total"],
            "device": self.resolved_device,
            "model_version": self.model_version,
            "checkpoint_checksum": self.checkpoint_sha256,
            "diagnostics": diagnostics,
            "warnings": self.warnings,
        }


def _elapsed_ms(started: float, finished: float) -> float:
    return max(0.0, (finished - started) * 1000.0)


def _diagnostics_payload(
    intensity: IntensityDiagnostics,
    structure: StructuralDiagnostics,
) -> dict[str, Any]:
    return {
        "intensity": intensity.to_dict(),
        "structure": structure.to_dict(),
    }


def _quality_indicators(
    input_intensity: IntensityDiagnostics,
    input_structure: StructuralDiagnostics,
    restored_intensity: IntensityDiagnostics,
    restored_structure: StructuralDiagnostics,
    service_result: SingleImageRestorationResult,
) -> dict[str, Any]:
    def intensity_value(result: IntensityDiagnostics, name: str) -> float:
        return result.measurements[name].value

    def structure_value(result: StructuralDiagnostics, name: str) -> float:
        return result.measurements[name].value

    input_sharpness = structure_value(input_structure, "laplacian_sharpness_variance")
    output_sharpness = structure_value(restored_structure, "laplacian_sharpness_variance")
    sharpness_ratio = None if input_sharpness <= 1e-12 else output_sharpness / input_sharpness
    return {
        "kind": "no_reference_assurance_indicators",
        "can_prove_reconstruction_correctness": False,
        "dimension_contract_satisfied": (
            service_result.restored_width == service_result.original_width * 2
            and service_result.restored_height == service_result.original_height * 2
        ),
        "finite_normalized_output": True,
        "lossless_png_output": True,
        "mean_intensity_delta": (
            intensity_value(restored_intensity, "mean")
            - intensity_value(input_intensity, "mean")
        ),
        "contrast_proxy_delta": (
            intensity_value(restored_intensity, "contrast_proxy")
            - intensity_value(input_intensity, "contrast_proxy")
        ),
        "sharpness_proxy_ratio": sharpness_ratio,
        "input_approximate_noise_sigma": structure_value(
            input_structure, "approximate_noise_sigma"
        ),
        "restored_approximate_noise_sigma": structure_value(
            restored_structure, "approximate_noise_sigma"
        ),
    }


class SemiRestorePipeline:
    """Own one manager and restoration service for the process lifetime."""

    def __init__(
        self,
        manager: ModelManager,
        *,
        config: PipelineConfig = DEFAULT_PIPELINE_CONFIG,
        preprocessing_limits: PreprocessingLimits = DEFAULT_LIMITS,
        max_direct_input_pixels: int = DEFAULT_DIRECT_INFERENCE_MAX_PIXELS,
        service: SingleImageRestorationService | None = None,
    ) -> None:
        if not isinstance(manager, ModelManager):
            raise TypeError("manager must be a ModelManager")
        self.manager = manager
        self.config = config
        self.preprocessing_limits = preprocessing_limits
        self.service = service or SingleImageRestorationService(
            manager,
            preprocessing_limits=preprocessing_limits,
            max_direct_input_pixels=max_direct_input_pixels,
        )

    @classmethod
    def from_config(
        cls,
        *,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        metadata_path: str | Path = DEFAULT_METADATA_PATH,
        model_config_path: str | Path = DEFAULT_CONFIG_PATH,
        device: str = "auto",
        pipeline_config: PipelineConfig = DEFAULT_PIPELINE_CONFIG,
        preprocessing_limits: PreprocessingLimits = DEFAULT_LIMITS,
        max_direct_input_pixels: int = DEFAULT_DIRECT_INFERENCE_MAX_PIXELS,
    ) -> SemiRestorePipeline:
        """Verify and allocate the model exactly once, failing closed on error."""

        manager = ModelManager(
            checkpoint_path=checkpoint_path,
            metadata_path=metadata_path,
            config_path=model_config_path,
            device=device,
        )
        try:
            manager.load()
            return cls(
                manager,
                config=pipeline_config,
                preprocessing_limits=preprocessing_limits,
                max_direct_input_pixels=max_direct_input_pixels,
            )
        except Exception:
            manager.close()
            raise

    def close(self) -> None:
        self.manager.close()

    def status(self) -> ModelManagerStatus:
        return self.manager.status()

    def restore_and_analyze(
        self,
        image: ImageInput,
        *,
        mode: InferenceMode | None = None,
    ) -> RestorationResult:
        """Preprocess once, analyze, restore, and package a lossless result."""

        total_started = time.perf_counter()
        preprocessing_started = time.perf_counter()
        preprocessed: PreprocessingResult = preprocess_sem_image(
            image, limits=self.preprocessing_limits
        )
        preprocessing_finished = time.perf_counter()
        canonical = preprocessed.tensor[0, 0]

        input_diagnostics_started = time.perf_counter()
        input_intensity = analyze_intensity(canonical)
        input_structure = analyze_structure(canonical)
        input_diagnostics_finished = time.perf_counter()

        selected_mode = self.config.mode if mode is None else mode
        restoration_started = time.perf_counter()
        if selected_mode == "direct":
            service_result = self.service.restore(
                image,
                output_bit_depth=self.config.output_bit_depth,
                preprocessed=preprocessed,
            )
        elif selected_mode == "tiled":
            service_result = self.service.restore_tiled(
                image,
                tile_size=self.config.tile_size,
                overlap=self.config.overlap,
                output_bit_depth=self.config.output_bit_depth,
                max_padded_pixels_per_tile=self.config.max_padded_pixels_per_tile,
                max_tile_count=self.config.max_tile_count,
                preprocessed=preprocessed,
            )
        else:
            raise PipelineError("Inference mode must be direct or tiled")
        restoration_finished = time.perf_counter()

        output_diagnostics_started = time.perf_counter()
        restored_intensity = analyze_intensity(service_result.restored_image)
        restored_structure = analyze_structure(service_result.restored_image)
        indicators = _quality_indicators(
            input_intensity,
            input_structure,
            restored_intensity,
            restored_structure,
            service_result,
        )
        output_diagnostics_finished = time.perf_counter()

        packaging_started = time.perf_counter()
        limitations = (
            "Restoration may hallucinate or oversmooth fine structures and defects.",
            "Out-of-domain acquisition conditions may reduce reliability.",
            "External validation is limited by downsample-only reference construction.",
            "No-reference diagnostics cannot prove reconstruction correctness.",
            "Suitability is a rule-based advisory, not a calibrated probability.",
            "Scientific model output is lossless PNG only.",
        )
        warnings = list(service_result.warnings)
        warnings.extend(input_intensity.warnings)
        warnings.extend(input_structure.warnings)
        if input_structure.recommendation == "bypass":
            warnings.append(
                "The bypass recommendation is advisory; restoration was still performed explicitly."
            )
        warnings = list(dict.fromkeys(warnings))
        metadata = service_result.metadata()
        postprocessing = dict(service_result.postprocessing_metadata)
        clipping_keys = (
            "raw_minimum",
            "raw_maximum",
            "clipped_minimum",
            "clipped_maximum",
            "values_below_zero",
            "fraction_below_zero",
            "values_above_one",
            "fraction_above_one",
            "total_values",
            "clipping_occurred",
        )
        packaging_finished = time.perf_counter()
        total_finished = packaging_finished
        result = RestorationResult(
            restored_image=service_result.restored_image,
            png_bytes=service_result.png_bytes,
            media_type=service_result.media_type,
            png_bit_depth=service_result.png_bit_depth,
            original_width=service_result.original_width,
            original_height=service_result.original_height,
            restored_width=service_result.restored_width,
            restored_height=service_result.restored_height,
            input_diagnostics=_diagnostics_payload(input_intensity, input_structure),
            restored_diagnostics=_diagnostics_payload(
                restored_intensity, restored_structure
            ),
            suitability_recommendation=input_structure.recommendation,
            suitability_reasons=input_structure.reasons,
            restoration_metadata=metadata,
            spatial_metadata=service_result.spatial_plan.to_dict(),
            tile_metadata=(
                None
                if service_result.tiled_metadata is None
                else service_result.tiled_metadata.to_dict()
            ),
            clipping_metadata={key: postprocessing[key] for key in clipping_keys},
            quality_indicators=indicators,
            model_name=service_result.model_name,
            model_version=service_result.model_version,
            checkpoint_sha256=service_result.checkpoint_sha256,
            training_revision=service_result.training_revision,
            resolved_device=service_result.resolved_device,
            timing_ms={
                "preprocessing": _elapsed_ms(preprocessing_started, preprocessing_finished),
                "input_diagnostics": _elapsed_ms(
                    input_diagnostics_started, input_diagnostics_finished
                ),
                "restoration_total": _elapsed_ms(restoration_started, restoration_finished),
                "model_inference": service_result.model_inference_latency_ms,
                "output_diagnostics": _elapsed_ms(
                    output_diagnostics_started, output_diagnostics_finished
                ),
                "packaging": _elapsed_ms(packaging_started, packaging_finished),
                "total": _elapsed_ms(total_started, total_finished),
            },
            warnings=tuple(warnings),
            limitations=limitations,
        )
        return result


__all__ = [
    "PIPELINE_VERSION",
    "DEFAULT_PIPELINE_CONFIG",
    "PipelineConfig",
    "PipelineError",
    "RestorationResult",
    "SemiRestorePipeline",
]
