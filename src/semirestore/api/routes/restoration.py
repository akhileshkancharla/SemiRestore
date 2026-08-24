"""Versioned restoration endpoint."""

from __future__ import annotations

import base64
from typing import Annotated

from fastapi import APIRouter, Depends, File, Request, UploadFile, status

from semirestore.api.application import ApplicationRuntime
from semirestore.api.dependencies import (
    get_runtime,
    require_inference_gate,
    require_model_service,
)
from semirestore.api.errors import RestorationFailedError
from semirestore.api.observability import observe_inference, observe_restoration
from semirestore.api.schemas import (
    AnalysisTimingResponse,
    AnalyzeInputResponse,
    AnalyzeResponse,
    ErrorResponse,
    InferenceResponse,
    ModelIdentityResponse,
    RestoredImageResponse,
    RestoreInputResponse,
    RestoreResponse,
    SuitabilityResponse,
)
from semirestore.api.uploads import ValidatedUpload, validate_upload
from semirestore.platform import (
    AnalysisResult,
    ModelServiceUnavailableError,
    RestorationResult,
)

router = APIRouter(prefix="/api/v1", tags=["restoration"])

RESTORATION_ERROR_RESPONSES = {
    status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
    status.HTTP_413_CONTENT_TOO_LARGE: {"model": ErrorResponse},
    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"model": ErrorResponse},
    status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ErrorResponse},
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
    status.HTTP_504_GATEWAY_TIMEOUT: {"model": ErrorResponse},
}

RuntimeDependency = Annotated[ApplicationRuntime, Depends(get_runtime)]
ImageUpload = Annotated[
    UploadFile | None,
    File(description="One PNG, JPEG, or single-frame TIFF SEM image."),
]


def _validate_service_result(
    result: object,
    upload: ValidatedUpload,
) -> RestorationResult:
    """Revalidate the adapter boundary before serializing its output."""
    try:
        if not isinstance(result, RestorationResult):
            raise TypeError("unexpected restoration result type")
        validated = RestorationResult(
            restored_image_bytes=result.restored_image_bytes,
            restored_media_type=result.restored_media_type,
            restored_width=result.restored_width,
            restored_height=result.restored_height,
            original_width=result.original_width,
            original_height=result.original_height,
            inference_latency_ms=result.inference_latency_ms,
            device=result.device,
            model_name=result.model_name,
            model_version=result.model_version,
            training_revision=result.training_revision,
            checkpoint_checksum=result.checkpoint_checksum,
            phase_latency_ms=result.phase_latency_ms,
            diagnostics=result.diagnostics,
            warnings=result.warnings,
        )
        if (
            validated.restored_media_type != "image/png"
            or not validated.restored_image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        ):
            raise ValueError("restoration result is not a lossless PNG")
        if (validated.original_width, validated.original_height) != (upload.width, upload.height):
            raise ValueError("result input dimensions do not match the validated upload")
        return validated
    except Exception as error:
        raise RestorationFailedError() from error


def _validate_analysis_result(result: object, upload: ValidatedUpload) -> AnalysisResult:
    """Revalidate safe diagnostic data before public serialization."""
    try:
        if not isinstance(result, AnalysisResult):
            raise TypeError("unexpected analysis result type")
        validated = AnalysisResult(
            original_width=result.original_width,
            original_height=result.original_height,
            diagnostics=result.diagnostics,
            suitability_recommendation=result.suitability_recommendation,
            suitability_reasons=result.suitability_reasons,
            warnings=result.warnings,
            analysis_latency_ms=result.analysis_latency_ms,
        )
        if (validated.original_width, validated.original_height) != (upload.width, upload.height):
            raise ValueError("analysis dimensions do not match the validated upload")
        return validated
    except Exception as error:
        raise RestorationFailedError() from error


def _serialize_restoration(
    result: RestorationResult,
    upload: ValidatedUpload,
) -> RestoreResponse:
    return RestoreResponse(
        image=RestoredImageResponse(
            media_type="image/png",
            content=base64.b64encode(result.restored_image_bytes).decode("ascii"),
            width=result.restored_width,
            height=result.restored_height,
        ),
        input=RestoreInputResponse(
            width=upload.width,
            height=upload.height,
            media_type=upload.media_type,
        ),
        inference=InferenceResponse(
            latency_ms=result.inference_latency_ms,
            device=result.device,
            phase_latency_ms=dict(result.phase_latency_ms),
        ),
        model=ModelIdentityResponse(
            name=result.model_name,
            version=result.model_version,
            training_revision=result.training_revision,
            checkpoint_checksum=result.checkpoint_checksum,
        ),
        diagnostics=dict(result.diagnostics),
        warnings=result.warnings,
    )


async def _execute_restoration(
    request: Request,
    runtime: ApplicationRuntime,
    upload: ValidatedUpload,
    *,
    include_analysis: bool,
) -> RestorationResult:
    async def invoke_service() -> object:
        service = require_model_service(runtime)
        if not runtime.model_health().ready:
            raise ModelServiceUnavailableError("model service is unavailable")
        gate = require_inference_gate(runtime)
        operation = service.restore_and_analyze if include_analysis else service.restore
        return await gate.run(lambda: operation(upload))

    async def execute() -> RestorationResult:
        service_result = await observe_inference(request, invoke_service)
        return _validate_service_result(service_result, upload)

    return await observe_restoration(request, execute)


@router.post(
    "/restore",
    response_model=RestoreResponse,
    responses=RESTORATION_ERROR_RESPONSES,
)
async def restore(
    request: Request,
    runtime: RuntimeDependency,
    image: ImageUpload = None,
) -> RestoreResponse:
    """Validate one upload and restore it with the lifespan-owned service."""
    validated_upload = await validate_upload(image, runtime.settings)

    result = await _execute_restoration(
        request,
        runtime,
        validated_upload,
        include_analysis=False,
    )
    return _serialize_restoration(result, validated_upload)


@router.post(
    "/restore-and-analyze",
    response_model=RestoreResponse,
    responses=RESTORATION_ERROR_RESPONSES,
)
async def restore_and_analyze(
    request: Request,
    runtime: RuntimeDependency,
    image: ImageUpload = None,
) -> RestoreResponse:
    """Restore one image and return the complete scientific result projection."""
    validated_upload = await validate_upload(image, runtime.settings)
    result = await _execute_restoration(
        request,
        runtime,
        validated_upload,
        include_analysis=True,
    )
    return _serialize_restoration(result, validated_upload)


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses=RESTORATION_ERROR_RESPONSES,
)
async def analyze(
    request: Request,
    runtime: RuntimeDependency,
    image: ImageUpload = None,
) -> AnalyzeResponse:
    """Run model-owned input diagnostics through the lifespan service."""
    validated_upload = await validate_upload(image, runtime.settings)

    async def invoke_service() -> object:
        service = require_model_service(runtime)
        if not runtime.model_health().ready:
            raise ModelServiceUnavailableError("model service is unavailable")
        gate = require_inference_gate(runtime)
        return await gate.run(lambda: service.analyze(validated_upload))

    service_result = await observe_inference(request, invoke_service)
    result = _validate_analysis_result(service_result, validated_upload)
    return AnalyzeResponse(
        input=AnalyzeInputResponse(
            width=validated_upload.width,
            height=validated_upload.height,
            media_type=validated_upload.media_type,
        ),
        analysis=AnalysisTimingResponse(latency_ms=result.analysis_latency_ms),
        diagnostics=dict(result.diagnostics),
        suitability=SuitabilityResponse(
            recommendation=result.suitability_recommendation,
            reasons=result.suitability_reasons,
        ),
        warnings=result.warnings,
    )
