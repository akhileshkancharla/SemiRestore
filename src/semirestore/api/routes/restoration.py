"""Versioned restoration endpoint."""

from __future__ import annotations

import base64
from typing import Annotated

from fastapi import APIRouter, Depends, File, Request, UploadFile

from semirestore.api.application import ApplicationRuntime
from semirestore.api.dependencies import (
    get_runtime,
    require_inference_gate,
    require_model_service,
)
from semirestore.api.errors import RestorationFailedError
from semirestore.api.observability import observe_inference
from semirestore.api.schemas import (
    InferenceResponse,
    ModelIdentityResponse,
    RestoredImageResponse,
    RestoreInputResponse,
    RestoreResponse,
)
from semirestore.api.uploads import ValidatedUpload, validate_upload
from semirestore.platform import ModelServiceUnavailableError, RestorationResult

router = APIRouter(prefix="/api/v1", tags=["restoration"])

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
            model_version=result.model_version,
            checkpoint_checksum=result.checkpoint_checksum,
            diagnostics=result.diagnostics,
            warnings=result.warnings,
        )
        if (validated.original_width, validated.original_height) != (upload.width, upload.height):
            raise ValueError("result input dimensions do not match the validated upload")
        return validated
    except Exception as error:
        raise RestorationFailedError() from error


@router.post("/restore", response_model=RestoreResponse)
async def restore(
    request: Request,
    runtime: RuntimeDependency,
    image: ImageUpload = None,
) -> RestoreResponse:
    """Validate one upload and restore it with the lifespan-owned service."""
    validated_upload = await validate_upload(image, runtime.settings)

    async def invoke_service() -> object:
        service = require_model_service(runtime)
        if not runtime.model_health().ready:
            raise ModelServiceUnavailableError("model service is unavailable")
        gate = require_inference_gate(runtime)
        return await gate.run(lambda: service.restore(validated_upload))

    service_result = await observe_inference(request, invoke_service)
    result = _validate_service_result(service_result, validated_upload)
    return RestoreResponse(
        image=RestoredImageResponse(
            media_type=result.restored_media_type,
            content=base64.b64encode(result.restored_image_bytes).decode("ascii"),
            width=result.restored_width,
            height=result.restored_height,
        ),
        input=RestoreInputResponse(
            width=validated_upload.width,
            height=validated_upload.height,
            media_type=validated_upload.media_type,
        ),
        inference=InferenceResponse(
            latency_ms=result.inference_latency_ms,
            device=result.device,
        ),
        model=ModelIdentityResponse(
            version=result.model_version,
            checkpoint_checksum=result.checkpoint_checksum,
        ),
        diagnostics=dict(result.diagnostics),
        warnings=result.warnings,
    )
