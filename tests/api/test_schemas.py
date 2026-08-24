from __future__ import annotations

from semirestore.api.schemas import (
    AnalysisTimingResponse,
    AnalyzeInputResponse,
    AnalyzeResponse,
    ErrorBody,
    ErrorCode,
    ErrorResponse,
    LiveResponse,
    ModelHealthResponse,
    ReadyResponse,
    SuitabilityResponse,
    VersionResponse,
)
from semirestore.platform import ModelServiceState


def test_operational_schemas_serialize_to_stable_json_values() -> None:
    assert LiveResponse().model_dump(mode="json") == {"status": "alive"}
    assert ReadyResponse(
        ready=False,
        state=ModelServiceState.UNAVAILABLE,
        unavailable_reason="model is unavailable",
    ).model_dump(mode="json") == {
        "ready": False,
        "state": "unavailable",
        "unavailable_reason": "model is unavailable",
    }
    assert ModelHealthResponse(
        ready=True,
        state=ModelServiceState.READY,
        device="cpu",
        model_version="v1",
        checkpoint_checksum="sha256:test",
    ).model_dump(mode="json") == {
        "ready": True,
        "state": "ready",
        "unavailable_reason": None,
        "device": "cpu",
        "model_version": "v1",
        "checkpoint_checksum": "sha256:test",
    }
    assert VersionResponse(version="0.1.0").model_dump(mode="json") == {
        "application": "semirestore",
        "version": "0.1.0",
    }


def test_error_envelope_serializes_optional_details_and_request_id() -> None:
    response = ErrorResponse(
        error=ErrorBody(
            code=ErrorCode.UPLOAD_TOO_LARGE,
            message="The uploaded file exceeds the configured size limit.",
            details={"maximum_bytes": 1024},
            request_id="request-123",
        )
    )

    assert response.model_dump(mode="json") == {
        "error": {
            "code": "upload_too_large",
            "message": "The uploaded file exceeds the configured size limit.",
            "details": {"maximum_bytes": 1024},
            "request_id": "request-123",
        }
    }


def test_error_envelope_keeps_request_id_shape_when_unavailable() -> None:
    response = ErrorResponse(
        error=ErrorBody(
            code=ErrorCode.INTERNAL_ERROR,
            message="An internal server error occurred.",
        )
    )

    assert response.model_dump(mode="json")["error"]["request_id"] is None


def test_analysis_schema_serializes_diagnostics_and_advisory_suitability() -> None:
    response = AnalyzeResponse(
        input=AnalyzeInputResponse(width=10, height=8, media_type="image/png"),
        analysis=AnalysisTimingResponse(latency_ms=4.5),
        diagnostics={"intensity": {"version": "v1"}},
        suitability=SuitabilityResponse(
            recommendation="warn",
            reasons=("Controlled public reason.",),
        ),
        warnings=("Controlled public warning.",),
    )

    assert response.model_dump(mode="json") == {
        "input": {"width": 10, "height": 8, "media_type": "image/png"},
        "analysis": {"latency_ms": 4.5},
        "diagnostics": {"intensity": {"version": "v1"}},
        "suitability": {
            "recommendation": "warn",
            "reasons": ["Controlled public reason."],
            "advisory_not_probability": True,
        },
        "warnings": ["Controlled public warning."],
    }
