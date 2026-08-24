"""End-to-end HTTP smoke validation for a running SemiRestore service."""

from __future__ import annotations

import base64
import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from io import BytesIO
from typing import Any
from urllib.parse import urlsplit

import httpx
from PIL import Image
from pydantic import ValidationError

from semirestore.api.schemas import AnalyzeResponse, RestoreResponse
from semirestore.platform.load_testing import synthetic_grayscale_png


class SmokeOperation(StrEnum):
    """Restoration operation exercised after the operational probes."""

    ANALYZE = "analyze"
    RESTORE = "restore"
    RESTORE_AND_ANALYZE = "restore-and-analyze"


_OPERATION_PATHS = {
    SmokeOperation.ANALYZE: "/api/v1/analyze",
    SmokeOperation.RESTORE: "/api/v1/restore",
    SmokeOperation.RESTORE_AND_ANALYZE: "/api/v1/restore-and-analyze",
}


class SmokeTestError(RuntimeError):
    """Safe base error for a failed smoke check."""


class SmokeTestUnavailableError(SmokeTestError):
    """Raised when the service is live but cannot accept model work."""


@dataclass(frozen=True, slots=True)
class SmokeTestConfig:
    """Validated, bounded settings for one smoke-test run."""

    base_url: str = "http://127.0.0.1:8000"
    operation: SmokeOperation = SmokeOperation.RESTORE_AND_ANALYZE
    timeout_seconds: float = 120.0
    width: int = 16
    height: int = 16

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username:
            raise ValueError("base_url must be an HTTP(S) URL without embedded credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain a query or fragment")
        if not isinstance(self.operation, SmokeOperation):
            raise ValueError("operation must be a supported SmokeOperation")
        if (
            not isinstance(self.timeout_seconds, (int, float))
            or isinstance(self.timeout_seconds, bool)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        for name in ("width", "height"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 4096:
                raise ValueError(f"{name} must be an integer from 1 through 4096")
        if self.width * self.height > 4_194_304:
            raise ValueError("synthetic smoke input cannot exceed 4194304 pixels")


@dataclass(frozen=True, slots=True)
class SmokeResponseSummary:
    """Safe projection of the validated operation response."""

    input_width: int
    input_height: int
    input_media_type: str
    diagnostic_sections: tuple[str, ...]
    warning_count: int
    suitability_recommendation: str | None = None
    restored_width: int | None = None
    restored_height: int | None = None
    restored_media_type: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    device: str | None = None


@dataclass(frozen=True, slots=True)
class SmokeReport:
    """Metadata-only record of a successful end-to-end smoke sequence."""

    schema_version: int
    operation: SmokeOperation
    checks: tuple[str, ...]
    model_state: str
    response: SmokeResponseSummary


def _json_object(response: httpx.Response) -> dict[str, Any]:
    try:
        document = response.json()
    except ValueError as error:
        raise SmokeTestError("A service response was not valid JSON.") from error
    if not isinstance(document, dict):
        raise SmokeTestError("A service response did not match the API contract.")
    return document


def _require_success(response: httpx.Response, check_name: str) -> dict[str, Any]:
    if not 200 <= response.status_code < 300:
        raise SmokeTestError(f"The {check_name} check did not succeed.")
    return _json_object(response)


def _validate_png(content: str, declared_width: int, declared_height: int) -> None:
    try:
        payload = base64.b64decode(content, validate=True)
        if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("not a PNG")
        with Image.open(BytesIO(payload)) as image:
            image.load()
            if image.format != "PNG" or image.size != (declared_width, declared_height):
                raise ValueError("PNG metadata mismatch")
    except (ValueError, OSError) as error:
        raise SmokeTestError("The restoration response did not contain a valid PNG.") from error


def validate_operation_payload(
    operation: SmokeOperation,
    document: dict[str, Any],
    *,
    input_width: int,
    input_height: int,
) -> SmokeResponseSummary:
    """Validate the public operation schema and return a content-free summary."""

    try:
        if operation is SmokeOperation.ANALYZE:
            response = AnalyzeResponse.model_validate(document)
            if (response.input.width, response.input.height) != (input_width, input_height):
                raise ValueError("input dimensions differ")
            return SmokeResponseSummary(
                input_width=response.input.width,
                input_height=response.input.height,
                input_media_type=response.input.media_type,
                diagnostic_sections=tuple(sorted(response.diagnostics)),
                warning_count=len(response.warnings),
                suitability_recommendation=response.suitability.recommendation,
            )

        response = RestoreResponse.model_validate(document)
        if (response.input.width, response.input.height) != (input_width, input_height):
            raise ValueError("input dimensions differ")
        _validate_png(response.image.content, response.image.width, response.image.height)
        return SmokeResponseSummary(
            input_width=response.input.width,
            input_height=response.input.height,
            input_media_type=response.input.media_type,
            diagnostic_sections=tuple(sorted(response.diagnostics)),
            warning_count=len(response.warnings),
            restored_width=response.image.width,
            restored_height=response.image.height,
            restored_media_type=response.image.media_type,
            model_name=response.model.name,
            model_version=response.model.version,
            device=response.inference.device,
        )
    except (ValidationError, ValueError) as error:
        raise SmokeTestError("The operation response did not match the API contract.") from error


async def run_smoke_test(
    config: SmokeTestConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> SmokeReport:
    """Run probes and one synthetic upload against the public HTTP interface."""

    image_bytes = synthetic_grayscale_png(config.width, config.height)
    try:
        async with httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            timeout=config.timeout_seconds,
            transport=transport,
        ) as client:
            live = _require_success(await client.get("/health/live"), "liveness")
            if live != {"status": "alive"}:
                raise SmokeTestError("The liveness response did not match the API contract.")

            ready_response = await client.get("/health/ready")
            ready = _json_object(ready_response)
            if ready_response.status_code != 200 or ready.get("ready") is not True:
                raise SmokeTestUnavailableError(
                    "The service is live but is not ready for restoration work."
                )
            if ready.get("state") != "ready":
                raise SmokeTestError("The readiness response did not match the API contract.")

            model = _require_success(await client.get("/health/model"), "model health")
            if model.get("ready") is not True or model.get("state") != "ready":
                raise SmokeTestError("The model-health response did not match readiness.")

            operation_path = _OPERATION_PATHS[config.operation]
            operation_response = await client.post(
                operation_path,
                files={"image": ("synthetic-smoke.png", image_bytes, "image/png")},
            )
            operation_document = _require_success(operation_response, "operation")
            summary = validate_operation_payload(
                config.operation,
                operation_document,
                input_width=config.width,
                input_height=config.height,
            )
    except SmokeTestError:
        raise
    except httpx.TimeoutException as error:
        raise SmokeTestError("The smoke test timed out.") from error
    except httpx.HTTPError as error:
        raise SmokeTestError("The smoke test could not reach the service.") from error

    return SmokeReport(
        schema_version=1,
        operation=config.operation,
        checks=("health/live", "health/ready", "health/model", "multipart-upload", "contract"),
        model_state=str(model["state"]),
        response=summary,
    )


def report_payload(report: SmokeReport) -> dict[str, Any]:
    """Serialize a successful report without carrying image data."""

    payload = asdict(report)
    payload["operation"] = report.operation.value
    return payload


__all__ = [
    "SmokeOperation",
    "SmokeReport",
    "SmokeResponseSummary",
    "SmokeTestConfig",
    "SmokeTestError",
    "SmokeTestUnavailableError",
    "report_payload",
    "run_smoke_test",
    "validate_operation_payload",
]
