"""Explicit test-only model-service fixtures.

Nothing in this module is imported by the production package.
"""

from __future__ import annotations

from collections.abc import Iterator
from io import BytesIO
from typing import cast

import pytest
from PIL import Image

from semirestore.api.uploads import ValidatedUpload
from semirestore.platform import (
    ModelHealth,
    ModelServiceState,
    ModelServiceUnavailableError,
    RestorationResult,
)

_DEFAULT_RESULT = object()


def _synthetic_png() -> bytes:
    output = BytesIO()
    Image.new("L", (2, 2), color=127).save(output, format="PNG")
    return output.getvalue()


class FakeModelService:
    """Lifecycle-only fake; it never performs or claims real restoration."""

    def __init__(
        self,
        *,
        health: ModelHealth | None = None,
        startup_error: Exception | None = None,
        shutdown_error: Exception | None = None,
        inference_error: Exception | None = None,
        restoration_result: object = _DEFAULT_RESULT,
    ) -> None:
        self.current_health = health or ModelHealth(
            state=ModelServiceState.READY,
            ready=True,
            device="test-device",
            model_version="synthetic-test-model",
            checkpoint_checksum="test-checksum",
        )
        self.startup_error = startup_error
        self.shutdown_error = shutdown_error
        self.inference_error = inference_error
        self.restoration_result = restoration_result
        self.startup_calls = 0
        self.shutdown_calls = 0
        self.restoration_calls = 0
        self.restoration_inputs: list[dict[str, str | int]] = []
        self.synthetic_output_bytes = _synthetic_png()

    async def startup(self) -> None:
        self.startup_calls += 1
        if self.startup_error is not None:
            raise self.startup_error

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        if self.shutdown_error is not None:
            raise self.shutdown_error

    def health(self) -> ModelHealth:
        return self.current_health

    async def restore(self, upload: ValidatedUpload) -> RestorationResult:
        self.restoration_calls += 1
        self.restoration_inputs.append(
            {
                "encoded_size": len(upload.encoded_bytes),
                "media_type": upload.media_type,
                "detected_format": upload.detected_format,
                "width": upload.width,
                "height": upload.height,
            }
        )
        if not self.current_health.ready:
            raise ModelServiceUnavailableError("synthetic test service is unready")
        if self.inference_error is not None:
            raise self.inference_error
        if self.restoration_result is not _DEFAULT_RESULT:
            return cast(RestorationResult, self.restoration_result)
        return RestorationResult(
            restored_image_bytes=self.synthetic_output_bytes,
            restored_media_type="image/png",
            restored_width=2,
            restored_height=2,
            original_width=upload.width,
            original_height=upload.height,
            inference_latency_ms=12.5,
            device="test-device",
            model_version="synthetic-test-model",
            checkpoint_checksum=f"sha256:{'a' * 64}",
            diagnostics={"synthetic": True, "source_format": upload.detected_format},
            warnings=("Synthetic test output; no real restoration was performed.",),
        )


@pytest.fixture
def fake_model_service() -> Iterator[FakeModelService]:
    """Provide an explicitly requested synthetic model-service test double."""
    yield FakeModelService()


@pytest.fixture
def fake_model_service_factory() -> type[FakeModelService]:
    """Expose the test fake constructor only to tests that request it."""
    return FakeModelService
