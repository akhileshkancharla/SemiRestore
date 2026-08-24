from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from semirestore.api import create_app
from semirestore.api.uploads import ValidatedUpload
from semirestore.model_manager import ModelManagerState, ModelManagerStatus
from semirestore.pipeline import PipelineError
from semirestore.pipeline import RestorationResult as PipelineResult
from semirestore.platform import (
    ModelService,
    ModelServiceInferenceError,
    ModelServiceInitializationError,
    ModelServiceState,
    RuntimeSettings,
    SemiRestoreModelService,
)
from semirestore.preprocessing import DEFAULT_LIMITS


def manager_status(*, ready: bool = True) -> ModelManagerStatus:
    return ModelManagerStatus(
        state=ModelManagerState.READY if ready else ModelManagerState.FAILED,
        ready=ready,
        model_name="naf_sr" if ready else None,
        architecture="statistics-conditioned NAF-SR" if ready else None,
        model_version="controlled-v1" if ready else None,
        training_revision="controlled-revision" if ready else None,
        resolved_device="cpu" if ready else None,
        parameter_count=42 if ready else None,
        checkpoint_path="private/checkpoint.pt",
        checkpoint_sha256="a" * 64 if ready else None,
        scale_factor=2 if ready else None,
        last_loading_error_category=None if ready else "checkpoint_verification",
        retry_permitted=not ready,
    )


def pipeline_result(width: int = 3, height: int = 2) -> PipelineResult:
    restored_width = width * 2
    restored_height = height * 2
    return PipelineResult(
        restored_image=np.full(
            (restored_height, restored_width),
            0.5,
            dtype=np.float32,
        ),
        png_bytes=b"\x89PNG\r\n\x1a\ncontrolled",
        media_type="image/png",
        png_bit_depth=16,
        original_width=width,
        original_height=height,
        restored_width=restored_width,
        restored_height=restored_height,
        input_diagnostics={"intensity": {"version": "controlled"}},
        restored_diagnostics={"intensity": {"version": "controlled"}},
        suitability_recommendation="restore",
        suitability_reasons=("Controlled suitability reason.",),
        restoration_metadata={"scale_factor": 2},
        spatial_metadata={"padding_required": False},
        tile_metadata=None,
        clipping_metadata={"clipping_occurred": False},
        quality_indicators={"dimension_contract_satisfied": True},
        model_name="naf_sr",
        model_version="controlled-v1",
        checkpoint_sha256="a" * 64,
        training_revision="controlled-revision",
        resolved_device="cpu",
        timing_ms={"restoration_total": 12.5, "total": 15.0},
        warnings=("Controlled public warning.",),
        limitations=("Controlled public limitation.",),
    )


class ControlledPipeline:
    def __init__(
        self,
        *,
        result: PipelineResult | None = None,
        status: ModelManagerStatus | None = None,
        inference_error: Exception | None = None,
        release: threading.Event | None = None,
    ) -> None:
        self.result = result or pipeline_result()
        self.current_status = status or manager_status()
        self.inference_error = inference_error
        self.release = release
        self.restore_started = threading.Event()
        self.restore_calls = 0
        self.close_calls = 0
        self.factory_thread: int | None = None
        self.restore_threads: list[int] = []
        self.close_thread: int | None = None
        self.preprocessing_limits = DEFAULT_LIMITS

    def status(self) -> ModelManagerStatus:
        return self.current_status

    def restore_and_analyze(self, image: bytes) -> PipelineResult:
        assert image
        self.restore_calls += 1
        self.restore_threads.append(threading.get_ident())
        self.restore_started.set()
        if self.release is not None:
            assert self.release.wait(timeout=2)
        if self.inference_error is not None:
            raise self.inference_error
        return self.result

    def close(self) -> None:
        self.close_calls += 1
        self.close_thread = threading.get_ident()


def validated_upload() -> ValidatedUpload:
    return ValidatedUpload(
        encoded_bytes=b"controlled-image",
        media_type="image/png",
        detected_format="PNG",
        width=3,
        height=2,
    )


def analysis_upload() -> ValidatedUpload:
    output = BytesIO()
    Image.new("L", (4, 3), color=96).save(output, format="PNG")
    return ValidatedUpload(
        encoded_bytes=output.getvalue(),
        media_type="image/png",
        detected_format="PNG",
        width=4,
        height=3,
    )


def png_upload() -> dict[str, tuple[str, bytes, str]]:
    output = BytesIO()
    Image.new("L", (3, 2), color=96).save(output, format="PNG")
    return {"image": ("input.png", output.getvalue(), "image/png")}


def test_adapter_loads_once_maps_real_result_and_closes_once(tmp_path: Path) -> None:
    pipeline = ControlledPipeline()
    factory_calls: list[dict[str, object]] = []
    main_thread = threading.get_ident()

    def factory(**kwargs: object) -> Any:
        pipeline.factory_thread = threading.get_ident()
        factory_calls.append(kwargs)
        return pipeline

    settings = RuntimeSettings(
        checkpoint_path=tmp_path / "model.pt",
        model_metadata_path=tmp_path / "checksums.json",
        model_config_path=tmp_path / "model.yaml",
        device_preference="cpu",
    )
    service = SemiRestoreModelService(settings, pipeline_factory=factory)

    async def exercise() -> tuple[Any, Any]:
        await service.startup()
        first = await service.restore(validated_upload())
        second = await service.restore(validated_upload())
        await service.shutdown()
        return first, second

    first, second = asyncio.run(exercise())

    assert isinstance(service, ModelService)
    assert len(factory_calls) == 1
    assert factory_calls[0] == {
        "checkpoint_path": tmp_path / "model.pt",
        "metadata_path": tmp_path / "checksums.json",
        "model_config_path": tmp_path / "model.yaml",
        "device": "cpu",
    }
    assert pipeline.factory_thread != main_thread
    assert pipeline.restore_threads and all(
        thread_id != main_thread for thread_id in pipeline.restore_threads
    )
    assert pipeline.close_thread != main_thread
    assert pipeline.restore_calls == 2
    assert pipeline.close_calls == 1
    assert first == second
    assert first.restored_image_bytes == pipeline.result.png_bytes
    assert first.restored_media_type == "image/png"
    assert (first.restored_width, first.restored_height) == (6, 4)
    assert first.diagnostics["pipeline_version"] == pipeline.result.pipeline_version
    assert first.model_version == "controlled-v1"
    assert first.checkpoint_checksum == "a" * 64
    assert service.health().state is ModelServiceState.STOPPED


def test_adapter_health_uses_safe_cached_manager_identity() -> None:
    pipeline = ControlledPipeline()
    service = SemiRestoreModelService(
        RuntimeSettings(),
        pipeline_factory=lambda **_: pipeline,
    )

    assert service.health().state is ModelServiceState.STARTING
    asyncio.run(service.startup())
    health = service.health()

    assert health.ready is True
    assert health.device == "cpu"
    assert health.model_version == "controlled-v1"
    assert health.checkpoint_checksum == "a" * 64
    assert "checkpoint.pt" not in repr(health)
    asyncio.run(service.shutdown())


def test_adapter_analyze_uses_public_model_diagnostics() -> None:
    pipeline = ControlledPipeline()
    service = SemiRestoreModelService(
        RuntimeSettings(),
        pipeline_factory=lambda **_: pipeline,
    )

    async def analyze() -> Any:
        await service.startup()
        result = await service.analyze(analysis_upload())
        await service.shutdown()
        return result

    result = asyncio.run(analyze())

    assert (result.original_width, result.original_height) == (4, 3)
    assert set(result.diagnostics) == {"preprocessing", "intensity", "structure"}
    assert result.suitability_recommendation in {"restore", "warn", "bypass"}
    assert result.suitability_reasons
    assert result.warnings
    assert result.analysis_latency_ms >= 0


def test_adapter_startup_failure_is_safe_and_closes_partial_pipeline() -> None:
    partial = ControlledPipeline(status=manager_status(ready=False))
    service = SemiRestoreModelService(
        RuntimeSettings(),
        pipeline_factory=lambda **_: partial,
    )

    with pytest.raises(ModelServiceInitializationError) as error:
        asyncio.run(service.startup())

    assert str(error.value) == "model service failed to initialize"
    assert service.health().state is ModelServiceState.UNAVAILABLE
    assert service.health().unavailable_reason == "model service failed to initialize"
    assert partial.close_calls == 1


def test_adapter_suppresses_pipeline_failure_details() -> None:
    pipeline = ControlledPipeline(
        inference_error=PipelineError("C:/private/checkpoint.pt tensor=[1] token=secret")
    )
    service = SemiRestoreModelService(
        RuntimeSettings(),
        pipeline_factory=lambda **_: pipeline,
    )

    async def fail() -> None:
        await service.startup()
        await service.restore(validated_upload())

    with pytest.raises(ModelServiceInferenceError) as error:
        asyncio.run(fail())

    assert str(error.value) == "model inference failed"
    for unsafe in ("private", "checkpoint.pt", "tensor", "token", "secret"):
        assert unsafe not in str(error.value)
    asyncio.run(service.shutdown())


def test_real_adapter_remains_behind_existing_application_gate() -> None:
    release = threading.Event()
    pipeline = ControlledPipeline(release=release)
    settings = RuntimeSettings(
        inference_concurrency_limit=1,
        concurrency_acquisition_timeout_seconds=0.02,
        inference_timeout_seconds=1.0,
    )
    service = SemiRestoreModelService(settings, pipeline_factory=lambda **_: pipeline)
    app = create_app(settings=settings, model_service_factory=lambda _: service)

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(client.post, "/api/v1/restore", files=png_upload())
        assert pipeline.restore_started.wait(timeout=1)
        try:
            started = time.monotonic()
            busy = client.post("/api/v1/restore", files=png_upload())
            elapsed = time.monotonic() - started
        finally:
            release.set()
        successful = first.result(timeout=2)

    assert successful.status_code == 200
    assert busy.status_code == 503
    assert busy.json()["error"]["code"] == "inference_busy"
    assert elapsed < 0.5
    assert pipeline.restore_calls == 1
    assert pipeline.close_calls == 1
