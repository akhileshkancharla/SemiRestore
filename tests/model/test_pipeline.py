from __future__ import annotations

import base64
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from semirestore.checkpoints import LoadedCheckpoint
from semirestore.model_manager import DEFAULT_CHECKPOINT_PATH, ModelManager, ModelNotReadyError
from semirestore.pipeline import PIPELINE_VERSION, PipelineConfig, SemiRestorePipeline
from semirestore.preprocessing import PreprocessingError
from semirestore.restoration_service import ManagerNotReadyError
from semirestore.structural_diagnostics import StructuralDiagnosticError


class PipelineModel(nn.Module):
    statistics_conditioning = True
    padder_size = 1
    scale = 2

    def __init__(self, *, delay: float = 0.0) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.9))
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        conditioning_statistics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del conditioning_statistics
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            return (
                F.interpolate(inputs, scale_factor=2, mode="bilinear", align_corners=False)
                * self.gain
            )
        finally:
            with self.lock:
                self.active -= 1


def _pipeline(
    *,
    model: PipelineModel | None = None,
    config: PipelineConfig | None = None,
) -> tuple[SemiRestorePipeline, PipelineModel, list[int]]:
    pipeline_model = PipelineModel() if model is None else model
    loads: list[int] = []

    def loader(**kwargs: object) -> LoadedCheckpoint:
        del kwargs
        loads.append(1)
        return LoadedCheckpoint(
            model=pipeline_model,
            device=torch.device("cpu"),
            checkpoint_path=Path("semirestore_conditioned.pt"),
            checkpoint_sha256="a" * 64,
            architecture="statistics-conditioned NAF-SR",
            model_name="naf_sr",
            parameter_count=sum(parameter.numel() for parameter in pipeline_model.parameters()),
            model_version="controlled-v1",
            training_revision="controlled-revision",
        )

    manager = ModelManager(loader=loader, device="cpu")
    manager.load()
    return SemiRestorePipeline(
        manager, config=PipelineConfig() if config is None else config
    ), pipeline_model, loads


def _image(height: int = 8, width: int = 10) -> np.ndarray:
    return np.linspace(0.05, 0.95, height * width, dtype=np.float32).reshape(height, width)


def test_controlled_direct_pipeline_contract() -> None:
    pipeline, _, _ = _pipeline()
    try:
        result = pipeline.restore_and_analyze(_image())
    finally:
        pipeline.close()

    assert result.original_width == 10
    assert result.original_height == 8
    assert result.restored_width == 20
    assert result.restored_height == 16
    assert result.restored_image.shape == (16, 20)
    assert result.media_type == "image/png"
    assert result.png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert result.quality_indicators["dimension_contract_satisfied"] is True
    assert result.quality_indicators["can_prove_reconstruction_correctness"] is False
    assert result.tile_metadata is None


def test_controlled_tiled_pipeline_exposes_plan_and_global_conditioning() -> None:
    pipeline, _, _ = _pipeline(
        config=PipelineConfig(mode="tiled", tile_size=6, overlap=2)
    )
    try:
        result = pipeline.restore_and_analyze(_image(9, 11))
    finally:
        pipeline.close()

    assert result.restored_image.shape == (18, 22)
    assert result.tile_metadata is not None
    assert result.tile_metadata["tile_count"] > 1
    assert result.tile_metadata["global_conditioning_reused"] is True


def test_preprocessing_occurs_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from semirestore import pipeline as pipeline_module

    calls = 0
    original = pipeline_module.preprocess_sem_image

    def counting_preprocess(image: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(image, **kwargs)

    monkeypatch.setattr(pipeline_module, "preprocess_sem_image", counting_preprocess)
    pipeline, _, _ = _pipeline()
    try:
        pipeline.restore_and_analyze(_image())
    finally:
        pipeline.close()

    assert calls == 1


def test_result_serialization_contains_png_and_no_device_objects() -> None:
    pipeline, _, _ = _pipeline()
    try:
        result = pipeline.restore_and_analyze(_image())
    finally:
        pipeline.close()

    payload = result.to_dict()
    rendered = json.dumps(payload, allow_nan=False)

    assert base64.b64decode(payload["restored_output"]["content"]) == result.png_bytes
    assert payload["pipeline_version"] == PIPELINE_VERSION
    assert "Tensor" not in rendered
    assert "array(" not in rendered
    assert result.metadata()["restored_output"]["content"] is None


def test_platform_projection_is_bounded_json_and_png_only() -> None:
    pipeline, _, _ = _pipeline()
    try:
        result = pipeline.restore_and_analyze(_image())
    finally:
        pipeline.close()

    projection = result.platform_projection()
    diagnostics = json.dumps(projection["diagnostics"], allow_nan=False).encode()

    assert projection["restored_media_type"] == "image/png"
    assert projection["restored_image_bytes"] == result.png_bytes
    assert len(diagnostics) < 65_536
    assert projection["device"] == "cpu"
    assert projection["model_version"] == "controlled-v1"
    assert projection["checkpoint_checksum"] == "a" * 64


def test_pipeline_records_required_warnings_and_limitations() -> None:
    pipeline, _, _ = _pipeline()
    try:
        result = pipeline.restore_and_analyze(np.full((8, 8), 0.5, dtype=np.float32))
    finally:
        pipeline.close()

    assert result.suitability_recommendation == "bypass"
    assert any("advisory" in warning for warning in result.warnings)
    assert any("hallucinate or oversmooth" in item for item in result.limitations)
    assert any("Out-of-domain" in item for item in result.limitations)
    assert any("downsample-only" in item for item in result.limitations)
    assert any("cannot prove" in item for item in result.limitations)
    assert any("lossless PNG" in item for item in result.limitations)


def test_phase_timings_and_provenance_are_finite() -> None:
    pipeline, _, _ = _pipeline()
    try:
        result = pipeline.restore_and_analyze(_image())
    finally:
        pipeline.close()

    assert all(value >= 0 and np.isfinite(value) for value in result.timing_ms.values())
    assert result.model_name == "naf_sr"
    assert result.model_version == "controlled-v1"
    assert result.checkpoint_sha256 == "a" * 64
    assert result.training_revision == "controlled-revision"


def test_one_model_allocation_is_reused_across_requests() -> None:
    pipeline, _, loads = _pipeline()
    try:
        first = pipeline.restore_and_analyze(_image())
        second = pipeline.restore_and_analyze(_image())
    finally:
        pipeline.close()

    assert loads == [1]
    assert first.checkpoint_sha256 == second.checkpoint_sha256


def test_concurrent_calls_preserve_shared_model_execution_lock() -> None:
    model = PipelineModel(delay=0.05)
    pipeline, _, _ = _pipeline(model=model)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(pipeline.restore_and_analyze, _image()) for _ in range(2)]
            results = [future.result(timeout=10) for future in futures]
    finally:
        pipeline.close()

    assert len(results) == 2
    assert model.max_active == 1


def test_invalid_input_and_closed_pipeline_fail_safely() -> None:
    pipeline, _, _ = _pipeline()
    with pytest.raises((PreprocessingError, StructuralDiagnosticError)):
        pipeline.restore_and_analyze(np.zeros((3, 3, 2), dtype=np.float32))
    pipeline.close()
    with pytest.raises((ManagerNotReadyError, ModelNotReadyError)):
        pipeline.restore_and_analyze(_image())


@pytest.fixture(scope="module")
def real_pipeline() -> SemiRestorePipeline:
    if not DEFAULT_CHECKPOINT_PATH.is_file():
        pytest.skip("verified ignored runtime checkpoint is not installed")
    pipeline = SemiRestorePipeline.from_config(device="cpu")
    yield pipeline
    pipeline.close()


@pytest.mark.local_checkpoint
def test_real_checkpoint_direct_pipeline(real_pipeline: SemiRestorePipeline) -> None:
    result = real_pipeline.restore_and_analyze(_image(9, 11), mode="direct")

    assert result.restored_image.shape == (18, 22)
    assert result.checkpoint_sha256 == (
        "273abd9d6dcfa9bdee71ac15016994962304b6c9d902898b4f4d503bed158c28"
    )
    assert result.model_version == "conditioned-d037473"
    assert result.media_type == "image/png"


@pytest.mark.local_checkpoint
def test_real_checkpoint_tiled_pipeline(real_pipeline: SemiRestorePipeline) -> None:
    original_config = real_pipeline.config
    real_pipeline.config = PipelineConfig(mode="tiled", tile_size=8, overlap=4)
    try:
        result = real_pipeline.restore_and_analyze(_image(9, 11), mode="tiled")
    finally:
        real_pipeline.config = original_config

    assert result.restored_image.shape == (18, 22)
    assert result.tile_metadata is not None
    assert result.tile_metadata["tile_count"] > 1
