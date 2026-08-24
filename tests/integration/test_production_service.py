from __future__ import annotations

import base64
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from torch import nn
from torch.nn import functional as F

from semirestore.api import create_app
from semirestore.api.observability import configure_application_logging
from semirestore.checkpoints import LoadedCheckpoint, load_checkpoint_metadata
from semirestore.model_manager import DEFAULT_CHECKPOINT_PATH, ModelManager, ModelManagerState
from semirestore.pipeline import SemiRestorePipeline
from semirestore.platform import RuntimeSettings, SemiRestoreModelService


class IntegrationModel(nn.Module):
    statistics_conditioning = True
    padder_size = 1
    scale = 2

    def __init__(self, *, release: threading.Event | None = None) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.9))
        self.release = release
        self.forward_started = threading.Event()
        self.forward_calls = 0
        self.active = 0
        self.maximum_active = 0
        self._lock = threading.Lock()

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        conditioning_statistics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del conditioning_statistics
        with self._lock:
            self.forward_calls += 1
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        self.forward_started.set()
        try:
            if self.release is not None:
                assert self.release.wait(timeout=3)
            return (
                F.interpolate(inputs, scale_factor=2, mode="bilinear", align_corners=False)
                * self.gain
            )
        finally:
            with self._lock:
                self.active -= 1


def png_bytes(size: tuple[int, int] = (10, 8)) -> bytes:
    width, height = size
    values = np.linspace(12, 240, width * height, dtype=np.uint8).reshape(height, width)
    output = BytesIO()
    Image.fromarray(values).save(output, format="PNG")
    return output.getvalue()


def upload(
    encoded: bytes | None = None,
    *,
    filename: str = "input.png",
) -> dict[str, tuple[str, bytes, str]]:
    return {"image": (filename, encoded or png_bytes(), "image/png")}


def build_pipeline(
    model: IntegrationModel,
    load_calls: list[int],
) -> SemiRestorePipeline:
    def loader(**_: object) -> LoadedCheckpoint:
        load_calls.append(1)
        return LoadedCheckpoint(
            model=model,
            device=torch.device("cpu"),
            checkpoint_path=Path("controlled-integration.pt"),
            checkpoint_sha256="c" * 64,
            architecture="statistics-conditioned NAF-SR",
            model_name="naf_sr",
            parameter_count=sum(parameter.numel() for parameter in model.parameters()),
            model_version="controlled-integration-v1",
            training_revision="controlled-integration-revision",
        )

    manager = ModelManager(loader=loader, device="cpu")
    manager.load()
    return SemiRestorePipeline(manager)


def integrated_app(
    *,
    settings: RuntimeSettings | None = None,
    model: IntegrationModel | None = None,
) -> tuple[object, IntegrationModel, list[int], list[SemiRestorePipeline]]:
    runtime_settings = settings or RuntimeSettings(device_preference="cpu")
    controlled_model = model or IntegrationModel()
    load_calls: list[int] = []
    pipelines: list[SemiRestorePipeline] = []

    def factory(**_: object) -> SemiRestorePipeline:
        pipeline = build_pipeline(controlled_model, load_calls)
        pipelines.append(pipeline)
        return pipeline

    service = SemiRestoreModelService(runtime_settings, pipeline_factory=factory)
    app = create_app(
        settings=runtime_settings,
        model_service_factory=lambda _: service,
    )
    return app, controlled_model, load_calls, pipelines


def capture_logs(app: object) -> StringIO:
    stream = StringIO()
    configure_application_logging(app.state.runtime.settings, stream=stream)  # type: ignore[attr-defined]
    return stream


def test_multipart_to_real_boundary_png_observability_and_nonpersistence(
    tmp_path: Path,
) -> None:
    app, model, load_calls, pipelines = integrated_app()
    stream = capture_logs(app)
    unsafe_filename = str(tmp_path / "private" / "best.pt")

    with TestClient(app) as client:
        ready = client.get("/health/ready")
        response = client.post(
            "/api/v1/restore",
            files=upload(filename=unsafe_filename),
            headers={"x-request-id": "integration-restore"},
        )
        metrics = client.get("/metrics")

    assert ready.status_code == 200
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "integration-restore"
    body = response.json()
    restored_bytes = base64.b64decode(body["image"]["content"])
    assert restored_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert body["image"]["media_type"] == "image/png"
    assert (body["image"]["width"], body["image"]["height"]) == (20, 16)
    assert body["diagnostics"]["input"]
    assert body["diagnostics"]["restored"]
    assert body["diagnostics"]["quality_indicators"][
        "dimension_contract_satisfied"
    ] is True
    assert body["diagnostics"]["suitability"]["advisory_not_probability"] is True
    assert body["model"]["checkpoint_checksum"] == "c" * 64
    assert body["warnings"]
    assert load_calls == [1]
    assert model.forward_calls == 1
    assert len(pipelines) == 1
    assert pipelines[0].status().state is ModelManagerState.CLOSED
    assert list(tmp_path.rglob("*")) == []

    assert 'outcome="success"' in metrics.text
    assert 'route="/api/v1/restore"' in metrics.text
    assert "integration-restore" not in metrics.text
    log_text = stream.getvalue()
    records = [json.loads(line) for line in log_text.splitlines() if line]
    assert any(
        record["event"] == "inference_completed"
        and record["request_id"] == "integration-restore"
        and record["outcome"] == "success"
        for record in records
    )
    for unsafe in (
        unsafe_filename,
        str(tmp_path),
        "best.pt",
        body["image"]["content"],
        "tensor([",
        "array(",
    ):
        assert unsafe not in log_text


def test_startup_and_missing_checkpoint_fail_closed_without_path_leak(tmp_path: Path) -> None:
    missing = tmp_path / "private" / "checkpoint.pt"
    settings = RuntimeSettings(environment="production", checkpoint_path=missing)
    app = create_app(settings=settings)
    stream = capture_logs(app)

    with TestClient(app) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")
        model = client.get("/health/model")
        restoration = client.post("/api/v1/restore", files=upload())

    assert live.status_code == 200
    assert ready.status_code == 503
    assert ready.json()["state"] == "unavailable"
    assert model.json()["ready"] is False
    assert model.json()["checkpoint_checksum"] is None
    assert restoration.status_code == 503
    assert restoration.json()["error"]["code"] == "model_unavailable"
    for rendered in (ready.text, model.text, restoration.text, stream.getvalue()):
        assert str(tmp_path) not in rendered
        assert "checkpoint.pt" not in rendered
        assert "Traceback" not in rendered


def test_malformed_and_oversized_uploads_never_reach_model() -> None:
    encoded = png_bytes()
    settings = RuntimeSettings(
        device_preference="cpu",
        max_encoded_upload_bytes=len(encoded) - 1,
    )
    app, model, _, _ = integrated_app(settings=settings)

    with TestClient(app) as client:
        malformed = client.post(
            "/api/v1/restore-and-analyze",
            files=upload(b"not an image"),
        )
        oversized = client.post(
            "/api/v1/restore",
            files=upload(encoded),
        )

    assert malformed.status_code == 422
    assert malformed.json()["error"]["code"] == "invalid_image"
    assert oversized.status_code == 413
    assert oversized.json()["error"]["code"] == "upload_too_large"
    assert model.forward_calls == 0


def test_real_adapter_backpressure_and_shutdown_are_deterministic() -> None:
    release = threading.Event()
    model = IntegrationModel(release=release)
    settings = RuntimeSettings(
        device_preference="cpu",
        inference_concurrency_limit=1,
        concurrency_acquisition_timeout_seconds=0.02,
        inference_timeout_seconds=2.0,
    )
    app, _, load_calls, pipelines = integrated_app(settings=settings, model=model)

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(client.post, "/api/v1/restore", files=upload())
        assert model.forward_started.wait(timeout=2)
        try:
            live = client.get("/health/live")
            busy = client.post(
                "/api/v1/restore",
                files=upload(),
                headers={"x-request-id": "integration-busy"},
            )
        finally:
            release.set()
        success = first.result(timeout=3)

    assert live.status_code == 200
    assert success.status_code == 200
    assert busy.status_code == 503
    assert busy.json()["error"]["code"] == "inference_busy"
    assert busy.json()["error"]["request_id"] == "integration-busy"
    assert model.forward_calls == 1
    assert model.maximum_active == 1
    assert load_calls == [1]
    assert pipelines[0].status().state is ModelManagerState.CLOSED


@pytest.mark.local_checkpoint
def test_verified_local_checkpoint_through_production_api() -> None:
    if not DEFAULT_CHECKPOINT_PATH.is_file():
        pytest.skip("verified ignored runtime checkpoint is unavailable")
    expected = load_checkpoint_metadata()
    app = create_app(settings=RuntimeSettings(device_preference="cpu"))

    with TestClient(app) as client:
        ready = client.get("/health/ready")
        response = client.post("/api/v1/restore", files=upload())

    assert ready.status_code == 200
    assert response.status_code == 200
    assert response.json()["image"]["media_type"] == "image/png"
    assert response.json()["model"]["checkpoint_checksum"] == expected.sha256
