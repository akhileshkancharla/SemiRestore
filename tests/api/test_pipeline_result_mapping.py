from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from fastapi.testclient import TestClient
from PIL import Image
from torch import nn
from torch.nn import functional as F

from semirestore.api import create_app
from semirestore.checkpoints import LoadedCheckpoint
from semirestore.model_manager import ModelManager
from semirestore.pipeline import SemiRestorePipeline
from semirestore.platform import RuntimeSettings, SemiRestoreModelService


class ControlledPipelineModel(nn.Module):
    statistics_conditioning = True
    padder_size = 1
    scale = 2

    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.9))
        self.forward_calls = 0

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        conditioning_statistics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del conditioning_statistics
        self.forward_calls += 1
        return (
            F.interpolate(inputs, scale_factor=2, mode="bilinear", align_corners=False)
            * self.gain
        )


def controlled_pipeline(
    model: ControlledPipelineModel,
    load_calls: list[int],
) -> SemiRestorePipeline:
    def loader(**_: object) -> LoadedCheckpoint:
        load_calls.append(1)
        return LoadedCheckpoint(
            model=model,
            device=torch.device("cpu"),
            checkpoint_path=Path("controlled-checkpoint.pt"),
            checkpoint_sha256="b" * 64,
            architecture="statistics-conditioned NAF-SR",
            model_name="naf_sr",
            parameter_count=sum(parameter.numel() for parameter in model.parameters()),
            model_version="controlled-api-v1",
            training_revision="controlled-api-revision",
        )

    manager = ModelManager(loader=loader, device="cpu")
    manager.load()
    return SemiRestorePipeline(manager)


def image_upload() -> dict[str, tuple[str, bytes, str]]:
    values = np.linspace(12, 240, 8 * 10, dtype=np.uint8).reshape(8, 10)
    output = BytesIO()
    Image.fromarray(values, mode="L").save(output, format="PNG")
    return {"image": ("controlled.png", output.getvalue(), "image/png")}


def controlled_app() -> tuple[object, ControlledPipelineModel, list[int], list[int]]:
    model = ControlledPipelineModel()
    load_calls: list[int] = []
    factory_calls: list[int] = []

    def factory(**_: object) -> SemiRestorePipeline:
        factory_calls.append(1)
        return controlled_pipeline(model, load_calls)

    settings = RuntimeSettings(device_preference="cpu")
    service = SemiRestoreModelService(settings, pipeline_factory=factory)
    app = create_app(settings=settings, model_service_factory=lambda _: service)
    return app, model, load_calls, factory_calls


def test_restore_maps_actual_pipeline_result_to_versioned_contract() -> None:
    app, model, load_calls, factory_calls = controlled_app()

    with TestClient(app) as client:
        response = client.post("/api/v1/restore", files=image_upload())

    assert response.status_code == 200
    body = response.json()
    assert body["image"]["media_type"] == "image/png"
    assert body["image"]["width"] == 20
    assert body["image"]["height"] == 16
    decoded = base64.b64decode(body["image"]["content"])
    assert decoded.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(BytesIO(decoded)) as restored:
        assert restored.format == "PNG"
        assert restored.size == (20, 16)
    assert body["input"] == {"width": 10, "height": 8, "media_type": "image/png"}
    assert body["model"] == {
        "name": "naf_sr",
        "version": "controlled-api-v1",
        "training_revision": "controlled-api-revision",
        "checkpoint_checksum": "b" * 64,
    }
    assert body["inference"]["device"] == "cpu"
    assert body["inference"]["latency_ms"] >= 0
    assert body["inference"]["phase_latency_ms"]["restoration_total"] >= 0
    assert set(body["diagnostics"]) == {
        "pipeline_version",
        "input",
        "restored",
        "suitability",
        "restoration",
        "spatial",
        "tiles",
        "clipping",
        "quality_indicators",
        "timing_ms",
        "limitations",
    }
    assert body["diagnostics"]["quality_indicators"][
        "dimension_contract_satisfied"
    ] is True
    assert body["diagnostics"]["quality_indicators"][
        "can_prove_reconstruction_correctness"
    ] is False
    assert body["diagnostics"]["suitability"]["advisory_not_probability"] is True
    assert body["warnings"]
    assert factory_calls == [1]
    assert load_calls == [1]
    assert model.forward_calls == 1
    for unsafe in ("array(", "tensor([", "controlled-checkpoint.pt", "Traceback"):
        assert unsafe not in response.text


def test_analyze_and_restore_and_analyze_use_same_real_adapter_instance() -> None:
    app, model, load_calls, factory_calls = controlled_app()

    with TestClient(app) as client:
        analysis = client.post("/api/v1/analyze", files=image_upload())
        combined = client.post("/api/v1/restore-and-analyze", files=image_upload())

    assert analysis.status_code == 200
    analysis_body = analysis.json()
    assert analysis_body["input"] == {
        "width": 10,
        "height": 8,
        "media_type": "image/png",
    }
    assert analysis_body["analysis"]["latency_ms"] >= 0
    assert set(analysis_body["diagnostics"]) == {
        "preprocessing",
        "intensity",
        "structure",
    }
    assert analysis_body["suitability"]["recommendation"] in {
        "restore",
        "warn",
        "bypass",
    }
    assert analysis_body["suitability"]["advisory_not_probability"] is True
    assert combined.status_code == 200
    assert combined.json()["image"]["media_type"] == "image/png"
    assert factory_calls == [1]
    assert load_calls == [1]
    assert model.forward_calls == 1


def test_model_routes_report_missing_checkpoint_without_leaking_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "private" / "best.pt"
    app = create_app(settings=RuntimeSettings(checkpoint_path=missing_path))

    with TestClient(app) as client:
        responses = [
            client.post("/api/v1/analyze", files=image_upload()),
            client.post("/api/v1/restore", files=image_upload()),
            client.post("/api/v1/restore-and-analyze", files=image_upload()),
        ]

    for response in responses:
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "model_unavailable"
        assert response.json()["error"]["request_id"] == response.headers["x-request-id"]
        assert str(tmp_path) not in response.text
        assert "best.pt" not in response.text
