from __future__ import annotations

import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from semirestore import (
    checkpoints,
    model_manager,
    postprocessing,
    preprocessing,
    restoration_service,
)
from semirestore.models import compute_conditioning_statistics
from semirestore.models.naf_sr import ConditioningStatisticsError

REAL_RUNTIME_CHECKPOINT = Path("artifacts/model/semirestore_conditioned.pt")
REAL_SHA256 = "273abd9d6dcfa9bdee71ac15016994962304b6c9d902898b4f4d503bed158c28"


class TileAwareModel(nn.Module):
    scale = 2
    padder_size = 8

    def __init__(self, *, constant: float | None = None, expanded_range: bool = False) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.constant = constant
        self.expanded_range = expanded_range
        self.calls = 0
        self.observed_statistics: list[torch.Tensor] = []
        self.observed_shapes: list[tuple[int, ...]] = []

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        conditioning_statistics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.calls += 1
        self.observed_shapes.append(tuple(inputs.shape))
        if conditioning_statistics is None:
            raise RuntimeError("global conditioning override is required")
        self.observed_statistics.append(conditioning_statistics.detach().cpu().clone())
        shape = (inputs.shape[0], 1, inputs.shape[-2] * 2, inputs.shape[-1] * 2)
        if self.constant is not None:
            return torch.full(shape, self.constant, dtype=inputs.dtype, device=inputs.device)
        output = F.interpolate(inputs, scale_factor=2, mode="nearest")
        return output * 3.0 - 1.0 if self.expanded_range else output


class TileLoader:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.calls = 0

    def __call__(self, **_kwargs: object) -> checkpoints.LoadedCheckpoint:
        self.calls += 1
        return checkpoints.LoadedCheckpoint(
            model=self.model,
            device=next(self.model.parameters()).device,
            checkpoint_path=Path("artifacts/model/synthetic.pt"),
            checkpoint_sha256="c" * 64,
            architecture="statistics-conditioned NAF-SR",
            model_name="naf_sr",
            parameter_count=sum(parameter.numel() for parameter in self.model.parameters()),
            model_version="synthetic-v1",
            training_revision="synthetic-revision",
        )


def _service(
    model: nn.Module | None = None,
    **kwargs: object,
) -> tuple[restoration_service.SingleImageRestorationService, TileLoader]:
    controlled = model or TileAwareModel()
    loader = TileLoader(controlled)
    manager = model_manager.ModelManager(loader=loader)
    manager.load()
    return restoration_service.SingleImageRestorationService(manager, **kwargs), loader


def test_tiled_service_preprocesses_and_computes_global_statistics_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = TileAwareModel()
    preprocessing_calls = 0
    postprocessing_calls = 0
    statistics_calls = 0

    def recording_preprocessor(
        *args: object,
        **kwargs: object,
    ) -> preprocessing.PreprocessingResult:
        nonlocal preprocessing_calls
        preprocessing_calls += 1
        return preprocessing.preprocess_sem_image(*args, **kwargs)  # type: ignore[arg-type]

    def recording_postprocessor(
        *args: object,
        **kwargs: object,
    ) -> postprocessing.PostprocessingResult:
        nonlocal postprocessing_calls
        postprocessing_calls += 1
        return postprocessing.postprocess_restoration(*args, **kwargs)  # type: ignore[arg-type]

    real_statistics = restoration_service.compute_conditioning_statistics

    def recording_statistics(inputs: torch.Tensor) -> torch.Tensor:
        nonlocal statistics_calls
        statistics_calls += 1
        return real_statistics(inputs)

    monkeypatch.setattr(
        restoration_service,
        "compute_conditioning_statistics",
        recording_statistics,
    )
    service, loader = _service(
        model,
        preprocessor=recording_preprocessor,
        postprocessor=recording_postprocessor,
    )
    image = np.linspace(0.0, 1.0, 20 * 20, dtype=np.float32).reshape(20, 20)

    result = service.restore_tiled(image, tile_size=16, overlap=4)

    assert preprocessing_calls == 1
    assert statistics_calls == 1
    assert postprocessing_calls == 1
    assert result.tiled_metadata is not None
    assert result.tiled_metadata.plan_summary["tile_count"] == 4
    assert model.calls == 4
    assert loader.calls == 1
    expected = compute_conditioning_statistics(torch.from_numpy(image)[None, None])
    assert len(model.observed_statistics) == 4
    for observed in model.observed_statistics:
        torch.testing.assert_close(observed, expected, atol=0, rtol=0)


def test_constant_tile_outputs_blend_without_seams() -> None:
    service, _loader = _service(TileAwareModel(constant=0.375))

    result = service.restore_tiled(
        np.zeros((20, 20), dtype=np.uint8),
        tile_size=16,
        overlap=4,
    )

    np.testing.assert_allclose(result.restored_image, np.float32(0.375), atol=1e-7, rtol=0)
    assert result.restored_image.shape == (40, 40)


def test_tile_outputs_are_not_clipped_before_one_global_postprocess() -> None:
    service, _loader = _service(TileAwareModel(expanded_range=True))
    image = np.linspace(0.0, 1.0, 20 * 20, dtype=np.float32).reshape(20, 20)

    result = service.restore_tiled(image, tile_size=16, overlap=4)

    assert result.postprocessing_metadata["raw_minimum"] == pytest.approx(-1.0)
    assert result.postprocessing_metadata["raw_maximum"] == pytest.approx(2.0)
    assert result.postprocessing_metadata["clipping_occurred"] is True
    assert float(result.restored_image.min()) == 0.0
    assert float(result.restored_image.max()) == 1.0


@pytest.mark.parametrize(("bit_depth", "mode"), [(8, "L"), (16, "I;16")])
def test_tiled_png_round_trip_preserves_exact_dimensions(bit_depth: int, mode: str) -> None:
    service, _loader = _service()

    result = service.restore_tiled(
        np.zeros((17, 19), dtype=np.uint8),
        tile_size=16,
        overlap=4,
        output_bit_depth=bit_depth,
    )

    assert result.png_bit_depth == bit_depth
    assert result.restored_image.shape == (34, 38)
    with Image.open(io.BytesIO(result.png_bytes)) as decoded:
        assert decoded.mode == mode
        assert decoded.size == (38, 34)


def test_tiled_metadata_and_timings_are_complete() -> None:
    service, _loader = _service()

    result = service.restore_tiled(
        np.zeros((20, 20), dtype=np.uint8),
        tile_size=16,
        overlap=4,
    )

    tiled = result.tiled_metadata
    assert tiled is not None
    assert tiled.global_conditioning_reused is True
    assert tiled.plan_summary["blending_method"] == "separable_linear_overlap_ramp"
    assert tiled.plan_summary["max_padded_pixels_per_tile"] == 256
    timings = (
        tiled.preprocessing_latency_ms,
        tiled.planning_latency_ms,
        tiled.cumulative_lock_wait_latency_ms,
        tiled.cumulative_transfer_latency_ms,
        tiled.cumulative_model_latency_ms,
        tiled.assembly_blending_latency_ms,
        tiled.postprocessing_latency_ms,
        tiled.total_latency_ms,
    )
    assert all(value >= 0.0 for value in timings)
    assert result.metadata()["tiled"] == tiled.to_dict()


def test_per_tile_resource_limit_is_mapped_before_model_execution() -> None:
    model = TileAwareModel()
    service, _loader = _service(model)

    with pytest.raises(restoration_service.RestorationTileResourceError):
        service.restore_tiled(
            np.zeros((20, 20), dtype=np.uint8),
            tile_size=16,
            overlap=4,
            max_padded_pixels_per_tile=255,
        )

    assert model.calls == 0


def test_global_conditioning_rejection_maps_safely() -> None:
    class RejectingModel(TileAwareModel):
        def forward(
            self,
            inputs: torch.Tensor,
            *,
            conditioning_statistics: torch.Tensor | None = None,
        ) -> torch.Tensor:
            raise ConditioningStatisticsError("secret override details")

    service, _loader = _service(RejectingModel())

    with pytest.raises(restoration_service.GlobalConditioningOverrideError):
        service.restore_tiled(np.zeros((8, 8), dtype=np.uint8), tile_size=8, overlap=2)


def test_tile_inference_exception_maps_safely() -> None:
    class FailingModel(TileAwareModel):
        def forward(
            self,
            inputs: torch.Tensor,
            *,
            conditioning_statistics: torch.Tensor | None = None,
        ) -> torch.Tensor:
            raise RuntimeError("secret device failure")

    service, _loader = _service(FailingModel())

    with pytest.raises(restoration_service.RestorationTileInferenceError) as error:
        service.restore_tiled(np.zeros((8, 8), dtype=np.uint8), tile_size=8, overlap=2)

    assert "secret" not in str(error.value)


@pytest.mark.parametrize("failure", ["shape", "nonfinite"])
def test_malformed_or_nonfinite_tile_output_maps_safely(failure: str) -> None:
    class InvalidOutputModel(TileAwareModel):
        def forward(
            self,
            inputs: torch.Tensor,
            *,
            conditioning_statistics: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if failure == "shape":
                return inputs
            return torch.full(
                (1, 1, inputs.shape[-2] * 2, inputs.shape[-1] * 2),
                float("nan"),
            )

    service, _loader = _service(InvalidOutputModel())

    with pytest.raises(restoration_service.InvalidTileOutputError):
        service.restore_tiled(np.zeros((8, 8), dtype=np.uint8), tile_size=8, overlap=2)


def test_zero_blending_weights_reject_partial_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _loader = _service()

    def zero_weights(
        _plan: object,
        tile: object,
    ) -> np.ndarray:
        height = tile.height * 2  # type: ignore[attr-defined]
        width = tile.width * 2  # type: ignore[attr-defined]
        return np.zeros((height, width), dtype=np.float32)

    monkeypatch.setattr(restoration_service, "blending_weights", zero_weights)

    with pytest.raises(restoration_service.RestorationTileAssemblyError):
        service.restore_tiled(np.zeros((8, 8), dtype=np.uint8), tile_size=8, overlap=2)


def test_nonfinite_assembled_output_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    maximum = torch.finfo(torch.float32).max
    service, _loader = _service(TileAwareModel(constant=maximum))

    def unit_weights(
        _plan: object,
        tile: object,
    ) -> np.ndarray:
        height = tile.height * 2  # type: ignore[attr-defined]
        width = tile.width * 2  # type: ignore[attr-defined]
        return np.ones((height, width), dtype=np.float32)

    monkeypatch.setattr(restoration_service, "blending_weights", unit_weights)

    with pytest.raises(restoration_service.RestorationTileAssemblyError):
        service.restore_tiled(np.zeros((12, 12), dtype=np.uint8), tile_size=8, overlap=4)


def test_tiled_postprocessing_failure_maps_safely() -> None:
    def failing_postprocessor(*_args: object, **_kwargs: object) -> object:
        raise postprocessing.PostprocessingError("secret assembled output")

    service, _loader = _service(postprocessor=failing_postprocessor)

    with pytest.raises(restoration_service.RestorationPostprocessingError) as error:
        service.restore_tiled(np.zeros((8, 8), dtype=np.uint8), tile_size=8, overlap=2)

    assert "secret" not in str(error.value)


@pytest.mark.parametrize(
    ("tile_size", "overlap", "max_tiles", "error_type"),
    [
        (7, 2, 100, restoration_service.InvalidTilePlanError),
        (16, 16, 100, restoration_service.InvalidTilePlanError),
        (8, 7, 2, restoration_service.RestorationTileCountError),
    ],
)
def test_invalid_tiled_configuration_maps_safely(
    tile_size: int,
    overlap: int,
    max_tiles: int,
    error_type: type[restoration_service.RestorationServiceError],
) -> None:
    service, _loader = _service()

    with pytest.raises(error_type):
        service.restore_tiled(
            np.zeros((20, 20), dtype=np.uint8),
            tile_size=tile_size,
            overlap=overlap,
            max_tile_count=max_tiles,
        )


def test_tiled_caller_input_unchanged_and_no_files_created(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _loader = _service()
    image = np.linspace(0.0, 1.0, 20 * 20, dtype=np.float32).reshape(20, 20)
    original = image.copy()
    monkeypatch.chdir(tmp_path)

    result = service.restore_tiled(image, tile_size=16, overlap=4)
    result.restored_image.fill(0.0)

    np.testing.assert_array_equal(image, original)
    assert list(tmp_path.iterdir()) == []


def test_direct_restore_remains_non_tiled_and_unchanged() -> None:
    class DirectCompatibleModel(TileAwareModel):
        def forward(
            self,
            inputs: torch.Tensor,
            *,
            conditioning_statistics: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if conditioning_statistics is None:
                self.calls += 1
                return F.interpolate(inputs, scale_factor=2, mode="nearest")
            return super().forward(inputs, conditioning_statistics=conditioning_statistics)

    model = DirectCompatibleModel()
    service, _loader = _service(model)
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)

    result = service.restore(image)

    expected = np.repeat(np.repeat(image.astype(np.float32) / 255.0, 2, axis=0), 2, axis=1)
    np.testing.assert_array_equal(result.restored_image, expected)
    assert result.tiled_metadata is None
    assert model.calls == 1


def test_concurrent_tiled_calls_preserve_forward_serialization() -> None:
    class ConcurrencyModel(TileAwareModel):
        def __init__(self) -> None:
            super().__init__()
            self.active = 0
            self.maximum_active = 0
            self.state_lock = threading.Lock()

        def forward(
            self,
            inputs: torch.Tensor,
            *,
            conditioning_statistics: torch.Tensor | None = None,
        ) -> torch.Tensor:
            with self.state_lock:
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
            try:
                time.sleep(0.01)
                return super().forward(
                    inputs,
                    conditioning_statistics=conditioning_statistics,
                )
            finally:
                with self.state_lock:
                    self.active -= 1

    model = ConcurrencyModel()
    service, _loader = _service(model)
    image = np.zeros((20, 20), dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(service.restore_tiled, image, tile_size=16, overlap=4)
            for _ in range(2)
        ]
        for future in futures:
            future.result(timeout=10)

    assert model.calls == 8
    assert model.maximum_active == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_tiled_restoration_uses_one_tile_tensor_at_a_time() -> None:
    model = TileAwareModel().to("cuda")
    loader = TileLoader(model)
    manager = model_manager.ModelManager(device="cuda:0", loader=loader)
    manager.load()
    service = restoration_service.SingleImageRestorationService(manager)

    result = service.restore_tiled(
        np.zeros((20, 20), dtype=np.uint8),
        tile_size=16,
        overlap=4,
    )

    assert result.resolved_device == "cuda:0"
    assert model.calls == 4
    assert all(shape[-2] <= 16 and shape[-1] <= 16 for shape in model.observed_shapes)


@pytest.mark.local_checkpoint
def test_real_checkpoint_tiled_comparison_and_seam_measurement() -> None:
    if not REAL_RUNTIME_CHECKPOINT.is_file():
        pytest.skip("verified ignored runtime checkpoint is unavailable")
    manager = model_manager.ModelManager(device="cpu")
    manager.load()
    service = restoration_service.SingleImageRestorationService(manager)
    image = np.linspace(0.0, 1.0, 9 * 11, dtype=np.float32).reshape(9, 11)

    direct = service.restore(image)
    tiled = service.restore_tiled(image, tile_size=8, overlap=4)

    absolute_difference = np.abs(tiled.restored_image - direct.restored_image)
    mean_absolute_difference = float(absolute_difference.mean())
    seam_mask = np.zeros_like(absolute_difference, dtype=bool)
    assert tiled.tiled_metadata is not None
    summary = tiled.tiled_metadata.plan_summary
    for start in summary["row_starts"][1:]:  # type: ignore[index,union-attr]
        boundary = int(start) * 2
        seam_mask[max(0, boundary - 1) : boundary + 2, :] = True
    for start in summary["column_starts"][1:]:  # type: ignore[index,union-attr]
        boundary = int(start) * 2
        seam_mask[:, max(0, boundary - 1) : boundary + 2] = True
    seam_mean_absolute_difference = float(absolute_difference[seam_mask].mean())
    nonseam_mean_absolute_difference = float(absolute_difference[~seam_mask].mean())
    print(
        "tiled_comparison "
        f"mae={mean_absolute_difference:.9f} "
        f"seam_mae={seam_mean_absolute_difference:.9f} "
        f"nonseam_mae={nonseam_mean_absolute_difference:.9f}"
    )

    assert tiled.restored_image.shape == direct.restored_image.shape == (18, 22)
    assert tiled.restored_image.dtype == np.float32
    assert np.isfinite(tiled.restored_image).all()
    assert float(tiled.restored_image.min()) >= 0.0
    assert float(tiled.restored_image.max()) <= 1.0
    assert tiled.checkpoint_sha256 == direct.checkpoint_sha256 == REAL_SHA256
    assert np.isfinite(mean_absolute_difference)
    assert np.isfinite(seam_mean_absolute_difference)
    assert np.isfinite(nonseam_mean_absolute_difference)
    with Image.open(io.BytesIO(tiled.png_bytes)) as decoded:
        assert decoded.mode == "I;16"
        assert decoded.size == (22, 18)

    manager.close()
