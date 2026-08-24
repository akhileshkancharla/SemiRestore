from __future__ import annotations

import io
import json
import threading
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
from semirestore.postprocessing import PostprocessingError

REAL_RUNTIME_CHECKPOINT = Path("artifacts/model/semirestore_conditioned.pt")
REAL_SHA256 = "273abd9d6dcfa9bdee71ac15016994962304b6c9d902898b4f4d503bed158c28"


class ControlledUpscaleModel(nn.Module):
    scale = 2

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = 0
        self.last_dtype: torch.dtype | None = None
        self.last_device: torch.device | None = None
        self.grad_enabled: bool | None = None
        self.inference_mode_enabled: bool | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.last_dtype = inputs.dtype
        self.last_device = inputs.device
        self.grad_enabled = torch.is_grad_enabled()
        self.inference_mode_enabled = torch.is_inference_mode_enabled()
        return F.interpolate(inputs, scale_factor=2, mode="nearest")


class ControlledLoader:
    def __init__(self, model: nn.Module | None = None) -> None:
        self.model = model or ControlledUpscaleModel()
        self.calls = 0

    def __call__(self, **_kwargs: object) -> checkpoints.LoadedCheckpoint:
        self.calls += 1
        return checkpoints.LoadedCheckpoint(
            model=self.model,
            device=torch.device("cpu"),
            checkpoint_path=Path("artifacts/model/synthetic.pt"),
            checkpoint_sha256="b" * 64,
            architecture="statistics-conditioned NAF-SR",
            model_name="naf_sr",
            parameter_count=sum(parameter.numel() for parameter in self.model.parameters()),
            model_version="synthetic-v1",
            training_revision="synthetic-revision",
        )


def _ready_service(
    *,
    model: nn.Module | None = None,
    max_pixels: int = restoration_service.DEFAULT_DIRECT_INFERENCE_MAX_PIXELS,
    **service_kwargs: object,
) -> tuple[
    restoration_service.SingleImageRestorationService,
    model_manager.ModelManager,
    ControlledLoader,
]:
    loader = ControlledLoader(model)
    manager = model_manager.ModelManager(loader=loader)
    manager.load()
    service = restoration_service.SingleImageRestorationService(
        manager,
        max_direct_input_pixels=max_pixels,
        **service_kwargs,
    )
    return service, manager, loader


def _png_bytes(array: np.ndarray) -> bytes:
    encoded = io.BytesIO()
    Image.fromarray(array).save(encoded, format="PNG")
    return encoded.getvalue()


def test_successful_cpu_restoration_has_exact_two_x_dimensions() -> None:
    service, _manager, loader = _ready_service()
    image = np.array([[0, 64, 255], [32, 128, 192]], dtype=np.uint8)

    result = service.restore(image)

    assert result.original_width == 3
    assert result.original_height == 2
    assert result.restored_width == 6
    assert result.restored_height == 4
    assert result.restored_image.shape == (4, 6)
    assert result.restored_image.dtype == np.float32
    assert result.restored_image.flags.c_contiguous
    assert result.scale_factor == 2
    assert loader.model.calls == 1


@pytest.mark.parametrize("source_type", ["path", "bytes", "numpy", "pil"])
def test_all_preprocessing_input_types_flow_through_service(
    source_type: str,
    tmp_path: Path,
) -> None:
    service, _manager, _loader = _ready_service()
    array = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    content = _png_bytes(array)
    if source_type == "path":
        source: object = tmp_path / "input.png"
        source.write_bytes(content)  # type: ignore[union-attr]
    elif source_type == "bytes":
        source = content
    elif source_type == "numpy":
        source = array
    else:
        source = Image.fromarray(array)

    result = service.restore(source)  # type: ignore[arg-type]

    assert result.preprocessing_metadata["source_type"] == source_type
    assert (result.restored_width, result.restored_height) == (4, 4)


@pytest.mark.parametrize(
    ("bit_depth", "mode", "dtype"),
    [(8, "L", np.uint8), (16, "I;16", np.uint16)],
)
def test_explicit_png_bit_depth_round_trip(
    bit_depth: int,
    mode: str,
    dtype: np.dtype[object],
) -> None:
    service, _manager, _loader = _ready_service()

    result = service.restore(
        np.array([[0.0, 0.5]], dtype=np.float32),
        output_bit_depth=bit_depth,
    )

    assert result.media_type == "image/png"
    assert result.png_bit_depth == bit_depth
    with Image.open(io.BytesIO(result.png_bytes)) as decoded:
        decoded_array = np.array(decoded)
        assert decoded.mode == mode
        assert decoded.size == (4, 2)
    assert decoded_array.dtype == dtype


def test_default_output_is_16_bit_png() -> None:
    service, _manager, _loader = _ready_service()

    result = service.restore(np.zeros((1, 1), dtype=np.uint8))

    assert result.png_bit_depth == 16
    with Image.open(io.BytesIO(result.png_bytes)) as decoded:
        assert decoded.mode == "I;16"


def test_model_and_checkpoint_load_are_reused_across_requests() -> None:
    service, manager, loader = _ready_service()
    expected_model = manager.model

    first = service.restore(np.zeros((2, 2), dtype=np.uint8))
    second = service.restore(np.ones((2, 2), dtype=np.float32))

    assert manager.model is expected_model
    assert loader.calls == 1
    assert loader.model.calls == 2
    assert first.model_name == second.model_name == "naf_sr"


def test_preprocessing_and_postprocessing_each_run_exactly_once() -> None:
    preprocessing_calls = 0
    postprocessing_calls = 0

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

    service, _manager, _loader = _ready_service(
        preprocessor=recording_preprocessor,
        postprocessor=recording_postprocessor,
    )

    service.restore(np.zeros((2, 2), dtype=np.uint8))

    assert preprocessing_calls == 1
    assert postprocessing_calls == 1


@pytest.mark.parametrize("state", ["unloaded", "failed", "closed"])
def test_manager_must_be_ready(state: str) -> None:
    if state == "failed":
        def failing_loader(**_kwargs: object) -> checkpoints.LoadedCheckpoint:
            raise checkpoints.CheckpointVerificationError("private checkpoint detail")

        manager = model_manager.ModelManager(loader=failing_loader)
        with pytest.raises(model_manager.ModelManagerLoadError):
            manager.load()
    else:
        manager = model_manager.ModelManager(loader=ControlledLoader())
        if state == "closed":
            manager.close()
    service = restoration_service.SingleImageRestorationService(manager)

    with pytest.raises(restoration_service.ManagerNotReadyError) as error:
        service.restore(np.zeros((1, 1), dtype=np.uint8))

    assert error.value.category is restoration_service.RestorationErrorCategory.MANAGER_NOT_READY


def test_invalid_input_maps_to_safe_category() -> None:
    service, _manager, _loader = _ready_service()

    with pytest.raises(restoration_service.InvalidRestorationInputError) as error:
        service.restore(b"not-an-image")

    assert error.value.category is restoration_service.RestorationErrorCategory.INVALID_INPUT


def test_preprocessing_resource_failure_maps_safely() -> None:
    limits = restoration_service.PreprocessingLimits(max_pixels=4)
    service, _manager, _loader = _ready_service(
        max_pixels=4,
        preprocessing_limits=limits,
    )

    with pytest.raises(restoration_service.RestorationResourceLimitError) as error:
        service.restore(np.zeros((3, 3), dtype=np.uint8))

    assert error.value.category is restoration_service.RestorationErrorCategory.RESOURCE_LIMIT


def test_unexpected_preprocessing_failure_maps_without_details() -> None:
    def failing_preprocessor(*_args: object, **_kwargs: object) -> object:
        raise preprocessing.PreprocessingError("secret decoder detail")

    service, _manager, _loader = _ready_service(preprocessor=failing_preprocessor)

    with pytest.raises(restoration_service.RestorationPreprocessingError) as error:
        service.restore(np.zeros((1, 1), dtype=np.uint8))

    assert error.value.category is restoration_service.RestorationErrorCategory.PREPROCESSING
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("bit_depth", [1, 12, 32, 16.0, True])
def test_invalid_output_bit_depth_is_rejected(bit_depth: object) -> None:
    service, _manager, loader = _ready_service()

    with pytest.raises(restoration_service.UnsupportedRestorationOutputError):
        service.restore(
            np.zeros((1, 1), dtype=np.uint8),
            output_bit_depth=bit_depth,  # type: ignore[arg-type]
        )

    assert loader.model.calls == 0


@pytest.mark.parametrize("failure", ["shape", "nonfinite"])
def test_invalid_model_output_maps_safely(failure: str) -> None:
    class InvalidOutputModel(ControlledUpscaleModel):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            if failure == "shape":
                return inputs
            return torch.full(
                (1, 1, inputs.shape[-2] * 2, inputs.shape[-1] * 2),
                float("nan"),
            )

    service, _manager, _loader = _ready_service(model=InvalidOutputModel())

    with pytest.raises(restoration_service.InvalidModelOutputError) as error:
        service.restore(np.zeros((2, 2), dtype=np.uint8))

    assert error.value.category is restoration_service.RestorationErrorCategory.INVALID_MODEL_OUTPUT


def test_model_exception_is_mapped_without_internal_details() -> None:
    class FailingModel(ControlledUpscaleModel):
        def forward(self, _inputs: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("secret tensor and device details")

    service, _manager, _loader = _ready_service(model=FailingModel())

    with pytest.raises(restoration_service.RestorationInferenceError) as error:
        service.restore(np.zeros((1, 1), dtype=np.uint8))

    assert error.value.category is restoration_service.RestorationErrorCategory.MODEL_INFERENCE
    assert "secret" not in str(error.value)


def test_postprocessing_exception_is_mapped_without_details() -> None:
    def failing_postprocessor(*_args: object, **_kwargs: object) -> object:
        raise PostprocessingError("secret output detail")

    service, _manager, _loader = _ready_service(postprocessor=failing_postprocessor)

    with pytest.raises(restoration_service.RestorationPostprocessingError) as error:
        service.restore(np.zeros((1, 1), dtype=np.uint8))

    assert error.value.category is restoration_service.RestorationErrorCategory.POSTPROCESSING
    assert "secret" not in str(error.value)


def test_caller_numpy_input_remains_unchanged() -> None:
    service, _manager, _loader = _ready_service()
    image = np.array([[0.0, 0.5], [0.75, 1.0]], dtype=np.float32)
    original = image.copy()

    result = service.restore(image)
    result.restored_image.fill(0.0)

    np.testing.assert_array_equal(image, original)


def test_inference_uses_fp32_device_tensor_without_gradients() -> None:
    model = ControlledUpscaleModel()
    service, _manager, _loader = _ready_service(model=model)

    service.restore(np.ones((2, 2), dtype=np.float64))

    assert model.last_dtype == torch.float32
    assert model.last_device == torch.device("cpu")
    assert model.grad_enabled is False
    assert model.inference_mode_enabled is True


def test_result_contains_complete_serialization_friendly_metadata() -> None:
    service, _manager, _loader = _ready_service()

    result = service.restore(np.array([[0, 255]], dtype=np.uint8))
    metadata = result.metadata()

    assert metadata["media_type"] == "image/png"
    assert metadata["png_bit_depth"] == 16
    assert metadata["original_width"] == 2
    assert metadata["restored_width"] == 4
    assert metadata["scale_factor"] == 2
    assert metadata["resolved_device"] == "cpu"
    assert metadata["model_name"] == "naf_sr"
    assert metadata["model_version"] == "synthetic-v1"
    assert metadata["training_revision"] == "synthetic-revision"
    assert metadata["checkpoint_sha256"] == "b" * 64
    assert isinstance(metadata["preprocessing"], dict)
    assert isinstance(metadata["postprocessing"], dict)
    json.dumps(metadata, allow_nan=False)


def test_phase_timings_are_nonnegative_and_consistent() -> None:
    service, _manager, _loader = _ready_service()

    result = service.restore(np.zeros((2, 2), dtype=np.uint8))

    phases = (
        result.preprocessing_latency_ms,
        result.device_transfer_latency_ms,
        result.inference_wait_latency_ms,
        result.model_inference_latency_ms,
        result.postprocessing_latency_ms,
    )
    assert all(value >= 0.0 for value in phases)
    assert result.total_latency_ms >= 0.0
    assert result.total_latency_ms + 1e-9 >= sum(phases)


def test_direct_inference_pixel_limit_rejects_before_model_execution() -> None:
    service, _manager, loader = _ready_service(max_pixels=16)

    with pytest.raises(restoration_service.RestorationResourceLimitError) as error:
        service.restore(np.zeros((5, 4), dtype=np.uint8))

    assert "tiled inference" in str(error.value)
    assert loader.model.calls == 0


def test_concurrent_calls_serialize_model_execution() -> None:
    class SynchronizationProbeModel(ControlledUpscaleModel):
        def __init__(self) -> None:
            super().__init__()
            self.active = 0
            self.maximum_active = 0
            self.state_lock = threading.Lock()
            self.first_entered = threading.Event()
            self.second_entered = threading.Event()
            self.release_first = threading.Event()

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            with self.state_lock:
                self.calls += 1
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
                if self.calls == 1:
                    self.first_entered.set()
                else:
                    self.second_entered.set()
            if self.calls == 1 and not self.release_first.wait(timeout=5):
                raise RuntimeError("test synchronization timed out")
            output = F.interpolate(inputs, scale_factor=2, mode="nearest")
            with self.state_lock:
                self.active -= 1
            return output

    model = SynchronizationProbeModel()
    service, _manager, _loader = _ready_service(model=model)
    image = np.zeros((2, 2), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(service.restore, image)
        assert model.first_entered.wait(timeout=5)
        second = executor.submit(service.restore, image)
        assert not model.second_entered.wait(timeout=0.2)
        model.release_first.set()
        first.result(timeout=5)
        second.result(timeout=5)

    assert model.calls == 2
    assert model.maximum_active == 1


def test_service_creates_no_temporary_input_or_output_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _manager, _loader = _ready_service()
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.iterdir())

    service.restore(_png_bytes(np.array([[0, 255]], dtype=np.uint8)))

    assert set(tmp_path.iterdir()) == before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_restoration_uses_selected_device() -> None:
    model = ControlledUpscaleModel().to("cuda")

    class CudaLoader(ControlledLoader):
        def __call__(self, **_kwargs: object) -> checkpoints.LoadedCheckpoint:
            loaded = super().__call__()
            return checkpoints.LoadedCheckpoint(
                model=loaded.model,
                device=torch.device("cuda:0"),
                checkpoint_path=loaded.checkpoint_path,
                checkpoint_sha256=loaded.checkpoint_sha256,
                architecture=loaded.architecture,
                model_name=loaded.model_name,
                parameter_count=loaded.parameter_count,
                model_version=loaded.model_version,
                training_revision=loaded.training_revision,
            )

    loader = CudaLoader(model)
    manager = model_manager.ModelManager(device="cuda:0", loader=loader)
    manager.load()
    service = restoration_service.SingleImageRestorationService(manager)

    result = service.restore(np.zeros((2, 2), dtype=np.uint8))

    assert model.last_device == torch.device("cuda:0")
    assert result.resolved_device == "cuda:0"


@pytest.mark.local_checkpoint
def test_real_checkpoint_restores_small_synthetic_image() -> None:
    if not REAL_RUNTIME_CHECKPOINT.is_file():
        pytest.skip("verified ignored runtime checkpoint is unavailable")
    manager = model_manager.ModelManager(device="cpu")
    manager.load()
    service = restoration_service.SingleImageRestorationService(manager)
    image = np.linspace(0.0, 1.0, 8 * 8, dtype=np.float32).reshape(8, 8)

    result = service.restore(image)

    assert result.original_width == 8
    assert result.original_height == 8
    assert result.restored_width == 16
    assert result.restored_height == 16
    assert result.restored_image.shape == (16, 16)
    assert result.restored_image.dtype == np.float32
    assert np.isfinite(result.restored_image).all()
    assert float(result.restored_image.min()) >= 0.0
    assert float(result.restored_image.max()) <= 1.0
    assert result.media_type == "image/png"
    assert result.png_bit_depth == 16
    assert result.checkpoint_sha256 == REAL_SHA256
    assert result.model_name == "naf_sr"
    assert result.model_version == "conditioned-d037473"
    with Image.open(io.BytesIO(result.png_bytes)) as decoded:
        assert decoded.mode == "I;16"
        assert decoded.size == (16, 16)

    manager.close()
