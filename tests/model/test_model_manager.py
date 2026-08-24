from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
from torch import nn

from semirestore import checkpoints, model_manager

REAL_RUNTIME_CHECKPOINT = Path("artifacts/model/semirestore_conditioned.pt")
REAL_SHA256 = "273abd9d6dcfa9bdee71ac15016994962304b6c9d902898b4f4d503bed158c28"
CPU_DEVICE = torch.device("cpu")


class TinyManagedModel(nn.Module):
    scale = 2

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)


def _loaded(
    *,
    model: nn.Module | None = None,
    checkpoint_path: Path = Path("artifacts/model/synthetic.pt"),
    device: torch.device = CPU_DEVICE,
) -> checkpoints.LoadedCheckpoint:
    loaded_model = model or TinyManagedModel()
    return checkpoints.LoadedCheckpoint(
        model=loaded_model,
        device=device,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256="a" * 64,
        architecture="statistics-conditioned NAF-SR",
        model_name="naf_sr",
        parameter_count=sum(parameter.numel() for parameter in loaded_model.parameters()),
        model_version="synthetic-v1",
        training_revision="synthetic-revision",
    )


class ControlledLoader:
    def __init__(self, loaded: checkpoints.LoadedCheckpoint | None = None) -> None:
        self.loaded = loaded or _loaded()
        self.calls = 0
        self.error: Exception | None = None
        self.last_kwargs: dict[str, object] | None = None

    def __call__(self, **kwargs: object) -> checkpoints.LoadedCheckpoint:
        self.calls += 1
        self.last_kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.loaded


def test_initial_state_is_unloaded_without_triggering_loader() -> None:
    loader = ControlledLoader()
    manager = model_manager.ModelManager(loader=loader)

    assert manager.state is model_manager.ModelManagerState.UNLOADED
    assert manager.is_ready is False
    assert manager.status().state is model_manager.ModelManagerState.UNLOADED
    assert loader.calls == 0


def test_successful_load_reaches_ready_and_reuses_same_model() -> None:
    loader = ControlledLoader()
    manager = model_manager.ModelManager(loader=loader)

    first = manager.load()
    second = manager.load()

    assert first is loader.loaded.model
    assert second is first
    assert manager.model is first
    assert manager.is_ready is True
    assert manager.state is model_manager.ModelManagerState.READY
    assert loader.calls == 1


def test_concurrent_load_calls_invoke_loader_once() -> None:
    entered = threading.Event()
    release = threading.Event()
    calls = 0
    calls_lock = threading.Lock()
    loaded = _loaded()

    def blocking_loader(**_kwargs: object) -> checkpoints.LoadedCheckpoint:
        nonlocal calls
        with calls_lock:
            calls += 1
        entered.set()
        if not release.wait(timeout=5):
            raise RuntimeError("test loader timed out")
        return loaded

    manager = model_manager.ModelManager(loader=blocking_loader)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(manager.load) for _ in range(8)]
        assert entered.wait(timeout=5)
        release.set()
        models = [future.result(timeout=5) for future in futures]

    assert calls == 1
    assert all(model is loaded.model for model in models)


def test_model_access_before_load_is_rejected_without_loading() -> None:
    loader = ControlledLoader()
    manager = model_manager.ModelManager(loader=loader)

    with pytest.raises(model_manager.ModelNotReadyError, match="unloaded"):
        _ = manager.model

    assert loader.calls == 0


def test_status_before_and_after_load_is_serialization_friendly() -> None:
    manager = model_manager.ModelManager(loader=ControlledLoader())

    before = manager.status().to_dict()
    manager.load()
    after = manager.status().to_dict()

    assert before["state"] == "unloaded"
    assert before["ready"] is False
    assert before["checkpoint_path"] == "artifacts/model/semirestore_conditioned.pt"
    assert before["checkpoint_sha256"] is None
    assert after == {
        "state": "ready",
        "ready": True,
        "model_name": "naf_sr",
        "architecture": "statistics-conditioned NAF-SR",
        "model_version": "synthetic-v1",
        "training_revision": "synthetic-revision",
        "resolved_device": "cpu",
        "parameter_count": 6,
        "checkpoint_path": "artifacts/model/synthetic.pt",
        "checkpoint_sha256": "a" * 64,
        "scale_factor": 2,
        "last_loading_error_category": None,
        "retry_permitted": False,
    }
    json.dumps(before)
    json.dumps(after)


def test_failed_load_is_safe_sticky_and_requires_explicit_reset() -> None:
    loader = ControlledLoader()
    loader.error = checkpoints.CheckpointVerificationError(
        "secret at C:/private/source/best.pt"
    )
    manager = model_manager.ModelManager(loader=loader)

    with pytest.raises(model_manager.ModelManagerLoadError) as first_error:
        manager.load()
    with pytest.raises(model_manager.ModelManagerLoadError) as repeated_error:
        manager.load()

    assert first_error.value.category == "checkpoint_verification"
    assert "private" not in str(first_error.value)
    assert repeated_error.value.category == "checkpoint_verification"
    assert loader.calls == 1
    status = manager.status()
    assert status.state is model_manager.ModelManagerState.FAILED
    assert status.ready is False
    assert status.last_loading_error_category == "checkpoint_verification"
    assert status.retry_permitted is True

    loader.error = None
    manager.reset_failure()
    assert manager.state is model_manager.ModelManagerState.UNLOADED
    assert manager.load() is loader.loaded.model
    assert loader.calls == 2


def test_unexpected_loader_failure_does_not_leak_details() -> None:
    loader = ControlledLoader()
    loader.error = RuntimeError("token=very-secret")
    manager = model_manager.ModelManager(loader=loader)

    with pytest.raises(model_manager.ModelManagerLoadError) as error:
        manager.load()

    assert error.value.category == "unexpected_loading_error"
    assert "secret" not in str(error.value)
    assert manager.status().last_loading_error_category == "unexpected_loading_error"


def test_close_from_unloaded_is_permanent_and_does_not_load() -> None:
    loader = ControlledLoader()
    manager = model_manager.ModelManager(loader=loader)

    manager.close()
    manager.close()

    assert manager.state is model_manager.ModelManagerState.CLOSED
    assert manager.is_ready is False
    assert loader.calls == 0
    with pytest.raises(model_manager.ModelManagerClosedError):
        manager.load()
    with pytest.raises(model_manager.ModelManagerClosedError):
        manager.reset_failure()
    with pytest.raises(model_manager.ModelManagerClosedError):
        _ = manager.model


def test_close_from_ready_releases_manager_reference_and_preserves_identity() -> None:
    loader = ControlledLoader()
    manager = model_manager.ModelManager(loader=loader)
    loaded_model = manager.load()

    manager.close()

    status = manager.status()
    assert status.state is model_manager.ModelManagerState.CLOSED
    assert status.ready is False
    assert status.model_name == "naf_sr"
    with pytest.raises(model_manager.ModelManagerClosedError):
        _ = manager.model
    with pytest.raises(model_manager.ModelManagerClosedError):
        manager.load()
    assert loaded_model is loader.loaded.model
    assert loader.calls == 1


def test_loaded_model_is_eval_with_gradients_disabled() -> None:
    model = TinyManagedModel()
    model.train()
    model.requires_grad_(True)
    manager = model_manager.ModelManager(loader=ControlledLoader(_loaded(model=model)))

    loaded_model = manager.load()

    assert loaded_model.training is False
    assert all(parameter.requires_grad is False for parameter in loaded_model.parameters())


def test_selected_device_metadata_is_preserved_without_moving_model() -> None:
    loaded = _loaded(device=torch.device("cuda:7"))
    loader = ControlledLoader(loaded)
    manager = model_manager.ModelManager(device="cuda:7", loader=loader)

    manager.load()

    assert manager.status().resolved_device == "cuda:7"
    assert loader.last_kwargs is not None
    assert loader.last_kwargs["device"] == "cuda:7"


def test_external_absolute_checkpoint_path_is_not_exposed(tmp_path: Path) -> None:
    absolute_checkpoint = tmp_path / "immutable-source" / "best.pt"
    manager = model_manager.ModelManager(
        checkpoint_path=absolute_checkpoint,
        loader=ControlledLoader(_loaded(checkpoint_path=absolute_checkpoint)),
    )

    before = manager.status().checkpoint_path
    manager.load()
    after = manager.status().checkpoint_path

    assert before == "best.pt"
    assert after == "best.pt"
    assert str(tmp_path) not in json.dumps(manager.status().to_dict())


def test_invalid_loaded_parameter_metadata_becomes_safe_failure() -> None:
    loaded = _loaded()
    invalid = checkpoints.LoadedCheckpoint(
        model=loaded.model,
        device=loaded.device,
        checkpoint_path=loaded.checkpoint_path,
        checkpoint_sha256=loaded.checkpoint_sha256,
        architecture=loaded.architecture,
        model_name=loaded.model_name,
        parameter_count=loaded.parameter_count + 1,
        model_version=loaded.model_version,
        training_revision=loaded.training_revision,
    )
    manager = model_manager.ModelManager(loader=ControlledLoader(invalid))

    with pytest.raises(model_manager.ModelManagerLoadError) as error:
        manager.load()

    assert error.value.category == "unexpected_loading_error"


@pytest.mark.local_checkpoint
def test_real_checkpoint_manager_lifecycle_integration() -> None:
    if not REAL_RUNTIME_CHECKPOINT.is_file():
        pytest.skip("verified ignored runtime checkpoint is unavailable")
    manager = model_manager.ModelManager(device="cpu")

    first = manager.load()
    second = manager.load()
    status = manager.status()

    assert first is second
    assert first is manager.model
    assert status.ready is True
    assert status.model_name == "naf_sr"
    assert status.architecture == "statistics-conditioned NAF-SR"
    assert status.model_version == "conditioned-d037473"
    assert status.training_revision == "d037473ddf4a3cd20eb3fef933991cd66749f4f2"
    assert status.resolved_device == "cpu"
    assert status.parameter_count == 9_111_684
    assert status.checkpoint_path == "artifacts/model/semirestore_conditioned.pt"
    assert status.checkpoint_sha256 == REAL_SHA256
    assert status.scale_factor == 2
    assert first.training is False
    assert all(parameter.requires_grad is False for parameter in first.parameters())

    manager.close()
    assert manager.status().state is model_manager.ModelManagerState.CLOSED
