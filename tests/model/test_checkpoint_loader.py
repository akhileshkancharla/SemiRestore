from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
import torch
from torch import nn

from semirestore import checkpoints

CONFIG_PATH = Path("configs/model/resolved_conditioned.yaml")
REAL_RUNTIME_CHECKPOINT = Path("artifacts/model/semirestore_conditioned.pt")
REAL_SHA256 = "273abd9d6dcfa9bdee71ac15016994962304b6c9d902898b4f4d503bed158c28"


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _write_metadata(
    path: Path,
    *,
    checkpoint: Path,
    expected_parameter_count: int,
    size_bytes: int | None = None,
    sha256: str | None = None,
) -> Path:
    payload = checkpoint.read_bytes() if checkpoint.is_file() else b"missing"
    document = {
        "schema_version": 1,
        "checkpoints": {
            "semirestore_conditioned": {
                "model_name": "naf_sr",
                "model_version": "synthetic-test",
                "architecture": "statistics-conditioned NAF-SR",
                "expected_parameter_count": expected_parameter_count,
                "runtime_artifact_path": str(checkpoint),
                "sha256": hashlib.sha256(payload).hexdigest() if sha256 is None else sha256,
                "size_bytes": len(payload) if size_bytes is None else size_bytes,
                "training_revision": "synthetic",
            }
        },
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _checkpoint_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload_factory: Callable[[Mapping[str, torch.Tensor]], object],
    *,
    expected_parameter_count: int | None = None,
) -> tuple[Path, Path, TinyModel]:
    source_model = TinyModel()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(payload_factory(source_model.state_dict()), checkpoint)
    count = _parameter_count(source_model)
    metadata = _write_metadata(
        tmp_path / "checksums.json",
        checkpoint=checkpoint,
        expected_parameter_count=(
            count if expected_parameter_count is None else expected_parameter_count
        ),
    )
    monkeypatch.setattr(checkpoints, "build_model", lambda _config: TinyModel())
    return checkpoint, metadata, source_model


def _load_synthetic(checkpoint: Path, metadata: Path) -> checkpoints.LoadedCheckpoint:
    return checkpoints.load_conditioned_checkpoint(
        checkpoint_path=checkpoint,
        metadata_path=metadata,
        config_path=CONFIG_PATH,
        device="cpu",
    )


def test_missing_checkpoint_is_rejected_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "missing.pt"
    metadata = _write_metadata(
        tmp_path / "checksums.json",
        checkpoint=checkpoint,
        expected_parameter_count=6,
    )
    load_called = False

    def forbidden_load(*_args: object, **_kwargs: object) -> object:
        nonlocal load_called
        load_called = True
        raise AssertionError("torch.load must not run")

    monkeypatch.setattr(checkpoints.torch, "load", forbidden_load)

    with pytest.raises(checkpoints.CheckpointVerificationError, match="does not exist"):
        _load_synthetic(checkpoint, metadata)

    assert load_called is False


def test_checksum_mismatch_prevents_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, metadata, _model = _checkpoint_case(
        tmp_path,
        monkeypatch,
        lambda state: {"model_state_dict": state},
    )
    contents = bytearray(checkpoint.read_bytes())
    contents[-1] ^= 1
    checkpoint.write_bytes(contents)
    load_called = False

    def forbidden_load(*_args: object, **_kwargs: object) -> object:
        nonlocal load_called
        load_called = True
        raise AssertionError("torch.load must not run")

    monkeypatch.setattr(checkpoints.torch, "load", forbidden_load)

    with pytest.raises(checkpoints.CheckpointVerificationError, match="SHA-256 mismatch"):
        _load_synthetic(checkpoint, metadata)

    assert load_called is False


def test_unexpected_container_structure_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, metadata, _model = _checkpoint_case(
        tmp_path,
        monkeypatch,
        lambda _state: ["not", "a", "mapping"],
    )

    with pytest.raises(checkpoints.CheckpointStructureError, match="container type"):
        _load_synthetic(checkpoint, metadata)


@pytest.mark.parametrize("container_key", [None, "model_state_dict", "state_dict", "model"])
def test_supported_state_dictionary_containers_load_strictly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    container_key: str | None,
) -> None:
    def payload(state: Mapping[str, torch.Tensor]) -> object:
        return dict(state) if container_key is None else {container_key: state}

    checkpoint, metadata, source_model = _checkpoint_case(tmp_path, monkeypatch, payload)

    loaded = _load_synthetic(checkpoint, metadata)

    for key, value in loaded.model.state_dict().items():
        torch.testing.assert_close(value, source_model.state_dict()[key])


def test_non_string_state_key_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, metadata, _model = _checkpoint_case(
        tmp_path,
        monkeypatch,
        lambda _state: {1: torch.ones(1)},
    )

    with pytest.raises(checkpoints.CheckpointStructureError, match="non-string"):
        _load_synthetic(checkpoint, metadata)


def test_non_tensor_state_value_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, metadata, _model = _checkpoint_case(
        tmp_path,
        monkeypatch,
        lambda _state: {"linear.weight": "not-a-tensor"},
    )

    with pytest.raises(checkpoints.CheckpointStructureError, match="non-tensor"):
        _load_synthetic(checkpoint, metadata)


def test_missing_state_key_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def payload(state: Mapping[str, torch.Tensor]) -> object:
        values = dict(state)
        values.pop("linear.bias")
        return {"model_state_dict": values}

    checkpoint, metadata, _model = _checkpoint_case(tmp_path, monkeypatch, payload)

    with pytest.raises(checkpoints.CheckpointCompatibilityError, match="missing key"):
        _load_synthetic(checkpoint, metadata)


def test_unexpected_state_key_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def payload(state: Mapping[str, torch.Tensor]) -> object:
        values = dict(state)
        values["unexpected.weight"] = torch.ones(1)
        return {"model_state_dict": values}

    checkpoint, metadata, _model = _checkpoint_case(tmp_path, monkeypatch, payload)

    with pytest.raises(checkpoints.CheckpointCompatibilityError, match="unexpected key"):
        _load_synthetic(checkpoint, metadata)


def test_tensor_shape_mismatch_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def payload(state: Mapping[str, torch.Tensor]) -> object:
        values = dict(state)
        values["linear.weight"] = torch.zeros((3, 2))
        return {"model_state_dict": values}

    checkpoint, metadata, _model = _checkpoint_case(tmp_path, monkeypatch, payload)

    with pytest.raises(checkpoints.CheckpointCompatibilityError, match="shape mismatch"):
        _load_synthetic(checkpoint, metadata)


def test_cpu_loading_freezes_model_and_returns_verified_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, metadata, _model = _checkpoint_case(
        tmp_path,
        monkeypatch,
        lambda state: {"model_state_dict": state},
    )
    real_torch_load = torch.load
    calls: list[dict[str, object]] = []

    def recording_load(*args: object, **kwargs: object) -> object:
        calls.append(dict(kwargs))
        return real_torch_load(*args, **kwargs)

    monkeypatch.setattr(checkpoints.torch, "load", recording_load)

    loaded = _load_synthetic(checkpoint, metadata)

    assert calls == [{"map_location": torch.device("cpu"), "weights_only": True}]
    assert loaded.device == torch.device("cpu")
    assert loaded.checkpoint_path == checkpoint
    assert loaded.model_name == "naf_sr"
    assert loaded.architecture == "statistics-conditioned NAF-SR"
    assert loaded.parameter_count == 6
    assert loaded.model_version == "synthetic-test"
    assert loaded.training_revision == "synthetic"
    assert loaded.checkpoint_sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert loaded.model.training is False
    assert all(parameter.requires_grad is False for parameter in loaded.model.parameters())


def test_parameter_count_is_enforced_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, metadata, _model = _checkpoint_case(
        tmp_path,
        monkeypatch,
        lambda state: {"model_state_dict": state},
        expected_parameter_count=7,
    )
    load_called = False

    def forbidden_load(*_args: object, **_kwargs: object) -> object:
        nonlocal load_called
        load_called = True
        raise AssertionError("torch.load must not run")

    monkeypatch.setattr(checkpoints.torch, "load", forbidden_load)

    with pytest.raises(checkpoints.CheckpointCompatibilityError, match="parameter count"):
        _load_synthetic(checkpoint, metadata)

    assert load_called is False


def test_auto_selects_cpu_when_cuda_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert checkpoints.resolve_device("auto") == torch.device("cpu")


def test_auto_selects_first_cuda_device_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    assert checkpoints.resolve_device("auto") == torch.device("cuda:0")
    assert checkpoints.resolve_device("cuda") == torch.device("cuda:0")
    assert checkpoints.resolve_device("cuda:1") == torch.device("cuda:1")


@pytest.mark.parametrize("requested", ["", "gpu", "cpu:0", "cuda:", "cuda:-1", "cuda:x"])
def test_malformed_device_request_is_rejected(requested: str) -> None:
    with pytest.raises(checkpoints.DeviceSelectionError, match="Device request|Malformed"):
        checkpoints.resolve_device(requested)


def test_unavailable_cuda_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(checkpoints.DeviceSelectionError, match="unavailable"):
        checkpoints.resolve_device("cuda:0")


def test_out_of_range_cuda_index_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(checkpoints.DeviceSelectionError, match="index 2"):
        checkpoints.resolve_device("cuda:2")


@pytest.mark.local_checkpoint
def test_real_conditioned_checkpoint_loads_safely_and_strictly() -> None:
    if not REAL_RUNTIME_CHECKPOINT.is_file():
        pytest.skip("verified ignored runtime checkpoint is unavailable")

    loaded = checkpoints.load_conditioned_checkpoint(device="cpu")

    assert loaded.checkpoint_sha256 == REAL_SHA256
    assert loaded.parameter_count == 9_111_684
    assert loaded.model_name == "naf_sr"
    assert loaded.architecture == "statistics-conditioned NAF-SR"
    assert loaded.device == torch.device("cpu")
    assert loaded.model.training is False
    assert all(parameter.requires_grad is False for parameter in loaded.model.parameters())
    tracked = subprocess.run(
        ["git", "ls-files", "*.pt", "*.pth", "*.ckpt"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert tracked.stdout.strip() == ""
