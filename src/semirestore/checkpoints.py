"""Checksum-gated loading for the frozen conditioned NAF-SR checkpoint."""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .config import ModelConfigError, build_model, load_model_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_PATH = PROJECT_ROOT / "artifacts" / "model" / "checksums.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "model" / "resolved_conditioned.yaml"
CHECKPOINT_KEY = "semirestore_conditioned"
HASH_CHUNK_SIZE = 1024 * 1024
SUPPORTED_STATE_KEYS = ("model_state_dict", "state_dict", "model")
CUDA_DEVICE_PATTERN = re.compile(r"cuda(?::(0|[1-9][0-9]*))?")


class CheckpointError(RuntimeError):
    """Base class for safe checkpoint-loading failures."""


class CheckpointMetadataError(CheckpointError):
    """Raised when trusted checkpoint metadata is missing or malformed."""


class CheckpointVerificationError(CheckpointError):
    """Raised before deserialization when checkpoint identity is invalid."""


class CheckpointStructureError(CheckpointError):
    """Raised when verified content is not a supported state-dictionary container."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised when verified weights do not strictly match the frozen architecture."""


class DeviceSelectionError(CheckpointError):
    """Raised when a requested execution device is malformed or unavailable."""


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Trusted checkpoint identity and model facts loaded from tracked JSON."""

    runtime_path: Path
    sha256: str
    size_bytes: int
    model_name: str
    architecture: str
    expected_parameter_count: int
    model_version: str | None
    training_revision: str | None


@dataclass(frozen=True, slots=True)
class FileIdentity:
    """Opaque checkpoint size and incremental SHA-256."""

    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """Frozen model and verified runtime metadata returned to later services."""

    model: nn.Module
    device: torch.device
    checkpoint_path: Path
    checkpoint_sha256: str
    architecture: str
    model_name: str
    parameter_count: int
    model_version: str | None
    training_revision: str | None


def _required_string(values: Mapping[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise CheckpointMetadataError(f"Checkpoint metadata field {key!r} is invalid")
    return value


def _optional_string(values: Mapping[str, object], key: str) -> str | None:
    value = values.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise CheckpointMetadataError(f"Checkpoint metadata field {key!r} is invalid")
    return value


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_checkpoint_metadata(
    path: str | Path = DEFAULT_METADATA_PATH,
) -> CheckpointMetadata:
    """Load trusted model identity and runtime location from tracked metadata."""

    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise CheckpointMetadataError(f"Checkpoint metadata does not exist: {metadata_path}")
    try:
        document = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CheckpointMetadataError(
            f"Could not safely read checkpoint metadata: {metadata_path}"
        ) from error

    if not isinstance(document, Mapping) or document.get("schema_version") != 1:
        raise CheckpointMetadataError("Checkpoint metadata has an unsupported schema")
    checkpoints = document.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise CheckpointMetadataError("Checkpoint metadata has no checkpoints mapping")
    values = checkpoints.get(CHECKPOINT_KEY)
    if not isinstance(values, Mapping):
        raise CheckpointMetadataError(f"Checkpoint metadata has no {CHECKPOINT_KEY!r} entry")

    size_bytes = values.get("size_bytes")
    parameter_count = values.get("expected_parameter_count")
    if type(size_bytes) is not int or size_bytes < 1:
        raise CheckpointMetadataError("Checkpoint metadata size_bytes must be positive")
    if type(parameter_count) is not int or parameter_count < 1:
        raise CheckpointMetadataError(
            "Checkpoint metadata expected_parameter_count must be positive"
        )
    sha256 = _required_string(values, "sha256")
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise CheckpointMetadataError("Checkpoint metadata sha256 must be lowercase hexadecimal")

    return CheckpointMetadata(
        runtime_path=_project_path(_required_string(values, "runtime_artifact_path")),
        sha256=sha256,
        size_bytes=size_bytes,
        model_name=_required_string(values, "model_name"),
        architecture=_required_string(values, "architecture"),
        expected_parameter_count=parameter_count,
        model_version=_optional_string(values, "model_version"),
        training_revision=_optional_string(values, "training_revision"),
    )


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve ``cpu``, ``cuda``, ``cuda:N``, or ``auto`` to an available device."""

    if not isinstance(requested, str) or not requested.strip():
        raise DeviceSelectionError("Device request must be one of: auto, cpu, cuda, cuda:N")
    normalized = requested.strip().lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "auto":
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            return torch.device("cpu")
        return torch.device("cuda", 0)

    match = CUDA_DEVICE_PATTERN.fullmatch(normalized)
    if match is None:
        raise DeviceSelectionError(
            f"Malformed device request {requested!r}; expected auto, cpu, cuda, or cuda:N"
        )
    if not torch.cuda.is_available():
        raise DeviceSelectionError(f"CUDA was requested but is unavailable: {requested!r}")
    index = int(match.group(1) or 0)
    device_count = torch.cuda.device_count()
    if index >= device_count:
        raise DeviceSelectionError(
            f"CUDA device index {index} is unavailable; detected {device_count} device(s)"
        )
    return torch.device("cuda", index)


def _file_identity(path: Path) -> FileIdentity:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_SIZE), b""):
            size_bytes += len(chunk)
            digest.update(chunk)
    return FileIdentity(size_bytes=size_bytes, sha256=digest.hexdigest())


def verify_checkpoint(
    path: str | Path,
    metadata: CheckpointMetadata,
) -> FileIdentity:
    """Verify a regular checkpoint's size and digest before deserialization."""

    checkpoint_path = Path(path)
    if checkpoint_path.is_symlink():
        raise CheckpointVerificationError(
            f"Runtime checkpoint must not be a symbolic link: {checkpoint_path}"
        )
    try:
        file_stat = checkpoint_path.stat()
    except FileNotFoundError as error:
        raise CheckpointVerificationError(
            f"Runtime checkpoint does not exist: {checkpoint_path}"
        ) from error
    except OSError as error:
        raise CheckpointVerificationError(
            f"Could not inspect runtime checkpoint: {checkpoint_path}"
        ) from error
    if not stat.S_ISREG(file_stat.st_mode):
        raise CheckpointVerificationError(
            f"Runtime checkpoint is not a regular file: {checkpoint_path}"
        )
    if file_stat.st_size != metadata.size_bytes:
        raise CheckpointVerificationError(
            f"Runtime checkpoint size mismatch: expected {metadata.size_bytes} bytes, "
            f"got {file_stat.st_size}"
        )
    try:
        identity = _file_identity(checkpoint_path)
    except OSError as error:
        raise CheckpointVerificationError(
            f"Could not hash runtime checkpoint: {checkpoint_path}"
        ) from error
    if identity.size_bytes != metadata.size_bytes or identity.sha256 != metadata.sha256:
        raise CheckpointVerificationError(
            f"Runtime checkpoint SHA-256 mismatch: expected {metadata.sha256}, "
            f"got {identity.sha256}"
        )
    return identity


def _extract_state_dict(payload: object) -> dict[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise CheckpointStructureError(
            f"Unsupported checkpoint container type: {type(payload).__name__}"
        )

    container_keys = [key for key in SUPPORTED_STATE_KEYS if key in payload]
    if len(container_keys) > 1:
        raise CheckpointStructureError(
            f"Checkpoint has ambiguous state-dictionary containers: {container_keys}"
        )
    if container_keys:
        container_key = container_keys[0]
        candidate = payload[container_key]
        if not isinstance(candidate, Mapping):
            raise CheckpointStructureError(
                f"Checkpoint container {container_key!r} must hold a mapping"
            )
    else:
        candidate = payload

    if not candidate:
        raise CheckpointStructureError("Checkpoint state dictionary is empty")
    non_string_keys = [repr(key) for key in candidate if not isinstance(key, str)]
    if non_string_keys:
        raise CheckpointStructureError(
            f"Checkpoint state dictionary contains non-string key(s): {non_string_keys[:5]}"
        )
    non_tensor_keys = [key for key, value in candidate.items() if not torch.is_tensor(value)]
    if non_tensor_keys:
        raise CheckpointStructureError(
            f"Checkpoint state dictionary contains non-tensor value(s): {non_tensor_keys[:5]}"
        )
    return dict(candidate)


def _strictly_load_state_dict(model: nn.Module, state_dict: Mapping[str, torch.Tensor]) -> None:
    expected = model.state_dict()
    expected_keys = set(expected)
    actual_keys = set(state_dict)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing key(s): {missing[:10]}")
        if unexpected:
            details.append(f"unexpected key(s): {unexpected[:10]}")
        raise CheckpointCompatibilityError(
            "Checkpoint keys do not strictly match the model: " + "; ".join(details)
        )

    shape_mismatches = []
    for key, expected_value in expected.items():
        actual_value = state_dict[key]
        if actual_value.shape != expected_value.shape:
            shape_mismatches.append(
                f"{key}: expected {tuple(expected_value.shape)}, got {tuple(actual_value.shape)}"
            )
    if shape_mismatches:
        raise CheckpointCompatibilityError(
            "Checkpoint tensor shape mismatch(es): " + "; ".join(shape_mismatches[:10])
        )

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise CheckpointCompatibilityError(
            "Checkpoint failed strict model-state loading"
        ) from error


def load_conditioned_checkpoint(
    *,
    checkpoint_path: str | Path | None = None,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    device: str = "auto",
) -> LoadedCheckpoint:
    """Verify, safely deserialize, and freeze the conditioned NAF-SR model."""

    metadata = load_checkpoint_metadata(metadata_path)
    resolved_path = Path(checkpoint_path) if checkpoint_path is not None else metadata.runtime_path
    identity = verify_checkpoint(resolved_path, metadata)
    resolved_device = resolve_device(device)

    try:
        config = load_model_config(config_path)
        model = build_model(config)
    except ModelConfigError as error:
        raise CheckpointCompatibilityError(
            "Could not construct the frozen checkpoint-compatible architecture"
        ) from error
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != metadata.expected_parameter_count:
        raise CheckpointCompatibilityError(
            f"Constructed model parameter count mismatch: expected "
            f"{metadata.expected_parameter_count}, got {parameter_count}"
        )
    model.to(resolved_device)

    try:
        payload = torch.load(
            resolved_path,
            map_location=resolved_device,
            weights_only=True,
        )
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as error:
        raise CheckpointStructureError(
            "Verified checkpoint could not be loaded with weights_only=True; "
            "unsafe fallback is disabled"
        ) from error

    state_dict = _extract_state_dict(payload)
    _strictly_load_state_dict(model, state_dict)
    model.eval()
    model.requires_grad_(False)

    return LoadedCheckpoint(
        model=model,
        device=resolved_device,
        checkpoint_path=resolved_path,
        checkpoint_sha256=identity.sha256,
        architecture=metadata.architecture,
        model_name=metadata.model_name,
        parameter_count=parameter_count,
        model_version=metadata.model_version,
        training_revision=metadata.training_revision,
    )


__all__ = [
    "CheckpointCompatibilityError",
    "CheckpointError",
    "CheckpointMetadata",
    "CheckpointMetadataError",
    "CheckpointStructureError",
    "CheckpointVerificationError",
    "DeviceSelectionError",
    "LoadedCheckpoint",
    "load_checkpoint_metadata",
    "load_conditioned_checkpoint",
    "resolve_device",
    "verify_checkpoint",
]
