"""Safe atomic persistence for resumable training and deployable best weights."""

from __future__ import annotations

import os
import re
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

TRAINING_CHECKPOINT_VERSION = 1
_ROTATED_PATTERN = re.compile(r"last-step-([0-9]{8,12})\.pt")


class TrainingCheckpointError(RuntimeError):
    """Raised when training persistence fails safe validation."""


def _partial_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")


def _prepare_destination(path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise TrainingCheckpointError("Checkpoint parent must be a real directory")
    if destination.exists() and (destination.is_symlink() or not destination.is_file()):
        raise TrainingCheckpointError("Checkpoint destination must be a regular file")
    return destination


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Serialize through a same-directory partial and atomically replace on success."""

    if not isinstance(payload, Mapping) or not payload:
        raise TrainingCheckpointError("Checkpoint payload must be a non-empty mapping")
    destination = _prepare_destination(path)
    partial = _partial_path(destination)
    try:
        torch.save(dict(payload), partial)
        if not partial.is_file() or partial.stat().st_size < 1:
            raise TrainingCheckpointError("Checkpoint serialization produced no regular file")
        os.replace(partial, destination)
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise TrainingCheckpointError(f"Could not atomically write {destination.name}") from error
    finally:
        if partial.exists():
            partial.unlink()
    return destination


def _atomic_copy(source: Path, destination: Path) -> None:
    partial = _partial_path(destination)
    try:
        shutil.copyfile(source, partial)
        os.replace(partial, destination)
    except OSError as error:
        raise TrainingCheckpointError(f"Could not update {destination.name}") from error
    finally:
        if partial.exists():
            partial.unlink()


def load_training_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load only safe tensor/primitive training state with ``weights_only=True``."""

    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise TrainingCheckpointError(f"Resume checkpoint is not a regular file: {source}")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError) as error:
        raise TrainingCheckpointError("Could not safely load resume checkpoint") from error
    if not isinstance(payload, Mapping):
        raise TrainingCheckpointError("Resume checkpoint root must be a mapping")
    required = {
        "format_version",
        "checkpoint_role",
        "configuration_fingerprint",
        "model_fingerprint",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "step",
        "epoch",
        "batch_in_epoch",
        "best_metric_name",
        "best_metric_value",
        "best_weights_source",
        "torch_rng_state",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise TrainingCheckpointError(f"Resume checkpoint is missing fields: {missing}")
    if payload.get("format_version") != TRAINING_CHECKPOINT_VERSION:
        raise TrainingCheckpointError("Unsupported training checkpoint format")
    if payload.get("checkpoint_role") != "training_resume":
        raise TrainingCheckpointError("Checkpoint is not resumable training state")
    for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict"):
        if not isinstance(payload.get(key), Mapping):
            raise TrainingCheckpointError(f"Resume checkpoint field {key!r} must be a mapping")
    return dict(payload)


class TrainingCheckpointManager:
    """Own a bounded set of explicitly named training checkpoint files."""

    def __init__(self, directory: str | Path, *, keep_last: int = 2) -> None:
        if type(keep_last) is not int or not 1 <= keep_last <= 10:
            raise TrainingCheckpointError("keep_last must be an integer in [1, 10]")
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.directory.is_symlink() or not self.directory.is_dir():
            raise TrainingCheckpointError("Checkpoint directory must be a real directory")
        self.keep_last = keep_last

    @property
    def last_path(self) -> Path:
        return self.directory / "last.pt"

    @property
    def best_path(self) -> Path:
        return self.directory / "best.pt"

    def save_resume(self, payload: Mapping[str, Any], *, step: int) -> Path:
        if type(step) is not int or step < 1:
            raise TrainingCheckpointError("Checkpoint step must be a positive integer")
        archive = self.directory / f"last-step-{step:08d}.pt"
        atomic_torch_save(payload, archive)
        _atomic_copy(archive, self.last_path)
        self._rotate()
        return self.last_path

    def save_best(self, payload: Mapping[str, Any]) -> Path:
        return atomic_torch_save(payload, self.best_path)

    def _rotate(self) -> None:
        candidates: list[tuple[int, Path]] = []
        for path in self.directory.iterdir():
            match = _ROTATED_PATTERN.fullmatch(path.name)
            if match and path.is_file() and not path.is_symlink():
                candidates.append((int(match.group(1)), path))
        candidates.sort(reverse=True)
        for _, path in candidates[self.keep_last :]:
            # The exact parent and strict filename pattern were checked above.
            path.unlink()


__all__ = [
    "TRAINING_CHECKPOINT_VERSION",
    "TrainingCheckpointError",
    "TrainingCheckpointManager",
    "atomic_torch_save",
    "load_training_checkpoint",
]
