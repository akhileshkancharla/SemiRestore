"""Validation helpers for raw paired SEM training arrays.

Training arrays intentionally use a different contract from deployment inputs:
finite real values are converted to float32, but they are never normalized or
clipped.  In particular, degraded values outside ``[0, 1]`` are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class DatasetValidationError(ValueError):
    """Raised when paired training data violates the scientific contract."""


@dataclass(frozen=True, slots=True)
class TrainingArrayInfo:
    """Serialization-friendly facts recorded before float32 conversion."""

    path: str
    shape: tuple[int, int]
    source_dtype: str
    minimum: float
    maximum: float


def load_training_array(path: str | Path) -> tuple[np.ndarray, TrainingArrayInfo]:
    """Load one finite 2D ``.npy`` image without normalization or clipping."""

    source = Path(path)
    if source.suffix.casefold() != ".npy":
        raise DatasetValidationError(
            f"Unsupported training image format for {source}; expected a .npy array"
        )
    if not source.is_file():
        raise DatasetValidationError(f"Training array is not a regular file: {source}")
    try:
        array = np.load(source, allow_pickle=False)
    except (OSError, ValueError, TypeError) as error:
        raise DatasetValidationError(f"Could not safely load training array: {source}") from error

    if not isinstance(array, np.ndarray):
        raise DatasetValidationError(f"Training file did not contain an array: {source}")
    if array.ndim != 2:
        raise DatasetValidationError(
            f"Training array must be one-channel 2D data; got shape {array.shape}: {source}"
        )
    if array.size == 0:
        raise DatasetValidationError(f"Training array is empty: {source}")
    if (
        np.issubdtype(array.dtype, np.bool_)
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise DatasetValidationError(
            f"Training array must have a real numeric dtype; got {array.dtype}: {source}"
        )
    if not np.isfinite(array).all():
        raise DatasetValidationError(f"Training array contains NaN or infinity: {source}")

    info = TrainingArrayInfo(
        path=str(source),
        shape=(int(array.shape[0]), int(array.shape[1])),
        source_dtype=str(array.dtype),
        minimum=float(array.min()),
        maximum=float(array.max()),
    )
    # A distinct, contiguous buffer prevents tensor writes from reaching NumPy's
    # loaded storage and deliberately leaves the scientific value range intact.
    converted = np.array(array, dtype=np.float32, order="C", copy=True)
    return converted, info


__all__ = ["DatasetValidationError", "TrainingArrayInfo", "load_training_array"]
