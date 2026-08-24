"""Safe manifest-backed paired SEM restoration data."""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .data import DatasetValidationError, TrainingArrayInfo, load_training_array

_SPLIT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_CANONICAL_COLUMNS = {
    "sample_id": "sample_id",
    "low_resolution_path": "lr_path",
    "high_resolution_path": "hr_path",
    "split": "split",
}
_HISTORICAL_COLUMNS = {
    "sample_id": "stem",
    "low_resolution_path": "input_relpath",
    "high_resolution_path": "target_relpath",
    "split": "split",
}


@dataclass(frozen=True, slots=True)
class PairRecord:
    """One validated LR/HR manifest record with stable identity."""

    sample_id: str
    low_resolution_path: Path
    high_resolution_path: Path
    split: str


def _safe_path(root: Path, text: str, *, role: str, sample_id: str) -> Path:
    if not text.strip():
        raise DatasetValidationError(f"Empty {role} path for sample {sample_id!r}")
    relative = Path(text.strip())
    if relative.is_absolute():
        raise DatasetValidationError(
            f"Manifest {role} path for sample {sample_id!r} must be relative"
        )
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise DatasetValidationError(
            f"Manifest {role} path for sample {sample_id!r} escapes the dataset root"
        )
    if not resolved.is_file():
        raise DatasetValidationError(
            f"Manifest {role} file is missing for sample {sample_id!r}: {resolved}"
        )
    return resolved


def _column_mapping(fieldnames: Sequence[str] | None) -> dict[str, str]:
    available = set(fieldnames or ())
    canonical = set(_CANONICAL_COLUMNS.values())
    historical = set(_HISTORICAL_COLUMNS.values())
    if canonical.issubset(available):
        return _CANONICAL_COLUMNS
    if historical.issubset(available):
        return _HISTORICAL_COLUMNS
    raise DatasetValidationError(
        "Manifest must contain either canonical columns "
        "(sample_id, lr_path, hr_path, split) or historical columns "
        "(stem, input_relpath, target_relpath, split)"
    )


def _validate_split(split: str, *, context: str) -> str:
    value = split.strip()
    if not _SPLIT_PATTERN.fullmatch(value):
        raise DatasetValidationError(
            f"{context} must be a non-empty portable split name; got {split!r}"
        )
    return value


def read_pair_manifest(
    manifest_path: str | Path,
    dataset_root: str | Path,
    *,
    split: str,
    allowed_splits: Sequence[str] | None = None,
) -> list[PairRecord]:
    """Parse, audit, and deterministically select one split from a paired CSV."""

    manifest = Path(manifest_path).expanduser().resolve()
    root = Path(dataset_root).expanduser().resolve()
    selected_split = _validate_split(split, context="Requested split")
    if not manifest.is_file():
        raise DatasetValidationError(f"Manifest is not a regular file: {manifest}")
    if not root.is_dir():
        raise DatasetValidationError(f"Dataset root is not a directory: {root}")

    allowed: set[str] | None = None
    if allowed_splits is not None:
        allowed = {
            _validate_split(value, context="Allowed split") for value in allowed_splits
        }
        if not allowed:
            raise DatasetValidationError("allowed_splits must not be empty")
        if selected_split not in allowed:
            raise DatasetValidationError(
                f"Requested split {selected_split!r} is not in allowed_splits"
            )

    records: list[PairRecord] = []
    ids: dict[str, tuple[str, str]] = {}
    paths: dict[str, tuple[str, str, str]] = {}
    try:
        handle = manifest.open("r", encoding="utf-8-sig", newline="")
    except OSError as error:
        raise DatasetValidationError(f"Could not read manifest: {manifest}") from error
    with handle:
        reader = csv.DictReader(handle)
        columns = _column_mapping(reader.fieldnames)
        for line_number, row in enumerate(reader, start=2):
            sample_id = (row.get(columns["sample_id"]) or "").strip()
            if not sample_id:
                raise DatasetValidationError(
                    f"Manifest row {line_number} has an empty sample identifier"
                )
            split_value = _validate_split(
                row.get(columns["split"]) or "", context=f"Split on row {line_number}"
            )
            if allowed is not None and split_value not in allowed:
                raise DatasetValidationError(
                    f"Manifest row {line_number} uses unsupported split {split_value!r}"
                )
            identity_key = sample_id.casefold()
            if identity_key in ids:
                previous_id, previous_split = ids[identity_key]
                raise DatasetValidationError(
                    f"Duplicate sample identifier {sample_id!r} across splits "
                    f"{previous_split!r} and {split_value!r} (previous spelling {previous_id!r})"
                )
            ids[identity_key] = (sample_id, split_value)

            low_path = _safe_path(
                root,
                row.get(columns["low_resolution_path"]) or "",
                role="low-resolution",
                sample_id=sample_id,
            )
            high_path = _safe_path(
                root,
                row.get(columns["high_resolution_path"]) or "",
                role="high-resolution",
                sample_id=sample_id,
            )
            for role, path in (("low-resolution", low_path), ("high-resolution", high_path)):
                path_key = str(path).casefold()
                if path_key in paths:
                    previous_id, previous_role, previous_split = paths[path_key]
                    raise DatasetValidationError(
                        f"Dataset file is reused by samples {previous_id!r} and {sample_id!r} "
                        f"({previous_role}/{previous_split} and {role}/{split_value}); "
                        "this creates duplicate or split leakage"
                    )
                paths[path_key] = (sample_id, role, split_value)

            records.append(PairRecord(sample_id, low_path, high_path, split_value))

    selected = [record for record in records if record.split == selected_split]
    if not selected:
        raise DatasetValidationError(
            f"Manifest contains no samples for requested split {selected_split!r}"
        )
    return sorted(selected, key=lambda record: (record.sample_id.casefold(), record.sample_id))


def _pair_metadata(
    record: PairRecord,
    low: TrainingArrayInfo,
    high: TrainingArrayInfo,
) -> dict[str, Any]:
    return {
        "sample_id": record.sample_id,
        "split": record.split,
        "low_resolution_path": low.path,
        "high_resolution_path": high.path,
        "low_resolution_shape": low.shape,
        "high_resolution_shape": high.shape,
        "low_resolution_source_dtype": low.source_dtype,
        "high_resolution_source_dtype": high.source_dtype,
        "low_resolution_min": low.minimum,
        "low_resolution_max": low.maximum,
        "high_resolution_min": high.minimum,
        "high_resolution_max": high.maximum,
        "value_policy": "raw_float32_no_normalization_or_clipping",
    }


def _load_pair(record: PairRecord) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    low_array, low_info = load_training_array(record.low_resolution_path)
    high_array, high_info = load_training_array(record.high_resolution_path)
    expected = (low_info.shape[0] * 2, low_info.shape[1] * 2)
    if high_info.shape != expected:
        raise DatasetValidationError(
            f"Pair {record.sample_id!r} has LR shape {low_info.shape} and HR shape "
            f"{high_info.shape}; expected exact 2x HR shape {expected}"
        )
    low_tensor = torch.from_numpy(low_array).unsqueeze(0)
    high_tensor = torch.from_numpy(high_array).unsqueeze(0)
    return low_tensor, high_tensor, _pair_metadata(record, low_info, high_info)


class PairedSEMDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]):
    """PyTorch dataset for validated raw LR/HR SEM pairs."""

    def __init__(self, records: Sequence[PairRecord], *, validate_on_init: bool = True) -> None:
        if not records:
            raise DatasetValidationError("PairedSEMDataset requires at least one record")
        self._records = tuple(records)
        if validate_on_init:
            for record in self._records:
                _load_pair(record)

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        dataset_root: str | Path,
        *,
        split: str,
        allowed_splits: Sequence[str] | None = None,
        validate_on_init: bool = True,
    ) -> PairedSEMDataset:
        records = read_pair_manifest(
            manifest_path,
            dataset_root,
            split=split,
            allowed_splits=allowed_splits,
        )
        return cls(records, validate_on_init=validate_on_init)

    @property
    def records(self) -> tuple[PairRecord, ...]:
        return self._records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        return _load_pair(self._records[index])


__all__ = ["PairRecord", "PairedSEMDataset", "read_pair_manifest"]
