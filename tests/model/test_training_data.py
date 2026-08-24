from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from semirestore.data import DatasetValidationError, load_training_array
from semirestore.training_data import PairedSEMDataset, read_pair_manifest


def _save_pair(
    root: Path,
    sample_id: str,
    *,
    low: np.ndarray | None = None,
    high: np.ndarray | None = None,
) -> tuple[str, str]:
    low_path = root / "lr" / f"{sample_id}.npy"
    high_path = root / "hr" / f"{sample_id}.npy"
    low_path.parent.mkdir(parents=True, exist_ok=True)
    high_path.parent.mkdir(parents=True, exist_ok=True)
    low_value = np.arange(6, dtype=np.float32).reshape(2, 3) if low is None else low
    high_value = np.arange(24, dtype=np.float32).reshape(4, 6) if high is None else high
    np.save(low_path, low_value)
    np.save(high_path, high_value)
    return low_path.relative_to(root).as_posix(), high_path.relative_to(root).as_posix()


def _write_manifest(root: Path, rows: list[dict[str, str]]) -> Path:
    path = root / "manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "lr_path", "hr_path", "split"))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(root: Path, sample_id: str, split: str = "train") -> dict[str, str]:
    low, high = _save_pair(root, sample_id)
    return {"sample_id": sample_id, "lr_path": low, "hr_path": high, "split": split}


def test_valid_pair_loading_returns_raw_tensors_and_metadata(tmp_path: Path) -> None:
    low = np.array([[-0.25, 0.5], [1.25, 2.0]], dtype=np.float32)
    high = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    low_path, high_path = _save_pair(tmp_path, "sample-1", low=low, high=high)
    manifest = _write_manifest(
        tmp_path,
        [{"sample_id": "sample-1", "lr_path": low_path, "hr_path": high_path, "split": "train"}],
    )

    dataset = PairedSEMDataset.from_manifest(manifest, tmp_path, split="train")
    degraded, target, metadata = dataset[0]

    assert degraded.shape == (1, 2, 2)
    assert target.shape == (1, 4, 4)
    assert degraded.dtype == target.dtype == torch.float32
    torch.testing.assert_close(degraded.squeeze(0), torch.from_numpy(low))
    assert metadata["sample_id"] == "sample-1"
    assert metadata["split"] == "train"
    assert metadata["value_policy"] == "raw_float32_no_normalization_or_clipping"


def test_historical_manifest_columns_are_supported(tmp_path: Path) -> None:
    low, high = _save_pair(tmp_path, "legacy")
    manifest = tmp_path / "historical.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("stem", "input_relpath", "target_relpath", "split", "input_min"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "stem": "legacy",
                "input_relpath": low,
                "target_relpath": high,
                "split": "val_ood",
                "input_min": "-0.1",
            }
        )

    records = read_pair_manifest(manifest, tmp_path, split="val_ood")

    assert [record.sample_id for record in records] == ["legacy"]


def test_exact_two_times_scale_is_validated_eagerly(tmp_path: Path) -> None:
    row = _row(tmp_path, "bad-scale")
    np.save(tmp_path / row["hr_path"], np.zeros((5, 6), dtype=np.float32))
    manifest = _write_manifest(tmp_path, [row])

    with pytest.raises(DatasetValidationError, match="expected exact 2x"):
        PairedSEMDataset.from_manifest(manifest, tmp_path, split="train")


def test_manifest_order_is_deterministic_by_casefolded_identifier(tmp_path: Path) -> None:
    rows = [_row(tmp_path, name) for name in ("zeta", "Beta", "alpha")]
    manifest = _write_manifest(tmp_path, rows)

    first = read_pair_manifest(manifest, tmp_path, split="train")
    second = read_pair_manifest(manifest, tmp_path, split="train")

    assert [record.sample_id for record in first] == ["alpha", "Beta", "zeta"]
    assert first == second


@pytest.mark.parametrize(
    ("fieldnames", "message"),
    [
        (("sample_id", "lr_path", "split"), "Manifest must contain"),
        (("sample_id", "lr_path", "hr_path", "split"), "empty sample identifier"),
    ],
)
def test_malformed_manifest(tmp_path: Path, fieldnames: tuple[str, ...], message: str) -> None:
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if "hr_path" in fieldnames:
            writer.writerow({name: "train" if name == "split" else "" for name in fieldnames})

    with pytest.raises(DatasetValidationError, match=message):
        read_pair_manifest(manifest, tmp_path, split="train")


def test_missing_manifest_and_dataset_files_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="Manifest is not"):
        read_pair_manifest(tmp_path / "missing.csv", tmp_path, split="train")

    manifest = _write_manifest(
        tmp_path,
        [{"sample_id": "missing", "lr_path": "no.npy", "hr_path": "none.npy", "split": "train"}],
    )
    with pytest.raises(DatasetValidationError, match="file is missing"):
        read_pair_manifest(manifest, tmp_path, split="train")


def test_duplicate_identifier_is_rejected_across_splits(tmp_path: Path) -> None:
    first = _row(tmp_path, "shared", "train")
    second_low, second_high = _save_pair(tmp_path, "other")
    second = {
        "sample_id": "SHARED",
        "lr_path": second_low,
        "hr_path": second_high,
        "split": "validation",
    }
    manifest = _write_manifest(tmp_path, [first, second])

    with pytest.raises(DatasetValidationError, match="Duplicate sample identifier"):
        read_pair_manifest(manifest, tmp_path, split="train")


def test_file_reuse_detects_split_leakage(tmp_path: Path) -> None:
    first = _row(tmp_path, "train-a", "train")
    _, second_high = _save_pair(tmp_path, "val-b")
    second = {
        "sample_id": "val-b",
        "lr_path": first["lr_path"],
        "hr_path": second_high,
        "split": "validation",
    }
    manifest = _write_manifest(tmp_path, [first, second])

    with pytest.raises(DatasetValidationError, match="split leakage"):
        read_pair_manifest(manifest, tmp_path, split="train")


@pytest.mark.parametrize("unsafe", ("../escape.npy", "..\\escape.npy"))
def test_path_traversal_is_rejected(tmp_path: Path, unsafe: str) -> None:
    outside = tmp_path.parent / "escape.npy"
    np.save(outside, np.zeros((2, 2), dtype=np.float32))
    _, high = _save_pair(tmp_path, "safe")
    manifest = _write_manifest(
        tmp_path,
        [{"sample_id": "unsafe", "lr_path": unsafe, "hr_path": high, "split": "train"}],
    )

    with pytest.raises(DatasetValidationError, match="escapes the dataset root"):
        read_pair_manifest(manifest, tmp_path, split="train")


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((2, 2, 1), dtype=np.float32),
        np.zeros((1, 2, 2), dtype=np.float32),
        np.zeros((2, 2), dtype=np.bool_),
        np.zeros((2, 2), dtype=np.complex64),
    ],
)
def test_wrong_dimensions_channels_and_dtypes_are_rejected(
    tmp_path: Path, array: np.ndarray
) -> None:
    path = tmp_path / "bad.npy"
    np.save(path, array)

    with pytest.raises(DatasetValidationError, match="2D|real numeric"):
        load_training_array(path)


@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_non_finite_training_arrays_are_rejected(tmp_path: Path, value: float) -> None:
    path = tmp_path / "bad.npy"
    array = np.zeros((2, 2), dtype=np.float32)
    array[0, 0] = value
    np.save(path, array)

    with pytest.raises(DatasetValidationError, match="NaN or infinity"):
        load_training_array(path)


def test_raw_out_of_range_values_are_preserved_and_source_is_unchanged(tmp_path: Path) -> None:
    source = np.array([[-2.5, 0.0], [1.5, 9.0]], dtype=np.float64)
    original = source.copy()
    path = tmp_path / "raw.npy"
    np.save(path, source)

    loaded, info = load_training_array(path)
    loaded[0, 0] = 100.0

    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(np.load(path), original)
    assert info.minimum == -2.5
    assert info.maximum == 9.0
    assert loaded.dtype == np.float32


def test_split_names_and_allowed_split_set_are_validated(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, [_row(tmp_path, "sample", "holdout")])

    with pytest.raises(DatasetValidationError, match="unsupported split"):
        read_pair_manifest(
            manifest,
            tmp_path,
            split="train",
            allowed_splits=("train", "validation", "test"),
        )
    with pytest.raises(DatasetValidationError, match="portable split"):
        read_pair_manifest(manifest, tmp_path, split="../train")


def test_dataset_is_dataloader_compatible(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, [_row(tmp_path, "a"), _row(tmp_path, "b")])
    dataset = PairedSEMDataset.from_manifest(manifest, tmp_path, split="train")

    degraded, target, metadata = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))

    assert degraded.shape == (2, 1, 2, 3)
    assert target.shape == (2, 1, 4, 6)
    assert metadata["sample_id"] == ["a", "b"]
    assert metadata["value_policy"] == [
        "raw_float32_no_normalization_or_clipping",
        "raw_float32_no_normalization_or_clipping",
    ]
