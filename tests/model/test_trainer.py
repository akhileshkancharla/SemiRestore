from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from semirestore.config import CONDITIONED_CHECKPOINT_CONFIG
from semirestore.degradations import DegradationConfig, ParameterRange
from semirestore.trainer import (
    ConditionedRestorationTrainer,
    TrainerConfig,
    TrainingConfigurationError,
    TrainingDataConfig,
    TrainingRuntimeError,
    cosine_warmup_factor,
    create_training_loaders,
)


class TinyPairedDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, str]]]):
    def __init__(self, count: int = 4) -> None:
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, str]]:
        low = torch.linspace(0.0, 1.0, 36).reshape(1, 6, 6) + index * 0.01
        high = F.interpolate(low[None], scale_factor=2, mode="nearest")[0] * 0.75 + 0.1
        return low, high, {"sample_id": f"sample-{index}", "split": "train"}


class TinyConditionedModel(nn.Module):
    statistics_conditioning = True
    scale = 2

    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.5))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.last_conditioning: torch.Tensor | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flattened = inputs.flatten(2)
        self.last_conditioning = torch.cat(
            (
                flattened.mean(2),
                flattened.std(2, unbiased=False),
                flattened.amin(2),
                flattened.amax(2),
            ),
            dim=1,
        ).detach()
        upsampled = F.interpolate(inputs, scale_factor=2, mode="nearest")
        return upsampled * self.gain + self.bias


def _loader(count: int = 4, *, batch_size: int = 2, shuffle: bool = False) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(99)
    return DataLoader(
        TinyPairedDataset(count),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _config(**overrides: object) -> TrainerConfig:
    values: dict[str, object] = {
        "seed": 2026,
        "device": "cpu",
        "deterministic": True,
        "batch_size": 2,
        "num_workers": 0,
        "max_steps": 4,
        "validation_interval": 2,
        "learning_rate": 0.05,
        "weight_decay": 0.0,
        "warmup_steps": 2,
        "gradient_clip_norm": 1.0,
        "charbonnier_epsilon": 1e-3,
        "d4_augmentation": False,
        "synthetic_probability": 0.0,
        "metric_data_range": 1.0,
        "metric_data_min": 0.0,
        "metric_range_policy": "clip",
    }
    values.update(overrides)
    return TrainerConfig(**values)


def test_one_cpu_training_step_updates_parameters_and_counts_step() -> None:
    model = TinyConditionedModel()
    trainer = ConditionedRestorationTrainer(model, _config())
    before = {name: value.detach().clone() for name, value in model.named_parameters()}

    summary = trainer.train_step(next(iter(_loader())))

    assert summary.step == 1
    assert summary.batch_size == 2
    assert summary.loss > 0
    assert any(not torch.equal(before[name], value) for name, value in model.named_parameters())
    assert isinstance(trainer.optimizer, torch.optim.AdamW)


def test_validation_does_not_update_parameters_or_optimizer_step() -> None:
    model = TinyConditionedModel()
    trainer = ConditionedRestorationTrainer(model, _config())
    trainer.train_step(next(iter(_loader())))
    before = deepcopy(model.state_dict())
    step_before = trainer.step

    summary = trainer.validate(_loader())

    assert summary.image_count == 4
    assert summary.mean_loss > 0
    assert trainer.step == step_before
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_conditioning_is_computed_during_training() -> None:
    model = TinyConditionedModel()
    trainer = ConditionedRestorationTrainer(model, _config())
    batch = next(iter(_loader()))

    trainer.train_step(batch)

    assert model.last_conditioning is not None
    assert model.last_conditioning.shape == (2, 4)
    expected_mean = batch[0].flatten(2).mean(2)
    torch.testing.assert_close(model.last_conditioning[:, :1], expected_mean)


def test_frozen_conditioned_nafsr_construction() -> None:
    trainer = ConditionedRestorationTrainer.build_conditioned(
        _config(max_steps=1, warmup_steps=0)
    )

    assert trainer.model.statistics_conditioning is True
    assert sum(parameter.numel() for parameter in trainer.model.parameters()) == 9_111_684
    assert CONDITIONED_CHECKPOINT_CONFIG.statistics_conditioning is True


def test_unconditioned_model_is_rejected() -> None:
    model = TinyConditionedModel()
    model.statistics_conditioning = False

    with pytest.raises(TrainingConfigurationError, match="active conditioning"):
        ConditionedRestorationTrainer(model, _config())


def test_historical_optimizer_scheduler_and_warmup_behavior() -> None:
    config = _config(learning_rate=0.1, max_steps=4, warmup_steps=2)
    trainer = ConditionedRestorationTrainer(TinyConditionedModel(), config)

    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
    first = trainer.train_step(next(iter(_loader(batch_size=1))))
    second = trainer.train_step(next(iter(_loader(batch_size=1))))
    third = trainer.train_step(next(iter(_loader(batch_size=1))))

    assert first.learning_rate == pytest.approx(0.1)
    assert second.learning_rate == pytest.approx(0.1)
    assert third.learning_rate == pytest.approx(0.05)
    assert cosine_warmup_factor(4, max_steps=4, warmup_steps=2) == pytest.approx(0.0)


def test_repeated_run_is_deterministic() -> None:
    initial = TinyConditionedModel().state_dict()

    def run() -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        model = TinyConditionedModel()
        model.load_state_dict(initial)
        trainer = ConditionedRestorationTrainer(
            model, _config(d4_augmentation=True, max_steps=3, validation_interval=3)
        )
        summary = trainer.fit(_loader(shuffle=True), _loader(), max_steps=3)
        return deepcopy(model.state_dict()), summary.as_dict()

    first_state, first_summary = run()
    second_state, second_summary = run()

    assert first_summary == second_summary
    for name in first_state:
        torch.testing.assert_close(first_state[name], second_state[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"batch_size": 0},
        {"learning_rate": 0.0},
        {"warmup_steps": 5, "max_steps": 4},
        {"synthetic_probability": 1.1},
        {"metric_range_policy": "infer"},
        {"deterministic": 1},
    ],
)
def test_malformed_trainer_configuration(overrides: dict[str, object]) -> None:
    with pytest.raises(TrainingConfigurationError):
        _config(**overrides)


def test_empty_training_dataset_is_rejected() -> None:
    trainer = ConditionedRestorationTrainer(TinyConditionedModel(), _config())
    empty = DataLoader(TinyPairedDataset(0), batch_size=1)

    with pytest.raises(TrainingRuntimeError, match="Training loader is empty"):
        trainer.fit(empty, _loader(), max_steps=1)


class NonFiniteLoss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (prediction - target).mean() * torch.tensor(float("nan"))


def test_non_finite_loss_stops_before_optimizer_step() -> None:
    model = TinyConditionedModel()
    trainer = ConditionedRestorationTrainer(model, _config(), loss_function=NonFiniteLoss())
    before = deepcopy(model.state_dict())

    with pytest.raises(TrainingRuntimeError, match="Non-finite training loss"):
        trainer.train_step(next(iter(_loader())))

    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_structured_metrics_are_strict_json_serializable() -> None:
    trainer = ConditionedRestorationTrainer(TinyConditionedModel(), _config())

    summary = trainer.fit(_loader(), _loader(), max_steps=2)
    payload = summary.as_dict()

    assert payload["completed_steps"] == 2
    assert payload["amp_enabled"] is False
    assert payload["ema_enabled"] is False
    assert payload["checkpoint_writes"] is False
    json.dumps(payload, allow_nan=False)


def test_configured_degradation_replaces_lr_deterministically() -> None:
    degradation = DegradationConfig(
        blur_sigma=ParameterRange(0.0, 0.0),
        gaussian_noise_std=ParameterRange(0.0, 0.0),
        speckle_std=ParameterRange(0.0, 0.0),
        additive_bias=ParameterRange(0.0, 0.0),
        downsample_modes=("area",),
        randomize_order=False,
    )
    model = TinyConditionedModel()
    trainer = ConditionedRestorationTrainer(
        model,
        _config(synthetic_probability=1.0),
        degradation_config=degradation,
    )
    degraded, target, metadata = next(iter(_loader()))

    summary = trainer.train_step((torch.zeros_like(degraded), target, metadata))

    assert summary.synthetic_samples == degraded.shape[0]
    assert model.last_conditioning is not None
    expected = F.interpolate(target, size=(6, 6), mode="area").flatten(2).mean(2)
    torch.testing.assert_close(model.last_conditioning[:, :1], expected)


def _write_manifest_fixture(root: Path) -> Path:
    rows: list[dict[str, str]] = []
    for split, sample_id in (("train", "train-a"), ("validation", "val-a")):
        low = np.linspace(0.0, 1.0, 36, dtype=np.float32).reshape(6, 6)
        high = np.repeat(np.repeat(low, 2, axis=0), 2, axis=1)
        low_path = root / "lr" / f"{sample_id}.npy"
        high_path = root / "hr" / f"{sample_id}.npy"
        low_path.parent.mkdir(parents=True, exist_ok=True)
        high_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(low_path, low)
        np.save(high_path, high)
        rows.append(
            {
                "sample_id": sample_id,
                "lr_path": low_path.relative_to(root).as_posix(),
                "hr_path": high_path.relative_to(root).as_posix(),
                "split": split,
            }
        )
    manifest = root / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "lr_path", "hr_path", "split"))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_small_end_to_end_manifest_training_smoke(tmp_path: Path) -> None:
    manifest = _write_manifest_fixture(tmp_path)
    config = _config(max_steps=1, warmup_steps=0, validation_interval=1)
    train_loader, validation_loader = create_training_loaders(
        TrainingDataConfig(manifest, tmp_path, validation_split="validation"), config
    )
    trainer = ConditionedRestorationTrainer(TinyConditionedModel(), config)

    summary = trainer.fit(train_loader, validation_loader, max_steps=1)

    assert summary.completed_steps == 1
    assert summary.validation[0].image_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_optional_cuda_training_smoke() -> None:
    trainer = ConditionedRestorationTrainer(
        TinyConditionedModel(),
        _config(device="cuda", max_steps=1, warmup_steps=0, validation_interval=1),
    )

    summary = trainer.fit(_loader(batch_size=1), _loader(batch_size=1), max_steps=1)

    assert summary.device.startswith("cuda")
