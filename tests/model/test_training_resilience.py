from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from semirestore.trainer import (
    ConditionedRestorationTrainer,
    TrainerConfig,
    TrainingRuntimeError,
)
from semirestore.training_checkpoints import (
    TrainingCheckpointError,
    TrainingCheckpointManager,
    atomic_torch_save,
    load_training_checkpoint,
)


class ResilienceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, str]]]):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, str]]:
        low = torch.linspace(0.0, 1.0, 36).reshape(1, 6, 6) + index * 0.01
        high = F.interpolate(low[None], scale_factor=2, mode="nearest")[0] * 0.8 + 0.05
        return low, high, {"sample_id": f"resilient-{index}"}


class ResilientTinyModel(nn.Module):
    statistics_conditioning = True

    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.4))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.interpolate(inputs, scale_factor=2, mode="nearest") * self.gain + self.bias


def _config(**overrides: object) -> TrainerConfig:
    values: dict[str, object] = {
        "seed": 2026,
        "device": "cpu",
        "deterministic": True,
        "batch_size": 2,
        "num_workers": 0,
        "max_steps": 4,
        "validation_interval": 2,
        "learning_rate": 0.02,
        "weight_decay": 0.0,
        "warmup_steps": 1,
        "gradient_clip_norm": 0.5,
        "charbonnier_epsilon": 1e-3,
        "d4_augmentation": False,
        "synthetic_probability": 0.0,
        "metric_data_range": 1.0,
        "metric_data_min": 0.0,
        "metric_range_policy": "clip",
        "amp_enabled": True,
        "ema_enabled": True,
        "ema_decay": 0.9,
        "best_validation_metric": "psnr_db",
    }
    values.update(overrides)
    return TrainerConfig(**values)


def _loader() -> DataLoader[Any]:
    return DataLoader(
        ResilienceDataset(),
        batch_size=2,
        shuffle=False,
        generator=torch.Generator().manual_seed(77),
    )


def _state_equal(first: dict[str, Any], second: dict[str, Any]) -> bool:
    if first.keys() != second.keys():
        return False
    for key in first:
        left, right = first[key], second[key]
        if isinstance(left, torch.Tensor):
            if not torch.equal(left, right):
                return False
        elif isinstance(left, dict):
            if not _state_equal(left, right):
                return False
        elif left != right:
            return False
    return True


def test_cpu_amp_configuration_remains_safe_fp32() -> None:
    trainer = ConditionedRestorationTrainer(ResilientTinyModel(), _config(amp_enabled=True))

    summary = trainer.fit(_loader(), _loader(), max_steps=1)

    assert trainer.amp_enabled is False
    assert trainer.scaler.is_enabled() is False
    assert summary.amp_enabled is False
    assert next(trainer.model.parameters()).dtype == torch.float32


def test_gradient_clipping_is_applied_after_unscale(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[tuple[float, bool]] = []
    original = torch.nn.utils.clip_grad_norm_

    def recording_clip(
        parameters: Any, max_norm: float, *, error_if_nonfinite: bool
    ) -> torch.Tensor:
        recorded.append((max_norm, error_if_nonfinite))
        return original(parameters, max_norm, error_if_nonfinite=error_if_nonfinite)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recording_clip)
    trainer = ConditionedRestorationTrainer(
        ResilientTinyModel(), _config(gradient_clip_norm=0.125)
    )

    trainer.train_step(next(iter(_loader())))

    assert recorded == [(0.125, True)]


def test_ema_matches_historical_lerp_update() -> None:
    model = ResilientTinyModel()
    trainer = ConditionedRestorationTrainer(model, _config(ema_decay=0.9))
    assert trainer.ema_model is not None
    initial_ema = deepcopy(trainer.ema_model.state_dict())

    trainer.train_step(next(iter(_loader())))

    for name, value in trainer.model.state_dict().items():
        expected = initial_ema[name] * 0.9 + value * 0.1
        torch.testing.assert_close(trainer.ema_model.state_dict()[name], expected)


def test_uninterrupted_and_resumed_training_are_equivalent(tmp_path: Path) -> None:
    initial = deepcopy(ResilientTinyModel().state_dict())

    uninterrupted_model = ResilientTinyModel()
    uninterrupted_model.load_state_dict(initial)
    uninterrupted = ConditionedRestorationTrainer(uninterrupted_model, _config())
    uninterrupted.fit(_loader(), _loader(), max_steps=4)

    interrupted_model = ResilientTinyModel()
    interrupted_model.load_state_dict(initial)
    manager = TrainingCheckpointManager(tmp_path / "checkpoints", keep_last=2)
    interrupted = ConditionedRestorationTrainer(
        interrupted_model,
        _config(),
        checkpoint_manager=manager,
    )
    interrupted.fit(_loader(), _loader(), max_steps=2)

    resumed = ConditionedRestorationTrainer(
        ResilientTinyModel(),
        _config(),
        checkpoint_manager=manager,
    )
    resumed.resume(manager.last_path)
    resumed.fit(_loader(), _loader(), max_steps=4)

    assert resumed.step == uninterrupted.step == 4
    assert resumed.epoch == uninterrupted.epoch
    assert resumed.batch_in_epoch == uninterrupted.batch_in_epoch
    assert _state_equal(resumed.model.state_dict(), uninterrupted.model.state_dict())
    assert _state_equal(resumed.optimizer.state_dict(), uninterrupted.optimizer.state_dict())
    assert resumed.scheduler.state_dict() == uninterrupted.scheduler.state_dict()
    assert resumed.ema_model is not None and uninterrupted.ema_model is not None
    assert _state_equal(resumed.ema_model.state_dict(), uninterrupted.ema_model.state_dict())


def test_resume_rejects_incompatible_configuration(tmp_path: Path) -> None:
    manager = TrainingCheckpointManager(tmp_path)
    trainer = ConditionedRestorationTrainer(
        ResilientTinyModel(), _config(), checkpoint_manager=manager
    )
    trainer.fit(_loader(), _loader(), max_steps=2)
    incompatible = ConditionedRestorationTrainer(
        ResilientTinyModel(), _config(learning_rate=0.01)
    )

    with pytest.raises(TrainingCheckpointError, match="configuration is incompatible"):
        incompatible.resume(manager.last_path)


def test_resume_rejects_best_inference_checkpoint(tmp_path: Path) -> None:
    path = atomic_torch_save(
        {
            "format_version": 1,
            "checkpoint_role": "best_inference",
            "model_state_dict": ResilientTinyModel().state_dict(),
        },
        tmp_path / "best.pt",
    )

    with pytest.raises(TrainingCheckpointError, match="missing fields|not resumable"):
        load_training_checkpoint(path)


def test_best_checkpoint_excludes_resume_only_state(tmp_path: Path) -> None:
    manager = TrainingCheckpointManager(tmp_path)
    trainer = ConditionedRestorationTrainer(
        ResilientTinyModel(), _config(), checkpoint_manager=manager
    )

    trainer.fit(_loader(), _loader(), max_steps=2)
    payload = torch.load(manager.best_path, map_location="cpu", weights_only=True)

    assert payload["checkpoint_role"] == "best_inference"
    assert "model_state_dict" in payload
    assert "optimizer_state_dict" not in payload
    assert "scheduler_state_dict" not in payload
    assert "scaler_state_dict" not in payload
    assert "ema_state_dict" not in payload


def test_bounded_rotation_keeps_only_named_recent_archives(tmp_path: Path) -> None:
    manager = TrainingCheckpointManager(tmp_path, keep_last=2)
    unrelated = tmp_path / "notes.pt"
    unrelated.write_bytes(b"preserve")
    for step in (1, 2, 3):
        manager.save_resume({"step": step}, step=step)

    assert sorted(path.name for path in tmp_path.glob("last-step-*.pt")) == [
        "last-step-00000002.pt",
        "last-step-00000003.pt",
    ]
    assert unrelated.read_bytes() == b"preserve"
    assert manager.last_path.is_file()


def test_atomic_failure_cleans_partial_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_save(payload: object, path: str | Path) -> None:
        del payload
        Path(path).write_bytes(b"incomplete")
        raise RuntimeError("simulated serialization failure")

    monkeypatch.setattr(torch, "save", failing_save)

    with pytest.raises(TrainingCheckpointError, match="atomically write"):
        atomic_torch_save({"value": torch.tensor(1)}, tmp_path / "last.pt")

    assert not (tmp_path / "last.pt").exists()
    assert list(tmp_path.glob("*.partial")) == []
    assert list(tmp_path.glob(".*.partial")) == []


class NonFiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor) -> torch.Tensor:
        return value.clone()

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        return (torch.full_like(gradient, float("nan")),)


class BadGradientLoss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        del target
        return NonFiniteGradient.apply(prediction).mean()


def test_non_finite_gradient_stops_optimizer_update() -> None:
    model = ResilientTinyModel()
    trainer = ConditionedRestorationTrainer(model, _config(), loss_function=BadGradientLoss())
    before = deepcopy(model.state_dict())

    with pytest.raises(TrainingRuntimeError, match="Non-finite training gradient"):
        trainer.train_step(next(iter(_loader())))

    assert _state_equal(model.state_dict(), before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_amp_uses_current_scaler_and_autocast() -> None:
    trainer = ConditionedRestorationTrainer(
        ResilientTinyModel(), _config(device="cuda", amp_enabled=True)
    )

    summary = trainer.fit(_loader(), _loader(), max_steps=1)

    assert trainer.amp_enabled is True
    assert trainer.scaler.is_enabled() is True
    assert summary.amp_enabled is True
