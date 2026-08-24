"""Core conditioned restoration trainer without Milestone 20 resilience features."""

from __future__ import annotations

import hashlib
import math
import os
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from .checkpoints import resolve_device
from .config import CONDITIONED_CHECKPOINT_CONFIG, ModelConfig, build_model
from .data import DatasetValidationError
from .degradations import DegradationConfig, degrade_sem_image
from .losses import CharbonnierLoss
from .metrics import compute_reference_metrics
from .training_data import PairedSEMDataset


class TrainingConfigurationError(ValueError):
    """Raised when core trainer settings are malformed."""


class TrainingRuntimeError(RuntimeError):
    """Raised when a training or validation invariant fails at runtime."""


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Validated settings supported by the Milestone 19 core trainer."""

    seed: int = 2026
    device: str = "cpu"
    deterministic: bool = True
    batch_size: int = 16
    num_workers: int = 0
    max_steps: int = 5000
    validation_interval: int = 250
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_steps: int = 100
    gradient_clip_norm: float = 1.0
    charbonnier_epsilon: float = 1e-3
    d4_augmentation: bool = True
    synthetic_probability: float = 0.0
    metric_data_range: float = 1.0
    metric_data_min: float = 0.0
    metric_range_policy: str = "clip"

    def __post_init__(self) -> None:
        integer_minima = {
            "seed": (self.seed, 0),
            "batch_size": (self.batch_size, 1),
            "num_workers": (self.num_workers, 0),
            "max_steps": (self.max_steps, 1),
            "validation_interval": (self.validation_interval, 1),
            "warmup_steps": (self.warmup_steps, 0),
        }
        for name, (value, minimum) in integer_minima.items():
            if type(value) is not int or value < minimum:
                raise TrainingConfigurationError(
                    f"{name} must be an integer greater than or equal to {minimum}"
                )
        numeric_minima = {
            "learning_rate": (self.learning_rate, 0.0, False),
            "weight_decay": (self.weight_decay, 0.0, True),
            "gradient_clip_norm": (self.gradient_clip_norm, 0.0, False),
            "charbonnier_epsilon": (self.charbonnier_epsilon, 0.0, False),
            "metric_data_range": (self.metric_data_range, 0.0, False),
        }
        for name, (value, minimum, inclusive) in numeric_minima.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TrainingConfigurationError(f"{name} must be numeric")
            valid_bound = value >= minimum if inclusive else value > minimum
            if not math.isfinite(value) or not valid_bound:
                operator = ">=" if inclusive else ">"
                raise TrainingConfigurationError(f"{name} must be finite and {operator} {minimum}")
        if self.warmup_steps > self.max_steps:
            raise TrainingConfigurationError("warmup_steps cannot exceed max_steps")
        if type(self.deterministic) is not bool or type(self.d4_augmentation) is not bool:
            raise TrainingConfigurationError("deterministic and d4_augmentation must be booleans")
        if (
            isinstance(self.synthetic_probability, bool)
            or not isinstance(self.synthetic_probability, (int, float))
            or not math.isfinite(self.synthetic_probability)
            or not 0.0 <= self.synthetic_probability <= 1.0
        ):
            raise TrainingConfigurationError("synthetic_probability must be in [0, 1]")
        if not math.isfinite(self.metric_data_min):
            raise TrainingConfigurationError("metric_data_min must be finite")
        if self.metric_range_policy not in ("reject", "clip"):
            raise TrainingConfigurationError("metric_range_policy must be 'reject' or 'clip'")


@dataclass(frozen=True, slots=True)
class TrainingDataConfig:
    """Manifest and split locations for paired training integration."""

    manifest_path: Path
    dataset_root: Path
    train_split: str = "train"
    validation_split: str = "val_ood"


@dataclass(frozen=True, slots=True)
class TrainingStepSummary:
    step: int
    epoch: int
    loss: float
    learning_rate: float
    batch_size: int
    synthetic_samples: int

    def as_dict(self) -> dict[str, int | float]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "synthetic_samples": self.synthetic_samples,
        }


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    step: int
    image_count: int
    mean_loss: float
    mean_psnr_db: float
    mean_ssim: float

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "step": self.step,
            "image_count": self.image_count,
            "mean_loss": self.mean_loss,
            "mean_psnr_db": _safe_number(self.mean_psnr_db),
            "mean_ssim": self.mean_ssim,
        }


@dataclass(frozen=True, slots=True)
class TrainingRunSummary:
    completed_steps: int
    completed_epochs: int
    training: tuple[TrainingStepSummary, ...]
    validation: tuple[ValidationSummary, ...]
    device: str
    deterministic: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "completed_steps": self.completed_steps,
            "completed_epochs": self.completed_epochs,
            "training": [item.as_dict() for item in self.training],
            "validation": [item.as_dict() for item in self.validation],
            "device": self.device,
            "deterministic": self.deterministic,
            "amp_enabled": False,
            "ema_enabled": False,
            "checkpoint_writes": False,
        }


def _safe_number(value: float) -> float | str:
    return "Infinity" if math.isinf(value) and value > 0 else value


def configure_training_seed(seed: int, *, deterministic: bool) -> torch.Generator:
    """Configure the historical Python/NumPy/PyTorch seed surfaces."""

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.use_deterministic_algorithms(False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def seed_dataloader_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def cosine_warmup_factor(step: int, *, max_steps: int, warmup_steps: int) -> float:
    """Historical linear-warmup/cosine-decay multiplier."""

    if type(step) is not int or step < 0:
        raise TrainingConfigurationError("scheduler step must be a non-negative integer")
    if type(max_steps) is not int or max_steps < 1:
        raise TrainingConfigurationError("max_steps must be positive")
    if type(warmup_steps) is not int or not 0 <= warmup_steps <= max_steps:
        raise TrainingConfigurationError("warmup_steps must be in [0, max_steps]")
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    remaining = max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / remaining))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def create_training_loaders(
    data: TrainingDataConfig,
    config: TrainerConfig,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Construct deterministic loaders from validated paired manifests."""

    train_dataset = PairedSEMDataset.from_manifest(
        data.manifest_path, data.dataset_root, split=data.train_split
    )
    validation_dataset = PairedSEMDataset.from_manifest(
        data.manifest_path, data.dataset_root, split=data.validation_split
    )
    generator = configure_training_seed(config.seed, deterministic=config.deterministic)
    common = {
        "num_workers": config.num_workers,
        "pin_memory": config.device.startswith("cuda"),
        "persistent_workers": config.num_workers > 0,
        "worker_init_fn": seed_dataloader_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config.batch_size, len(train_dataset)),
        shuffle=True,
        generator=generator,
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=min(config.batch_size, len(validation_dataset)),
        shuffle=False,
        **common,
    )
    return train_loader, validation_loader


def _stable_choice(seed: int, sample_id: str, epoch: int, purpose: str) -> float:
    payload = f"semirestore-trainer-v1\0{purpose}\0{seed}\0{epoch}\0{sample_id}".encode()
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(1 << 64)


def _stable_transform(seed: int, sample_id: str, epoch: int) -> int:
    return int(_stable_choice(seed, sample_id, epoch, "d4") * 8) % 8


def _apply_d4(image: torch.Tensor, transform: int) -> torch.Tensor:
    result = torch.rot90(image, transform % 4, dims=(-2, -1))
    if transform >= 4:
        result = torch.flip(result, dims=(-1,))
    return result.contiguous()


def _sample_ids(metadata: Mapping[str, Any], batch_size: int) -> tuple[str, ...]:
    raw = metadata.get("sample_id")
    if isinstance(raw, str):
        values = (raw,)
    elif isinstance(raw, Sequence):
        values = tuple(raw)
    else:
        raise TrainingRuntimeError("Batch metadata must provide sample_id values")
    if len(values) != batch_size or any(
        not isinstance(value, str) or not value for value in values
    ):
        raise TrainingRuntimeError("Batch metadata has invalid sample_id values")
    return values


Batch = tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]


class ConditionedRestorationTrainer:
    """Small, testable training core for the conditioned NAF-SR model."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        *,
        degradation_config: DegradationConfig | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        loss_function: nn.Module | None = None,
    ) -> None:
        if config.synthetic_probability > 0 and degradation_config is None:
            raise TrainingConfigurationError(
                "synthetic_probability requires a degradation configuration"
            )
        if getattr(model, "statistics_conditioning", True) is not True:
            raise TrainingConfigurationError("Conditioned training requires active conditioning")
        self.config = config
        self.device = resolve_device(config.device)
        self.model = model.to(self.device)
        self.degradation_config = degradation_config
        self.loss_function = (
            CharbonnierLoss(config.charbonnier_epsilon)
            if loss_function is None
            else loss_function
        ).to(self.device)
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = scheduler or torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: cosine_warmup_factor(
                step, max_steps=config.max_steps, warmup_steps=config.warmup_steps
            ),
        )
        self.step = 0
        self.epoch = 0

    @classmethod
    def build_conditioned(
        cls,
        config: TrainerConfig,
        *,
        model_config: ModelConfig = CONDITIONED_CHECKPOINT_CONFIG,
        degradation_config: DegradationConfig | None = None,
        model_factory: Callable[[ModelConfig], nn.Module] = build_model,
    ) -> ConditionedRestorationTrainer:
        model_config.require_checkpoint_compatible()
        if not model_config.statistics_conditioning:
            raise TrainingConfigurationError("Model configuration must enable conditioning")
        configure_training_seed(config.seed, deterministic=config.deterministic)
        return cls(
            model_factory(model_config),
            config,
            degradation_config=degradation_config,
        )

    def _prepare_training_batch(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor, int]:
        degraded, target, metadata = batch
        if degraded.ndim != 4 or target.ndim != 4 or degraded.shape[0] != target.shape[0]:
            raise TrainingRuntimeError("Training batch must contain aligned NCHW tensors")
        sample_ids = _sample_ids(metadata, degraded.shape[0])
        prepared_low: list[torch.Tensor] = []
        prepared_high: list[torch.Tensor] = []
        synthetic_count = 0
        for index, sample_id in enumerate(sample_ids):
            low_item = degraded[index]
            high_item = target[index]
            if _stable_choice(self.config.seed, sample_id, self.epoch, "synthetic") < float(
                self.config.synthetic_probability
            ):
                assert self.degradation_config is not None
                low_item = degrade_sem_image(
                    high_item,
                    self.degradation_config,
                    sample_id=sample_id,
                    base_seed=self.config.seed,
                    epoch=self.epoch,
                ).tensor
                synthetic_count += 1
            if self.config.d4_augmentation:
                transform = _stable_transform(self.config.seed, sample_id, self.epoch)
                low_item = _apply_d4(low_item, transform)
                high_item = _apply_d4(high_item, transform)
            prepared_low.append(low_item)
            prepared_high.append(high_item)
        low_batch = torch.stack(prepared_low).to(self.device)
        high_batch = torch.stack(prepared_high).to(self.device)
        return low_batch, high_batch, synthetic_count

    def train_step(self, batch: Batch) -> TrainingStepSummary:
        """Run one finite CPU/CUDA optimizer step without checkpoint side effects."""

        degraded, target, synthetic_count = self._prepare_training_batch(batch)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        prediction = self.model(degraded)
        if prediction.shape != target.shape or not bool(torch.isfinite(prediction).all().item()):
            raise TrainingRuntimeError("Model produced an invalid training prediction")
        loss = self.loss_function(prediction, target)
        if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
            raise TrainingRuntimeError(f"Non-finite training loss at step {self.step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        return TrainingStepSummary(
            step=self.step,
            epoch=self.epoch,
            loss=float(loss.detach().item()),
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            batch_size=int(degraded.shape[0]),
            synthetic_samples=synthetic_count,
        )

    def validate(self, loader: DataLoader[Any]) -> ValidationSummary:
        """Evaluate paired references without updating parameters or optimizer state."""

        if len(loader) == 0:
            raise TrainingRuntimeError("Validation loader is empty")
        was_training = self.model.training
        self.model.eval()
        losses: list[float] = []
        psnr_values: list[float] = []
        ssim_values: list[float] = []
        image_count = 0
        try:
            with torch.inference_mode():
                for degraded, target, metadata in loader:
                    if degraded.ndim != 4 or target.ndim != 4:
                        raise TrainingRuntimeError("Validation batch must contain NCHW tensors")
                    ids = _sample_ids(metadata, degraded.shape[0])
                    degraded = degraded.to(self.device)
                    target = target.to(self.device)
                    prediction = self.model(degraded)
                    if prediction.shape != target.shape or not bool(
                        torch.isfinite(prediction).all().item()
                    ):
                        raise TrainingRuntimeError(
                            "Model produced an invalid validation prediction"
                        )
                    loss = self.loss_function(prediction, target)
                    if not bool(torch.isfinite(loss).item()):
                        raise TrainingRuntimeError("Validation loss is non-finite")
                    metrics = compute_reference_metrics(
                        prediction,
                        target,
                        data_range=self.config.metric_data_range,
                        data_min=self.config.metric_data_min,
                        range_policy=self.config.metric_range_policy,  # type: ignore[arg-type]
                        sample_ids=ids,
                    )
                    losses.extend([float(loss.item())] * len(ids))
                    psnr_values.extend(item.psnr_db for item in metrics.per_image)
                    ssim_values.extend(item.ssim for item in metrics.per_image)
                    image_count += len(ids)
        finally:
            self.model.train(was_training)
        return ValidationSummary(
            step=self.step,
            image_count=image_count,
            mean_loss=float(sum(losses) / len(losses)),
            mean_psnr_db=float(torch.tensor(psnr_values, dtype=torch.float64).mean().item()),
            mean_ssim=float(sum(ssim_values) / len(ssim_values)),
        )

    def fit(
        self,
        train_loader: DataLoader[Any],
        validation_loader: DataLoader[Any],
        *,
        max_steps: int | None = None,
    ) -> TrainingRunSummary:
        """Run the bounded core loop; no training begins until this is called."""

        target_steps = self.config.max_steps if max_steps is None else max_steps
        if type(target_steps) is not int or not 1 <= target_steps <= self.config.max_steps:
            raise TrainingConfigurationError("max_steps must be in [1, config.max_steps]")
        if len(train_loader) == 0:
            raise TrainingRuntimeError("Training loader is empty")
        training: list[TrainingStepSummary] = []
        validation: list[ValidationSummary] = []
        iterator = iter(train_loader)
        while self.step < target_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                self.epoch += 1
                iterator = iter(train_loader)
                batch = next(iterator)
            training.append(self.train_step(batch))
            if self.step % self.config.validation_interval == 0 or self.step == target_steps:
                validation.append(self.validate(validation_loader))
        return TrainingRunSummary(
            completed_steps=self.step,
            completed_epochs=self.epoch,
            training=tuple(training),
            validation=tuple(validation),
            device=str(self.device),
            deterministic=self.config.deterministic,
        )


def create_manifest_training(
    data: TrainingDataConfig,
    config: TrainerConfig,
    *,
    model_config: ModelConfig = CONDITIONED_CHECKPOINT_CONFIG,
    degradation_config: DegradationConfig | None = None,
) -> tuple[ConditionedRestorationTrainer, DataLoader[Any], DataLoader[Any]]:
    """Integrate conditioned construction with paired manifest loaders."""

    try:
        loaders = create_training_loaders(data, config)
    except DatasetValidationError:
        raise
    trainer = ConditionedRestorationTrainer.build_conditioned(
        config,
        model_config=model_config,
        degradation_config=degradation_config,
    )
    return trainer, *loaders


__all__ = [
    "ConditionedRestorationTrainer",
    "TrainerConfig",
    "TrainingConfigurationError",
    "TrainingDataConfig",
    "TrainingRunSummary",
    "TrainingRuntimeError",
    "TrainingStepSummary",
    "ValidationSummary",
    "configure_training_seed",
    "cosine_warmup_factor",
    "create_manifest_training",
    "create_training_loaders",
]
