"""Conditioned restoration trainer with safe resumable state."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
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
from .training_checkpoints import (
    TRAINING_CHECKPOINT_VERSION,
    TrainingCheckpointError,
    TrainingCheckpointManager,
    load_training_checkpoint,
)
from .training_data import PairedSEMDataset


class TrainingConfigurationError(ValueError):
    """Raised when core trainer settings are malformed."""


class TrainingRuntimeError(RuntimeError):
    """Raised when a training or validation invariant fails at runtime."""


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Validated settings for reproducible resilient training."""

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
    amp_enabled: bool = True
    ema_enabled: bool = True
    ema_decay: float = 0.999
    best_validation_metric: str = "psnr_db"

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
        boolean_values = (
            self.deterministic,
            self.d4_augmentation,
            self.amp_enabled,
            self.ema_enabled,
        )
        if any(type(value) is not bool for value in boolean_values):
            raise TrainingConfigurationError(
                "deterministic, d4_augmentation, amp_enabled, and ema_enabled must be booleans"
            )
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
        if (
            isinstance(self.ema_decay, bool)
            or not isinstance(self.ema_decay, (int, float))
            or not math.isfinite(self.ema_decay)
            or not 0.0 <= self.ema_decay < 1.0
        ):
            raise TrainingConfigurationError("ema_decay must be finite and in [0, 1)")
        if self.best_validation_metric not in ("psnr_db", "ssim", "loss"):
            raise TrainingConfigurationError(
                "best_validation_metric must be psnr_db, ssim, or loss"
            )


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
    weights_source: str = "raw"

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "step": self.step,
            "image_count": self.image_count,
            "mean_loss": self.mean_loss,
            "mean_psnr_db": _safe_number(self.mean_psnr_db),
            "mean_ssim": self.mean_ssim,
            "weights_source": self.weights_source,
        }


@dataclass(frozen=True, slots=True)
class TrainingRunSummary:
    completed_steps: int
    completed_epochs: int
    training: tuple[TrainingStepSummary, ...]
    validation: tuple[ValidationSummary, ...]
    device: str
    deterministic: bool
    amp_enabled: bool
    ema_enabled: bool
    checkpoint_writes: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "completed_steps": self.completed_steps,
            "completed_epochs": self.completed_epochs,
            "training": [item.as_dict() for item in self.training],
            "validation": [item.as_dict() for item in self.validation],
            "device": self.device,
            "deterministic": self.deterministic,
            "amp_enabled": self.amp_enabled,
            "ema_enabled": self.ema_enabled,
            "checkpoint_writes": self.checkpoint_writes,
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


def _configuration_fingerprint(config: TrainerConfig) -> str:
    rendered = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode()).hexdigest()


def _model_fingerprint(model: nn.Module) -> str:
    structure = [
        (name, list(value.shape), str(value.dtype))
        for name, value in model.state_dict().items()
    ]
    rendered = json.dumps(structure, separators=(",", ":"))
    return hashlib.sha256(rendered.encode()).hexdigest()


@torch.no_grad()
def _update_ema(ema_model: nn.Module, model: nn.Module, *, decay: float) -> None:
    ema_parameters = dict(ema_model.named_parameters())
    for name, parameter in model.named_parameters():
        ema_parameters[name].lerp_(parameter.detach(), 1.0 - decay)
    ema_buffers = dict(ema_model.named_buffers())
    for name, buffer in model.named_buffers():
        ema_buffers[name].copy_(buffer.detach())


def _optimizer_to(optimizer: Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


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
        checkpoint_manager: TrainingCheckpointManager | None = None,
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
        self.amp_enabled = config.amp_enabled and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp_enabled)
        self.ema_model = (
            deepcopy(self.model).eval().requires_grad_(False)
            if config.ema_enabled
            else None
        )
        self.checkpoint_manager = checkpoint_manager
        self.step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.best_metric_value = (
            math.inf if config.best_validation_metric == "loss" else -math.inf
        )
        self.best_weights_source = "raw"
        self._epoch_loader_generator_state: torch.Tensor | None = None
        self._resume_loader_generator_state: torch.Tensor | None = None

    @classmethod
    def build_conditioned(
        cls,
        config: TrainerConfig,
        *,
        model_config: ModelConfig = CONDITIONED_CHECKPOINT_CONFIG,
        degradation_config: DegradationConfig | None = None,
        model_factory: Callable[[ModelConfig], nn.Module] = build_model,
        checkpoint_manager: TrainingCheckpointManager | None = None,
    ) -> ConditionedRestorationTrainer:
        model_config.require_checkpoint_compatible()
        if not model_config.statistics_conditioning:
            raise TrainingConfigurationError("Model configuration must enable conditioning")
        configure_training_seed(config.seed, deterministic=config.deterministic)
        return cls(
            model_factory(model_config),
            config,
            degradation_config=degradation_config,
            checkpoint_manager=checkpoint_manager,
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
        """Run one finite CPU/CUDA optimizer step with optional CUDA AMP."""

        degraded, target, synthetic_count = self._prepare_training_batch(batch)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            prediction = self.model(degraded)
            loss = self.loss_function(prediction, target)
        if prediction.shape != target.shape or not bool(torch.isfinite(prediction).all().item()):
            raise TrainingRuntimeError("Model produced an invalid training prediction")
        if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
            raise TrainingRuntimeError(f"Non-finite training loss at step {self.step + 1}")
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        try:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm,
                error_if_nonfinite=True,
            )
        except RuntimeError as error:
            self.optimizer.zero_grad(set_to_none=True)
            raise TrainingRuntimeError(
                f"Non-finite training gradient at step {self.step + 1}"
            ) from error
        if not bool(torch.isfinite(gradient_norm).item()):
            self.optimizer.zero_grad(set_to_none=True)
            raise TrainingRuntimeError(
                f"Non-finite training gradient at step {self.step + 1}"
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        if self.ema_model is not None:
            _update_ema(self.ema_model, self.model, decay=self.config.ema_decay)
        self.step += 1
        self.batch_in_epoch += 1
        return TrainingStepSummary(
            step=self.step,
            epoch=self.epoch,
            loss=float(loss.detach().item()),
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            batch_size=int(degraded.shape[0]),
            synthetic_samples=synthetic_count,
        )

    def validate(
        self,
        loader: DataLoader[Any],
        *,
        model: nn.Module | None = None,
        weights_source: str = "raw",
    ) -> ValidationSummary:
        """Evaluate paired references without updating parameters or optimizer state."""

        if len(loader) == 0:
            raise TrainingRuntimeError("Validation loader is empty")
        evaluated_model = self.model if model is None else model
        was_training = evaluated_model.training
        evaluated_model.eval()
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
                    with torch.autocast(
                        device_type=self.device.type,
                        enabled=self.amp_enabled,
                    ):
                        prediction = evaluated_model(degraded)
                        loss = self.loss_function(prediction, target)
                    if prediction.shape != target.shape or not bool(
                        torch.isfinite(prediction).all().item()
                    ):
                        raise TrainingRuntimeError(
                            "Model produced an invalid validation prediction"
                        )
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
            evaluated_model.train(was_training)
        return ValidationSummary(
            step=self.step,
            image_count=image_count,
            mean_loss=float(sum(losses) / len(losses)),
            mean_psnr_db=float(torch.tensor(psnr_values, dtype=torch.float64).mean().item()),
            mean_ssim=float(sum(ssim_values) / len(ssim_values)),
            weights_source=weights_source,
        )

    def _metric_value(self, summary: ValidationSummary) -> float:
        values = {
            "psnr_db": summary.mean_psnr_db,
            "ssim": summary.mean_ssim,
            "loss": summary.mean_loss,
        }
        return values[self.config.best_validation_metric]

    def _is_better(self, value: float, reference: float) -> bool:
        if self.config.best_validation_metric == "loss":
            return value < reference
        return value > reference

    def _select_validation(
        self,
        raw: ValidationSummary,
        ema: ValidationSummary | None,
    ) -> ValidationSummary:
        if ema is None:
            return raw
        return ema if self._is_better(self._metric_value(ema), self._metric_value(raw)) else raw

    def _resume_payload(self) -> dict[str, Any]:
        return {
            "format_version": TRAINING_CHECKPOINT_VERSION,
            "checkpoint_role": "training_resume",
            "configuration_fingerprint": _configuration_fingerprint(self.config),
            "model_fingerprint": _model_fingerprint(self.model),
            "model_state_dict": {
                name: value.detach().cpu() for name, value in self.model.state_dict().items()
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "ema_state_dict": (
                None
                if self.ema_model is None
                else {
                    name: value.detach().cpu()
                    for name, value in self.ema_model.state_dict().items()
                }
            ),
            "step": self.step,
            "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch,
            "best_metric_name": self.config.best_validation_metric,
            "best_metric_value": self.best_metric_value,
            "best_weights_source": self.best_weights_source,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "epoch_loader_generator_state": self._epoch_loader_generator_state,
        }

    def _best_payload(self, model: nn.Module) -> dict[str, Any]:
        return {
            "format_version": TRAINING_CHECKPOINT_VERSION,
            "checkpoint_role": "best_inference",
            "model_state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "model_fingerprint": _model_fingerprint(model),
            "step": self.step,
            "validation_metric": self.config.best_validation_metric,
            "validation_metric_value": self.best_metric_value,
            "selected_weights": self.best_weights_source,
        }

    def _save_validation_checkpoint(
        self,
        selected: ValidationSummary,
    ) -> None:
        if self.checkpoint_manager is None:
            return
        selected_value = self._metric_value(selected)
        improved = self._is_better(selected_value, self.best_metric_value)
        if improved:
            self.best_metric_value = selected_value
            self.best_weights_source = selected.weights_source
        self.checkpoint_manager.save_resume(self._resume_payload(), step=self.step)
        if improved:
            selected_model = self.ema_model if selected.weights_source == "ema" else self.model
            assert selected_model is not None
            self.checkpoint_manager.save_best(self._best_payload(selected_model))

    def resume(self, path: str | Path) -> None:
        """Restore compatible safe training state before continuing ``fit``."""

        try:
            payload = load_training_checkpoint(path)
        except TrainingCheckpointError:
            raise
        if payload["configuration_fingerprint"] != _configuration_fingerprint(self.config):
            raise TrainingCheckpointError("Resume configuration is incompatible")
        if payload["model_fingerprint"] != _model_fingerprint(self.model):
            raise TrainingCheckpointError("Resume model architecture is incompatible")
        if payload["best_metric_name"] != self.config.best_validation_metric:
            raise TrainingCheckpointError("Resume best-metric policy is incompatible")
        try:
            self.model.load_state_dict(payload["model_state_dict"], strict=True)
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
            _optimizer_to(self.optimizer, self.device)
            self.scheduler.load_state_dict(payload["scheduler_state_dict"])
            self.scaler.load_state_dict(payload["scaler_state_dict"])
            ema_state = payload.get("ema_state_dict")
            if self.ema_model is None:
                if ema_state is not None:
                    raise TrainingCheckpointError("Resume EMA setting is incompatible")
            elif not isinstance(ema_state, Mapping):
                raise TrainingCheckpointError("Resume checkpoint is missing EMA state")
            else:
                self.ema_model.load_state_dict(ema_state, strict=True)
        except (RuntimeError, ValueError, KeyError) as error:
            raise TrainingCheckpointError("Resume state is incompatible") from error
        step, epoch, batch_in_epoch = (
            payload.get("step"),
            payload.get("epoch"),
            payload.get("batch_in_epoch"),
        )
        if any(type(value) is not int or value < 0 for value in (step, epoch, batch_in_epoch)):
            raise TrainingCheckpointError("Resume step/epoch state is invalid")
        if step > self.config.max_steps:
            raise TrainingCheckpointError("Resume step exceeds configured max_steps")
        best_value = payload.get("best_metric_value")
        best_source = payload.get("best_weights_source")
        if not isinstance(best_value, (int, float)) or best_source not in ("raw", "ema"):
            raise TrainingCheckpointError("Resume best-model state is invalid")
        rng_state = payload.get("torch_rng_state")
        if not isinstance(rng_state, torch.Tensor):
            raise TrainingCheckpointError("Resume RNG state is invalid")
        torch.set_rng_state(rng_state)
        cuda_states = payload.get("cuda_rng_states", [])
        if torch.cuda.is_available() and isinstance(cuda_states, list) and cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)
        loader_state = payload.get("epoch_loader_generator_state")
        if loader_state is not None and not isinstance(loader_state, torch.Tensor):
            raise TrainingCheckpointError("Resume DataLoader generator state is invalid")
        self.step = step
        self.epoch = epoch
        self.batch_in_epoch = batch_in_epoch
        self.best_metric_value = float(best_value)
        self.best_weights_source = str(best_source)
        self._resume_loader_generator_state = loader_state

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
        if target_steps <= self.step:
            raise TrainingConfigurationError("max_steps must be greater than the current step")
        if len(train_loader) == 0:
            raise TrainingRuntimeError("Training loader is empty")
        training: list[TrainingStepSummary] = []
        validation: list[ValidationSummary] = []
        loader_generator = getattr(train_loader, "generator", None)
        if self._resume_loader_generator_state is not None:
            if not isinstance(loader_generator, torch.Generator):
                raise TrainingRuntimeError(
                    "Deterministic resume requires the training DataLoader generator"
                )
            loader_generator.set_state(self._resume_loader_generator_state)
        self._epoch_loader_generator_state = (
            None
            if not isinstance(loader_generator, torch.Generator)
            else loader_generator.get_state()
        )
        iterator = iter(train_loader)
        for _ in range(self.batch_in_epoch):
            try:
                next(iterator)
            except StopIteration as error:
                raise TrainingRuntimeError(
                    "Resume batch position exceeds the current training epoch"
                ) from error
        while self.step < target_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self._epoch_loader_generator_state = (
                    None
                    if not isinstance(loader_generator, torch.Generator)
                    else loader_generator.get_state()
                )
                iterator = iter(train_loader)
                batch = next(iterator)
            training.append(self.train_step(batch))
            if self.step % self.config.validation_interval == 0 or self.step == target_steps:
                raw_validation = self.validate(
                    validation_loader,
                    model=self.model,
                    weights_source="raw",
                )
                ema_validation = (
                    None
                    if self.ema_model is None
                    else self.validate(
                        validation_loader,
                        model=self.ema_model,
                        weights_source="ema",
                    )
                )
                selected = self._select_validation(raw_validation, ema_validation)
                validation.append(selected)
                self._save_validation_checkpoint(selected)
        return TrainingRunSummary(
            completed_steps=self.step,
            completed_epochs=self.epoch,
            training=tuple(training),
            validation=tuple(validation),
            device=str(self.device),
            deterministic=self.config.deterministic,
            amp_enabled=self.amp_enabled,
            ema_enabled=self.ema_model is not None,
            checkpoint_writes=self.checkpoint_manager is not None,
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
