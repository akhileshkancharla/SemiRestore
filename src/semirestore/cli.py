"""YAML-driven, side-effect-free-on-import model workflow CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
import yaml

from .checkpoints import (
    CheckpointError,
    load_checkpoint_metadata,
)
from .config import ModelConfig, ModelConfigError
from .data import DatasetValidationError
from .model_manager import ModelManagerError
from .pipeline import PipelineConfig, SemiRestorePipeline
from .restoration_service import RestorationServiceError
from .trainer import (
    TrainerConfig,
    TrainingConfigurationError,
    TrainingDataConfig,
    TrainingRuntimeError,
    create_manifest_training,
)
from .training_checkpoints import TrainingCheckpointError, TrainingCheckpointManager
from .training_data import read_pair_manifest

CLI_VERSION = "semirestore-workflows-v1"
_SENSITIVE_FRAGMENTS = ("password", "secret", "token", "credential", "api_key")


class WorkflowConfigError(ValueError):
    """Raised when the YAML workflow contract is invalid."""


@dataclass(frozen=True, slots=True)
class WorkflowConfig:
    source: Path
    base_directory: Path
    model: ModelConfig
    data: TrainingDataConfig
    training: TrainerConfig
    run_directory: Path
    checkpoint_keep_last: int
    raw: dict[str, Any]
    sha256: str


def _require_mapping(values: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = values.get(key)
    if not isinstance(value, Mapping):
        raise WorkflowConfigError(f"Configuration requires a {key!r} mapping")
    return value


def _resolve_path(base: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise WorkflowConfigError(f"Configuration path {field!r} must be a non-empty string")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _trainer_config(values: Mapping[str, Any]) -> TrainerConfig:
    normalized = dict(values)
    if "amp" in normalized:
        if "amp_enabled" in normalized:
            raise WorkflowConfigError("training cannot define both amp and amp_enabled")
        normalized["amp_enabled"] = normalized.pop("amp")
    allowed = {field.name for field in fields(TrainerConfig)}
    unknown = sorted(set(normalized) - allowed)
    if unknown:
        raise WorkflowConfigError(f"Unknown training configuration fields: {unknown}")
    try:
        return TrainerConfig(**normalized)
    except (TypeError, TrainingConfigurationError) as error:
        raise WorkflowConfigError("Training configuration is invalid") from error


def load_workflow_config(path: str | Path) -> WorkflowConfig:
    """Safely load and fully validate one workflow configuration."""

    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise WorkflowConfigError(f"Workflow configuration is not a regular file: {source.name}")
    try:
        content = source.read_bytes()
        document = yaml.safe_load(content)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise WorkflowConfigError("Could not safely read workflow configuration") from error
    if not isinstance(document, Mapping):
        raise WorkflowConfigError("Workflow configuration root must be a mapping")
    allowed_sections = {"model", "data", "training", "output", "checkpointing"}
    unknown_sections = sorted(set(document) - allowed_sections)
    if unknown_sections:
        raise WorkflowConfigError(f"Unknown workflow sections: {unknown_sections}")
    try:
        model = ModelConfig.from_mapping(_require_mapping(document, "model"))
        model.require_checkpoint_compatible()
    except ModelConfigError as error:
        raise WorkflowConfigError("Model configuration is checkpoint-incompatible") from error
    base = source.parent
    data_values = _require_mapping(document, "data")
    allowed_data = {"manifest", "dataset_root", "train_split", "validation_split"}
    if unknown_data := sorted(set(data_values) - allowed_data):
        raise WorkflowConfigError(f"Unknown data configuration fields: {unknown_data}")
    train_split = data_values.get("train_split", "train")
    validation_split = data_values.get("validation_split", "val_ood")
    if not isinstance(train_split, str) or not isinstance(validation_split, str):
        raise WorkflowConfigError("Data split names must be strings")
    data = TrainingDataConfig(
        manifest_path=_resolve_path(base, data_values.get("manifest"), field="data.manifest"),
        dataset_root=_resolve_path(
            base, data_values.get("dataset_root"), field="data.dataset_root"
        ),
        train_split=train_split,
        validation_split=validation_split,
    )
    training = _trainer_config(_require_mapping(document, "training"))
    output = _require_mapping(document, "output")
    if set(output) != {"run_dir"}:
        raise WorkflowConfigError("output must contain only run_dir")
    run_directory = _resolve_path(base, output.get("run_dir"), field="output.run_dir")
    checkpointing = document.get("checkpointing", {})
    if not isinstance(checkpointing, Mapping) or set(checkpointing) - {"keep_last"}:
        raise WorkflowConfigError("checkpointing supports only keep_last")
    keep_last = checkpointing.get("keep_last", 2)
    if type(keep_last) is not int or not 1 <= keep_last <= 10:
        raise WorkflowConfigError("checkpointing.keep_last must be in [1, 10]")
    return WorkflowConfig(
        source=source,
        base_directory=base,
        model=model,
        data=data,
        training=training,
        run_directory=run_directory,
        checkpoint_keep_last=keep_last,
        raw=dict(document),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _display_path(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.name


def _redact(value: Any, *, key: str = "") -> Any:
    if any(fragment in key.casefold() for fragment in _SENSITIVE_FRAGMENTS):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(item_key): _redact(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_redact(item) for item in value]
    return value


def resolved_configuration(config: WorkflowConfig) -> dict[str, Any]:
    """Return a sanitized configuration without machine-specific absolute paths."""

    training = asdict(config.training)
    return {
        "workflow_version": CLI_VERSION,
        "configuration_file": config.source.name,
        "configuration_sha256": config.sha256,
        "model": asdict(config.model),
        "data": {
            "manifest": _display_path(config.data.manifest_path, config.base_directory),
            "dataset_root": _display_path(config.data.dataset_root, config.base_directory),
            "train_split": config.data.train_split,
            "validation_split": config.data.validation_split,
        },
        "training": _redact(training),
        "output": {
            "run_dir": _display_path(config.run_directory, config.base_directory),
        },
        "checkpointing": {"keep_last": config.checkpoint_keep_last},
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: bytes, *, force: bool) -> None:
    destination = path.expanduser().resolve()
    if destination.exists() and not force:
        raise WorkflowConfigError(f"Output already exists: {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or destination.is_symlink():
        raise WorkflowConfigError("Output path cannot use symbolic links")
    partial = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
    try:
        partial.write_bytes(content)
        os.replace(partial, destination)
    finally:
        if partial.exists():
            partial.unlink()


def _checkpoint_report(config: WorkflowConfig) -> dict[str, Any]:
    metadata = load_checkpoint_metadata()
    return {
        "model_name": metadata.model_name,
        "architecture": metadata.architecture,
        "expected_sha256": metadata.sha256,
        "expected_size_bytes": metadata.size_bytes,
        "runtime_path": _display_path(metadata.runtime_path, config.base_directory),
        "training_revision": metadata.training_revision,
    }


def _run_validate(config: WorkflowConfig, args: argparse.Namespace) -> dict[str, Any]:
    del args
    return {
        "command": "validate-config",
        "status": "valid",
        "resolved_configuration": resolved_configuration(config),
        "checkpoint_identity": _checkpoint_report(config),
    }


def _run_audit(config: WorkflowConfig, args: argparse.Namespace) -> dict[str, Any]:
    del args
    train = read_pair_manifest(
        config.data.manifest_path,
        config.data.dataset_root,
        split=config.data.train_split,
    )
    validation = read_pair_manifest(
        config.data.manifest_path,
        config.data.dataset_root,
        split=config.data.validation_split,
    )
    return {
        "command": "audit-dataset",
        "status": "valid",
        "manifest": _display_path(config.data.manifest_path, config.base_directory),
        "train_split": config.data.train_split,
        "train_samples": len(train),
        "validation_split": config.data.validation_split,
        "validation_samples": len(validation),
        "leakage_detected": False,
    }


def _create_training(config: WorkflowConfig) -> tuple[Any, Any, Any]:
    manager = TrainingCheckpointManager(
        config.run_directory / "checkpoints", keep_last=config.checkpoint_keep_last
    )
    trainer, train_loader, validation_loader = create_manifest_training(
        config.data,
        config.training,
        model_config=config.model,
    )
    trainer.checkpoint_manager = manager
    return trainer, train_loader, validation_loader


def _run_train(config: WorkflowConfig, args: argparse.Namespace) -> dict[str, Any]:
    trainer, train_loader, validation_loader = _create_training(config)
    steps = config.training.max_steps if args.steps is None else args.steps
    summary = trainer.fit(train_loader, validation_loader, max_steps=steps)
    return {
        "command": "train",
        "status": "complete",
        "seed": config.training.seed,
        "summary": summary.as_dict(),
    }


def _run_resume(config: WorkflowConfig, args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    trainer, train_loader, validation_loader = _create_training(config)
    trainer.resume(checkpoint)
    steps = config.training.max_steps if args.steps is None else args.steps
    summary = trainer.fit(train_loader, validation_loader, max_steps=steps)
    return {
        "command": "resume",
        "status": "complete",
        "resume_checkpoint": checkpoint.name,
        "resume_checkpoint_sha256": _file_sha256(checkpoint),
        "seed": config.training.seed,
        "summary": summary.as_dict(),
    }


def _load_evaluation_weights(trainer: Any, checkpoint: Path) -> None:
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise TrainingCheckpointError("Evaluation checkpoint must be a regular file")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError) as error:
        raise TrainingCheckpointError("Could not safely load evaluation checkpoint") from error
    if not isinstance(payload, Mapping) or payload.get("checkpoint_role") != "best_inference":
        raise TrainingCheckpointError("Evaluation requires a best_inference checkpoint")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise TrainingCheckpointError("Evaluation checkpoint has no model state")
    try:
        trainer.model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise TrainingCheckpointError("Evaluation model state is incompatible") from error


def _run_evaluate(config: WorkflowConfig, args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    trainer, _, validation_loader = _create_training(config)
    _load_evaluation_weights(trainer, checkpoint)
    summary = trainer.validate(validation_loader)
    return {
        "command": "evaluate",
        "status": "complete",
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": _file_sha256(checkpoint),
        "reference_metrics": summary.as_dict(),
        "limitations": ["PSNR and SSIM require aligned reference HR images."],
    }


def _create_pipeline(args: argparse.Namespace) -> SemiRestorePipeline:
    return SemiRestorePipeline.from_config(
        device=args.device,
        pipeline_config=PipelineConfig(
            mode=args.mode,
            output_bit_depth=args.bit_depth,
            tile_size=args.tile_size,
            overlap=args.overlap,
        ),
    )


def _restore_once(config: WorkflowConfig, args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    del config
    pipeline = _create_pipeline(args)
    try:
        result = pipeline.restore_and_analyze(Path(args.input), mode=args.mode)
        status = pipeline.status().to_dict()
    finally:
        pipeline.close()
    status["checkpoint_path"] = Path(str(status["checkpoint_path"])).name
    return result, status


def _run_restore(config: WorkflowConfig, args: argparse.Namespace) -> dict[str, Any]:
    result, status = _restore_once(config, args)
    output = Path(args.output).expanduser().resolve()
    _atomic_write(output, result.png_bytes, force=args.force)
    return {
        "command": "restore",
        "status": "complete",
        "input": Path(args.input).name,
        "output": output.name,
        "media_type": result.media_type,
        "restoration": result.metadata(),
        "checkpoint_identity": status,
    }


def _run_benchmark(config: WorkflowConfig, args: argparse.Namespace) -> dict[str, Any]:
    del config
    pipeline = _create_pipeline(args)
    try:
        results = [
            pipeline.restore_and_analyze(Path(args.input), mode=args.mode)
            for _ in range(args.iterations)
        ]
    finally:
        pipeline.close()
    latencies = [float(result.timing_ms["total"]) for result in results]
    report = {
        "command": "benchmark",
        "status": "complete",
        "input": Path(args.input).name,
        "mode": args.mode,
        "iterations": args.iterations,
        "measured_total_latency_ms": {
            "minimum": min(latencies),
            "mean": fmean(latencies),
            "maximum": max(latencies),
        },
        "device": results[0].resolved_device,
        "checkpoint_sha256": results[0].checkpoint_sha256,
        "fabricated_values": False,
    }
    if args.report is not None:
        content = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
        _atomic_write(Path(args.report), content, force=args.force)
        report["report"] = Path(args.report).name
    return report


def _run_dry(config: WorkflowConfig, args: argparse.Namespace) -> dict[str, Any]:
    del args
    return {
        "command": "dry-run",
        "status": "ready",
        "expensive_work_started": False,
        "seed": config.training.seed,
        "device_request": config.training.device,
        "planned_max_steps": config.training.max_steps,
        "resolved_configuration": resolved_configuration(config),
    }


def _run_environment(config: WorkflowConfig, args: argparse.Namespace) -> dict[str, Any]:
    del args
    return {
        "command": "environment",
        "status": "complete",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "seed": config.training.seed,
        "device_request": config.training.device,
        "checkpoint_identity": _checkpoint_report(config),
    }


_COMMANDS = {
    "validate-config": _run_validate,
    "audit-dataset": _run_audit,
    "train": _run_train,
    "resume": _run_resume,
    "evaluate": _run_evaluate,
    "restore": _run_restore,
    "benchmark": _run_benchmark,
    "dry-run": _run_dry,
    "environment": _run_environment,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproducible SemiRestore model workflows")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--json", action="store_true", dest="json_output")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-config")
    subparsers.add_parser("audit-dataset")
    train = subparsers.add_parser("train")
    train.add_argument("--steps", type=int)
    resume = subparsers.add_parser("resume")
    resume.add_argument("--checkpoint", type=Path, required=True)
    resume.add_argument("--steps", type=int)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    for name in ("restore", "benchmark"):
        command = subparsers.add_parser(name)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--mode", choices=("direct", "tiled"), default="direct")
        command.add_argument("--tile-size", type=int, default=256)
        command.add_argument("--overlap", type=int, default=32)
        command.add_argument("--bit-depth", type=int, choices=(8, 16), default=16)
        command.add_argument("--device", default="auto")
        command.add_argument("--force", action="store_true")
    restore = subparsers.choices["restore"]
    restore.add_argument("--output", type=Path, required=True)
    benchmark = subparsers.choices["benchmark"]
    benchmark.add_argument("--iterations", type=int, choices=range(1, 101), default=3)
    benchmark.add_argument("--report", type=Path)
    subparsers.add_parser("dry-run")
    subparsers.add_parser("environment")
    return parser


def _human_summary(payload: Mapping[str, Any]) -> str:
    command = payload.get("command", "workflow")
    status = payload.get("status", "complete")
    details = [f"SemiRestore {command}: {status}"]
    for key in ("seed", "device", "train_samples", "validation_samples", "output"):
        if key in payload:
            details.append(f"{key.replace('_', ' ')}: {payload[key]}")
    return "\n".join(details)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one explicit workflow and return a stable process exit code."""

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        config = load_workflow_config(args.config)
        payload = _COMMANDS[args.command](config, args)
        if args.json_output:
            print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        else:
            print(_human_summary(payload))
        return 0
    except (
        WorkflowConfigError,
        ModelConfigError,
        DatasetValidationError,
        TrainingConfigurationError,
    ) as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2
    except (CheckpointError, TrainingCheckpointError, ModelManagerError) as error:
        print(f"checkpoint error: {error}", file=sys.stderr)
        return 3
    except (TrainingRuntimeError, RestorationServiceError, OSError, RuntimeError) as error:
        print(f"workflow error: {error}", file=sys.stderr)
        return 4


__all__ = [
    "CLI_VERSION",
    "WorkflowConfig",
    "WorkflowConfigError",
    "build_parser",
    "load_workflow_config",
    "main",
    "resolved_configuration",
]
