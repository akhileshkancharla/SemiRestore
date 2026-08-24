"""Validated configuration for the frozen conditioned restoration model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from .models import NAFSR


class ModelConfigError(ValueError):
    """Raised when model configuration is malformed or checkpoint-incompatible."""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Architecture values required to construct the conditioned NAF-SR model."""

    name: str
    width: int
    encoder_blocks: tuple[int, ...]
    middle_blocks: int
    decoder_blocks: tuple[int, ...]
    dropout: float
    statistics_conditioning: bool
    conditioning_hidden: int

    @property
    def scale(self) -> int:
        """Return the architecture's fixed spatial output scale."""

        return NAFSR.scale

    def model_kwargs(self) -> dict[str, object]:
        """Return only keyword arguments accepted by :class:`NAFSR`."""

        return {
            "width": self.width,
            "encoder_blocks": self.encoder_blocks,
            "middle_blocks": self.middle_blocks,
            "decoder_blocks": self.decoder_blocks,
            "dropout": self.dropout,
            "statistics_conditioning": self.statistics_conditioning,
            "conditioning_hidden": self.conditioning_hidden,
        }

    def require_checkpoint_compatible(self) -> None:
        """Reject architecture values incompatible with the trusted checkpoint."""

        mismatches = []
        for field in fields(self):
            expected = getattr(CONDITIONED_CHECKPOINT_CONFIG, field.name)
            actual = getattr(self, field.name)
            if actual != expected:
                mismatches.append(f"{field.name}: expected {expected!r}, got {actual!r}")
        if mismatches:
            details = "; ".join(mismatches)
            raise ModelConfigError(
                f"Configuration is incompatible with the conditioned checkpoint ({details})"
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> ModelConfig:
        """Validate and normalize an untrusted model configuration mapping."""

        required = {field.name for field in fields(cls)}
        supplied = set(values)
        missing = sorted(required - supplied)
        unknown = sorted(supplied - required)
        if missing:
            raise ModelConfigError(f"Missing model configuration field(s): {missing}")
        if unknown:
            raise ModelConfigError(f"Unknown model configuration field(s): {unknown}")

        name = _require_string(values, "name")
        width = _require_integer(values, "width", minimum=4)
        encoder_blocks = _require_blocks(values, "encoder_blocks")
        middle_blocks = _require_integer(values, "middle_blocks", minimum=1)
        decoder_blocks = _require_blocks(values, "decoder_blocks")
        if len(encoder_blocks) != len(decoder_blocks):
            raise ModelConfigError("encoder_blocks and decoder_blocks must have equal lengths")
        dropout = _require_float(values, "dropout", minimum=0.0, maximum_exclusive=1.0)
        statistics_conditioning = _require_boolean(values, "statistics_conditioning")
        conditioning_hidden = _require_integer(values, "conditioning_hidden", minimum=4)

        return cls(
            name=name,
            width=width,
            encoder_blocks=encoder_blocks,
            middle_blocks=middle_blocks,
            decoder_blocks=decoder_blocks,
            dropout=dropout,
            statistics_conditioning=statistics_conditioning,
            conditioning_hidden=conditioning_hidden,
        )


CONDITIONED_CHECKPOINT_CONFIG = ModelConfig(
    name="naf_sr",
    width=48,
    encoder_blocks=(2, 2, 4),
    middle_blocks=6,
    decoder_blocks=(2, 2, 2),
    dropout=0.0,
    statistics_conditioning=True,
    conditioning_hidden=64,
)


def _require_string(values: Mapping[str, Any], key: str) -> str:
    value = values[key]
    if not isinstance(value, str) or not value.strip():
        raise ModelConfigError(f"{key} must be a non-empty string")
    return value


def _require_integer(values: Mapping[str, Any], key: str, *, minimum: int) -> int:
    value = values[key]
    if type(value) is not int or value < minimum:
        raise ModelConfigError(f"{key} must be an integer greater than or equal to {minimum}")
    return value


def _require_blocks(values: Mapping[str, Any], key: str) -> tuple[int, ...]:
    value = values[key]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ModelConfigError(f"{key} must be a non-empty sequence of positive integers")
    if any(type(count) is not int or count < 1 for count in value):
        raise ModelConfigError(f"{key} must contain only positive integers")
    return tuple(value)


def _require_float(
    values: Mapping[str, Any],
    key: str,
    *,
    minimum: float,
    maximum_exclusive: float,
) -> float:
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelConfigError(f"{key} must be numeric")
    converted = float(value)
    if not minimum <= converted < maximum_exclusive:
        raise ModelConfigError(f"{key} must be in [{minimum}, {maximum_exclusive})")
    return converted


def _require_boolean(values: Mapping[str, Any], key: str) -> bool:
    value = values[key]
    if type(value) is not bool:
        raise ModelConfigError(f"{key} must be a boolean")
    return value


def load_model_config(path: str | Path) -> ModelConfig:
    """Safely load the checkpoint-compatible model section from a YAML file."""

    resolved_path = Path(path)
    if not resolved_path.is_file():
        raise ModelConfigError(f"Model configuration file does not exist: {resolved_path}")
    try:
        with resolved_path.open("r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as error:
        raise ModelConfigError(
            f"Could not safely read model configuration: {resolved_path}"
        ) from error

    if not isinstance(document, Mapping):
        raise ModelConfigError("Model configuration root must be a mapping")
    unknown_sections = sorted(set(document) - {"model"})
    if unknown_sections:
        raise ModelConfigError(
            f"Deployment configuration contains unsupported section(s): {unknown_sections}"
        )
    model_values = document.get("model")
    if not isinstance(model_values, Mapping):
        raise ModelConfigError("Model configuration must contain a 'model' mapping")

    config = ModelConfig.from_mapping(model_values)
    config.require_checkpoint_compatible()
    return config


def build_model(config: ModelConfig) -> NAFSR:
    """Construct the frozen NAF-SR architecture from validated configuration."""

    config.require_checkpoint_compatible()
    return NAFSR(**config.model_kwargs())


__all__ = [
    "CONDITIONED_CHECKPOINT_CONFIG",
    "ModelConfig",
    "ModelConfigError",
    "build_model",
    "load_model_config",
]
