from __future__ import annotations

from pathlib import Path

import pytest

from semirestore.config import ModelConfigError, build_model, load_model_config

CONFIG_PATH = Path("configs/model/resolved_conditioned.yaml")


def test_authoritative_model_configuration_loads() -> None:
    config = load_model_config(CONFIG_PATH)

    assert config.name == "naf_sr"
    assert config.width == 48
    assert config.encoder_blocks == (2, 2, 4)
    assert config.middle_blocks == 6
    assert config.decoder_blocks == (2, 2, 2)
    assert config.statistics_conditioning is True
    assert config.conditioning_hidden == 64
    assert config.scale == 2


def test_validated_configuration_constructs_the_conditioned_model() -> None:
    config = load_model_config(CONFIG_PATH)

    model = build_model(config)

    assert model.model_config() == {
        "width": 48,
        "encoder_blocks": [2, 2, 4],
        "middle_blocks": 6,
        "decoder_blocks": [2, 2, 2],
        "dropout": 0.0,
        "statistics_conditioning": True,
        "conditioning_hidden": 64,
    }


def test_loader_rejects_checkpoint_incompatible_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "wrong-width.yaml"
    path.write_text(CONFIG_PATH.read_text(encoding="utf-8").replace("width: 48", "width: 32"))

    with pytest.raises(ModelConfigError, match=r"width: expected 48, got 32"):
        load_model_config(path)


def test_loader_rejects_experiment_sections(tmp_path: Path) -> None:
    path = tmp_path / "training-config.yaml"
    path.write_text(
        CONFIG_PATH.read_text(encoding="utf-8") + "\ntraining:\n  max_steps: 5000\n",
        encoding="utf-8",
    )

    with pytest.raises(ModelConfigError, match="unsupported section"):
        load_model_config(path)


def test_loader_rejects_unknown_model_fields(tmp_path: Path) -> None:
    path = tmp_path / "unknown-field.yaml"
    path.write_text(
        CONFIG_PATH.read_text(encoding="utf-8") + "  output_channels: 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ModelConfigError, match="Unknown model configuration"):
        load_model_config(path)


def test_loader_rejects_unsafe_yaml_tags(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.yaml"
    path.write_text("!!python/object:builtins.object {}", encoding="utf-8")

    with pytest.raises(ModelConfigError, match="safely read"):
        load_model_config(path)


def test_loader_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ModelConfigError, match="does not exist"):
        load_model_config(tmp_path / "missing.yaml")
