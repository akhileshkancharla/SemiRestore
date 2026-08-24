from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from semirestore.platform import RuntimeSettings
from semirestore.platform.settings import DEFAULT_MEDIA_TYPES


def test_settings_have_safe_development_defaults() -> None:
    settings = RuntimeSettings()

    assert settings.environment == "development"
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.allowed_media_types == DEFAULT_MEDIA_TYPES
    assert settings.inference_concurrency_limit == 1
    assert settings.enable_fake_model_service is False
    assert settings.model_config_path is None
    assert settings.model_metadata_path is None
    assert settings.checkpoint_path is None


def test_settings_load_typed_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEMIRESTORE_ENVIRONMENT", "staging")
    monkeypatch.setenv("SEMIRESTORE_PORT", "9000")
    monkeypatch.setenv("SEMIRESTORE_LOG_LEVEL", "warning")
    monkeypatch.setenv("SEMIRESTORE_JSON_LOGGING", "false")
    monkeypatch.setenv("SEMIRESTORE_ALLOWED_MEDIA_TYPES", '["IMAGE/PNG", "image/tiff"]')
    monkeypatch.setenv("SEMIRESTORE_INFERENCE_CONCURRENCY_LIMIT", "2")
    monkeypatch.setenv("SEMIRESTORE_MODEL_CONFIG_PATH", "configs/runtime.yaml")
    monkeypatch.setenv("SEMIRESTORE_MODEL_METADATA_PATH", "artifacts/model/checksums.json")
    monkeypatch.setenv("SEMIRESTORE_CHECKPOINT_PATH", "artifacts/model/model.pt")
    monkeypatch.setenv("SEMIRESTORE_DEVICE_PREFERENCE", "cpu")
    monkeypatch.setenv("SEMIRESTORE_ENABLE_FAKE_MODEL_SERVICE", "true")

    settings = RuntimeSettings()

    assert settings.environment == "staging"
    assert settings.port == 9000
    assert settings.log_level == "WARNING"
    assert settings.json_logging is False
    assert settings.allowed_media_types == ("image/png", "image/tiff")
    assert settings.inference_concurrency_limit == 2
    assert settings.model_config_path == Path("configs/runtime.yaml")
    assert settings.model_metadata_path == Path("artifacts/model/checksums.json")
    assert settings.checkpoint_path == Path("artifacts/model/model.pt")
    assert settings.device_preference == "cpu"
    assert settings.enable_fake_model_service is True


@pytest.mark.parametrize(
    ("environment_name", "value"),
    [
        ("SEMIRESTORE_PORT", "0"),
        ("SEMIRESTORE_MAX_ENCODED_UPLOAD_BYTES", "0"),
        ("SEMIRESTORE_MAX_DECODED_IMAGE_WIDTH", "0"),
        ("SEMIRESTORE_MAX_DECODED_IMAGE_HEIGHT", "0"),
        ("SEMIRESTORE_MAX_DECODED_PIXEL_COUNT", "0"),
        ("SEMIRESTORE_INFERENCE_CONCURRENCY_LIMIT", "0"),
        ("SEMIRESTORE_CONCURRENCY_ACQUISITION_TIMEOUT_SECONDS", "0"),
        ("SEMIRESTORE_INFERENCE_TIMEOUT_SECONDS", "0"),
        ("SEMIRESTORE_CONCURRENCY_ACQUISITION_TIMEOUT_SECONDS", "inf"),
        ("SEMIRESTORE_INFERENCE_TIMEOUT_SECONDS", "nan"),
    ],
)
def test_settings_reject_non_positive_limits(
    monkeypatch: pytest.MonkeyPatch, environment_name: str, value: str
) -> None:
    monkeypatch.setenv(environment_name, value)

    with pytest.raises(ValidationError):
        RuntimeSettings()


def test_settings_reject_duplicate_media_types(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "SEMIRESTORE_ALLOWED_MEDIA_TYPES",
        '["image/png", "IMAGE/PNG"]',
    )

    with pytest.raises(ValidationError):
        RuntimeSettings()
