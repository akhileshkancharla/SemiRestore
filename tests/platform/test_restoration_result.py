from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pytest

from semirestore.platform import RestorationResult


def make_result(**overrides: Any) -> RestorationResult:
    values: dict[str, Any] = {
        "restored_image_bytes": b"encoded image",
        "restored_media_type": "image/png",
        "restored_width": 4,
        "restored_height": 3,
        "original_width": 4,
        "original_height": 3,
    }
    values.update(overrides)
    return RestorationResult(**values)


def test_result_copies_json_diagnostics_and_freezes_warnings() -> None:
    diagnostics = {"summary": {"synthetic": True}, "values": [1, 2]}
    result = make_result(
        inference_latency_ms=0,
        device="cpu",
        model_version="model-v1",
        checkpoint_checksum=f"sha256:{'a' * 64}",
        diagnostics=diagnostics,
        warnings=("Synthetic warning.",),
    )
    diagnostics["summary"] = {"synthetic": False}

    assert result.inference_latency_ms == 0.0
    assert isinstance(result.diagnostics, Mapping)
    assert result.diagnostics["summary"] == {"synthetic": True}
    assert result.warnings == ("Synthetic warning.",)


@pytest.mark.parametrize(
    "overrides",
    [
        {"restored_image_bytes": b""},
        {"restored_image_bytes": bytearray(b"not immutable")},
        {"restored_media_type": "image/gif"},
        {"restored_width": 0},
        {"restored_height": -1},
        {"original_width": 0},
        {"original_height": False},
        {"inference_latency_ms": -0.1},
        {"inference_latency_ms": math.inf},
        {"inference_latency_ms": math.nan},
        {"device": "C:/private/device"},
        {"model_version": "version\nsecret"},
        {"checkpoint_checksum": "../private/checkpoint"},
        {"diagnostics": {"value": object()}},
        {"diagnostics": {"value": math.nan}},
        {"warnings": ["mutable warning"]},
        {"warnings": ("C:/private/checkpoint",)},
    ],
)
def test_result_rejects_invalid_or_unsafe_values(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        make_result(**overrides)
