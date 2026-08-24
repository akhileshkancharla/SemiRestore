from __future__ import annotations

import pytest
import torch

from semirestore.models import NAFSR


def _small_model() -> NAFSR:
    return NAFSR(
        width=4,
        encoder_blocks=(1,),
        middle_blocks=1,
        decoder_blocks=(1,),
        statistics_conditioning=True,
        conditioning_hidden=4,
    )


def test_naf_sr_restores_one_channel_at_twice_the_input_size() -> None:
    torch.manual_seed(2026)
    inputs = torch.rand((1, 1, 5, 7), dtype=torch.float32)

    output = _small_model()(inputs)

    assert output.shape == (1, 1, 10, 14)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_naf_sr_rejects_non_grayscale_nchw_input() -> None:
    with pytest.raises(ValueError, match="one channel"):
        _small_model()(torch.zeros((1, 3, 8, 8)))
