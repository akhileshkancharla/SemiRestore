from __future__ import annotations

import pytest
import torch

from semirestore.models import LayerNorm2d, NAFBlock, SimpleGate


def test_layer_norm_2d_preserves_shape_and_normalizes_channels() -> None:
    inputs = torch.tensor(
        [[[[1.0, 4.0]], [[2.0, 5.0]], [[3.0, 6.0]]]],
        dtype=torch.float32,
    )

    output = LayerNorm2d(3)(inputs)

    assert output.shape == inputs.shape
    torch.testing.assert_close(output.mean(dim=1), torch.zeros((1, 1, 2)), atol=1e-6, rtol=0)


def test_layer_norm_2d_rejects_wrong_channel_count() -> None:
    with pytest.raises(ValueError, match="expected N,4,H,W"):
        LayerNorm2d(4)(torch.zeros((1, 3, 2, 2)))


def test_simple_gate_multiplies_channel_halves() -> None:
    inputs = torch.tensor([[[[1.0]], [[2.0]], [[3.0]], [[4.0]]]])

    output = SimpleGate()(inputs)

    torch.testing.assert_close(output, torch.tensor([[[[3.0]], [[8.0]]]]))


def test_simple_gate_rejects_odd_channel_count() -> None:
    with pytest.raises(ValueError, match="even channel count"):
        SimpleGate()(torch.zeros((1, 3, 2, 2)))


def test_naf_block_is_identity_at_initialization() -> None:
    torch.manual_seed(2026)
    inputs = torch.randn((2, 8, 5, 7))

    output = NAFBlock(8)(inputs)

    torch.testing.assert_close(output, inputs, atol=0, rtol=0)


@pytest.mark.parametrize("dropout", [-0.1, 1.0])
def test_naf_block_rejects_invalid_dropout(dropout: float) -> None:
    with pytest.raises(ValueError, match="dropout must be"):
        NAFBlock(8, dropout=dropout)
