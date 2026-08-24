from __future__ import annotations

import math
from collections.abc import Iterator

import pytest
import torch
from torch.nn import functional as F

from semirestore.config import build_model, load_model_config
from semirestore.models import NAFSR

EXPECTED_PARAMETER_COUNT = 9_111_684


@pytest.fixture(scope="module")
def limited_torch_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 2))
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def conditioned_model(limited_torch_threads: None) -> NAFSR:
    del limited_torch_threads
    torch.manual_seed(2026)
    model = build_model(load_model_config("configs/model/resolved_conditioned.yaml"))
    return model.eval()


def test_frozen_architecture_has_expected_parameter_count(conditioned_model: NAFSR) -> None:
    parameter_count = sum(parameter.numel() for parameter in conditioned_model.parameters())

    assert parameter_count == EXPECTED_PARAMETER_COUNT


def test_frozen_architecture_has_expected_stages_and_pixel_shuffle(
    conditioned_model: NAFSR,
) -> None:
    assert [len(stage) for stage in conditioned_model.encoders] == [2, 2, 4]
    assert len(conditioned_model.middle) == 6
    assert [len(stage) for stage in conditioned_model.decoders] == [2, 2, 2]
    assert conditioned_model.padder_size == 8
    assert conditioned_model.conditioning_channels == (48, 96, 192, 384, 192, 96, 48)
    assert conditioned_model.sr_head[0].in_channels == 48
    assert conditioned_model.sr_head[0].out_channels == 4
    assert isinstance(conditioned_model.sr_head[1], torch.nn.PixelShuffle)
    assert conditioned_model.sr_head[1].upscale_factor == 2


def test_conditioner_receives_mean_std_min_max_in_that_order(
    conditioned_model: NAFSR,
) -> None:
    assert conditioned_model.conditioner is not None
    observed: list[torch.Tensor] = []

    def capture_statistics(_module: torch.nn.Module, arguments: tuple[torch.Tensor, ...]) -> None:
        observed.append(arguments[0].detach().clone())

    hook = conditioned_model.conditioner[0].register_forward_pre_hook(capture_statistics)
    inputs = torch.tensor([[[[0.0, 0.25], [0.5, 1.0]]]], dtype=torch.float32)
    try:
        parameters = conditioned_model._conditioning(inputs)
    finally:
        hook.remove()

    expected = torch.tensor(
        [[0.4375, math.sqrt(0.13671875), 0.0, 1.0]],
        dtype=torch.float32,
    )
    assert len(observed) == 1
    torch.testing.assert_close(observed[0], expected)
    assert parameters is not None
    assert [tuple(item.shape) for item in parameters] == [
        (1, 2 * channels) for channels in conditioned_model.conditioning_channels
    ]
    assert all(torch.count_nonzero(item) == 0 for item in parameters)


def test_scale_and_shift_conditioning_modulates_features() -> None:
    features = torch.ones((1, 2, 1, 1))
    parameters = torch.tensor([[0.0, 0.0, 1.0, -1.0]])

    conditioned = NAFSR._apply_condition(features, parameters)

    expected = torch.tensor([[[[1.1]], [[0.9]]]])
    torch.testing.assert_close(conditioned, expected)


def test_frozen_model_returns_finite_two_x_output_for_odd_dimensions(
    conditioned_model: NAFSR,
) -> None:
    inputs = torch.linspace(0.0, 1.0, 9 * 11).reshape(1, 1, 9, 11)

    with torch.inference_mode():
        output = conditioned_model(inputs)

    assert output.shape == (1, 1, 18, 22)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_zero_learned_head_reduces_output_to_bicubic_interpolation() -> None:
    model = NAFSR(
        width=4,
        encoder_blocks=(1,),
        middle_blocks=1,
        decoder_blocks=(1,),
    ).eval()
    torch.nn.init.zeros_(model.sr_head[0].weight)
    torch.nn.init.zeros_(model.sr_head[0].bias)
    inputs = torch.tensor([[[[0.0, 0.25], [0.5, 1.0]]]])

    with torch.inference_mode():
        output = model(inputs)
        expected = F.interpolate(
            inputs,
            scale_factor=2,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

    torch.testing.assert_close(output, expected, atol=0, rtol=0)
