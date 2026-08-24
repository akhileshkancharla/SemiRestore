from __future__ import annotations

import json

import pytest
import torch
from torch.nn import functional as F

from semirestore import spatial
from semirestore.models import NAFSR


def test_aligned_spatial_plan_has_no_padding() -> None:
    plan = spatial.create_spatial_plan(
        original_width=16,
        original_height=8,
        alignment=8,
        scale_factor=2,
    )

    assert plan.padded_width == 16
    assert plan.padded_height == 8
    assert plan.right_padding == 0
    assert plan.bottom_padding == 0
    assert plan.unpadded_input_pixels == 128
    assert plan.padded_input_pixels == 128
    assert plan.padding_overhead_pixels == 0
    assert plan.padding_overhead_fraction == 0.0
    assert plan.internal_padding_required is False


def test_unaligned_spatial_plan_calculates_right_bottom_and_restored_extents() -> None:
    plan = spatial.create_spatial_plan(
        original_width=11,
        original_height=9,
        alignment=8,
        scale_factor=2,
    )

    assert plan.padded_width == 16
    assert plan.padded_height == 16
    assert plan.right_padding == 5
    assert plan.bottom_padding == 7
    assert plan.unpadded_input_pixels == 99
    assert plan.padded_input_pixels == 256
    assert plan.padding_overhead_pixels == 157
    assert plan.padding_overhead_fraction == pytest.approx(157 / 99)
    assert plan.internal_restored_width == 32
    assert plan.internal_restored_height == 32
    assert plan.final_restored_width == 22
    assert plan.final_restored_height == 18
    assert plan.scale_factor == 2
    assert plan.internal_padding_required is True


def test_spatial_plan_is_serialization_friendly() -> None:
    plan = spatial.create_spatial_plan(
        original_width=7,
        original_height=5,
        alignment=8,
        scale_factor=2,
    )

    serialized = json.dumps(plan.to_dict(), allow_nan=False)

    assert '"alignment": 8' in serialized
    assert plan.to_dict()["final_restored_width"] == 14


def test_planning_does_not_allocate_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_allocation(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("spatial planning must not allocate tensors")

    monkeypatch.setattr(torch, "empty", forbidden_allocation)
    monkeypatch.setattr(torch, "zeros", forbidden_allocation)
    monkeypatch.setattr(torch, "tensor", forbidden_allocation)

    plan = spatial.create_spatial_plan(
        original_width=13,
        original_height=17,
        alignment=8,
        scale_factor=2,
    )

    assert (plan.padded_width, plan.padded_height) == (16, 24)


@pytest.mark.parametrize(
    ("width", "height"),
    [(0, 1), (1, 0), (-1, 1), (1, -1), (True, 1), (1, False), (1.0, 1)],
)
def test_invalid_dimensions_are_rejected(width: object, height: object) -> None:
    with pytest.raises(spatial.SpatialPlanningError, match="positive integer"):
        spatial.create_spatial_plan(
            original_width=width,  # type: ignore[arg-type]
            original_height=height,  # type: ignore[arg-type]
            alignment=8,
            scale_factor=2,
        )


@pytest.mark.parametrize("alignment", [0, -1, True, 1.0, 3, 12])
def test_invalid_alignment_is_rejected(alignment: object) -> None:
    with pytest.raises(spatial.SpatialPlanningError, match="alignment"):
        spatial.create_spatial_plan(
            original_width=8,
            original_height=8,
            alignment=alignment,  # type: ignore[arg-type]
            scale_factor=2,
        )


@pytest.mark.parametrize("scale_factor", [0, -1, True, 1.0, 1, 3, 4])
def test_invalid_or_unsupported_scale_is_rejected(scale_factor: object) -> None:
    with pytest.raises(spatial.SpatialPlanningError, match="scale_factor"):
        spatial.create_spatial_plan(
            original_width=8,
            original_height=8,
            alignment=8,
            scale_factor=scale_factor,  # type: ignore[arg-type]
        )


def test_unreasonable_dimensions_are_rejected_before_computation() -> None:
    with pytest.raises(spatial.SpatialPlanningError, match="cannot exceed"):
        spatial.create_spatial_plan(
            original_width=spatial.MAX_PLANNED_INPUT_DIMENSION + 1,
            original_height=1,
            alignment=8,
            scale_factor=2,
        )


def test_unreasonable_padded_pixel_count_is_rejected() -> None:
    with pytest.raises(spatial.SpatialPlanningError, match="pixel count cannot exceed"):
        spatial.create_spatial_plan(
            original_width=100_000,
            original_height=100_000,
            alignment=8,
            scale_factor=2,
        )


def _audited_model() -> NAFSR:
    return NAFSR(
        width=4,
        encoder_blocks=(1, 1, 1),
        middle_blocks=1,
        decoder_blocks=(1, 1, 1),
        statistics_conditioning=True,
        conditioning_hidden=4,
    ).eval()


def test_actual_model_conditions_before_right_bottom_replicate_padding() -> None:
    model = _audited_model()
    assert model.padder_size == 8
    assert model.conditioner is not None
    observed_statistics: list[torch.Tensor] = []
    observed_padded_inputs: list[torch.Tensor] = []

    def capture_statistics(_module: torch.nn.Module, arguments: tuple[torch.Tensor, ...]) -> None:
        observed_statistics.append(arguments[0].detach().clone())

    def capture_padded_input(_module: torch.nn.Module, arguments: tuple[torch.Tensor, ...]) -> None:
        observed_padded_inputs.append(arguments[0].detach().clone())

    statistics_hook = model.conditioner[0].register_forward_pre_hook(capture_statistics)
    intro_hook = model.intro.register_forward_pre_hook(capture_padded_input)
    inputs = torch.arange(15, dtype=torch.float32).reshape(1, 1, 3, 5) / 14.0
    try:
        with torch.inference_mode():
            output = model(inputs)
    finally:
        statistics_hook.remove()
        intro_hook.remove()

    flattened = inputs.flatten(2)
    expected_statistics = torch.cat(
        (
            flattened.mean(2),
            flattened.std(2, unbiased=False),
            flattened.amin(2),
            flattened.amax(2),
        ),
        dim=1,
    )
    expected_padding = F.pad(inputs, (0, 3, 0, 5), mode="replicate")
    assert len(observed_statistics) == 1
    assert len(observed_padded_inputs) == 1
    torch.testing.assert_close(observed_statistics[0], expected_statistics)
    torch.testing.assert_close(observed_padded_inputs[0], expected_padding)
    assert output.shape == (1, 1, 6, 10)


def test_actual_model_crops_unaligned_learned_extent_and_uses_original_bicubic() -> None:
    model = _audited_model()
    torch.nn.init.zeros_(model.sr_head[0].weight)
    torch.nn.init.zeros_(model.sr_head[0].bias)
    inputs = torch.linspace(0.0, 1.0, 5 * 7).reshape(1, 1, 5, 7)
    internal_shapes: list[tuple[int, ...]] = []

    def capture_internal_extent(
        _module: torch.nn.Module,
        _arguments: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        internal_shapes.append(tuple(output.shape))

    hook = model.sr_head.register_forward_hook(capture_internal_extent)

    try:
        with torch.inference_mode():
            output = model(inputs)
            expected = F.interpolate(
                inputs,
                scale_factor=2,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
    finally:
        hook.remove()

    assert internal_shapes == [(1, 1, 16, 16)]
    assert output.shape == (1, 1, 10, 14)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
