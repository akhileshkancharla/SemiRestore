from __future__ import annotations

import pytest
import torch

from semirestore.models import (
    NAFSR,
    ConditioningStatisticsError,
    compute_conditioning_statistics,
    validate_conditioning_statistics,
)


def _conditioned_model() -> NAFSR:
    model = NAFSR(
        width=4,
        encoder_blocks=(1,),
        middle_blocks=1,
        decoder_blocks=(1,),
        statistics_conditioning=True,
        conditioning_hidden=4,
    ).eval()
    assert model.conditioner is not None
    torch.nn.init.normal_(model.conditioner[-1].weight, std=0.01)
    torch.nn.init.normal_(model.conditioner[-1].bias, std=0.01)
    return model


def test_statistics_use_mean_std_min_max_order() -> None:
    inputs = torch.tensor([[[[0.0, 0.25], [0.5, 1.0]]]])

    statistics = compute_conditioning_statistics(inputs)

    expected = torch.tensor([[0.4375, 0.36975497, 0.0, 1.0]])
    torch.testing.assert_close(statistics, expected)


def test_direct_forward_is_unchanged_with_equivalent_override() -> None:
    torch.manual_seed(2026)
    model = _conditioned_model()
    inputs = torch.rand((1, 1, 5, 7))
    statistics = compute_conditioning_statistics(inputs)

    with torch.inference_mode():
        direct = model(inputs)
        overridden = model(inputs, conditioning_statistics=statistics)

    torch.testing.assert_close(overridden, direct, atol=0, rtol=0)


def test_override_reaches_conditioner_in_exact_supplied_order() -> None:
    model = _conditioned_model()
    assert model.conditioner is not None
    observed: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, arguments: tuple[torch.Tensor, ...]) -> None:
        observed.append(arguments[0].detach().clone())

    hook = model.conditioner[0].register_forward_pre_hook(capture)
    inputs = torch.zeros((1, 1, 2, 2))
    override = torch.tensor([[0.4, 0.2, 0.1, 0.9]])
    try:
        with torch.inference_mode():
            model(inputs, conditioning_statistics=override)
    finally:
        hook.remove()

    assert len(observed) == 1
    torch.testing.assert_close(observed[0], override, atol=0, rtol=0)


@pytest.mark.parametrize(
    "statistics",
    [
        None,
        [0.5, 0.1, 0.0, 1.0],
        torch.zeros((4,)),
        torch.zeros((1, 5)),
        torch.zeros((2, 4)),
        torch.zeros((1, 4), dtype=torch.float64),
        torch.tensor([[0.5, float("nan"), 0.0, 1.0]]),
        torch.tensor([[0.5, -0.1, 0.0, 1.0]]),
    ],
)
def test_invalid_override_is_rejected(statistics: object) -> None:
    inputs = torch.zeros((1, 1, 2, 2))
    if statistics is None:
        statistics = torch.sparse_coo_tensor(
            torch.tensor([[0], [0]]),
            torch.tensor([0.5]),
            size=(1, 4),
            check_invariants=False,
        )

    with pytest.raises(ConditioningStatisticsError):
        validate_conditioning_statistics(statistics, inputs)


def test_override_device_must_match_input() -> None:
    inputs = torch.zeros((1, 1, 2, 2))
    statistics = torch.empty((1, 4), device="meta")

    with pytest.raises(ConditioningStatisticsError, match="device"):
        validate_conditioning_statistics(statistics, inputs)


def test_unconditioned_model_rejects_unused_override() -> None:
    model = NAFSR(
        width=4,
        encoder_blocks=(1,),
        middle_blocks=1,
        decoder_blocks=(1,),
    ).eval()

    with pytest.raises(ConditioningStatisticsError, match="unconditioned"):
        model(torch.zeros((1, 1, 2, 2)), conditioning_statistics=torch.zeros((1, 4)))


def test_forward_extension_adds_no_parameters_or_state_keys() -> None:
    model = _conditioned_model()
    keys_before = tuple(model.state_dict())
    count_before = sum(parameter.numel() for parameter in model.parameters())
    inputs = torch.rand((1, 1, 4, 4))

    with torch.inference_mode():
        model(inputs, conditioning_statistics=compute_conditioning_statistics(inputs))

    assert tuple(model.state_dict()) == keys_before
    assert sum(parameter.numel() for parameter in model.parameters()) == count_before
