from __future__ import annotations

import json
import math

import pytest
import torch

from semirestore.losses import CharbonnierLoss
from semirestore.metrics import (
    compute_reference_metrics,
    peak_signal_to_noise_ratio,
    structural_similarity_index,
)


def test_charbonnier_identical_inputs_equal_epsilon() -> None:
    loss = CharbonnierLoss(1e-3)
    image = torch.zeros((2, 1, 4, 5))

    result = loss(image, image)

    assert result.item() == pytest.approx(1e-3)


def test_charbonnier_known_numerical_example() -> None:
    prediction = torch.tensor([0.0, 3.0])
    target = torch.tensor([0.0, 0.0])

    result = CharbonnierLoss(1.0)(prediction, target)

    assert result.item() == pytest.approx((1.0 + math.sqrt(10.0)) / 2.0)


def test_charbonnier_preserves_gradient_flow() -> None:
    prediction = torch.tensor([1.0, -2.0], requires_grad=True)
    target = torch.zeros(2)

    CharbonnierLoss()(prediction, target).backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad) == 2


@pytest.mark.parametrize("epsilon", (0.0, -1.0, float("nan"), float("inf"), True))
def test_charbonnier_rejects_invalid_epsilon(epsilon: float) -> None:
    with pytest.raises(ValueError):
        CharbonnierLoss(epsilon)


def test_charbonnier_validates_shape_dtype_and_finiteness() -> None:
    loss = CharbonnierLoss()
    with pytest.raises(ValueError, match="shape mismatch"):
        loss(torch.zeros(2), torch.zeros(3))
    with pytest.raises(ValueError, match="floating-point"):
        loss(torch.zeros(2, dtype=torch.int64), torch.zeros(2, dtype=torch.int64))
    with pytest.raises(ValueError, match="finite"):
        loss(torch.tensor([float("nan")]), torch.zeros(1))


def test_psnr_identical_image_is_positive_infinity() -> None:
    image = torch.full((11, 12), 0.25)

    score = peak_signal_to_noise_ratio(image, image, data_range=1.0)

    assert score.shape == (1,)
    assert torch.isposinf(score).all()


def test_psnr_known_unit_error_is_zero_db() -> None:
    prediction = torch.zeros((11, 11))
    target = torch.ones((11, 11))

    score = peak_signal_to_noise_ratio(prediction, target, data_range=1.0)

    assert score.item() == pytest.approx(0.0, abs=1e-12)


def test_psnr_known_half_range_error() -> None:
    prediction = torch.zeros((1, 11, 11))
    target = torch.full((1, 11, 11), 0.5)

    score = peak_signal_to_noise_ratio(prediction, target, data_range=1.0)

    assert score.item() == pytest.approx(6.020599913279624)


def test_ssim_identical_image_is_one() -> None:
    image = torch.linspace(0.0, 1.0, 13 * 15).reshape(13, 15)

    score = structural_similarity_index(image, image, data_range=1.0)

    assert score.item() == pytest.approx(1.0, abs=1e-12)


def test_ssim_known_constant_images_matches_luminance_term() -> None:
    prediction = torch.zeros((11, 11))
    target = torch.ones((11, 11))

    score = structural_similarity_index(prediction, target, data_range=1.0)

    expected = 0.01**2 / (1.0 + 0.01**2)
    assert score.item() == pytest.approx(expected, rel=1e-10)


def test_single_and_batched_inputs_have_consistent_results() -> None:
    prediction = torch.stack((torch.zeros((11, 11)), torch.full((11, 11), 0.5)))
    target = torch.ones((2, 1, 11, 11))

    batch_psnr = peak_signal_to_noise_ratio(prediction, target, data_range=1.0)
    batch_ssim = structural_similarity_index(prediction, target, data_range=1.0)

    assert batch_psnr.shape == batch_ssim.shape == (2,)
    for index in range(2):
        single_psnr = peak_signal_to_noise_ratio(
            prediction[index], target[index], data_range=1.0
        )
        single_ssim = structural_similarity_index(
            prediction[index], target[index], data_range=1.0
        )
        torch.testing.assert_close(batch_psnr[index], single_psnr[0])
        torch.testing.assert_close(batch_ssim[index], single_ssim[0])


def test_explicit_non_unit_data_range() -> None:
    prediction = torch.zeros((11, 11))
    target = torch.full((11, 11), 127.5)

    score = peak_signal_to_noise_ratio(prediction, target, data_range=255.0)

    assert score.item() == pytest.approx(6.020599913279624)


def test_reject_policy_does_not_silently_clip() -> None:
    prediction = torch.full((11, 11), 1.1)
    target = torch.ones((11, 11))

    with pytest.raises(ValueError, match="exceed explicit range"):
        peak_signal_to_noise_ratio(prediction, target, data_range=1.0)


def test_explicit_clip_policy_matches_historical_scoring_boundary() -> None:
    prediction = torch.full((11, 11), 1.1)
    target = torch.ones((11, 11))

    score = peak_signal_to_noise_ratio(
        prediction, target, data_range=1.0, range_policy="clip"
    )

    assert torch.isposinf(score).all()


@pytest.mark.parametrize(
    ("prediction", "target", "message"),
    [
        (torch.zeros((2, 11)), torch.zeros((3, 11)), "shape mismatch"),
        (torch.zeros((1, 3, 11, 11)), torch.zeros((1, 3, 11, 11)), "grayscale"),
        (torch.zeros((11, 11), dtype=torch.int32), torch.zeros((11, 11)), "floating-point"),
        (torch.full((11, 11), float("nan")), torch.zeros((11, 11)), "NaN or infinity"),
    ],
)
def test_metrics_validate_shape_channel_dtype_and_finiteness(
    prediction: torch.Tensor, target: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        peak_signal_to_noise_ratio(prediction, target, data_range=1.0)


@pytest.mark.parametrize("data_range", (0.0, -1.0, float("nan"), float("inf")))
def test_metrics_require_valid_explicit_data_range(data_range: float) -> None:
    with pytest.raises(ValueError, match="data_range"):
        peak_signal_to_noise_ratio(
            torch.zeros((11, 11)), torch.zeros((11, 11)), data_range=data_range
        )


def test_ssim_rejects_images_smaller_than_historical_window() -> None:
    with pytest.raises(ValueError, match="at least 11x11"):
        structural_similarity_index(torch.zeros((10, 11)), torch.zeros((10, 11)), data_range=1.0)


def test_metric_summary_has_per_image_aggregate_and_strict_json_form() -> None:
    prediction = torch.stack((torch.zeros((11, 11)), torch.ones((11, 11))))
    target = torch.stack((torch.ones((11, 11)), torch.ones((11, 11))))

    summary = compute_reference_metrics(
        prediction,
        target,
        data_range=1.0,
        sample_ids=("different", "perfect"),
    )
    payload = summary.as_dict()

    assert [item.sample_id for item in summary.per_image] == ["different", "perfect"]
    assert math.isinf(summary.mean_psnr_db)
    assert payload["per_image"][1]["psnr_db"] == "Infinity"
    json.dumps(payload, allow_nan=False)


def test_metric_summary_validates_sample_identifiers() -> None:
    images = torch.zeros((2, 11, 11))

    with pytest.raises(ValueError, match="sample_ids"):
        compute_reference_metrics(images, images, data_range=1.0, sample_ids=("only-one",))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cpu_cuda_metric_agreement() -> None:
    generator = torch.Generator().manual_seed(4)
    prediction = torch.rand((2, 1, 15, 17), generator=generator)
    target = torch.rand((2, 1, 15, 17), generator=generator)

    cpu_psnr = peak_signal_to_noise_ratio(prediction, target, data_range=1.0)
    cpu_ssim = structural_similarity_index(prediction, target, data_range=1.0)
    cuda_psnr = peak_signal_to_noise_ratio(
        prediction.cuda(), target.cuda(), data_range=1.0
    ).cpu()
    cuda_ssim = structural_similarity_index(
        prediction.cuda(), target.cuda(), data_range=1.0
    ).cpu()

    torch.testing.assert_close(cuda_psnr, cpu_psnr, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(cuda_ssim, cpu_ssim, rtol=1e-10, atol=1e-10)
