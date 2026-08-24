"""Restoration training losses preserved from the authoritative run."""

from __future__ import annotations

import math

import torch
from torch import nn


class CharbonnierLoss(nn.Module):
    """Mean ``sqrt((prediction - target)^2 + epsilon^2)`` fidelity loss."""

    def __init__(self, epsilon: float = 1e-3) -> None:
        super().__init__()
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
            raise ValueError("Charbonnier epsilon must be numeric")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("Charbonnier epsilon must be finite and positive")
        self.epsilon = float(epsilon)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise TypeError("Charbonnier inputs must be tensors")
        if prediction.shape != target.shape:
            raise ValueError(
                f"Charbonnier shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}"
            )
        if prediction.numel() == 0:
            raise ValueError("Charbonnier inputs must not be empty")
        if not prediction.dtype.is_floating_point or not target.dtype.is_floating_point:
            raise ValueError("Charbonnier inputs must use floating-point dtypes")
        if prediction.device != target.device:
            raise ValueError("Charbonnier inputs must be on the same device")
        if not bool(torch.isfinite(prediction).all().item()) or not bool(
            torch.isfinite(target).all().item()
        ):
            raise ValueError("Charbonnier inputs must contain only finite values")
        difference = prediction - target
        return torch.sqrt(difference.square() + self.epsilon**2).mean()


__all__ = ["CharbonnierLoss"]
