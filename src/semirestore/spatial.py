"""Pure integer spatial planning for NAF-SR internal alignment behavior."""

from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_SCALE_FACTOR = 2
MAX_PLANNED_INPUT_DIMENSION = 1_000_000
MAX_PLANNED_INPUT_PIXELS = 1_000_000_000
MAX_SAFE_INTEGER = (1 << 63) - 1


class SpatialPlanningError(ValueError):
    """Raised when dimensions cannot form a supported, bounded spatial plan."""


@dataclass(frozen=True, slots=True)
class SpatialPlan:
    """Allocation-free description of internal alignment and exact output crop."""

    original_width: int
    original_height: int
    alignment: int
    padded_width: int
    padded_height: int
    right_padding: int
    bottom_padding: int
    unpadded_input_pixels: int
    padded_input_pixels: int
    padding_overhead_pixels: int
    padding_overhead_fraction: float
    internal_restored_width: int
    internal_restored_height: int
    final_restored_width: int
    final_restored_height: int
    scale_factor: int
    internal_padding_required: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        """Return a JSON-compatible spatial metadata mapping."""

        return {
            "original_width": self.original_width,
            "original_height": self.original_height,
            "alignment": self.alignment,
            "padded_width": self.padded_width,
            "padded_height": self.padded_height,
            "right_padding": self.right_padding,
            "bottom_padding": self.bottom_padding,
            "unpadded_input_pixels": self.unpadded_input_pixels,
            "padded_input_pixels": self.padded_input_pixels,
            "padding_overhead_pixels": self.padding_overhead_pixels,
            "padding_overhead_fraction": self.padding_overhead_fraction,
            "internal_restored_width": self.internal_restored_width,
            "internal_restored_height": self.internal_restored_height,
            "final_restored_width": self.final_restored_width,
            "final_restored_height": self.final_restored_height,
            "scale_factor": self.scale_factor,
            "internal_padding_required": self.internal_padding_required,
        }


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise SpatialPlanningError(f"{name} must be a positive integer")
    return value


def _checked_product(left: int, right: int, name: str) -> int:
    if left > MAX_SAFE_INTEGER // right:
        raise SpatialPlanningError(f"{name} exceeds the supported integer range")
    return left * right


def create_spatial_plan(
    *,
    original_width: int,
    original_height: int,
    alignment: int,
    scale_factor: int,
) -> SpatialPlan:
    """Calculate alignment padding and restored extents without tensor allocation."""

    width = _positive_integer(original_width, "original_width")
    height = _positive_integer(original_height, "original_height")
    aligned_to = _positive_integer(alignment, "alignment")
    scale = _positive_integer(scale_factor, "scale_factor")
    if aligned_to & (aligned_to - 1):
        raise SpatialPlanningError("alignment must be a power of two")
    if scale != SUPPORTED_SCALE_FACTOR:
        raise SpatialPlanningError(
            f"scale_factor must be the supported value {SUPPORTED_SCALE_FACTOR}"
        )
    if width > MAX_PLANNED_INPUT_DIMENSION or height > MAX_PLANNED_INPUT_DIMENSION:
        raise SpatialPlanningError(
            f"input dimensions cannot exceed {MAX_PLANNED_INPUT_DIMENSION} pixels"
        )
    if aligned_to > MAX_PLANNED_INPUT_DIMENSION:
        raise SpatialPlanningError(
            f"alignment cannot exceed {MAX_PLANNED_INPUT_DIMENSION} pixels"
        )

    padded_width = ((width + aligned_to - 1) // aligned_to) * aligned_to
    padded_height = ((height + aligned_to - 1) // aligned_to) * aligned_to
    if (
        padded_width > MAX_PLANNED_INPUT_DIMENSION
        or padded_height > MAX_PLANNED_INPUT_DIMENSION
    ):
        raise SpatialPlanningError("aligned dimensions exceed the supported planning bound")
    unpadded_pixels = _checked_product(width, height, "unpadded input pixel count")
    padded_pixels = _checked_product(
        padded_width,
        padded_height,
        "padded input pixel count",
    )
    if padded_pixels > MAX_PLANNED_INPUT_PIXELS:
        raise SpatialPlanningError(
            f"padded input pixel count cannot exceed {MAX_PLANNED_INPUT_PIXELS}"
        )
    internal_width = _checked_product(padded_width, scale, "internal restored width")
    internal_height = _checked_product(padded_height, scale, "internal restored height")
    final_width = _checked_product(width, scale, "final restored width")
    final_height = _checked_product(height, scale, "final restored height")
    overhead_pixels = padded_pixels - unpadded_pixels

    return SpatialPlan(
        original_width=width,
        original_height=height,
        alignment=aligned_to,
        padded_width=padded_width,
        padded_height=padded_height,
        right_padding=padded_width - width,
        bottom_padding=padded_height - height,
        unpadded_input_pixels=unpadded_pixels,
        padded_input_pixels=padded_pixels,
        padding_overhead_pixels=overhead_pixels,
        padding_overhead_fraction=overhead_pixels / unpadded_pixels,
        internal_restored_width=internal_width,
        internal_restored_height=internal_height,
        final_restored_width=final_width,
        final_restored_height=final_height,
        scale_factor=scale,
        internal_padding_required=overhead_pixels > 0,
    )


__all__ = [
    "MAX_PLANNED_INPUT_DIMENSION",
    "MAX_PLANNED_INPUT_PIXELS",
    "SUPPORTED_SCALE_FACTOR",
    "SpatialPlan",
    "SpatialPlanningError",
    "create_spatial_plan",
]
