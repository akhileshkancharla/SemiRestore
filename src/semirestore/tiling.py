"""Deterministic tile planning and positive overlap blending weights."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spatial import SpatialPlanningError, create_spatial_plan

DEFAULT_MAX_TILE_COUNT = 4096
BLENDING_METHOD = "separable_linear_overlap_ramp"


class TilingError(ValueError):
    """Base class for safe tile planning and assembly failures."""


class InvalidTileConfigurationError(TilingError):
    """Raised when tile size, overlap, dimensions, or model geometry is invalid."""


class ExcessiveTileCountError(TilingError):
    """Raised when a deterministic plan would contain too many tiles."""


class TileResourceLimitError(TilingError):
    """Raised when an individual aligned tile exceeds its compute limit."""


class TileAssemblyError(TilingError):
    """Raised when blending weights or assembled output are invalid."""


@dataclass(frozen=True, slots=True)
class TileCoordinate:
    """One half-open input tile and its exact scaled output coordinate."""

    index: int
    row: int
    column: int
    top: int
    left: int
    bottom: int
    right: int
    output_top: int
    output_left: int
    output_bottom: int
    output_right: int
    padded_input_pixels: int

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def width(self) -> int:
        return self.right - self.left

    def to_dict(self) -> dict[str, int]:
        return {
            "index": self.index,
            "row": self.row,
            "column": self.column,
            "top": self.top,
            "left": self.left,
            "bottom": self.bottom,
            "right": self.right,
            "output_top": self.output_top,
            "output_left": self.output_left,
            "output_bottom": self.output_bottom,
            "output_right": self.output_right,
            "padded_input_pixels": self.padded_input_pixels,
        }


@dataclass(frozen=True, slots=True)
class TilePlan:
    """Compact deterministic row-major plan for a complete image."""

    image_width: int
    image_height: int
    tile_size: int
    overlap: int
    stride: int
    alignment: int
    scale_factor: int
    row_starts: tuple[int, ...]
    column_starts: tuple[int, ...]
    tiles: tuple[TileCoordinate, ...]
    max_padded_pixels_per_tile: int
    per_tile_compute_limit: int
    blending_method: str = BLENDING_METHOD

    @property
    def tile_count(self) -> int:
        return len(self.tiles)

    def to_summary(self) -> dict[str, object]:
        """Return compact JSON-compatible metadata without per-tile telemetry."""

        return {
            "tile_size": self.tile_size,
            "overlap": self.overlap,
            "stride": self.stride,
            "tile_count": self.tile_count,
            "tile_rows": len(self.row_starts),
            "tile_columns": len(self.column_starts),
            "row_starts": list(self.row_starts),
            "column_starts": list(self.column_starts),
            "coordinate_convention": "half_open_input_row_major",
            "alignment": self.alignment,
            "scale_factor": self.scale_factor,
            "max_padded_pixels_per_tile": self.max_padded_pixels_per_tile,
            "per_tile_compute_limit": self.per_tile_compute_limit,
            "blending_method": self.blending_method,
            "first_tile": self.tiles[0].to_dict(),
            "last_tile": self.tiles[-1].to_dict(),
        }


def _integer(value: object, name: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise InvalidTileConfigurationError(f"{name} must be an integer >= {minimum}")
    return value


def _axis_starts(length: int, tile_size: int, stride: int) -> tuple[int, ...]:
    starts = [0]
    while starts[-1] + tile_size < length:
        starts.append(starts[-1] + stride)
    return tuple(starts)


def _axis_tile_count(length: int, tile_size: int, stride: int) -> int:
    if length <= tile_size:
        return 1
    return ((length - tile_size + stride - 1) // stride) + 1


def create_tile_plan(
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
    overlap: int,
    alignment: int,
    scale_factor: int,
    max_padded_pixels_per_tile: int,
    max_tile_count: int = DEFAULT_MAX_TILE_COUNT,
) -> TilePlan:
    """Plan maximum-extent overlapping tiles with deterministic fixed stride."""

    size = _integer(tile_size, "tile_size", minimum=1)
    shared = _integer(overlap, "overlap", minimum=0)
    tile_limit = _integer(
        max_padded_pixels_per_tile,
        "max_padded_pixels_per_tile",
        minimum=1,
    )
    count_limit = _integer(max_tile_count, "max_tile_count", minimum=1)
    if shared >= size:
        raise InvalidTileConfigurationError("overlap must be smaller than tile_size")
    try:
        full_plan = create_spatial_plan(
            original_width=image_width,
            original_height=image_height,
            alignment=alignment,
            scale_factor=scale_factor,
        )
    except SpatialPlanningError as error:
        raise InvalidTileConfigurationError("Image or model spatial values are invalid") from error
    if size % full_plan.alignment != 0:
        raise InvalidTileConfigurationError("tile_size must be divisible by model alignment")

    stride = size - shared
    row_count = _axis_tile_count(full_plan.original_height, size, stride)
    column_count = _axis_tile_count(full_plan.original_width, size, stride)
    if row_count > count_limit // column_count:
        raise ExcessiveTileCountError(
            f"Tile count exceeds configured maximum {count_limit}"
        )
    tile_count = row_count * column_count
    if tile_count > count_limit:
        raise ExcessiveTileCountError(
            f"Tile count {tile_count} exceeds configured maximum {count_limit}"
        )
    row_starts = _axis_starts(full_plan.original_height, size, stride)
    column_starts = _axis_starts(full_plan.original_width, size, stride)

    tiles: list[TileCoordinate] = []
    maximum_padded_pixels = 0
    for row, top in enumerate(row_starts):
        bottom = min(top + size, full_plan.original_height)
        for column, left in enumerate(column_starts):
            right = min(left + size, full_plan.original_width)
            tile_spatial = create_spatial_plan(
                original_width=right - left,
                original_height=bottom - top,
                alignment=full_plan.alignment,
                scale_factor=full_plan.scale_factor,
            )
            maximum_padded_pixels = max(
                maximum_padded_pixels,
                tile_spatial.padded_input_pixels,
            )
            if tile_spatial.padded_input_pixels > tile_limit:
                raise TileResourceLimitError(
                    f"Aligned tile requires {tile_spatial.padded_input_pixels} pixels, "
                    f"exceeding per-tile limit {tile_limit}"
                )
            tiles.append(
                TileCoordinate(
                    index=len(tiles),
                    row=row,
                    column=column,
                    top=top,
                    left=left,
                    bottom=bottom,
                    right=right,
                    output_top=top * full_plan.scale_factor,
                    output_left=left * full_plan.scale_factor,
                    output_bottom=bottom * full_plan.scale_factor,
                    output_right=right * full_plan.scale_factor,
                    padded_input_pixels=tile_spatial.padded_input_pixels,
                )
            )

    return TilePlan(
        image_width=full_plan.original_width,
        image_height=full_plan.original_height,
        tile_size=size,
        overlap=shared,
        stride=stride,
        alignment=full_plan.alignment,
        scale_factor=full_plan.scale_factor,
        row_starts=row_starts,
        column_starts=column_starts,
        tiles=tuple(tiles),
        max_padded_pixels_per_tile=maximum_padded_pixels,
        per_tile_compute_limit=tile_limit,
    )


def _linear_axis_weights(length: int, leading_overlap: int, trailing_overlap: int) -> np.ndarray:
    weights = np.ones(length, dtype=np.float32)
    if leading_overlap:
        weights[:leading_overlap] *= np.arange(1, leading_overlap + 1, dtype=np.float32) / (
            leading_overlap + 1
        )
    if trailing_overlap:
        weights[-trailing_overlap:] *= np.arange(
            trailing_overlap,
            0,
            -1,
            dtype=np.float32,
        ) / (trailing_overlap + 1)
    return weights


def blending_weights(plan: TilePlan, tile: TileCoordinate) -> np.ndarray:
    """Return a positive separable linear ramp for one scaled output tile."""

    if tile.index >= plan.tile_count or plan.tiles[tile.index] != tile:
        raise TileAssemblyError("Tile does not belong to the supplied plan")
    scale = plan.scale_factor
    top_overlap = 0
    bottom_overlap = 0
    left_overlap = 0
    right_overlap = 0
    if tile.row > 0:
        previous_top = plan.row_starts[tile.row - 1]
        previous_bottom = min(previous_top + plan.tile_size, plan.image_height)
        top_overlap = max(0, previous_bottom - tile.top) * scale
    if tile.row + 1 < len(plan.row_starts):
        next_top = plan.row_starts[tile.row + 1]
        bottom_overlap = max(0, tile.bottom - next_top) * scale
    if tile.column > 0:
        previous_left = plan.column_starts[tile.column - 1]
        previous_right = min(previous_left + plan.tile_size, plan.image_width)
        left_overlap = max(0, previous_right - tile.left) * scale
    if tile.column + 1 < len(plan.column_starts):
        next_left = plan.column_starts[tile.column + 1]
        right_overlap = max(0, tile.right - next_left) * scale

    height = tile.height * scale
    width = tile.width * scale
    vertical = _linear_axis_weights(height, top_overlap, bottom_overlap)
    horizontal = _linear_axis_weights(width, left_overlap, right_overlap)
    weights = np.ascontiguousarray(vertical[:, None] * horizontal[None, :])
    if not np.isfinite(weights).all() or float(weights.min()) <= 0.0:
        raise TileAssemblyError("Blending weights must be finite and strictly positive")
    return weights


__all__ = [
    "BLENDING_METHOD",
    "DEFAULT_MAX_TILE_COUNT",
    "ExcessiveTileCountError",
    "InvalidTileConfigurationError",
    "TileAssemblyError",
    "TileCoordinate",
    "TilePlan",
    "TileResourceLimitError",
    "TilingError",
    "blending_weights",
    "create_tile_plan",
]
