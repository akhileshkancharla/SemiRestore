from __future__ import annotations

import json

import numpy as np
import pytest

from semirestore import tiling


def _plan(
    width: int,
    height: int,
    *,
    tile_size: int = 16,
    overlap: int = 4,
    limit: int = 256,
    max_tiles: int = 100,
) -> tiling.TilePlan:
    return tiling.create_tile_plan(
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        overlap=overlap,
        alignment=8,
        scale_factor=2,
        max_padded_pixels_per_tile=limit,
        max_tile_count=max_tiles,
    )


@pytest.mark.parametrize("dimensions", [(7, 5), (16, 16)])
def test_image_smaller_than_or_equal_to_tile_uses_one_tile(
    dimensions: tuple[int, int],
) -> None:
    plan = _plan(*dimensions)

    assert plan.tile_count == 1
    tile = plan.tiles[0]
    assert (tile.left, tile.top, tile.right, tile.bottom) == (0, 0, *dimensions)


def test_nondivisible_plan_has_deterministic_half_open_coordinates() -> None:
    plan = _plan(31, 25)

    assert plan.column_starts == (0, 12, 24)
    assert plan.row_starts == (0, 12)
    assert plan.tile_count == 6
    assert plan.tiles[0].to_dict()["index"] == 0
    assert (plan.tiles[-1].left, plan.tiles[-1].right) == (24, 31)
    assert (plan.tiles[-1].top, plan.tiles[-1].bottom) == (12, 25)
    assert plan.tiles[-1].output_right == 62
    assert plan.tiles[-1].output_bottom == 50
    assert plan.to_summary() == _plan(31, 25).to_summary()
    json.dumps(plan.to_summary(), allow_nan=False)


@pytest.mark.parametrize("dimensions", [(13, 11), (1, 31), (31, 1), (17, 29)])
def test_odd_prime_narrow_tall_and_rectangular_plans_cover_every_pixel(
    dimensions: tuple[int, int],
) -> None:
    width, height = dimensions
    plan = _plan(width, height)
    coverage = np.zeros((height, width), dtype=np.uint16)

    for tile in plan.tiles:
        coverage[tile.top : tile.bottom, tile.left : tile.right] += 1

    assert int(coverage.min()) >= 1
    assert plan.tiles[0].top == plan.tiles[0].left == 0
    assert max(tile.right for tile in plan.tiles) == width
    assert max(tile.bottom for tile in plan.tiles) == height


@pytest.mark.parametrize("tile_size", [0, -1, True, 7, 12])
def test_invalid_or_unaligned_tile_size_is_rejected(tile_size: object) -> None:
    with pytest.raises(tiling.InvalidTileConfigurationError, match="tile_size"):
        tiling.create_tile_plan(
            image_width=16,
            image_height=16,
            tile_size=tile_size,  # type: ignore[arg-type]
            overlap=0,
            alignment=8,
            scale_factor=2,
            max_padded_pixels_per_tile=256,
        )


@pytest.mark.parametrize("overlap", [-1, True, 1.0, 16, 17])
def test_invalid_overlap_is_rejected(overlap: object) -> None:
    with pytest.raises(tiling.InvalidTileConfigurationError, match="overlap"):
        tiling.create_tile_plan(
            image_width=16,
            image_height=16,
            tile_size=16,
            overlap=overlap,  # type: ignore[arg-type]
            alignment=8,
            scale_factor=2,
            max_padded_pixels_per_tile=256,
        )


def test_excessive_tile_count_is_rejected_before_coordinate_materialization() -> None:
    with pytest.raises(tiling.ExcessiveTileCountError, match="Tile count"):
        _plan(100, 100, tile_size=8, overlap=7, limit=64, max_tiles=10)


def test_per_tile_aligned_resource_limit_is_enforced() -> None:
    with pytest.raises(tiling.TileResourceLimitError, match="Aligned tile"):
        _plan(17, 17, tile_size=16, overlap=4, limit=255)


def test_blending_weights_are_positive_and_cover_scaled_output() -> None:
    plan = _plan(31, 25)
    accumulated = np.zeros((50, 62), dtype=np.float32)

    for tile in plan.tiles:
        weights = tiling.blending_weights(plan, tile)
        assert weights.shape == (tile.height * 2, tile.width * 2)
        assert float(weights.min()) > 0.0
        accumulated[
            tile.output_top : tile.output_bottom,
            tile.output_left : tile.output_right,
        ] += weights

    assert np.isfinite(accumulated).all()
    assert float(accumulated.min()) > 0.0


def test_constant_raw_tiles_remain_constant_after_weighted_assembly() -> None:
    plan = _plan(31, 25)
    accumulated = np.zeros((50, 62), dtype=np.float32)
    total_weights = np.zeros_like(accumulated)

    for tile in plan.tiles:
        weights = tiling.blending_weights(plan, tile)
        target = np.s_[
            tile.output_top : tile.output_bottom,
            tile.output_left : tile.output_right,
        ]
        accumulated[target] += np.float32(0.375) * weights
        total_weights[target] += weights

    assembled = accumulated / total_weights
    np.testing.assert_allclose(assembled, np.float32(0.375), atol=1e-7, rtol=0)
