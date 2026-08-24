# Memory-aware tiled restoration contract

Milestone 15 adds an explicit tiled path for images that pass full-image decode
validation but exceed the direct aligned-compute limit. Direct `restore()` is
unchanged and remains the default. Callers must deliberately select
`restore_tiled()`; the service does not silently switch modes or retry CUDA
out-of-memory failures.

## Global conditioning override

The checkpoint conditions every feature stage from input mean, population
standard deviation, minimum, and maximum, in that exact order. Computing those
values independently per tile would make brightness and contrast conditioning
vary spatially and could create restoration seams.

The model forward method therefore accepts an optional validated
`conditioning_statistics` tensor with dense shape `(batch,4)`. It must match
the input tensor's dtype and device, contain finite values, and have a
nonnegative standard deviation. An override is rejected for an unconditioned
model. The extension adds no modules, parameters, buffers, or state-dictionary
keys. With no override, the original full-input computation runs exactly as
before; regression tests compare direct output with an equivalent override at
zero tolerance, while strict real-checkpoint loading continues to validate all
keys and the 9,111,684 parameter count.

Tiled restoration preprocesses the complete image once on CPU and computes one
mean/std/min/max tensor from that original unpadded image. The small statistics
tensor is transferred to the model device once and the identical tensor is
passed to every tile. Tile-local statistics are never computed.

## Tile coordinates and context

`tile_size` is the maximum full input extent of a tile, including shared
context. It must be a positive multiple of the model's alignment. `overlap` is
the number of input pixels shared by adjacent tiles and must satisfy
`0 <= overlap < tile_size`. The stride is `tile_size - overlap`.

Coordinates are deterministic row-major, half-open input rectangles
`[top,bottom) x [left,right)`. Starts advance by the fixed stride until the
current tile reaches the image boundary. Edge tiles may be smaller than
`tile_size`; the model handles their alignment internally. Output coordinates
are each input coordinate multiplied by two. Images smaller than or equal to a
tile use one tile.

Overlap supplies convolutional context shared with neighboring tiles. Every
tile contributes its complete output, but shared output margins receive
separable linear ramp weights. At an overlap of `N` output pixels, the entering
tile ramps from `1/(N+1)` to `N/(N+1)` and the leaving tile uses the reverse.
Vertical and horizontal ramps are multiplied. Outer image edges remain weight
one, and ramp endpoints are strictly positive, preventing zero-weight corners.

Raw float32 predictions and weights accumulate separately on CPU. No tile is
clipped. After all tiles, every output weight must be finite and positive; the
accumulator is normalized once, checked for finiteness, then sent through the
existing global postprocessing and PNG conversion exactly once. This preserves
exact `2H x 2W` output and 8-bit/16-bit lossless PNG behavior.

Overlap reduces boundary discontinuities but does not make tiled inference
mathematically identical to full-image inference. NAF-SR has a finite but
potentially large receptive field, and a tile lacks context outside its shared
margin. Larger overlap generally improves context continuity while increasing
tile count and compute.

## Planning and resource limits

The planner validates positive integer tile size, nonnegative integer overlap,
alignment compatibility, supported scale, safe dimensions, per-tile aligned
compute, and a configurable maximum tile count. Booleans are rejected as
integers. Tile count is bounded arithmetically before coordinate lists are
materialized.

Each tile receives its own spatial plan, and its internally padded pixel count
must not exceed `max_padded_pixels_per_tile`. By default this uses the service's
262,144-pixel direct compute limit, but callers may deliberately choose a lower
or higher bounded value. The full decoded source must still satisfy the
preprocessing width, height, pixel, and encoded-byte limits.

Normal metadata contains a compact deterministic summary: tile size, overlap,
stride, row and column starts, row/column counts, total tile count, alignment,
scale, actual maximum padded tile pixels, configured per-tile limit, blending
method, and first/last coordinates. It avoids large per-tile telemetry.

## Memory, concurrency, and timing

The full normalized source remains on CPU. Only one sliced FP32 tile is moved
to the selected device at a time. Each forward call uses
`torch.inference_mode()` and the existing per-service inference lock. The raw
tile is returned to CPU, accumulated, and device references are deleted before
the next iteration. The checkpoint is never reloaded, no autograd graph is
retained, no tile is written to disk, and `torch.cuda.empty_cache()` is not
called per tile.

The service serializes each forward call, while preprocessing, planning,
transfers, and CPU assembly remain outside the lock. Concurrent tiled requests
may interleave between tiles but can never execute the shared model
simultaneously.

Metadata reports preprocessing, planning, cumulative inference-lock wait,
cumulative input/output transfer, cumulative model, assembly/blending,
postprocessing/PNG, and total wall latency. CUDA synchronization surrounds only
measurements that require completion visibility and therefore affects timing.

## Errors and scientific limitations

Typed safe errors cover invalid configuration, excessive tile count, per-tile
resource rejection, global-conditioning rejection, tile inference, malformed
or non-finite tile output, invalid/zero blending weights, non-finite assembly,
and global postprocessing. Partially assembled images are never returned.

Tiled output should be compared descriptively with direct output for workloads
where both fit. Difference and seam-region metrics describe implementation
behavior; without ground truth they are not accuracy thresholds or quality
claims. Diagnostics, automatic overlap selection, tiling retries, and hardware-
specific capacity tuning remain outside this milestone.
