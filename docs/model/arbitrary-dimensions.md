# Arbitrary SEM image dimension contract

Milestone 14 makes the existing checkpoint-compatible NAF-SR spatial behavior
explicit and observable. It adds planning and validation only: model
mathematics, internal padding, inference count, PNG encoding, and checkpoint
compatibility are unchanged. Tiled and batch inference remain out of scope.

## Audited internal implementation

The configured conditioned model has three encoder downsampling stages, so
`NAFSR.padder_size` is `2 ** 3 = 8`. In `forward`, the original input height and
width are recorded before any padding. `_pad` computes the smallest multiples
of eight and calls:

```text
F.pad(input, (left=0, right=pad_width, top=0, bottom=pad_height), mode="replicate")
```

Only the right and bottom margins are extended. Aligned inputs receive zero
padding. Unaligned inputs use edge replication, not zeros or reflection.

Conditioning is deliberately calculated before `_pad`. Its global mean,
population standard deviation, minimum, and maximum therefore describe only
the original SEM image; replicated padding pixels do not change those
statistics. This order is checkpoint behavior and has not been moved.

The padded tensor enters the learned encoder/decoder and 2x super-resolution
head. That head can produce the aligned internal extent, after which the
learned residual is cropped to `original_height * 2` and
`original_width * 2`. Separately, bicubic interpolation operates directly on
the original unpadded input with a scale factor of two. The cropped learned
residual and original-extent bicubic result are then added, guaranteeing a
final `2H x 2W` output.

No external padding layer was added because doing so would duplicate the model
contract and could alter conditioning, bicubic behavior, cropping, or resource
accounting.

## Spatial plan

`create_spatial_plan` uses integer-only ceiling alignment and allocates no image
tensors. Its typed result records:

- original dimensions and model alignment;
- padded dimensions and right/bottom padding;
- raw and padded input pixel counts;
- padding overhead pixels and the overhead fraction relative to raw pixels;
- internal restored dimensions before crop;
- final restored dimensions after crop;
- scale factor and whether padding is required.

The planner accepts positive integer dimensions, a positive power-of-two
alignment, and the supported scale factor two. Booleans, malformed values,
unsupported scales, dimensions above 1,000,000, padded inputs above one billion
pixels, and calculations outside the signed 64-bit range are rejected. These
planning bounds prevent nonsensical metadata; preprocessing and direct
inference impose much smaller allocation limits.

Examples with alignment eight and scale two:

| Input | Padded input | Right/bottom pad | Internal restored | Final restored |
|---|---|---|---|---|
| `8x16` | `8x16` | `0/0` | `16x32` | `16x32` |
| `9x11` | `16x16` | `5/7` | `32x32` | `18x22` |

Dimensions are shown as height by width; right/bottom padding is width then
height.

## Restoration and resource accounting

The service reads alignment from the actual loaded model's `padder_size` and
scale from verified manager identity, creates the plan before model execution,
and passes the original tensor to the model unchanged. It validates the
observed postprocessed extent against the planned final extent and exposes the
full plan in each result.

The existing configurable direct-inference limit now means **aligned padded
input pixels**, not raw pixels. This is a compatibility clarification: aligned
area better represents encoder/decoder activation work. For example, a `15x17`
input contains 255 raw pixels but aligns to `16x24`, so it consumes 384 pixels
of the limit. The default remains 262,144 padded pixels. It is a conservative
pre-tiling guard, not a universal safety guarantee for every CPU or GPU.

When padding overhead is at least 25% of raw input area, the result includes a
warning with the added compute pixels and percentage. Small, narrow, or tall
images may have high relative overhead even though their absolute cost is low.

## Supported dimensions and limitations

Aligned, odd, prime, mixed aligned/unaligned, sub-alignment, narrow, tall,
square, and rectangular positive dimensions are supported when they pass
preprocessing and aligned-compute limits. The final dimensions remain exactly
twice the original dimensions. Planning does not prove that an arbitrarily
large image fits available host or device memory; larger workloads require the
explicit memory-aware tiled path introduced in Milestone 15.
