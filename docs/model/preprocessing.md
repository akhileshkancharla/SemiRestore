# SEM image preprocessing contract

Milestone 10 defines the validated, single-image CPU boundary used before the
frozen conditioned NAF-SR model. It does not perform inference, padding, output
conversion, or diagnostics.

## Historical convention discovered

The training revision `d037473ddf4a3cd20eb3fef933991cd66749f4f2` loaded
organizer `.npy` samples with `numpy.load(..., allow_pickle=False)`, required a
finite real 2D array, converted it directly to contiguous float32, and did not
clip or normalize it. The training record labels this policy
`raw_float32_no_clip`.

The dataset audit reports float32 degraded inputs of shape `128x128`, with a
global range from approximately `-0.278563` to `2.158005`. Targets were float32
in `[0,1]`. The model's internal mean/std/min/max conditioner therefore saw raw
out-of-range degraded values during training.

The public raster-image boundary required for deployment is intentionally
narrower: every accepted image becomes `[0,1]` without silently changing an
accepted float image's relative intensity scale. Historical raw `.npy` loading
is not part of this milestone. Out-of-range float arrays are rejected rather
than clipped, min-max normalized, or reinterpreted as `[0,255]` data. This
difference from the historical competition input distribution must remain
visible in model limitations and later suitability analysis.

## Canonical result

`preprocess_sem_image` returns `PreprocessingResult` containing:

- one contiguous CPU tensor with shape `(1, 1, H, W)`;
- `torch.float32` values in `[0,1]`;
- original dimensions, dtype, source type, and decoded PIL mode where relevant;
- original minimum and maximum intensity;
- explicit normalization and channel-conversion labels;
- warnings and preprocessing version `semirestore-preprocessing-v1`.

The input array is never modified in place, and encoded uploads are decoded in
memory without being written to disk.

## Intensity policy

| Input dtype | Accepted interpretation | Conversion |
|---|---|---|
| `uint8` | full unsigned 8-bit range | divide by 255 |
| `uint16` | full unsigned 16-bit range | divide by 65,535 |
| `float32` | already normalized `[0,1]` | copied without rescaling |
| `float64` | already normalized `[0,1]` | converted to float32 without rescaling |

Boolean, signed integer, other unsigned integer widths, float16, complex, and
object arrays are rejected. Floating-point NaN, infinity, negative values, and
values above one are rejected. No input uses per-image min-max normalization.

## Grayscale policy

Two-dimensional arrays and `HxWx1` arrays are accepted directly. `HxWx3` and
PIL/encoded RGB inputs are accepted only when all three channels are exactly
identical; they are then collapsed and the conversion is recorded. Arbitrary
RGB conversion is rejected because channel weighting could change scientific
intensity meaning. Alpha, palette, CMYK, two-channel, and other layouts or PIL
modes are unsupported.

Supported PIL modes are `L`, `I;16`, `I;16L`, `I;16B`, `F`, and conditionally
identical `RGB`.

## Resource policy

The default limits are:

- maximum width: 8,192 pixels;
- maximum height: 8,192 pixels;
- maximum total pixels: 16,777,216;
- maximum encoded input size: 67,108,864 bytes.

Dimensions are checked before tensor allocation. Encoded paths are size-checked
before a bounded read. Pillow's decompression-bomb protections remain enabled;
warnings are promoted to local decode failures without changing Pillow's global
settings.

Constant images are valid deterministic inputs. They receive a zero-dynamic-
range warning rather than being rescaled or rejected.
