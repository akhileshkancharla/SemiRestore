# Restoration output postprocessing contract

Milestone 11 defines the boundary between one raw conditioned NAF-SR output and
scientifically traceable restored image data. It performs no inference,
padding, cropping, resizing, diagnostics, storage, or API work.

## Established clipping behavior

The historical inference implementation at training revision
`d037473ddf4a3cd20eb3fef933991cd66749f4f2` checked output shape and finiteness,
then applied `clamp(0, 1)` before converting output to a float32 CPU NumPy
array. The new boundary preserves that behavior without mutating the caller's
tensor.

Clipping is required because the model output head is not range bounded and
can produce small values below zero or above one. Postprocessing records the
raw and clipped ranges and the count and fraction clipped on each side. This is
not accuracy validation: a value can lie in `[0,1]` and still differ from the
unknown ground truth, while clipping can hide the magnitude of an out-of-range
prediction. The recorded statistics make that transformation visible.

No per-image min-max normalization, contrast stretching, sharpening,
denoising, gamma correction, histogram equalization, learned processing, or
silent resizing is performed.

## Validated tensor contract

`postprocess_restoration` accepts a dense, strided PyTorch float16, bfloat16,
float32, or float64 tensor with shape `(1, 1, H_out, W_out)`. CPU and CUDA
tensors are supported. The tensor is detached, clipped out of place, converted
to float32, and transferred to CPU during postprocessing. Sparse, integer,
Boolean, complex, non-finite, empty, multi-batch, multi-channel, and incorrectly
ranked tensors are rejected.

Original dimensions are optional, but width and height must be supplied
together as positive integers. When present, the output must be exactly twice
the original width and height. Incorrect output dimensions are rejected rather
than resized. Padding and final cropping remain outside this milestone.

The result contains a contiguous two-dimensional float32 NumPy image in
`[0,1]`, restored dimensions, original dimensions when supplied, scale factor
two, source dtype and device, clipping statistics, warnings, and version
`semirestore-postprocessing-v1`.

## Quantization and lossless output

Quantization uses deterministic round-half-up conversion:

- 8-bit: `floor(value * 255 + 0.5)`, producing grayscale `uint8` values in
  `[0,255]`;
- 16-bit: `floor(value * 65535 + 0.5)`, producing grayscale `uint16` values in
  `[0,65535]`.

Eight-bit PNG is smaller and broadly interoperable but has only 256 intensity
levels. Sixteen-bit PNG preserves 65,536 intensity levels and is preferred
when downstream scientific tools support it. Pillow's single-channel `L` and
`I;16` PNG paths are used and verified by exact decode round trips in the test
suite. PNG is used because it is lossless; JPEG is intentionally unsupported
because its quantization and compression artifacts alter restored pixels.

PNG bytes are produced in memory and preserve the exact restored dimensions.
Postprocessing alone cannot establish restoration fidelity, recover clipped
information, or measure accuracy without paired ground truth and an evaluation
protocol.
