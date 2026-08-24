# Single-image restoration service contract

Milestone 13 connects the validated preprocessing boundary, a ready persistent
`ModelManager`, one frozen NAF-SR forward call, validated postprocessing, and
lossless PNG encoding. It does not add padding, cropping, tiling, batching,
diagnostics, quality claims, API routes, or an async platform adapter.

## Complete data flow

For each successful call, `SingleImageRestorationService.restore`:

1. requires an already-ready manager and obtains its persistent model without
   invoking `load()`;
2. preprocesses one supported Path, encoded byte string, NumPy array, or PIL
   image exactly once into a contiguous CPU FP32 `(1,1,H,W)` tensor;
3. rejects images above the configured direct-inference pixel limit;
4. transfers only the input tensor to the manager's resolved device as FP32;
5. invokes the same model exactly once under `torch.inference_mode()`, without
   autocast;
6. postprocesses exactly once, enforcing the exact 2x output dimensions and
   recording clipping;
7. returns a grayscale float32 array and an in-memory 8-bit or 16-bit PNG.

Inputs and outputs are never written to disk. No checkpoint loading, hashing,
or deserialization occurs per restoration.

## Existing internal padding

The migrated NAF-SR forward path already replicate-pads its internal feature
input on the bottom and right to its encoder-compatible multiple and crops the
learned 2x residual back to the requested output extent. The bicubic residual
path uses the original input dimensions. This service deliberately adds no
second padding or cropping implementation. Milestone 14 will formalize and
validate arbitrary-dimension behavior at the service level.

## Result and scientific output

`SingleImageRestorationResult` contains the restored contiguous float32
grayscale array, PNG bytes, bit depth, dimensions, scale, nested preprocessing
and postprocessing metadata, phase timings, manager identity, checkpoint
SHA-256, and combined warnings. Its `metadata()` method returns a JSON-compatible
mapping without image arrays, tensors, or encoded payload contents.

Output is always `image/png`. Sixteen-bit PNG is the default to retain more
quantized intensity levels; callers may explicitly request 8-bit PNG. JPEG and
TIFF output are not supported because the completed scientific output boundary
only guarantees exact lossless grayscale PNG round trips. No confidence,
accuracy, or quality score is fabricated.

## Direct-inference resource limit

The default direct input limit is 262,144 internally aligned/padded pixels (for
example, an already aligned 512x512 image). As formalized in Milestone 14, this
uses the model's actual `padder_size` rather than raw image area because padded
area better represents encoder/decoder work. This conservative guard remains
lower than preprocessing's decode limit and is not a universal CPU/GPU safety
claim. Controlled deployments may deliberately override it, but it cannot
exceed the configured preprocessing pixel limit. Rejections explicitly advise
using the separately selected `restore_tiled()` path documented for Milestone 15.

## Concurrency and timing

Each service owns one inference lock. Preprocessing, postprocessing, and PNG
encoding remain outside it; the single model forward call is serialized so
concurrent callers cannot use one PyTorch model instance simultaneously. This
is conservative and predictable but limits each service instance to one active
forward call. The platform's outer inference gate still controls admission,
timeouts, and capacity across requests.

Timings use the monotonic high-resolution performance clock and report
preprocessing, input-device transfer, inference-lock wait, model inference,
postprocessing including PNG encoding, and total wall time. Values are
nonnegative, and total wall time includes phase and orchestration overhead. CUDA
is synchronized around transfer and forward measurement only where needed for
accurate timings; those synchronizations themselves affect benchmark behavior.

## Safe error categories

The service maps failures to stable typed categories:

- `manager_not_ready`;
- `invalid_input`;
- `resource_limit`;
- `unsupported_output`;
- `preprocessing_failure`;
- `device_transfer_failure`;
- `model_inference_failure`;
- `invalid_model_output`;
- `postprocessing_failure`.

Errors do not include checkpoint paths, tensor contents, internal exception
messages, or tracebacks from underlying operations.

## Platform adapter boundary

The refreshed read-only audit of `origin/work/platform-likhitha` confirms an
async `ModelService` protocol with `startup()`, `shutdown()`, `restore()`, and
synchronous `health()`. A later platform-owned adapter can load/close the
manager during lifespan, run this blocking service through an async thread
boundary, translate the result to platform `RestorationResult`, and map safe
error categories. Platform `ValidatedUpload.encoded_bytes` is directly
acceptable here. The adapter remains deferred and no platform module is
imported or modified by this milestone.
