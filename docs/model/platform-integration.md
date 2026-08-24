# Platform adapter mapping

This audit inspected `origin/work/platform-likhitha` at
`91d2276d206dda0c1e6dc161bb511da98cf64558`. No platform-owned file was changed.
The actual platform protocol requires async `startup()`, async `shutdown()`,
async `restore(ValidatedUpload)`, and synchronous `health()`.

## Lifecycle mapping

- `startup`: call `SemiRestorePipeline.from_config(...)` once in
  `asyncio.to_thread`; retain that pipeline and its verified status. Translate
  checkpoint, configuration, or device failures to
  `ModelServiceInitializationError` and remain unavailable.
- `shutdown`: stop accepting work, detach the retained pipeline, and call its
  idempotent-effect `close()` once. Report the platform `stopped` state.
- `health`: read the retained pipeline's cached manager status synchronously.
  Map ready state, actual device, model version, and checksum into
  `ModelHealth`; never load, hash, or run inference here.
- `restore`: pass `ValidatedUpload.encoded_bytes` directly to
  `pipeline.restore_and_analyze` through `asyncio.to_thread`. The platform
  `InferenceGate` remains the only admission controller.

## Result mapping

`RestorationResult.platform_projection()` maps PNG bytes/media type,
restored/original dimensions, restoration latency, actual device, model version,
checkpoint checksum, bounded JSON diagnostics, and immutable public warnings
to the platform-owned `RestorationResult`. The API then base64-encodes those PNG
bytes into `RestoreResponse.image`; input transport metadata continues to come
from `ValidatedUpload`.

Only `image/png` is emitted even though the platform schema can transport JPEG
or TIFF. Model diagnostics are deliberately projected into primitives and stay
below the platform's 65,536-byte limit for the standard result. Paths, image
arrays, tensors, exception messages, and checkpoint contents never cross the
boundary.

## Error mapping

- Startup/configuration/checkpoint/device failure:
  `ModelServiceInitializationError`.
- Restore before readiness: `ModelServiceUnavailableError`.
- Known preprocessing, diagnostic, inference, output, or serialization failure:
  `ModelServiceInferenceError` with no original public message.
- `asyncio.CancelledError`: re-raise unchanged.

The platform route maps these to safe 503 or 500 responses. The adapter must
not add a semaphore, fake fallback, per-request checkpoint load, or another
upload policy. One application process owns one adapter and one pipeline;
additional Uvicorn workers would allocate additional models.
