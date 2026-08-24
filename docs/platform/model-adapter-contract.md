# Model adapter contract

## Purpose and ownership

The future adapter is the only translation layer between the platform-owned
`ModelService` protocol and the model-owned API:

```python
from semirestore.pipeline import SemiRestorePipeline

pipeline = SemiRestorePipeline.from_config(...)
result = pipeline.restore_and_analyze(image)
```

Those imports are intentionally absent from the platform branch because the
final pipeline is not available here. This document defines the integration
contract; it is not an implementation of the model or adapter.

Create exactly one adapter instance per application process. The application
lifespan calls its factory once, `startup()` once, reuses the same instance for
health and every restoration request, and calls `shutdown()` once. Never build a
pipeline or load a checkpoint in a route or per request.

## Lifecycle and checkpoint verification

Before advertising readiness, `startup()` must:

1. require the model configuration and checkpoint settings expected by the
   final pipeline;
2. verify that each required artifact exists, is the intended regular file, and
   is compatible with the model/configuration version;
3. compute the public checkpoint checksum from the exact bytes that will be
   loaded, using a stable algorithm such as SHA-256;
4. resolve the device policy and load the checkpoint through
   `SemiRestorePipeline.from_config(...)` exactly once;
5. complete any model-owned warm-up or integrity checks required to prove that
   `restore_and_analyze` can accept work;
6. retain only the long-lived pipeline and safe public identity metadata;
7. change health to `READY` only after every required step succeeds.

Partial startup must leave the adapter unready and release any resources it
already acquired. `shutdown()` must be idempotent in effect, stop accepting new
work, release model/device resources, and set a safe stopped state. The platform
performs best-effort cleanup after startup or shutdown failures, so the adapter
must not rely on a second call.

### Missing or invalid checkpoints

A missing, non-regular, unreadable, corrupt, incompatible, or unverifiable
checkpoint must raise `ModelServiceInitializationError` during startup. It must
never cause random weights, an embedded demo model, or a fake service to become
ready. The application will remain live, report HTTP 503 from `/health/ready`,
report safe unavailable health, and return `model_unavailable` for restoration.

Do not put a checkpoint path or raw loader exception in health, responses, logs,
or metrics. A safe fixed reason such as `model checkpoint is unavailable` may be
used inside adapter health; the application already uses a generic reason when
startup itself fails.

## Readiness and health mapping

The adapter's `health()` method is synchronous, fast, side-effect-free, and must
not load files, run inference, or probe a remote service. Map adapter state to
`ModelHealth` as follows:

| Adapter condition | `state` | `ready` | Public metadata |
| --- | --- | ---: | --- |
| Initialization in progress | `starting` | false | safe reason; identity only if verified |
| Verified pipeline can accept work | `ready` | true | device, model version, checksum |
| Missing/failed/degraded pipeline | `unavailable` | false | safe reason; no unverified claims |
| Resources released | `stopped` | false | safe stopped reason |

`ready` must be true if and only if `state` is `ready`. A ready service cannot
have `unavailable_reason`; an unready service must have one. `device`,
`model_version`, and `checkpoint_checksum` are optional safe public identifiers,
not paths or arbitrary exception text.

Device policy should map `device_preference=cpu` to CPU. With `auto`, a verified
CUDA configuration may be selected and CPU is the fallback when CUDA is not
usable. An explicit `cuda` request should fail initialization rather than
silently changing deployment semantics. The model team must verify these rules
against checkpoint loading and report the actual selected device, not merely
the requested device.

## Input mapping

`restore()` receives one immutable `ValidatedUpload`:

| Field | Adapter use |
| --- | --- |
| `encoded_bytes` | Decode in memory into the exact image object expected by the pipeline |
| `media_type` | Canonical transport type already matched to detected content |
| `detected_format` | `PNG`, `JPEG`, or `TIFF` identity already verified by the platform |
| `width`, `height` | Original dimensions to preserve in the result contract |

The platform has already bounded the encoded read, verified the format and
single-frame structure, fully decoded the image, and enforced dimensions. The
adapter must not duplicate those transport/security decisions or introduce a
second competing upload policy. It may perform the one model-required in-memory
decode and the scientific preprocessing owned by `SemiRestorePipeline`.

Any PIL image, NumPy array, or tensor created for `restore_and_analyze(image)` is
strictly adapter-internal and must be closed or released before return. Do not
persist the upload and do not turn the multipart filename into a filesystem
path.

## Output mapping

Translate the model result into one `RestorationResult` before crossing back to
the platform:

| `RestorationResult` field | Required mapping |
| --- | --- |
| `restored_image_bytes` | Non-empty encoded PNG, JPEG, or single-frame TIFF bytes |
| `restored_media_type` | Media type matching those exact encoded bytes |
| `restored_width`, `restored_height` | Dimensions of the encoded restoration |
| `original_width`, `original_height` | Exact dimensions from `ValidatedUpload` |
| `inference_latency_ms` | Optional finite, non-negative model-measured latency |
| `device` | Optional actual safe device identifier |
| `model_version` | Optional safe version for the loaded model/config pair |
| `checkpoint_checksum` | Optional checksum of the exact verified checkpoint |
| `diagnostics` | Bounded JSON-compatible mapping safe for public response |
| `warnings` | Immutable tuple of bounded, printable, public suitability warnings |

Encode restored pixels inside the adapter. No tensor, NumPy array, PIL object,
file handle, generator, custom class, NaN/infinity, or lazy device object may
cross the boundary. `RestorationResult` enforces non-empty bytes, supported media
types, positive dimensions, finite latency, safe identities, JSON diagnostics,
and safe warnings, and the route revalidates that result before serialization.

The adapter must deliberately project model diagnostics into JSON primitives;
it must not serialize an object's `repr` or dump an unrestricted model result.
Suitability warnings are model-owned scientific guidance, not proof of
correctness. They must omit paths, secrets, tensor dumps, and input content.

## Exception translation

Translate at the adapter boundary without exposing original messages:

| Condition | Exception crossing the boundary |
| --- | --- |
| Configuration/checkpoint/device failure during startup | `ModelServiceInitializationError` |
| A started service temporarily cannot accept work | `ModelServiceUnavailableError` |
| Expected model inference or output-conversion failure | `ModelServiceInferenceError` |
| Caller/task cancellation | re-raise `asyncio.CancelledError` unchanged |

The exception message is for private debugging only and is never a public
contract. Platform handlers map availability failures to HTTP 503
`model_unavailable`, known inference failures to HTTP 500 `restoration_failed`,
and unexpected exceptions to generic HTTP 500 `internal_error`. Do not attach
images, arrays, tensors, paths, checkpoint contents, or secrets to exceptions.

## Synchronous inference behind the async boundary

`ModelService.restore()` is asynchronous even if
`pipeline.restore_and_analyze(image)` is synchronous. CPU-bound synchronous work
should run outside the event-loop thread, for example through
`asyncio.to_thread`, while the adapter retains ownership of temporary objects.
Do not create an independent adapter semaphore: the application-level
`InferenceGate` already controls admission and timeout around the call.

Cancellation of the awaiting coroutine cannot forcibly stop a Python worker
thread, native library call, or already-submitted GPU kernel. GPU timing may also
require explicit synchronization before producing bytes and latency. These
limitations must be tested with the real pipeline; timeout capacity settings
must not assume that underlying work stopped immediately.

The safe deployment policy remains one Uvicorn worker. Extra workers each create
their own adapter and load their own checkpoint, multiplying CPU/GPU memory and
effective inference concurrency.

## Integration invariants

The real integration must preserve all of these invariants:

- one pipeline construction and verified checkpoint load per process startup;
- no model construction or checkpoint load per request;
- no fake fallback in production or after initialization failure;
- no duplicate platform upload validation or competing transport preprocessing;
- all scientific preprocessing remains inside the model pipeline/adapter;
- the application gate remains the only platform admission controller;
- no non-serializable or device-backed object crosses the protocol;
- shutdown releases long-lived resources once;
- public health, diagnostics, warnings, errors, logs, and metrics remain safe.
