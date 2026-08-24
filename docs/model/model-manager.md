# Persistent model manager contract

Milestone 12 introduces the process-local lifecycle owner for the verified
statistics-conditioned NAF-SR model. It does not preprocess images, run
inference, postprocess output, pad or tile inputs, compute diagnostics, or
implement API/platform adapters.

## Platform protocol audit

The read-only audit of `origin/work/platform-likhitha` found the platform
contract in `src/semirestore/platform/model_service.py`. The application owns
one `ModelService` for each FastAPI lifespan and expects:

- asynchronous `startup()` and `shutdown()` lifecycle methods;
- synchronous `health()` returning safe state, readiness, device, model
  version, checkpoint checksum, and an unavailable reason;
- asynchronous `restore(ValidatedUpload)` returning the platform-owned
  `RestorationResult`;
- public states `starting`, `ready`, `unavailable`, and `stopped`.

The platform schemas are in `src/semirestore/api/schemas.py`, not
`src/semirestore/platform/schemas.py`. The restoration route checks health,
runs `restore()` through the platform inference-concurrency gate, and expects
lossless image bytes plus identity and diagnostic metadata.

`ModelManager` is deliberately the synchronous lifecycle dependency beneath a
future platform service adapter. That adapter can run `load()` without blocking
the event loop, map manager states to platform health states, and use manager
identity in responses. The required `restore()` method combines preprocessing,
inference, and postprocessing and therefore remains deferred to Milestone 13.
No imports from the not-yet-merged platform package are added on this branch.

The intended adapter mapping is `unloaded`/`loading` to `starting`, `ready` to
`ready`, `failed` to `unavailable`, and `closed` to `stopped`. Failed health
must use a stable public reason derived from the manager's category rather than
the underlying exception message.

One contract mismatch is recorded for later integration: the platform result
currently permits PNG, JPEG, and TIFF, while the scientific postprocessing
boundary intentionally emits only lossless grayscale PNG. The future adapter
should advertise and return PNG rather than weakening the model boundary.

## Lifecycle and retry policy

Each manager starts `unloaded` and moves through `loading` to `ready`, or to
`failed` when loading cannot complete. `load()` delegates to the checksum-gated
safe loader and publishes the model only after the complete load succeeds.
Concurrent callers wait for the same attempt; a successful model is loaded
once and every later call or property access returns the same instance.

Failures expose only a stable category such as `checkpoint_verification` or
`device_selection`; checkpoint exception messages and tracebacks are not part
of status. Failed managers never retry implicitly. An operator must call
`reset_failure()` before a new `load()` attempt. Reset is rejected during
loading, after readiness, and after closure.

`close()` is permanent and idempotent. It drops the manager's strong reference
to the model and prevents loading, reset, or model access afterward. Status and
readiness remain inspectable. Identity metadata already established by a
successful load is tensor-free and remains available after closure. External
callers that retained the returned model reference must release it separately.

## Status and safety

`status()` never triggers loading. It returns an immutable snapshot with state,
readiness, model name, architecture, model version and training revision,
resolved device, parameter count, safe runtime checkpoint path, verified
SHA-256, scale factor, failure category, and retry permission. Paths outside
the project are reduced to their filename; immutable training-source paths,
checkpoint contents, tensors, exceptions, tracebacks, and secrets are never
included.

The safe loader remains responsible for configuration validation, size and
checksum verification, `weights_only=True` deserialization, strict state-dict
compatibility, and device selection. The manager additionally enforces eval
mode and disables parameter gradients without moving the selected model.

## Thread and process model

Lifecycle state, readiness, status, model publication, reset, and closure are
thread-safe. The manager does not hold its lifecycle lock while checkpoint
verification and loading run. This guarantee does not make future inference
calls automatically safe for arbitrary concurrency; the platform's inference
gate or a model-service policy must serialize or bound inference as appropriate
for the selected device.

Use one manager per application process. Independent worker processes have
separate Python address spaces and cannot transparently share the PyTorch model
object. Consequently, each worker loads its own checkpoint and owns a separate
copy of model parameters and runtime buffers. On GPU, each worker therefore
consumes its own device-memory allocation; worker count must be sized against
available GPU memory. Distributed serving and multiprocessing are outside this
milestone.
