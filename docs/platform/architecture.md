# Architecture

## Architecture

SemiRestore is split into a transport/platform layer and a future model layer:

- `semirestore.platform` owns typed runtime settings, safe health metadata, the
  `ModelService` protocol, its exception taxonomy, and `RestorationResult`.
- `semirestore.api` owns application construction, lifespan state, dependency
  injection, upload validation, HTTP serialization, error handling, request
  correlation, logs, metrics, and inference admission control.
- The future model adapter will own the translation to and from
  `SemiRestorePipeline`. It must satisfy the existing protocol without moving
  model behavior into the API.

Each application object has an isolated `ApplicationRuntime`. That runtime owns
one settings object, one Prometheus registry, and—only while the lifespan is
active—one inference gate and at most one model-service instance. Dependencies
retrieve those exact application-local objects; routes never construct a model
service.

## Application lifecycle

`create_app()` builds the FastAPI application without loading a model. On
lifespan startup it:

1. constructs one `InferenceGate` from the validated settings;
2. calls the explicitly supplied model-service factory at most once;
3. calls `startup()` once on the returned service;
4. publishes that same successfully started instance through application state;
5. marks startup complete, even when no adapter was configured or initialization
   failed, so operational endpoints remain available.

No factory means no service. `SEMIRESTORE_ENABLE_FAKE_MODEL_SERVICE` defaults to
false and does not activate production fake behavior. A failed service startup
is contained, the partial service receives one best-effort `shutdown()`, and
readiness remains false with a generic safe reason.

On lifespan shutdown the application first removes the service and gate from
runtime state, marks the service stopped, and then calls the started service's
`shutdown()` once. A shutdown exception cannot prevent application shutdown.
Resources are process-local: multiple workers would repeat construction and
checkpoint ownership.

## Dependency and request flow

A restoration request follows one platform-owned path:

1. request middleware selects a safe request ID and starts monotonic timing;
2. multipart upload validation reads a bounded byte count, verifies the declared
   and detected image format, fully decodes the image, and checks dimensions;
3. the route resolves the lifespan-owned service and checks current readiness;
4. the lifespan-owned gate admits, rejects, or times out the operation;
5. the adapter returns a `RestorationResult`, which is revalidated at the
   boundary;
6. the API Base64-encodes output bytes and serializes only typed safe metadata;
7. bounded metrics and privacy-safe completion events record the outcome.

Upload validation finishes before inference admission. Invalid content cannot
hold scarce model capacity, and no model construction or checkpoint loading
occurs per request.

## Privacy boundary

Inputs and restored outputs remain in memory. The platform does not create
upload, output, checkpoint, cache, or result files. Multipart filenames are not
trusted as paths and are never used for persistence.

Public responses, application logs, and metric labels exclude raw exceptions,
stack traces, query strings, multipart boundaries, filenames, image bytes,
Base64 payloads, local or checkpoint paths, tensors, secrets, and internal
environment details. The adapter must preserve this boundary. Diagnostics and
warnings are public data and therefore must be deliberately safe and bounded
before they enter `RestorationResult`.
