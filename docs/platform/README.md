# SemiRestore platform ownership

The platform track owns the service boundary around the SemiRestore model. Its
code lives in `src/semirestore/platform/` and `src/semirestore/api/`; its tests
live in `tests/platform/` and `tests/api/`. Model architecture, checkpoints,
scientific image processing, inference, training, evaluation, and diagnostics
remain model-track responsibilities.

## Milestone 1 boundary

This milestone provides typed runtime settings and package namespaces only. It
does not create an HTTP application, load a model, inspect a checkpoint, or
perform restoration. Later milestones will consume these settings through a
narrow platform-side model-service adapter.

`RuntimeSettings` reads environment variables prefixed with `SEMIRESTORE_`.
Defaults are development-safe: the service binds to loopback, inference
concurrency is one, and the development fake model service is disabled. Model
configuration and checkpoint paths are opaque optional values that will be
passed to the future adapter; platform settings do not open or validate them.

The settings cover runtime identity and networking, logging, encoded and
decoded upload limits, accepted media types, inference concurrency and
timeouts, model-boundary paths, device preference, and explicit fake-service
enablement. Secrets do not belong in committed configuration.

## Local environment

The project requires Python 3.11 or newer. On a Windows system where `python`
still selects Python 3.10, create the environment explicitly with an installed
supported interpreter, for example:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[platform,dev]"
.\.venv\Scripts\python -m pytest
```

Docker is not required for this milestone. Container packaging is deferred to
Platform Milestone 11.

## Restoration response transport

`POST /api/v1/restore` accepts one multipart field named `image`. The response
keeps the restored image and its metadata together by encoding the image bytes
as Base64 in JSON. Base64 increases the encoded payload size by approximately
one-third; a binary response may be added later if large-image workloads make
that overhead material. Uploaded and restored images are not persisted.

## Inference capacity and timeouts

Each application lifespan owns one bounded inference controller configured by
`SEMIRESTORE_INFERENCE_CONCURRENCY_LIMIT`,
`SEMIRESTORE_CONCURRENCY_ACQUISITION_TIMEOUT_SECONDS`, and
`SEMIRESTORE_INFERENCE_TIMEOUT_SECONDS`. Upload size, decoding, and dimension
validation finish before a request waits for inference capacity, so invalid
uploads cannot occupy an expensive model slot. A request that cannot acquire a
slot in time receives HTTP 503 `inference_busy`; inference that exceeds its
execution timeout receives HTTP 504 `inference_timeout`. Slots are released on
success, failure, timeout, and cancellation.

Async cancellation stops a cooperative restoration coroutine. If a future
adapter offloads blocking CPU or GPU work to a thread or native runtime,
cancelling the awaiting coroutine may not physically stop that work or GPU
kernels immediately. Capacity-release behavior must be reviewed with the real
adapter before claiming hard inference cancellation.

## Request correlation and application logs

Every HTTP request receives an opaque request ID before routing. A caller may
provide `X-Request-ID` when its value is 1–64 printable ASCII characters from
letters, digits, `.`, `_`, `:`, and `-`, beginning with a letter or digit.
Missing, ambiguous, or invalid values are replaced with a generated UUID. The
validated ID is returned in the `X-Request-ID` response header and included in
the stable API error envelope.

Application events use the isolated `semirestore` logger. The
`SEMIRESTORE_LOG_LEVEL`, `SEMIRESTORE_JSON_LOGGING`, and
`SEMIRESTORE_ENVIRONMENT` settings control its level, format, and environment
field. JSON is the default; human-readable output is available for local use.
HTTP 2xx/3xx completions log at INFO, 4xx at WARNING, and 5xx at ERROR. Each
completion records monotonic elapsed time, the resolved route template (or
`<unmatched>`), status class, and stable error code where applicable.

Inference events record the platform-observed interval around model-service
availability checks, inference-capacity waiting, and the adapter call. This is
not model-reported latency and can therefore be longer than the optional
`inference.latency_ms` returned by an adapter. Logged outcomes are limited to
success, busy, timeout, unavailable, failed, and cancellation. Request and
inference logs never inspect or emit query strings, request or response bodies,
multipart boundaries, filenames, headers other than the validated request ID,
filesystem or checkpoint paths, raw exceptions, images, tensors, or secrets.
Cancellation continues to propagate and emits at most one deliberately named
cancellation event; it is never reported as a successful completion or an
`internal_error`.

## Prometheus-compatible metrics

`GET /metrics` exposes only the current application's isolated Prometheus
registry using the official text exposition content type and a
`Cache-Control: no-store` response header. It remains available without model
readiness and is deliberately
excluded from OpenAPI and ordinary HTTP request metrics, avoiding scrape-driven
self-noise. Each application instance owns a fresh registry; tests and multiple
applications therefore cannot collide or share time-series values.

The registry contains these collectors:

- `semirestore_http_requests_total` counter and
  `semirestore_http_request_duration_seconds` histogram, labelled by bounded
  method, resolved route template, and `2xx`, `3xx`, `4xx`, `5xx`, or
  `cancelled` status class.
- `semirestore_restoration_requests_total` counter, labelled by `success`,
  `busy`, `timeout`, `unavailable`, `failed`, or `cancelled` outcome.
- `semirestore_inference_duration_seconds` histogram with the same bounded
  outcome labels. It measures platform orchestration after upload validation,
  including capacity acquisition and the adapter call, rather than the model's
  optional self-reported latency.
- `semirestore_inference_active`, `semirestore_inference_waiting`, and
  `semirestore_inference_capacity` gauges. Waiting covers capacity acquisition;
  active begins only after acquisition, and both return to zero on every
  terminal or cancellation path.
- `semirestore_inference_busy_total` and
  `semirestore_inference_timeouts_total` counters, which distinguish capacity
  acquisition rejection from execution timeout after acquisition.

HTTP duration buckets are 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5,
5, and 10 seconds. Inference buckets are 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
1, 2.5, 5, 10, 30, 60, and 120 seconds. Unusual methods become `OTHER`, and
unmatched paths become `<unmatched>`.

Request IDs, raw paths, query strings, filenames, client content types, image
dimensions or content, exception messages, diagnostics, model versions,
devices, checkpoint details, and other user-controlled values are never metric
labels. A readiness gauge is intentionally deferred: current health is obtained
on demand and no lifecycle hook can guarantee a continuously accurate gauge
without risking stale or misleading readiness data.
