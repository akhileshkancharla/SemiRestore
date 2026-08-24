# Development and operations

## Local development

Python 3.11 or newer is required. Install the platform and development optional
dependencies into a virtual environment:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[platform,dev]"
```

Start the existing application factory with one worker:

```powershell
.\.venv\Scripts\python -m uvicorn semirestore.api:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

The command uses the production adapter. With the verified ignored checkpoint
installed it can become ready; without it, startup remains live but unready and
does not activate a fake. Check the running service with:

```powershell
curl.exe -i http://127.0.0.1:8000/health/live
curl.exe -i http://127.0.0.1:8000/health/ready
curl.exe -i http://127.0.0.1:8000/health/model
curl.exe -i http://127.0.0.1:8000/version
curl.exe -i http://127.0.0.1:8000/metrics
```

When the checkpoint is absent, the expected readiness status is HTTP 503 and
restoration returns `model_unavailable`. Tests supply a fake service explicitly
through `create_app(model_service_factory=...)`; production code has no fake
fallback.

## Runtime settings

`RuntimeSettings` reads case-insensitive environment variables with the
`SEMIRESTORE_` prefix. Values are validated before the application is built.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `SEMIRESTORE_ENVIRONMENT` | `development` | Safe runtime identity used in logs |
| `SEMIRESTORE_HOST` | `127.0.0.1` | Documented bind host |
| `SEMIRESTORE_PORT` | `8000` | Documented bind port |
| `SEMIRESTORE_LOG_LEVEL` | `INFO` | Application logger threshold |
| `SEMIRESTORE_JSON_LOGGING` | `true` | JSON or local text log format |
| `SEMIRESTORE_MAX_ENCODED_UPLOAD_BYTES` | `10485760` | Encoded upload bound |
| `SEMIRESTORE_MAX_DECODED_IMAGE_WIDTH` | `16384` | Decoded width bound |
| `SEMIRESTORE_MAX_DECODED_IMAGE_HEIGHT` | `16384` | Decoded height bound |
| `SEMIRESTORE_MAX_DECODED_PIXEL_COUNT` | `100000000` | Decoded pixel bound |
| `SEMIRESTORE_ALLOWED_MEDIA_TYPES` | PNG, JPEG, TIFF | Declared media allow-list |
| `SEMIRESTORE_INFERENCE_CONCURRENCY_LIMIT` | `1` | Process-local capacity |
| `SEMIRESTORE_CONCURRENCY_ACQUISITION_TIMEOUT_SECONDS` | `1.0` | Capacity wait bound |
| `SEMIRESTORE_INFERENCE_TIMEOUT_SECONDS` | `120.0` | Adapter call wait bound |
| `SEMIRESTORE_MODEL_CONFIG_PATH` | unset | Optional resolved model YAML override |
| `SEMIRESTORE_MODEL_METADATA_PATH` | unset | Optional trusted checkpoint manifest override |
| `SEMIRESTORE_CHECKPOINT_PATH` | unset | Optional ignored runtime checkpoint override |
| `SEMIRESTORE_DEVICE_PREFERENCE` | `auto` | `auto`, `cpu`, or `cuda` preference |
| `SEMIRESTORE_ENABLE_FAKE_MODEL_SERVICE` | `false` | Reserved; never enables production fake behavior |

Uvicorn command-line host and port must agree with the environment used by an
operator. The production adapter verifies configuration and checkpoint inputs
during startup. Do not put secrets in these variables or commit environment
files containing secrets. The complete authoritative table, including Compose
and dashboard settings, is [environment.md](environment.md).

## Request IDs

Every HTTP response receives `X-Request-ID`. A caller-provided value is retained
only when it is one unambiguous 1–64 character ASCII identifier beginning with a
letter or digit and otherwise containing letters, digits, `.`, `_`, `:`, or `-`.
Invalid, duplicate, missing, or non-ASCII values are replaced with an opaque
generated UUID. Error envelopes carry the same validated value.

Request IDs aid correlation but are not authentication tokens. They are logged,
but never used as Prometheus labels.

## Structured logging

The isolated `semirestore` logger emits JSON by default and a safe text format
for local development. HTTP completion events contain only a fixed field set:
environment, request ID, method, resolved route template, status/status class,
monotonic duration, and stable error code. Inference events add a bounded
outcome, readiness category, and platform-observed duration.

HTTP 2xx/3xx logs at INFO, 4xx at WARNING, and 5xx at ERROR. The logger never
formats arbitrary exception objects or request/response content. Cancellation
propagates and receives an explicit cancellation event rather than success or
`internal_error`.

## Metrics

`GET /metrics` returns the application-local Prometheus registry with
`Cache-Control: no-store`. It remains available while the model is unready and
is excluded from OpenAPI and ordinary HTTP metrics to prevent scrape self-noise.

Collectors cover HTTP count/duration, restoration outcomes, inference
orchestration duration, active/waiting/capacity gauges, busy rejections, and
execution timeouts. Labels are limited to bounded methods, route templates,
status classes, and outcomes. They never contain request IDs, raw paths, query
strings, filenames, media metadata, model identity, exception text, or content.

No readiness metric is exposed because current readiness is obtained on demand;
the application has no update hook that can guarantee an accurate continuous
gauge without stale or misleading values. Alert against `/health/ready` and the
bounded failure metrics instead.

## Concurrency and backpressure

One process-local `InferenceGate` is created at startup. A validated request
waits up to `concurrency_acquisition_timeout_seconds` for one of
`inference_concurrency_limit` slots. When capacity stays full, the request gets
HTTP 503 `inference_busy`. Health and upload-validation requests remain outside
the held inference slot.

The gate releases capacity after success, known failure, unexpected failure,
timeout, or cancellation. Capacity is not distributed across processes, so a
worker count greater than one also creates multiple services and independently
multiplies effective concurrency and model memory.

## Timeouts and cancellation

After acquiring capacity, the platform waits at most
`inference_timeout_seconds` for the adapter coroutine. Expiry cancels the task,
returns HTTP 504 `inference_timeout`, records a bounded timeout outcome, and
releases the slot.

Cancellation is cooperative. It stops the async wait, but synchronous CPU work
offloaded to a thread and submitted GPU kernels may continue after their caller
has timed out. The real adapter must document and test that behavior; operators
must size concurrency and timeout values so abandoned underlying work cannot
overload the process.
