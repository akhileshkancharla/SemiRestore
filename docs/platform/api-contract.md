# API contract

## Endpoints

| Method and path | Success contract | Purpose |
| --- | --- | --- |
| `GET /health/live` | HTTP 200 `LiveResponse` | Process liveness only |
| `GET /health/ready` | HTTP 200 or 503 `ReadyResponse` | Ability to accept restoration work |
| `GET /health/model` | HTTP 200 `ModelHealthResponse` | Safe model readiness and identity metadata |
| `GET /version` | HTTP 200 `VersionResponse` | Application name and package version |
| `POST /api/v1/restore` | HTTP 200 `RestoreResponse` | Validate and restore one multipart image |
| `GET /metrics` | Prometheus text | Operational metrics; intentionally excluded from OpenAPI |

OpenAPI documents the health and version operations, the multipart restoration
body, its typed success response, and the stable error envelope for every
documented restoration failure status.

## Health and readiness

Liveness never consults model readiness. A process can be live while its model
is starting, missing, failed, or stopped.

Readiness is true only when `ModelHealth.state` is `ready` and the adapter says
restoration work can be accepted. An unready response uses HTTP 503 and a safe
reason. `/health/model` always uses HTTP 200 so operators can inspect the safe
state; it may expose only `ready`, `state`, `device`, `model_version`,
`checkpoint_checksum`, and `unavailable_reason`.

An unconfigured application reports `unavailable`, does not claim a device,
model version, or checkpoint checksum, and rejects restoration with
`model_unavailable`. A configured path alone does not imply that the path
exists, that a checkpoint was verified, or that the service is ready.

## Restoration contract

`POST /api/v1/restore` consumes `multipart/form-data` with exactly one supported
field:

- field name: `image`
- accepted declared and detected formats: PNG (`image/png`), JPEG
  (`image/jpeg`), and single-frame TIFF (`image/tiff`)

The JSON success body is:

```json
{
  "image": {
    "encoding": "base64",
    "media_type": "image/png",
    "content": "<base64>",
    "width": 1024,
    "height": 768
  },
  "input": {
    "width": 1024,
    "height": 768,
    "media_type": "image/tiff"
  },
  "inference": {
    "latency_ms": 125.0,
    "device": "cpu"
  },
  "model": {
    "version": "<safe-public-version>",
    "checkpoint_checksum": "sha256:<digest>"
  },
  "diagnostics": {},
  "warnings": []
}
```

The output media type is independent of the input media type. Dimensions must
describe the corresponding encoded image and input. Optional inference and
model fields remain explicit `null` values when the adapter cannot provide
them. Diagnostics must be JSON-compatible; warnings must be safe public text.

## Base64 transport

Keeping bytes and metadata in one JSON response simplifies a typed first API,
but Base64 adds approximately one-third size overhead before JSON framing. The
current endpoint is appropriate for bounded requests, not bulk result transfer.
A future binary or object-storage transport would require a separately versioned
contract and privacy review.

## Upload security

Validation is in-memory and bounded by:

- `max_encoded_upload_bytes` while reading the stream;
- `max_decoded_image_width` and `max_decoded_image_height`;
- `max_decoded_pixel_count` to limit decompression amplification;
- the configured allow-list intersected with PNG, JPEG, and TIFF support;
- format detection matching the declared media type;
- complete image decoding and a single-frame requirement.

Empty, oversized, unsupported, malformed, mismatched, multi-frame, or excessive
images are rejected before the adapter and inference gate are called. Safe error
details may report configured limits or the supported media types; they never
echo uploaded content or filenames.

## Error contract

Every API error has one stable shape. `request_id` is present as a string after
HTTP middleware runs and remains nullable in the reusable schema for contexts
where no request ID is available.

```json
{
  "error": {
    "code": "model_unavailable",
    "message": "The model service is unavailable.",
    "details": null,
    "request_id": "opaque-correlation-id"
  }
}
```

| Code | HTTP status | Meaning |
| --- | ---: | --- |
| `invalid_request` | 400, 404, 405, or 422 | Malformed transport, route/method, or request validation |
| `empty_upload` | 400 | No image bytes were supplied |
| `unsupported_media_type` | 415 | Declared/detected format is unsupported or mismatched |
| `upload_too_large` | 413 | Encoded byte limit was exceeded |
| `invalid_image` | 422 | Image structure or decoding is invalid |
| `image_dimensions_exceeded` | 413 | Width, height, or pixel limit was exceeded |
| `model_unavailable` | 503 | No ready service can accept work |
| `inference_busy` | 503 | Capacity was not acquired in time |
| `inference_timeout` | 504 | Execution exceeded its configured timeout |
| `restoration_failed` | 500 | A known model inference failure occurred |
| `internal_error` | 500 | An unexpected failure occurred |

Known platform and model-service exceptions map without reading their exception
messages. FastAPI validation exposes only issue location and stable type, never
the rejected value or internal validation context. Unexpected exceptions become
generic `internal_error`; raw messages, paths, and stack traces are suppressed.
