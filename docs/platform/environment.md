# Environment-variable reference

`RuntimeSettings` reads case-insensitive variables with the `SEMIRESTORE_`
prefix. Invalid settings prevent application construction. Do not put secrets
in these values or commit `.env` files.

## API runtime

| Variable | Default | Validation and effect |
| --- | --- | --- |
| `SEMIRESTORE_ENVIRONMENT` | `development` | Non-blank, at most 64 characters; safe log identity |
| `SEMIRESTORE_HOST` | `127.0.0.1` | Non-blank documented bind host; Uvicorn arguments remain authoritative |
| `SEMIRESTORE_PORT` | `8000` | Integer 1-65535; Uvicorn arguments remain authoritative |
| `SEMIRESTORE_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL` |
| `SEMIRESTORE_JSON_LOGGING` | `true` | JSON when true, local text when false |
| `SEMIRESTORE_MAX_ENCODED_UPLOAD_BYTES` | `10485760` | Positive encoded multipart image limit |
| `SEMIRESTORE_MAX_DECODED_IMAGE_WIDTH` | `16384` | Positive decoded width limit |
| `SEMIRESTORE_MAX_DECODED_IMAGE_HEIGHT` | `16384` | Positive decoded height limit |
| `SEMIRESTORE_MAX_DECODED_PIXEL_COUNT` | `100000000` | Positive decoded width times height limit |
| `SEMIRESTORE_ALLOWED_MEDIA_TYPES` | PNG, JPEG, TIFF | Non-empty unique JSON array of supported MIME types |
| `SEMIRESTORE_INFERENCE_CONCURRENCY_LIMIT` | `1` | Positive process-local inference capacity |
| `SEMIRESTORE_CONCURRENCY_ACQUISITION_TIMEOUT_SECONDS` | `1.0` | Finite positive wait for capacity |
| `SEMIRESTORE_INFERENCE_TIMEOUT_SECONDS` | `120.0` | Finite positive adapter wait |
| `SEMIRESTORE_MODEL_CONFIG_PATH` | model default | Resolved model YAML supplied to the production adapter |
| `SEMIRESTORE_MODEL_METADATA_PATH` | model default | Trusted checkpoint manifest supplied to the adapter |
| `SEMIRESTORE_CHECKPOINT_PATH` | model default | Ignored verified runtime checkpoint supplied to the adapter |
| `SEMIRESTORE_DEVICE_PREFERENCE` | `auto` | `auto`, `cpu`, or `cuda`; the CPU container sets `cpu` |
| `SEMIRESTORE_ENABLE_FAKE_MODEL_SERVICE` | `false` | Reserved and rejected when true; never activates a production fake |

Pydantic settings expects complex values such as the media allow-list as JSON,
for example `SEMIRESTORE_ALLOWED_MEDIA_TYPES=["image/png","image/tiff"]` in an
environment mechanism that preserves the quotes.

Leaving model paths unset delegates to the tracked defaults:
`configs/model/resolved_conditioned.yaml`, `artifacts/model/checksums.json`, and
the ignored `artifacts/model/semirestore_conditioned.pt`. Alternate files must
remain compatible and should be mounted read-only. Public health never exposes
their filesystem paths.

## Compose and dashboard variables

`.env.example` contains only safe local defaults. Compose additionally reads:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SEMIRESTORE_CHECKPOINT_HOST_PATH` | `./artifacts/model/semirestore_conditioned.pt` | Host file mounted read-only at `/models/semirestore_conditioned.pt` |
| `SEMIRESTORE_API_PORT` | `8000` | Published API port |
| `SEMIRESTORE_DASHBOARD_PORT` | `5173` | Published dashboard port |
| `SEMIRESTORE_PROMETHEUS_PORT` | `9090` | Published Prometheus port |
| `SEMIRESTORE_DASHBOARD_API_BASE_URL` | `/service` | Dashboard build-time same-origin API prefix |
| `SEMIRESTORE_DEV_API_URL` | `http://127.0.0.1:8000` | Vite development proxy target; not used by Compose |
| `SEMIRESTORE_API_CPUS` / `SEMIRESTORE_API_MEMORY` | `2.0` / `4g` | API container limits |
| `SEMIRESTORE_DASHBOARD_CPUS` / `SEMIRESTORE_DASHBOARD_MEMORY` | `0.5` / `128m` | Dashboard limits |
| `SEMIRESTORE_PROMETHEUS_CPUS` / `SEMIRESTORE_PROMETHEUS_MEMORY` | `0.5` / `256m` | Prometheus limits |
| `SEMIRESTORE_PROMETHEUS_RETENTION` | `24h` | Local Prometheus retention |

Compose pins production fake behavior to false and supplies read-only in-image
model configuration/metadata paths. Resource values are starting points, not
measured capacity recommendations. Validate them with the real checkpoint and
representative images before production use.
