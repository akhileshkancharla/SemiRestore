# Local Compose stack

## CPU startup

Docker Compose is optional; the API can still run directly with the documented
Python/Uvicorn command. For the container stack, copy the safe template and set
the checkpoint source if it differs from the ignored default location:

```sh
cp .env.example .env
docker compose up --build api
```

The API mounts `SEMIRESTORE_CHECKPOINT_HOST_PATH` read-only at
`/models/semirestore_conditioned.pt`. The default host path is
`./artifacts/model/semirestore_conditioned.pt`, which is ignored by Git. The
tracked configuration and checksum manifest remain inside the image. A missing,
wrong, or incompatible checkpoint leaves the container live but makes the
Compose readiness check fail; no fake restoration is activated.

API-only operation does not require any profile. Optional services are enabled
deliberately:

```sh
docker compose --profile dashboard up --build
docker compose --profile observability up --build api prometheus
docker compose --profile dashboard --profile observability up --build
```

The dashboard profile builds the React/Vite inspection console and serves its
static assets from an unprivileged Nginx runtime on
`http://localhost:${SEMIRESTORE_DASHBOARD_PORT:-5173}`. It waits for API
readiness, then proxies browser requests from `/service` to the API over the
Compose `frontend` network. `SEMIRESTORE_DASHBOARD_API_BASE_URL` is a build-time
URL for deployments that need another same-origin prefix; `/service` is the
safe local default. API-only operation remains independent of the dashboard.

Prometheus starts when the API process starts because `/metrics` remains useful
while the model is unready; its scrape configuration uses only the private
`backend` network.

Resource defaults are intentionally conservative and configurable through the
safe `.env.example` keys. The API retains one worker and one process-local model
instance. The checkpoint mount, Prometheus configuration mount, and model image
inputs are read-only. No secret or machine-specific absolute path belongs in
Compose or the example environment file.

## Dashboard usage

Open the dashboard after its API status shows ready. Select or drop one PNG,
JPEG, or single-frame TIFF. The browser validates basic type/dimensions and
creates a temporary local preview; server validation remains authoritative.
Choose Analyze, Restore, or Restore and analyze. A request can be cancelled, a
recoverable failure retains the selection, and a duplicate submission is
blocked.

Successful restoration shows original and exact returned PNG views, a direct
comparison slider, synchronized zoom/pan, dimensions, download, diagnostics,
suitability guidance, warnings, model provenance, and returned timings. The UI
labels display scaling and applies no sharpening or enhancement filters.
Suitability is advisory and not a probability. Missing fields remain explicitly
unavailable; the UI does not invent values.

Replacing an image or result and unmounting the workspace revoke browser object
URLs. Neither the dashboard nor API permanently stores uploads or restorations.

## Dashboard development

The frontend requires Node.js 24 or newer. During direct development, Vite
proxies `/service` to `http://127.0.0.1:8000`; override that development target
with `SEMIRESTORE_DEV_API_URL` when needed:

```sh
cd web
npm install
npm run dev
```

Run `npm run lint`, `npm run test:run`, and `npm run build` before packaging.
The dashboard includes the upload, analysis, restoration, comparison,
diagnostics, readiness, cancellation, safe-error, and exact-PNG download flows.

## Health and shutdown

The image-level health check uses `/health/live`. Compose uses `/health/ready`
for model-dependent service ordering, while Prometheus depends only on process
startup. Uvicorn receives container signals directly through its exec-form
command, has a 30-second graceful-shutdown timeout, and Compose allows 35
seconds before forced termination.

## Optional CUDA deployment

The committed image is CPU-only. Do not enable CUDA merely by changing
`SEMIRESTORE_DEVICE_PREFERENCE`; that image has no CUDA PyTorch runtime. An
optional GPU deployment requires a separately reviewed CUDA-compatible image,
NVIDIA Container Toolkit on the host, a compatible driver/runtime/PyTorch
combination, and explicit device reservations in a local override file.

For such a validated image, an uncommitted `compose.cuda.yaml` may override the
API image, set `SEMIRESTORE_DEVICE_PREFERENCE=cuda`, and request one GPU. Keep
one API worker, revalidate GPU memory against the real checkpoint and inference
limit, and test timeout/cancellation behavior. No CUDA build or Compose run was
performed on this workstation.
