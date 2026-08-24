# SemiRestore

SemiRestore is an AI-assisted service for restoring degraded, single-channel
semiconductor scanning electron microscope (SEM) images. It combines a
checksum-gated statistics-conditioned NAF-SR pipeline, a FastAPI service, an
operational dashboard, Prometheus metrics, and secure CPU container packaging.

Restored images are estimates, not ground truth. Diagnostics and suitability
recommendations are advisory measurements and heuristics, not probabilities or
proof of restoration correctness.

## Quick start

Python 3.11 or newer is required. This checkout uses an ignored Python 3.13
virtual environment:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[platform,dev]"
.\.venv\Scripts\python.exe scripts/model/install_local_checkpoint.py
.\.venv\Scripts\python.exe -m uvicorn semirestore.api:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

The installer verifies the source bytes against the tracked manifest before
atomically creating the ignored runtime checkpoint. If the immutable default
source is not present, supply `--source path/to/best.pt`. Without the verified
checkpoint, the API remains live but returns HTTP 503 from readiness; it never
falls back to fake restoration.

In another terminal, run the generated-image smoke sequence:

```powershell
.\.venv\Scripts\python.exe scripts/platform/smoke_test.py --operation restore-and-analyze
```

The command checks liveness, readiness, model health, multipart upload, typed
response fields, and the decoded PNG. It emits metadata only. The complete
instructions and expected failures are in the
[smoke-test guide](docs/platform/smoke-testing.md).

## Architecture and operation

One application process owns one verified model adapter, one pipeline, one
inference gate, and one metrics registry. Requests flow through bounded
in-memory upload validation, readiness/admission control, the model-owned
pipeline, and strict response serialization. Uploads and restored images are
not permanently stored by the platform.

- [Platform handoff and documentation index](docs/platform/README.md)
- [Architecture and data flow](docs/platform/architecture.md)
- [Deployment guide](docs/platform/deployment.md)
- [Operations runbook](docs/platform/runbook.md)
- [Environment-variable reference](docs/platform/environment.md)
- [Troubleshooting](docs/platform/troubleshooting.md)
- [API contract](docs/platform/api-contract.md)
- [Model documentation](docs/model/architecture.md)

The dashboard is developed from `web/`. With the API running, use:

```powershell
Set-Location web
npm ci
npm run dev
```

The dashboard accepts PNG, JPEG, and single-frame TIFF images, supports analysis
and restoration workflows, compares the original with the exact returned PNG,
and displays only diagnostics and provenance supplied by the API. See
[dashboard usage](docs/platform/local-stack.md#dashboard-usage).

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
Set-Location web
npm ci
npm run lint
npm test -- --run
npm run build
npm audit
```

Checkpoint-dependent tests skip explicitly when the ignored checkpoint is
absent. CI requires no checkpoint and performs static Docker and Compose policy
checks; an actual image build, container run, CUDA validation, and real
checkpoint smoke test are separate hardware-dependent release gates.

## Repository boundaries

Model behavior, training, evaluation, architecture, preprocessing, and
checkpoint compatibility live under `src/semirestore/`, `configs/model/`,
`scripts/model/`, `tests/model/`, and `docs/model/`. Platform/API code lives
under `src/semirestore/platform/` and `src/semirestore/api/`; the dashboard is
under `web/`.

Checkpoint binaries, `local/artifacts/`, uploads, generated restorations,
reports, caches, secrets, `node_modules`, and frontend build output must not be
committed. The immutable historical experiment export under `local/artifacts/`
must never be edited.
