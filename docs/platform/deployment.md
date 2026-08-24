# Deployment

## CPU deployment

CPU is the portable deployment target. The Dockerfile uses a two-stage build,
creates a non-editable SemiRestore wheel, derives the Torch requirement from
that wheel's metadata, installs Torch from the CPU index, and installs the one
validated `semirestore-*.whl` with the `platform` extra. The runtime uses the
non-root numeric identity `10001:10001` and one Uvicorn worker.

One process owns one adapter, checkpoint, metrics registry, and concurrency
gate. Keep `--workers 1` unless memory, checkpoint ownership, capacity, and
observability semantics have been deliberately redesigned and tested.

## Docker usage

Docker is unavailable on the workstation used for this handoff. Static
Dockerfile policy tests passed, but no image build and no container execution
occurred. A real build and smoke test remain a pre-merge task on a Docker-capable
machine.

These are the exact intended validation commands; they have not been run here:

```sh
docker build -t semirestore:platform .
docker run --rm -p 8000:8000 semirestore:platform
```

Without the runtime checkpoint mount, expected container behavior is:

- `/health/live` succeeds;
- `/health/ready` returns HTTP 503 unready;
- `/health/model` safely reports unavailable with no model/checkpoint claim;
- `/metrics` remains available;
- restoration returns `model_unavailable`.

After starting the container on a capable machine, exercise those endpoints and
inspect startup/shutdown logs. A successful image build alone is not a service
smoke test.

## Container health check

The OCI `HEALTHCHECK` uses Python's standard library to call
`http://127.0.0.1:8000/health/live`. It deliberately does not call readiness or
inference. Missing model artifacts should make the instance unready for traffic,
not restart an otherwise healthy API process.

Deployments should route traffic only after `/health/ready` succeeds. Liveness
and readiness must remain separate probes.

## Runtime model artifacts

The image does not contain or download checkpoints, uploads, outputs, or private
datasets. It contains only the tracked resolved model configuration and checksum
manifest, and points the integrated adapter to those files with
`SEMIRESTORE_MODEL_CONFIG_PATH` and `SEMIRESTORE_MODEL_METADATA_PATH`. Mount the
verified checkpoint read-only at the configured runtime path:

```sh
docker run --rm -p 8000:8000 \
  --mount type=bind,src=/absolute/host/semirestore_conditioned.pt,dst=/models/semirestore_conditioned.pt,readonly \
  semirestore:platform
```

The production adapter verifies existence, regular-file type, compatibility,
and checksum during startup. A missing or invalid mount leaves the API live but
unready; it never enables synthetic restoration. Alternate read-only paths may
be supplied with all three model environment variables. Secrets should use
deployment-managed secret mechanisms, not Docker `ARG`, image `ENV`, or
committed files.

Where supported, operators should also evaluate a read-only root filesystem, a
small writable `/tmp` tmpfs, dropped capabilities, and `no-new-privileges`.
These runtime controls are not automatically applied by the Dockerfile.

## GPU limitations

GPU packaging is not implemented. A real GPU deployment requires a compatible
NVIDIA runtime, CUDA/PyTorch base-image strategy, driver/library compatibility,
checkpoint validation, memory sizing, and explicit synchronization/cancellation
tests. Exposing a GPU device to the CPU image is not sufficient.

GPU work may continue after the async request is cancelled, and each extra
worker may load another checkpoint into device memory. One worker and a
concurrency limit validated against the actual model remain the safe default.

## Known limitations

- The real adapter is integrated, but readiness still requires the external
  verified checkpoint.
- Checkpoint contents and scientific behavior remain model-owned.
- Docker build and container execution have not been performed locally.
- GPU packaging and runtime validation are not implemented.
- Base64 adds approximately one-third response overhead.
- Cancellation may not immediately stop underlying thread or GPU work.
- A single process/worker is the safe default.
- No authentication or authorization layer exists.
- No persistent result storage exists; images are intentionally non-persistent.
- No distributed queue or cross-process concurrency controller exists.
- No readiness metric is exposed because continuously accurate readiness cannot
  currently be guaranteed.
- Platform diagnostics do not claim model scientific correctness.
- Restored output is an estimate and is not guaranteed ground truth.
