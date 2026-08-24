# Production handoff checklist

## Integrated and covered by repository tests

- [x] The production application factory constructs one real
  `SemiRestoreModelService` per process lifespan.
- [x] The adapter loads the model-owned `SemiRestorePipeline` once, reuses it,
  exposes safe cached health, and closes it on shutdown.
- [x] Checkpoint existence, regular-file type, trusted size/SHA-256,
  architecture compatibility, and safe `weights_only=True` loading are enforced.
- [x] Missing, invalid, or incompatible checkpoints leave the API live but
  unready without a fake fallback.
- [x] PNG, JPEG, and single-frame TIFF uploads receive bounded in-memory format,
  decode, and dimension validation before inference admission.
- [x] Analyze, restore, and restore-and-analyze operations use the typed model
  boundary and strict safe response schemas.
- [x] Concurrency, backpressure, timeout, cancellation, request ID, structured
  logging, metrics, and error behavior have focused platform tests.
- [x] Dashboard upload, cancellation, comparison, exact PNG download,
  diagnostics, warnings, readiness, and safe-error flows have frontend tests.
- [x] CI runs the complete checkpoint-free backend and frontend validation and
  rejects tracked checkpoints/artifacts/generated frontend output.
- [x] The generated-image load and smoke harnesses contain no user image data or
  response content in their reports.

These checks establish software-contract behavior. They do not establish
restoration quality, scientific suitability, production capacity, or hardware
performance.

## Required release-environment verification

- [ ] Install or read-only mount the exact checkpoint named by the trusted
  manifest; confirm its public checksum through `/health/model` without
  exposing its path.
- [ ] Run the optional `local_checkpoint` tests on CPU with the verified ignored
  checkpoint and review all non-checkpoint tests and expected skips.
- [ ] Build the Docker image on a Docker-capable machine and verify its non-root
  identity, one-worker command, health check, read-only mount, and signal flow.
- [ ] Start the container without a checkpoint and confirm live/unready behavior,
  then start it with the verified checkpoint and require readiness.
- [ ] Run `scripts/platform/smoke_test.py --operation restore-and-analyze`
  against the deployed service and inspect the safe metadata-only report.
- [ ] Exercise liveness, readiness, model health, version, metrics, analyze,
  restore, and restore-and-analyze through the actual deployment proxy.
- [ ] Run representative CPU load/resilience tests and record hardware, image
  digest, package/model/checkpoint identities, inputs, worker count, and settings.
- [ ] Verify SIGTERM traffic drain and shutdown within the documented 35-second
  Compose grace period, including the deployment orchestrator's behavior.
- [ ] Confirm logs, responses, dashboards, metrics, and temporary files contain
  no images, Base64 payloads, filenames, checkpoint paths, secrets, exception
  text, tensors, or unbounded labels.
- [ ] Validate backup/rollback image and checkpoint compatibility and execute the
  [runbook rollback](runbook.md#rollback) before production exposure.
- [ ] Supply TLS, authentication, authorization, network access policy, rate
  limiting, vulnerability management, and retention rules outside the service.

## CUDA-specific optional gates

- [ ] Build a separately reviewed CUDA-compatible image; the committed image is
  CPU-only.
- [ ] Verify NVIDIA host driver, container runtime, CUDA, and PyTorch compatibility.
- [ ] Confirm actual device reporting, GPU synchronization/timing, memory use,
  cancellation behavior, and one-model-per-worker ownership.
- [ ] Re-run real-checkpoint smoke and representative load tests without
  comparing their results directly to a different CPU environment.

CUDA and real container execution were not performed on the workstation used
for this handoff. No benchmark result is claimed.
