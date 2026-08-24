# Teammate integration checklist

## Model-owned prerequisites

Before wiring a production adapter, the model branch must provide and test:

- [ ] `semirestore.pipeline.SemiRestorePipeline` with stable
  `from_config(...)` and `restore_and_analyze(image)` contracts.
- [ ] A versioned configuration schema and an explicit compatibility contract
  among code, configuration, and checkpoint.
- [ ] A real checkpoint acquisition and integrity process; no checkpoint belongs
  in Git, the wheel, or the container image.
- [ ] Deterministic checkpoint loading with an agreed CPU map-location and device
  selection/fallback policy.
- [ ] A stable model-version identifier and checksum policy for safe provenance.
- [ ] The exact accepted in-memory image representation, color/channel/range
  semantics, and model-owned preprocessing rules.
- [ ] A result contract identifying restored pixels, scientific diagnostics,
  suitability warnings, and model-measured latency.
- [ ] JSON-safe diagnostic projection and public-safe warning vocabulary.
- [ ] Expected initialization, availability, and inference failure categories.
- [ ] Scientific validation showing what diagnostics mean; neither diagnostics
  nor restored output should be presented as ground truth.

## Adapter integration

After those prerequisites land:

- [ ] Implement the real adapter exactly as specified in
  [model-adapter-contract.md](model-adapter-contract.md).
- [ ] Load and verify the checkpoint once in adapter `startup()` and release it
  in `shutdown()`.
- [ ] Map `ValidatedUpload` to the pipeline input without persistence or a second
  transport policy.
- [ ] Map model output to fully validated `RestorationResult` bytes and metadata.
- [ ] Translate known failures to the three model-service exception categories
  and propagate cancellation.
- [ ] Verify CPU fallback and report the actual device, model version, and exact
  checkpoint checksum.
- [ ] Test thread/native/GPU behavior after timeout and cancellation; tune the
  application gate against real resource use.
- [ ] Wire the production application factory to the real adapter. The current
  no-argument `create_app` must remain unready until that explicit wiring exists;
  never bridge the gap with a fake.
- [ ] Keep the Docker/Uvicorn factory target consistent if integration introduces
  a separate production factory.

## Integration verification

- [ ] Run all platform unit and end-to-end tests with Python 3.11 or newer.
- [ ] Add real-adapter tests for ready startup, missing/corrupt/incompatible
  checkpoints, CPU inference, output serialization, failure translation,
  shutdown, timeout, and repeated request reuse.
- [ ] Confirm live-but-unready behavior with a missing checkpoint and ready
  behavior only with the verified compatible checkpoint.
- [ ] Confirm responses, logs, and metrics contain no paths, raw exceptions,
  images, tensors, secrets, or unbounded labels.
- [ ] Confirm uploads and outputs are not persisted and immutable artifacts are
  not tracked by Git.
- [ ] Re-run Ruff and `git diff --check`.
- [ ] On a Docker-capable machine, run the documented image build and container
  smoke test. This was not performed on the platform workstation.
- [ ] Exercise liveness, readiness, model health, metrics, successful restoration,
  SIGTERM shutdown, non-root execution, read-only checkpoint mounting, and the
  one-worker policy in the built container.
- [ ] Review authentication, authorization, network exposure, and rate limiting
  before exposing the API outside a trusted environment; none are implemented.
