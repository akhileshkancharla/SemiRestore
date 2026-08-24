# Continuous integration

## Automated validation

GitHub Actions runs with repository contents read-only and cancels superseded
runs for the same branch or pull request. The backend/platform job installs the
CPU PyTorch runtime and the project development/platform dependencies, runs
Ruff and the complete pytest suite, repeats the focused container, Compose,
dashboard and CI policy checks, validates `compose.yaml`, and rejects tracked
checkpoint files, `local/artifacts/`, or generated frontend output.

The frontend job uses `npm ci` with the committed lockfile, then runs ESLint,
Vitest, the TypeScript/Vite production build, and `npm audit` with a high-severity
failure threshold. Dependency caches are keyed from the Python project metadata
and frontend lockfile. CI does not download a checkpoint, use repository
secrets, upload user images, or publish build artifacts.

Checkpoint-dependent tests use the existing explicit marker and skip when the
ignored verified checkpoint is absent. The production application remains live
but unready when no checkpoint is supplied; CI never substitutes a fake model.

## Hardware-dependent validation

CI validates the CPU code path and static container policy. It does not establish
restoration quality, production throughput, GPU memory requirements, CUDA
compatibility, or real-checkpoint readiness. Those claims require the verified
checkpoint and representative deployment hardware.

The workflow validates Compose syntax when Docker tooling is available on the
runner. A Docker image build and container execution remain separate release
gates. CUDA deployment requires a separately reviewed GPU image and an NVIDIA
runtime; the committed image remains CPU-only.
