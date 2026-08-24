# SemiRestore

SemiRestore is an AI-assisted restoration system for degraded, single-channel
semiconductor scanning electron microscope (SEM) images. This repository is
being assembled as a clean engineering home for a previously trained,
statistics-conditioned NAF-SR model and the diagnostics that surround it.

## Model-engineering workspace

The model track owns these paths:

- `src/semirestore/`
- `configs/model/`
- `scripts/model/`
- `artifacts/model/`
- `reports/model/`
- `tests/model/`
- `docs/model/`
- model dependency groups in `pyproject.toml`

Platform, API, frontend, observability, deployment, and workflow code are owned
by the platform track. Shared, implementation-independent interfaces belong in
`contracts/` when they become necessary.

The historical experiment export under `local/artifacts/` is immutable,
ignored source material. It must never be edited, staged, or committed.
Deployable checkpoints are installed separately into `artifacts/model/` and
also remain outside normal Git history.

## Development

SemiRestore uses a `src` package layout and requires Python 3.11 or newer.
After installing the model development dependencies, run the model tests with:

```console
python -m pytest tests/model
```

Checkpoint provenance, installation, inference contracts, and limitations will
be documented alongside the corresponding tested implementation milestones.
