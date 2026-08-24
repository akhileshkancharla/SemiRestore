# SemiRestore platform ownership

The platform track owns the service boundary around the SemiRestore model. Its
code lives in `src/semirestore/platform/` and `src/semirestore/api/`; its tests
live in `tests/platform/` and `tests/api/`. Model architecture, checkpoints,
scientific image processing, inference, training, evaluation, and diagnostics
remain model-track responsibilities.

## Milestone 1 boundary

This milestone provides typed runtime settings and package namespaces only. It
does not create an HTTP application, load a model, inspect a checkpoint, or
perform restoration. Later milestones will consume these settings through a
narrow platform-side model-service adapter.

`RuntimeSettings` reads environment variables prefixed with `SEMIRESTORE_`.
Defaults are development-safe: the service binds to loopback, inference
concurrency is one, and the development fake model service is disabled. Model
configuration and checkpoint paths are opaque optional values that will be
passed to the future adapter; platform settings do not open or validate them.

The settings cover runtime identity and networking, logging, encoded and
decoded upload limits, accepted media types, inference concurrency and
timeouts, model-boundary paths, device preference, and explicit fake-service
enablement. Secrets do not belong in committed configuration.

## Local environment

The project requires Python 3.11 or newer. On a Windows system where `python`
still selects Python 3.10, create the environment explicitly with an installed
supported interpreter, for example:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[platform,dev]"
.\.venv\Scripts\python -m pytest
```

Docker is not required for this milestone. Container packaging is deferred to
Platform Milestone 11.

## Restoration response transport

`POST /api/v1/restore` accepts one multipart field named `image`. The response
keeps the restored image and its metadata together by encoding the image bytes
as Base64 in JSON. Base64 increases the encoded payload size by approximately
one-third; a binary response may be added later if large-image workloads make
that overhead material. Uploaded and restored images are not persisted.
