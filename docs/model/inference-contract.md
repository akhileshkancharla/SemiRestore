# Final model inference contract

`SemiRestorePipeline.from_config(...)` verifies and loads the installed
checkpoint once. `restore_and_analyze(image, mode="direct"|"tiled")` accepts
the same in-memory image inputs as model preprocessing. It does not persist the
input or create tile files.

The pipeline returns `semirestore.pipeline.RestorationResult` containing the
normalized restored array, lossless PNG bytes, original/restored dimensions,
input and restored diagnostics, advisory suitability decision and reasons,
spatial/tile and clipping metadata, no-reference assurance indicators, model
version/checksum/device, phase timings, warnings, and limitations.
`to_dict()` is strict JSON-compatible and embeds PNG bytes as base64;
`metadata()` omits the bytes. `platform_projection()` returns the exact field
mapping needed by the platform-owned result without importing platform code.

All successful outputs are grayscale PNG with media type `image/png` and exact
2× dimensions. Quality indicators confirm mechanical invariants and describe
input/output changes; they do not claim accuracy. A structural `bypass`
recommendation remains advisory, so an explicit pipeline call still restores
and records a warning.
