# Paired SEM restoration data

`PairedSEMDataset` adapts the historical manifest-backed NumPy loader while
making path, identity, split, and pair validation explicit. Canonical manifests
use `sample_id,lr_path,hr_path,split`; the historical
`stem,input_relpath,target_relpath,split` schema remains accepted.

Every path must be relative to the configured dataset root and resolve inside
it. Sample identifiers and files must be unique across the entire manifest, so
duplicate samples and train/validation/test leakage fail before a split is
selected. Returned samples are ordered deterministically by identifier and
contain `(low_resolution, high_resolution, metadata)`, with tensors shaped
`1xHxW` and `1x2Hx2W`.

## Scientific value policy

Training `.npy` arrays are finite, real, one-channel arrays converted to
contiguous float32. They are not normalized or clipped. Negative degraded
values and values above one are therefore preserved exactly (subject only to
float32 conversion), matching the historical training domain. This loader must
not call deployment preprocessing, whose strict input boundary has different
normalization rules.

Encoded image formats are intentionally not accepted at this boundary. Their
integer scaling and color-mode policies would require a separately documented
scientific decision; callers should provide the authoritative raw `.npy`
arrays instead. No dataset, manifest, or generated sample is tracked by this
implementation.
