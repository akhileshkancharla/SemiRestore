# Restoration losses and reference metrics

The authoritative conditioned run used only mean Charbonnier loss,
`sqrt(error² + epsilon²)`, with `epsilon=0.001`. Loss computation operates on
raw training-domain floats and never performs display clipping.

PSNR and SSIM are full-reference metrics: both require aligned ground-truth HR
imagery and must not be reported as production confidence for an unpaired SEM
input. Callers must provide `data_range`; the library never infers it from an
image. The default range policy rejects out-of-range values. Historical
evaluation behavior is available only through the explicit `range_policy="clip"`
choice, which clamps prediction and target at the scoring boundary.

PSNR reports positive infinity for exact reconstruction. SSIM preserves the
historical evaluation policy: an 11×11 Gaussian window, sigma 1.5, constants
K1=0.01 and K2=0.03, and population (not sample) covariance. Both functions
return one value per image. `compute_reference_metrics` adds deterministic
per-image records and aggregate means; its `as_dict()` form encodes infinite
PSNR as the string `"Infinity"` so strict JSON serialization remains possible.
