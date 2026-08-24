# Resilient mixed-precision training

CUDA training uses the current `torch.autocast` and `torch.amp.GradScaler` APIs
when `amp_enabled` is configured. CPU training always remains FP32. Gradients
are unscaled before configurable norm clipping; non-finite loss, prediction,
gradient norm, or validation output stops the step before an optimizer update.

The historical EMA policy is available with decay 0.999. Raw and EMA weights
are both evaluated, and the configured PSNR, SSIM, or loss policy chooses the
validation candidate. `best.pt` contains only the selected model state and
primitive provenance. `last.pt` and bounded `last-step-N.pt` archives are
training-resume artifacts containing model, optimizer, scheduler, scaler, EMA,
step/epoch position, best-selection state, and deterministic RNG state.

Writes use same-directory partial files and atomic replacement. Rotation only
recognizes strict `last-step-N.pt` names inside the configured checkpoint
directory and retains the requested count. Resume loading uses
`weights_only=True`, validates format role plus configuration and model
fingerprints, and never weakens the separately checksum-gated deployment
loader. Interrupted temporary files are cleaned after failure.
