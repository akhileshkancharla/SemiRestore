# Conditioned restoration trainer

The Milestone 19 trainer provides an explicit, bounded core loop for the
statistics-conditioned NAF-SR architecture. Construction validates the frozen
checkpoint-compatible model configuration. Training uses the authoritative
AdamW settings, mean Charbonnier loss, linear warmup, cosine decay, gradient
clipping, paired manifest data, deterministic D4 transforms, and optional
sample-seeded raw degradation. The default synthetic probability remains the
authoritative value of zero.

Validation is inference-only, never updates parameters, and computes paired HR
loss, PSNR, and SSIM using an explicit range policy. Step, epoch, learning-rate,
loss, sample, and validation summaries contain only serialization-friendly
values. Dependency injection permits controlled small-model tests without
starting a 9.1-million-parameter training run.

No work begins at import or construction time. Callers must explicitly invoke
`train_step`, `validate`, or bounded `fit`. This milestone deliberately has no
AMP, EMA, resume workflow, checkpoint serialization, or checkpoint rotation;
those resilience features belong to Milestone 20.
