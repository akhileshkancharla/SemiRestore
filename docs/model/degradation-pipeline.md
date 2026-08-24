# Reproducible SEM degradation

The degradation pipeline is an adapted migration of the historical training
utility. It applies reflective Gaussian blur, additive Gaussian noise and bias,
multiplicative Gaussian speckle, and exact 2× area or antialiased bicubic
downsampling. The configured operation order is either explicit or a uniformly
sampled permutation. Outputs are never clipped.

Every invocation derives a private seed from the base seed, epoch, stable sample
identifier, and degradation-version string. Parameter sampling, operation order,
and noise use only a local generator. Results therefore do not depend on
DataLoader worker count, batch composition, or traversal order, and the caller's
HR tensor is never mutated.

The result metadata records all sampled strengths, the selected interpolation,
the realized operation order, seed inputs, boundary/antialias behavior, shapes,
and clipping policy. Display/export clipping and deployment preprocessing are
separate boundaries and must not be applied to raw degraded training arrays.

The authoritative conditioned run configured `synthetic_probability: 0.0`; its
15% synthetic-degradation ablation was rejected. This module preserves the
historical controlled-degradation behavior for reproducible experiments, but it
does not claim that synthetic degradation produced the deployed checkpoint.
