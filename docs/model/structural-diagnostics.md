# Structural suitability analysis

Structural diagnostics operate on one normalized `[0,1]` grayscale image at
its current resolution. They report reflected 4-neighbor Laplacian variance,
Sobel magnitude and energy, edge density at magnitude 0.10, residual energy
from a reflected 3×3 binomial Gaussian low-pass, and an approximate noise sigma
from the valid-support Immerkaer kernel. Kernel scaling and boundary behavior
are returned with every result.

The versioned advisory rules return `bypass` for essentially flat content,
`warn` for high approximate noise or texture/noise ambiguity, and `restore` for
possible blur or when no caution threshold fires. Every recommendation includes
the exact triggered rule and a reason. It is not a calibrated probability,
accuracy estimate, or assertion that restoration will preserve defects.

All measurements are resolution-sensitive. High-frequency energy may be real
SEM detail, noise, or both, and the noise estimate can be biased upward by edges
and texture. Comparisons across acquisition magnification, pixel pitch, or
resampling scale require explicit qualification.
