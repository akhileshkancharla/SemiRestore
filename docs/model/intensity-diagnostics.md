# Intensity degradation diagnostics

Intensity diagnostics accept one finite normalized `[0,1]` grayscale image and
report mean, population standard deviation, minimum, maximum, observed dynamic
range, robust P95−P05 contrast, 256-bin Shannon entropy, and fractions near the
lower and upper representable boundaries. The saturation thresholds are
`1/255` and `254/255`.

The versioned qualitative rules label constant, dark, bright, low-contrast, or
otherwise nominal intensity profiles. Saturation above 1% adds a warning. These
labels are deterministic heuristics, not learned probabilities. They do not use
a clean HR reference and therefore cannot measure restoration accuracy,
reconstruction correctness, defect preservation, or model confidence. No
composite display score is emitted because no defensible calibration data has
been established.
