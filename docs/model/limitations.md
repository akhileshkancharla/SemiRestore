# Scientific and operational limitations

- Restoration may hallucinate or oversmooth structures, including real defect
  evidence. The source image must remain available for comparison.
- No-reference intensity, structural, clipping, and before/after indicators
  cannot prove that reconstructed pixels are correct.
- Suitability is an explainable threshold ruleset, not confidence, probability,
  accuracy, or an autonomous disposition decision.
- Out-of-domain microscopes, magnification, pixel pitch, dose, noise, contrast,
  sample preparation, or materials may invalidate expected behavior.
- External validation based on synthetic 2× downsampling is weaker than native
  aligned degraded/clean SEM evidence.
- Structural values are resolution-sensitive. High-frequency energy may be
  useful detail, noise, or both.
- Direct and tiled inference can differ because tiles have finite receptive
  field context despite global conditioning and overlap.
- The verified checkpoint must be installed at runtime and is never fetched or
  silently replaced by the pipeline.
- Only lossless grayscale PNG is a supported scientific model output.
- Exactly one model allocation should be created per process. Multiple server
  workers duplicate checkpoint memory.
