# SemiRestore conditioned NAF-SR model card

## Model and intended use

SemiRestore uses a statistics-conditioned, one-channel NAF-SR network for 2×
restoration of degraded semiconductor SEM imagery. The deployed model has
9,111,684 parameters and is identified as `conditioned-d037473`. Its verified
checkpoint SHA-256 is
`273abd9d6dcfa9bdee71ac15016994962304b6c9d902898b4f4d503bed158c28`.
The checkpoint must be installed locally; it is intentionally excluded from
Git and container source.

The output is an evidence-preserving visualization aid, not ground truth. A
qualified operator should compare restoration with the source image and the
acquisition context before drawing scientific or manufacturing conclusions.

## Training and evaluation provenance

Architecture and weights derive from historical training revision
`d037473ddf4a3cd20eb3fef933991cd66749f4f2`. The selected configuration used
Charbonnier loss, AdamW, D4 augmentation, statistics conditioning, and no
synthetic degradation (`synthetic_probability: 0.0`). The recorded best
validation PSNR was 25.251129150390625 dB.

External controlled validation is downsample-only and therefore weaker than
native paired degraded/clean validation. PSNR and SSIM require aligned HR
references. Production intensity and structural diagnostics are no-reference
heuristics and cannot demonstrate reconstruction correctness.

## Risks and limitations

- The network can hallucinate plausible texture, erase defects, or oversmooth
  fine structures.
- Acquisition conditions outside the training distribution may produce
  unreliable restoration.
- Tiled and direct inference may differ because boundary context differs.
- Suitability labels are deterministic rules, not calibrated probabilities.
- Scientific output is lossless 8-bit or 16-bit grayscale PNG only.
- One persistent model allocation is required per process; additional worker
  processes multiply memory use.
