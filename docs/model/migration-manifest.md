# Model migration manifest

This document records the provenance and intended treatment of model-engineering
material migrated into SemiRestore. It does not assert that historical source
or pretrained weights were created or trained in this repository.

## Authoritative sources

### Conditioned deployment run

- Immutable source directory:
  `local/artifacts/naf_sr_conditioned_d037473/`
- Historical training revision:
  [`d037473ddf4a3cd20eb3fef933991cd66749f4f2`](https://github.com/FaisalTabrez/MithaiMafia/commit/d037473ddf4a3cd20eb3fef933991cd66749f4f2)
- Deployment checkpoint source: `best.pt`
- Deployment checkpoint size: `36,565,383` bytes
- Deployment checkpoint SHA-256:
  `273abd9d6dcfa9bdee71ac15016994962304b6c9d902898b4f4d503bed158c28`
- Runtime destination: `artifacts/model/semirestore_conditioned.pt`
- Training checkpoint `last.pt`: excluded; it is a 182,960,366-byte resumable
  training artifact, not the compact deployment checkpoint.

The `best.pt` digest was recomputed from the immutable local export and matches
the previously recorded trusted digest. The checkpoint remains ignored and is
installed locally only after verification. It must not enter normal Git history.

### Earlier implementation

- Repository: <https://github.com/FaisalTabrez/MithaiMafia>
- Revision recorded by the authoritative training run:
  `d037473ddf4a3cd20eb3fef933991cd66749f4f2`
- Newer revision inspected for source inventory only:
  `72c5ff8d6f69bfdc083b7d6864dd169fcadca0dd`

Checkpoint-compatible architecture behavior is sourced from the recorded
training revision. Later upstream changes are not assumed compatible merely
because they are newer.

## Classification vocabulary

- **Direct migration**: reusable source copied with packaging/import-only changes.
- **Adapted migration**: historical logic retained but refactored or validated for
  the new package and public interfaces.
- **New implementation**: behavior not present in the historical implementation.
- **Runtime artifact**: required binary installed locally and excluded from Git.
- **Evidence**: selected small historical output preserved without changing values.

## Source migration map

| Original source | Destination | Classification | Important adaptations and compatibility notes |
|---|---|---|---|
| `src/semirestore/models/naf_blocks.py` at training revision | `src/semirestore/models/naf_blocks.py` | Adapted migration | Preserve layer normalization, gating, attention, residual scaling, parameter names, and tensor behavior; add only validation or typing that does not alter checkpoint keys. |
| `src/semirestore/models/naf_sr.py` at training revision | `src/semirestore/models/naf_sr.py` | Adapted migration | Preserve three-scale encoder/decoder, mean/std/min/max conditioner, stage channel order, PixelShuffle head, bicubic residual, and parameter names. Architecture dimensions remain frozen. |
| `src/semirestore/models/__init__.py` at training revision | `src/semirestore/models/__init__.py` | Adapted migration | Expose only models migrated into this repository and construct them from validated configuration. |
| Run `resolved_config.yaml` | `configs/model/resolved_conditioned.yaml` | Adapted migration | Preserve the authoritative model subsection exactly; separate deployment-compatible model settings from historical absolute data/run paths. |
| `src/semirestore/checkpoints.py` | `src/semirestore/checkpoints.py` | Adapted migration | Add checksum gating, `weights_only=True`, strict container validation, explicit mismatch errors, inference freezing, and verified model identity. |
| `src/semirestore/data.py` and `training_data.py` | model-owned data modules | Adapted migration | Retain safe paired-array loading while adding image-facing validation, supported input types, scale checks, and serialization-friendly metadata. |
| `src/semirestore/degradations.py` | `src/semirestore/degradations.py` | Adapted migration | Retain blur/noise/downsampling logic, add deterministic seed handling, and return applied parameters. |
| `src/semirestore/losses.py` | `src/semirestore/losses.py` | Adapted migration | Preserve Charbonnier behavior and document input/range assumptions. |
| `src/semirestore/metrics.py` | `src/semirestore/metrics.py` | Adapted migration | Preserve reference PSNR/SSIM policies; keep reference metrics distinct from no-reference diagnostics. |
| `src/semirestore/inference.py` | model-owned inference modules | Adapted migration | Refactor directory-oriented inference into reusable preprocessing, postprocessing, model-manager, padding, tiling, and single-image services. |
| `train.py` | model package plus `scripts/model/` CLI | Adapted migration | Preserve optimizer, warmup, EMA, AMP, clipping, resume, and seed behavior behind validated YAML configuration. |
| `evaluation.py` and `evaluate_metrics.py` | model package plus `scripts/model/` CLI | Adapted migration | Retain reference evaluation policies and add dry-run/config validation suitable for the new layout. |
| No historical equivalent | diagnostics and assurance modules | New implementation | Deterministic intensity/structural diagnostics remain external to the pretrained network conditioner. |
| No historical equivalent | `src/semirestore/pipeline.py` and public result contracts | New implementation | Provide the serialization-friendly `restore_and_analyze` handoff without exposing PyTorch internals. |

The final pipeline implementation was completed after a read-only audit of
`origin/work/platform-likhitha` revision
`91d2276d206dda0c1e6dc161bb511da98cf64558`. The platform protocol and response
types remain platform-owned and were not copied into this branch. The model
package instead exposes a primitive `platform_projection()` mapping documented
in `docs/model/platform-integration.md`.

## Artifact and evidence decisions

| Immutable source | Planned tracked representation | Treatment | Reason |
|---|---|---|---|
| Conditioned `best.pt` | `artifacts/model/checksums.json` and documentation | Runtime artifact represented by metadata; binary installed locally | Required deployable weights; verified binary must remain ignored. |
| Conditioned `resolved_config.yaml` | `configs/model/resolved_conditioned.yaml` | Adapted migration | Authoritative checkpoint architecture and training settings. |
| Conditioned `summary.json` | `reports/model/training_summary.json` | Evidence candidate | Small summary containing model selection and training facts. |
| Conditioned `environment.json` | `reports/model/training_environment.json` | Evidence candidate | Small reproducibility record. |
| Conditioned validation ID/OOD summaries | `reports/model/validation_id.json` and `validation_ood.json` | Evidence candidates | Aggregate evaluation evidence; retain original numerical values. |
| Conditioned `ablations.csv` | `reports/model/ablations.csv` | Evidence candidate | Small selection evidence for statistics conditioning. |
| Root bicubic validation summary/metrics | `reports/model/` | Evidence candidates pending relevance review | Reference baseline evidence; copy only if understandable without historical manifests. |
| Root `dataset_audit.json` | `reports/model/dataset_audit.json` | Evidence candidate | Small dataset provenance summary without raw image data. |
| `last.pt`, older `*.pt`, `final_selection_2776ab8/model.pt` | none | Excluded | Training-resume state, obsolete variants, or redundant checkpoint copies. |
| `history.csv` | none by default | Excluded | Large step-level history is not required for deployment provenance. |
| `manifest*.csv`, provisional manifests | none | Excluded | Large historical path manifests are unnecessary and may encode obsolete local paths. |
| Per-image validation/external metrics | none by default | Excluded | Large result tables are unnecessary for the model handoff. |
| Generated `.npy` files and panels | none | Excluded | Generated outputs, not reusable source or compact aggregate evidence. |
| Submission materials and unrelated model variants | none | Excluded | Outside the deployed conditioned-model scope. |

Evidence candidates are not automatically approved for migration by this
manifest. Each is inspected for size, relevance, privacy, and standalone meaning
before copying. Renamed evidence retains its original path in this document.

## Frozen compatibility facts

The authoritative run configuration and summary establish these invariants:

- model name: `naf_sr`
- input/output channels: one grayscale channel
- width: `48`
- encoder blocks: `[2, 2, 4]`
- middle blocks: `6`
- decoder blocks: `[2, 2, 2]`
- statistics conditioning: enabled using mean, standard deviation, minimum,
  and maximum
- conditioning hidden size: `64`
- output scale: `2x`
- parameter count: `9,111,684`
- completed training steps: `5,000`
- recorded best validation PSNR: `25.251129150390625` dB
- recorded environment: Tesla T4 and PyTorch `2.11.0+cu128`

External degradation diagnostics are new assurance measurements. They are not
additional inputs to the pretrained conditioner and must not alter the frozen
checkpoint architecture.
