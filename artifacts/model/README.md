# SemiRestore model artifacts

Checkpoint binaries in this directory are runtime artifacts. They are not
tracked in Git or Git LFS. Machine-readable checkpoint identity is recorded in
[`checksums.json`](checksums.json).

## Verified conditioned checkpoint

| Field | Value |
|---|---|
| Model name | `naf_sr` |
| Model version | `conditioned-d037473` |
| Architecture | statistics-conditioned NAF-SR |
| Output scale | 2x |
| Expected parameters | 9,111,684 |
| Immutable source | `local/artifacts/naf_sr_conditioned_d037473/best.pt` |
| Runtime destination | `artifacts/model/semirestore_conditioned.pt` |
| Expected size | 36,565,383 bytes |
| Expected SHA-256 | `273abd9d6dcfa9bdee71ac15016994962304b6c9d902898b4f4d503bed158c28` |
| Training revision | `d037473ddf4a3cd20eb3fef933991cd66749f4f2` |
| Checkpoint role | deployable best-validation weights |
| Verification status | verified against the immutable local source |

The source file was treated as opaque data during metadata registration. Its
size and SHA-256 were recomputed without deserializing or printing its binary
contents. Both values matched the previously trusted identity exactly.

## Usage and storage rules

- Use `best.pt` as the deployment source. It represents the selected
  best-validation weights intended for inference.
- Do not use or migrate `last.pt`. It is a much larger resumable-training
  artifact containing state that is unnecessary for deployment.
- Never edit, rename, move, delete, stage, or force-add files beneath
  `local/artifacts/`; that tree is immutable source material.
- Keep `artifacts/model/semirestore_conditioned.pt` ignored. A verified local
  runtime copy may exist there after a separately authorized installation step,
  but it must not be committed.
- Verify both byte size and SHA-256 before any future checkpoint load. Treat a
  mismatch as a hard failure rather than updating the trusted metadata.
- Do not deserialize an unverified checkpoint. Safe checkpoint loading belongs
  to a later, independently reviewed milestone.

The binary is excluded from Git because it is a generated runtime dependency,
not source code; keeping it external avoids repository bloat and prevents an
unreviewed serialized object from being distributed through normal source
history. The checksum manifest provides a stable identity without claiming the
checkpoint was trained in this repository.

## Verified local installation

From the repository root, verify and install the authoritative source at the
default ignored runtime destination with:

```console
python scripts/model/install_local_checkpoint.py
```

Alternative paths may be supplied explicitly:

```console
python scripts/model/install_local_checkpoint.py \
  --source path/to/best.pt \
  --destination artifacts/model/semirestore_conditioned.pt
```

The installer loads the trusted size, digest, and default paths from
`checksums.json`. It verifies the source, copies through a temporary `.partial`
file in the destination directory, verifies that copy, and only then replaces
the destination atomically. A correctly installed destination is an idempotent
success. A different regular-file destination is preserved unless `--force` is
explicitly supplied. Failed partial copies are removed.

Installation only copies verified opaque bytes. It does not deserialize or
otherwise inspect the PyTorch checkpoint.
