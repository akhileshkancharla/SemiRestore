# Reproducible model workflows

Run `python scripts/model/workflows.py --config CONFIG COMMAND`. Available
commands are `validate-config`, `audit-dataset`, `train`, `resume`, `evaluate`,
`restore`, `benchmark`, `dry-run`, and `environment`. Put the global `--json`
flag before the command for machine-readable output.

Only explicit `train` or `resume` commands update model parameters. Importing
the CLI and the validation, dry-run, audit, and environment commands start no
training. Restore and benchmark load the checksum-verified runtime checkpoint;
evaluation accepts only safe `best_inference` training checkpoints. Resume
accepts only compatible `training_resume` state.

Reports use configuration-relative paths or basenames, redact sensitive-key
fields, include the deterministic seed and checkpoint identity, and never emit
checkpoint contents. Benchmark values are measured for the requested input and
iterations; no reference values are fabricated. Exit code 2 identifies
configuration/data errors, 3 checkpoint failures, and 4 runtime failures.
