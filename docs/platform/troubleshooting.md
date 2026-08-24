# Troubleshooting

Start with liveness, readiness, model health, the response `X-Request-ID`, and
bounded Prometheus outcomes. Client responses intentionally suppress raw
exceptions, paths, stack traces, upload content, and environment details.

| Symptom or code | Meaning | Safe operator action |
| --- | --- | --- |
| Liveness cannot be reached | Process, port, proxy, or network failure | Check process/container state, port mapping, and proxy health |
| Liveness 200; readiness 503 | Adapter cannot currently accept work | Check the safe model-health state and private startup logs; verify mount/config/checksum compatibility |
| `invalid_request` | Route, method, multipart, or request validation failed | Confirm the documented method, path, and `image` field |
| `empty_upload` | No image bytes were supplied | Re-select a non-empty source file |
| `unsupported_media_type` | Declared/detected type is unsupported or mismatched | Use PNG, JPEG, or single-frame TIFF with the correct media type |
| `upload_too_large` | Encoded limit was exceeded | Use a bounded source or review the configured limit and capacity |
| `invalid_image` | Image decoding/structure failed | Re-export a valid, non-truncated single-frame image |
| `image_dimensions_exceeded` | Width, height, or pixel limit was exceeded | Reduce dimensions or deliberately review all memory limits |
| `model_unavailable` | No ready model service can accept work | Restore readiness; never enable a fake fallback |
| `inference_busy` | Capacity was not acquired before its wait timeout | Reduce request rate or validate a capacity change with load tests |
| `inference_timeout` | Adapter wait exceeded its execution timeout | Inspect saturation/input size; underlying native work may continue |
| `restoration_failed` | A known model/inference failure was safely mapped | Correlate the request ID with private bounded logs; do not ask clients for secrets |
| `internal_error` | An unexpected failure was safely suppressed | Correlate request ID, revision, and metrics; preserve the generic public response |

## Checkpoint startup failures

Run the installer from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/model/install_local_checkpoint.py
```

It requires a regular source file with the exact tracked size and SHA-256 and
installs atomically to the ignored runtime destination. A mismatch is a hard
failure. Do not edit the manifest to fit an untrusted file, deserialize an
unverified checkpoint, commit the runtime binary, or expose its path through a
public response.

For Compose, confirm `SEMIRESTORE_CHECKPOINT_HOST_PATH` names an existing host
file and that the read-only bind reaches `/models/semirestore_conditioned.pt`.
The committed CPU image cannot become a CUDA image by changing only the device
preference.

## Dashboard and observability

If the dashboard loads but shows no API state, confirm its built API base URL,
the `/service` proxy, browser network response, and API readiness. The dashboard
does not store uploads remotely; replacing a selection revokes the old local
preview URL. A recoverable API failure intentionally keeps the selected image.

If Prometheus cannot scrape, call `/metrics` directly, verify that Prometheus is
on the private `backend` network, and inspect its read-only configuration mount.
The metrics endpoint remains available while the model is unready and has no
readiness gauge by design. Never add request IDs, filenames, raw routes, model
identity, or exception text as labels.
