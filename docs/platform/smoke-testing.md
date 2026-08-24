# End-to-end smoke testing

The smoke command exercises one running service through its public HTTP
interface. It creates a deterministic grayscale PNG in memory, then checks this
sequence:

1. `GET /health/live` returns the exact liveness contract.
2. `GET /health/ready` says the process can accept restoration work.
3. `GET /health/model` agrees that the model is ready.
4. A multipart `image` upload is sent to the selected operation.
5. The JSON is validated against `AnalyzeResponse` or `RestoreResponse`.
6. Restoration output is strict Base64, has a PNG signature, decodes fully, and
   has dimensions matching the response metadata.

Run the complete operation against a locally started API:

```powershell
.\.venv\Scripts\python.exe scripts/platform/smoke_test.py `
  --base-url http://127.0.0.1:8000 `
  --operation restore-and-analyze `
  --timeout 120 `
  --width 16 `
  --height 16
```

Valid operations are `analyze`, `restore`, and `restore-and-analyze`. Input
dimensions and the client timeout are bounded. The URL must be HTTP(S), cannot
embed credentials, and cannot include a query or fragment.

A successful report contains only check names, dimensions, media types,
diagnostic section names, warning count, and returned device/model identifiers.
It never contains image bytes, Base64 content, filenames, response bodies,
paths, raw server errors, or secrets. Failure messages are fixed and safe.

An HTTP 503 readiness response is an expected failed smoke run when the verified
checkpoint is absent. Install the checkpoint and restart the one-worker API;
never enable a fake service to make a production smoke check pass.

## Automated and real-checkpoint coverage

The regular integration tests run the complete sequence with controlled HTTP
responses and validate an actual FastAPI response produced by an explicit
test-only fake. This proves harness behavior and contract compatibility, not
model correctness.

The optional test below uses the production adapter on CPU and skips with
`verified ignored runtime checkpoint is unavailable` when the ignored runtime
checkpoint does not exist:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/integration/test_smoke_test.py -m local_checkpoint -ra
```

For a release, repeat the command against the deployed CPU container and record
the image digest, package/model version, checkpoint checksum, settings, and host
outside the smoke output. CUDA smoke validation requires a separately reviewed
GPU image and host; it was not performed for this handoff.
