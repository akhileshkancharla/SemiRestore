# Operations runbook

## Deploy and start

1. Select a reviewed source revision and immutable image tag.
2. Install or mount the checkpoint using the trusted manifest; never bake it
   into source control or the image.
3. Review the environment against [environment.md](environment.md), keep one
   Uvicorn worker, and keep fake-service behavior disabled.
4. Start the CPU service using the direct Uvicorn or Compose command in
   [deployment.md](deployment.md).
5. Require `/health/live` for process liveness and `/health/ready` before
   routing restoration traffic.
6. Run [the end-to-end smoke command](smoke-testing.md) against the deployment.
7. Confirm Prometheus scraping and privacy-safe structured completion logs.

The dashboard is optional and must not determine API health. Authentication,
authorization, TLS termination, rate limiting, and network exposure policy must
be supplied by the deployment environment before use outside a trusted network.

## Operational checks

```powershell
curl.exe -i http://127.0.0.1:8000/health/live
curl.exe -i http://127.0.0.1:8000/health/ready
curl.exe -i http://127.0.0.1:8000/health/model
curl.exe -i http://127.0.0.1:8000/version
curl.exe -i http://127.0.0.1:8000/metrics
```

- Liveness says only that the API process is running. It must stay independent
  of checkpoint/model readiness.
- Readiness says whether this process can accept restoration work. Remove an
  unready instance from traffic without treating it as a crash loop.
- Model health exposes only readiness/state, device, model version, checksum,
  and a safe unavailable reason.
- Correlate failures with `X-Request-ID`; never use request IDs as credentials
  or Prometheus labels.

Monitor bounded request counts/status classes, HTTP and inference duration,
restoration outcomes, active/waiting/capacity gauges, busy rejections, and
timeouts. Alert on sustained readiness failures, `internal_error`,
`restoration_failed`, capacity saturation, busy rejection, and timeouts. No
single latency threshold is prescribed because production performance was not
measured on this workstation.

## Graceful shutdown

Stop sending new traffic, wait for readiness-based removal to propagate, then
send SIGTERM to Uvicorn. The container runs Uvicorn directly with a 30-second
graceful-shutdown timeout; Compose allows 35 seconds before forced termination.
One lifespan shutdown detaches the inference gate and service, stops accepting
work, and closes the retained pipeline. Cancellation cannot guarantee that an
already running native/thread/GPU operation stops immediately.

Do not delete uploads or outputs during shutdown: the platform stores neither
on disk. Prometheus has its own named volume and retention policy.

## Rollback

1. Remove the failing revision from traffic while preserving logs and metrics.
2. Redeploy the last reviewed immutable image and its compatible configuration.
3. Mount the checkpoint identity approved for that image; do not weaken checksum
   or compatibility validation to force startup.
4. Verify liveness, readiness, model health/version/checksum, and metrics.
5. Run the smoke command before restoring traffic.

There is no platform upload/result database migration to reverse. Treat changes
to the checkpoint, model configuration, API schema, and dashboard API base URL
as separate compatibility inputs. If rollback readiness fails, keep the service
out of traffic and use [troubleshooting.md](troubleshooting.md).
