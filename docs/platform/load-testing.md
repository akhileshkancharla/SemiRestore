# Load and resilience testing

The asynchronous load harness exercises one public endpoint at a time with a
deterministically generated grayscale PNG. It never reads or commits a user or
dataset image. Concurrency, duration, request rate, endpoint, client timeout,
and input dimensions are explicit command-line settings.

```sh
.venv/Scripts/python scripts/platform/load_test.py \
  --endpoint restore-and-analyze \
  --base-url http://127.0.0.1:8000 \
  --concurrency 1 \
  --duration 30 \
  --rate 1 \
  --timeout 120 \
  --width 512 \
  --height 512
```

Use `live`, `ready`, `analyze`, `restore`, or `restore-and-analyze` to isolate
the route being measured. JSON reports default to the ignored
`load-results/latest.json` path. They contain safe request status/error codes
and client-observed latency, plus requests, successes, failures, throughput,
p50/p95/p99 latency, backpressure rejections, and timeout counts. They contain
no image bytes or response bodies.

No production benchmark was run during this milestone because this workstation
has no verified checkpoint-backed server. Automated tests exercised every route
against a controlled local ASGI service and separately verified backpressure and
timeout accounting. Those checks validate harness behavior, not production
performance or restoration quality.

## CPU interpretation

Run one endpoint/configuration combination at a time, record the exact host CPU,
input dimensions, worker count, model version, checkpoint identity, and runtime
settings outside the generated report, and compare only like-for-like runs. One
Uvicorn worker owns one process-local model instance. Additional workers load
additional model instances and change memory use and scheduling behavior.

## GPU interpretation

GPU results require the separately reviewed CUDA image, compatible driver and
runtime, and the verified checkpoint. GPU memory is not transparently shared
between worker processes; each worker owns its own model instance. Record warmup,
device, CUDA/PyTorch versions, memory use, and concurrency policy. Do not compare
CPU and GPU latency as if the environments were equivalent, and do not infer
restoration quality from throughput or latency.
