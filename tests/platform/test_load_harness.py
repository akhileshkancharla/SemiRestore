from __future__ import annotations

import asyncio
import json
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image

from semirestore.platform.load_testing import (
    LoadEndpoint,
    LoadTestConfig,
    RequestSample,
    aggregate_samples,
    report_payload,
    run_load_test,
    synthetic_grayscale_png,
    write_report,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_url", "file:///private"),
        ("base_url", "http://user:secret@localhost:8000"),
        ("concurrency", 0),
        ("duration_seconds", 0),
        ("request_rate", float("inf")),
        ("timeout_seconds", -1),
        ("width", 0),
        ("height", 20_000),
    ],
)
def test_configuration_rejects_unsafe_or_unbounded_values(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        LoadTestConfig(**{field: value})  # type: ignore[arg-type]


def test_synthetic_input_is_deterministic_grayscale_png() -> None:
    first = synthetic_grayscale_png(13, 7)
    second = synthetic_grayscale_png(13, 7)

    assert first == second
    assert first.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(BytesIO(first)) as image:
        assert image.mode == "L"
        assert image.size == (13, 7)


def controlled_app() -> FastAPI:
    app = FastAPI()

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready")
    async def ready() -> dict[str, object]:
        return {"ready": True, "state": "ready", "unavailable_reason": None}

    async def success(_request: Request) -> JSONResponse:
        return JSONResponse({"controlled": True})

    app.post("/api/v1/analyze")(success)
    app.post("/api/v1/restore")(success)
    app.post("/api/v1/restore-and-analyze")(success)
    return app


@pytest.mark.parametrize("endpoint", list(LoadEndpoint))
def test_each_public_operation_runs_against_controlled_local_service(
    endpoint: LoadEndpoint,
) -> None:
    run = asyncio.run(
        run_load_test(
            LoadTestConfig(
                base_url="http://controlled.test",
                endpoint=endpoint,
                concurrency=2,
                duration_seconds=0.001,
                request_rate=1,
                timeout_seconds=1,
                width=8,
                height=6,
            ),
            transport=httpx.ASGITransport(app=controlled_app()),
        )
    )

    assert run.summary.requests == 1
    assert run.summary.successes == 1
    assert run.summary.failures == 0
    assert run.samples[0].status_code == 200


def test_aggregation_accounts_for_failures_percentiles_backpressure_and_timeouts() -> None:
    samples = [
        RequestSample(10, True, 200),
        RequestSample(20, False, 503, "inference_busy"),
        RequestSample(30, False, 504, "inference_timeout"),
        RequestSample(40, False, None, "client_timeout"),
    ]

    summary = aggregate_samples(samples, elapsed_seconds=2)

    assert summary.requests == 4
    assert summary.successes == 1
    assert summary.failures == 3
    assert summary.throughput_requests_per_second == 2
    assert summary.p50_latency_ms == pytest.approx(25)
    assert summary.p95_latency_ms == pytest.approx(38.5)
    assert summary.p99_latency_ms == pytest.approx(39.7)
    assert summary.backpressure_rejections == 1
    assert summary.timeout_count == 2


def test_http_failures_and_client_timeout_are_accounted_without_response_bodies() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/restore":
            return httpx.Response(
                503,
                json={
                    "error": {
                        "code": "inference_busy",
                        "message": "safe",
                        "details": None,
                        "request_id": None,
                    }
                },
            )
        raise httpx.ReadTimeout("private transport detail", request=request)

    busy = asyncio.run(
        run_load_test(
            LoadTestConfig(
                base_url="http://controlled.test",
                endpoint=LoadEndpoint.RESTORE,
                duration_seconds=0.001,
                request_rate=1,
            ),
            transport=httpx.MockTransport(handler),
        )
    )
    timeout = asyncio.run(
        run_load_test(
            LoadTestConfig(
                base_url="http://controlled.test",
                endpoint=LoadEndpoint.LIVE,
                duration_seconds=0.001,
                request_rate=1,
            ),
            transport=httpx.MockTransport(handler),
        )
    )

    assert busy.summary.backpressure_rejections == 1
    assert busy.samples[0].error_code == "inference_busy"
    assert timeout.summary.timeout_count == 1
    assert timeout.samples[0].error_code == "client_timeout"
    assert "private transport detail" not in json.dumps(report_payload(timeout))


def test_report_generation_contains_configuration_summary_and_safe_samples(
    tmp_path: Path,
) -> None:
    run = asyncio.run(
        run_load_test(
            LoadTestConfig(
                base_url="http://controlled.test",
                endpoint=LoadEndpoint.LIVE,
                duration_seconds=0.001,
                request_rate=1,
            ),
            transport=httpx.ASGITransport(app=controlled_app()),
        )
    )
    destination = tmp_path / "report.json"

    write_report(run, destination)
    loaded = json.loads(destination.read_text(encoding="utf-8"))

    assert loaded["schema_version"] == 1
    assert loaded["config"]["endpoint"] == "live"
    assert loaded["summary"]["requests"] == 1
    assert loaded["samples"][0]["success"] is True
    assert "restoration-quality benchmark" in loaded["measurement_scope"]


def test_raw_results_are_ignored_and_documentation_qualifies_interpretation() -> None:
    ignore = (ROOT / "load-results" / ".gitignore").read_text(encoding="utf-8")
    docs = (ROOT / "docs" / "platform" / "load-testing.md").read_text(encoding="utf-8")
    normalized_docs = " ".join(docs.split())

    assert ignore.splitlines() == ["*", "!.gitignore"]
    assert "No production benchmark was run" in docs
    assert "## CPU interpretation" in docs
    assert "## GPU interpretation" in docs
    assert "One Uvicorn worker owns one process-local model instance" in normalized_docs
    assert "GPU memory is not transparently shared" in normalized_docs
