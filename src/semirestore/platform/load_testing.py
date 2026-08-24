"""Reproducible asynchronous HTTP load harness for the public platform API."""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import asdict, dataclass
from enum import StrEnum
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import numpy as np
from PIL import Image


class LoadEndpoint(StrEnum):
    LIVE = "live"
    READY = "ready"
    ANALYZE = "analyze"
    RESTORE = "restore"
    RESTORE_AND_ANALYZE = "restore-and-analyze"


_ENDPOINT_PATHS = {
    LoadEndpoint.LIVE: "/health/live",
    LoadEndpoint.READY: "/health/ready",
    LoadEndpoint.ANALYZE: "/api/v1/analyze",
    LoadEndpoint.RESTORE: "/api/v1/restore",
    LoadEndpoint.RESTORE_AND_ANALYZE: "/api/v1/restore-and-analyze",
}


@dataclass(frozen=True, slots=True)
class LoadTestConfig:
    base_url: str = "http://127.0.0.1:8000"
    endpoint: LoadEndpoint = LoadEndpoint.LIVE
    concurrency: int = 1
    duration_seconds: float = 10.0
    request_rate: float = 1.0
    timeout_seconds: float = 30.0
    width: int = 64
    height: int = 64

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username:
            raise ValueError("base_url must be an HTTP(S) URL without embedded credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain a query or fragment")
        if not isinstance(self.endpoint, LoadEndpoint):
            raise ValueError("endpoint must be a supported LoadEndpoint")
        for name in ("concurrency", "width", "height"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("duration_seconds", "request_rate", "timeout_seconds"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.width > 16_384 or self.height > 16_384:
            raise ValueError("synthetic input dimensions cannot exceed 16384 pixels per axis")
        if self.width * self.height > 16_777_216:
            raise ValueError("synthetic input cannot exceed 16777216 pixels")


@dataclass(frozen=True, slots=True)
class RequestSample:
    latency_ms: float
    success: bool
    status_code: int | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("latency_ms must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class LoadSummary:
    requests: int
    successes: int
    failures: int
    elapsed_seconds: float
    throughput_requests_per_second: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    backpressure_rejections: int
    timeout_count: int


@dataclass(frozen=True, slots=True)
class LoadRun:
    config: LoadTestConfig
    summary: LoadSummary
    samples: tuple[RequestSample, ...]


def synthetic_grayscale_png(width: int, height: int) -> bytes:
    if width < 1 or height < 1:
        raise ValueError("synthetic image dimensions must be positive")
    rows = np.arange(height, dtype=np.uint32)[:, None]
    columns = np.arange(width, dtype=np.uint32)[None, :]
    image = ((rows * 17 + columns * 37 + 11) % 256).astype(np.uint8)
    output = BytesIO()
    Image.fromarray(image, mode="L").save(output, format="PNG")
    return output.getvalue()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def aggregate_samples(samples: list[RequestSample], elapsed_seconds: float) -> LoadSummary:
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be finite and positive")
    latencies = [sample.latency_ms for sample in samples]
    successes = sum(sample.success for sample in samples)
    backpressure = sum(sample.error_code == "inference_busy" for sample in samples)
    timeouts = sum(
        sample.error_code in {"inference_timeout", "client_timeout"} for sample in samples
    )
    return LoadSummary(
        requests=len(samples),
        successes=successes,
        failures=len(samples) - successes,
        elapsed_seconds=elapsed_seconds,
        throughput_requests_per_second=len(samples) / elapsed_seconds,
        p50_latency_ms=_percentile(latencies, 0.50),
        p95_latency_ms=_percentile(latencies, 0.95),
        p99_latency_ms=_percentile(latencies, 0.99),
        backpressure_rejections=backpressure,
        timeout_count=timeouts,
    )


def _error_code(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        return "unexpected_response"
    if not isinstance(body, dict):
        return "unexpected_response"
    error = body.get("error")
    if not isinstance(error, dict) or not isinstance(error.get("code"), str):
        return "unexpected_response"
    return error["code"]


async def _one_request(
    client: httpx.AsyncClient,
    config: LoadTestConfig,
    image_bytes: bytes,
) -> RequestSample:
    started = time.perf_counter()
    try:
        endpoint_path = _ENDPOINT_PATHS[config.endpoint]
        if config.endpoint in {LoadEndpoint.LIVE, LoadEndpoint.READY}:
            response = await client.get(endpoint_path)
        else:
            response = await client.post(
                endpoint_path,
                files={"image": ("synthetic-load.png", image_bytes, "image/png")},
            )
        latency_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        success = 200 <= response.status_code < 300
        return RequestSample(
            latency_ms=latency_ms,
            success=success,
            status_code=response.status_code,
            error_code=None if success else _error_code(response),
        )
    except httpx.TimeoutException:
        return RequestSample(
            latency_ms=max(0.0, (time.perf_counter() - started) * 1000.0),
            success=False,
            error_code="client_timeout",
        )
    except httpx.HTTPError:
        return RequestSample(
            latency_ms=max(0.0, (time.perf_counter() - started) * 1000.0),
            success=False,
            error_code="transport_error",
        )


async def run_load_test(
    config: LoadTestConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> LoadRun:
    image_bytes = synthetic_grayscale_png(config.width, config.height)
    request_count = max(1, math.ceil(config.duration_seconds * config.request_rate))
    semaphore = asyncio.Semaphore(config.concurrency)
    started = time.perf_counter()

    async with httpx.AsyncClient(
        base_url=config.base_url.rstrip("/"),
        timeout=config.timeout_seconds,
        transport=transport,
    ) as client:
        async def scheduled_request(index: int) -> RequestSample:
            target = started + index / config.request_rate
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            async with semaphore:
                return await _one_request(client, config, image_bytes)

        samples = await asyncio.gather(
            *(scheduled_request(index) for index in range(request_count))
        )

    elapsed = max(time.perf_counter() - started, 1e-9)
    collected = list(samples)
    return LoadRun(
        config=config,
        summary=aggregate_samples(collected, elapsed),
        samples=tuple(collected),
    )


def report_payload(run: LoadRun) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "measurement_scope": "HTTP client observations; not a restoration-quality benchmark",
        "config": {
            **asdict(run.config),
            "endpoint": run.config.endpoint.value,
        },
        "summary": asdict(run.summary),
        "samples": [asdict(sample) for sample in run.samples],
    }


def write_report(run: LoadRun, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report_payload(run), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "LoadEndpoint",
    "LoadRun",
    "LoadSummary",
    "LoadTestConfig",
    "RequestSample",
    "aggregate_samples",
    "report_payload",
    "run_load_test",
    "synthetic_grayscale_png",
    "write_report",
]
