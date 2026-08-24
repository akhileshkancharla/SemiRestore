from __future__ import annotations

import asyncio

import pytest

from semirestore.api.concurrency import InferenceGate
from semirestore.api.errors import InferenceBusyError, InferenceTimeoutError
from semirestore.api.metrics import PlatformMetrics


def value(metrics: PlatformMetrics, name: str) -> float:
    result = metrics.registry.get_sample_value(name)
    assert result is not None
    return result


def make_gate(
    metrics: PlatformMetrics,
    *,
    acquisition_timeout: float = 0.2,
    execution_timeout: float = 0.2,
) -> InferenceGate:
    return InferenceGate(
        concurrency_limit=1,
        acquisition_timeout_seconds=acquisition_timeout,
        execution_timeout_seconds=execution_timeout,
        metrics=metrics,
    )


def test_configured_capacity_and_deterministic_waiting_active_lifecycle() -> None:
    async def scenario() -> None:
        metrics = PlatformMetrics(inference_capacity=1)
        gate = make_gate(metrics)
        first_started = asyncio.Event()
        release_first = asyncio.Event()

        async def first() -> None:
            first_started.set()
            await release_first.wait()

        first_task = asyncio.create_task(gate.run(first))
        await first_started.wait()
        assert value(metrics, "semirestore_inference_capacity") == 1
        assert value(metrics, "semirestore_inference_active") == 1
        assert value(metrics, "semirestore_inference_waiting") == 0

        waiting_task = asyncio.create_task(gate.run(lambda: _value(None)))
        await asyncio.sleep(0)
        assert value(metrics, "semirestore_inference_active") == 1
        assert value(metrics, "semirestore_inference_waiting") == 1

        waiting_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting_task
        assert value(metrics, "semirestore_inference_waiting") == 0
        assert value(metrics, "semirestore_inference_active") == 1

        release_first.set()
        await first_task
        assert value(metrics, "semirestore_inference_active") == 0
        assert value(metrics, "semirestore_inference_waiting") == 0

    asyncio.run(scenario())


def test_active_gauge_returns_to_zero_after_success_failure_and_timeout() -> None:
    async def scenario() -> None:
        metrics = PlatformMetrics(inference_capacity=1)
        gate = make_gate(metrics, execution_timeout=0.01)

        assert await gate.run(lambda: _value("success")) == "success"
        assert value(metrics, "semirestore_inference_active") == 0

        async def fail() -> None:
            raise RuntimeError("private failure")

        with pytest.raises(RuntimeError):
            await gate.run(fail)
        assert value(metrics, "semirestore_inference_active") == 0

        blocker = asyncio.Event()
        with pytest.raises(InferenceTimeoutError):
            await gate.run(blocker.wait)
        assert value(metrics, "semirestore_inference_active") == 0
        assert value(metrics, "semirestore_inference_waiting") == 0
        assert value(metrics, "semirestore_inference_timeouts_total") == 1

    asyncio.run(scenario())


def test_active_gauge_returns_to_zero_after_cancellation() -> None:
    async def scenario() -> None:
        metrics = PlatformMetrics(inference_capacity=1)
        gate = make_gate(metrics)
        started = asyncio.Event()
        blocker = asyncio.Event()

        async def operation() -> None:
            started.set()
            await blocker.wait()

        task = asyncio.create_task(gate.run(operation))
        await started.wait()
        assert value(metrics, "semirestore_inference_active") == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert value(metrics, "semirestore_inference_active") == 0
        assert value(metrics, "semirestore_inference_waiting") == 0

    asyncio.run(scenario())


def test_busy_rejection_decrements_waiting_without_changing_active() -> None:
    async def scenario() -> None:
        metrics = PlatformMetrics(inference_capacity=1)
        gate = make_gate(metrics, acquisition_timeout=0.01)
        started = asyncio.Event()
        release = asyncio.Event()

        async def operation() -> None:
            started.set()
            await release.wait()

        active_task = asyncio.create_task(gate.run(operation))
        await started.wait()
        with pytest.raises(InferenceBusyError):
            await gate.run(lambda: _value(None))
        assert value(metrics, "semirestore_inference_busy_total") == 1
        assert value(metrics, "semirestore_inference_waiting") == 0
        assert value(metrics, "semirestore_inference_active") == 1
        release.set()
        await active_task
        assert value(metrics, "semirestore_inference_active") == 0

    asyncio.run(scenario())


def test_concurrency_gauges_never_become_negative() -> None:
    async def scenario() -> None:
        metrics = PlatformMetrics(inference_capacity=1)
        gate = make_gate(metrics)
        for _ in range(3):
            await gate.run(lambda: _value(None))
        assert value(metrics, "semirestore_inference_active") >= 0
        assert value(metrics, "semirestore_inference_waiting") >= 0

    asyncio.run(scenario())


async def _value(value: object) -> object:
    return value
