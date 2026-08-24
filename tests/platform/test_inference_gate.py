from __future__ import annotations

import asyncio
import math

import pytest

from semirestore.api.concurrency import InferenceGate
from semirestore.api.errors import InferenceBusyError, InferenceTimeoutError
from semirestore.platform import ModelServiceInferenceError


def run(coroutine: object) -> object:
    return asyncio.run(coroutine)  # type: ignore[arg-type]


def make_gate(
    *,
    limit: int = 1,
    acquisition_timeout: float = 0.2,
    execution_timeout: float = 0.2,
) -> InferenceGate:
    return InferenceGate(
        concurrency_limit=limit,
        acquisition_timeout_seconds=acquisition_timeout,
        execution_timeout_seconds=execution_timeout,
    )


def test_limit_one_waiter_proceeds_only_after_release() -> None:
    async def scenario() -> None:
        gate = make_gate()
        first_started = asyncio.Event()
        second_started = asyncio.Event()
        release_first = asyncio.Event()

        async def first() -> str:
            first_started.set()
            await release_first.wait()
            return "first"

        async def second() -> str:
            second_started.set()
            return "second"

        first_task = asyncio.create_task(gate.run(first))
        await first_started.wait()
        second_task = asyncio.create_task(gate.run(second))
        await asyncio.sleep(0)
        assert not second_started.is_set()

        release_first.set()
        assert await asyncio.gather(first_task, second_task) == ["first", "second"]
        assert second_started.is_set()

    run(scenario())


def test_limit_greater_than_one_never_exceeds_configured_capacity() -> None:
    async def scenario() -> None:
        gate = make_gate(limit=2)
        release = asyncio.Event()
        saturated = asyncio.Event()
        active = 0
        maximum_active = 0

        async def operation() -> None:
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            if active == 2:
                saturated.set()
            try:
                await release.wait()
            finally:
                active -= 1

        tasks = [asyncio.create_task(gate.run(operation)) for _ in range(3)]
        await saturated.wait()
        await asyncio.sleep(0)
        assert active == 2
        assert maximum_active == 2

        release.set()
        await asyncio.gather(*tasks)
        assert maximum_active == 2

    run(scenario())


def test_acquisition_timeout_does_not_invoke_rejected_operation() -> None:
    async def scenario() -> None:
        gate = make_gate(acquisition_timeout=0.01)
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        rejected_called = False

        async def first() -> None:
            first_started.set()
            await release_first.wait()

        async def rejected() -> None:
            nonlocal rejected_called
            rejected_called = True

        first_task = asyncio.create_task(gate.run(first))
        await first_started.wait()
        with pytest.raises(InferenceBusyError):
            await gate.run(rejected)
        assert rejected_called is False

        release_first.set()
        await first_task

    run(scenario())


def test_execution_timeout_cancels_operation_and_releases_capacity() -> None:
    async def scenario() -> None:
        gate = make_gate(execution_timeout=0.01)
        never_release = asyncio.Event()
        cancelled = asyncio.Event()

        async def timed_out() -> None:
            try:
                await never_release.wait()
            finally:
                cancelled.set()

        with pytest.raises(InferenceTimeoutError):
            await gate.run(timed_out)
        assert cancelled.is_set()
        assert await gate.run(lambda: _value("later")) == "later"

    run(scenario())


@pytest.mark.parametrize(
    "error",
    [ModelServiceInferenceError("known failure"), RuntimeError("unexpected failure")],
)
def test_capacity_is_released_after_known_and_unexpected_failures(error: Exception) -> None:
    async def scenario() -> None:
        gate = make_gate()

        async def fail() -> None:
            raise error

        with pytest.raises(type(error)):
            await gate.run(fail)
        assert await gate.run(lambda: _value("later")) == "later"

    run(scenario())


def test_client_cancellation_propagates_and_releases_capacity() -> None:
    async def scenario() -> None:
        gate = make_gate()
        started = asyncio.Event()
        cancelled = asyncio.Event()
        blocker = asyncio.Event()

        async def operation() -> None:
            started.set()
            try:
                await blocker.wait()
            finally:
                cancelled.set()

        task = asyncio.create_task(gate.run(operation))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()
        assert await gate.run(lambda: _value("later")) == "later"

    run(scenario())


def test_cancelled_acquisition_does_not_over_release() -> None:
    async def scenario() -> None:
        gate = make_gate()
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        waiting_operation_called = False

        async def first() -> None:
            first_started.set()
            await release_first.wait()

        async def waiting() -> None:
            nonlocal waiting_operation_called
            waiting_operation_called = True

        first_task = asyncio.create_task(gate.run(first))
        await first_started.wait()
        waiting_task = asyncio.create_task(gate.run(waiting))
        await asyncio.sleep(0)
        waiting_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting_task
        assert waiting_operation_called is False

        release_first.set()
        await first_task

        blocker = asyncio.Event()
        third_started = asyncio.Event()
        fourth_started = asyncio.Event()

        async def third() -> None:
            third_started.set()
            await blocker.wait()

        async def fourth() -> None:
            fourth_started.set()

        third_task = asyncio.create_task(gate.run(third))
        await third_started.wait()
        fourth_task = asyncio.create_task(gate.run(fourth))
        await asyncio.sleep(0)
        assert not fourth_started.is_set()
        blocker.set()
        await asyncio.gather(third_task, fourth_task)

    run(scenario())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"concurrency_limit": 0},
        {"acquisition_timeout_seconds": 0},
        {"acquisition_timeout_seconds": math.inf},
        {"execution_timeout_seconds": -1},
        {"execution_timeout_seconds": math.nan},
    ],
)
def test_controller_rejects_invalid_limits(kwargs: dict[str, float | int]) -> None:
    values: dict[str, float | int] = {
        "concurrency_limit": 1,
        "acquisition_timeout_seconds": 1.0,
        "execution_timeout_seconds": 1.0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        InferenceGate(**values)  # type: ignore[arg-type]


async def _value(value: str) -> str:
    return value
