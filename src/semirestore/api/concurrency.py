"""Application-owned bounded inference capacity."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from typing import TypeVar

from semirestore.api.errors import InferenceBusyError, InferenceTimeoutError

ResultT = TypeVar("ResultT")


class InferenceGate:
    """Bound concurrent inference and its acquisition and execution waits."""

    def __init__(
        self,
        *,
        concurrency_limit: int,
        acquisition_timeout_seconds: float,
        execution_timeout_seconds: float,
    ) -> None:
        if concurrency_limit < 1:
            raise ValueError("concurrency limit must be at least one")
        if not math.isfinite(acquisition_timeout_seconds) or acquisition_timeout_seconds <= 0:
            raise ValueError("acquisition timeout must be finite and positive")
        if not math.isfinite(execution_timeout_seconds) or execution_timeout_seconds <= 0:
            raise ValueError("execution timeout must be finite and positive")
        self.concurrency_limit = concurrency_limit
        self.acquisition_timeout_seconds = acquisition_timeout_seconds
        self.execution_timeout_seconds = execution_timeout_seconds
        self._semaphore = asyncio.BoundedSemaphore(concurrency_limit)

    async def run(self, operation: Callable[[], Awaitable[ResultT]]) -> ResultT:
        """Run one operation inside the bounded inference-capacity slot."""
        acquired = False
        try:
            try:
                await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=self.acquisition_timeout_seconds,
                )
            except TimeoutError as error:
                raise InferenceBusyError() from error
            acquired = True

            task = asyncio.create_task(operation())
            try:
                done, _pending = await asyncio.wait(
                    {task},
                    timeout=self.execution_timeout_seconds,
                )
                if not done:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    raise InferenceTimeoutError()
                return await task
            except asyncio.CancelledError:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise
        finally:
            if acquired:
                self._semaphore.release()
