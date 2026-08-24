"""Explicit test-only model-service fixtures.

Nothing in this module is imported by the production package.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from semirestore.platform import ModelHealth, ModelServiceState


class FakeModelService:
    """Lifecycle-only fake; it never performs or claims real restoration."""

    def __init__(
        self,
        *,
        health: ModelHealth | None = None,
        startup_error: Exception | None = None,
    ) -> None:
        self.current_health = health or ModelHealth(
            state=ModelServiceState.READY,
            ready=True,
            device="test-device",
            model_version="synthetic-test-model",
            checkpoint_checksum="test-checksum",
        )
        self.startup_error = startup_error
        self.startup_calls = 0
        self.shutdown_calls = 0

    async def startup(self) -> None:
        self.startup_calls += 1
        if self.startup_error is not None:
            raise self.startup_error

    async def shutdown(self) -> None:
        self.shutdown_calls += 1

    def health(self) -> ModelHealth:
        return self.current_health


@pytest.fixture
def fake_model_service() -> Iterator[FakeModelService]:
    """Provide an explicitly requested synthetic model-service test double."""
    yield FakeModelService()
