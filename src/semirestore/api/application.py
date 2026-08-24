"""FastAPI application construction and model-service lifespan."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI

from semirestore import __version__
from semirestore.api.errors import register_exception_handlers
from semirestore.platform import ModelHealth, ModelService, ModelServiceState, RuntimeSettings

ModelServiceFactory = Callable[[RuntimeSettings], ModelService]


@dataclass(slots=True)
class ApplicationRuntime:
    """Mutable process-local state owned by one application instance."""

    settings: RuntimeSettings
    model_service: ModelService | None = None
    startup_complete: bool = False
    unavailable_reason: str = "model service startup is in progress"

    def model_health(self) -> ModelHealth:
        """Return safe current model health without leaking implementation errors."""
        if self.model_service is None:
            state = (
                ModelServiceState.UNAVAILABLE
                if self.startup_complete
                else ModelServiceState.STARTING
            )
            return ModelHealth(
                state=state,
                ready=False,
                unavailable_reason=self.unavailable_reason,
            )
        try:
            return self.model_service.health()
        except Exception:
            return ModelHealth(
                state=ModelServiceState.UNAVAILABLE,
                ready=False,
                unavailable_reason="model health check failed",
            )


def create_app(
    *,
    settings: RuntimeSettings | None = None,
    model_service_factory: ModelServiceFactory | None = None,
) -> FastAPI:
    """Create an application whose supplied model service is lifespan-scoped."""
    runtime = ApplicationRuntime(settings=settings or RuntimeSettings())

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        service: ModelService | None = None
        try:
            if model_service_factory is None:
                runtime.unavailable_reason = "model service adapter is not configured"
            else:
                service = model_service_factory(runtime.settings)
                await service.startup()
                runtime.model_service = service
                runtime.unavailable_reason = ""
        except Exception:
            runtime.unavailable_reason = "model service failed to initialize"
            if service is not None:
                try:
                    await service.shutdown()
                except Exception:
                    pass
        finally:
            runtime.startup_complete = True

        try:
            yield
        finally:
            started_service = runtime.model_service
            runtime.model_service = None
            runtime.startup_complete = False
            runtime.unavailable_reason = "model service has stopped"
            if started_service is not None:
                try:
                    await started_service.shutdown()
                except Exception:
                    pass

    app = FastAPI(title="SemiRestore", version=__version__, lifespan=lifespan)
    app.state.runtime = runtime
    register_exception_handlers(app)
    from semirestore.api.routes.operations import router as operations_router

    app.include_router(operations_router)
    return app
