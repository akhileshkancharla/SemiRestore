"""Safe request correlation, timing, and application-owned logging."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import IO, Any, TypeVar
from uuid import uuid4

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from semirestore.api.errors import InferenceBusyError, InferenceTimeoutError
from semirestore.api.metrics import PlatformMetrics, RestorationOutcome
from semirestore.api.schemas import ErrorBody, ErrorCode, ErrorResponse
from semirestore.platform import (
    ModelServiceInferenceError,
    ModelServiceInitializationError,
    ModelServiceUnavailableError,
    RuntimeSettings,
)

LOGGER_NAME = "semirestore"
REQUEST_ID_HEADER = "x-request-id"
UNMATCHED_ROUTE = "<unmatched>"

_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}\Z", re.ASCII)
_HANDLER_MARKER = "_semirestore_owned_handler"
_LOG_FIELDS = (
    "environment",
    "request_id",
    "method",
    "route",
    "status",
    "status_class",
    "duration_ms",
    "inference_duration_ms",
    "outcome",
    "error_code",
    "model_readiness",
)

ResultT = TypeVar("ResultT")


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _event_payload(record: logging.LogRecord) -> dict[str, object]:
    supplied = getattr(record, "semirestore_fields", {})
    fields = supplied if isinstance(supplied, Mapping) else {}
    payload: dict[str, object] = {
        "timestamp": _utc_timestamp(),
        "level": record.levelname,
        "event": str(getattr(record, "semirestore_event", "application_event")),
    }
    payload.update({name: fields.get(name) for name in _LOG_FIELDS})
    return payload


class _SafeJsonFormatter(logging.Formatter):
    """Serialize only the explicit SemiRestore logging contract."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(_event_payload(record), separators=(",", ":"), ensure_ascii=True)


class _SafeTextFormatter(logging.Formatter):
    """Render the same safe contract in a readable local-development format."""

    def format(self, record: logging.LogRecord) -> str:
        payload = _event_payload(record)
        leading = (
            f"{payload.pop('timestamp')} {payload.pop('level')} "
            f"event={payload.pop('event')}"
        )
        fields = " ".join(
            f"{name}={json.dumps(value, ensure_ascii=True)}"
            for name, value in payload.items()
            if value is not None
        )
        return f"{leading} {fields}" if fields else leading


def configure_application_logging(
    settings: RuntimeSettings,
    *,
    stream: IO[str] | None = None,
) -> logging.Logger:
    """Configure one isolated, idempotent handler for SemiRestore events."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(settings.log_level)
    logger.propagate = False

    owned_handlers = [
        handler for handler in logger.handlers if getattr(handler, _HANDLER_MARKER, False)
    ]
    if owned_handlers:
        handler = owned_handlers[0]
        for duplicate in owned_handlers[1:]:
            logger.removeHandler(duplicate)
    else:
        handler = logging.StreamHandler()
        setattr(handler, _HANDLER_MARKER, True)
        logger.addHandler(handler)

    if isinstance(handler, logging.StreamHandler):
        handler.setStream(stream if stream is not None else sys.stderr)
    handler.setLevel(settings.log_level)
    handler.setFormatter(_SafeJsonFormatter() if settings.json_logging else _SafeTextFormatter())
    return logger


def emit_event(level: int, event: str, **fields: object) -> None:
    """Emit one event whose formatter cannot inspect arbitrary application data."""
    safe_fields = {name: fields.get(name) for name in _LOG_FIELDS}
    logging.getLogger(LOGGER_NAME).log(
        level,
        event,
        extra={"semirestore_event": event, "semirestore_fields": safe_fields},
    )


def select_request_id(headers: Sequence[tuple[bytes, bytes]]) -> str:
    """Accept one narrowly valid client ID or generate an opaque replacement."""
    values = [value for name, value in headers if name.lower() == b"x-request-id"]
    if len(values) == 1:
        try:
            candidate = values[0].decode("ascii")
        except UnicodeDecodeError:
            candidate = ""
        if _REQUEST_ID_PATTERN.fullmatch(candidate) is not None:
            return candidate
    return uuid4().hex


def _request_id_from_scope(scope: Scope) -> str | None:
    state = scope.get("state")
    if not isinstance(state, dict):
        return None
    request_id = state.get("request_id")
    if isinstance(request_id, str) and _REQUEST_ID_PATTERN.fullmatch(request_id) is not None:
        return request_id
    return None


def _route_template(scope: Scope) -> str:
    route = scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) and path.startswith("/") else UNMATCHED_ROUTE


def _method(scope: Scope) -> str:
    method = scope.get("method")
    if isinstance(method, str) and method.isascii() and method.isalpha() and len(method) <= 16:
        return method.upper()
    return "<invalid>"


def _elapsed_seconds(start_ns: int) -> float:
    return max(0, time.perf_counter_ns() - start_ns) / 1_000_000_000


def _milliseconds(duration_seconds: float) -> float:
    return round(duration_seconds * 1_000, 3)


def _completion_level(status_code: int) -> int:
    if status_code >= 500:
        return logging.ERROR
    if status_code >= 400:
        return logging.WARNING
    return logging.INFO


def _stable_error_code(value: object) -> str | None:
    if isinstance(value, str) and value in ErrorCode:
        return value
    return None


class RequestObservabilityMiddleware:
    """Correlate and log HTTP requests without reading application payloads."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        environment: str,
        metrics: PlatformMetrics,
    ) -> None:
        self.app = app
        self.environment = environment
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_ns = time.perf_counter_ns()
        state = scope.setdefault("state", {})
        request_id = select_request_id(scope.get("headers", ()))
        state["request_id"] = request_id
        response_started = False
        response_status: int | None = None

        async def send_with_request_id(message: Message) -> None:
            nonlocal response_started, response_status
            if message["type"] == "http.response.start":
                response_started = True
                response_status = message["status"]
                updated = dict(message)
                updated["headers"] = list(message.get("headers", ()))
                MutableHeaders(scope=updated)[REQUEST_ID_HEADER] = request_id
                message = updated
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        except asyncio.CancelledError:
            duration_seconds = _elapsed_seconds(start_ns)
            route = _route_template(scope)
            if route != "/metrics":
                self.metrics.observe_http(
                    method=_method(scope),
                    route=route,
                    status_class="cancelled",
                    duration_seconds=duration_seconds,
                )
            emit_event(
                logging.WARNING,
                "http_request_cancelled",
                environment=self.environment,
                request_id=request_id,
                method=_method(scope),
                route=route,
                duration_ms=_milliseconds(duration_seconds),
                outcome="cancelled",
            )
            raise
        except Exception:
            if response_started:
                raise
            state["error_code"] = ErrorCode.INTERNAL_ERROR.value
            envelope = ErrorResponse(
                error=ErrorBody(
                    code=ErrorCode.INTERNAL_ERROR,
                    message="An internal server error occurred.",
                    request_id=request_id,
                )
            )
            response = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=envelope.model_dump(mode="json"),
            )
            await response(scope, receive, send_with_request_id)

        final_status = response_status or status.HTTP_500_INTERNAL_SERVER_ERROR
        duration_seconds = _elapsed_seconds(start_ns)
        route = _route_template(scope)
        status_class = f"{final_status // 100}xx"
        if route != "/metrics":
            self.metrics.observe_http(
                method=_method(scope),
                route=route,
                status_class=status_class,
                duration_seconds=duration_seconds,
            )
        emit_event(
            _completion_level(final_status),
            "http_request_completed",
            environment=self.environment,
            request_id=request_id,
            method=_method(scope),
            route=route,
            status=final_status,
            status_class=status_class,
            duration_ms=_milliseconds(duration_seconds),
            error_code=_stable_error_code(state.get("error_code")),
        )


async def observe_inference(
    request: Request,
    operation: Callable[[], Awaitable[ResultT]],
) -> ResultT:
    """Log the platform-observed gate/service interval and re-raise its outcome."""
    start_ns = time.perf_counter_ns()
    common = {
        "environment": request.app.state.runtime.settings.environment,
        "request_id": _request_id_from_scope(request.scope),
        "method": _method(request.scope),
        "route": _route_template(request.scope),
    }
    try:
        result = await operation()
    except asyncio.CancelledError:
        duration_seconds = _elapsed_seconds(start_ns)
        request.app.state.runtime.metrics.observe_inference(
            outcome="cancelled",
            duration_seconds=duration_seconds,
        )
        emit_event(
            logging.WARNING,
            "inference_cancelled",
            **common,
            inference_duration_ms=_milliseconds(duration_seconds),
            outcome="cancelled",
            model_readiness="ready",
        )
        raise
    except InferenceBusyError:
        _emit_inference_outcome(
            request, common, start_ns, "busy", ErrorCode.INFERENCE_BUSY, "ready"
        )
        raise
    except InferenceTimeoutError:
        _emit_inference_outcome(
            request, common, start_ns, "timeout", ErrorCode.INFERENCE_TIMEOUT, "ready"
        )
        raise
    except (ModelServiceInitializationError, ModelServiceUnavailableError):
        _emit_inference_outcome(
            request,
            common,
            start_ns,
            "unavailable",
            ErrorCode.MODEL_UNAVAILABLE,
            "unavailable",
        )
        raise
    except ModelServiceInferenceError:
        _emit_inference_outcome(
            request,
            common,
            start_ns,
            "failed",
            ErrorCode.RESTORATION_FAILED,
            "ready",
        )
        raise
    except Exception:
        _emit_inference_outcome(
            request, common, start_ns, "failed", ErrorCode.INTERNAL_ERROR, "ready"
        )
        raise

    _emit_inference_outcome(request, common, start_ns, "success", None, "ready")
    return result


def _emit_inference_outcome(
    request: Request,
    common: Mapping[str, Any],
    start_ns: int,
    outcome: RestorationOutcome,
    error_code: ErrorCode | None,
    model_readiness: str,
) -> None:
    duration_seconds = _elapsed_seconds(start_ns)
    request.app.state.runtime.metrics.observe_inference(
        outcome=outcome,
        duration_seconds=duration_seconds,
    )
    level = logging.INFO if outcome == "success" else logging.WARNING
    if outcome == "failed":
        level = logging.ERROR
    emit_event(
        level,
        "inference_completed",
        **common,
        inference_duration_ms=_milliseconds(duration_seconds),
        outcome=outcome,
        error_code=error_code.value if error_code is not None else None,
        model_readiness=model_readiness,
    )


async def observe_restoration(
    request: Request,
    operation: Callable[[], Awaitable[ResultT]],
) -> ResultT:
    """Record exactly one bounded outcome for a validated restoration attempt."""
    outcome: RestorationOutcome | None = None
    try:
        result = await operation()
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    except InferenceBusyError:
        outcome = "busy"
        raise
    except InferenceTimeoutError:
        outcome = "timeout"
        raise
    except (ModelServiceInitializationError, ModelServiceUnavailableError):
        outcome = "unavailable"
        raise
    except Exception:
        outcome = "failed"
        raise
    else:
        outcome = "success"
        return result
    finally:
        if outcome is not None:
            request.app.state.runtime.metrics.record_restoration(outcome)
