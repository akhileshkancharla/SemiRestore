"""Application-local, bounded-cardinality Prometheus metrics."""

from __future__ import annotations

from typing import Final, Literal

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

HTTP_DURATION_BUCKETS: Final = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)
INFERENCE_DURATION_BUCKETS: Final = (
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
)

HTTP_METHODS: Final = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"})
STATUS_CLASSES: Final = frozenset({"2xx", "3xx", "4xx", "5xx", "cancelled"})
OUTCOMES: Final = frozenset(
    {"success", "busy", "timeout", "unavailable", "failed", "cancelled"}
)

RestorationOutcome = Literal[
    "success", "busy", "timeout", "unavailable", "failed", "cancelled"
]


def bounded_method(method: str) -> str:
    """Return one supported method label or the bounded fallback."""
    normalized = method.upper()
    return normalized if normalized in HTTP_METHODS else "OTHER"


def bounded_route(route: str) -> str:
    """Accept only resolved route templates or the fixed unmatched label."""
    return route if route == "<unmatched>" or route.startswith("/") else "<unmatched>"


class PlatformMetrics:
    """One isolated Prometheus registry and its SemiRestore collectors."""

    def __init__(self, *, inference_capacity: int) -> None:
        self.registry = CollectorRegistry()
        self.http_requests = Counter(
            "semirestore_http_requests_total",
            "Completed SemiRestore HTTP requests.",
            ("method", "route", "status_class"),
            registry=self.registry,
        )
        self.http_request_duration = Histogram(
            "semirestore_http_request_duration_seconds",
            "Total SemiRestore HTTP request duration in seconds.",
            ("method", "route", "status_class"),
            buckets=HTTP_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.restoration_requests = Counter(
            "semirestore_restoration_requests_total",
            "Validated restoration attempts by bounded outcome.",
            ("outcome",),
            registry=self.registry,
        )
        self.inference_duration = Histogram(
            "semirestore_inference_duration_seconds",
            "Platform-observed inference orchestration duration in seconds.",
            ("outcome",),
            buckets=INFERENCE_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.inference_active = Gauge(
            "semirestore_inference_active",
            "Inference operations currently holding capacity.",
            registry=self.registry,
        )
        self.inference_waiting = Gauge(
            "semirestore_inference_waiting",
            "Inference operations currently waiting to acquire capacity.",
            registry=self.registry,
        )
        self.inference_capacity = Gauge(
            "semirestore_inference_capacity",
            "Configured application-local inference capacity.",
            registry=self.registry,
        )
        self.inference_busy = Counter(
            "semirestore_inference_busy_total",
            "Inference operations rejected while acquiring capacity.",
            registry=self.registry,
        )
        self.inference_timeouts = Counter(
            "semirestore_inference_timeouts_total",
            "Inference operations that exceeded the execution timeout.",
            registry=self.registry,
        )
        self.inference_capacity.set(inference_capacity)

    def observe_http(
        self,
        *,
        method: str,
        route: str,
        status_class: str,
        duration_seconds: float,
    ) -> None:
        """Record one completed request using only bounded labels."""
        labels = {
            "method": bounded_method(method),
            "route": bounded_route(route),
            "status_class": (
                status_class if status_class in STATUS_CLASSES else "5xx"
            ),
        }
        self.http_requests.labels(**labels).inc()
        self.http_request_duration.labels(**labels).observe(max(0.0, duration_seconds))

    def record_restoration(self, outcome: RestorationOutcome) -> None:
        """Record exactly one terminal outcome for a validated restoration."""
        self.restoration_requests.labels(outcome=self._bounded_outcome(outcome)).inc()

    def observe_inference(
        self,
        *,
        outcome: RestorationOutcome,
        duration_seconds: float,
    ) -> None:
        """Record one platform-observed inference interval."""
        self.inference_duration.labels(outcome=self._bounded_outcome(outcome)).observe(
            max(0.0, duration_seconds)
        )

    @staticmethod
    def _bounded_outcome(outcome: str) -> str:
        return outcome if outcome in OUTCOMES else "failed"
