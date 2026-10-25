"""Prometheus metrics behind /metrics, scoped to one registry per app instance for test isolation."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)


class Metrics:
    """The handful of series that describe the service's health."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "ledgerlens_http_requests_total", "HTTP requests", ["method", "path", "status"], registry=self.registry
        )
        self.request_latency = Histogram(
            "ledgerlens_http_request_seconds",
            "HTTP request latency",
            ["path"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.documents = Counter(
            "ledgerlens_documents_total", "Documents processed", ["doc_type", "decision"], registry=self.registry
        )
        self.pipeline_latency = Histogram(
            "ledgerlens_pipeline_seconds", "Pipeline latency per page", buckets=LATENCY_BUCKETS, registry=self.registry
        )
        self.queue_depth = Gauge(
            "ledgerlens_review_queue_pending", "Documents waiting for review", registry=self.registry
        )
        self.corrections = Counter(
            "ledgerlens_review_fields_total", "Reviewed fields", ["outcome"], registry=self.registry
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)
