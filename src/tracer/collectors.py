"""Metric collectors that aggregate data from trace spans."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from src.tracer.spans import Span, SpanKind, SpanStatus


@dataclass
class LatencyDistribution:
    """Statistical distribution of latency values."""

    count: int = 0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    std_dev_ms: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean_ms": round(self.mean_ms, 3),
            "median_ms": round(self.median_ms, 3),
            "p90_ms": round(self.p90_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "std_dev_ms": round(self.std_dev_ms, 3),
        }


@dataclass
class ThroughputMetrics:
    """Request throughput measurements."""

    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    timeouts: int = 0
    requests_per_second: float = 0.0
    error_rate: float = 0.0
    wall_clock_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "successful": self.successful,
            "failed": self.failed,
            "timeouts": self.timeouts,
            "requests_per_second": round(self.requests_per_second, 3),
            "error_rate": round(self.error_rate, 4),
            "wall_clock_seconds": round(self.wall_clock_seconds, 3),
        }


@dataclass
class SpanBreakdown:
    """Timing breakdown by span kind."""

    preprocessing: LatencyDistribution = field(default_factory=LatencyDistribution)
    inference: LatencyDistribution = field(default_factory=LatencyDistribution)
    postprocessing: LatencyDistribution = field(default_factory=LatencyDistribution)
    network: LatencyDistribution = field(default_factory=LatencyDistribution)
    total: LatencyDistribution = field(default_factory=LatencyDistribution)

    def to_dict(self) -> dict[str, Any]:
        return {
            "preprocessing": self.preprocessing.to_dict(),
            "inference": self.inference.to_dict(),
            "postprocessing": self.postprocessing.to_dict(),
            "network": self.network.to_dict(),
            "total": self.total.to_dict(),
        }


class MetricCollector:
    """Aggregates spans into structured performance metrics."""

    def __init__(self) -> None:
        self._spans: list[Span] = []

    def ingest(self, spans: list[Span]) -> None:
        self._spans.extend(spans)

    def reset(self) -> None:
        self._spans.clear()

    @staticmethod
    def _percentile(values: list[float], pct: float) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        k = (len(sorted_vals) - 1) * (pct / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_vals[int(k)]
        return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)

    def _compute_distribution(self, durations: list[float]) -> LatencyDistribution:
        if not durations:
            return LatencyDistribution()
        return LatencyDistribution(
            count=len(durations),
            mean_ms=statistics.mean(durations),
            median_ms=statistics.median(durations),
            p90_ms=self._percentile(durations, 90),
            p95_ms=self._percentile(durations, 95),
            p99_ms=self._percentile(durations, 99),
            min_ms=min(durations),
            max_ms=max(durations),
            std_dev_ms=statistics.stdev(durations) if len(durations) > 1 else 0.0,
        )

    def compute_latency_breakdown(self) -> SpanBreakdown:
        """Break down latency by span kind."""
        by_kind: dict[SpanKind, list[float]] = {}
        for span in self._spans:
            by_kind.setdefault(span.kind, []).append(span.duration_ms)

        breakdown = SpanBreakdown()
        kind_map = {
            SpanKind.PREPROCESSING: "preprocessing",
            SpanKind.INFERENCE: "inference",
            SpanKind.POSTPROCESSING: "postprocessing",
            SpanKind.NETWORK: "network",
        }
        for kind, attr_name in kind_map.items():
            if kind in by_kind:
                setattr(breakdown, attr_name, self._compute_distribution(by_kind[kind]))

        request_spans = by_kind.get(SpanKind.REQUEST, [])
        if request_spans:
            breakdown.total = self._compute_distribution(request_spans)

        return breakdown

    def compute_throughput(self) -> ThroughputMetrics:
        """Compute request throughput metrics."""
        request_spans = [s for s in self._spans if s.kind == SpanKind.REQUEST]
        if not request_spans:
            return ThroughputMetrics()

        total = len(request_spans)
        successful = sum(1 for s in request_spans if s.status == SpanStatus.OK)
        failed = sum(1 for s in request_spans if s.status == SpanStatus.ERROR)
        timeouts = sum(1 for s in request_spans if s.status == SpanStatus.TIMEOUT)

        start_times = [s.start_time for s in request_spans if s.start_time > 0]
        end_times = [s.end_time for s in request_spans if s.end_time > 0]

        wall_clock = 0.0
        rps = 0.0
        if start_times and end_times:
            wall_clock = max(end_times) - min(start_times)
            if wall_clock > 0:
                rps = total / wall_clock

        return ThroughputMetrics(
            total_requests=total,
            successful=successful,
            failed=failed,
            timeouts=timeouts,
            requests_per_second=rps,
            error_rate=failed / total if total > 0 else 0.0,
            wall_clock_seconds=wall_clock,
        )

    def compute_confidence_distribution(self) -> dict[str, Any]:
        """Extract confidence score distribution from span attributes."""
        scores: list[float] = []
        for span in self._spans:
            if "confidence" in span.attributes:
                scores.append(float(span.attributes["confidence"]))

        if not scores:
            return {"count": 0, "buckets": {}}

        buckets = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0,
        }
        for score in scores:
            if score < 0.2:
                buckets["0.0-0.2"] += 1
            elif score < 0.4:
                buckets["0.2-0.4"] += 1
            elif score < 0.6:
                buckets["0.4-0.6"] += 1
            elif score < 0.8:
                buckets["0.6-0.8"] += 1
            else:
                buckets["0.8-1.0"] += 1

        return {
            "count": len(scores),
            "mean": round(statistics.mean(scores), 4),
            "median": round(statistics.median(scores), 4),
            "std_dev": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "buckets": buckets,
        }

    def collect_all(self) -> dict[str, Any]:
        """Collect all metrics into a single structure."""
        return {
            "latency": self.compute_latency_breakdown().to_dict(),
            "throughput": self.compute_throughput().to_dict(),
            "confidence": self.compute_confidence_distribution(),
        }
