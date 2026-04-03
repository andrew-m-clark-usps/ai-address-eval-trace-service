"""Core tracing engine for model evaluation."""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

import requests

from src.config import ModelEndpointConfig
from src.tracer.collectors import MetricCollector
from src.tracer.spans import Span, SpanKind, SpanStatus

logger = logging.getLogger(__name__)


@dataclass
class TraceRecord:
    """A complete trace of a single address verification request."""

    trace_id: str
    input_address: dict[str, Any]
    output_result: dict[str, Any] | None = None
    spans: list[Span] = field(default_factory=list)
    error: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "input_address": self.input_address,
            "output_result": self.output_result,
            "spans": [s.to_dict() for s in self.spans],
            "error": self.error,
            "timestamp": self.timestamp,
        }


class Tracer:
    """Captures detailed traces of model inference requests.

    Wraps calls to the external model endpoint, capturing timing data,
    input/output payloads, and intermediate processing steps.
    """

    def __init__(self, model_config: ModelEndpointConfig) -> None:
        self._config = model_config
        self._session = requests.Session()
        if model_config.headers:
            self._session.headers.update(model_config.headers)
        self._traces: list[TraceRecord] = []
        self._collector = MetricCollector()

    @property
    def traces(self) -> list[TraceRecord]:
        return list(self._traces)

    @property
    def collector(self) -> MetricCollector:
        return self._collector

    @contextmanager
    def _span(
        self, name: str, kind: SpanKind, trace_id: str, parent_id: str | None = None
    ) -> Generator[Span, None, None]:
        span = Span(name=name, kind=kind, trace_id=trace_id, parent_id=parent_id)
        span.start()
        try:
            yield span
        except Exception as exc:
            span.finish(status=SpanStatus.ERROR, error=str(exc))
            raise
        else:
            span.finish()

    def trace_request(self, address: dict[str, Any]) -> TraceRecord:
        """Execute a traced verification request against the model.

        Captures spans for: preprocessing, network call, inference (server-side),
        postprocessing, and the overall request.
        """
        trace_id = uuid.uuid4().hex[:24]
        record = TraceRecord(trace_id=trace_id, input_address=address)
        all_spans: list[Span] = []

        request_span = Span(name="verify_address", kind=SpanKind.REQUEST, trace_id=trace_id)
        request_span.start()

        # Preprocessing: normalize input
        with self._span("normalize_input", SpanKind.PREPROCESSING, trace_id) as pre_span:
            payload = self._build_payload(address)
            pre_span.set_attribute("input_fields", list(address.keys()))
            pre_span.set_attribute("payload_bytes", len(json.dumps(payload)))
            all_spans.append(pre_span)

        # Network + Inference
        with self._span(
            "model_call", SpanKind.NETWORK, trace_id, parent_id=request_span.span_id
        ) as net_span:
            try:
                response = self._session.post(
                    self._config.verify_url,
                    json=payload,
                    timeout=self._config.timeout_seconds,
                )
                net_span.set_attribute("http_status", response.status_code)
                net_span.set_attribute("response_bytes", len(response.content))

                if response.status_code == 200:
                    result = response.json()
                else:
                    result = {"error": f"HTTP {response.status_code}", "body": response.text[:500]}
                    net_span.finish(
                        status=SpanStatus.ERROR, error=f"HTTP {response.status_code}"
                    )

            except requests.Timeout:
                result = {"error": "timeout"}
                net_span.finish(status=SpanStatus.TIMEOUT, error="Request timed out")
                record.error = "timeout"

            except requests.ConnectionError:
                result = {"error": "connection_failed"}
                net_span.finish(status=SpanStatus.ERROR, error="Connection failed")
                record.error = "connection_failed"

            except requests.RequestException as exc:
                result = {"error": str(exc)}
                net_span.finish(status=SpanStatus.ERROR, error=str(exc))
                record.error = str(exc)

            all_spans.append(net_span)

        # Postprocessing: extract and structure result
        with self._span("process_response", SpanKind.POSTPROCESSING, trace_id) as post_span:
            structured = self._structure_result(result)
            post_span.set_attribute("result_keys", list(structured.keys()))
            if "confidence" in structured:
                post_span.set_attribute("confidence", structured["confidence"])
                request_span.set_attribute("confidence", structured["confidence"])
            all_spans.append(post_span)

        # Finalize request span
        if record.error:
            request_span.finish(status=SpanStatus.ERROR, error=record.error)
        else:
            request_span.finish()

        request_span.set_attribute("input_address", address)
        request_span.set_attribute("output_result", structured)
        all_spans.insert(0, request_span)

        record.output_result = structured
        record.spans = all_spans

        self._traces.append(record)
        self._collector.ingest(all_spans)

        return record

    def trace_batch(self, addresses: list[dict[str, Any]]) -> list[TraceRecord]:
        """Trace a batch of address verification requests sequentially."""
        records = []
        for addr in addresses:
            record = self.trace_request(addr)
            records.append(record)
        return records

    def check_health(self) -> dict[str, Any]:
        """Check if the model endpoint is reachable."""
        try:
            resp = self._session.get(
                self._config.health_url, timeout=self._config.timeout_seconds
            )
            return {"status": "healthy" if resp.status_code == 200 else "unhealthy",
                    "http_status": resp.status_code}
        except requests.RequestException as exc:
            return {"status": "unreachable", "error": str(exc)}

    def get_metrics(self) -> dict[str, Any]:
        """Return aggregated metrics from all collected spans."""
        return self._collector.collect_all()

    def export_traces(self, path: str | Path) -> None:
        """Export all trace records to a JSON file."""
        data = [t.to_dict() for t in self._traces]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def reset(self) -> None:
        """Clear all traces and metrics."""
        self._traces.clear()
        self._collector.reset()

    @staticmethod
    def _build_payload(address: dict[str, Any]) -> dict[str, Any]:
        """Normalize address into model-expected payload format."""
        return {
            "address": {
                "street": address.get("street", ""),
                "city": address.get("city", ""),
                "state": address.get("state", ""),
                "zip": address.get("zip", ""),
                "unit": address.get("unit", ""),
            }
        }

    @staticmethod
    def _structure_result(raw: dict[str, Any]) -> dict[str, Any]:
        """Extract structured fields from model response."""
        if "error" in raw:
            return raw
        return {
            "verified": raw.get("verified", False),
            "confidence": raw.get("confidence", 0.0),
            "standardized": raw.get("standardized_address", raw.get("standardized", {})),
            "components": raw.get("components", {}),
            "corrections": raw.get("corrections", []),
            "metadata": raw.get("metadata", {}),
        }
