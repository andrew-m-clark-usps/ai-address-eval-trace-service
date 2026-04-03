"""Span definitions for structured trace capture."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SpanKind(Enum):
    """Classification of trace spans."""

    REQUEST = "request"
    PREPROCESSING = "preprocessing"
    INFERENCE = "inference"
    POSTPROCESSING = "postprocessing"
    VALIDATION = "validation"
    NETWORK = "network"


class SpanStatus(Enum):
    """Span completion status."""

    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class SpanEvent:
    """Discrete event within a span."""

    name: str
    timestamp: float = field(default_factory=time.time)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """A single unit of traced work.

    Captures timing, metadata, and nested structure for any operation
    within the model evaluation pipeline.
    """

    name: str
    kind: SpanKind
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: str | None = None
    trace_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    status: SpanStatus = SpanStatus.OK
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    error_message: str = ""

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        if self.end_time <= 0 or self.start_time <= 0:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    def start(self) -> Span:
        self.start_time = time.time()
        return self

    def finish(self, status: SpanStatus = SpanStatus.OK, error: str = "") -> Span:
        self.end_time = time.time()
        self.status = status
        self.error_message = error
        return self

    def add_event(self, name: str, **attrs: Any) -> None:
        self.events.append(SpanEvent(name=name, attributes=attrs))

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "trace_id": self.trace_id,
            "name": self.name,
            "kind": self.kind.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 3),
            "status": self.status.value,
            "error_message": self.error_message,
            "attributes": self.attributes,
            "events": [
                {
                    "name": e.name,
                    "timestamp": e.timestamp,
                    "attributes": e.attributes,
                }
                for e in self.events
            ],
        }
