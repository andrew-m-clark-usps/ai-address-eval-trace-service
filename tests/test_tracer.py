"""Comprehensive tests for the AI address eval-trace service."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.config import ModelEndpointConfig, ServiceConfig
from src.dashboard.generator import DashboardGenerator
from src.evaluator.datasets import EvalCase, EvalDataset
from src.evaluator.metrics import ComponentScore, EvaluationResult, MetricsCalculator
from src.evaluator.runner import EvaluationRunner
from src.tracer.collectors import (
    LatencyDistribution,
    MetricCollector,
    SpanBreakdown,
    ThroughputMetrics,
)
from src.tracer.core import Tracer, TraceRecord
from src.tracer.spans import Span, SpanEvent, SpanKind, SpanStatus

# ---------------------------------------------------------------------------
# src.tracer.spans
# ---------------------------------------------------------------------------


class TestSpanKind:
    def test_enum_values(self):
        assert SpanKind.REQUEST.value == "request"
        assert SpanKind.PREPROCESSING.value == "preprocessing"
        assert SpanKind.INFERENCE.value == "inference"
        assert SpanKind.POSTPROCESSING.value == "postprocessing"
        assert SpanKind.VALIDATION.value == "validation"
        assert SpanKind.NETWORK.value == "network"

    def test_all_members(self):
        assert len(SpanKind) == 6


class TestSpanStatus:
    def test_enum_values(self):
        assert SpanStatus.OK.value == "ok"
        assert SpanStatus.ERROR.value == "error"
        assert SpanStatus.TIMEOUT.value == "timeout"

    def test_all_members(self):
        assert len(SpanStatus) == 3


class TestSpanEvent:
    def test_basic_construction(self):
        evt = SpanEvent(name="cache_hit")
        assert evt.name == "cache_hit"
        assert evt.timestamp > 0
        assert evt.attributes == {}

    def test_with_attributes(self):
        evt = SpanEvent(name="lookup", attributes={"key": "val"})
        assert evt.attributes["key"] == "val"


class TestSpan:
    def test_defaults(self):
        span = Span(name="op", kind=SpanKind.REQUEST)
        assert span.name == "op"
        assert span.kind == SpanKind.REQUEST
        assert len(span.span_id) == 16
        assert span.parent_id is None
        assert span.trace_id == ""
        assert span.start_time == 0.0
        assert span.end_time == 0.0
        assert span.status == SpanStatus.OK
        assert span.attributes == {}
        assert span.events == []
        assert span.error_message == ""

    def test_duration_ms_zero_when_not_started(self):
        span = Span(name="x", kind=SpanKind.INFERENCE)
        assert span.duration_ms == 0.0

    def test_duration_ms_zero_when_only_started(self):
        span = Span(name="x", kind=SpanKind.INFERENCE)
        span.start()
        assert span.duration_ms == 0.0

    def test_start_finish_timing(self):
        span = Span(name="work", kind=SpanKind.NETWORK)
        span.start()
        time.sleep(0.02)
        span.finish()
        assert span.start_time > 0
        assert span.end_time >= span.start_time
        assert span.duration_ms >= 15  # at least ~20ms minus tolerance
        assert span.status == SpanStatus.OK

    def test_finish_with_error(self):
        span = Span(name="fail", kind=SpanKind.INFERENCE)
        span.start()
        span.finish(status=SpanStatus.ERROR, error="boom")
        assert span.status == SpanStatus.ERROR
        assert span.error_message == "boom"

    def test_finish_with_timeout(self):
        span = Span(name="slow", kind=SpanKind.NETWORK)
        span.start()
        span.finish(status=SpanStatus.TIMEOUT, error="timed out")
        assert span.status == SpanStatus.TIMEOUT

    def test_start_returns_self(self):
        span = Span(name="s", kind=SpanKind.REQUEST)
        assert span.start() is span

    def test_finish_returns_self(self):
        span = Span(name="s", kind=SpanKind.REQUEST).start()
        assert span.finish() is span

    def test_add_event(self):
        span = Span(name="op", kind=SpanKind.PREPROCESSING)
        span.add_event("step1", count=5, label="a")
        assert len(span.events) == 1
        assert span.events[0].name == "step1"
        assert span.events[0].attributes == {"count": 5, "label": "a"}

    def test_add_multiple_events(self):
        span = Span(name="op", kind=SpanKind.PREPROCESSING)
        span.add_event("a")
        span.add_event("b")
        span.add_event("c")
        assert len(span.events) == 3

    def test_set_attribute(self):
        span = Span(name="op", kind=SpanKind.POSTPROCESSING)
        span.set_attribute("key", "value")
        assert span.attributes["key"] == "value"

    def test_set_attribute_overwrite(self):
        span = Span(name="op", kind=SpanKind.POSTPROCESSING)
        span.set_attribute("k", 1)
        span.set_attribute("k", 2)
        assert span.attributes["k"] == 2

    def test_to_dict_structure(self):
        span = Span(
            name="test",
            kind=SpanKind.INFERENCE,
            span_id="abc123",
            parent_id="parent1",
            trace_id="trace1",
            start_time=1000.0,
            end_time=1000.05,
            status=SpanStatus.OK,
            attributes={"model": "v2"},
        )
        span.add_event("ev", flag=True)
        d = span.to_dict()

        assert d["span_id"] == "abc123"
        assert d["parent_id"] == "parent1"
        assert d["trace_id"] == "trace1"
        assert d["name"] == "test"
        assert d["kind"] == "inference"
        assert d["start_time"] == 1000.0
        assert d["end_time"] == 1000.05
        assert d["duration_ms"] == 50.0
        assert d["status"] == "ok"
        assert d["attributes"]["model"] == "v2"
        assert len(d["events"]) == 1
        assert d["events"][0]["name"] == "ev"
        assert d["events"][0]["attributes"] == {"flag": True}

    def test_to_dict_duration_rounded(self):
        span = Span(
            name="x",
            kind=SpanKind.REQUEST,
            start_time=1.0,
            end_time=1.0001234,
        )
        d = span.to_dict()
        assert d["duration_ms"] == round(0.1234, 3)

    def test_unique_span_ids(self):
        ids = {Span(name="s", kind=SpanKind.REQUEST).span_id for _ in range(100)}
        assert len(ids) == 100

    def test_manual_times(self):
        span = Span(
            name="manual",
            kind=SpanKind.REQUEST,
            start_time=100.0,
            end_time=100.5,
        )
        assert span.duration_ms == 500.0


# ---------------------------------------------------------------------------
# src.tracer.collectors
# ---------------------------------------------------------------------------


class TestLatencyDistribution:
    def test_defaults(self):
        ld = LatencyDistribution()
        assert ld.count == 0
        assert ld.mean_ms == 0.0

    def test_to_dict_rounding(self):
        ld = LatencyDistribution(
            count=5,
            mean_ms=1.23456,
            median_ms=1.5,
            p90_ms=2.0,
            p95_ms=2.5,
            p99_ms=3.0,
            min_ms=0.5,
            max_ms=4.0,
            std_dev_ms=0.456789,
        )
        d = ld.to_dict()
        assert d["count"] == 5
        assert d["mean_ms"] == 1.235
        assert d["std_dev_ms"] == 0.457


class TestThroughputMetrics:
    def test_defaults(self):
        t = ThroughputMetrics()
        assert t.total_requests == 0
        assert t.error_rate == 0.0

    def test_to_dict(self):
        t = ThroughputMetrics(
            total_requests=100,
            successful=90,
            failed=8,
            timeouts=2,
            requests_per_second=33.333,
            error_rate=0.08,
            wall_clock_seconds=3.0,
        )
        d = t.to_dict()
        assert d["total_requests"] == 100
        assert d["requests_per_second"] == 33.333
        assert d["error_rate"] == 0.08


class TestSpanBreakdown:
    def test_defaults_are_empty_distributions(self):
        sb = SpanBreakdown()
        assert sb.preprocessing.count == 0
        assert sb.inference.count == 0
        assert sb.total.count == 0

    def test_to_dict_keys(self):
        d = SpanBreakdown().to_dict()
        assert set(d.keys()) == {
            "preprocessing",
            "inference",
            "postprocessing",
            "network",
            "total",
        }


def _make_span(kind, start, end, status=SpanStatus.OK, **attrs):
    return Span(
        name="s",
        kind=kind,
        start_time=start,
        end_time=end,
        status=status,
        attributes=attrs,
    )


class TestMetricCollector:
    def test_empty_collector(self):
        mc = MetricCollector()
        assert mc.collect_all()["throughput"]["total_requests"] == 0

    def test_ingest_and_reset(self):
        mc = MetricCollector()
        spans = [_make_span(SpanKind.REQUEST, 1.0, 1.05)]
        mc.ingest(spans)
        assert mc.compute_throughput().total_requests == 1
        mc.reset()
        assert mc.compute_throughput().total_requests == 0

    def test_latency_breakdown_single_kind(self):
        mc = MetricCollector()
        mc.ingest([
            _make_span(SpanKind.PREPROCESSING, 1.0, 1.01),
            _make_span(SpanKind.PREPROCESSING, 2.0, 2.02),
        ])
        bd = mc.compute_latency_breakdown()
        assert bd.preprocessing.count == 2
        assert bd.preprocessing.mean_ms == pytest.approx(15.0, abs=1)
        assert bd.inference.count == 0

    def test_latency_breakdown_request_goes_to_total(self):
        mc = MetricCollector()
        mc.ingest([_make_span(SpanKind.REQUEST, 1.0, 1.1)])
        bd = mc.compute_latency_breakdown()
        assert bd.total.count == 1
        assert bd.total.mean_ms == pytest.approx(100.0, abs=1)

    def test_throughput_counts(self):
        mc = MetricCollector()
        mc.ingest([
            _make_span(SpanKind.REQUEST, 1.0, 1.1, SpanStatus.OK),
            _make_span(SpanKind.REQUEST, 1.0, 1.2, SpanStatus.ERROR),
            _make_span(SpanKind.REQUEST, 1.0, 1.3, SpanStatus.TIMEOUT),
        ])
        t = mc.compute_throughput()
        assert t.total_requests == 3
        assert t.successful == 1
        assert t.failed == 1
        assert t.timeouts == 1
        assert t.error_rate == pytest.approx(1 / 3, abs=0.01)
        assert t.wall_clock_seconds == pytest.approx(0.3, abs=0.01)
        assert t.requests_per_second > 0

    def test_throughput_empty(self):
        mc = MetricCollector()
        mc.ingest([_make_span(SpanKind.PREPROCESSING, 1.0, 1.1)])
        t = mc.compute_throughput()
        assert t.total_requests == 0

    def test_confidence_distribution_empty(self):
        mc = MetricCollector()
        d = mc.compute_confidence_distribution()
        assert d["count"] == 0
        assert d["buckets"] == {}

    def test_confidence_distribution_buckets(self):
        mc = MetricCollector()
        spans = [
            _make_span(SpanKind.INFERENCE, 1, 2, confidence=0.1),
            _make_span(SpanKind.INFERENCE, 1, 2, confidence=0.3),
            _make_span(SpanKind.INFERENCE, 1, 2, confidence=0.5),
            _make_span(SpanKind.INFERENCE, 1, 2, confidence=0.7),
            _make_span(SpanKind.INFERENCE, 1, 2, confidence=0.9),
        ]
        mc.ingest(spans)
        d = mc.compute_confidence_distribution()
        assert d["count"] == 5
        assert d["buckets"]["0.0-0.2"] == 1
        assert d["buckets"]["0.2-0.4"] == 1
        assert d["buckets"]["0.4-0.6"] == 1
        assert d["buckets"]["0.6-0.8"] == 1
        assert d["buckets"]["0.8-1.0"] == 1
        assert d["min"] == 0.1
        assert d["max"] == 0.9

    def test_confidence_distribution_single_value(self):
        mc = MetricCollector()
        mc.ingest([_make_span(SpanKind.REQUEST, 1, 2, confidence=0.95)])
        d = mc.compute_confidence_distribution()
        assert d["count"] == 1
        assert d["std_dev"] == 0.0

    def test_collect_all_keys(self):
        mc = MetricCollector()
        result = mc.collect_all()
        assert set(result.keys()) == {"latency", "throughput", "confidence"}

    def test_percentile_single_value(self):
        assert MetricCollector._percentile([42.0], 50) == 42.0
        assert MetricCollector._percentile([42.0], 99) == 42.0

    def test_percentile_empty(self):
        assert MetricCollector._percentile([], 50) == 0.0

    def test_percentile_multiple(self):
        vals = list(range(1, 101))  # 1..100
        assert MetricCollector._percentile(vals, 50) == pytest.approx(50.5, abs=0.5)
        assert MetricCollector._percentile(vals, 99) == pytest.approx(99.01, abs=0.5)


# ---------------------------------------------------------------------------
# src.evaluator.datasets
# ---------------------------------------------------------------------------


class TestEvalCase:
    def test_construction(self):
        case = EvalCase(
            case_id="t1",
            input_address={"street": "123 Main"},
            expected_output={"verified": True},
        )
        assert case.case_id == "t1"
        assert case.category == "general"
        assert case.description == ""

    def test_to_dict(self):
        case = EvalCase(
            case_id="t2",
            input_address={"street": "a"},
            expected_output={"verified": False},
            category="test",
            description="desc",
        )
        d = case.to_dict()
        assert d["case_id"] == "t2"
        assert d["category"] == "test"
        assert d["description"] == "desc"


class TestEvalDataset:
    def test_construction(self):
        ds = EvalDataset(name="ds", version="1.0")
        assert ds.name == "ds"
        assert ds.cases == []

    def test_to_dict_empty(self):
        ds = EvalDataset(name="empty", version="0.1")
        d = ds.to_dict()
        assert d["total_cases"] == 0
        assert d["categories"] == []
        assert d["cases"] == []

    def test_to_dict_with_cases(self):
        ds = EvalDataset(
            name="ds",
            version="1.0",
            cases=[
                EvalCase("c1", {"s": "1"}, {"v": True}, category="a"),
                EvalCase("c2", {"s": "2"}, {"v": False}, category="b"),
                EvalCase("c3", {"s": "3"}, {"v": True}, category="a"),
            ],
        )
        d = ds.to_dict()
        assert d["total_cases"] == 3
        assert set(d["categories"]) == {"a", "b"}

    def test_from_file(self, tmp_path):
        data = {
            "name": "file-ds",
            "version": "2.0",
            "cases": [
                {
                    "case_id": "f1",
                    "input_address": {"street": "100 Main"},
                    "expected_output": {"verified": True},
                    "category": "cat1",
                    "description": "test case",
                },
                {
                    "case_id": "f2",
                    "input_address": {"street": "200 Oak"},
                    "expected_output": {"verified": False},
                },
            ],
        }
        fp = tmp_path / "dataset.json"
        fp.write_text(json.dumps(data))

        ds = EvalDataset.from_file(fp)
        assert ds.name == "file-ds"
        assert ds.version == "2.0"
        assert len(ds.cases) == 2
        assert ds.cases[0].case_id == "f1"
        assert ds.cases[0].category == "cat1"
        assert ds.cases[1].category == "general"
        assert ds.cases[1].description == ""

    def test_from_file_defaults(self, tmp_path):
        fp = tmp_path / "minimal.json"
        fp.write_text(json.dumps({"cases": []}))
        ds = EvalDataset.from_file(fp)
        assert ds.name == "unnamed"
        assert ds.version == "1.0"

    def test_default_dataset(self):
        ds = EvalDataset.default_dataset()
        assert ds.name == "default-address-eval"
        assert ds.version == "1.0"
        assert len(ds.cases) == 16

    def test_default_dataset_categories(self):
        ds = EvalDataset.default_dataset()
        categories = {c.category for c in ds.cases}
        expected = {
            "standard",
            "abbreviation",
            "error_correction",
            "missing_data",
            "po_box",
            "rural",
            "military",
            "zip_plus_4",
            "unit",
            "invalid",
        }
        assert categories == expected

    def test_default_dataset_case_ids_unique(self):
        ds = EvalDataset.default_dataset()
        ids = [c.case_id for c in ds.cases]
        assert len(ids) == len(set(ids))

    def test_roundtrip_file(self, tmp_path):
        ds = EvalDataset.default_dataset()
        fp = tmp_path / "roundtrip.json"
        fp.write_text(json.dumps(ds.to_dict()))
        loaded = EvalDataset.from_file(fp)
        assert len(loaded.cases) == len(ds.cases)
        assert loaded.name == ds.name


# ---------------------------------------------------------------------------
# src.evaluator.metrics
# ---------------------------------------------------------------------------


class TestComponentScore:
    def test_defaults(self):
        cs = ComponentScore(component="street")
        assert cs.accuracy == 0.0
        assert cs.error_rate == 0.0

    def test_accuracy(self):
        cs = ComponentScore(component="city", correct=8, incorrect=2, total=10)
        assert cs.accuracy == pytest.approx(0.8)
        assert cs.error_rate == pytest.approx(0.2)

    def test_zero_total(self):
        cs = ComponentScore(component="zip", total=0)
        assert cs.accuracy == 0.0
        assert cs.error_rate == 0.0

    def test_to_dict(self):
        cs = ComponentScore(
            component="state", correct=3, incorrect=1, missing=0, total=4
        )
        d = cs.to_dict()
        assert d["component"] == "state"
        assert d["accuracy"] == 0.75
        assert d["error_rate"] == 0.25


class TestEvaluationResult:
    def test_defaults(self):
        r = EvaluationResult(
            case_id="c1",
            input_address={"street": "1"},
            expected={"verified": True},
            actual=None,
        )
        assert r.passed is False
        assert r.exact_match is False
        assert r.confidence == 0.0
        assert r.latency_ms == 0.0

    def test_to_dict(self):
        r = EvaluationResult(
            case_id="c2",
            input_address={"street": "x"},
            expected={"verified": True},
            actual={"verified": True},
            passed=True,
            exact_match=True,
            component_matches={"street": True},
            confidence=0.95,
            latency_ms=12.345,
            category="standard",
        )
        d = r.to_dict()
        assert d["case_id"] == "c2"
        assert d["passed"] is True
        assert d["confidence"] == 0.95
        assert d["latency_ms"] == 12.345


def _make_results(passed_flags, exact_flags=None, categories=None):
    """Helper to create a batch of EvaluationResult objects."""
    exact_flags = exact_flags or passed_flags
    categories = categories or ["general"] * len(passed_flags)
    results = []
    for i, (p, e, cat) in enumerate(zip(passed_flags, exact_flags, categories)):
        results.append(
            EvaluationResult(
                case_id=f"r{i}",
                input_address={"street": str(i)},
                expected={"verified": True},
                actual={"verified": True} if p else {"verified": False},
                passed=p,
                exact_match=e,
                confidence=0.9 if p else 0.3,
                category=cat,
                component_matches={"street": p, "city": p},
            )
        )
    return results


class TestMetricsCalculator:
    def test_empty(self):
        mc = MetricsCalculator()
        assert mc.total == 0
        assert mc.overall_accuracy() == 0.0
        assert mc.exact_match_rate() == 0.0
        assert mc.error_rate() == 0.0

    def test_all_passed(self):
        results = _make_results([True, True, True])
        mc = MetricsCalculator(results)
        assert mc.total == 3
        assert mc.overall_accuracy() == 1.0
        assert mc.exact_match_rate() == 1.0

    def test_mixed_results(self):
        results = _make_results(
            [True, True, False, False],
            [True, False, False, False],
        )
        mc = MetricsCalculator(results)
        assert mc.overall_accuracy() == 0.5
        assert mc.exact_match_rate() == 0.25

    def test_error_rate(self):
        results = [
            EvaluationResult(
                case_id="e1",
                input_address={},
                expected={},
                actual={},
                error="fail",
            ),
            EvaluationResult(
                case_id="e2",
                input_address={},
                expected={},
                actual={},
                error="",
            ),
        ]
        mc = MetricsCalculator(results)
        assert mc.error_rate() == 0.5

    def test_add_result(self):
        mc = MetricsCalculator()
        mc.add_result(
            EvaluationResult(
                case_id="a1",
                input_address={},
                expected={},
                actual={},
                passed=True,
            )
        )
        assert mc.total == 1

    def test_add_results(self):
        mc = MetricsCalculator()
        mc.add_results(_make_results([True, False]))
        assert mc.total == 2

    def test_component_scores(self):
        results = _make_results([True, True, False])
        mc = MetricsCalculator(results)
        scores = mc.component_scores()
        assert "street" in scores
        assert scores["street"].correct == 2
        assert scores["street"].incorrect == 1
        assert scores["street"].total == 3

    def test_category_breakdown(self):
        results = _make_results(
            [True, False, True],
            categories=["a", "a", "b"],
        )
        mc = MetricsCalculator(results)
        bd = mc.category_breakdown()
        assert bd["a"]["total"] == 2
        assert bd["a"]["passed"] == 1
        assert bd["a"]["accuracy"] == 0.5
        assert bd["b"]["total"] == 1
        assert bd["b"]["passed"] == 1

    def test_category_breakdown_uncategorized(self):
        r = EvaluationResult(
            case_id="u",
            input_address={},
            expected={},
            actual={},
            passed=True,
            category="",
        )
        mc = MetricsCalculator([r])
        bd = mc.category_breakdown()
        assert "uncategorized" in bd

    def test_confidence_correlation(self):
        results = _make_results([True, True, False])
        mc = MetricsCalculator(results)
        cc = mc.confidence_correlation()
        assert cc["correct_samples"] == 2
        assert cc["incorrect_samples"] == 1
        assert cc["correct_mean_confidence"] == 0.9
        assert cc["incorrect_mean_confidence"] == 0.3

    def test_confidence_correlation_no_confidence(self):
        r = EvaluationResult(
            case_id="z",
            input_address={},
            expected={},
            actual={},
            passed=True,
            confidence=0.0,
        )
        mc = MetricsCalculator([r])
        cc = mc.confidence_correlation()
        assert cc["correct_samples"] == 0

    def test_compute_all_structure(self):
        results = _make_results([True, False])
        mc = MetricsCalculator(results)
        data = mc.compute_all()
        assert "summary" in data
        assert "component_scores" in data
        assert "category_breakdown" in data
        assert "confidence_correlation" in data
        assert "results" in data
        assert data["summary"]["total_cases"] == 2


# ---------------------------------------------------------------------------
# src.evaluator.runner (static methods + simulation)
# ---------------------------------------------------------------------------


class TestEvaluationRunnerStaticMethods:
    def test_check_passed_verified_match(self):
        assert EvaluationRunner._check_passed(
            {"verified": True}, {"verified": True}
        )

    def test_check_passed_verified_mismatch(self):
        assert not EvaluationRunner._check_passed(
            {"verified": True}, {"verified": False}
        )

    def test_check_passed_error_in_actual(self):
        assert not EvaluationRunner._check_passed(
            {"verified": True}, {"error": "fail", "verified": True}
        )

    def test_check_passed_error_expected_and_actual(self):
        assert EvaluationRunner._check_passed(
            {"error": "expected"}, {"error": "got"}
        )

    def test_check_passed_no_verified_key(self):
        assert EvaluationRunner._check_passed({}, {"something": 1})

    def test_check_exact_match_both_empty(self):
        assert EvaluationRunner._check_exact_match(
            {"standardized": {}}, {"standardized": {}}
        )

    def test_check_exact_match_no_standardized(self):
        assert EvaluationRunner._check_exact_match({}, {})

    def test_check_exact_match_equal(self):
        std = {"street": "123 MAIN ST", "city": "SPRINGFIELD"}
        assert EvaluationRunner._check_exact_match(
            {"standardized": std}, {"standardized": std}
        )

    def test_check_exact_match_not_equal(self):
        assert not EvaluationRunner._check_exact_match(
            {"standardized": {"street": "A"}},
            {"standardized": {"street": "B"}},
        )

    def test_check_components_empty_expected(self):
        assert EvaluationRunner._check_components({}, {"standardized": {"street": "X"}}) == {}

    def test_check_components_match(self):
        expected = {
            "standardized": {
                "street": "123 MAIN ST",
                "city": "NYC",
                "state": "NY",
                "zip": "10001",
            }
        }
        actual = {
            "standardized": {
                "street": "123 MAIN ST",
                "city": "NYC",
                "state": "NY",
                "zip": "10001",
            }
        }
        matches = EvaluationRunner._check_components(expected, actual)
        assert all(matches.values())
        assert set(matches.keys()) == {"street", "city", "state", "zip"}

    def test_check_components_partial_match(self):
        expected = {"standardized": {"street": "A", "city": "B", "state": "C", "zip": "D"}}
        actual = {"standardized": {"street": "A", "city": "X", "state": "C", "zip": "D"}}
        matches = EvaluationRunner._check_components(expected, actual)
        assert matches["street"] is True
        assert matches["city"] is False
        assert matches["state"] is True

    def test_check_components_missing_actual(self):
        expected = {"standardized": {"street": "A", "city": "B"}}
        actual = {"standardized": {}}
        matches = EvaluationRunner._check_components(expected, actual)
        assert matches["street"] is False
        assert matches["city"] is False


class TestEvaluationRunnerSimulation:
    def test_run_with_simulated_results(self):
        config = ModelEndpointConfig()
        tracer = Tracer(config)
        runner = EvaluationRunner(tracer)
        dataset = EvalDataset.default_dataset()

        metrics = runner.run_with_simulated_results(dataset)

        assert metrics["simulated"] is True
        assert metrics["dataset"]["total_cases"] == 16
        assert "summary" in metrics
        assert "component_scores" in metrics
        assert metrics["summary"]["total_cases"] == 16
        assert 0.0 <= metrics["summary"]["overall_accuracy"] <= 1.0
        assert 0.0 <= metrics["summary"]["exact_match_rate"] <= 1.0
        assert 0.0 <= metrics["summary"]["error_rate"] <= 1.0
        assert len(metrics["results"]) == 16

    def test_simulation_deterministic(self):
        """Same seed should yield identical results."""
        config = ModelEndpointConfig()

        tracer1 = Tracer(config)
        runner1 = EvaluationRunner(tracer1)
        m1 = runner1.run_with_simulated_results(EvalDataset.default_dataset())

        tracer2 = Tracer(config)
        runner2 = EvaluationRunner(tracer2)
        m2 = runner2.run_with_simulated_results(EvalDataset.default_dataset())

        assert m1["summary"]["overall_accuracy"] == m2["summary"]["overall_accuracy"]
        assert m1["summary"]["exact_match_rate"] == m2["summary"]["exact_match_rate"]
        for r1, r2 in zip(m1["results"], m2["results"]):
            assert r1["passed"] == r2["passed"]
            assert r1["exact_match"] == r2["exact_match"]
            assert r1["confidence"] == r2["confidence"]

    def test_simulation_produces_traces(self):
        config = ModelEndpointConfig()
        tracer = Tracer(config)
        runner = EvaluationRunner(tracer)
        runner.run_with_simulated_results(EvalDataset.default_dataset())

        assert len(tracer.traces) == 16
        for trace in tracer.traces:
            assert len(trace.spans) == 4
            kinds = {s.kind for s in trace.spans}
            expected_kinds = {
                SpanKind.REQUEST,
                SpanKind.PREPROCESSING,
                SpanKind.NETWORK,
                SpanKind.POSTPROCESSING,
            }
            assert kinds == expected_kinds

    def test_simulation_trace_metrics(self):
        config = ModelEndpointConfig()
        tracer = Tracer(config)
        runner = EvaluationRunner(tracer)
        runner.run_with_simulated_results(EvalDataset.default_dataset())

        metrics = tracer.get_metrics()
        assert metrics["throughput"]["total_requests"] == 16
        assert metrics["confidence"]["count"] > 0

    def test_runner_results_property(self):
        config = ModelEndpointConfig()
        tracer = Tracer(config)
        runner = EvaluationRunner(tracer)
        runner.run_with_simulated_results(EvalDataset.default_dataset())

        assert len(runner.results) == 16
        assert runner.results is not runner._results  # defensive copy

    def test_runner_calculator_property(self):
        config = ModelEndpointConfig()
        tracer = Tracer(config)
        runner = EvaluationRunner(tracer)
        runner.run_with_simulated_results(EvalDataset.default_dataset())

        calc = runner.calculator
        assert calc.total == 16


# ---------------------------------------------------------------------------
# src.dashboard.generator
# ---------------------------------------------------------------------------


class TestDashboardGenerator:
    @pytest.fixture()
    def sim_data(self):
        """Run simulation and return (eval_metrics, trace_metrics, traces)."""
        config = ModelEndpointConfig()
        tracer = Tracer(config)
        runner = EvaluationRunner(tracer)
        eval_metrics = runner.run_with_simulated_results(
            EvalDataset.default_dataset()
        )
        trace_metrics = tracer.get_metrics()
        traces = [t.to_dict() for t in tracer.traces]
        return eval_metrics, trace_metrics, traces

    def test_generate_creates_file(self, tmp_path, sim_data):
        eval_m, trace_m, traces = sim_data
        gen = DashboardGenerator(str(tmp_path))
        path = gen.generate(eval_m, trace_m, traces, run_id="test-run")
        assert path.exists()
        assert path.name == "dashboard.html"
        content = path.read_text()
        assert "test-run" in content
        assert "<!DOCTYPE html>" in content

    def test_generate_default_run_id(self, tmp_path, sim_data):
        eval_m, trace_m, traces = sim_data
        gen = DashboardGenerator(str(tmp_path))
        path = gen.generate(eval_m, trace_m, traces)
        assert path.exists()

    def test_save_history_creates_file(self, tmp_path, sim_data):
        eval_m, trace_m, _ = sim_data
        gen = DashboardGenerator(str(tmp_path))
        history = gen.save_history("run-1", eval_m, trace_m)
        assert len(history) == 1
        assert history[0]["run_id"] == "run-1"

        history_path = tmp_path / "run_history.json"
        assert history_path.exists()

    def test_save_history_appends(self, tmp_path, sim_data):
        eval_m, trace_m, _ = sim_data
        gen = DashboardGenerator(str(tmp_path))
        gen.save_history("run-1", eval_m, trace_m)
        history = gen.save_history("run-2", eval_m, trace_m)
        assert len(history) == 2
        assert history[0]["run_id"] == "run-1"
        assert history[1]["run_id"] == "run-2"

    def test_save_history_caps_at_100(self, tmp_path, sim_data):
        eval_m, trace_m, _ = sim_data
        gen = DashboardGenerator(str(tmp_path))
        for i in range(105):
            gen.save_history(f"run-{i}", eval_m, trace_m)
        history_path = tmp_path / "run_history.json"
        history = json.loads(history_path.read_text())
        assert len(history) == 100
        assert history[0]["run_id"] == "run-5"

    def test_save_history_corrupt_file(self, tmp_path, sim_data):
        eval_m, trace_m, _ = sim_data
        gen = DashboardGenerator(str(tmp_path))
        history_path = tmp_path / "run_history.json"
        history_path.write_text("not valid json {{{")
        history = gen.save_history("run-1", eval_m, trace_m)
        assert len(history) == 1

    def test_save_history_entry_fields(self, tmp_path, sim_data):
        eval_m, trace_m, _ = sim_data
        gen = DashboardGenerator(str(tmp_path))
        history = gen.save_history("test-run", eval_m, trace_m)
        entry = history[0]
        expected_keys = {
            "run_id",
            "timestamp",
            "accuracy",
            "exact_match",
            "error_rate",
            "total_cases",
            "mean_latency",
            "p95_latency",
            "throughput",
        }
        assert set(entry.keys()) == expected_keys

    def test_output_dir_created(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "dir"
        gen = DashboardGenerator(str(nested))
        assert nested.exists()
        assert gen._output_dir == nested

    def test_generate_with_history(self, tmp_path, sim_data):
        eval_m, trace_m, traces = sim_data
        gen = DashboardGenerator(str(tmp_path))
        history = [
            {"run_id": "old-1", "accuracy": 0.8, "exact_match": 0.7},
            {"run_id": "old-2", "accuracy": 0.85, "exact_match": 0.75},
        ]
        path = gen.generate(eval_m, trace_m, traces, history=history)
        assert path.exists()


# ---------------------------------------------------------------------------
# src.config
# ---------------------------------------------------------------------------


class TestModelEndpointConfig:
    def test_defaults(self):
        cfg = ModelEndpointConfig()
        assert cfg.base_url == "http://localhost:8080"
        assert cfg.verify_endpoint == "/api/v1/verify"
        assert cfg.health_endpoint == "/api/v1/health"
        assert cfg.timeout_seconds == 30.0
        assert cfg.headers == {}
        assert cfg.auth_token == ""

    def test_verify_url(self):
        cfg = ModelEndpointConfig(base_url="http://host:9090")
        assert cfg.verify_url == "http://host:9090/api/v1/verify"

    def test_verify_url_strips_trailing_slash(self):
        cfg = ModelEndpointConfig(base_url="http://host:9090/")
        assert cfg.verify_url == "http://host:9090/api/v1/verify"

    def test_health_url(self):
        cfg = ModelEndpointConfig(base_url="http://host:9090")
        assert cfg.health_url == "http://host:9090/api/v1/health"


class TestServiceConfig:
    def test_defaults(self):
        cfg = ServiceConfig()
        assert isinstance(cfg.model, ModelEndpointConfig)
        assert cfg.output_dir == "output"
        assert cfg.data_dir == "data/eval_sets"
        assert cfg.max_concurrent == 4

    def test_from_env_defaults(self, monkeypatch):
        monkeypatch.delenv("MODEL_BASE_URL", raising=False)
        monkeypatch.delenv("MODEL_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("OUTPUT_DIR", raising=False)
        monkeypatch.delenv("DATA_DIR", raising=False)
        monkeypatch.delenv("MODEL_TIMEOUT", raising=False)
        monkeypatch.delenv("MODEL_VERIFY_ENDPOINT", raising=False)
        monkeypatch.delenv("MODEL_HEALTH_ENDPOINT", raising=False)

        cfg = ServiceConfig.from_env()
        assert cfg.model.base_url == "http://localhost:8080"
        assert cfg.output_dir == "output"

    def test_from_env_custom(self, monkeypatch):
        monkeypatch.setenv("MODEL_BASE_URL", "http://custom:1234")
        monkeypatch.setenv("MODEL_VERIFY_ENDPOINT", "/v2/verify")
        monkeypatch.setenv("MODEL_HEALTH_ENDPOINT", "/v2/health")
        monkeypatch.setenv("MODEL_TIMEOUT", "10")
        monkeypatch.setenv("MODEL_AUTH_TOKEN", "secret")
        monkeypatch.setenv("OUTPUT_DIR", "out2")
        monkeypatch.setenv("DATA_DIR", "data2")

        cfg = ServiceConfig.from_env()
        assert cfg.model.base_url == "http://custom:1234"
        assert cfg.model.verify_endpoint == "/v2/verify"
        assert cfg.model.timeout_seconds == 10.0
        assert cfg.model.auth_token == "secret"
        assert cfg.model.headers["Authorization"] == "Bearer secret"
        assert cfg.output_dir == "out2"
        assert cfg.data_dir == "data2"

    def test_from_env_no_auth_header_when_empty(self, monkeypatch):
        monkeypatch.delenv("MODEL_AUTH_TOKEN", raising=False)
        cfg = ServiceConfig.from_env()
        assert "Authorization" not in cfg.model.headers

    def test_from_file(self, tmp_path):
        data = {
            "model": {
                "base_url": "http://file:5555",
                "verify_endpoint": "/check",
                "health_endpoint": "/ping",
                "timeout_seconds": 15.0,
                "auth_token": "tok",
            },
            "output_dir": "file_out",
            "data_dir": "file_data",
            "max_concurrent": 8,
        }
        fp = tmp_path / "config.json"
        fp.write_text(json.dumps(data))

        cfg = ServiceConfig.from_file(fp)
        assert cfg.model.base_url == "http://file:5555"
        assert cfg.model.verify_endpoint == "/check"
        assert cfg.model.timeout_seconds == 15.0
        assert cfg.model.headers["Authorization"] == "Bearer tok"
        assert cfg.output_dir == "file_out"
        assert cfg.max_concurrent == 8

    def test_from_file_minimal(self, tmp_path):
        fp = tmp_path / "minimal.json"
        fp.write_text(json.dumps({}))
        cfg = ServiceConfig.from_file(fp)
        assert cfg.model.base_url == "http://localhost:8080"
        assert cfg.output_dir == "output"


# ---------------------------------------------------------------------------
# src.tracer.core (TraceRecord + Tracer utilities)
# ---------------------------------------------------------------------------


class TestTraceRecord:
    def test_to_dict(self):
        span = Span(name="s", kind=SpanKind.REQUEST, trace_id="t1")
        rec = TraceRecord(
            trace_id="t1",
            input_address={"street": "1"},
            output_result={"verified": True},
            spans=[span],
        )
        d = rec.to_dict()
        assert d["trace_id"] == "t1"
        assert d["input_address"] == {"street": "1"}
        assert len(d["spans"]) == 1


class TestTracerExport:
    def test_export_traces(self, tmp_path):
        config = ModelEndpointConfig()
        tracer = Tracer(config)
        runner = EvaluationRunner(tracer)
        runner.run_with_simulated_results(EvalDataset.default_dataset())

        export_path = tmp_path / "traces.json"
        tracer.export_traces(str(export_path))
        assert export_path.exists()

        data = json.loads(export_path.read_text())
        assert len(data) == 16
        assert "trace_id" in data[0]
        assert "spans" in data[0]

    def test_export_creates_parent_dirs(self, tmp_path):
        config = ModelEndpointConfig()
        tracer = Tracer(config)
        runner = EvaluationRunner(tracer)
        runner.run_with_simulated_results(EvalDataset.default_dataset())

        export_path = tmp_path / "nested" / "deep" / "traces.json"
        tracer.export_traces(str(export_path))
        assert export_path.exists()

    def test_reset_clears_all(self):
        config = ModelEndpointConfig()
        tracer = Tracer(config)
        runner = EvaluationRunner(tracer)
        runner.run_with_simulated_results(EvalDataset.default_dataset())

        assert len(tracer.traces) > 0
        tracer.reset()
        assert len(tracer.traces) == 0
        assert tracer.get_metrics()["throughput"]["total_requests"] == 0


# ---------------------------------------------------------------------------
# src.main (parse_args + run with --simulate)
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_defaults(self):
        from src.main import parse_args

        args = parse_args([])
        assert args.config is None
        assert args.model_url is None
        assert args.dataset is None
        assert args.output_dir == "output"
        assert args.simulate is False
        assert args.verbose is False
        assert args.export_traces is None

    def test_simulate_flag(self):
        from src.main import parse_args

        args = parse_args(["--simulate"])
        assert args.simulate is True

    def test_verbose_short(self):
        from src.main import parse_args

        args = parse_args(["-v"])
        assert args.verbose is True

    def test_all_options(self):
        from src.main import parse_args

        args = parse_args([
            "--config", "cfg.json",
            "--model-url", "http://m:80",
            "--dataset", "ds.json",
            "--output-dir", "out",
            "--simulate",
            "--verbose",
            "--export-traces", "traces.json",
        ])
        assert args.config == "cfg.json"
        assert args.model_url == "http://m:80"
        assert args.dataset == "ds.json"
        assert args.output_dir == "out"
        assert args.simulate is True
        assert args.verbose is True
        assert args.export_traces == "traces.json"


class TestMainRun:
    def test_run_simulate(self, tmp_path):
        from src.main import parse_args, run

        output_dir = str(tmp_path / "output")
        args = parse_args(["--simulate", "--output-dir", output_dir])
        exit_code = run(args)
        assert exit_code == 0

        dashboard = Path(output_dir) / "dashboard.html"
        assert dashboard.exists()
        history = Path(output_dir) / "run_history.json"
        assert history.exists()

    def test_run_simulate_with_export(self, tmp_path):
        from src.main import parse_args, run

        output_dir = str(tmp_path / "output")
        export_path = str(tmp_path / "traces.json")
        args = parse_args([
            "--simulate",
            "--output-dir", output_dir,
            "--export-traces", export_path,
        ])
        exit_code = run(args)
        assert exit_code == 0

        traces = json.loads(Path(export_path).read_text())
        assert len(traces) == 16

    def test_run_simulate_with_custom_dataset(self, tmp_path):
        from src.main import parse_args, run

        ds_data = {
            "name": "custom",
            "version": "1.0",
            "cases": [
                {
                    "case_id": "c1",
                    "input_address": {"street": "1 Main", "city": "X", "state": "Y", "zip": "0"},
                    "expected_output": {
                        "verified": True,
                        "standardized": {
                            "street": "1 MAIN",
                            "city": "X",
                            "state": "Y",
                            "zip": "0",
                        },
                    },
                    "category": "standard",
                },
            ],
        }
        ds_path = tmp_path / "custom_ds.json"
        ds_path.write_text(json.dumps(ds_data))

        output_dir = str(tmp_path / "output")
        args = parse_args([
            "--simulate",
            "--dataset", str(ds_path),
            "--output-dir", output_dir,
        ])
        exit_code = run(args)
        assert exit_code == 0

    def test_run_simulate_with_config_file(self, tmp_path):
        from src.main import parse_args, run

        cfg_data = {
            "model": {"base_url": "http://test:9999"},
            "output_dir": str(tmp_path / "cfg_output"),
        }
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(cfg_data))

        args = parse_args([
            "--simulate",
            "--config", str(cfg_path),
            "--output-dir", str(tmp_path / "cfg_output"),
        ])
        exit_code = run(args)
        assert exit_code == 0

    def test_run_simulate_with_model_url_override(self, tmp_path):
        from src.main import parse_args, run

        output_dir = str(tmp_path / "output")
        args = parse_args([
            "--simulate",
            "--model-url", "http://override:1111",
            "--output-dir", output_dir,
        ])
        exit_code = run(args)
        assert exit_code == 0


# ---------------------------------------------------------------------------
# End-to-end integration: full simulation pipeline
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_pipeline(self, tmp_path):
        """Full end-to-end: dataset → simulation → metrics → dashboard → export."""
        config = ModelEndpointConfig()
        tracer = Tracer(config)
        runner = EvaluationRunner(tracer)
        dataset = EvalDataset.default_dataset()

        # Run simulation
        eval_metrics = runner.run_with_simulated_results(dataset)
        trace_metrics = tracer.get_metrics()
        traces_data = [t.to_dict() for t in tracer.traces]

        # Verify eval metrics
        summary = eval_metrics["summary"]
        assert summary["total_cases"] == 16
        assert 0 <= summary["overall_accuracy"] <= 1
        assert 0 <= summary["exact_match_rate"] <= 1

        # Verify trace metrics
        assert trace_metrics["throughput"]["total_requests"] == 16
        assert trace_metrics["confidence"]["count"] > 0

        # Generate dashboard
        gen = DashboardGenerator(str(tmp_path))
        history = gen.save_history("e2e-run", eval_metrics, trace_metrics)
        path = gen.generate(
            eval_metrics, trace_metrics, traces_data,
            run_id="e2e-run", history=history,
        )
        assert path.exists()
        html = path.read_text()
        assert len(html) > 1000
        assert "e2e-run" in html

        # Export traces
        export_path = tmp_path / "traces.json"
        tracer.export_traces(str(export_path))
        exported = json.loads(export_path.read_text())
        assert len(exported) == 16

        # Verify history file
        history_path = tmp_path / "run_history.json"
        assert history_path.exists()
        hist = json.loads(history_path.read_text())
        assert hist[0]["run_id"] == "e2e-run"
        assert hist[0]["total_cases"] == 16
