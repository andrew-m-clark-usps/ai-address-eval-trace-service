"""Evaluation runner that orchestrates traced evaluations."""

from __future__ import annotations

import logging
import time
from typing import Any

from src.evaluator.datasets import EvalCase, EvalDataset
from src.evaluator.metrics import EvaluationResult, MetricsCalculator
from src.tracer.core import TraceRecord, Tracer

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Runs evaluation cases through the tracer and computes metrics."""

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer
        self._calculator = MetricsCalculator()
        self._results: list[EvaluationResult] = []

    @property
    def results(self) -> list[EvaluationResult]:
        return list(self._results)

    @property
    def calculator(self) -> MetricsCalculator:
        return self._calculator

    def run_dataset(self, dataset: EvalDataset) -> dict[str, Any]:
        """Run all cases in a dataset and return metrics."""
        logger.info("Running evaluation: %s (%d cases)", dataset.name, len(dataset.cases))
        start = time.time()

        for case in dataset.cases:
            result = self._evaluate_case(case)
            self._results.append(result)
            self._calculator.add_result(result)

        elapsed = time.time() - start
        logger.info("Evaluation complete in %.2fs", elapsed)

        metrics = self._calculator.compute_all()
        metrics["dataset"] = {
            "name": dataset.name,
            "version": dataset.version,
            "total_cases": len(dataset.cases),
        }
        metrics["elapsed_seconds"] = round(elapsed, 3)
        return metrics

    def _evaluate_case(self, case: EvalCase) -> EvaluationResult:
        """Evaluate a single case against the model."""
        trace = self._tracer.trace_request(case.input_address)
        actual = trace.output_result or {}

        # Determine pass/fail
        passed = self._check_passed(case.expected_output, actual)
        exact = self._check_exact_match(case.expected_output, actual)
        components = self._check_components(case.expected_output, actual)

        # Get latency from the request span
        latency_ms = 0.0
        if trace.spans:
            latency_ms = trace.spans[0].duration_ms

        return EvaluationResult(
            case_id=case.case_id,
            input_address=case.input_address,
            expected=case.expected_output,
            actual=actual,
            passed=passed,
            exact_match=exact,
            component_matches=components,
            confidence=float(actual.get("confidence", 0.0)),
            latency_ms=latency_ms,
            error=trace.error,
            category=case.category,
        )

    @staticmethod
    def _check_passed(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
        """Check if the result meets minimum pass criteria."""
        if "error" in actual and "error" not in expected:
            return False
        if "verified" in expected:
            if actual.get("verified") != expected["verified"]:
                return False
        return True

    @staticmethod
    def _check_exact_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
        """Check if the standardized output exactly matches expected."""
        expected_std = expected.get("standardized", {})
        actual_std = actual.get("standardized", {})
        if not expected_std and not actual_std:
            return True
        return expected_std == actual_std

    @staticmethod
    def _check_components(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, bool]:
        """Check individual address components."""
        expected_std = expected.get("standardized", {})
        actual_std = actual.get("standardized", {})
        if not expected_std:
            return {}
        matches = {}
        for key in ("street", "city", "state", "zip"):
            if key in expected_std:
                matches[key] = expected_std.get(key, "") == actual_std.get(key, "")
        return matches

    def run_with_simulated_results(self, dataset: EvalDataset) -> dict[str, Any]:
        """Run evaluation with simulated model responses for testing.

        Generates realistic trace data without requiring a live model endpoint.
        This is useful for development, CI pipelines, and dashboard testing.
        """
        import random
        import uuid

        from src.tracer.spans import Span, SpanKind, SpanStatus

        logger.info(
            "Running simulated evaluation: %s (%d cases)", dataset.name, len(dataset.cases)
        )
        start = time.time()

        rng = random.Random(42)

        for case in dataset.cases:
            trace_id = uuid.uuid4().hex[:24]

            # Simulate varying latencies
            base_latency = rng.gauss(45, 15)
            base_latency = max(5, base_latency)

            # Simulate whether model gets it right
            expects_valid = case.expected_output.get("verified", False)
            is_error_case = case.category in ("invalid", "missing_data")
            is_hard_case = case.category in ("error_correction", "abbreviation")

            if is_error_case:
                model_correct = rng.random() < 0.85
                confidence = rng.uniform(0.15, 0.55)
            elif is_hard_case:
                model_correct = rng.random() < 0.70
                confidence = rng.uniform(0.45, 0.85)
            else:
                model_correct = rng.random() < 0.90
                confidence = rng.uniform(0.70, 0.98)

            # Build simulated output
            if model_correct:
                actual = {
                    "verified": expects_valid,
                    "confidence": round(confidence, 4),
                    "standardized": case.expected_output.get("standardized", {}),
                    "components": {},
                    "corrections": [],
                    "metadata": {},
                }
            else:
                actual = {
                    "verified": not expects_valid,
                    "confidence": round(rng.uniform(0.2, 0.6), 4),
                    "standardized": {},
                    "components": {},
                    "corrections": [],
                    "metadata": {},
                }

            # Create trace spans
            now = time.time()
            pre_dur = rng.uniform(0.5, 3.0)
            net_dur = rng.uniform(10, base_latency)
            post_dur = rng.uniform(0.5, 2.0)
            total_dur = pre_dur + net_dur + post_dur

            spans = [
                Span(
                    name="verify_address",
                    kind=SpanKind.REQUEST,
                    trace_id=trace_id,
                    start_time=now,
                    end_time=now + total_dur / 1000,
                    status=SpanStatus.OK,
                    attributes={"confidence": confidence},
                ),
                Span(
                    name="normalize_input",
                    kind=SpanKind.PREPROCESSING,
                    trace_id=trace_id,
                    start_time=now,
                    end_time=now + pre_dur / 1000,
                ),
                Span(
                    name="model_call",
                    kind=SpanKind.NETWORK,
                    trace_id=trace_id,
                    start_time=now + pre_dur / 1000,
                    end_time=now + (pre_dur + net_dur) / 1000,
                    attributes={"http_status": 200, "response_bytes": rng.randint(200, 800)},
                ),
                Span(
                    name="process_response",
                    kind=SpanKind.POSTPROCESSING,
                    trace_id=trace_id,
                    start_time=now + (pre_dur + net_dur) / 1000,
                    end_time=now + total_dur / 1000,
                    attributes={"confidence": confidence},
                ),
            ]

            self._tracer._collector.ingest(spans)
            record = TraceRecord(
                trace_id=trace_id,
                input_address=case.input_address,
                output_result=actual,
                spans=spans,
            )
            self._tracer._traces.append(record)

            passed = self._check_passed(case.expected_output, actual)
            exact = self._check_exact_match(case.expected_output, actual)
            components = self._check_components(case.expected_output, actual)

            result = EvaluationResult(
                case_id=case.case_id,
                input_address=case.input_address,
                expected=case.expected_output,
                actual=actual,
                passed=passed,
                exact_match=exact,
                component_matches=components,
                confidence=confidence,
                latency_ms=total_dur,
                error="",
                category=case.category,
            )
            self._results.append(result)
            self._calculator.add_result(result)

        elapsed = time.time() - start
        metrics = self._calculator.compute_all()
        metrics["dataset"] = {
            "name": dataset.name,
            "version": dataset.version,
            "total_cases": len(dataset.cases),
        }
        metrics["elapsed_seconds"] = round(elapsed, 3)
        metrics["simulated"] = True
        return metrics
