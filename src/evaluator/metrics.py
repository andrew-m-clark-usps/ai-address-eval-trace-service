"""Evaluation metrics computation for address verification."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ComponentScore:
    """Accuracy score for a single address component."""

    component: str
    correct: int = 0
    incorrect: int = 0
    missing: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def error_rate(self) -> float:
        return self.incorrect / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "correct": self.correct,
            "incorrect": self.incorrect,
            "missing": self.missing,
            "total": self.total,
            "accuracy": round(self.accuracy, 4),
            "error_rate": round(self.error_rate, 4),
        }


@dataclass
class EvaluationResult:
    """Result of a single evaluation case."""

    case_id: str
    input_address: dict[str, Any]
    expected: dict[str, Any]
    actual: dict[str, Any] | None
    passed: bool = False
    exact_match: bool = False
    component_matches: dict[str, bool] = field(default_factory=dict)
    confidence: float = 0.0
    latency_ms: float = 0.0
    error: str = ""
    category: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "input_address": self.input_address,
            "expected": self.expected,
            "actual": self.actual,
            "passed": self.passed,
            "exact_match": self.exact_match,
            "component_matches": self.component_matches,
            "confidence": round(self.confidence, 4),
            "latency_ms": round(self.latency_ms, 3),
            "error": self.error,
            "category": self.category,
        }


class MetricsCalculator:
    """Computes evaluation metrics from a set of results."""

    def __init__(self, results: list[EvaluationResult] | None = None) -> None:
        self._results: list[EvaluationResult] = results or []

    def add_result(self, result: EvaluationResult) -> None:
        self._results.append(result)

    def add_results(self, results: list[EvaluationResult]) -> None:
        self._results.extend(results)

    @property
    def total(self) -> int:
        return len(self._results)

    def overall_accuracy(self) -> float:
        """Fraction of cases that passed."""
        if not self._results:
            return 0.0
        return sum(1 for r in self._results if r.passed) / len(self._results)

    def exact_match_rate(self) -> float:
        """Fraction of cases with exact output match."""
        if not self._results:
            return 0.0
        return sum(1 for r in self._results if r.exact_match) / len(self._results)

    def error_rate(self) -> float:
        """Fraction of cases that produced errors."""
        if not self._results:
            return 0.0
        return sum(1 for r in self._results if r.error) / len(self._results)

    def component_scores(self) -> dict[str, ComponentScore]:
        """Per-component accuracy breakdown."""
        components: dict[str, ComponentScore] = {}
        for result in self._results:
            for comp, matched in result.component_matches.items():
                if comp not in components:
                    components[comp] = ComponentScore(component=comp)
                score = components[comp]
                score.total += 1
                if matched:
                    score.correct += 1
                else:
                    score.incorrect += 1
        return components

    def category_breakdown(self) -> dict[str, dict[str, Any]]:
        """Accuracy breakdown by test category."""
        categories: dict[str, list[EvaluationResult]] = {}
        for result in self._results:
            cat = result.category or "uncategorized"
            categories.setdefault(cat, []).append(result)

        breakdown = {}
        for cat, results in categories.items():
            passed = sum(1 for r in results if r.passed)
            total = len(results)
            breakdown[cat] = {
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "accuracy": round(passed / total, 4) if total > 0 else 0.0,
            }
        return breakdown

    def confidence_correlation(self) -> dict[str, Any]:
        """Analyze relationship between confidence and correctness."""
        correct_conf = [r.confidence for r in self._results if r.passed and r.confidence > 0]
        incorrect_conf = [
            r.confidence for r in self._results if not r.passed and r.confidence > 0
        ]
        return {
            "correct_mean_confidence": (
                round(statistics.mean(correct_conf), 4) if correct_conf else 0.0
            ),
            "incorrect_mean_confidence": (
                round(statistics.mean(incorrect_conf), 4) if incorrect_conf else 0.0
            ),
            "correct_samples": len(correct_conf),
            "incorrect_samples": len(incorrect_conf),
        }

    def compute_all(self) -> dict[str, Any]:
        """Compute all metrics."""
        return {
            "summary": {
                "total_cases": self.total,
                "overall_accuracy": round(self.overall_accuracy(), 4),
                "exact_match_rate": round(self.exact_match_rate(), 4),
                "error_rate": round(self.error_rate(), 4),
            },
            "component_scores": {
                k: v.to_dict() for k, v in self.component_scores().items()
            },
            "category_breakdown": self.category_breakdown(),
            "confidence_correlation": self.confidence_correlation(),
            "results": [r.to_dict() for r in self._results],
        }
