"""Tests for the MLflow integration module."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.evaluator.mlflow_integration import MLflowTracker, is_available

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_eval_metrics() -> dict[str, Any]:
    """Minimal eval metrics matching the real structure."""
    return {
        "summary": {
            "total_cases": 16,
            "overall_accuracy": 0.9375,
            "exact_match_rate": 0.9375,
            "error_rate": 0.0,
        },
        "component_scores": {
            "street": {"component": "street", "accuracy": 0.9, "error_rate": 0.1},
            "city": {"component": "city", "accuracy": 0.95, "error_rate": 0.05},
        },
        "category_breakdown": {
            "standard": {"total": 2, "passed": 2, "failed": 0, "accuracy": 1.0},
            "invalid": {"total": 2, "passed": 1, "failed": 1, "accuracy": 0.5},
        },
        "confidence_correlation": {
            "correct_mean_confidence": 0.85,
            "incorrect_mean_confidence": 0.4,
        },
        "results": [],
    }


def _sample_trace_metrics() -> dict[str, Any]:
    """Minimal trace metrics matching the real structure."""
    return {
        "latency": {
            "total": {
                "count": 16,
                "mean_ms": 30.5,
                "median_ms": 28.0,
                "p90_ms": 45.0,
                "p95_ms": 50.0,
                "p99_ms": 55.0,
                "min_ms": 10.0,
                "max_ms": 60.0,
            },
        },
        "throughput": {
            "total_requests": 16,
            "successful": 16,
            "failed": 0,
            "requests_per_second": 100.0,
            "error_rate": 0.0,
            "wall_clock_seconds": 0.16,
        },
        "confidence": {
            "count": 16,
            "mean": 0.75,
            "median": 0.78,
            "std_dev": 0.15,
            "min": 0.2,
            "max": 0.98,
            "buckets": {},
        },
    }


# ---------------------------------------------------------------------------
# Tests: is_available
# ---------------------------------------------------------------------------

class TestIsAvailable:
    """Test the is_available sentinel."""

    def test_returns_bool(self):
        result = is_available()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Tests: MLflowTracker without mlflow installed
# ---------------------------------------------------------------------------

class TestMLflowTrackerNoMlflow:
    """Ensure graceful degradation when mlflow is not installed."""

    @patch("src.evaluator.mlflow_integration.MLFLOW_AVAILABLE", False)
    def test_init_without_mlflow(self):
        tracker = MLflowTracker()
        assert tracker._active is False

    @patch("src.evaluator.mlflow_integration.MLFLOW_AVAILABLE", False)
    def test_log_params_noop(self):
        tracker = MLflowTracker()
        # Should not raise
        tracker.log_params({"key": "value"})

    @patch("src.evaluator.mlflow_integration.MLFLOW_AVAILABLE", False)
    def test_log_summary_metrics_noop(self):
        tracker = MLflowTracker()
        tracker.log_summary_metrics(_sample_eval_metrics())

    @patch("src.evaluator.mlflow_integration.MLFLOW_AVAILABLE", False)
    def test_log_trace_metrics_noop(self):
        tracker = MLflowTracker()
        tracker.log_trace_metrics(_sample_trace_metrics())

    @patch("src.evaluator.mlflow_integration.MLFLOW_AVAILABLE", False)
    def test_log_component_scores_noop(self):
        tracker = MLflowTracker()
        tracker.log_component_scores(_sample_eval_metrics())

    @patch("src.evaluator.mlflow_integration.MLFLOW_AVAILABLE", False)
    def test_log_category_breakdown_noop(self):
        tracker = MLflowTracker()
        tracker.log_category_breakdown(_sample_eval_metrics())

    @patch("src.evaluator.mlflow_integration.MLFLOW_AVAILABLE", False)
    def test_log_artifact_file_noop(self):
        tracker = MLflowTracker()
        tracker.log_artifact_file("/nonexistent/file.json")

    @patch("src.evaluator.mlflow_integration.MLFLOW_AVAILABLE", False)
    def test_log_json_artifact_noop(self):
        tracker = MLflowTracker()
        tracker.log_json_artifact({"key": "value"}, "test.json")

    @patch("src.evaluator.mlflow_integration.MLFLOW_AVAILABLE", False)
    def test_start_end_run_noop(self):
        tracker = MLflowTracker()
        tracker.start_run("test-run")
        tracker.end_run()

    @patch("src.evaluator.mlflow_integration.MLFLOW_AVAILABLE", False)
    def test_log_evaluation_run_noop(self):
        tracker = MLflowTracker()
        # Should not raise even with full arguments
        tracker.log_evaluation_run(
            run_id="test-123",
            eval_metrics=_sample_eval_metrics(),
            trace_metrics=_sample_trace_metrics(),
            traces=[{"trace_id": "abc"}],
            dataset_info={"name": "test", "version": "1.0"},
            model_url="http://localhost:8080",
            simulate=True,
            dashboard_path="/tmp/dashboard.html",
        )


# ---------------------------------------------------------------------------
# Tests: MLflowTracker with mocked mlflow
# ---------------------------------------------------------------------------

class TestMLflowTrackerWithMock:
    """Test MLflowTracker behaviour using a mocked mlflow module."""

    @pytest.fixture()
    def mock_mlflow(self):
        """Patch mlflow functions and mark as available."""
        with (
            patch("src.evaluator.mlflow_integration.MLFLOW_AVAILABLE", True),
            patch("src.evaluator.mlflow_integration.mlflow") as mock_ml,
        ):
            # Make set_experiment succeed
            mock_ml.set_experiment = MagicMock()
            mock_ml.set_tracking_uri = MagicMock()
            mock_ml.start_run = MagicMock()
            mock_ml.end_run = MagicMock()
            mock_ml.log_params = MagicMock()
            mock_ml.log_metrics = MagicMock()
            mock_ml.log_artifact = MagicMock()
            yield mock_ml

    def test_init_sets_experiment(self, mock_mlflow):
        tracker = MLflowTracker(experiment_name="test-exp")
        assert tracker._active is True
        mock_mlflow.set_experiment.assert_called_once_with("test-exp")

    def test_init_with_tracking_uri(self, mock_mlflow):
        tracker = MLflowTracker(tracking_uri="http://mlflow:5000")
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://mlflow:5000")
        assert tracker._active is True

    def test_start_and_end_run(self, mock_mlflow):
        tracker = MLflowTracker()
        tracker.start_run("my-run")
        mock_mlflow.start_run.assert_called_once_with(run_name="my-run")
        tracker.end_run()
        mock_mlflow.end_run.assert_called_once()

    def test_log_params(self, mock_mlflow):
        tracker = MLflowTracker()
        tracker.start_run()
        tracker.log_params({"key": "value", "num": 42})
        mock_mlflow.log_params.assert_called_once()
        logged = mock_mlflow.log_params.call_args[0][0]
        assert logged["key"] == "value"
        assert logged["num"] == "42"

    def test_log_params_truncates_long_values(self, mock_mlflow):
        tracker = MLflowTracker()
        tracker.start_run()
        long_val = "x" * 600
        tracker.log_params({"long": long_val})
        logged = mock_mlflow.log_params.call_args[0][0]
        assert len(logged["long"]) == 500

    def test_log_summary_metrics(self, mock_mlflow):
        tracker = MLflowTracker()
        tracker.start_run()
        tracker.log_summary_metrics(_sample_eval_metrics())
        mock_mlflow.log_metrics.assert_called_once()
        metrics = mock_mlflow.log_metrics.call_args[0][0]
        assert "overall_accuracy" in metrics
        assert "exact_match_rate" in metrics
        assert "error_rate" in metrics
        assert "total_cases" in metrics
        assert metrics["overall_accuracy"] == 0.9375

    def test_log_trace_metrics(self, mock_mlflow):
        tracker = MLflowTracker()
        tracker.start_run()
        tracker.log_trace_metrics(_sample_trace_metrics())
        mock_mlflow.log_metrics.assert_called_once()
        metrics = mock_mlflow.log_metrics.call_args[0][0]
        assert "latency_mean_ms" in metrics
        assert "throughput_requests_per_second" in metrics
        assert "confidence_mean" in metrics

    def test_log_component_scores(self, mock_mlflow):
        tracker = MLflowTracker()
        tracker.start_run()
        tracker.log_component_scores(_sample_eval_metrics())
        mock_mlflow.log_metrics.assert_called_once()
        metrics = mock_mlflow.log_metrics.call_args[0][0]
        assert "component_street_accuracy" in metrics
        assert "component_city_accuracy" in metrics
        assert metrics["component_street_accuracy"] == 0.9

    def test_log_category_breakdown(self, mock_mlflow):
        tracker = MLflowTracker()
        tracker.start_run()
        tracker.log_category_breakdown(_sample_eval_metrics())
        mock_mlflow.log_metrics.assert_called_once()
        metrics = mock_mlflow.log_metrics.call_args[0][0]
        assert "category_standard_accuracy" in metrics
        assert "category_invalid_accuracy" in metrics
        assert metrics["category_standard_accuracy"] == 1.0

    def test_log_artifact_file(self, mock_mlflow, tmp_path):
        tracker = MLflowTracker()
        tracker.start_run()
        fpath = tmp_path / "test.json"
        fpath.write_text('{"hello": "world"}')
        tracker.log_artifact_file(fpath)
        mock_mlflow.log_artifact.assert_called_once_with(str(fpath))

    def test_log_json_artifact(self, mock_mlflow):
        tracker = MLflowTracker()
        tracker.start_run()
        tracker.log_json_artifact({"key": "value"}, "out.json")
        mock_mlflow.log_artifact.assert_called_once()
        artifact_path = mock_mlflow.log_artifact.call_args[0][0]
        assert artifact_path.endswith("out.json")

    def test_log_evaluation_run_full(self, mock_mlflow, tmp_path):
        """End-to-end test of log_evaluation_run."""
        dashboard = tmp_path / "dashboard.html"
        dashboard.write_text("<html></html>")

        tracker = MLflowTracker()
        tracker.log_evaluation_run(
            run_id="20260403-120000",
            eval_metrics=_sample_eval_metrics(),
            trace_metrics=_sample_trace_metrics(),
            traces=[{"trace_id": "abc"}],
            dataset_info={"name": "default", "version": "1.0", "total_cases": 16},
            model_url="http://localhost:8080",
            simulate=True,
            dashboard_path=str(dashboard),
        )

        # Verify run lifecycle
        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.end_run.assert_called_once()

        # Verify params were logged
        mock_mlflow.log_params.assert_called_once()
        params = mock_mlflow.log_params.call_args[0][0]
        assert params["run_id"] == "20260403-120000"
        assert params["simulate"] == "True"
        assert params["dataset_name"] == "default"

        # Verify metrics were logged (summary + trace + component + category)
        assert mock_mlflow.log_metrics.call_count == 4

        # Verify artifacts (eval_metrics.json + trace_metrics.json + traces.json + dashboard)
        assert mock_mlflow.log_artifact.call_count == 4

    def test_log_evaluation_run_without_optional_args(self, mock_mlflow):
        """log_evaluation_run works with minimal arguments."""
        tracker = MLflowTracker()
        tracker.log_evaluation_run(
            run_id="run-1",
            eval_metrics=_sample_eval_metrics(),
            trace_metrics=_sample_trace_metrics(),
            traces=[],
        )
        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.end_run.assert_called_once()

    def test_log_evaluation_run_ends_on_exception(self, mock_mlflow):
        """Ensure end_run is called even if logging raises."""
        mock_mlflow.log_params.side_effect = RuntimeError("boom")
        tracker = MLflowTracker()
        with pytest.raises(RuntimeError, match="boom"):
            tracker.log_evaluation_run(
                run_id="fail-run",
                eval_metrics=_sample_eval_metrics(),
                trace_metrics=_sample_trace_metrics(),
                traces=[],
            )
        # end_run must still be called for cleanup
        mock_mlflow.end_run.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: MLflowTracker with empty / edge-case metrics
# ---------------------------------------------------------------------------

class TestMLflowTrackerEdgeCases:
    """Edge cases in metric extraction."""

    @pytest.fixture()
    def mock_mlflow(self):
        with (
            patch("src.evaluator.mlflow_integration.MLFLOW_AVAILABLE", True),
            patch("src.evaluator.mlflow_integration.mlflow") as mock_ml,
        ):
            mock_ml.set_experiment = MagicMock()
            mock_ml.set_tracking_uri = MagicMock()
            mock_ml.start_run = MagicMock()
            mock_ml.end_run = MagicMock()
            mock_ml.log_params = MagicMock()
            mock_ml.log_metrics = MagicMock()
            mock_ml.log_artifact = MagicMock()
            yield mock_ml

    def test_empty_eval_metrics(self, mock_mlflow):
        tracker = MLflowTracker()
        tracker.start_run()
        tracker.log_summary_metrics({})
        # log_metrics not called when there's nothing to log
        mock_mlflow.log_metrics.assert_not_called()

    def test_empty_trace_metrics(self, mock_mlflow):
        tracker = MLflowTracker()
        tracker.start_run()
        tracker.log_trace_metrics({})
        mock_mlflow.log_metrics.assert_not_called()

    def test_empty_component_scores(self, mock_mlflow):
        tracker = MLflowTracker()
        tracker.start_run()
        tracker.log_component_scores({"component_scores": {}})
        mock_mlflow.log_metrics.assert_not_called()

    def test_empty_category_breakdown(self, mock_mlflow):
        tracker = MLflowTracker()
        tracker.start_run()
        tracker.log_category_breakdown({"category_breakdown": {}})
        mock_mlflow.log_metrics.assert_not_called()

    def test_missing_dashboard_path(self, mock_mlflow):
        """log_evaluation_run skips dashboard artifact if path doesn't exist."""
        tracker = MLflowTracker()
        tracker.log_evaluation_run(
            run_id="no-dash",
            eval_metrics=_sample_eval_metrics(),
            trace_metrics=_sample_trace_metrics(),
            traces=[],
            dashboard_path="/nonexistent/dashboard.html",
        )
        # Only 2 JSON artifacts (eval + trace), no dashboard, no traces (empty)
        assert mock_mlflow.log_artifact.call_count == 2


# ---------------------------------------------------------------------------
# Tests: CLI argument parsing for MLflow flags
# ---------------------------------------------------------------------------

class TestMLflowCLIArgs:
    """Verify the new MLflow CLI arguments are parsed correctly."""

    def test_mlflow_flag_default(self):
        from src.main import parse_args

        args = parse_args(["--simulate"])
        assert args.mlflow is False
        assert args.mlflow_uri is None
        assert args.mlflow_experiment == "address-verification-eval"

    def test_mlflow_flag_enabled(self):
        from src.main import parse_args

        args = parse_args(["--simulate", "--mlflow"])
        assert args.mlflow is True

    def test_mlflow_uri(self):
        from src.main import parse_args

        args = parse_args([
            "--simulate", "--mlflow",
            "--mlflow-uri", "http://mlflow:5000",
        ])
        assert args.mlflow_uri == "http://mlflow:5000"

    def test_mlflow_experiment(self):
        from src.main import parse_args

        args = parse_args([
            "--simulate", "--mlflow",
            "--mlflow-experiment", "my-experiment",
        ])
        assert args.mlflow_experiment == "my-experiment"
