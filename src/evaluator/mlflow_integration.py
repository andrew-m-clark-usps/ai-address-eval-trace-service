"""MLflow integration for experiment tracking and artifact management.

Provides an optional MLflow logging layer that captures evaluation metrics,
trace data, parameters, and artifacts from each evaluation run. Enabled
via the ``--mlflow`` CLI flag; when disabled the existing pipeline is
completely unaffected.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import mlflow
    from mlflow.exceptions import MlflowException

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    MlflowException = Exception  # type: ignore[assignment,misc]
    MLFLOW_AVAILABLE = False


def is_available() -> bool:
    """Return True if the mlflow package is installed."""
    return MLFLOW_AVAILABLE


class MLflowTracker:
    """Logs evaluation runs to an MLflow tracking server.

    All public methods are safe to call even if MLflow is not installed;
    they degrade to no-ops with a warning.

    Parameters
    ----------
    experiment_name:
        MLflow experiment name.  Created automatically if it does not exist.
    tracking_uri:
        MLflow tracking server URI.  ``None`` uses the default local
        ``./mlruns`` directory.
    """

    def __init__(
        self,
        experiment_name: str = "address-verification-eval",
        tracking_uri: str | None = None,
    ) -> None:
        self._experiment_name = experiment_name
        self._tracking_uri = tracking_uri
        self._active = False
        self._run_context: Any = None

        if not MLFLOW_AVAILABLE:
            logger.warning(
                "mlflow is not installed – tracking is disabled.  "
                "Install with: pip install mlflow"
            )
            return

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        try:
            mlflow.set_experiment(experiment_name)
            self._active = True
        except MlflowException as exc:
            logger.error("Failed to set MLflow experiment: %s", exc)
            self._active = False

    # ------------------------------------------------------------------
    # Context manager for a single evaluation run
    # ------------------------------------------------------------------

    def start_run(self, run_name: str | None = None) -> None:
        """Begin a new MLflow run."""
        if not self._active:
            return
        self._run_context = mlflow.start_run(run_name=run_name)

    def end_run(self) -> None:
        """End the current MLflow run."""
        if not self._active:
            return
        mlflow.end_run()
        self._run_context = None

    # ------------------------------------------------------------------
    # Parameter logging
    # ------------------------------------------------------------------

    def log_params(self, params: dict[str, Any]) -> None:
        """Log run parameters (dataset name, model URL, etc.)."""
        if not self._active:
            return
        safe: dict[str, str] = {}
        for key, value in params.items():
            str_val = str(value)
            # MLflow truncates param values at 500 chars
            safe[key] = str_val[:500]
        mlflow.log_params(safe)

    # ------------------------------------------------------------------
    # Metric logging
    # ------------------------------------------------------------------

    def log_summary_metrics(self, eval_metrics: dict[str, Any]) -> None:
        """Log top-level evaluation summary metrics."""
        if not self._active:
            return
        summary = eval_metrics.get("summary", {})
        flat: dict[str, float] = {}
        for key in (
            "overall_accuracy",
            "exact_match_rate",
            "error_rate",
            "total_cases",
        ):
            if key in summary:
                flat[key] = float(summary[key])
        if flat:
            mlflow.log_metrics(flat)

    def log_trace_metrics(self, trace_metrics: dict[str, Any]) -> None:
        """Log latency, throughput, and confidence metrics from traces."""
        if not self._active:
            return
        metrics: dict[str, float] = {}

        # Latency
        latency = trace_metrics.get("latency", {}).get("total", {})
        for key in ("mean_ms", "median_ms", "p90_ms", "p95_ms", "p99_ms",
                     "min_ms", "max_ms"):
            if key in latency:
                metrics[f"latency_{key}"] = float(latency[key])

        # Throughput
        throughput = trace_metrics.get("throughput", {})
        for key in ("requests_per_second", "error_rate", "wall_clock_seconds"):
            if key in throughput:
                metrics[f"throughput_{key}"] = float(throughput[key])

        # Confidence
        confidence = trace_metrics.get("confidence", {})
        for key in ("mean", "median", "std_dev", "min", "max"):
            if key in confidence:
                metrics[f"confidence_{key}"] = float(confidence[key])

        if metrics:
            mlflow.log_metrics(metrics)

    def log_component_scores(self, eval_metrics: dict[str, Any]) -> None:
        """Log per-component accuracy metrics (street, city, state, zip)."""
        if not self._active:
            return
        scores = eval_metrics.get("component_scores", {})
        metrics: dict[str, float] = {}
        for comp_name, comp_data in scores.items():
            if isinstance(comp_data, dict):
                acc = comp_data.get("accuracy")
                if acc is not None:
                    metrics[f"component_{comp_name}_accuracy"] = float(acc)
                err = comp_data.get("error_rate")
                if err is not None:
                    metrics[f"component_{comp_name}_error_rate"] = float(err)
        if metrics:
            mlflow.log_metrics(metrics)

    def log_category_breakdown(self, eval_metrics: dict[str, Any]) -> None:
        """Log per-category accuracy metrics."""
        if not self._active:
            return
        breakdown = eval_metrics.get("category_breakdown", {})
        metrics: dict[str, float] = {}
        for cat_name, cat_data in breakdown.items():
            if isinstance(cat_data, dict):
                acc = cat_data.get("accuracy")
                if acc is not None:
                    metrics[f"category_{cat_name}_accuracy"] = float(acc)
        if metrics:
            mlflow.log_metrics(metrics)

    # ------------------------------------------------------------------
    # Artifact logging
    # ------------------------------------------------------------------

    def log_artifact_file(self, path: str | Path) -> None:
        """Log an existing file as an MLflow artifact."""
        if not self._active:
            return
        mlflow.log_artifact(str(path))

    def log_json_artifact(
        self, data: Any, filename: str
    ) -> None:
        """Serialize *data* to JSON and log as an MLflow artifact."""
        if not self._active:
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / filename
            with open(filepath, "w") as fh:
                json.dump(data, fh, indent=2, default=str)
            mlflow.log_artifact(str(filepath))

    # ------------------------------------------------------------------
    # Convenience: log everything from a completed run
    # ------------------------------------------------------------------

    def log_evaluation_run(
        self,
        *,
        run_id: str,
        eval_metrics: dict[str, Any],
        trace_metrics: dict[str, Any],
        traces: list[dict[str, Any]],
        dataset_info: dict[str, Any] | None = None,
        model_url: str = "",
        simulate: bool = False,
        dashboard_path: str | Path | None = None,
    ) -> None:
        """Log a complete evaluation run in a single call.

        This is the primary public API.  It opens an MLflow run, logs all
        parameters / metrics / artifacts, and closes the run.
        """
        if not self._active:
            logger.debug("MLflow tracking inactive – skipping log_evaluation_run")
            return

        self.start_run(run_name=run_id)
        try:
            # --- Parameters ---
            params: dict[str, Any] = {
                "run_id": run_id,
                "simulate": str(simulate),
                "model_url": model_url or "N/A",
            }
            if dataset_info:
                for key, val in dataset_info.items():
                    params[f"dataset_{key}"] = val
            self.log_params(params)

            # --- Metrics ---
            self.log_summary_metrics(eval_metrics)
            self.log_trace_metrics(trace_metrics)
            self.log_component_scores(eval_metrics)
            self.log_category_breakdown(eval_metrics)

            # --- Artifacts ---
            self.log_json_artifact(eval_metrics, "eval_metrics.json")
            self.log_json_artifact(trace_metrics, "trace_metrics.json")
            if traces:
                self.log_json_artifact(traces, "traces.json")
            if dashboard_path and Path(dashboard_path).exists():
                self.log_artifact_file(dashboard_path)

        finally:
            self.end_run()

        logger.info("MLflow run logged: %s", run_id)
