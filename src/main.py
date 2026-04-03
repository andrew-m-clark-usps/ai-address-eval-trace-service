"""Entry point for the address verification eval-trace service."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

from src.config import ServiceConfig
from src.dashboard.generator import DashboardGenerator
from src.evaluator.datasets import EvalDataset
from src.evaluator.mlflow_integration import MLflowTracker
from src.evaluator.mlflow_integration import is_available as mlflow_available
from src.evaluator.runner import EvaluationRunner
from src.tracer.core import Tracer

logger = logging.getLogger("eval-trace")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="eval-trace",
        description="Run evaluations and traces against an AI address verification model",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--model-url",
        type=str,
        default=None,
        help="Base URL of the model endpoint (overrides config)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to custom evaluation dataset JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for dashboard output (default: output)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run with simulated model responses (no live endpoint required)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--export-traces",
        type=str,
        default=None,
        help="Export raw traces to JSON file",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow experiment tracking (requires mlflow package)",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=None,
        help="MLflow tracking server URI (default: local ./mlruns)",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="address-verification-eval",
        help="MLflow experiment name (default: address-verification-eval)",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    """Execute the evaluation and trace pipeline."""
    setup_logging(args.verbose)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    # Configuration
    if args.config:
        config = ServiceConfig.from_file(args.config)
    else:
        config = ServiceConfig.from_env()

    if args.model_url:
        config.model.base_url = args.model_url

    config.output_dir = args.output_dir
    config.run_id = run_id

    logger.info("Starting eval-trace run: %s", run_id)

    # Load dataset
    if args.dataset:
        dataset = EvalDataset.from_file(args.dataset)
        logger.info("Loaded dataset from %s: %d cases", args.dataset, len(dataset.cases))
    else:
        dataset = EvalDataset.default_dataset()
        logger.info("Using default dataset: %d cases", len(dataset.cases))

    # Initialize tracer and runner
    tracer = Tracer(config.model)
    runner = EvaluationRunner(tracer)

    # Run evaluation
    if args.simulate:
        logger.info("Running in simulation mode")
        eval_metrics = runner.run_with_simulated_results(dataset)
    else:
        # Check model health first
        health = tracer.check_health()
        if health["status"] == "unreachable":
            logger.warning(
                "Model endpoint unreachable at %s: %s",
                config.model.base_url,
                health.get("error", "unknown"),
            )
            logger.info("Falling back to simulation mode")
            eval_metrics = runner.run_with_simulated_results(dataset)
        else:
            logger.info("Model endpoint healthy: %s", health)
            eval_metrics = runner.run_dataset(dataset)

    # Collect trace metrics
    trace_metrics = tracer.get_metrics()

    # Export traces if requested
    if args.export_traces:
        tracer.export_traces(args.export_traces)
        logger.info("Traces exported to %s", args.export_traces)

    # Generate dashboard
    dashboard = DashboardGenerator(config.output_dir)

    history = dashboard.save_history(run_id, eval_metrics, trace_metrics)
    logger.info("Run history updated (%d entries)", len(history))

    traces_data = [t.to_dict() for t in tracer.traces]
    output_path = dashboard.generate(
        eval_metrics=eval_metrics,
        trace_metrics=trace_metrics,
        traces=traces_data,
        run_id=run_id,
        history=history,
    )

    logger.info("Dashboard generated: %s", output_path)

    # MLflow tracking
    if args.mlflow:
        if not mlflow_available():
            logger.error(
                "MLflow tracking requested but mlflow is not installed.  "
                "Install with: pip install mlflow"
            )
        else:
            logger.info("Logging run to MLflow")
            tracker = MLflowTracker(
                experiment_name=args.mlflow_experiment,
                tracking_uri=args.mlflow_uri,
            )
            tracker.log_evaluation_run(
                run_id=run_id,
                eval_metrics=eval_metrics,
                trace_metrics=trace_metrics,
                traces=traces_data,
                dataset_info={
                    "name": dataset.name,
                    "version": dataset.version,
                    "total_cases": len(dataset.cases),
                },
                model_url=config.model.base_url,
                simulate=args.simulate,
                dashboard_path=output_path,
            )

    logger.info("Run complete: %s", run_id)

    # Print summary to stdout
    summary = eval_metrics.get("summary", {})
    print(f"\n{'='*60}")
    print(f"  Run ID:         {run_id}")
    print(f"  Cases:          {summary.get('total_cases', 0)}")
    print(f"  Accuracy:       {summary.get('overall_accuracy', 0)*100:.1f}%")
    print(f"  Exact Match:    {summary.get('exact_match_rate', 0)*100:.1f}%")
    print(f"  Error Rate:     {summary.get('error_rate', 0)*100:.1f}%")
    latency = trace_metrics.get("latency", {}).get("total", {}).get("mean_ms", 0)
    print(f"  Mean Latency:   {latency:.1f}ms")
    print(f"  Dashboard:      {output_path}")
    if args.mlflow and mlflow_available():
        print(f"  MLflow:         Experiment '{args.mlflow_experiment}'")
    print(f"{'='*60}\n")

    return 0


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    sys.exit(run(args))


if __name__ == "__main__":
    main()
