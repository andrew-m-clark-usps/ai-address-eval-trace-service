# AI Address Eval Trace Service

Trace and evaluation service for AI address verification models. Captures span-level performance data, runs structured evaluation suites, and renders results into a single-file HTML dashboard that regenerates on each run.

---

## Overview

This service wraps an external AI address verification model endpoint and provides:

- **Tracing** — Span-based instrumentation capturing preprocessing, network, inference, and postprocessing phases with latency distributions (min, median, mean, p90, p95, p99, max), throughput, and confidence scoring.
- **Evaluation** — Structured test suites covering 10 address categories (standard, abbreviation, error correction, missing data, PO Box, rural, military, ZIP+4, unit, invalid). Per-component accuracy (street, city, state, ZIP), category breakdowns, and confidence-correctness correlation.
- **Dashboard** — Static single-file HTML report with a dark professional theme, Chart.js visualizations, expandable per-case detail rows, and historical trend tracking across runs.
- **Simulation Mode** — Generates realistic trace data without requiring a live model endpoint, useful for development, CI pipelines, and dashboard testing.
- **MLflow Integration** — Optional experiment tracking via MLflow. Log metrics, parameters, and artifacts (dashboard, traces, dataset) to a local or remote MLflow tracking server for cross-run comparison and model lifecycle management.

## Quick Start

### Install

```bash
pip install -e ".[dev]"
```

### Run (Simulation Mode)

No live model endpoint required:

```bash
python -m src.main --simulate
```

This generates `output/dashboard.html` with full evaluation metrics and trace data.

### Run (Live Model)

```bash
python -m src.main --model-url http://localhost:8080
```

If the model endpoint is unreachable, the service automatically falls back to simulation mode.

### Run with Custom Dataset

```bash
python -m src.main --simulate --dataset data/eval_sets/addresses.json
```

## CLI Reference

```
usage: eval-trace [-h] [--config CONFIG] [--model-url MODEL_URL]
                  [--dataset DATASET] [--output-dir OUTPUT_DIR]
                  [--simulate] [--verbose] [--export-traces EXPORT_TRACES]
                  [--mlflow] [--mlflow-uri MLFLOW_URI]
                  [--mlflow-experiment MLFLOW_EXPERIMENT]

Options:
  --config CONFIG           Path to JSON config file
  --model-url MODEL_URL     Base URL of the model endpoint
  --dataset DATASET         Path to custom evaluation dataset JSON
  --output-dir OUTPUT_DIR   Directory for dashboard output (default: output)
  --simulate                Run with simulated model responses
  --verbose, -v             Enable debug logging
  --export-traces FILE      Export raw traces to JSON file
  --mlflow                  Enable MLflow experiment tracking
  --mlflow-uri URI          MLflow tracking server URI (default: local ./mlruns)
  --mlflow-experiment NAME  MLflow experiment name (default: address-verification-eval)
```

## Configuration

### Environment Variables

| Variable                | Default                   | Description                |
|-------------------------|---------------------------|----------------------------|
| `MODEL_BASE_URL`        | `http://localhost:8080`   | Model endpoint base URL    |
| `MODEL_VERIFY_ENDPOINT` | `/api/v1/verify`          | Verification endpoint path |
| `MODEL_HEALTH_ENDPOINT` | `/api/v1/health`          | Health check endpoint path |
| `MODEL_TIMEOUT`         | `30`                      | Request timeout (seconds)  |
| `MODEL_AUTH_TOKEN`      | (empty)                   | Bearer token for auth      |
| `OUTPUT_DIR`            | `output`                  | Dashboard output directory |
| `DATA_DIR`              | `data/eval_sets`          | Evaluation data directory  |

### Config File

```json
{
  "model": {
    "base_url": "http://localhost:8080",
    "verify_endpoint": "/api/v1/verify",
    "health_endpoint": "/api/v1/health",
    "timeout_seconds": 30
  },
  "output_dir": "output",
  "data_dir": "data/eval_sets",
  "max_concurrent": 4
}
```

```bash
python -m src.main --config config.json --simulate
```

## Evaluation Dataset Format

Datasets are JSON files with this structure:

```json
{
  "name": "custom-eval",
  "version": "1.0",
  "cases": [
    {
      "case_id": "std-001",
      "input_address": {
        "street": "1600 Pennsylvania Avenue NW",
        "city": "Washington",
        "state": "DC",
        "zip": "20500"
      },
      "expected_output": {
        "verified": true,
        "standardized": {
          "street": "1600 PENNSYLVANIA AVE NW",
          "city": "WASHINGTON",
          "state": "DC",
          "zip": "20500"
        }
      },
      "category": "standard",
      "description": "Known valid address"
    }
  ]
}
```

A default dataset with 16 cases is included at `data/eval_sets/addresses.json` and also built into the code.

## Test Categories

| Category           | Description                                |
|--------------------|--------------------------------------------|
| `standard`         | Known valid addresses                      |
| `abbreviation`     | Directional/suffix standardization         |
| `error_correction` | Typos in street and city names             |
| `missing_data`     | Missing required address components        |
| `po_box`           | PO Box addresses                           |
| `rural`            | Rural routes and highway contract routes   |
| `military`         | APO/FPO military addresses                 |
| `zip_plus_4`       | ZIP+4 code validation                      |
| `unit`             | Suite, floor, apartment designations       |
| `invalid`          | Completely invalid or garbage input        |

## Dashboard

The generated dashboard (`output/dashboard.html`) includes:

- **KPI Row** — Overall accuracy, exact match rate, mean latency, error rate, throughput, mean confidence
- **Category Accuracy** — Stacked bar chart of pass/fail by test category
- **Latency Distribution** — Bar chart showing min, median, mean, p90, p95, p99, max
- **Component Accuracy** — Doughnut chart of per-component (street, city, state, ZIP) accuracy
- **Confidence Distribution** — Histogram of model confidence scores
- **Processing Breakdown** — Doughnut chart of preprocessing, network, inference, postprocessing time
- **Trend Charts** — Accuracy and latency trends across historical runs (when available)
- **Component Scores Table** — Detailed per-component correct/incorrect/total/accuracy
- **Category Breakdown Table** — Per-category pass/fail/accuracy
- **Confidence Analysis** — Mean confidence for correct vs incorrect predictions
- **Individual Results** — Expandable rows showing expected vs actual output for each case

The dashboard is a self-contained HTML file that updates each time the service runs. Run history is persisted in `output/run_history.json`.

## Project Structure

```
src/
  config.py             # Service and model endpoint configuration
  main.py               # CLI entry point and pipeline orchestration
  tracer/
    spans.py            # Span, SpanKind, SpanStatus, SpanEvent definitions
    collectors.py       # MetricCollector, latency/throughput/confidence aggregation
    core.py             # Tracer engine wrapping model HTTP calls
  evaluator/
    datasets.py         # EvalCase, EvalDataset with built-in test cases
    metrics.py          # MetricsCalculator, ComponentScore, EvaluationResult
    runner.py           # EvaluationRunner with live and simulation modes
    mlflow_integration.py  # Optional MLflow experiment tracking
  dashboard/
    generator.py        # Static HTML dashboard generator with Jinja2
tests/
  test_tracer.py        # Test suite covering all modules
  test_mlflow_integration.py  # Tests for MLflow integration
data/
  eval_sets/
    addresses.json      # Default evaluation dataset (16 cases, 10 categories)
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Install with MLflow support
pip install -e ".[mlflow]"

# Run tests
pytest

# Run linter
ruff check src/ tests/

# Run with verbose logging
python -m src.main --simulate --verbose

# Export traces for analysis
python -m src.main --simulate --export-traces traces.json
```

## MLflow Experiment Tracking

Enable MLflow to log evaluation runs for cross-run comparison, metric trending, and artifact management.

### Setup

```bash
pip install -e ".[mlflow]"
```

### Run with MLflow (local tracking)

```bash
python -m src.main --simulate --mlflow
```

This creates a `./mlruns` directory with experiment data. View the results:

```bash
mlflow ui
# Open http://localhost:5000
```

### Run with remote MLflow server

```bash
python -m src.main --simulate --mlflow --mlflow-uri http://mlflow-server:5000
```

### What gets logged

Each evaluation run logs the following to MLflow:

**Parameters:**
- `run_id`, `simulate`, `model_url`
- `dataset_name`, `dataset_version`, `dataset_total_cases`

**Metrics:**
- `overall_accuracy`, `exact_match_rate`, `error_rate`, `total_cases`
- `latency_mean_ms`, `latency_p95_ms`, `latency_p99_ms`, etc.
- `throughput_requests_per_second`, `throughput_error_rate`
- `confidence_mean`, `confidence_median`, `confidence_std_dev`
- `component_street_accuracy`, `component_city_accuracy`, etc.
- `category_standard_accuracy`, `category_invalid_accuracy`, etc.

**Artifacts:**
- `eval_metrics.json` — Full evaluation results
- `trace_metrics.json` — Trace-level performance data
- `traces.json` — Raw trace records
- `dashboard.html` — Generated dashboard report

### Framework Selection

MLflow was selected after evaluating 8 ML evaluation frameworks. See
[docs/framework_analysis.md](docs/framework_analysis.md) for the full analysis
comparing TensorFlow Model Analysis, MLflow, Weights & Biases, Hugging Face
Evaluate, PyCaret, ClearML, Keras Tuner, and specialized LLM evaluation
frameworks.

## Requirements

- Python 3.10+
- `requests` — HTTP client for model endpoint calls
- `jinja2` — HTML template rendering
- `mlflow` — Experiment tracking (optional, install with `pip install -e ".[mlflow]"`)
- `pytest` — Testing (dev)
- `ruff` — Linting (dev)
