# AI Address Eval Trace Service -- Repo Overview and Notes Setup

Date: 2026-04-27
Tags: repository-overview, project-setup, notes-infrastructure

## Summary

This note captures the structure and capabilities of the AI Address Eval Trace Service repository. It also documents the creation of the `notes/` directory as a persistent location for dated project notes going forward.

## Details

The repository implements a trace and evaluation service for AI address verification models. It wraps an external model endpoint and provides three core capabilities:

1. **Tracing** -- Span-based instrumentation that captures preprocessing, network, inference, and postprocessing phases with detailed latency distributions (min, median, mean, p90, p95, p99, max), throughput, and confidence scoring.

2. **Evaluation** -- Structured test suites covering 10 address categories (standard, abbreviation, error correction, missing data, PO Box, rural, military, ZIP+4, unit, invalid). Includes per-component accuracy for street, city, state, and ZIP.

3. **Dashboard** -- Static single-file HTML report (`output/dashboard.html`) with Chart.js visualizations, expandable per-case detail rows, and historical trend tracking across runs.

The service also supports a simulation mode for development and CI without a live model endpoint, and optional MLflow integration for experiment tracking.

### Project layout

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
  test_tracer.py
  test_mlflow_integration.py
data/
  eval_sets/
    addresses.json      # Default evaluation dataset (16 cases, 10 categories)
notes/                  # Dated project notes (new)
```

### Key commands

```bash
# Install
pip install -e ".[dev]"

# Run in simulation mode
python -m src.main --simulate

# Run tests
pytest

# Lint
ruff check src/ tests/
```

## Decisions

- Created `notes/` at the repo root since no existing notes directory (`notes/`, `docs/notes/`, or `.notes/`) was found.
- Chose `notes/` over `docs/notes/` to keep the path short and top-level visible, matching the existing flat structure (`docs/`, `data/`, `src/`, `tests/`).

## Open questions

- Should a `.gitkeep` or `README.md` be added inside `notes/` to describe its purpose and conventions for future contributors?
- Should notes be excluded from linting or other CI checks?

## References

- `README.md` in the repo root -- primary source for project overview, CLI reference, and configuration details used in this note.
- `docs/framework_analysis.md` -- documents the evaluation of eight ML frameworks that led to selecting MLflow for experiment tracking.
