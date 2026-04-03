# ML Evaluation Framework Analysis for AI Address Verification

## Objective

Evaluate TensorFlow Model Analysis (TFMA), MLflow, Weights & Biases (W&B),
Hugging Face Evaluate, PyCaret, ClearML, Keras Tuner, and specialized LLM
evaluation frameworks to find an all-in-one solution for evaluating a custom
AI address verification model.

---

## Framework Assessment

### 1. TensorFlow Model Analysis (TFMA)

| Criterion | Rating | Notes |
|---|---|---|
| Custom model support | Low | Designed for TF SavedModel/Keras; requires adapters for non-TF models |
| Experiment tracking | None | No native experiment tracking—needs separate tooling |
| Metric logging | Medium | Rich sliced metrics, but limited to TF ecosystem |
| Artifact management | None | No model registry or artifact store |
| Dashboard/Viz | Medium | Fairness Indicators widget in Jupyter; no standalone dashboard |
| Deployment overhead | High | Requires Apache Beam, TF Serving stack |
| Address-verification fit | Low | Oriented toward classification/regression on tabular/image data |

**Verdict:** Not suitable. Heavy TF dependency, no experiment tracking, poor fit
for a custom REST-based address verification model.

### 2. MLflow

| Criterion | Rating | Notes |
|---|---|---|
| Custom model support | High | Framework-agnostic `mlflow.pyfunc`; works with any callable |
| Experiment tracking | High | Native experiments, runs, nested runs |
| Metric logging | High | `log_metric`, `log_metrics`, `log_param`, `log_artifact` |
| Artifact management | High | Built-in artifact store (local, S3, GCS, Azure Blob) |
| Model registry | High | Full lifecycle: Staging → Production → Archived |
| Dashboard/Viz | High | Built-in tracking UI with comparison views |
| REST API | High | Full REST API for programmatic access |
| Deployment overhead | Low | `pip install mlflow`; local or remote tracking server |
| Address-verification fit | High | Can wrap any Python function as a model; log custom address metrics |
| Open source | Yes | Apache 2.0 license |

**Verdict: Best all-in-one fit.** Framework-agnostic, full experiment tracking,
artifact management, model registry, and built-in UI. Can directly wrap the
existing `Tracer` and `EvaluationRunner` without architectural changes.

### 3. Weights & Biases (W&B)

| Criterion | Rating | Notes |
|---|---|---|
| Custom model support | High | Framework-agnostic logging |
| Experiment tracking | High | Excellent visualization and comparison tools |
| Metric logging | High | `wandb.log()` with automatic step tracking |
| Artifact management | High | W&B Artifacts for datasets and models |
| Dashboard/Viz | Very High | Best-in-class interactive dashboards |
| Deployment overhead | Medium | Requires W&B account; cloud-hosted (self-hosted available) |
| Address-verification fit | High | Flexible enough for custom metrics |
| Open source | Partial | Client is open-source; server is proprietary |
| Cost | Medium | Free tier limited; paid for teams |

**Verdict:** Excellent capabilities but requires external SaaS dependency and
account management. Not ideal for a self-contained, open-source USPS tool.

### 4. Hugging Face Evaluate

| Criterion | Rating | Notes |
|---|---|---|
| Custom model support | Medium | Focused on NLP/LLM models |
| Experiment tracking | None | No experiment tracking—just metric computation |
| Metric logging | Medium | Computes metrics (BLEU, ROUGE, accuracy); no persistence |
| Artifact management | None | No artifact store |
| Dashboard/Viz | None | No dashboard |
| Deployment overhead | Low | Lightweight `pip install evaluate` |
| Address-verification fit | Low | NLP-oriented; address verification needs custom metrics |

**Verdict:** Too narrow. Only computes metrics—no tracking, no artifacts, no
dashboard. Would need to be combined with other tools.

### 5. PyCaret

| Criterion | Rating | Notes |
|---|---|---|
| Custom model support | Low | AutoML for tabular data; wraps sklearn/XGBoost/LightGBM |
| Experiment tracking | Medium | Built-in MLflow integration |
| Metric logging | Medium | Standard classification/regression metrics |
| Dashboard/Viz | Medium | Basic plots; relies on MLflow for tracking UI |
| Deployment overhead | Medium | Heavy dependencies (sklearn, etc.) |
| Address-verification fit | Low | Not designed for custom model evaluation |

**Verdict:** AutoML tool, not an evaluation framework. Wrong tool for the job.

### 6. ClearML

| Criterion | Rating | Notes |
|---|---|---|
| Custom model support | High | Framework-agnostic |
| Experiment tracking | High | Auto-logging with `Task.init()` |
| Metric logging | High | Scalars, plots, tables |
| Artifact management | High | Built-in artifact management |
| Dashboard/Viz | High | Full web UI |
| Deployment overhead | Medium | Requires ClearML server (self-hosted or cloud) |
| Address-verification fit | Medium | General-purpose but more complex setup |
| Open source | Yes | Apache 2.0 (server and client) |

**Verdict:** Strong contender but more complex to set up than MLflow. Server
infrastructure requirement adds operational overhead.

### 7. Keras Tuner

| Criterion | Rating | Notes |
|---|---|---|
| Custom model support | Low | Keras/TF only |
| Experiment tracking | None | Hyperparameter search only |
| Metric logging | Low | Only during tuning |
| Artifact management | None | No artifact store |
| Dashboard/Viz | None | No dashboard |
| Deployment overhead | Medium | Requires TF/Keras |
| Address-verification fit | None | Hyperparameter tuner, not an eval framework |

**Verdict:** Completely wrong tool. This is a hyperparameter tuner, not an
evaluation or tracking framework.

### 8. Specialized LLM Evaluation Frameworks

Includes: LangSmith, Ragas, DeepEval, Promptfoo, Phoenix (Arize).

| Criterion | Rating | Notes |
|---|---|---|
| Custom model support | Medium | Designed for LLM/prompt chains |
| Experiment tracking | Varies | LangSmith/Phoenix have tracking; others are metric-only |
| Metric logging | Medium | Faithfulness, relevance, hallucination—not address metrics |
| Artifact management | Low | Limited to prompt/response logging |
| Dashboard/Viz | Medium | LangSmith and Phoenix have UIs |
| Deployment overhead | Medium | Various SaaS dependencies |
| Address-verification fit | Low | Address verification is not an LLM task |

**Verdict:** Wrong paradigm. These evaluate LLM text generation quality
(hallucination, faithfulness, coherence). Address verification is a structured
input/output classification and standardization task.

---

## Decision Matrix

| Framework | Custom Model | Tracking | Metrics | Artifacts | Dashboard | Open Source | Fit | **Score** |
|---|---|---|---|---|---|---|---|---|
| **MLflow** | ★★★ | ★★★ | ★★★ | ★★★ | ★★★ | ★★★ | ★★★ | **21/21** |
| W&B | ★★★ | ★★★ | ★★★ | ★★★ | ★★★ | ★★ | ★★★ | 20/21 |
| ClearML | ★★★ | ★★★ | ★★★ | ★★★ | ★★★ | ★★★ | ★★ | 20/21 |
| TFMA | ★ | ★ | ★★ | ★ | ★★ | ★★★ | ★ | 11/21 |
| HF Evaluate | ★★ | ★ | ★★ | ★ | ★ | ★★★ | ★ | 11/21 |
| PyCaret | ★ | ★★ | ★★ | ★ | ★★ | ★★★ | ★ | 12/21 |
| Keras Tuner | ★ | ★ | ★ | ★ | ★ | ★★★ | ★ | 9/21 |
| LLM Frameworks | ★★ | ★★ | ★★ | ★ | ★★ | ★★ | ★ | 12/21 |

---

## Recommendation: MLflow

**MLflow is the clear winner** for this AI address verification eval-trace
service because:

1. **Framework-agnostic**: Works with any model callable—perfect for wrapping
   the existing REST-based address verification model.
2. **All-in-one**: Single tool provides experiment tracking, metric logging,
   parameter logging, artifact storage, model registry, and built-in UI.
3. **Zero architecture changes**: Integrates as an additional logging layer
   alongside the existing `Tracer` and `EvaluationRunner`.
4. **Open source**: Apache 2.0 license, no vendor lock-in.
5. **Low overhead**: `pip install mlflow` and go. Local tracking server
   included—no external infrastructure required.
6. **Production-ready**: Used at scale by thousands of organizations.
7. **Extensible**: Custom metrics, custom model flavors, plugin system.

### Integration Architecture

```
Existing Pipeline:
  EvalDataset → EvaluationRunner → MetricsCalculator → DashboardGenerator

With MLflow:
  EvalDataset → EvaluationRunner → MetricsCalculator → DashboardGenerator
                      ↓                    ↓                    ↓
                 MLflow Run          MLflow Metrics       MLflow Artifacts
                 (params,            (accuracy,           (dashboard.html,
                  dataset info)       latency,             traces.json,
                                      component scores)    dataset.json)
```

MLflow adds a parallel logging path without modifying any existing behavior.
The `--mlflow` flag enables it; when disabled, everything works exactly as
before.
