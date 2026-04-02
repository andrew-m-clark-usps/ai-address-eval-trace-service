"""Static HTML dashboard generator."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, BaseLoader

logger = logging.getLogger(__name__)


class DashboardGenerator:
    """Generates a single-file HTML dashboard from evaluation and trace data."""

    def __init__(self, output_dir: str = "output") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._env = Environment(loader=BaseLoader(), autoescape=True)

    def generate(
        self,
        eval_metrics: dict[str, Any],
        trace_metrics: dict[str, Any],
        traces: list[dict[str, Any]],
        run_id: str = "",
        history: list[dict[str, Any]] | None = None,
    ) -> Path:
        """Generate the dashboard HTML file."""
        now = datetime.now(timezone.utc)
        run_id = run_id or now.strftime("%Y%m%d-%H%M%S")

        context = {
            "run_id": run_id,
            "timestamp": now.isoformat(),
            "timestamp_display": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "eval_metrics": eval_metrics,
            "trace_metrics": trace_metrics,
            "traces": traces[:50],
            "history": history or [],
            "eval_json": json.dumps(eval_metrics, default=str),
            "trace_json": json.dumps(trace_metrics, default=str),
            "history_json": json.dumps(history or [], default=str),
        }

        template = self._env.from_string(DASHBOARD_TEMPLATE)
        html = template.render(**context)

        output_path = self._output_dir / "dashboard.html"
        output_path.write_text(html)
        logger.info("Dashboard written to %s", output_path)
        return output_path

    def save_history(
        self,
        run_id: str,
        eval_metrics: dict[str, Any],
        trace_metrics: dict[str, Any],
        history_file: str = "run_history.json",
    ) -> list[dict[str, Any]]:
        """Append current run to history and return full history."""
        history_path = self._output_dir / history_file
        history: list[dict[str, Any]] = []

        if history_path.exists():
            try:
                history = json.loads(history_path.read_text())
            except (json.JSONDecodeError, OSError):
                history = []

        entry = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "accuracy": eval_metrics.get("summary", {}).get("overall_accuracy", 0),
            "exact_match": eval_metrics.get("summary", {}).get("exact_match_rate", 0),
            "error_rate": eval_metrics.get("summary", {}).get("error_rate", 0),
            "total_cases": eval_metrics.get("summary", {}).get("total_cases", 0),
            "mean_latency": trace_metrics.get("latency", {}).get("total", {}).get("mean_ms", 0),
            "p95_latency": trace_metrics.get("latency", {}).get("total", {}).get("p95_ms", 0),
            "throughput": trace_metrics.get("throughput", {}).get("requests_per_second", 0),
        }
        history.append(entry)

        # Keep last 100 runs
        history = history[-100:]
        history_path.write_text(json.dumps(history, indent=2, default=str))
        return history


DASHBOARD_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Address Verification Model -- Eval &amp; Trace Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg-primary:#0a0e17;--bg-secondary:#111827;--bg-card:#1a2234;--bg-card-alt:#1e2a3f;
  --border:#2a3650;--border-hover:#3b4f70;
  --text-primary:#e8ecf4;--text-secondary:#8b95a8;--text-dim:#5c6578;
  --accent:#3b82f6;--accent-glow:rgba(59,130,246,0.15);
  --success:#10b981;--warning:#f59e0b;--danger:#ef4444;
  --font-mono:'SF Mono','Cascadia Code','Fira Code',monospace;
  --font-sans:-apple-system,BlinkMacSystemFont,'Inter','Segoe UI',sans-serif;
  --radius:6px;
}
html{font-size:14px;-webkit-font-smoothing:antialiased}
body{font-family:var(--font-sans);background:var(--bg-primary);color:var(--text-primary);line-height:1.5;min-height:100vh}
a{color:var(--accent);text-decoration:none}

.shell{max-width:1440px;margin:0 auto;padding:24px 32px 64px}
header{display:flex;align-items:center;justify-content:space-between;padding:20px 0 16px;border-bottom:1px solid var(--border);margin-bottom:28px}
header h1{font-size:1.15rem;font-weight:600;letter-spacing:-0.01em;color:var(--text-primary)}
header .meta{font-family:var(--font-mono);font-size:0.75rem;color:var(--text-dim);text-align:right;line-height:1.7}

.kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:28px}
.kpi{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);padding:18px 20px;position:relative;overflow:hidden;transition:border-color .15s}
.kpi:hover{border-color:var(--border-hover)}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:2px}
.kpi.blue::before{background:var(--accent)}.kpi.green::before{background:var(--success)}.kpi.amber::before{background:var(--warning)}.kpi.red::before{background:var(--danger)}
.kpi .label{font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-secondary);margin-bottom:6px}
.kpi .value{font-family:var(--font-mono);font-size:1.6rem;font-weight:700;letter-spacing:-0.02em}
.kpi .sub{font-family:var(--font-mono);font-size:0.7rem;color:var(--text-dim);margin-top:4px}

.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:28px}
.grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px;margin-bottom:28px}
@media(max-width:900px){.grid-2,.grid-3{grid-template-columns:1fr}}

.card{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);padding:20px 22px;transition:border-color .15s}
.card:hover{border-color:var(--border-hover)}
.card h2{font-size:0.8rem;text-transform:uppercase;letter-spacing:0.06em;color:var(--text-secondary);margin-bottom:16px;font-weight:600}
.card canvas{width:100%!important;max-height:260px}

.section-label{font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-dim);margin:32px 0 14px;font-weight:600}

table{width:100%;border-collapse:collapse;font-size:0.8rem}
th{text-align:left;font-weight:600;color:var(--text-secondary);padding:10px 12px;border-bottom:1px solid var(--border);font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em}
td{padding:9px 12px;border-bottom:1px solid rgba(42,54,80,0.5);color:var(--text-primary);font-family:var(--font-mono);font-size:0.78rem}
tr:hover td{background:rgba(59,130,246,0.04)}

.badge{display:inline-block;padding:2px 8px;border-radius:3px;font-size:0.68rem;font-weight:600;letter-spacing:0.03em}
.badge.pass{background:rgba(16,185,129,0.12);color:var(--success)}
.badge.fail{background:rgba(239,68,68,0.12);color:var(--danger)}
.badge.warn{background:rgba(245,158,11,0.12);color:var(--warning)}

.trace-row{cursor:pointer}
.trace-detail{display:none;background:var(--bg-secondary);padding:0}
.trace-detail.open{display:table-row}
.trace-detail td{padding:12px 20px;font-size:0.75rem;color:var(--text-secondary)}
.trace-detail pre{white-space:pre-wrap;font-family:var(--font-mono);font-size:0.72rem;line-height:1.6;color:var(--text-dim);max-height:240px;overflow-y:auto;padding:10px;background:var(--bg-primary);border-radius:var(--radius);border:1px solid var(--border)}

.bar-inline{display:inline-block;height:6px;border-radius:3px;vertical-align:middle;margin-left:6px}

footer{text-align:center;padding:36px 0 12px;font-size:0.68rem;color:var(--text-dim);border-top:1px solid var(--border);margin-top:40px}
</style>
</head>
<body>
<div class="shell">

<header>
  <h1>Address Verification Model &mdash; Evaluation &amp; Trace Report</h1>
  <div class="meta">
    Run <strong>{{ run_id }}</strong><br>
    {{ timestamp_display }}
  </div>
</header>

<!-- KPI Row -->
<div class="kpi-row">
  <div class="kpi blue">
    <div class="label">Overall Accuracy</div>
    <div class="value">{{ "%.1f"|format(eval_metrics.summary.overall_accuracy * 100) }}%</div>
    <div class="sub">{{ eval_metrics.summary.total_cases }} cases evaluated</div>
  </div>
  <div class="kpi green">
    <div class="label">Exact Match Rate</div>
    <div class="value">{{ "%.1f"|format(eval_metrics.summary.exact_match_rate * 100) }}%</div>
    <div class="sub">Standardized output match</div>
  </div>
  <div class="kpi amber">
    <div class="label">Mean Latency</div>
    <div class="value">{{ "%.1f"|format(trace_metrics.latency.total.mean_ms) }}<span style="font-size:0.7em;color:var(--text-dim)">ms</span></div>
    <div class="sub">p95: {{ "%.1f"|format(trace_metrics.latency.total.p95_ms) }}ms</div>
  </div>
  <div class="kpi red">
    <div class="label">Error Rate</div>
    <div class="value">{{ "%.1f"|format(eval_metrics.summary.error_rate * 100) }}%</div>
    <div class="sub">{{ trace_metrics.throughput.failed }} failed of {{ trace_metrics.throughput.total_requests }}</div>
  </div>
  <div class="kpi blue">
    <div class="label">Throughput</div>
    <div class="value">{{ "%.1f"|format(trace_metrics.throughput.requests_per_second) }}<span style="font-size:0.6em;color:var(--text-dim)"> req/s</span></div>
    <div class="sub">{{ "%.2f"|format(trace_metrics.throughput.wall_clock_seconds) }}s total</div>
  </div>
  <div class="kpi green">
    <div class="label">Confidence</div>
    <div class="value">{{ "%.2f"|format(trace_metrics.confidence.mean if trace_metrics.confidence.count > 0 else 0) }}</div>
    <div class="sub">Mean model confidence</div>
  </div>
</div>

<!-- Charts Row 1 -->
<div class="grid-2">
  <div class="card">
    <h2>Category Accuracy</h2>
    <canvas id="chart-category"></canvas>
  </div>
  <div class="card">
    <h2>Latency Distribution</h2>
    <canvas id="chart-latency"></canvas>
  </div>
</div>

<div class="grid-3">
  <div class="card">
    <h2>Component Accuracy</h2>
    <canvas id="chart-components"></canvas>
  </div>
  <div class="card">
    <h2>Confidence Distribution</h2>
    <canvas id="chart-confidence"></canvas>
  </div>
  <div class="card">
    <h2>Processing Breakdown</h2>
    <canvas id="chart-breakdown"></canvas>
  </div>
</div>

{% if history|length > 1 %}
<div class="grid-2">
  <div class="card">
    <h2>Accuracy Trend</h2>
    <canvas id="chart-trend-accuracy"></canvas>
  </div>
  <div class="card">
    <h2>Latency Trend</h2>
    <canvas id="chart-trend-latency"></canvas>
  </div>
</div>
{% endif %}

<!-- Component Scores Table -->
<div class="section-label">Component-Level Scores</div>
<div class="card" style="padding:0;overflow:hidden">
<table>
<thead><tr><th>Component</th><th>Correct</th><th>Incorrect</th><th>Total</th><th>Accuracy</th><th></th></tr></thead>
<tbody>
{% for key, comp in eval_metrics.component_scores.items() %}
<tr>
  <td>{{ comp.component }}</td>
  <td>{{ comp.correct }}</td>
  <td>{{ comp.incorrect }}</td>
  <td>{{ comp.total }}</td>
  <td>{{ "%.1f"|format(comp.accuracy * 100) }}%</td>
  <td><span class="bar-inline" style="width:{{ (comp.accuracy * 80)|int }}px;background:var(--success)"></span><span class="bar-inline" style="width:{{ ((1 - comp.accuracy) * 80)|int }}px;background:var(--danger)"></span></td>
</tr>
{% endfor %}
</tbody>
</table>
</div>

<!-- Category Breakdown Table -->
<div class="section-label">Category Breakdown</div>
<div class="card" style="padding:0;overflow:hidden">
<table>
<thead><tr><th>Category</th><th>Passed</th><th>Failed</th><th>Total</th><th>Accuracy</th></tr></thead>
<tbody>
{% for cat, data in eval_metrics.category_breakdown.items() %}
<tr>
  <td>{{ cat }}</td>
  <td>{{ data.passed }}</td>
  <td>{{ data.failed }}</td>
  <td>{{ data.total }}</td>
  <td><span class="badge {{ 'pass' if data.accuracy >= 0.8 else ('warn' if data.accuracy >= 0.5 else 'fail') }}">{{ "%.0f"|format(data.accuracy * 100) }}%</span></td>
</tr>
{% endfor %}
</tbody>
</table>
</div>

<!-- Confidence Correlation -->
<div class="section-label">Confidence Analysis</div>
<div class="card">
  <table style="max-width:480px">
    <tr><td style="color:var(--text-secondary)">Mean confidence (correct predictions)</td><td><strong>{{ "%.4f"|format(eval_metrics.confidence_correlation.correct_mean_confidence) }}</strong></td></tr>
    <tr><td style="color:var(--text-secondary)">Mean confidence (incorrect predictions)</td><td><strong>{{ "%.4f"|format(eval_metrics.confidence_correlation.incorrect_mean_confidence) }}</strong></td></tr>
    <tr><td style="color:var(--text-secondary)">Correct sample count</td><td>{{ eval_metrics.confidence_correlation.correct_samples }}</td></tr>
    <tr><td style="color:var(--text-secondary)">Incorrect sample count</td><td>{{ eval_metrics.confidence_correlation.incorrect_samples }}</td></tr>
  </table>
</div>

<!-- Individual Results -->
<div class="section-label">Individual Evaluation Results</div>
<div class="card" style="padding:0;overflow:hidden">
<table>
<thead><tr><th>Case</th><th>Category</th><th>Status</th><th>Exact</th><th>Confidence</th><th>Latency</th><th>Input</th></tr></thead>
<tbody>
{% for r in eval_metrics.results %}
<tr class="trace-row" onclick="toggleDetail('detail-{{ r.case_id }}')">
  <td>{{ r.case_id }}</td>
  <td>{{ r.category }}</td>
  <td><span class="badge {{ 'pass' if r.passed else 'fail' }}">{{ 'PASS' if r.passed else 'FAIL' }}</span></td>
  <td><span class="badge {{ 'pass' if r.exact_match else 'fail' }}">{{ 'YES' if r.exact_match else 'NO' }}</span></td>
  <td>{{ "%.3f"|format(r.confidence) }}</td>
  <td>{{ "%.1f"|format(r.latency_ms) }}ms</td>
  <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{{ r.input_address.street }}, {{ r.input_address.city }}</td>
</tr>
<tr class="trace-detail" id="detail-{{ r.case_id }}">
  <td colspan="7">
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
      <div><strong style="color:var(--text-secondary)">Expected:</strong><pre>{{ r.expected | tojson(indent=2) }}</pre></div>
      <div><strong style="color:var(--text-secondary)">Actual:</strong><pre>{{ r.actual | tojson(indent=2) }}</pre></div>
    </div>
  </td>
</tr>
{% endfor %}
</tbody>
</table>
</div>

<footer>
  Generated {{ timestamp_display }} &middot; Run {{ run_id }}
</footer>

</div>

<script>
const evalData = {{ eval_json }};
const traceData = {{ trace_json }};
const historyData = {{ history_json }};

const COLORS = {
  blue:'#3b82f6',green:'#10b981',amber:'#f59e0b',red:'#ef4444',
  purple:'#8b5cf6',cyan:'#06b6d4',pink:'#ec4899',slate:'#64748b',
  blueA:'rgba(59,130,246,0.2)',greenA:'rgba(16,185,129,0.2)',
  amberA:'rgba(245,158,11,0.2)',redA:'rgba(239,68,68,0.2)'
};

Chart.defaults.color = '#8b95a8';
Chart.defaults.borderColor = 'rgba(42,54,80,0.5)';
Chart.defaults.font.family = "-apple-system,BlinkMacSystemFont,'Inter','Segoe UI',sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.labels.boxWidth = 10;
Chart.defaults.plugins.legend.labels.padding = 14;

// Category Accuracy
(function(){
  const bd = evalData.category_breakdown || {};
  const labels = Object.keys(bd);
  const acc = labels.map(k => bd[k].accuracy * 100);
  const passed = labels.map(k => bd[k].passed);
  const failed = labels.map(k => bd[k].failed);
  new Chart(document.getElementById('chart-category'),{
    type:'bar',
    data:{
      labels: labels,
      datasets:[
        {label:'Passed',data:passed,backgroundColor:COLORS.green,borderRadius:3},
        {label:'Failed',data:failed,backgroundColor:COLORS.red,borderRadius:3}
      ]
    },
    options:{
      responsive:true,
      plugins:{legend:{position:'top'}},
      scales:{x:{grid:{display:false}},y:{beginAtZero:true,title:{display:true,text:'Cases'}}}
    }
  });
})();

// Latency Distribution
(function(){
  const lat = traceData.latency || {};
  const total = lat.total || {};
  const labels = ['Min','Median','Mean','P90','P95','P99','Max'];
  const values = [total.min_ms,total.median_ms,total.mean_ms,total.p90_ms,total.p95_ms,total.p99_ms,total.max_ms];
  new Chart(document.getElementById('chart-latency'),{
    type:'bar',
    data:{
      labels: labels,
      datasets:[{
        data:values,
        backgroundColor:[COLORS.green,COLORS.blue,COLORS.blue,COLORS.amber,COLORS.amber,COLORS.red,COLORS.red],
        borderRadius:3
      }]
    },
    options:{
      responsive:true,
      plugins:{legend:{display:false}},
      scales:{x:{grid:{display:false}},y:{beginAtZero:true,title:{display:true,text:'ms'}}}
    }
  });
})();

// Component Accuracy
(function(){
  const cs = evalData.component_scores || {};
  const labels = Object.keys(cs);
  const acc = labels.map(k => cs[k].accuracy * 100);
  new Chart(document.getElementById('chart-components'),{
    type:'doughnut',
    data:{
      labels: labels,
      datasets:[{
        data: acc,
        backgroundColor:[COLORS.blue,COLORS.green,COLORS.amber,COLORS.purple,COLORS.cyan,COLORS.pink],
        borderWidth:0,
        hoverOffset:6
      }]
    },
    options:{
      responsive:true,
      cutout:'60%',
      plugins:{legend:{position:'bottom'}}
    }
  });
})();

// Confidence Distribution
(function(){
  const conf = traceData.confidence || {};
  const buckets = conf.buckets || {};
  new Chart(document.getElementById('chart-confidence'),{
    type:'bar',
    data:{
      labels: Object.keys(buckets),
      datasets:[{
        data: Object.values(buckets),
        backgroundColor:COLORS.blue,
        borderRadius:3
      }]
    },
    options:{
      responsive:true,
      plugins:{legend:{display:false}},
      scales:{x:{grid:{display:false},title:{display:true,text:'Confidence Range'}},y:{beginAtZero:true,title:{display:true,text:'Count'}}}
    }
  });
})();

// Processing Breakdown
(function(){
  const lat = traceData.latency || {};
  const keys = ['preprocessing','inference','network','postprocessing'];
  const means = keys.map(k => (lat[k] || {}).mean_ms || 0);
  new Chart(document.getElementById('chart-breakdown'),{
    type:'doughnut',
    data:{
      labels: keys.map(k => k.charAt(0).toUpperCase() + k.slice(1)),
      datasets:[{
        data: means,
        backgroundColor:[COLORS.cyan,COLORS.blue,COLORS.amber,COLORS.green],
        borderWidth:0
      }]
    },
    options:{
      responsive:true,
      cutout:'55%',
      plugins:{legend:{position:'bottom'}}
    }
  });
})();

// History Trends
if(historyData.length > 1){
  const tLabels = historyData.map(h => h.run_id || h.timestamp.slice(0,10));

  const accCanvas = document.getElementById('chart-trend-accuracy');
  if(accCanvas){
    new Chart(accCanvas,{
      type:'line',
      data:{
        labels:tLabels,
        datasets:[
          {label:'Accuracy',data:historyData.map(h=>h.accuracy*100),borderColor:COLORS.blue,backgroundColor:COLORS.blueA,fill:true,tension:0.3,pointRadius:3},
          {label:'Exact Match',data:historyData.map(h=>h.exact_match*100),borderColor:COLORS.green,backgroundColor:COLORS.greenA,fill:true,tension:0.3,pointRadius:3}
        ]
      },
      options:{responsive:true,scales:{y:{beginAtZero:true,max:100,title:{display:true,text:'%'}}}}
    });
  }

  const latCanvas = document.getElementById('chart-trend-latency');
  if(latCanvas){
    new Chart(latCanvas,{
      type:'line',
      data:{
        labels:tLabels,
        datasets:[
          {label:'Mean',data:historyData.map(h=>h.mean_latency),borderColor:COLORS.amber,tension:0.3,pointRadius:3},
          {label:'P95',data:historyData.map(h=>h.p95_latency),borderColor:COLORS.red,tension:0.3,pointRadius:3}
        ]
      },
      options:{responsive:true,scales:{y:{beginAtZero:true,title:{display:true,text:'ms'}}}}
    });
  }
}

function toggleDetail(id){
  const el = document.getElementById(id);
  if(el) el.classList.toggle('open');
}
</script>
</body>
</html>'''
