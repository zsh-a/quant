"""HTML / DataFrame report for an :class:`EvalRunner` execution.

The :class:`EvalReport` carries the per-sample :class:`SampleResult`
records and exposes:

* :meth:`to_dataframe` — a flat ``pandas.DataFrame`` of every row.
* :meth:`bucket_metrics` — pattern / regime / htf_aligned / source
  precision-recall-F1 with Wilson 95% confidence intervals.
* :meth:`cost_summary` — aggregate latency, token usage, and cache hit
  rate (only meaningful for LLM analysts; rule analysts fill 0s).
* :meth:`to_html` — a self-contained HTML page with the headline
  metrics and one table per bucket dimension.
"""

from __future__ import annotations

import html
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from src.brooks.eval.runner import SampleResult

__all__ = ["EvalReport", "BucketMetrics"]


@dataclass
class BucketMetrics:
    """Precision / recall / F1 with Wilson CI for one bucket."""

    bucket: str
    value: str
    samples: int
    pattern_precision: float
    pattern_recall: float
    pattern_f1: float
    side_accuracy: float
    hit_rate_1r: float
    hit_rate_2r: float
    avg_realized_r: float
    pattern_precision_ci: tuple[float, float]
    hit_rate_1r_ci: tuple[float, float]

    def to_row(self) -> dict:
        return {
            "bucket": self.bucket,
            "value": self.value,
            "samples": self.samples,
            "pattern_precision": round(self.pattern_precision, 4),
            "pattern_recall": round(self.pattern_recall, 4),
            "pattern_f1": round(self.pattern_f1, 4),
            "side_accuracy": round(self.side_accuracy, 4),
            "hit_rate_1r": round(self.hit_rate_1r, 4),
            "hit_rate_2r": round(self.hit_rate_2r, 4),
            "avg_realized_r": round(self.avg_realized_r, 4),
            "pattern_precision_ci_lo": round(self.pattern_precision_ci[0], 4),
            "pattern_precision_ci_hi": round(self.pattern_precision_ci[1], 4),
            "hit_rate_1r_ci_lo": round(self.hit_rate_1r_ci[0], 4),
            "hit_rate_1r_ci_hi": round(self.hit_rate_1r_ci[1], 4),
        }


@dataclass
class EvalReport:
    """Aggregated output of :meth:`EvalRunner.run`."""

    results: List[SampleResult] = field(default_factory=list)
    analyst_name: str = "analyst"
    dataset_size: int = 0

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    def overall(self) -> BucketMetrics:
        return _bucket_metrics("overall", "all", self.results)

    def bucket_metrics(self, dimension: str) -> List[BucketMetrics]:
        """Group results by ``dimension`` and compute per-bucket metrics.

        Supported dimensions: ``pattern``, ``regime``, ``htf_aligned``,
        ``source``.
        """
        getter = _DIMENSION_GETTERS.get(dimension)
        if getter is None:
            raise ValueError(f"unknown bucket dimension: {dimension!r}")
        groups: dict[str, List[SampleResult]] = {}
        for r in self.results:
            key = str(getter(r))
            groups.setdefault(key, []).append(r)
        return [_bucket_metrics(dimension, key, rows) for key, rows in sorted(groups.items())]

    def cost_summary(self) -> dict:
        if not self.results:
            return {
                "samples": 0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "cache_hit_rate": 0.0,
            }
        latencies = [r.latency_ms for r in self.results]
        in_tokens = sum(r.input_tokens for r in self.results)
        out_tokens = sum(r.output_tokens for r in self.results)
        cache_eligible = [r for r in self.results if r.input_tokens or r.output_tokens]
        cache_hits = sum(1 for r in cache_eligible if r.cache_hit)
        cache_hit_rate = cache_hits / len(cache_eligible) if cache_eligible else 0.0
        return {
            "samples": len(self.results),
            "avg_latency_ms": round(statistics.fmean(latencies), 3),
            "p95_latency_ms": round(_percentile(latencies, 95), 3),
            "total_input_tokens": in_tokens,
            "total_output_tokens": out_tokens,
            "cache_hit_rate": round(cache_hit_rate, 4),
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dataframe(self):
        import pandas as pd

        rows = [r.to_row() for r in self.results]
        return pd.DataFrame(rows)

    def to_json(self) -> str:
        payload = {
            "analyst": self.analyst_name,
            "dataset_size": self.dataset_size,
            "overall": self.overall().to_row(),
            "buckets": {dim: [b.to_row() for b in self.bucket_metrics(dim)] for dim in _DIMENSION_GETTERS},
            "cost": self.cost_summary(),
            "rows": [r.to_row() for r in self.results],
        }
        return json.dumps(payload, indent=2, default=_json_default)

    def to_html(self, path: Optional[Path | str] = None) -> str:
        """Render an HTML report. If ``path`` is provided, also writes it."""
        body_parts: List[str] = []
        body_parts.append(_render_header(self.analyst_name, self.dataset_size, len(self.results)))
        body_parts.append(_render_overall(self.overall()))
        body_parts.append(_render_cost(self.cost_summary()))
        for dim in _DIMENSION_GETTERS:
            body_parts.append(_render_bucket_table(dim, self.bucket_metrics(dim)))
        body_parts.append(_render_sample_table(self.results[:50]))
        html_doc = _HTML_TEMPLATE.format(
            title=html.escape(f"Brooks Eval — {self.analyst_name}"),
            body="\n".join(body_parts),
        )
        if path is not None:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(html_doc, encoding="utf-8")
        return html_doc


# ---------------------------------------------------------------------------
# Bucketing helpers
# ---------------------------------------------------------------------------


def _dim_pattern(r: SampleResult) -> str:
    return r.expected_pattern


def _dim_regime(r: SampleResult) -> str:
    return r.regime


def _dim_htf(r: SampleResult) -> str:
    return "aligned" if r.htf_aligned else "unaligned"


def _dim_source(r: SampleResult) -> str:
    return r.source


_DIMENSION_GETTERS = {
    "pattern": _dim_pattern,
    "regime": _dim_regime,
    "htf_aligned": _dim_htf,
    "source": _dim_source,
}


def _bucket_metrics(dimension: str, value: str, rows: List[SampleResult]) -> BucketMetrics:
    n = len(rows)
    if n == 0:
        zero_ci = (0.0, 0.0)
        return BucketMetrics(
            bucket=dimension,
            value=value,
            samples=0,
            pattern_precision=0.0,
            pattern_recall=0.0,
            pattern_f1=0.0,
            side_accuracy=0.0,
            hit_rate_1r=0.0,
            hit_rate_2r=0.0,
            avg_realized_r=0.0,
            pattern_precision_ci=zero_ci,
            hit_rate_1r_ci=zero_ci,
        )

    # Pattern-level precision & recall over this bucket.
    predicted = [r for r in rows if r.predicted_signal is not None]
    matched = [r for r in rows if r.pattern_match]
    side_matched = [r for r in rows if r.side_match]
    precision = len(matched) / len(predicted) if predicted else 0.0
    recall = len(matched) / n
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    side_accuracy = len(side_matched) / n

    hit_1r_n = sum(1 for r in rows if r.hit_1r == "hit")
    hit_2r_n = sum(1 for r in rows if r.hit_2r == "hit")
    resolved_1r = sum(1 for r in rows if r.hit_1r in ("hit", "miss"))
    resolved_2r = sum(1 for r in rows if r.hit_2r in ("hit", "miss"))
    hit_rate_1r = hit_1r_n / resolved_1r if resolved_1r else 0.0
    hit_rate_2r = hit_2r_n / resolved_2r if resolved_2r else 0.0

    realized = [r.realized_r for r in rows if r.realized_r is not None]
    avg_r = statistics.fmean(realized) if realized else 0.0

    return BucketMetrics(
        bucket=dimension,
        value=value,
        samples=n,
        pattern_precision=precision,
        pattern_recall=recall,
        pattern_f1=f1,
        side_accuracy=side_accuracy,
        hit_rate_1r=hit_rate_1r,
        hit_rate_2r=hit_rate_2r,
        avg_realized_r=avg_r,
        pattern_precision_ci=_wilson(len(matched), max(1, len(predicted))),
        hit_rate_1r_ci=_wilson(hit_1r_n, max(1, resolved_1r)),
    )


def _wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion at confidence ``z``."""
    if trials <= 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials)) / denom
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    return (round(lo, 4), round(hi, 4))


def _percentile(values: Iterable[float], pct: float) -> float:
    arr = sorted(values)
    if not arr:
        return 0.0
    if len(arr) == 1:
        return arr[0]
    rank = (pct / 100.0) * (len(arr) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return arr[lo]
    return arr[lo] + (arr[hi] - arr[lo]) * (rank - lo)


def _json_default(o):
    if isinstance(o, set):
        return sorted(o)
    return str(o)


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 24px; color: #1f2933; }}
h1 {{ font-size: 1.4rem; margin-bottom: 0.2rem; }}
h2 {{ font-size: 1.1rem; margin-top: 1.5rem; border-bottom: 1px solid #d9e2ec; padding-bottom: 4px; }}
table {{ border-collapse: collapse; margin: 0.4rem 0 1rem; font-size: 0.85rem; }}
th, td {{ border: 1px solid #cbd5e0; padding: 4px 8px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
tr:nth-child(even) {{ background: #f7fafc; }}
.kpi {{ display: inline-block; background: #edf2f7; padding: 6px 10px; margin: 4px 8px 4px 0; border-radius: 4px; font-size: 0.9rem; }}
.kpi b {{ font-size: 1.05rem; }}
.muted {{ color: #627d98; font-size: 0.8rem; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def _render_header(analyst_name: str, dataset_size: int, sample_count: int) -> str:
    return (
        f"<h1>Brooks Eval — {html.escape(analyst_name)}</h1>"
        f"<p class='muted'>dataset size: {dataset_size} · scored: {sample_count}</p>"
    )


def _render_overall(b: BucketMetrics) -> str:
    return (
        "<h2>Overall</h2>"
        + _kpi("samples", b.samples)
        + _kpi("pattern P", _pct(b.pattern_precision))
        + _kpi("pattern R", _pct(b.pattern_recall))
        + _kpi("pattern F1", _pct(b.pattern_f1))
        + _kpi("side acc", _pct(b.side_accuracy))
        + _kpi("hit-1R", _pct(b.hit_rate_1r))
        + _kpi("hit-2R", _pct(b.hit_rate_2r))
        + _kpi("avg R", f"{b.avg_realized_r:.2f}")
    )


def _render_cost(cost: dict) -> str:
    return (
        "<h2>Cost</h2>"
        + _kpi("avg latency", f"{cost.get('avg_latency_ms', 0.0):.1f} ms")
        + _kpi("p95 latency", f"{cost.get('p95_latency_ms', 0.0):.1f} ms")
        + _kpi("input tokens", cost.get("total_input_tokens", 0))
        + _kpi("output tokens", cost.get("total_output_tokens", 0))
        + _kpi("cache hit", _pct(cost.get("cache_hit_rate", 0.0)))
    )


def _render_bucket_table(dimension: str, buckets: List[BucketMetrics]) -> str:
    headers = [
        dimension,
        "n",
        "P",
        "R",
        "F1",
        "side",
        "hit-1R",
        "1R CI",
        "hit-2R",
        "avg R",
    ]
    rows = []
    for b in buckets:
        rows.append(
            "<tr>"
            f"<td>{html.escape(b.value)}</td>"
            f"<td>{b.samples}</td>"
            f"<td>{_pct(b.pattern_precision)}</td>"
            f"<td>{_pct(b.pattern_recall)}</td>"
            f"<td>{_pct(b.pattern_f1)}</td>"
            f"<td>{_pct(b.side_accuracy)}</td>"
            f"<td>{_pct(b.hit_rate_1r)}</td>"
            f"<td>{_pct(b.hit_rate_1r_ci[0])}–{_pct(b.hit_rate_1r_ci[1])}</td>"
            f"<td>{_pct(b.hit_rate_2r)}</td>"
            f"<td>{b.avg_realized_r:.2f}</td>"
            "</tr>"
        )
    body = "<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>"
    body += "<tbody>" + ("\n".join(rows) or "<tr><td colspan='10'>no data</td></tr>") + "</tbody>"
    return f"<h2>By {html.escape(dimension)}</h2><table>{body}</table>"


def _render_sample_table(rows: List[SampleResult]) -> str:
    if not rows:
        return "<h2>Samples</h2><p class='muted'>no samples</p>"
    headers = ["id", "expect", "predict", "side", "regime", "hit-1R", "realized R", "latency"]
    body_rows = []
    for r in rows:
        pred = f"{r.predicted_signal.pattern}/{r.predicted_signal.side}" if r.predicted_signal else "—"
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(r.sample_id)}</td>"
            f"<td>{html.escape(r.expected_pattern)}</td>"
            f"<td>{html.escape(pred)}</td>"
            f"<td>{html.escape(r.expected_side)}</td>"
            f"<td>{html.escape(r.regime)}</td>"
            f"<td>{r.hit_1r}</td>"
            f"<td>{r.realized_r if r.realized_r is None else f'{r.realized_r:.2f}'}</td>"
            f"<td>{r.latency_ms:.1f} ms</td>"
            "</tr>"
        )
    body = "<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>"
    body += "<tbody>" + "\n".join(body_rows) + "</tbody>"
    return f"<h2>Samples (first {len(rows)})</h2><table>{body}</table>"


def _kpi(label: str, value) -> str:
    return f"<span class='kpi'>{html.escape(label)} <b>{html.escape(str(value))}</b></span>"


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"
