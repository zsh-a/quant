"""Brooks evaluation pipeline.

This package wires together the four ingredients needed to grade a Brooks
:class:`~src.brooks.analyst.base.Analyst`:

* :mod:`src.brooks.eval.golden` — load a hand-curated or auto-labeled
  dataset of :class:`GoldenSample`'s (parquet or jsonl).
* :mod:`src.brooks.eval.auto_label` — bootstrap silver labels from
  rule + multi-LLM consensus voting.
* :mod:`src.brooks.eval.runner` — replay each sample through an analyst,
  score signals via the optional :class:`TraderEquation`, and emit per-bucket
  metrics.
* :mod:`src.brooks.eval.report` — turn an :class:`EvalReport` into an HTML
  page (with per-pattern / regime / htf bucket tables and Wilson CIs).
"""

from src.brooks.eval.auto_label import AutoLabeler
from src.brooks.eval.golden import GoldenDataset, GoldenSample
from src.brooks.eval.report import EvalReport
from src.brooks.eval.runner import EvalRunner, SampleResult

__all__ = [
    "AutoLabeler",
    "EvalReport",
    "EvalRunner",
    "GoldenDataset",
    "GoldenSample",
    "SampleResult",
]
