"""Rendering surfaces for :class:`src.brooks.context.BrooksContext`.

:mod:`src.brooks.render.text` is the canonical text renderer consumed by
the LLM analyst; :mod:`src.brooks.render.chart` is the matplotlib-backed
image renderer shared by the VLM analyst, UI previews, and eval reports.
"""

from __future__ import annotations

from src.brooks.render.chart import ChartStyle, render_annotated, render_chart
from src.brooks.render.text import render_context_text

__all__ = [
    "render_context_text",
    "render_chart",
    "render_annotated",
    "ChartStyle",
]
