"""Rendering surfaces for :class:`src.brooks.context.BrooksContext`.

:mod:`src.brooks.render.text` is the canonical text renderer consumed by
the LLM analyst; :mod:`src.brooks.render.chart` (Phase 4) will add an
image renderer for VLM inputs.
"""

from __future__ import annotations

from src.brooks.render.text import render_context_text

__all__ = ["render_context_text"]
