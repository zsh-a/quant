"""Analyst plugin layer.

Importing this package triggers registration of all built-in analysts
with :class:`AnalystRegistry`.
"""

from src.brooks.analyst import rule  # noqa: F401 — imported for side-effect
from src.brooks.analyst.base import Analyst, AnalystRegistry
from src.brooks.analyst.rule import RuleAnalyst

__all__ = [
    "Analyst",
    "AnalystRegistry",
    "RuleAnalyst",
]
