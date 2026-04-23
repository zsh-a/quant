"""Analyst plugin layer.

Importing this package triggers registration of all built-in analysts
with :class:`AnalystRegistry`.
"""

from src.brooks.analyst import ensemble, llm, rule  # noqa: F401 — imported for side-effect
from src.brooks.analyst.base import Analyst, AnalystRegistry
from src.brooks.analyst.ensemble import CriticAnalyst, RouterAnalyst, VoteAnalyst
from src.brooks.analyst.llm import LLMAnalyst, LLMSignalBatch
from src.brooks.analyst.rule import RuleAnalyst

__all__ = [
    "Analyst",
    "AnalystRegistry",
    "CriticAnalyst",
    "LLMAnalyst",
    "LLMSignalBatch",
    "RouterAnalyst",
    "RuleAnalyst",
    "VoteAnalyst",
]
