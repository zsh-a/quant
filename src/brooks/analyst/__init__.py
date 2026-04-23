"""Analyst plugin layer.

Importing this package triggers registration of all built-in analysts
with :class:`AnalystRegistry`.
"""

from src.brooks.analyst import ensemble, llm, rule, vlm  # noqa: F401 — imported for side-effect
from src.brooks.analyst.base import Analyst, AnalystRegistry
from src.brooks.analyst.ensemble import CriticAnalyst, RouterAnalyst, VoteAnalyst
from src.brooks.analyst.llm import LLMAnalyst, LLMSignalBatch
from src.brooks.analyst.rule import RuleAnalyst
from src.brooks.analyst.vlm import VLMAnalyst, VLMSignalBatch

__all__ = [
    "Analyst",
    "AnalystRegistry",
    "CriticAnalyst",
    "LLMAnalyst",
    "LLMSignalBatch",
    "RouterAnalyst",
    "RuleAnalyst",
    "VLMAnalyst",
    "VLMSignalBatch",
    "VoteAnalyst",
]
