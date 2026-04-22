"""LLM-backed analyst — drives a :class:`Provider` with a versioned
:class:`PromptBundle` and surfaces its structured output as unified
:class:`Signal` records.

This is the Phase 3.4 replacement for ``src/strategies/brooks_llm_pipeline.py``:
the provider, prompt assets, context renderer, and output schema are all
injected, so the analyst itself stays under 150 lines and has zero hardcoded
vendor SDK calls.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from src.alpha.llm.cache import CacheSpec
from src.alpha.llm.provider import Provider
from src.brooks.analyst.base import AnalystRegistry
from src.brooks.context import BrooksContext
from src.brooks.prompts import PromptBundle
from src.brooks.render.text import render_context_text
from src.brooks.schema import Signal

__all__ = ["LLMAnalyst", "LLMSignalBatch"]


class LLMSignalBatch(BaseModel):
    """Structured output schema given to the :class:`Provider`.

    The model returns one batch per call: an explanation string plus zero
    or more :class:`Signal` records. Pydantic validation at the provider
    boundary enforces the unified Signal contract.
    """

    reasoning: str = ""
    signals: list[Signal]


@AnalystRegistry.register("llm")
class LLMAnalyst:
    """Run a :class:`Provider` against a versioned :class:`PromptBundle`."""

    name: str

    def __init__(
        self,
        provider: Provider,
        model: str,
        prompts: Optional[PromptBundle] = None,
        cache_ttl_seconds: int = 3600,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        context_budget_tokens: int = 2000,
    ) -> None:
        self._provider = provider
        self._model = model
        self._prompts = prompts if prompts is not None else PromptBundle.load()
        self._cache = CacheSpec(
            blocks=["concept_manual", "fewshot"],
            ttl_seconds=cache_ttl_seconds,
        )
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._context_budget_tokens = context_budget_tokens
        self.name = f"llm:{model}"

    async def analyze(self, ctx: BrooksContext) -> list[Signal]:
        user_text = render_context_text(ctx, budget_tokens=self._context_budget_tokens)
        messages = self._prompts.build_messages(user_context=user_text)
        resp = await self._provider.complete(
            messages=messages,
            schema=LLMSignalBatch,
            cache=self._cache,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

        signals = list(resp.parsed.signals)
        meta_extras = {
            "latency_ms": resp.latency_ms,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "cache_read_tokens": resp.usage.cache_read_tokens,
            "cache_creation_tokens": resp.usage.cache_creation_tokens,
            "cache_hit": resp.cache_hit,
            "model": resp.model or self._model,
        }
        for s in signals:
            s.source = self.name
            s.meta.update(meta_extras)
        return signals
