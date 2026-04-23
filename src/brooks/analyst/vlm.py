"""VLM-backed analyst — feeds a rendered OHLCV chart and a compact text
summary into a multimodal :class:`Provider` and surfaces the structured
output as unified :class:`Signal` records.

The design mirrors :class:`src.brooks.analyst.llm.LLMAnalyst`: provider,
prompt assets, context renderer, and output schema are all injected so
the analyst itself stays small and has zero hard-coded vendor SDK
calls. Only two things change relative to the text analyst:

* The live user message is multimodal (``[image, text]``) instead of
  text-only.
* The cache namespace for few-shot uses ``fewshot_vlm`` so that sharing
  a prompt directory between text and vision analysts does not collide
  on the provider side.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from pydantic import BaseModel, Field

from src.alpha.llm.cache import CacheSpec
from src.alpha.llm.provider import ImagePart, Provider
from src.brooks.analyst.base import AnalystRegistry
from src.brooks.context import BrooksContext
from src.brooks.prompts import PromptBundle
from src.brooks.render.chart import ChartStyle, render_chart
from src.brooks.render.text import render_context_text
from src.brooks.schema import Signal

__all__ = ["VLMAnalyst", "VLMSignalBatch"]


class VLMSignalBatch(BaseModel):
    """Structured output schema given to the multimodal :class:`Provider`.

    ``signals`` is the Signal list the aggregator consumes. ``annotations``
    is a free-form list of overlay descriptors the VLM returns so callers
    can re-render the chart with pattern bounding boxes (bar ranges or
    image-normalized bboxes) drawn back on top — handy for eval reports
    and for the human-in-the-loop golden set.
    """

    reasoning: str = ""
    signals: list[Signal]
    annotations: list[dict] = Field(default_factory=list)


@AnalystRegistry.register("vlm")
class VLMAnalyst:
    """Run a multimodal :class:`Provider` against a VLM :class:`PromptBundle`."""

    name: str

    def __init__(
        self,
        provider: Provider,
        model: str,
        prompts: Optional[PromptBundle] = None,
        include_htf: bool = True,
        chart_style: Optional[ChartStyle] = None,
        cache_ttl_seconds: int = 3600,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        context_budget_tokens: int = 1000,
    ) -> None:
        self._provider = provider
        self._model = model
        self._prompts = prompts if prompts is not None else PromptBundle.load_vlm()
        style = chart_style or ChartStyle()
        if not include_htf and style.include_htf_inset:
            style = replace(style, include_htf_inset=False)
        self._chart_style = style
        self._cache = CacheSpec(
            blocks=["concept_manual", "fewshot_vlm"],
            ttl_seconds=cache_ttl_seconds,
        )
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._context_budget_tokens = context_budget_tokens
        self.name = f"vlm:{model}"

    async def analyze(self, ctx: BrooksContext) -> list[Signal]:
        chart_bytes = render_chart(ctx, self._chart_style)
        user_text = render_context_text(ctx, budget_tokens=self._context_budget_tokens)
        image = ImagePart(data=chart_bytes, media_type="image/png")
        messages = self._prompts.build_messages_multimodal(
            user_text=user_text,
            user_image=image,
        )
        resp = await self._provider.complete(
            messages=messages,
            schema=VLMSignalBatch,
            cache=self._cache,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

        signals = list(resp.parsed.signals)
        annotations = list(resp.parsed.annotations)
        meta_extras = {
            "latency_ms": resp.latency_ms,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "cache_read_tokens": resp.usage.cache_read_tokens,
            "cache_creation_tokens": resp.usage.cache_creation_tokens,
            "cache_hit": resp.cache_hit,
            "model": resp.model or self._model,
            "chart_bytes": len(chart_bytes),
            "annotations": annotations,
        }
        for s in signals:
            s.source = self.name
            s.meta.update(meta_extras)
        return signals
