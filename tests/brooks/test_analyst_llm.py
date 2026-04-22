"""MockProvider-driven tests for :class:`LLMAnalyst` (Phase 3.4).

The provider is the only collaborator that hits the network in production,
so a synchronous in-process ``MockProvider`` lets us exercise every branch
of the analyst — happy path, schema violation, provider failure, cache hit
metadata — without any I/O.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, List, Optional

import pytest
from pydantic import BaseModel, ValidationError

from src.alpha.llm.cache import CacheSpec
from src.alpha.llm.provider import Message, Response, Usage
from src.brooks.analyst import AnalystRegistry, LLMAnalyst, LLMSignalBatch
from src.brooks.analyst.base import Analyst
from src.brooks.context import Bar, BrooksContext, TFSnapshot
from src.brooks.prompts import PromptBundle
from src.brooks.schema import Signal

# ---------------------------------------------------------------------------
# MockProvider
# ---------------------------------------------------------------------------


@dataclass
class MockProvider:
    """In-process Provider double.

    * ``payload`` — what the provider should return as ``Response.parsed``
      (must already be a ``schema`` instance, mirroring real provider behavior).
    * ``raw_payload`` — alternative: a plain dict that will be revalidated
      through ``schema``. Lets us simulate schema violations.
    * ``raise_exc`` — exception to raise instead of completing.
    """

    name: str = "mock"
    payload: Optional[BaseModel] = None
    raw_payload: Optional[dict] = None
    raise_exc: Optional[Exception] = None
    usage: Usage = field(
        default_factory=lambda: Usage(
            input_tokens=120,
            output_tokens=45,
            cache_creation_tokens=0,
            cache_read_tokens=0,
        )
    )
    latency_ms: float = 12.5
    model: str = "mock-model"
    cache_hit: bool = False
    last_messages: List[Message] = field(default_factory=list)
    last_cache: Optional[CacheSpec] = None
    last_kwargs: dict = field(default_factory=dict)
    call_count: int = 0

    async def complete(
        self,
        messages: list[Message],
        schema: type,
        cache: Optional[CacheSpec] = None,
        seed: Optional[int] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> Response[Any]:
        self.call_count += 1
        self.last_messages = messages
        self.last_cache = cache
        self.last_kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
        }
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.raw_payload is not None:
            parsed = schema.model_validate(self.raw_payload)
        else:
            assert self.payload is not None, "MockProvider needs payload or raw_payload"
            parsed = self.payload
        return Response(
            parsed=parsed,
            raw={"mock": True},
            usage=self.usage,
            latency_ms=self.latency_ms,
            model=self.model,
            cache_hit=self.cache_hit,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ctx() -> BrooksContext:
    bars = [
        Bar(
            timestamp_ns=1_000_000_000 * (i + 1),
            open=100.0 + i,
            high=100.5 + i,
            low=99.5 + i,
            close=100.2 + i,
            volume=1.0,
        )
        for i in range(5)
    ]
    return BrooksContext(symbol="BTCUSDT", primary=TFSnapshot(interval="5m", bars=bars))


@pytest.fixture
def ctx() -> BrooksContext:
    return _ctx()


@pytest.fixture
def stub_prompts() -> PromptBundle:
    """Tiny in-memory PromptBundle so tests don't read prompts/brooks/."""
    return PromptBundle(
        system_text="SYSTEM",
        concept_manual="MANUAL",
        fewshot=[],
        schema_description="SCHEMA",
    )


def _signal(**overrides) -> Signal:
    base = dict(
        pattern="h2",
        side="long",
        signal_bar_idx=4,
        entry_px=100.5,
        stop_px=99.8,
        target_px=102.0,
        probability=0.65,
        quality=0.7,
        reasoning="LLM-detected H2 bull pullback",
        source="placeholder",  # analyst overwrites this
        meta={},
    )
    base.update(overrides)
    return Signal(**base)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Registry / Protocol conformance
# ---------------------------------------------------------------------------


def test_registry_knows_llm() -> None:
    assert "llm" in AnalystRegistry.all()
    assert AnalystRegistry.get("llm") is LLMAnalyst


def test_llm_analyst_satisfies_analyst_protocol(stub_prompts) -> None:
    provider = MockProvider(payload=LLMSignalBatch(reasoning="r", signals=[]))
    analyst = LLMAnalyst(provider=provider, model="x", prompts=stub_prompts)
    assert isinstance(analyst, Analyst)
    assert analyst.name == "llm:x"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_analyze_returns_signals_with_source_and_meta(ctx, stub_prompts) -> None:
    payload = LLMSignalBatch(
        reasoning="strong setup",
        signals=[_signal(), _signal(side="short", entry_px=99.0, stop_px=100.5)],
    )
    provider = MockProvider(
        payload=payload,
        usage=Usage(
            input_tokens=200,
            output_tokens=80,
            cache_creation_tokens=10,
            cache_read_tokens=150,
        ),
        latency_ms=42.0,
        model="claude-opus-4-7",
        cache_hit=True,
    )
    analyst = LLMAnalyst(
        provider=provider, model="claude-opus-4-7", prompts=stub_prompts
    )

    out = _run(analyst.analyze(ctx))
    assert provider.call_count == 1
    assert len(out) == 2
    for s in out:
        assert s.source == "llm:claude-opus-4-7"
        assert s.meta["latency_ms"] == 42.0
        assert s.meta["input_tokens"] == 200
        assert s.meta["output_tokens"] == 80
        assert s.meta["cache_read_tokens"] == 150
        assert s.meta["cache_creation_tokens"] == 10
        assert s.meta["cache_hit"] is True
        assert s.meta["model"] == "claude-opus-4-7"


def test_analyze_passes_cache_spec_with_concept_and_fewshot_blocks(
    ctx, stub_prompts
) -> None:
    provider = MockProvider(payload=LLMSignalBatch(reasoning="", signals=[]))
    analyst = LLMAnalyst(
        provider=provider,
        model="m",
        prompts=stub_prompts,
        cache_ttl_seconds=1800,
    )
    _run(analyst.analyze(ctx))
    cache = provider.last_cache
    assert cache is not None
    assert set(cache.blocks) == {"concept_manual", "fewshot"}
    assert cache.ttl_seconds == 1800


def test_analyze_forwards_max_tokens_and_temperature(ctx, stub_prompts) -> None:
    provider = MockProvider(payload=LLMSignalBatch(reasoning="", signals=[]))
    analyst = LLMAnalyst(
        provider=provider,
        model="m",
        prompts=stub_prompts,
        max_tokens=2048,
        temperature=0.3,
    )
    _run(analyst.analyze(ctx))
    assert provider.last_kwargs["max_tokens"] == 2048
    assert provider.last_kwargs["temperature"] == 0.3


def test_analyze_builds_messages_from_prompt_bundle(ctx, stub_prompts) -> None:
    provider = MockProvider(payload=LLMSignalBatch(reasoning="", signals=[]))
    analyst = LLMAnalyst(provider=provider, model="m", prompts=stub_prompts)
    _run(analyst.analyze(ctx))
    msgs = provider.last_messages
    # PromptBundle without fewshot -> [system, user]
    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert msgs[-1].role == "user"
    # Live user content is the rendered context (mentions LTF block)
    live_text = msgs[-1].content[0].text
    assert "LTF" in live_text


def test_analyze_returns_empty_when_provider_returns_no_signals(
    ctx, stub_prompts
) -> None:
    provider = MockProvider(payload=LLMSignalBatch(reasoning="quiet", signals=[]))
    analyst = LLMAnalyst(provider=provider, model="m", prompts=stub_prompts)
    out = _run(analyst.analyze(ctx))
    assert out == []


def test_cache_miss_records_cache_hit_false(ctx, stub_prompts) -> None:
    provider = MockProvider(
        payload=LLMSignalBatch(reasoning="", signals=[_signal()]),
        usage=Usage(input_tokens=10, output_tokens=5, cache_read_tokens=0),
        cache_hit=False,
    )
    analyst = LLMAnalyst(provider=provider, model="m", prompts=stub_prompts)
    out = _run(analyst.analyze(ctx))
    assert out[0].meta["cache_hit"] is False
    assert out[0].meta["cache_read_tokens"] == 0


# ---------------------------------------------------------------------------
# Schema validation failure
# ---------------------------------------------------------------------------


def test_analyze_raises_when_provider_payload_violates_schema(
    ctx, stub_prompts
) -> None:
    """If the provider returns a dict that fails Pydantic validation, the
    error must propagate — silently dropping malformed output would mask
    upstream prompt regressions."""
    bad_payload = {
        "reasoning": "broken",
        "signals": [
            {
                "pattern": "h2",
                "side": "sideways",  # invalid Literal
                "signal_bar_idx": 0,
                "entry_px": 100.0,
                "stop_px": 99.0,
                "probability": 0.5,
                "quality": 0.5,
                "source": "x",
            }
        ],
    }
    provider = MockProvider(raw_payload=bad_payload)
    analyst = LLMAnalyst(provider=provider, model="m", prompts=stub_prompts)
    with pytest.raises(ValidationError):
        _run(analyst.analyze(ctx))


def test_analyze_raises_when_signal_violates_invariant(ctx, stub_prompts) -> None:
    """entry_px == stop_px violates Signal's model_validator."""
    bad_payload = {
        "reasoning": "edge case",
        "signals": [
            {
                "pattern": "h2",
                "side": "long",
                "signal_bar_idx": 0,
                "entry_px": 100.0,
                "stop_px": 100.0,  # invariant violation
                "probability": 0.5,
                "quality": 0.5,
                "source": "x",
            }
        ],
    }
    provider = MockProvider(raw_payload=bad_payload)
    analyst = LLMAnalyst(provider=provider, model="m", prompts=stub_prompts)
    with pytest.raises(ValidationError):
        _run(analyst.analyze(ctx))


# ---------------------------------------------------------------------------
# Provider exception propagation
# ---------------------------------------------------------------------------


def test_analyze_propagates_provider_exception(ctx, stub_prompts) -> None:
    provider = MockProvider(raise_exc=RuntimeError("upstream 500"))
    analyst = LLMAnalyst(provider=provider, model="m", prompts=stub_prompts)
    with pytest.raises(RuntimeError, match="upstream 500"):
        _run(analyst.analyze(ctx))
