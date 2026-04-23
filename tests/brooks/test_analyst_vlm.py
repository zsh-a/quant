"""MockProvider-driven tests for :class:`VLMAnalyst` (Phase 4.2).

Mirrors ``tests/brooks/test_analyst_llm.py`` — an in-process
``MockProvider`` exercises every branch of the analyst (happy path,
schema violation, provider failure, cache metadata) without any network
or real chart rendering dependency.

Chart rendering itself is exercised against the real ``render_chart`` so
the test validates that the analyst actually produces PNG bytes and puts
them into a provider-visible ``ImagePart``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, List, Optional

import pytest
from pydantic import BaseModel, ValidationError

from src.alpha.llm.cache import CacheSpec
from src.alpha.llm.provider import ImagePart, Message, Response, TextPart, Usage
from src.brooks.analyst import (
    AnalystRegistry,
    VLMAnalyst,
    VLMSignalBatch,
)
from src.brooks.analyst.base import Analyst
from src.brooks.context import Bar, BrooksContext, TFSnapshot
from src.brooks.prompts import PromptBundle
from src.brooks.render.chart import ChartStyle
from src.brooks.schema import Signal

# ---------------------------------------------------------------------------
# MockProvider
# ---------------------------------------------------------------------------


@dataclass
class MockProvider:
    name: str = "mock"
    payload: Optional[BaseModel] = None
    raw_payload: Optional[dict] = None
    raise_exc: Optional[Exception] = None
    usage: Usage = field(
        default_factory=lambda: Usage(
            input_tokens=150,
            output_tokens=60,
            cache_creation_tokens=0,
            cache_read_tokens=0,
        )
    )
    latency_ms: float = 18.0
    model: str = "mock-vlm"
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
        for i in range(8)
    ]
    return BrooksContext(symbol="BTCUSDT", primary=TFSnapshot(interval="5m", bars=bars))


@pytest.fixture
def ctx() -> BrooksContext:
    return _ctx()


@pytest.fixture
def stub_prompts() -> PromptBundle:
    """Tiny in-memory VLM bundle so tests don't touch the real prompts dir."""
    return PromptBundle(
        system_text="SYSTEM-VLM",
        concept_manual="MANUAL",
        fewshot=[{"user": "CTX", "assistant": {"signals": [], "annotations": []}}],
        schema_description="SCHEMA",
    )


def _signal(**overrides) -> Signal:
    base = dict(
        pattern="h2",
        side="long",
        signal_bar_idx=7,
        entry_px=100.5,
        stop_px=99.8,
        target_px=102.0,
        probability=0.65,
        quality=0.7,
        reasoning="VLM-detected H2 bull pullback",
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


def test_registry_knows_vlm() -> None:
    assert "vlm" in AnalystRegistry.all()
    assert AnalystRegistry.get("vlm") is VLMAnalyst


def test_vlm_analyst_satisfies_analyst_protocol(stub_prompts) -> None:
    provider = MockProvider(payload=VLMSignalBatch(reasoning="r", signals=[]))
    analyst = VLMAnalyst(provider=provider, model="x", prompts=stub_prompts)
    assert isinstance(analyst, Analyst)
    assert analyst.name == "vlm:x"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_analyze_returns_signals_with_source_and_chart_bytes_meta(ctx, stub_prompts) -> None:
    payload = VLMSignalBatch(
        reasoning="clean H2",
        signals=[_signal(), _signal(side="short", entry_px=99.0, stop_px=100.5)],
        annotations=[{"pattern": "H2", "bar_range": [5, 7], "label": "H2"}],
    )
    provider = MockProvider(
        payload=payload,
        usage=Usage(
            input_tokens=320,
            output_tokens=90,
            cache_creation_tokens=12,
            cache_read_tokens=200,
        ),
        latency_ms=55.0,
        model="claude-opus-4-7",
        cache_hit=True,
    )
    analyst = VLMAnalyst(provider=provider, model="claude-opus-4-7", prompts=stub_prompts)

    out = _run(analyst.analyze(ctx))
    assert provider.call_count == 1
    assert len(out) == 2
    for s in out:
        assert s.source == "vlm:claude-opus-4-7"
        assert s.meta["latency_ms"] == 55.0
        assert s.meta["input_tokens"] == 320
        assert s.meta["output_tokens"] == 90
        assert s.meta["cache_read_tokens"] == 200
        assert s.meta["cache_creation_tokens"] == 12
        assert s.meta["cache_hit"] is True
        assert s.meta["model"] == "claude-opus-4-7"
        # Chart bytes: real PNG, positive length.
        assert isinstance(s.meta["chart_bytes"], int)
        assert s.meta["chart_bytes"] > 0
        assert s.meta["annotations"] == [{"pattern": "H2", "bar_range": [5, 7], "label": "H2"}]


def test_analyze_passes_cache_spec_with_concept_and_vlm_fewshot_blocks(ctx, stub_prompts) -> None:
    provider = MockProvider(payload=VLMSignalBatch(reasoning="", signals=[]))
    analyst = VLMAnalyst(
        provider=provider,
        model="m",
        prompts=stub_prompts,
        cache_ttl_seconds=1800,
    )
    _run(analyst.analyze(ctx))
    cache = provider.last_cache
    assert cache is not None
    assert set(cache.blocks) == {"concept_manual", "fewshot_vlm"}
    assert cache.ttl_seconds == 1800


def test_analyze_forwards_max_tokens_and_temperature(ctx, stub_prompts) -> None:
    provider = MockProvider(payload=VLMSignalBatch(reasoning="", signals=[]))
    analyst = VLMAnalyst(
        provider=provider,
        model="m",
        prompts=stub_prompts,
        max_tokens=2048,
        temperature=0.25,
    )
    _run(analyst.analyze(ctx))
    assert provider.last_kwargs["max_tokens"] == 2048
    assert provider.last_kwargs["temperature"] == 0.25


def test_analyze_builds_multimodal_messages_with_image_and_text(ctx, stub_prompts) -> None:
    provider = MockProvider(payload=VLMSignalBatch(reasoning="", signals=[]))
    analyst = VLMAnalyst(provider=provider, model="m", prompts=stub_prompts)
    _run(analyst.analyze(ctx))
    msgs = provider.last_messages
    # Bundle carries one fewshot record -> [system, fewshot, user]
    assert len(msgs) == 3
    assert msgs[0].role == "system"
    assert msgs[1].role == "user"
    assert msgs[-1].role == "user"

    # System message carries concept_manual cache marker.
    assert any(isinstance(p, TextPart) and p.cache == "concept_manual" for p in msgs[0].content)

    # Fewshot message uses the VLM-specific cache namespace.
    assert len(msgs[1].content) == 1
    assert msgs[1].content[0].cache == "fewshot_vlm"

    # Live user message = [image, text]; image first so the model anchors
    # on the chart before reading the compact text summary.
    live = msgs[-1]
    assert len(live.content) == 2
    assert isinstance(live.content[0], ImagePart)
    assert live.content[0].media_type == "image/png"
    assert len(live.content[0].data) > 0
    assert isinstance(live.content[1], TextPart)
    # The rendered text still carries the LTF block marker.
    assert "LTF" in live.content[1].text
    # No cache marker on the per-call live user message.
    assert live.content[0].cache is None
    assert live.content[1].cache is None


def test_analyze_returns_empty_when_provider_returns_no_signals(ctx, stub_prompts) -> None:
    provider = MockProvider(payload=VLMSignalBatch(reasoning="quiet", signals=[]))
    analyst = VLMAnalyst(provider=provider, model="m", prompts=stub_prompts)
    out = _run(analyst.analyze(ctx))
    assert out == []


def test_cache_miss_records_cache_hit_false(ctx, stub_prompts) -> None:
    provider = MockProvider(
        payload=VLMSignalBatch(reasoning="", signals=[_signal()]),
        usage=Usage(input_tokens=30, output_tokens=10, cache_read_tokens=0),
        cache_hit=False,
    )
    analyst = VLMAnalyst(provider=provider, model="m", prompts=stub_prompts)
    out = _run(analyst.analyze(ctx))
    assert out[0].meta["cache_hit"] is False
    assert out[0].meta["cache_read_tokens"] == 0
    assert out[0].meta["chart_bytes"] > 0


def test_include_htf_false_disables_htf_inset_on_chart_style(stub_prompts) -> None:
    provider = MockProvider(payload=VLMSignalBatch(reasoning="", signals=[]))
    analyst = VLMAnalyst(
        provider=provider,
        model="m",
        prompts=stub_prompts,
        include_htf=False,
    )
    # Inspect the analyst's effective ChartStyle — the constructor must
    # flip `include_htf_inset` off when `include_htf=False`.
    assert analyst._chart_style.include_htf_inset is False


def test_custom_chart_style_is_respected(stub_prompts, ctx) -> None:
    provider = MockProvider(payload=VLMSignalBatch(reasoning="", signals=[]))
    small = ChartStyle(width=320, height=200, dpi=80, show_volume=False)
    big = ChartStyle(width=1024, height=640, dpi=100, show_volume=False)
    small_analyst = VLMAnalyst(provider=provider, model="m", prompts=stub_prompts, chart_style=small)
    _run(small_analyst.analyze(ctx))
    # Reset and run a bigger-style one.
    provider2 = MockProvider(payload=VLMSignalBatch(reasoning="", signals=[_signal()]))
    big_analyst = VLMAnalyst(provider=provider2, model="m", prompts=stub_prompts, chart_style=big)
    out = _run(big_analyst.analyze(ctx))
    # Bigger canvas → more bytes. Both positive, big > small by a clear margin.
    small_image: ImagePart = provider.last_messages[-1].content[0]
    big_image: ImagePart = provider2.last_messages[-1].content[0]
    assert len(big_image.data) > len(small_image.data)
    # And the reported chart_bytes matches the actual image length.
    assert out[0].meta["chart_bytes"] == len(big_image.data)


# ---------------------------------------------------------------------------
# Schema validation failure
# ---------------------------------------------------------------------------


def test_analyze_raises_when_provider_payload_violates_schema(ctx, stub_prompts) -> None:
    """Provider dicts that fail Pydantic validation must propagate. A silent
    drop would mask upstream prompt regressions."""
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
        "annotations": [],
    }
    provider = MockProvider(raw_payload=bad_payload)
    analyst = VLMAnalyst(provider=provider, model="m", prompts=stub_prompts)
    with pytest.raises(ValidationError):
        _run(analyst.analyze(ctx))


def test_analyze_raises_when_signal_violates_invariant(ctx, stub_prompts) -> None:
    """entry_px == stop_px violates the Signal model_validator."""
    bad_payload = {
        "reasoning": "edge case",
        "signals": [
            {
                "pattern": "h2",
                "side": "long",
                "signal_bar_idx": 0,
                "entry_px": 100.0,
                "stop_px": 100.0,
                "probability": 0.5,
                "quality": 0.5,
                "source": "x",
            }
        ],
        "annotations": [],
    }
    provider = MockProvider(raw_payload=bad_payload)
    analyst = VLMAnalyst(provider=provider, model="m", prompts=stub_prompts)
    with pytest.raises(ValidationError):
        _run(analyst.analyze(ctx))


# ---------------------------------------------------------------------------
# Provider exception propagation
# ---------------------------------------------------------------------------


def test_analyze_propagates_provider_exception(ctx, stub_prompts) -> None:
    provider = MockProvider(raise_exc=RuntimeError("upstream 500"))
    analyst = VLMAnalyst(provider=provider, model="m", prompts=stub_prompts)
    with pytest.raises(RuntimeError, match="upstream 500"):
        _run(analyst.analyze(ctx))


# ---------------------------------------------------------------------------
# PromptBundle.load_vlm integration
# ---------------------------------------------------------------------------


def test_load_vlm_loads_system_vlm_and_fewshot_vlm(tmp_path) -> None:
    """``PromptBundle.load_vlm`` must read ``system_vlm.md`` and
    ``fewshot/vlm_examples.jsonl`` — *not* the text-analyst files."""
    root = tmp_path / "prompts" / "brooks"
    (root / "fewshot").mkdir(parents=True)
    (root / "system_vlm.md").write_text("# VLM SYSTEM\nbody\n", encoding="utf-8")
    # The text-analyst file is intentionally different so we catch mixups.
    (root / "system_analyst.md").write_text("# TEXT SYSTEM (should be ignored)\n", encoding="utf-8")
    (root / "concept_manual.md").write_text("# MANUAL\n", encoding="utf-8")
    (root / "fewshot" / "vlm_examples.jsonl").write_text(
        '{"user": "CTX", "assistant": {"signals": [], "annotations": []}}\n',
        encoding="utf-8",
    )

    bundle = PromptBundle.load_vlm(root)
    assert "VLM SYSTEM" in bundle.system_text
    assert "should be ignored" not in bundle.system_text
    assert "MANUAL" in bundle.concept_manual
    assert len(bundle.fewshot) == 1
    assert bundle.fewshot[0]["user"] == "CTX"
    # Schema description defaults to VLMSignalBatch.
    assert "VLMSignalBatch" in bundle.schema_description


def test_build_messages_multimodal_emits_image_and_text_on_live_user(
    tmp_path,
) -> None:
    root = tmp_path / "prompts" / "brooks"
    (root / "fewshot").mkdir(parents=True)
    (root / "system_vlm.md").write_text("# sys\n", encoding="utf-8")
    (root / "concept_manual.md").write_text("# man\n", encoding="utf-8")
    (root / "fewshot" / "vlm_examples.jsonl").write_text(
        '{"user": "CTX", "assistant": {"signals": [], "annotations": []}}\n',
        encoding="utf-8",
    )
    bundle = PromptBundle.load_vlm(root)
    image = ImagePart(data=b"\x89PNG\r\n\x1a\n", media_type="image/png")
    msgs = bundle.build_messages_multimodal(user_text="live", user_image=image)
    assert [m.role for m in msgs] == ["system", "user", "user"]
    # Fewshot cache namespace is VLM-specific.
    assert msgs[1].content[0].cache == "fewshot_vlm"
    # Live user message = [image, text] in that order.
    live = msgs[-1]
    assert isinstance(live.content[0], ImagePart)
    assert live.content[0].data == b"\x89PNG\r\n\x1a\n"
    assert isinstance(live.content[1], TextPart)
    assert live.content[1].text == "live"


def test_build_messages_multimodal_without_fewshot_returns_two_messages(
    tmp_path,
) -> None:
    root = tmp_path / "prompts" / "brooks"
    (root / "fewshot").mkdir(parents=True)
    (root / "system_vlm.md").write_text("# sys\n", encoding="utf-8")
    (root / "concept_manual.md").write_text("# man\n", encoding="utf-8")
    # No few-shot file on disk → bundle has empty fewshot list.
    bundle = PromptBundle.load_vlm(root)
    image = ImagePart(data=b"\x89PNG\r\n\x1a\n", media_type="image/png")
    msgs = bundle.build_messages_multimodal(user_text="live", user_image=image)
    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert msgs[-1].role == "user"
    # Concept manual cache marker is still present.
    assert any(isinstance(p, TextPart) and p.cache == "concept_manual" for p in msgs[0].content)


def test_default_load_vlm_uses_real_prompt_dir() -> None:
    """Smoke test: the checked-in ``prompts/brooks/system_vlm.md`` and
    ``fewshot/vlm_examples.jsonl`` must load successfully with
    ``PromptBundle.load_vlm()`` so the analyst's default bundle works
    without any test overrides."""
    bundle = PromptBundle.load_vlm()
    assert bundle.system_text.strip().startswith("#")
    assert "Brooks" in bundle.system_text
    assert "VLM" in bundle.system_text
    assert len(bundle.fewshot) >= 3
    for ex in bundle.fewshot:
        assistant = ex["assistant"]
        assert "signals" in assistant
        assert "annotations" in assistant
