"""Structural tests for the unified LLM/VLM Provider protocol."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, is_dataclass
from typing import get_args, get_type_hints

import pytest
from pydantic import BaseModel

from src.alpha.llm import (
    CacheSpec,
    ContentPart,
    ImagePart,
    Message,
    Provider,
    Response,
    TextPart,
    Usage,
)

from .conftest import MockProvider


class DummyOutput(BaseModel):
    formula: str
    score: float


class OtherOutput(BaseModel):
    label: str


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


def test_text_and_image_parts_are_dataclasses():
    t = TextPart(text="hello", cache="system")
    assert is_dataclass(t)
    assert t.text == "hello"
    assert t.cache == "system"

    t2 = TextPart(text="no cache")
    assert t2.cache is None

    img = ImagePart(data=b"\x89PNG", media_type="image/png", cache="context")
    assert is_dataclass(img)
    assert img.media_type == "image/png"
    assert img.cache == "context"

    default_img = ImagePart(data=b"bytes")
    assert default_img.media_type == "image/png"
    assert default_img.cache is None


def test_message_accepts_mixed_content_parts():
    msg = Message(
        role="user",
        content=[
            TextPart(text="Describe this chart:"),
            ImagePart(data=b"\xff\xd8\xff\xe0", media_type="image/jpeg"),
            TextPart(text="Return JSON only.", cache="instructions"),
        ],
    )
    assert msg.role == "user"
    assert len(msg.content) == 3
    assert isinstance(msg.content[0], TextPart)
    assert isinstance(msg.content[1], ImagePart)
    assert isinstance(msg.content[2], TextPart)

    # ContentPart union covers both part types.
    for part in msg.content:
        assert isinstance(part, (TextPart, ImagePart))
    # Union alias resolves to the expected members.
    assert set(ContentPart.__args__) == {TextPart, ImagePart}


def test_usage_defaults_and_fields():
    u = Usage(input_tokens=10, output_tokens=3)
    assert u.cache_creation_tokens == 0
    assert u.cache_read_tokens == 0

    u2 = Usage(input_tokens=10, output_tokens=3, cache_creation_tokens=50, cache_read_tokens=90)
    assert u2.cache_creation_tokens == 50
    assert u2.cache_read_tokens == 90


def test_cache_spec_is_serializable():
    spec = CacheSpec(blocks=["system", "schema"], ttl_seconds=3600)
    payload = asdict(spec)
    assert payload == {"blocks": ["system", "schema"], "ttl_seconds": 3600}

    default_spec = CacheSpec()
    assert default_spec.blocks == []
    assert default_spec.ttl_seconds == 300
    # Separate default instances must not share the same list object.
    default_spec.blocks.append("a")
    assert CacheSpec().blocks == []


def test_response_is_generic_over_pydantic_schema():
    parsed = DummyOutput(formula="cs_rank(close)", score=0.42)
    usage = Usage(input_tokens=12, output_tokens=4, cache_read_tokens=11)
    resp: Response[DummyOutput] = Response(
        parsed=parsed,
        raw={"id": "abc"},
        usage=usage,
        latency_ms=123.4,
        model="mock-v1",
        cache_hit=True,
    )
    assert isinstance(resp.parsed, DummyOutput)
    assert resp.parsed.score == pytest.approx(0.42)
    assert resp.usage.cache_read_tokens == 11
    assert resp.cache_hit is True
    # Parameterised generic alias preserves the type argument.
    assert get_args(Response[DummyOutput]) == (DummyOutput,)


def test_mock_provider_satisfies_protocol_runtime_and_structurally():
    provider = MockProvider(
        fixed_response=DummyOutput(formula="close", score=1.0),
        usage=Usage(input_tokens=5, output_tokens=2),
    )
    # runtime_checkable Protocol verifies the required method / attribute surface.
    assert isinstance(provider, Provider)
    assert provider.name == "mock"

    # Structural: the assignment target-typed as Provider must accept MockProvider.
    typed: Provider = provider  # noqa: F841 — static contract assertion

    async def _call() -> Response[DummyOutput]:
        return await provider.complete(
            messages=[Message(role="user", content=[TextPart(text="hi")])],
            schema=DummyOutput,
            cache=CacheSpec(blocks=["system"], ttl_seconds=300),
            seed=7,
            max_tokens=128,
            temperature=0.0,
        )

    result = asyncio.run(_call())
    assert isinstance(result, Response)
    assert isinstance(result.parsed, DummyOutput)
    assert result.parsed.formula == "close"
    assert result.usage.input_tokens == 5
    assert result.model == "mock-model-v1"
    assert provider.calls and provider.calls[0]["seed"] == 7
    assert provider.calls[0]["cache"].blocks == ["system"]


def test_mock_provider_rejects_schema_mismatch():
    provider = MockProvider(fixed_response=DummyOutput(formula="x", score=0.0))

    async def _call():
        await provider.complete(
            messages=[Message(role="user", content=[TextPart(text="hi")])],
            schema=OtherOutput,
        )

    with pytest.raises(TypeError):
        asyncio.run(_call())


def test_provider_protocol_signature_exposes_expected_names():
    # `complete` is defined on the Protocol; asserting it exists guards against
    # accidental renames of the single async entry-point.
    assert callable(Provider.complete)
    hints = get_type_hints(Provider.complete)
    assert "messages" in hints
    assert "schema" in hints
    assert "cache" in hints
    assert "return" in hints
