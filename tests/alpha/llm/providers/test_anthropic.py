"""Tests for `AnthropicProvider` — schema-backed tool use, cache_control, image blocks."""

from __future__ import annotations

import asyncio
import base64
import os
from dataclasses import dataclass, field
from typing import Any

import pytest

from src.alpha.llm.provider import CacheSpec, ImagePart, Message, TextPart
from src.alpha.llm.providers.anthropic import AnthropicProvider

from .conftest import HelloOut

# ----------------------------- fake Anthropic SDK --------------------------------


@dataclass
class _ToolUseBlock:
    type: str
    name: str
    input: dict


@dataclass
class _AnthropicUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class _AnthropicResponse:
    content: list
    usage: _AnthropicUsage = field(default_factory=_AnthropicUsage)
    model: str = "claude-opus-4-7"


class _FakeMessages:
    def __init__(self, response: _AnthropicResponse) -> None:
        self.response = response
        self.last_kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> _AnthropicResponse:
        self.last_kwargs = kwargs
        return self.response


class _FakeAnthropicClient:
    def __init__(self, response: _AnthropicResponse) -> None:
        self.messages = _FakeMessages(response)


def _fake(greeting: str, **usage_kwargs: Any) -> _FakeAnthropicClient:
    resp = _AnthropicResponse(
        content=[_ToolUseBlock(type="tool_use", name="respond", input={"greeting": greeting})],
        usage=_AnthropicUsage(**usage_kwargs),
    )
    return _FakeAnthropicClient(resp)


# ----------------------------- mock round-trip -----------------------------------


def test_roundtrip_parses_tool_use_and_extracts_cache_usage():
    fake = _fake(
        "hi",
        input_tokens=11,
        output_tokens=3,
        cache_creation_input_tokens=200,
        cache_read_input_tokens=999,
    )
    provider = AnthropicProvider(model="claude-haiku-4-5", client=fake)
    messages = [
        Message(role="system", content=[TextPart(text="You are helpful.", cache="system")]),
        Message(role="user", content=[TextPart(text="Say hi.")]),
    ]
    result = asyncio.run(
        provider.complete(
            messages=messages,
            schema=HelloOut,
            cache=CacheSpec(blocks=["system"], ttl_seconds=3600),
            max_tokens=128,
        )
    )

    assert isinstance(result.parsed, HelloOut)
    assert result.parsed.greeting == "hi"
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 3
    assert result.usage.cache_creation_tokens == 200
    assert result.usage.cache_read_tokens == 999
    assert result.cache_hit is True
    assert result.latency_ms >= 0.0
    assert result.model == "claude-opus-4-7"


def test_tool_schema_and_cache_control_are_applied():
    fake = _fake("hi")
    provider = AnthropicProvider(model="claude-haiku-4-5", client=fake)
    messages = [
        Message(
            role="system",
            content=[
                TextPart(text="Always answer briefly.", cache="system"),
                TextPart(text="JSON only.", cache="system"),
            ],
        ),
        Message(role="user", content=[TextPart(text="Say hi.")]),
    ]
    asyncio.run(
        provider.complete(
            messages=messages,
            schema=HelloOut,
            cache=CacheSpec(blocks=["system"], ttl_seconds=3600),
        )
    )

    kwargs = fake.messages.last_kwargs
    assert kwargs is not None

    # The tool mirrors the Pydantic schema and is forced via tool_choice.
    tool = kwargs["tools"][0]
    assert tool["name"] == "respond"
    assert tool["input_schema"]["properties"]["greeting"]["type"] == "string"
    assert kwargs["tool_choice"] == {"type": "tool", "name": "respond"}

    # Only the last text part in the "system" block carries cache_control (1h TTL).
    system = kwargs["system"]
    assert "cache_control" not in system[0]
    assert system[1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_image_part_renders_as_base64_image_block():
    fake = _fake("ok")
    provider = AnthropicProvider(model="claude-haiku-4-5", client=fake)
    img = b"\x89PNG\r\n\x1a\nmini"
    asyncio.run(
        provider.complete(
            messages=[
                Message(
                    role="user",
                    content=[
                        ImagePart(data=img, media_type="image/png"),
                        TextPart(text="Describe the image."),
                    ],
                )
            ],
            schema=HelloOut,
        )
    )

    user_msg = fake.messages.last_kwargs["messages"][0]
    assert user_msg["role"] == "user"
    assert user_msg["content"][0]["type"] == "image"
    assert user_msg["content"][0]["source"]["media_type"] == "image/png"
    assert user_msg["content"][0]["source"]["data"] == base64.b64encode(img).decode()


def test_missing_tool_use_block_raises():
    @dataclass
    class _TextBlock:
        type: str
        text: str

    resp = _AnthropicResponse(content=[_TextBlock(type="text", text="sorry")])
    provider = AnthropicProvider(model="claude-haiku-4-5", client=_FakeAnthropicClient(resp))
    with pytest.raises(ValueError):
        asyncio.run(
            provider.complete(
                messages=[Message(role="user", content=[TextPart(text="hi")])],
                schema=HelloOut,
            )
        )


# ----------------------------- live round-trip -----------------------------------


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="no ANTHROPIC_API_KEY")
def test_live_roundtrip_hello_schema():
    provider = AnthropicProvider(model="claude-haiku-4-5")
    result = asyncio.run(
        provider.complete(
            messages=[
                Message(
                    role="user",
                    content=[TextPart(text='Respond with the JSON object {"greeting":"hi"}.')],
                )
            ],
            schema=HelloOut,
            max_tokens=256,
        )
    )
    assert isinstance(result.parsed, HelloOut)
    assert result.parsed.greeting  # non-empty
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0
    assert result.latency_ms > 0
