"""Tests for `OpenAIProvider` — strict json_schema response_format, cached_tokens, images."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any

import pytest

from src.alpha.llm.provider import ImagePart, Message, TextPart
from src.alpha.llm.providers.openai import OpenAIProvider

from .conftest import HelloOut

# ------------------------------- fake OpenAI SDK ---------------------------------


@dataclass
class _Message:
    content: str = ""


@dataclass
class _Choice:
    message: _Message


@dataclass
class _PromptDetails:
    cached_tokens: int = 0


@dataclass
class _OpenAIUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tokens_details: _PromptDetails = field(default_factory=_PromptDetails)


@dataclass
class _OpenAIResponse:
    choices: list
    usage: _OpenAIUsage = field(default_factory=_OpenAIUsage)
    model: str = "gpt-4.1-mini"


class _FakeCompletions:
    def __init__(self, response: _OpenAIResponse) -> None:
        self.response = response
        self.last_kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> _OpenAIResponse:
        self.last_kwargs = kwargs
        return self.response


class _FakeChat:
    def __init__(self, response: _OpenAIResponse) -> None:
        self.completions = _FakeCompletions(response)


class _FakeOpenAIClient:
    def __init__(self, response: _OpenAIResponse) -> None:
        self.chat = _FakeChat(response)


def _fake(greeting: str, **usage_kwargs: Any) -> _FakeOpenAIClient:
    cached = usage_kwargs.pop("cached_tokens", 0)
    resp = _OpenAIResponse(
        choices=[_Choice(message=_Message(content=json.dumps({"greeting": greeting})))],
        usage=_OpenAIUsage(
            prompt_tokens=usage_kwargs.pop("prompt_tokens", 0),
            completion_tokens=usage_kwargs.pop("completion_tokens", 0),
            prompt_tokens_details=_PromptDetails(cached_tokens=cached),
        ),
    )
    return _FakeOpenAIClient(resp)


# ------------------------------- mock round-trip ---------------------------------


def test_roundtrip_parses_json_schema_and_cached_tokens():
    fake = _fake("hi", prompt_tokens=50, completion_tokens=3, cached_tokens=40)
    provider = OpenAIProvider(model="gpt-4.1-mini", client=fake)
    result = asyncio.run(
        provider.complete(
            messages=[Message(role="user", content=[TextPart(text='Output {"greeting":"hi"}.')])],
            schema=HelloOut,
            seed=42,
            max_tokens=128,
        )
    )
    assert result.parsed.greeting == "hi"
    assert result.usage.input_tokens == 50
    assert result.usage.output_tokens == 3
    assert result.usage.cache_read_tokens == 40
    assert result.cache_hit is True
    assert result.model == "gpt-4.1-mini"
    assert result.latency_ms >= 0.0


def test_response_format_is_strict_json_schema_and_seed_forwarded():
    fake = _fake("hi")
    provider = OpenAIProvider(model="gpt-4.1-mini", client=fake)
    asyncio.run(
        provider.complete(
            messages=[Message(role="user", content=[TextPart(text="hi")])],
            schema=HelloOut,
            seed=7,
        )
    )
    kwargs = fake.chat.completions.last_kwargs
    assert kwargs is not None
    assert kwargs["seed"] == 7

    rf = kwargs["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "HelloOut"
    assert rf["json_schema"]["strict"] is True

    schema = rf["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["greeting"]


def test_image_part_renders_as_data_url():
    fake = _fake("ok")
    provider = OpenAIProvider(model="gpt-4.1-mini", client=fake)
    img = b"\xff\xd8\xff"
    asyncio.run(
        provider.complete(
            messages=[
                Message(
                    role="user",
                    content=[
                        TextPart(text="Describe:"),
                        ImagePart(data=img, media_type="image/jpeg"),
                    ],
                )
            ],
            schema=HelloOut,
        )
    )
    msg = fake.chat.completions.last_kwargs["messages"][0]
    image_block = msg["content"][1]
    assert image_block["type"] == "image_url"
    assert image_block["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_list_content_response_is_concatenated():
    @dataclass
    class _TextPart:
        type: str
        text: str

    @dataclass
    class _MessageWithParts:
        content: list

    resp = _OpenAIResponse(
        choices=[
            _Choice(
                message=_MessageWithParts(
                    content=[
                        _TextPart(type="text", text='{"greeting":'),
                        _TextPart(type="text", text='"yo"}'),
                    ]
                )
            )
        ],
        usage=_OpenAIUsage(prompt_tokens=1, completion_tokens=2),
    )
    provider = OpenAIProvider(model="gpt-4.1-mini", client=_FakeOpenAIClient(resp))
    result = asyncio.run(
        provider.complete(
            messages=[Message(role="user", content=[TextPart(text="hi")])],
            schema=HelloOut,
        )
    )
    assert result.parsed.greeting == "yo"


# ------------------------------- live round-trip ---------------------------------


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="no OPENAI_API_KEY")
def test_live_roundtrip_hello_schema():
    provider = OpenAIProvider(model="gpt-4.1-mini")
    result = asyncio.run(
        provider.complete(
            messages=[Message(role="user", content=[TextPart(text='Return JSON {"greeting":"hi"}.')])],
            schema=HelloOut,
            max_tokens=64,
        )
    )
    assert result.parsed.greeting  # non-empty string
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0
    assert result.latency_ms > 0
