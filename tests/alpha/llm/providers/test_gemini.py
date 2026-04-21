"""Tests for `GeminiProvider` — response_schema parse, inline_data image, usage_metadata."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

import pytest

from src.alpha.llm.provider import ImagePart, Message, TextPart
from src.alpha.llm.providers.gemini import GeminiProvider

from .conftest import HelloOut


# ----------------------------- fake google-genai ---------------------------------


@dataclass
class _GeminiUsage:
    prompt_token_count: int = 0
    candidates_token_count: int = 0
    cached_content_token_count: int = 0


@dataclass
class _GeminiResponse:
    parsed: Any = None
    text: str = ""
    usage_metadata: _GeminiUsage = field(default_factory=_GeminiUsage)
    model_version: str = "gemini-2.5-flash"


class _FakeModels:
    def __init__(self, response: _GeminiResponse) -> None:
        self.response = response
        self.last_kwargs: dict[str, Any] | None = None

    async def generate_content(self, **kwargs: Any) -> _GeminiResponse:
        self.last_kwargs = kwargs
        return self.response


class _FakeAio:
    def __init__(self, response: _GeminiResponse) -> None:
        self.models = _FakeModels(response)


class _FakeGeminiClient:
    def __init__(self, response: _GeminiResponse) -> None:
        self.aio = _FakeAio(response)


# ----------------------------- mock round-trip -----------------------------------


def test_roundtrip_uses_parsed_and_usage_metadata():
    resp = _GeminiResponse(
        parsed=HelloOut(greeting="hi"),
        usage_metadata=_GeminiUsage(
            prompt_token_count=8,
            candidates_token_count=2,
            cached_content_token_count=5,
        ),
    )
    fake = _FakeGeminiClient(resp)
    provider = GeminiProvider(model="gemini-2.5-flash", client=fake)
    result = asyncio.run(
        provider.complete(
            messages=[Message(role="user", content=[TextPart(text="Say hi")])],
            schema=HelloOut,
            seed=123,
        )
    )
    assert isinstance(result.parsed, HelloOut)
    assert result.parsed.greeting == "hi"
    assert result.usage.input_tokens == 8
    assert result.usage.output_tokens == 2
    assert result.usage.cache_read_tokens == 5
    assert result.cache_hit is True
    assert result.model == "gemini-2.5-flash"
    assert result.latency_ms >= 0.0


def test_config_forwards_schema_and_seed_and_system_instruction():
    fake = _FakeGeminiClient(_GeminiResponse(parsed=HelloOut(greeting="ok")))
    provider = GeminiProvider(model="gemini-2.5-flash", client=fake)
    asyncio.run(
        provider.complete(
            messages=[
                Message(role="system", content=[TextPart(text="Answer briefly.")]),
                Message(role="user", content=[TextPart(text="Hi")]),
            ],
            schema=HelloOut,
            seed=99,
            max_tokens=64,
            temperature=0.2,
        )
    )
    cfg = fake.aio.models.last_kwargs["config"]
    assert cfg["response_mime_type"] == "application/json"
    assert cfg["response_schema"] is HelloOut
    assert cfg["seed"] == 99
    assert cfg["max_output_tokens"] == 64
    assert cfg["temperature"] == pytest.approx(0.2)
    assert cfg["system_instruction"] == "Answer briefly."

    # System messages are split out of `contents`.
    contents = fake.aio.models.last_kwargs["contents"]
    assert len(contents) == 1
    assert contents[0]["role"] == "user"


def test_falls_back_to_text_when_parsed_is_missing():
    resp = _GeminiResponse(
        parsed=None,
        text='{"greeting":"hey"}',
        usage_metadata=_GeminiUsage(prompt_token_count=3, candidates_token_count=2),
    )
    provider = GeminiProvider(model="gemini-2.5-flash", client=_FakeGeminiClient(resp))
    result = asyncio.run(
        provider.complete(
            messages=[Message(role="user", content=[TextPart(text="Hi")])],
            schema=HelloOut,
        )
    )
    assert result.parsed.greeting == "hey"


def test_image_part_uses_inline_data_with_raw_bytes():
    fake = _FakeGeminiClient(_GeminiResponse(parsed=HelloOut(greeting="ok")))
    provider = GeminiProvider(model="gemini-2.5-flash", client=fake)
    img = b"\x89PNG\r\n"
    asyncio.run(
        provider.complete(
            messages=[
                Message(
                    role="user",
                    content=[
                        ImagePart(data=img, media_type="image/png"),
                        TextPart(text="What is this?"),
                    ],
                )
            ],
            schema=HelloOut,
        )
    )
    parts = fake.aio.models.last_kwargs["contents"][0]["parts"]
    assert parts[0]["inline_data"]["mime_type"] == "image/png"
    assert parts[0]["inline_data"]["data"] == img
    assert parts[1]["text"] == "What is this?"


def test_missing_parsed_and_text_raises():
    resp = _GeminiResponse(parsed=None, text="", usage_metadata=_GeminiUsage())
    provider = GeminiProvider(model="gemini-2.5-flash", client=_FakeGeminiClient(resp))
    with pytest.raises(ValueError):
        asyncio.run(
            provider.complete(
                messages=[Message(role="user", content=[TextPart(text="hi")])],
                schema=HelloOut,
            )
        )


# ----------------------------- live round-trip -----------------------------------


@pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
    reason="no GOOGLE_API_KEY / GEMINI_API_KEY",
)
def test_live_roundtrip_hello_schema():
    provider = GeminiProvider(model="gemini-2.5-flash")
    result = asyncio.run(
        provider.complete(
            messages=[
                Message(
                    role="user",
                    content=[TextPart(text='Respond with JSON {"greeting":"hi"}.')],
                )
            ],
            schema=HelloOut,
            max_tokens=64,
        )
    )
    assert result.parsed.greeting  # non-empty string
    assert result.usage.input_tokens > 0
    assert result.latency_ms > 0
