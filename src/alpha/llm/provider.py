"""
Unified LLM/VLM provider interface.

Providers (Anthropic, OpenAI, Gemini, Qwen, ...) implement the `Provider` Protocol
to deliver structured output, prompt caching, and multimodal (text + image) inputs
behind a single async `complete()` call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Literal, Optional, Protocol, TypeVar, Union, runtime_checkable

from pydantic import BaseModel

from .cache import CacheSpec

T = TypeVar("T", bound=BaseModel)


@dataclass
class TextPart:
    text: str
    cache: Optional[str] = None


@dataclass
class ImagePart:
    data: bytes
    media_type: Literal["image/png", "image/jpeg"] = "image/png"
    cache: Optional[str] = None


ContentPart = Union[TextPart, ImagePart]


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: list[ContentPart]


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass
class Response(Generic[T]):
    parsed: T
    raw: dict[str, Any] = field(default_factory=dict)
    usage: Usage = field(default_factory=lambda: Usage(input_tokens=0, output_tokens=0))
    latency_ms: float = 0.0
    model: str = ""
    cache_hit: bool = False


@runtime_checkable
class Provider(Protocol):
    name: str

    async def complete(
        self,
        messages: list[Message],
        schema: type[T],
        cache: Optional[CacheSpec] = None,
        seed: Optional[int] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> Response[T]: ...


__all__ = [
    "TextPart",
    "ImagePart",
    "ContentPart",
    "Message",
    "Usage",
    "Response",
    "Provider",
    "CacheSpec",
]
