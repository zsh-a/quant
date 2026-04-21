"""Anthropic Claude provider — Messages API + tool-use structured output + cache_control."""

from __future__ import annotations

import os
from typing import Any, Optional, TypeVar

from pydantic import BaseModel

from ..cache import CacheSpec
from ..provider import ImagePart, Message, Response, TextPart, Usage
from ._common import (
    Timer,
    encode_image_base64,
    get_attr,
    pydantic_json_schema,
    raw_to_dict,
)

T = TypeVar("T", bound=BaseModel)

_STRUCTURED_TOOL_NAME = "respond"


class AnthropicProvider:
    """Anthropic Claude provider.

    * Structured output: declares a single `respond` tool whose `input_schema`
      mirrors the Pydantic class, and forces `tool_choice` to that tool so the
      model returns a JSON object in `tool_use.input`.
    * Prompt caching: for each `CacheSpec.blocks` name, the *last* content part
      carrying that cache name is decorated with `cache_control={"type":
      "ephemeral"}`. A 1-hour TTL is requested when `ttl_seconds >= 3600`.
    * Images: rendered as `image` content blocks with base64 `source`.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        client: Any = None,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if client is not None:
            self._client = client
            return
        if not self._api_key:
            raise ValueError("ANTHROPIC_API_KEY missing for AnthropicProvider")
        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=self._api_key)

    async def complete(
        self,
        messages: list[Message],
        schema: type[T],
        cache: Optional[CacheSpec] = None,
        seed: Optional[int] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> Response[T]:
        system_blocks, chat_messages = self._render(messages, cache)

        tools = [
            {
                "name": _STRUCTURED_TOOL_NAME,
                "description": f"Return the final response as an instance of {schema.__name__}.",
                "input_schema": pydantic_json_schema(schema),
            }
        ]
        request: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            "tools": tools,
            "tool_choice": {"type": "tool", "name": _STRUCTURED_TOOL_NAME},
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_blocks:
            request["system"] = system_blocks

        with Timer() as timer:
            raw = await self._client.messages.create(**request)

        parsed = self._parse_tool_output(raw, schema)
        usage = self._extract_usage(raw)
        return Response(
            parsed=parsed,
            raw=raw_to_dict(raw),
            usage=usage,
            latency_ms=timer.elapsed_ms,
            model=get_attr(raw, "model", self.model),
            cache_hit=usage.cache_read_tokens > 0,
        )

    # ---- rendering --------------------------------------------------------

    def _render(
        self,
        messages: list[Message],
        cache: Optional[CacheSpec],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        cache_blocks = set(cache.blocks) if cache else set()
        ttl_seconds = cache.ttl_seconds if cache else 300
        last_for_block = self._last_part_per_block(messages, cache_blocks)

        system_blocks: list[dict[str, Any]] = []
        chat_messages: list[dict[str, Any]] = []
        for mi, message in enumerate(messages):
            rendered_parts: list[dict[str, Any]] = []
            for pi, part in enumerate(message.content):
                block = self._render_part(part)
                if (
                    part.cache is not None
                    and part.cache in cache_blocks
                    and last_for_block.get(part.cache) == (mi, pi)
                ):
                    block["cache_control"] = _cache_control(ttl_seconds)
                rendered_parts.append(block)

            if message.role == "system":
                system_blocks.extend(rendered_parts)
            else:
                chat_messages.append({"role": message.role, "content": rendered_parts})
        return system_blocks, chat_messages

    @staticmethod
    def _last_part_per_block(
        messages: list[Message], cache_blocks: set[str]
    ) -> dict[str, tuple[int, int]]:
        last: dict[str, tuple[int, int]] = {}
        for mi, msg in enumerate(messages):
            for pi, part in enumerate(msg.content):
                if part.cache is not None and part.cache in cache_blocks:
                    last[part.cache] = (mi, pi)
        return last

    @staticmethod
    def _render_part(part: TextPart | ImagePart) -> dict[str, Any]:
        if isinstance(part, TextPart):
            return {"type": "text", "text": part.text}
        if isinstance(part, ImagePart):
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": part.media_type,
                    "data": encode_image_base64(part.data),
                },
            }
        raise TypeError(f"Unsupported content part: {type(part).__name__}")

    # ---- response parsing -------------------------------------------------

    @staticmethod
    def _parse_tool_output(raw: Any, schema: type[T]) -> T:
        for block in get_attr(raw, "content", []) or []:
            if get_attr(block, "type") == "tool_use" and get_attr(block, "name") == _STRUCTURED_TOOL_NAME:
                payload = get_attr(block, "input", {}) or {}
                return schema.model_validate(payload)
        raise ValueError(
            f"Anthropic response contained no tool_use block named {_STRUCTURED_TOOL_NAME!r}"
        )

    @staticmethod
    def _extract_usage(raw: Any) -> Usage:
        u = get_attr(raw, "usage")
        if u is None:
            return Usage(input_tokens=0, output_tokens=0)
        return Usage(
            input_tokens=int(get_attr(u, "input_tokens", 0) or 0),
            output_tokens=int(get_attr(u, "output_tokens", 0) or 0),
            cache_creation_tokens=int(get_attr(u, "cache_creation_input_tokens", 0) or 0),
            cache_read_tokens=int(get_attr(u, "cache_read_input_tokens", 0) or 0),
        )


def _cache_control(ttl_seconds: int) -> dict[str, Any]:
    cc: dict[str, Any] = {"type": "ephemeral"}
    if ttl_seconds >= 3600:
        cc["ttl"] = "1h"
    return cc
