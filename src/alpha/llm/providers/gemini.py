"""Google Gemini provider (google-genai SDK)."""

from __future__ import annotations

import os
from typing import Any, Optional, TypeVar

from pydantic import BaseModel

from ..cache import CacheSpec
from ..provider import ImagePart, Message, Response, TextPart, Usage
from ._common import Timer, get_attr, raw_to_dict

T = TypeVar("T", bound=BaseModel)


class GeminiProvider:
    """Google Gemini provider backed by the async surface of `google-genai`.

    * Structured output: passes the Pydantic class directly as
      `response_schema` with `response_mime_type="application/json"`; the SDK
      populates `response.parsed` with a validated instance.
    * Prompt caching: the Gemini explicit `cached_content` API is not yet wired
      here (tracked as a follow-up). The `cache` argument is accepted for
      Protocol compatibility and the API-reported cache-read tokens are
      surfaced via `Usage.cache_read_tokens`.
    * Images: rendered as `inline_data` parts carrying raw bytes.
    """

    name = "gemini"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        client: Any = None,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if client is not None:
            self._client = client
            return
        if not self._api_key:
            raise ValueError("GOOGLE_API_KEY missing for GeminiProvider")
        from google import genai

        self._client = genai.Client(api_key=self._api_key)

    async def complete(
        self,
        messages: list[Message],
        schema: type[T],
        cache: Optional[CacheSpec] = None,
        seed: Optional[int] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> Response[T]:
        system_instruction, contents = self._split(messages)

        config: dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_schema": schema,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if seed is not None:
            config["seed"] = seed
        if system_instruction:
            config["system_instruction"] = system_instruction

        with Timer() as timer:
            raw = await self._client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

        parsed = self._extract_parsed(raw, schema)
        usage = self._extract_usage(raw)
        return Response(
            parsed=parsed,
            raw=raw_to_dict(raw),
            usage=usage,
            latency_ms=timer.elapsed_ms,
            model=get_attr(raw, "model_version", self.model),
            cache_hit=usage.cache_read_tokens > 0,
        )

    # ---- rendering --------------------------------------------------------

    def _split(self, messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        system_fragments: list[str] = []
        contents: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                for part in message.content:
                    if isinstance(part, TextPart):
                        system_fragments.append(part.text)
                continue
            role = "user" if message.role == "user" else "model"
            contents.append(
                {
                    "role": role,
                    "parts": [self._render_part(p) for p in message.content],
                }
            )
        return "\n\n".join(system_fragments), contents

    @staticmethod
    def _render_part(part: TextPart | ImagePart) -> dict[str, Any]:
        if isinstance(part, TextPart):
            return {"text": part.text}
        if isinstance(part, ImagePart):
            return {
                "inline_data": {
                    "mime_type": part.media_type,
                    "data": part.data,
                }
            }
        raise TypeError(f"Unsupported content part: {type(part).__name__}")

    # ---- response parsing -------------------------------------------------

    @staticmethod
    def _extract_parsed(raw: Any, schema: type[T]) -> T:
        parsed = get_attr(raw, "parsed")
        if isinstance(parsed, schema):
            return parsed
        text = get_attr(raw, "text")
        if isinstance(text, str) and text:
            return schema.model_validate_json(text)
        for candidate in get_attr(raw, "candidates", []) or []:
            content = get_attr(candidate, "content")
            for part in get_attr(content, "parts", []) or []:
                pt = get_attr(part, "text")
                if pt:
                    return schema.model_validate_json(pt)
        raise ValueError("Gemini response contained no parsed output or text content")

    @staticmethod
    def _extract_usage(raw: Any) -> Usage:
        u = get_attr(raw, "usage_metadata")
        if u is None:
            return Usage(input_tokens=0, output_tokens=0)
        return Usage(
            input_tokens=int(get_attr(u, "prompt_token_count", 0) or 0),
            output_tokens=int(get_attr(u, "candidates_token_count", 0) or 0),
            cache_creation_tokens=0,
            cache_read_tokens=int(get_attr(u, "cached_content_token_count", 0) or 0),
        )
