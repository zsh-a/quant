"""OpenAI provider — Chat Completions + strict json_schema response_format."""

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
    raw_to_dict,
    strict_openai_schema,
)

T = TypeVar("T", bound=BaseModel)


class OpenAIProvider:
    """OpenAI (or OpenAI-compatible) Chat Completions provider.

    * Structured output: `response_format={"type": "json_schema", "strict":
      true, ...}` with the Pydantic JSON schema, then validated client-side via
      `BaseModel.model_validate_json`.
    * Prompt caching: automatic at the OpenAI platform; the
      `usage.prompt_tokens_details.cached_tokens` field populates
      `Usage.cache_read_tokens` and `Response.cache_hit`.
    * Images: rendered as `image_url` content parts with a `data:` URL.
    """

    name = "openai"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Any = None,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL")
        if client is not None:
            self._client = client
            return
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY missing for OpenAIProvider")
        import openai

        kwargs: dict[str, Any] = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = openai.AsyncOpenAI(**kwargs)

    async def complete(
        self,
        messages: list[Message],
        schema: type[T],
        cache: Optional[CacheSpec] = None,
        seed: Optional[int] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> Response[T]:
        rendered = [self._render_message(m) for m in messages]
        request: dict[str, Any] = {
            "model": self.model,
            "messages": rendered,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": strict_openai_schema(schema),
                    "strict": True,
                },
            },
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if seed is not None:
            request["seed"] = seed

        with Timer() as timer:
            raw = await self._client.chat.completions.create(**request)

        parsed = schema.model_validate_json(_first_choice_text(raw))
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

    @staticmethod
    def _render_message(message: Message) -> dict[str, Any]:
        parts: list[dict[str, Any]] = []
        for part in message.content:
            if isinstance(part, TextPart):
                parts.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                b64 = encode_image_base64(part.data)
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{part.media_type};base64,{b64}"},
                    }
                )
            else:
                raise TypeError(f"Unsupported content part: {type(part).__name__}")
        return {"role": message.role, "content": parts}

    # ---- response parsing -------------------------------------------------

    @staticmethod
    def _extract_usage(raw: Any) -> Usage:
        u = get_attr(raw, "usage")
        if u is None:
            return Usage(input_tokens=0, output_tokens=0)
        input_tokens = int(get_attr(u, "prompt_tokens", 0) or 0)
        output_tokens = int(get_attr(u, "completion_tokens", 0) or 0)
        details = get_attr(u, "prompt_tokens_details")
        cached = int(get_attr(details, "cached_tokens", 0) or 0)
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=0,
            cache_read_tokens=cached,
        )


def _first_choice_text(raw: Any) -> str:
    choices = get_attr(raw, "choices") or []
    if not choices:
        raise ValueError("OpenAI response contained no choices")
    msg = get_attr(choices[0], "message")
    content = get_attr(msg, "content", "") or ""
    if isinstance(content, list):
        # Some OpenAI-compatible backends return structured parts instead of a string.
        content = "".join(
            get_attr(p, "text", "") or ""
            for p in content
            if get_attr(p, "type") == "text"
        )
    return content
