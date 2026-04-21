"""Shared test fixtures for the alpha.llm provider protocol suite."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from pydantic import BaseModel

from src.alpha.llm.provider import CacheSpec, Message, Response, Usage


class MockProvider:
    """Minimal in-memory Provider used to exercise the Protocol surface."""

    name: str = "mock"

    def __init__(
        self,
        fixed_response: BaseModel,
        usage: Optional[Usage] = None,
        cache_hit: bool = False,
        model: str = "mock-model-v1",
        latency_ms: float = 1.0,
    ) -> None:
        self._resp = fixed_response
        self._usage = usage if usage is not None else Usage(input_tokens=0, output_tokens=0)
        self._cache_hit = cache_hit
        self._model = model
        self._latency_ms = latency_ms
        self.calls: list[dict] = []

    async def complete(
        self,
        messages: list[Message],
        schema: type[BaseModel],
        cache: Optional[CacheSpec] = None,
        seed: Optional[int] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> Response:
        self.calls.append(
            {
                "messages": messages,
                "schema": schema,
                "cache": cache,
                "seed": seed,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if not isinstance(self._resp, schema):
            raise TypeError(f"fixed_response {type(self._resp).__name__} is not an instance of {schema.__name__}")
        return Response(
            parsed=self._resp,
            raw={"mock": True},
            usage=replace(self._usage),
            latency_ms=self._latency_ms,
            model=self._model,
            cache_hit=self._cache_hit,
        )
