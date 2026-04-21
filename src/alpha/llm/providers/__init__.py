"""Concrete LLM/VLM providers behind the unified `Provider` protocol.

Anthropic / OpenAI / Gemini providers each implement `complete()` with structured
output (schema-enforced), prompt caching metadata, and multimodal (text+image)
input. SDK imports are deferred to provider `__init__` so the vendor packages are
only required when the corresponding provider is actually instantiated.
"""

from __future__ import annotations

from .anthropic import AnthropicProvider
from .gemini import GeminiProvider
from .openai import OpenAIProvider

__all__ = ["AnthropicProvider", "OpenAIProvider", "GeminiProvider"]
