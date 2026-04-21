"""Prompt-cache spec shared across LLM/VLM providers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CacheSpec:
    blocks: list[str] = field(default_factory=list)
    ttl_seconds: int = 300


__all__ = ["CacheSpec"]
