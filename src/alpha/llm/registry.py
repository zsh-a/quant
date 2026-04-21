"""Model registry — loads `config/llm/registry.yaml` and builds Provider instances."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from .provider import Provider

_DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "llm" / "registry.yaml"
)


@dataclass
class Pricing:
    input_per_1m: float = 0.0
    output_per_1m: float = 0.0
    cache_write_per_1m: float = 0.0
    cache_read_per_1m: float = 0.0


@dataclass
class ModelSpec:
    id: str
    provider: str
    api_id: str
    context_window: int = 0
    max_output: int = 0
    supports_image: bool = False
    supports_tool_use: bool = False
    supports_cache: bool = False
    cache_min_tokens: int = 0
    pricing: Pricing = field(default_factory=Pricing)
    notes: Optional[str] = None


ProviderFactory = Callable[..., Provider]


class ModelRegistry:
    """Registry of `ModelSpec`s keyed by model id with per-provider factories."""

    def __init__(self, models: dict[str, ModelSpec]) -> None:
        self._models = models
        self._factories: dict[str, ProviderFactory] = {
            "anthropic": _build_anthropic,
            "openai": _build_openai,
            "gemini": _build_gemini,
        }

    # ---- loading ----------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Optional[os.PathLike[str] | str] = None) -> "ModelRegistry":
        yaml_path = Path(path) if path else _DEFAULT_REGISTRY_PATH
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        models: dict[str, ModelSpec] = {}
        for item in raw.get("models", []) or []:
            pricing_kwargs = dict(item.get("pricing") or {})
            remaining = {k: v for k, v in item.items() if k != "pricing"}
            spec = ModelSpec(pricing=Pricing(**pricing_kwargs), **remaining)
            if spec.id in models:
                raise ValueError(f"Duplicate model id in registry: {spec.id!r}")
            models[spec.id] = spec
        return cls(models)

    # ---- introspection ----------------------------------------------------

    def list(self, provider: Optional[str] = None) -> list[ModelSpec]:
        specs = list(self._models.values())
        if provider is not None:
            specs = [s for s in specs if s.provider == provider]
        return specs

    def ids(self) -> list[str]:
        return list(self._models.keys())

    def get(self, model_id: str) -> ModelSpec:
        if model_id not in self._models:
            raise KeyError(f"Unknown model id: {model_id!r}")
        return self._models[model_id]

    # ---- factory wiring ---------------------------------------------------

    def register_factory(self, provider: str, factory: ProviderFactory) -> None:
        self._factories[provider] = factory

    def build(self, model_id: str, **overrides: Any) -> Provider:
        spec = self.get(model_id)
        factory = self._factories.get(spec.provider)
        if factory is None:
            raise ValueError(f"No factory registered for provider {spec.provider!r}")
        return factory(spec, **overrides)


def _build_anthropic(spec: ModelSpec, **overrides: Any) -> Provider:
    from .providers.anthropic import AnthropicProvider

    return AnthropicProvider(model=spec.api_id, **overrides)


def _build_openai(spec: ModelSpec, **overrides: Any) -> Provider:
    from .providers.openai import OpenAIProvider

    return OpenAIProvider(model=spec.api_id, **overrides)


def _build_gemini(spec: ModelSpec, **overrides: Any) -> Provider:
    from .providers.gemini import GeminiProvider

    return GeminiProvider(model=spec.api_id, **overrides)


__all__ = ["Pricing", "ModelSpec", "ModelRegistry"]
