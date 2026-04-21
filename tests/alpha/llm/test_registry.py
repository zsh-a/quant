"""Tests for `ModelRegistry` — yaml loading, lookups, factory wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.alpha.llm.provider import Provider
from src.alpha.llm.registry import ModelRegistry, ModelSpec

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_YAML = REPO_ROOT / "config" / "llm" / "registry.yaml"


def test_default_registry_yaml_loads_all_models():
    registry = ModelRegistry.from_yaml(DEFAULT_YAML)
    ids = set(registry.ids())
    # At minimum: the three hero models per provider exist.
    assert {"claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"} <= ids
    assert {"gpt-4.1", "gpt-4.1-mini"} <= ids
    assert {"gemini-2.5-pro", "gemini-2.5-flash"} <= ids


def test_model_spec_pricing_roundtrip():
    registry = ModelRegistry.from_yaml(DEFAULT_YAML)
    opus = registry.get("claude-opus-4-7")
    assert opus.provider == "anthropic"
    assert opus.api_id == "claude-opus-4-7"
    assert opus.context_window == 200000
    assert opus.supports_cache is True
    assert opus.pricing.input_per_1m == pytest.approx(15.00)
    assert opus.pricing.cache_read_per_1m == pytest.approx(1.50)


def test_list_filters_by_provider():
    registry = ModelRegistry.from_yaml(DEFAULT_YAML)
    anthropic_models = registry.list(provider="anthropic")
    assert anthropic_models and all(m.provider == "anthropic" for m in anthropic_models)


def test_registry_build_uses_registered_factory():
    spec = ModelSpec(id="fake", provider="fake-provider", api_id="fake-api")
    registry = ModelRegistry({"fake": spec})

    built: dict[str, object] = {}

    class _Stub:
        name = "fake-provider"

        def __init__(self, model: str, **_: object) -> None:
            self.model = model

        async def complete(self, *args, **kwargs):  # pragma: no cover - stub
            raise NotImplementedError

    def _factory(s: ModelSpec, **overrides) -> Provider:
        instance = _Stub(model=s.api_id, **overrides)
        built["instance"] = instance
        return instance  # type: ignore[return-value]

    registry.register_factory("fake-provider", _factory)
    provider = registry.build("fake", extra="kw")
    assert provider is built["instance"]
    assert getattr(provider, "model") == "fake-api"


def test_unknown_model_id_raises():
    registry = ModelRegistry.from_yaml(DEFAULT_YAML)
    with pytest.raises(KeyError):
        registry.get("does-not-exist")


def test_unknown_provider_factory_raises():
    spec = ModelSpec(id="rogue", provider="unregistered", api_id="x")
    registry = ModelRegistry({"rogue": spec})
    with pytest.raises(ValueError):
        registry.build("rogue")
