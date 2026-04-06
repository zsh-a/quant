"""Strategy registry — maps string names to factory functions.

Each strategy file registers itself at import time via the ``@register_strategy``
decorator, passing a ``StrategyMeta`` that carries both the registry key and
UI metadata (label, brief, detail, always_on).

The ``build_strategies`` function constructs instances from a set of names,
passing a typed ``StrategyInfra`` bundle instead of raw ``**kwargs``.

Usage::

    @register_strategy(StrategyMeta(
        registry_name="alpha_probe",
        label="Alpha Probe",
        brief="Experimental probe strategy",
    ))
    def _build(infra: StrategyInfra):
        return AlphaProbeStrategy(compiler=infra.compiler, llm=infra.llm_backend)

    # In service.py:
    infra = StrategyInfra(compiler=..., vm=..., ...)
    strategies = build_strategies({"mcts", "alpha_probe"}, infra)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .base import StrategyMeta


@dataclass
class StrategyInfra:
    """Typed infrastructure bundle passed to all strategy factories.

    Every field is explicitly named — no ``**kwargs`` leakage.
    """

    compiler: Any       # FormulaCompiler
    vm: Any             # StackVM
    schema: Any         # TensorSchema
    registry: Any       # OperatorRegistry
    llm_backend: Any    # LLM backend instance
    # --- strategy-specific config (with defaults) ---
    neural_sample_batch: int = 4096
    mcts_frequency: int = 1
    enum_max: int = 500
    enum_top_k: int = 30


_REGISTRY: dict[str, tuple[Callable[[StrategyInfra], Any], StrategyMeta]] = {}


def register_strategy(meta: StrategyMeta) -> Callable:
    """Decorator: register a strategy factory under ``meta.registry_name``.

    The decorated function receives a single ``StrategyInfra`` argument.
    """
    def decorator(factory_fn: Callable[[StrategyInfra], Any]) -> Callable:
        _REGISTRY[meta.registry_name] = (factory_fn, meta)
        return factory_fn
    return decorator


def available_strategies() -> list[str]:
    """Return sorted list of registered strategy names."""
    return sorted(_REGISTRY.keys())


def get_strategy_meta(name: str) -> StrategyMeta | None:
    """Return metadata for a single strategy, or None if not found."""
    entry = _REGISTRY.get(name)
    return entry[1] if entry else None


def get_all_meta() -> dict[str, StrategyMeta]:
    """Return metadata for all registered strategies."""
    return {name: entry[1] for name, entry in _REGISTRY.items()}


def build_strategies(names: set[str], infra: StrategyInfra) -> list:
    """Build strategy instances from a set of names.

    All strategies receive the same typed ``StrategyInfra`` bundle.
    """
    strategies = []
    for name in sorted(names):  # deterministic order
        entry = _REGISTRY.get(name)
        if entry is None:
            raise ValueError(
                f"Unknown strategy {name!r}. Available: {available_strategies()}"
            )
        factory_fn, _meta = entry
        strategies.append(factory_fn(infra))
    return strategies


# Backward compatibility alias
def build_extra_strategies(names: set[str], **infra: Any) -> list:
    """Legacy wrapper — prefer ``build_strategies(names, StrategyInfra(...))``.

    Converts ``**kwargs`` to ``StrategyInfra`` for backward compatibility.
    """
    typed_infra = StrategyInfra(
        compiler=infra.get("compiler"),
        vm=infra.get("vm"),
        schema=infra.get("schema"),
        registry=infra.get("registry"),
        llm_backend=infra.get("llm_backend"),
        neural_sample_batch=infra.get("neural_sample_batch", 4096),
        mcts_frequency=infra.get("mcts_frequency", 1),
        enum_max=infra.get("enum_max", 500),
        enum_top_k=infra.get("enum_top_k", 30),
    )
    return build_strategies(names, typed_infra)
