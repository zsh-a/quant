"""Strategy registry — maps string names to factory functions.

Each strategy file registers itself at import time via the ``@register_strategy``
decorator.  The ``build_extra_strategies`` function constructs instances from
a set of names, passing shared infrastructure objects as keyword arguments.

Usage::

    @register_strategy("alpha_probe")
    def _build(*, compiler, vm, schema, llm_backend, **kw):
        return AlphaProbeStrategy(compiler=compiler, llm=llm_backend)

    # In service.py:
    strategies = build_extra_strategies({"mcts", "alpha_probe"}, compiler=..., vm=..., ...)
"""

from __future__ import annotations

from typing import Any, Callable

_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_strategy(name: str) -> Callable:
    """Decorator: register a strategy factory under *name*."""
    def decorator(factory_fn: Callable[..., Any]) -> Callable[..., Any]:
        _REGISTRY[name] = factory_fn
        return factory_fn
    return decorator


def available_strategies() -> list[str]:
    """Return sorted list of registered strategy names."""
    return sorted(_REGISTRY.keys())


def build_extra_strategies(names: set[str], **infra: Any) -> list:
    """Build strategy instances from a set of names.

    All keyword arguments (compiler, vm, schema, llm_backend, etc.) are
    forwarded to each factory function.  Factories pick what they need
    via ``**kwargs``.
    """
    strategies = []
    for name in sorted(names):  # deterministic order
        factory = _REGISTRY.get(name)
        if factory is None:
            raise ValueError(
                f"Unknown strategy {name!r}. Available: {available_strategies()}"
            )
        strategies.append(factory(**infra))
    return strategies
