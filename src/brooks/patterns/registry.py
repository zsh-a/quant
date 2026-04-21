"""Name → :class:`PatternDetector` registry.

Use as a class decorator::

    @PatternRegistry.register("h2")
    class H2Detector(PatternDetector):
        ...

Then build instances with::

    det = PatternRegistry.build("h2", max_leg_bars=12)
"""

from __future__ import annotations

from typing import Dict, List, Type

from src.brooks.patterns.base import PatternDetector

_REGISTRY: Dict[str, Type[PatternDetector]] = {}


class PatternRegistry:
    """Decorator + factory for :class:`PatternDetector` subclasses."""

    @classmethod
    def register(cls, name: str):
        def decorator(klass: Type[PatternDetector]) -> Type[PatternDetector]:
            if name in _REGISTRY and _REGISTRY[name] is not klass:
                raise ValueError(
                    f"PatternRegistry: name {name!r} already registered "
                    f"to {_REGISTRY[name].__name__}"
                )
            _REGISTRY[name] = klass
            klass.name = name
            return klass

        return decorator

    @classmethod
    def build(cls, name: str, **params) -> PatternDetector:
        if name not in _REGISTRY:
            raise KeyError(f"PatternRegistry: unknown detector {name!r}")
        return _REGISTRY[name](**params)

    @classmethod
    def all(cls) -> List[str]:
        return list(_REGISTRY.keys())

    @classmethod
    def get(cls, name: str) -> Type[PatternDetector]:
        if name not in _REGISTRY:
            raise KeyError(f"PatternRegistry: unknown detector {name!r}")
        return _REGISTRY[name]
