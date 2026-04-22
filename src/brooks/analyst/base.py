"""Analyst plugin protocol and registry.

An :class:`Analyst` consumes a :class:`~src.brooks.context.BrooksContext`
and produces zero-or-more unified :class:`~src.brooks.schema.Signal`
records. Concrete analysts (rule, LLM, VLM) register themselves via the
:class:`AnalystRegistry` decorator and are built by name.
"""

from __future__ import annotations

from typing import Dict, List, Protocol, Type, runtime_checkable

from src.brooks.context import BrooksContext
from src.brooks.schema import Signal


@runtime_checkable
class Analyst(Protocol):
    """The minimal interface every analyst implements."""

    name: str

    async def analyze(self, ctx: BrooksContext) -> List[Signal]: ...


_ANALYSTS: Dict[str, Type] = {}


class AnalystRegistry:
    """Decorator + factory for :class:`Analyst` implementations."""

    @classmethod
    def register(cls, name: str):
        def wrap(klass: Type) -> Type:
            if name in _ANALYSTS and _ANALYSTS[name] is not klass:
                raise ValueError(f"AnalystRegistry: name {name!r} already registered to {_ANALYSTS[name].__name__}")
            _ANALYSTS[name] = klass
            klass.name = name
            return klass

        return wrap

    @classmethod
    def build(cls, name: str, **params) -> Analyst:
        if name not in _ANALYSTS:
            raise KeyError(f"AnalystRegistry: unknown analyst {name!r}")
        return _ANALYSTS[name](**params)

    @classmethod
    def all(cls) -> List[str]:
        return list(_ANALYSTS.keys())

    @classmethod
    def get(cls, name: str) -> Type:
        if name not in _ANALYSTS:
            raise KeyError(f"AnalystRegistry: unknown analyst {name!r}")
        return _ANALYSTS[name]
