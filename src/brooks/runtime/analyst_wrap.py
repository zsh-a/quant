"""DI-friendly throttled analyst — replaces the legacy ``analyze`` monkey-patch.

The live task previously rewrote ``analyst.analyze`` in-place to enforce a
wall-clock LLM rate limit. Now we wrap the analyst in :class:`ThrottledAnalyst`
and slot it into the strategy via ``strategy._analyst = ThrottledAnalyst(...)``.
The wrapper's behaviour is uniform across modes; live + replay differ only
by which :class:`~src.brooks.runtime.clock.Clock` they pair the wrapper with.
"""

from __future__ import annotations

from typing import Any, Callable, List

from loguru import logger

from src.brooks.runtime.clock import Clock
from src.brooks.runtime.throttle import Throttle


class ThrottledAnalyst:
    """Wrap an analyst's ``async analyze(ctx)`` with a min-gap throttle.

    ``key_fn`` decides which throttle bucket a context lands in — typically
    ``f"{ctx.symbol}:{interval}"`` so each (symbol, interval) pair throttles
    independently.
    """

    def __init__(
        self,
        inner_analyst: Any,
        throttle: Throttle,
        clock: Clock,
        key_fn: Callable[[Any], str] = lambda ctx: getattr(ctx, "symbol", ""),
    ):
        self._inner = inner_analyst
        self._throttle = throttle
        self._clock = clock
        self._key_fn = key_fn
        # Surface the wrapped analyst's identity so strategy code that
        # introspects ``analyst.name`` keeps working transparently.
        self.name = getattr(inner_analyst, "name", "")

    @property
    def inner(self) -> Any:
        return self._inner

    async def analyze(self, ctx: Any) -> List[Any]:
        key = self._key_fn(ctx)
        now = self._clock.now_seconds()
        if self._throttle.should_skip(key, now):
            logger.debug(
                "ThrottledAnalyst skip key={} gap<{}s",
                key,
                self._throttle.min_gap,
            )
            return []
        # Mark first so an exception in inner.analyze still spaces calls.
        self._throttle.mark_called(key, now)
        result = await self._inner.analyze(ctx)
        return result

    def __getattr__(self, item: str) -> Any:
        # Forward any other attribute lookup (e.g. analyst-specific config
        # readers) to the wrapped analyst.
        return getattr(self._inner, item)


# Convenience: build a ready-to-use throttled analyst pair.
def build_throttled_analyst(
    inner_analyst: Any,
    *,
    min_gap_seconds: float,
    clock: Clock,
    key_fn: Callable[[Any], str] = lambda ctx: getattr(ctx, "symbol", ""),
) -> ThrottledAnalyst:
    name = getattr(inner_analyst, "name", "") or ""
    if not (name.startswith("llm:") or name.startswith("vlm:")):
        # Rule analysts are free — return the inner instance untouched.
        # Returning the original lets ``strategy._analyst`` keep its exact
        # type for any code that does isinstance() checks.
        return inner_analyst  # type: ignore[return-value]
    return ThrottledAnalyst(
        inner_analyst,
        Throttle(min_gap_seconds=min_gap_seconds),
        clock,
        key_fn=key_fn,
    )


__all__ = ["ThrottledAnalyst", "build_throttled_analyst"]
