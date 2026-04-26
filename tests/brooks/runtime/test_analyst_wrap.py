"""ThrottledAnalyst wraps an analyst with a min-gap throttle.

The wrapper is uniform across modes — live + replay differ only by which
clock they pair the throttle with. Tests stub the inner analyst with a
counter so we can verify how many calls actually reach it.
"""

import asyncio
from dataclasses import dataclass

from src.brooks.runtime.analyst_wrap import ThrottledAnalyst, build_throttled_analyst
from src.brooks.runtime.clock import BarClock
from src.brooks.runtime.throttle import Throttle


@dataclass
class _Ctx:
    symbol: str = "BTC/USDT"


class _CountingAnalyst:
    name = "llm:test-analyst"

    def __init__(self):
        self.calls = 0
        self.last_ctx = None

    async def analyze(self, ctx):
        self.calls += 1
        self.last_ctx = ctx
        return [{"signal": self.calls}]


class _RuleAnalyst:
    name = "rule"

    async def analyze(self, ctx):  # pragma: no cover — exercised via build helper
        return []


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


def test_first_call_passes_through():
    inner = _CountingAnalyst()
    clock = BarClock()
    wrapped = ThrottledAnalyst(inner, Throttle(min_gap_seconds=300), clock)
    clock.advance_to(1000.0)
    result = _run(wrapped.analyze(_Ctx()))
    assert inner.calls == 1
    assert result == [{"signal": 1}]


def test_call_within_gap_skipped():
    inner = _CountingAnalyst()
    clock = BarClock()
    wrapped = ThrottledAnalyst(inner, Throttle(min_gap_seconds=300), clock)

    clock.advance_to(1000.0)
    _run(wrapped.analyze(_Ctx()))

    clock.advance_to(1100.0)  # only 100s of bar-time later
    result = _run(wrapped.analyze(_Ctx()))
    assert inner.calls == 1, "second call should be skipped by throttle"
    assert result == []


def test_call_beyond_gap_reaches_inner():
    inner = _CountingAnalyst()
    clock = BarClock()
    wrapped = ThrottledAnalyst(inner, Throttle(min_gap_seconds=300), clock)

    clock.advance_to(1000.0)
    _run(wrapped.analyze(_Ctx()))

    clock.advance_to(1500.0)  # 500s — beyond the gap
    _run(wrapped.analyze(_Ctx()))

    assert inner.calls == 2


def test_per_symbol_buckets_independent():
    inner = _CountingAnalyst()
    clock = BarClock()
    wrapped = ThrottledAnalyst(
        inner,
        Throttle(min_gap_seconds=300),
        clock,
        key_fn=lambda ctx: ctx.symbol,
    )
    clock.advance_to(1000.0)
    _run(wrapped.analyze(_Ctx(symbol="BTC/USDT")))
    _run(wrapped.analyze(_Ctx(symbol="ETH/USDT")))
    assert inner.calls == 2


def test_inner_exception_still_marks_called():
    """We mark first so an exception cannot bypass the throttle."""

    class _BoomAnalyst:
        name = "llm:boom"

        async def analyze(self, ctx):
            raise RuntimeError("network down")

    clock = BarClock()
    wrapped = ThrottledAnalyst(_BoomAnalyst(), Throttle(min_gap_seconds=300), clock)
    clock.advance_to(1000.0)
    try:
        _run(wrapped.analyze(_Ctx()))
    except RuntimeError:
        pass
    # Within the gap, even the next attempt is throttled — good.
    clock.advance_to(1100.0)
    result = _run(wrapped.analyze(_Ctx()))
    assert result == []


def test_build_helper_skips_rule_analysts():
    """build_throttled_analyst returns rule analysts untouched."""
    rule = _RuleAnalyst()
    out = build_throttled_analyst(rule, min_gap_seconds=300, clock=BarClock())
    assert out is rule


def test_build_helper_wraps_llm_analysts():
    inner = _CountingAnalyst()
    out = build_throttled_analyst(inner, min_gap_seconds=300, clock=BarClock())
    assert isinstance(out, ThrottledAnalyst)
    assert out.inner is inner
