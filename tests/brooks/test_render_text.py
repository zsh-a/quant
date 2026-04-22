"""Tests for src/brooks/render/text.py — the canonical context renderer."""

from __future__ import annotations

from typing import List

import pytest

from src.brooks.context import AccountSnapshot, Bar, BrooksContext, TFSnapshot
from src.brooks.render import render_context_text


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _bar(i: int, *, bull: bool = True, base: float = 100.0, step: float = 0.25) -> Bar:
    o = base + i * step
    if bull:
        c = o + 0.8
        h = c + 0.1
        l = o - 0.1
    else:
        c = o - 0.8
        h = o + 0.1
        l = c - 0.1
    return Bar(
        timestamp_ns=1_700_000_000_000_000_000 + i * 300_000_000_000,
        open=o,
        high=h,
        low=l,
        close=c,
        volume=1_000 + i,
    )


def _bull_run(n: int, base: float = 100.0) -> List[Bar]:
    return [_bar(i, bull=True, base=base) for i in range(n)]


def _trendy_with_pullbacks(n: int, base: float = 100.0) -> List[Bar]:
    """Mostly-bull trend with short pullbacks — yields confirmed swings.

    A straight monotonic run has no fractal pivots, so the HTF swing
    line would be empty. Interleaving 2-bar pullbacks every 6 bars gives
    the fractal swing detector real high/low pivots to confirm.
    """
    bars: List[Bar] = []
    i = 0
    price = base
    ts = 1_700_000_000_000_000_000
    while len(bars) < n:
        for _ in range(6):
            o = price
            c = price + 0.8
            h = c + 0.2
            l = o - 0.15
            bars.append(
                Bar(timestamp_ns=ts, open=o, high=h, low=l, close=c, volume=1)
            )
            price = c
            ts += 300_000_000_000
            if len(bars) >= n:
                return bars
        for _ in range(2):
            o = price
            c = price - 0.5
            h = o + 0.1
            l = c - 0.2
            bars.append(
                Bar(timestamp_ns=ts, open=o, high=h, low=l, close=c, volume=1)
            )
            price = c
            ts += 300_000_000_000
            if len(bars) >= n:
                return bars
        i += 1
    return bars


def _tiktoken_count(text: str) -> int:
    import tiktoken

    return len(tiktoken.get_encoding("cl100k_base").encode(text))


@pytest.fixture
def ctx_bullish_60() -> BrooksContext:
    return BrooksContext(
        symbol="BTCUSDT",
        primary=TFSnapshot(interval="5m", bars=_trendy_with_pullbacks(60)),
        htf={
            "1h": TFSnapshot(interval="1h", bars=_trendy_with_pullbacks(40, base=80.0)),
        },
        account=AccountSnapshot(equity=10_000.0, cash=5_000.0),
        now_ns=1_700_000_000_000_000_000,
    )


@pytest.fixture
def ctx_small() -> BrooksContext:
    return BrooksContext(
        symbol="X",
        primary=TFSnapshot(interval="5m", bars=_bull_run(10)),
    )


# ---------------------------------------------------------------------------
# Block headers and content
# ---------------------------------------------------------------------------


def test_ltf_block_header_and_newest_bar(ctx_bullish_60: BrooksContext) -> None:
    text = render_context_text(ctx_bullish_60, budget_tokens=5_000)
    assert "== LTF 5m" in text
    assert "current idx=0" in text
    assert "#0" in text
    # Oldest-window index should also appear in a generous budget.
    assert "#-59" in text


def test_htf_block_has_regime_and_always_in_line(ctx_bullish_60: BrooksContext) -> None:
    text = render_context_text(ctx_bullish_60, budget_tokens=5_000)
    assert "== HTF 1h" in text
    # Regime line format: "regime=<name> (conf=<0.xx>) always_in=<side>"
    assert "regime=" in text
    assert "always_in=" in text
    assert "conf=" in text


def test_htf_block_has_swing_summary(ctx_bullish_60: BrooksContext) -> None:
    text = render_context_text(ctx_bullish_60, budget_tokens=5_000)
    # At least one of swing high/low should appear with a bars-ago count.
    assert "last_swing_" in text
    assert "bars)" in text


def test_bar_line_fields(ctx_small: BrooksContext) -> None:
    """A bar line must contain the canonical fields documented in the
    acceptance section of the task spec: kind, body=%, close=hi/mid/lo,
    ema=, leg_up/leg_down (when moving), atr=.
    """
    text = render_context_text(ctx_small, budget_tokens=5_000)
    lines = [ln for ln in text.splitlines() if ln.startswith("#")]
    assert lines
    last = lines[-1]
    # kind
    assert any(k in last for k in ("bull", "bear", "doji"))
    assert "body=" in last
    assert "%" in last
    assert "close=" in last
    assert "ema=" in last
    assert "atr=" in last


def test_bullish_run_reports_leg_up(ctx_small: BrooksContext) -> None:
    text = render_context_text(ctx_small, budget_tokens=5_000)
    # A long unbroken bullish run must produce a leg_up= field somewhere.
    assert "leg_up=" in text


def test_ltf_only_context_omits_htf_section() -> None:
    ctx = BrooksContext(
        symbol="X", primary=TFSnapshot(interval="5m", bars=_bull_run(20))
    )
    text = render_context_text(ctx, budget_tokens=5_000)
    assert "HTF" not in text
    assert "LTF 5m" in text


# ---------------------------------------------------------------------------
# Budget / trimming (acceptance criterion)
# ---------------------------------------------------------------------------


def test_respects_tight_budget_500_tokens(ctx_bullish_60: BrooksContext) -> None:
    text = render_context_text(ctx_bullish_60, budget_tokens=500)
    assert _tiktoken_count(text) <= 500
    # LTF block is never dropped entirely — the newest bar must survive.
    assert "LTF 5m" in text
    assert "#0" in text


def test_generous_budget_keeps_all_bars(ctx_bullish_60: BrooksContext) -> None:
    text = render_context_text(ctx_bullish_60, budget_tokens=10_000)
    # All 60 LTF bars present (oldest = #-59).
    assert "#-59" in text
    # Header mentions 60 bars.
    assert "(last 60 bars" in text


def test_invalid_budget_raises() -> None:
    ctx = BrooksContext(symbol="X", primary=TFSnapshot(interval="5m", bars=_bull_run(3)))
    with pytest.raises(ValueError):
        render_context_text(ctx, budget_tokens=0)


def test_budget_trims_ltf_before_htf(ctx_bullish_60: BrooksContext) -> None:
    """With a moderate budget, the HTF summary (regime/always_in line) must
    survive trimming while older LTF bars are sacrificed."""
    text = render_context_text(ctx_bullish_60, budget_tokens=800)
    assert _tiktoken_count(text) <= 800
    # HTF summary survives.
    assert "regime=" in text
    assert "always_in=" in text
    # LTF #0 survives.
    assert "#0" in text


def test_render_text_tolerates_empty_primary() -> None:
    ctx = BrooksContext(
        symbol="X", primary=TFSnapshot(interval="5m", bars=[])
    )
    text = render_context_text(ctx, budget_tokens=500)
    assert "LTF 5m" in text


def test_render_text_handles_single_bar() -> None:
    bar = Bar(timestamp_ns=0, open=100.0, high=101.0, low=99.0, close=100.5, volume=1)
    ctx = BrooksContext(
        symbol="X", primary=TFSnapshot(interval="5m", bars=[bar])
    )
    text = render_context_text(ctx, budget_tokens=500)
    assert "#0" in text
    assert "bull" in text or "bear" in text or "doji" in text


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_render_text_is_deterministic(ctx_bullish_60: BrooksContext) -> None:
    a = render_context_text(ctx_bullish_60, budget_tokens=500)
    b = render_context_text(ctx_bullish_60, budget_tokens=500)
    assert a == b
