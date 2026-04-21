"""Tests for src/brooks/context.py — BrooksContext and its renderers."""

from __future__ import annotations

import pytest

from src.alpha.llm.provider import Message, TextPart
from src.brooks.context import (
    AccountSnapshot,
    Bar,
    BrooksContext,
    TFSnapshot,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _bar(i: int, *, bull: bool = True, base: float = 100.0) -> Bar:
    """Build a deterministic synthetic bar indexed by ``i``."""
    step = 0.25
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


def _bars(n: int, *, bull: bool = True) -> list[Bar]:
    return [_bar(i, bull=bull) for i in range(n)]


def _tiktoken_count(text: str) -> int:
    import tiktoken

    return len(tiktoken.get_encoding("cl100k_base").encode(text))


@pytest.fixture
def ctx_ltf_only() -> BrooksContext:
    return BrooksContext(
        symbol="BTCUSDT",
        primary=TFSnapshot(interval="5m", bars=_bars(60)),
    )


@pytest.fixture
def ctx_with_htf() -> BrooksContext:
    return BrooksContext(
        symbol="BTCUSDT",
        primary=TFSnapshot(interval="5m", bars=_bars(60)),
        htf={
            "1h": TFSnapshot(interval="1h", bars=_bars(24)),
            "1d": TFSnapshot(interval="1d", bars=_bars(10)),
        },
        account=AccountSnapshot(
            equity=100_000.0, cash=50_000.0, open_positions={"BTCUSDT": 0.5}
        ),
        now_ns=1_700_000_000_000_000_000,
    )


# ---------------------------------------------------------------------------
# render_text — content + token budget
# ---------------------------------------------------------------------------


def test_render_text_includes_ltf_and_htf_blocks(ctx_with_htf: BrooksContext) -> None:
    text = ctx_with_htf.render_text(budget_tokens=10_000)
    assert "== LTF 5m" in text
    assert "== HTF 1h" in text
    assert "== HTF 1d" in text
    # Newest bar index 0 must appear; at least one older negative index too.
    assert "#0 " in text
    assert "#-1" in text


def test_render_text_htf_absent_shows_only_ltf(ctx_ltf_only: BrooksContext) -> None:
    text = ctx_ltf_only.render_text(budget_tokens=10_000)
    assert "== LTF 5m" in text
    assert "HTF" not in text


def test_render_text_respects_tight_budget_with_tiktoken(
    ctx_with_htf: BrooksContext,
) -> None:
    text = ctx_with_htf.render_text(budget_tokens=500)
    assert _tiktoken_count(text) <= 500
    # LTF block is never dropped entirely even under tight budgets.
    assert "== LTF 5m" in text
    assert "#0 " in text


def test_render_text_generous_budget_keeps_all_bars(
    ctx_with_htf: BrooksContext,
) -> None:
    text = ctx_with_htf.render_text(budget_tokens=10_000)
    assert "(last 60)" in text  # full LTF
    assert "(last 24)" in text  # full HTF 1h
    assert "(last 10)" in text  # full HTF 1d


def test_render_text_bar_classification() -> None:
    # Strong bull bar: close at high, large body.
    bull = Bar(
        timestamp_ns=0, open=100.0, high=110.1, low=99.9, close=110.0, volume=1
    )
    bear = Bar(
        timestamp_ns=0, open=110.0, high=110.1, low=99.9, close=100.0, volume=1
    )
    doji = Bar(
        timestamp_ns=0, open=100.0, high=101.0, low=99.0, close=100.05, volume=1
    )
    ctx = BrooksContext(
        symbol="X",
        primary=TFSnapshot(interval="5m", bars=[doji, bear, bull]),
    )
    text = ctx.render_text(budget_tokens=10_000)
    lines = [ln for ln in text.splitlines() if ln.startswith("#")]
    assert len(lines) == 3
    # Oldest (doji) → newest (bull). Most recent is index 0.
    assert "doji" in lines[0]
    assert "bear" in lines[1] and "close=lo" in lines[1]
    assert "bull" in lines[2] and "close=hi" in lines[2] and "#0" in lines[2]


def test_render_text_handles_zero_range_bar() -> None:
    flat = Bar(timestamp_ns=0, open=100.0, high=100.0, low=100.0, close=100.0, volume=0)
    ctx = BrooksContext(
        symbol="X", primary=TFSnapshot(interval="5m", bars=[flat])
    )
    text = ctx.render_text(budget_tokens=10_000)
    assert "doji" in text
    assert "body=0%" in text
    assert "close=mid" in text


# ---------------------------------------------------------------------------
# render_chart — Phase 4 stub
# ---------------------------------------------------------------------------


def test_render_chart_raises_not_implemented(ctx_ltf_only: BrooksContext) -> None:
    with pytest.raises(NotImplementedError):
        ctx_ltf_only.render_chart()


# ---------------------------------------------------------------------------
# to_llm_messages — provider interop
# ---------------------------------------------------------------------------


def test_to_llm_messages_text_mode_shape(ctx_with_htf: BrooksContext) -> None:
    messages = ctx_with_htf.to_llm_messages(mode="text")
    assert isinstance(messages, list) and len(messages) == 1
    msg = messages[0]
    assert isinstance(msg, Message)
    assert msg.role == "user"
    assert len(msg.content) == 1
    part = msg.content[0]
    assert isinstance(part, TextPart)
    assert "== LTF 5m" in part.text
    assert "== HTF 1h" in part.text


def test_to_llm_messages_multimodal_raises(ctx_ltf_only: BrooksContext) -> None:
    with pytest.raises(NotImplementedError):
        ctx_ltf_only.to_llm_messages(mode="multimodal")


def test_to_llm_messages_unknown_mode_raises(ctx_ltf_only: BrooksContext) -> None:
    with pytest.raises(ValueError):
        ctx_ltf_only.to_llm_messages(mode="ocr")


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------


def test_bar_is_frozen() -> None:
    b = _bar(0)
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError, subclass of AttributeError
        b.close = 0.0  # type: ignore[misc]


def test_brooks_context_defaults() -> None:
    ctx = BrooksContext(symbol="X", primary=TFSnapshot(interval="5m", bars=[]))
    assert ctx.htf == {}
    assert ctx.account is None
    assert ctx.now_ns == 0
