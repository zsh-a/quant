"""Unified analyst input — :class:`BrooksContext` and rendering.

All analysts (rule / LLM / VLM) consume a single ``BrooksContext`` value.
The context owns the rendering surface so downstream code never has to
construct prompts or charts directly:

* :meth:`BrooksContext.render_text` — compact markdown-ish listing of HTF
  and LTF bars, budgeted by token count.
* :meth:`BrooksContext.render_chart` — OHLCV PNG (Phase 4; currently
  raises :class:`NotImplementedError`).
* :meth:`BrooksContext.to_llm_messages` — conversion to the
  :mod:`src.alpha.llm.provider` ``Message`` list.

Feature / structure / regime fields are left off the ``TFSnapshot``
dataclass for now; they will be added in Phase 2 once the detectors land.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.alpha.llm.provider import Message, TextPart

__all__ = [
    "Bar",
    "TFSnapshot",
    "AccountSnapshot",
    "BrooksContext",
]


@dataclass(frozen=True)
class Bar:
    """Single OHLCV bar, aligned to an interval boundary."""

    timestamp_ns: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class TFSnapshot:
    """Recent bars at a single timeframe.

    ``bars`` is ordered oldest → newest. Phase 2 will extend this struct
    with ``features`` / ``structure`` / ``regime`` once those detectors
    exist.
    """

    interval: str
    bars: list[Bar] = field(default_factory=list)


@dataclass
class AccountSnapshot:
    equity: float
    cash: float
    open_positions: dict[str, float] = field(default_factory=dict)


@dataclass
class BrooksContext:
    """All inputs any analyst needs for one decision."""

    symbol: str
    primary: TFSnapshot
    htf: dict[str, TFSnapshot] = field(default_factory=dict)
    account: Optional[AccountSnapshot] = None
    now_ns: int = 0

    # ---- rendering ---------------------------------------------------

    def render_text(self, budget_tokens: int = 2000) -> str:
        """Render HTF summaries + LTF bar sequence as compact text.

        The output is trimmed so that the tiktoken ``cl100k_base`` token
        count stays within ``budget_tokens``. Trimming drops the oldest
        LTF bars first (HTF context is preferred); if LTF is already
        minimal, HTF bar counts are reduced next.
        """
        htf_items = sorted(self.htf.items())
        ltf_n = len(self.primary.bars)
        htf_ns: list[int] = [len(s.bars) for _, s in htf_items]

        min_ltf = min(ltf_n, 3) if ltf_n else 0
        min_htf = [min(n, 2) if n else 0 for n in htf_ns]

        text = _compose(self.primary, htf_items, ltf_n, htf_ns)
        while _count_tokens(text) > budget_tokens:
            trimmed = False
            if ltf_n > min_ltf:
                step = max(1, ltf_n // 5)
                ltf_n = max(min_ltf, ltf_n - step)
                trimmed = True
            else:
                for i, n in enumerate(htf_ns):
                    if n > min_htf[i]:
                        step = max(1, n // 5)
                        htf_ns[i] = max(min_htf[i], n - step)
                        trimmed = True
                        break
            if not trimmed:
                break
            text = _compose(self.primary, htf_items, ltf_n, htf_ns)
        return text

    def render_chart(self, include_htf: bool = True) -> bytes:
        """OHLCV → PNG. Phase 4 implementation; currently a stub."""
        raise NotImplementedError("render_chart is implemented in Phase 4")

    def to_llm_messages(self, mode: str = "text") -> list[Message]:
        """Convert the context into the provider-facing message list.

        ``mode="text"`` returns a single user message containing the
        :meth:`render_text` output. ``mode="multimodal"`` additionally
        embeds the :meth:`render_chart` PNG and therefore also raises
        :class:`NotImplementedError` until Phase 4.
        """
        if mode == "text":
            return [Message(role="user", content=[TextPart(text=self.render_text())])]
        if mode == "multimodal":
            raise NotImplementedError("multimodal mode depends on render_chart (Phase 4)")
        raise ValueError(f"unknown mode: {mode!r}")


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _classify(bar: Bar) -> tuple[str, int, str]:
    """Return (kind, body_pct, close_pos) for compact rendering."""
    rng = bar.high - bar.low
    body_abs = abs(bar.close - bar.open)
    body_pct = int(round(100 * body_abs / rng)) if rng > 0 else 0

    if body_pct <= 10:
        kind = "doji"
    elif bar.close > bar.open:
        kind = "bull"
    elif bar.close < bar.open:
        kind = "bear"
    else:
        kind = "doji"

    if rng > 0:
        rel = (bar.close - bar.low) / rng
        if rel >= 2 / 3:
            close_pos = "hi"
        elif rel >= 1 / 3:
            close_pos = "mid"
        else:
            close_pos = "lo"
    else:
        close_pos = "mid"
    return kind, body_pct, close_pos


def _format_bar(bar: Bar, idx: int, idx_width: int) -> str:
    kind, body_pct, close_pos = _classify(bar)
    idx_str = f"#{idx}".ljust(idx_width + 1)  # "+1" for the '#'
    return f"{idx_str} {kind} body={body_pct}% close={close_pos}"


def _render_block(label: str, snap: TFSnapshot, n: int) -> str:
    bars = snap.bars[-n:] if n and n < len(snap.bars) else list(snap.bars)
    header = f"== {label} {snap.interval} (last {len(bars)}) =="
    if not bars:
        return header
    total = len(bars)
    max_abs_idx = total - 1
    idx_width = len(str(max_abs_idx)) + 1  # leading '-' on every non-zero idx
    lines = [header]
    for i, bar in enumerate(bars):
        idx = -(total - 1 - i)  # newest = 0, oldest = -(total-1)
        lines.append(_format_bar(bar, idx, idx_width))
    return "\n".join(lines)


def _compose(
    primary: TFSnapshot,
    htf_items: list[tuple[str, TFSnapshot]],
    ltf_n: int,
    htf_ns: list[int],
) -> str:
    parts: list[str] = []
    for (_interval, snap), n in zip(htf_items, htf_ns):
        parts.append(_render_block("HTF", snap, n))
    parts.append(_render_block("LTF", primary, ltf_n))
    return "\n\n".join(parts)


def _count_tokens(text: str) -> int:
    """tiktoken ``cl100k_base`` count, with a char/4 fallback."""
    try:
        import tiktoken  # type: ignore

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except ImportError:
        return max(1, len(text) // 4)
