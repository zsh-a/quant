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

Phase 3.5 extends :class:`TFSnapshot` with optional ``features``,
``structure``, and ``regime`` fields so that HTF snapshots can be passed
directly to detectors and the EV gate without streaming reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from src.alpha.llm.provider import Message, TextPart

if TYPE_CHECKING:  # pragma: no cover — type-only imports, avoid cycles
    from src.brooks.features import ExtendedBarFeatures
    from src.brooks.regime import RegimeSnapshot
    from src.brooks.structure import MarketStructure

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
    """Recent bars at a single timeframe with optional derived state.

    ``bars`` is ordered oldest → newest. Phase 3.5 populates the optional
    ``features`` / ``structure`` / ``regime`` fields when available so
    that downstream consumers (detectors, EV gate, renderer) do not need
    to replay every bar through the streaming extractors.
    """

    interval: str
    bars: list[Bar] = field(default_factory=list)
    features: Optional["ExtendedBarFeatures"] = None
    structure: Optional["MarketStructure"] = None
    regime: Optional["RegimeSnapshot"] = None


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

    # ---- HTF helpers -------------------------------------------------

    def htf_alignment_for(self, side: str) -> str:
        """Classify HTF alignment against ``side`` as a three-way tag.

        Alignment uses the trending sign of :class:`BrooksRegime`: a
        ``bull`` regime aligns with ``long``; ``bear`` with ``short``.
        Non-trending regimes (ranges, breakout, climax, unknown) are
        treated as neutral — they never block alignment but also never
        confirm it on their own.

        Returns one of:

        * ``"conflict"`` — at least one HTF trending regime opposes ``side``
        * ``"aligned"``  — at least one HTF trending regime matches ``side``
          and none oppose it
        * ``"neutral"``  — no HTF carries a trending regime (or no HTF)
        """
        saw_agree = False
        for snap in self.htf.values():
            regime = snap.regime
            if regime is None:
                continue
            direction = _regime_trend_side(regime)
            if direction is None:
                continue
            if direction != side:
                return "conflict"
            saw_agree = True
        return "aligned" if saw_agree else "neutral"

    def htf_aligned_for(self, side: str) -> bool:
        """Boolean convenience: ``True`` iff ``htf_alignment_for == "aligned"``."""
        return self.htf_alignment_for(side) == "aligned"

    def htf_alignment_score(self, side: str) -> float:
        """Weighted alignment score in ``[-1, 1]``.

        Aggregates each HTF's trend direction with a weight equal to its
        ``RegimeSnapshot.confidence``: ``+conf`` when trending in
        ``side``, ``-conf`` when opposed, ``0`` otherwise. The final
        score is the mean over HTFs that carry a regime (``0`` if none).
        """
        values: list[float] = []
        for snap in self.htf.values():
            regime = snap.regime
            if regime is None:
                continue
            direction = _regime_trend_side(regime)
            if direction is None:
                values.append(0.0)
                continue
            values.append(regime.confidence if direction == side else -regime.confidence)
        if not values:
            return 0.0
        return sum(values) / len(values)

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


def _regime_trend_side(regime) -> Optional[str]:
    """Map a :class:`RegimeSnapshot` to ``"long"``/``"short"``/``None``.

    Bull trend regimes vote ``long``, bear trend regimes vote ``short``.
    Ranges, breakouts, climax and unknown are neutral — they return
    ``None`` so the alignment helpers can skip them.
    """
    from src.brooks.regime import BrooksRegime

    r = regime.regime if hasattr(regime, "regime") else regime
    if r in (BrooksRegime.STRONG_BULL_TREND, BrooksRegime.WEAK_BULL_TREND):
        return "long"
    if r in (BrooksRegime.STRONG_BEAR_TREND, BrooksRegime.WEAK_BEAR_TREND):
        return "short"
    return None
