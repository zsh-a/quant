"""Compact text renderer for :class:`BrooksContext`.

The output is consumed by the LLM analyst as a ``user`` message. It is
designed to be:

* **Dense** — one line per bar, no wasted tokens.
* **Budgeted** — the tiktoken ``cl100k_base`` count of the returned
  string stays ≤ ``budget_tokens``.  Trimming drops the oldest LTF bars
  first (most recent bars have the highest signal density), then trims
  HTF bar counts, and only finally drops HTF blocks outright.
* **Feature-rich** — per-bar fields (body %, close position, EMA
  relation, leg length, ATR) and the HTF regime/always-in line are all
  computed on the fly from raw OHLC via the existing
  :mod:`src.brooks.features`, :mod:`src.brooks.structure`, and
  :mod:`src.brooks.regime` primitives, so this renderer depends only on
  ``TFSnapshot.bars`` and never on transient feature caches.

This module is the Phase 3.3 **canonical** implementation.  The
placeholder :meth:`BrooksContext.render_text` predating Phase 3.3 still
works for simple cases; new callers (in particular the LLM analyst)
should import :func:`render_context_text` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.brooks.context import BrooksContext, TFSnapshot
from src.brooks.features import BarFeatureExtractor, ExtendedBarFeatures, SwingPoint
from src.brooks.regime import BrooksRegimeClassifier, RegimeSnapshot
from src.brooks.structure import MarketStructure, MarketStructureTracker

__all__ = ["render_context_text"]


_DEFAULT_BUDGET = 2000


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_context_text(ctx: BrooksContext, budget_tokens: int = _DEFAULT_BUDGET) -> str:
    """Render ``ctx`` as compact markdown.

    See the module docstring for the budget / trimming policy. The
    output will never exceed ``budget_tokens`` cl100k tokens unless the
    minimum ("current LTF bar only, no HTF") already overshoots — in
    which case the absolute minimum is returned and the caller gets at
    least one bar of context back.
    """
    if budget_tokens <= 0:
        raise ValueError("budget_tokens must be > 0")

    primary_block = _build_tf_block(ctx.primary, is_htf=False)
    htf_blocks: List[_TFBlock] = []
    for interval, snap in sorted(ctx.htf.items()):
        htf_blocks.append(_build_tf_block(snap, is_htf=True))

    ltf_n = len(primary_block.bars_text)
    min_ltf = min(ltf_n, 1) if ltf_n else 0

    text = _compose(primary_block, htf_blocks, ltf_n)
    if _count_tokens(text) <= budget_tokens:
        return text

    while _count_tokens(text) > budget_tokens:
        trimmed = False
        # Step 1: shave LTF from the oldest side (newest bars have the
        # highest signal density).
        if ltf_n > min_ltf:
            step = max(1, ltf_n // 5)
            ltf_n = max(min_ltf, ltf_n - step)
            trimmed = True
        elif htf_blocks:
            # Step 2: drop HTF summary blocks from oldest (smallest) to
            # newest. HTF blocks are already summary-only (no per-bar
            # lines), so the only axis of trimming is block count.
            htf_blocks.pop(0)
            trimmed = True
        if not trimmed:
            break
        text = _compose(primary_block, htf_blocks, ltf_n)

    return text


# ---------------------------------------------------------------------------
# Block assembly
# ---------------------------------------------------------------------------


@dataclass
class _TFBlock:
    interval: str
    bars_text: List[str]
    summary_lines: List[str]
    is_htf: bool
    total_bars: int


def _build_tf_block(snap: TFSnapshot, *, is_htf: bool) -> _TFBlock:
    """Compute features / structure / regime for this snapshot.

    Produces one ``_TFBlock``. For LTF blocks the returned
    ``bars_text`` lists per-bar one-liners (oldest → newest). For HTF
    blocks we only need the regime / swing summary, so ``bars_text``
    stays empty.
    """
    features: List[ExtendedBarFeatures] = []
    structures: List[MarketStructure] = []

    ext = BarFeatureExtractor()
    tracker = MarketStructureTracker(ext)
    for bar in snap.bars:
        feat = ext.on_bar(bar.timestamp_ns, bar.open, bar.high, bar.low, bar.close)
        features.append(feat)
        structures.append(_clone_structure(tracker.on_features(feat)))

    if is_htf:
        bars_text: List[str] = []
        summary_lines = _htf_summary_lines(features, structures, ext.confirmed_swings)
    else:
        bars_text = [
            _format_bar_line(f, idx, len(features)) for idx, f in enumerate(features)
        ]
        summary_lines = []

    return _TFBlock(
        interval=snap.interval,
        bars_text=bars_text,
        summary_lines=summary_lines,
        is_htf=is_htf,
        total_bars=len(features),
    )


def _clone_structure(state: MarketStructure) -> MarketStructure:
    """``MarketStructureTracker.on_features`` mutates a single state.

    We keep per-bar snapshots by shallow-copying the dataclass at each
    step; the fields we rely on downstream are all primitives, lists,
    and dataclasses so a shallow copy is enough.
    """
    return MarketStructure(
        current_idx=state.current_idx,
        ema20=state.ema20,
        atr14=state.atr14,
        confirmed_swing_highs=list(state.confirmed_swing_highs),
        confirmed_swing_lows=list(state.confirmed_swing_lows),
        always_in=state.always_in,
        breakout_state=state.breakout_state,
        current_leg_dir=state.current_leg_dir,
        current_leg_start_idx=state.current_leg_start_idx,
        current_leg_length=state.current_leg_length,
        micro_channel_top=state.micro_channel_top,
        micro_channel_bot=state.micro_channel_bot,
        last_close=state.last_close,
        last_high=state.last_high,
        last_low=state.last_low,
        last_breakout_lookback_high=state.last_breakout_lookback_high,
        last_breakout_lookback_low=state.last_breakout_lookback_low,
    )


# ---------------------------------------------------------------------------
# Per-bar line
# ---------------------------------------------------------------------------


def _format_bar_line(feat: ExtendedBarFeatures, pos_in_window: int, total: int) -> str:
    """Format one ``ExtendedBarFeatures`` as a compact one-liner.

    ``pos_in_window`` runs oldest=0 → newest=total-1; the rendered index
    is ``-(total-1-pos)`` so the newest bar shows ``#0``.
    """
    idx = -(total - 1 - pos_in_window)
    width = len(str(total - 1)) + 1  # '+1' for the leading '-' on older bars
    idx_str = f"#{idx}".ljust(width + 1)

    kind = "bull" if feat.is_bull else "bear"
    if feat.is_doji:
        kind = "doji"

    close_pos = {"high": "hi", "mid": "mid", "low": "lo"}[feat.close_position]
    ema_rel = feat.ema_relation  # 'above' / 'at' / 'below'

    fields = [
        kind,
        f"body={feat.body_pct}%",
        f"close={close_pos}",
        f"ema={ema_rel}",
    ]
    if feat.leg_dir == "up":
        fields.append(f"leg_up={feat.leg_length}")
    elif feat.leg_dir == "down":
        fields.append(f"leg_down={feat.leg_length}")
    fields.append(f"atr={_fmt_price(feat.atr14)}")

    return f"{idx_str} " + " ".join(fields)


# ---------------------------------------------------------------------------
# HTF summary header
# ---------------------------------------------------------------------------


def _htf_summary_lines(
    features: List[ExtendedBarFeatures],
    structures: List[MarketStructure],
    swings: List[SwingPoint],
) -> List[str]:
    if not features or not structures:
        return []

    last_feat = features[-1]
    last_struct = structures[-1]

    classifier = BrooksRegimeClassifier()
    regime_snap = _classify_stream(classifier, features, structures)

    lines: List[str] = []

    regime_str = regime_snap.regime.value
    conf = regime_snap.confidence
    always_in = last_struct.always_in
    lines.append(
        f"regime={regime_str} (conf={conf:.2f}) always_in={always_in}"
    )

    swing_line = _swing_line(last_feat, swings)
    if swing_line:
        lines.append(swing_line)

    dist_line = _dist_line(last_feat, swings)
    if dist_line:
        lines.append(dist_line)

    return lines


def _classify_stream(
    classifier: BrooksRegimeClassifier,
    features: List[ExtendedBarFeatures],
    structures: List[MarketStructure],
) -> RegimeSnapshot:
    """Replay the bar stream through the classifier so breakout state
    is tracked correctly (BREAKOUT_MODE depends on always_in
    transitions, not just the final snapshot).
    """
    last: Optional[RegimeSnapshot] = None
    for i in range(len(features)):
        window_start = max(0, i + 1 - classifier.tr_lookback)
        window = features[window_start : i + 1]
        last = classifier.classify(window, structures[i])
    assert last is not None
    return last


def _swing_line(last_feat: ExtendedBarFeatures, swings: List[SwingPoint]) -> Optional[str]:
    confirmed = [s for s in swings if s.confirmed_at_idx <= last_feat.bar_idx]
    last_low = _latest(confirmed, "low")
    last_high = _latest(confirmed, "high")
    parts = []
    if last_low is not None:
        age = last_feat.bar_idx - last_low.bar_idx
        parts.append(f"last_swing_low={_fmt_price(last_low.price)} (-{age} bars)")
    if last_high is not None:
        age = last_feat.bar_idx - last_high.bar_idx
        parts.append(f"last_swing_high={_fmt_price(last_high.price)} (-{age} bars)")
    if not parts:
        return None
    return " | ".join(parts)


def _dist_line(last_feat: ExtendedBarFeatures, swings: List[SwingPoint]) -> Optional[str]:
    confirmed = [s for s in swings if s.confirmed_at_idx <= last_feat.bar_idx]
    last_low = _latest(confirmed, "low")
    last_high = _latest(confirmed, "high")
    atr = last_feat.atr14 if last_feat.atr14 > 0 else None
    if atr is None:
        return None
    pieces: List[str] = []
    if last_low is not None:
        d = abs(last_feat.close - last_low.price) / atr
        pieces.append(f"dist_to_htf_swing_low={d:.1f} ATR")
    if last_high is not None:
        d = abs(last_high.price - last_feat.close) / atr
        pieces.append(f"dist_to_htf_swing_high={d:.1f} ATR")
    if not pieces:
        return None
    return " | ".join(pieces)


def _latest(swings: List[SwingPoint], kind: str) -> Optional[SwingPoint]:
    for s in reversed(swings):
        if s.kind == kind:
            return s
    return None


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def _compose(
    primary: _TFBlock,
    htf_blocks: List[_TFBlock],
    ltf_n: int,
) -> str:
    parts: List[str] = []
    for block in htf_blocks:
        parts.append(_render_block(block, 0))
    parts.append(_render_block(primary, ltf_n))
    return "\n\n".join(parts)


def _render_block(block: _TFBlock, n: int) -> str:
    label = "HTF" if block.is_htf else "LTF"
    if block.is_htf:
        header = f"== {label} {block.interval} =="
    else:
        header = f"== {label} {block.interval} (last {n} bars, current idx=0) =="

    lines: List[str] = [header]
    lines.extend(block.summary_lines)
    if not block.is_htf:
        tail = (
            block.bars_text[-n:]
            if n and n < len(block.bars_text)
            else list(block.bars_text)
        )
        lines.extend(tail)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_price(p: float) -> str:
    if abs(p) >= 1000:
        return f"{p:.2f}"
    if abs(p) >= 10:
        return f"{p:.2f}"
    return f"{p:.2f}"


def _count_tokens(text: str) -> int:
    try:
        import tiktoken  # type: ignore

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except ImportError:
        return max(1, len(text) // 4)
