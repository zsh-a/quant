"""L3 — Brooks 7-state market regime classifier.

Replaces the 3-state (bull/bear/sideways) detector in
``src/analysis/regime_detector.py`` with Brooks' canonical taxonomy:

    STRONG_BULL_TREND / WEAK_BULL_TREND
    STRONG_BEAR_TREND / WEAK_BEAR_TREND
    TIGHT_TRADING_RANGE / BROAD_TRADING_RANGE
    BREAKOUT_MODE / CLIMAX
    (UNKNOWN when data is insufficient)

The classifier consumes outputs from the L1 :class:`BarFeatureExtractor`
(``ExtendedBarFeatures``) and the L2 :class:`MarketStructureTracker`
(``MarketStructure``).  It is designed for streaming use: call
:meth:`classify` once per bar with the most recent ``tr_lookback`` features
and the current structure snapshot.  The classifier keeps a tiny amount of
internal state (last breakout bar + side) so that ``BREAKOUT_MODE`` can
decay naturally across several bars instead of flashing on for one bar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .features import ExtendedBarFeatures
from .structure import MarketStructure

__all__ = [
    "BrooksRegime",
    "RegimeSnapshot",
    "BrooksRegimeClassifier",
]


class BrooksRegime(str, Enum):
    STRONG_BULL_TREND = "strong_bull_trend"
    WEAK_BULL_TREND = "weak_bull_trend"
    STRONG_BEAR_TREND = "strong_bear_trend"
    WEAK_BEAR_TREND = "weak_bear_trend"
    TIGHT_TRADING_RANGE = "tight_trading_range"
    BROAD_TRADING_RANGE = "broad_trading_range"
    BREAKOUT_MODE = "breakout_mode"
    CLIMAX = "climax"
    UNKNOWN = "unknown"


@dataclass
class RegimeSnapshot:
    """Classifier output for a single bar."""

    regime: BrooksRegime
    confidence: float
    reasons: List[str] = field(default_factory=list)

    consecutive_trend_bars: int = 0
    ema_atr_distance: float = 0.0
    swing_range_atr: float = 0.0
    bar_overlap_ratio: float = 0.0

    def to_dict(self) -> dict:
        return {
            "regime": self.regime.value,
            "confidence": round(self.confidence, 4),
            "reasons": list(self.reasons),
            "consecutive_trend_bars": self.consecutive_trend_bars,
            "ema_atr_distance": round(self.ema_atr_distance, 4),
            "swing_range_atr": round(self.swing_range_atr, 4),
            "bar_overlap_ratio": round(self.bar_overlap_ratio, 4),
        }


class BrooksRegimeClassifier:
    """Stateful 7-state Brooks regime classifier.

    Parameters
    ----------
    climax_consecutive_bars
        Number of same-direction trend bars required to mark a CLIMAX.
    climax_ema_dist_atr
        ``|close - EMA20| / ATR14`` threshold for CLIMAX.
    strong_trend_retracement_pct
        Max pullback (as a fraction of the current leg) tolerated by a
        strong trend (typical Brooks value ≈ 0.3).
    tight_tr_overlap_threshold
        Average pairwise bar overlap threshold separating tight vs broad TR.
    tr_lookback
        Window of bars used for trend / TR statistics.  Also the minimum
        number of bars required before the classifier emits anything other
        than ``UNKNOWN``.
    breakout_decay_bars
        BREAKOUT_MODE lasts this many bars after the last breakout.
    """

    def __init__(
        self,
        climax_consecutive_bars: int = 3,
        climax_ema_dist_atr: float = 2.0,
        strong_trend_retracement_pct: float = 0.3,
        tight_tr_overlap_threshold: float = 0.5,
        tr_lookback: int = 20,
        breakout_decay_bars: int = 5,
    ):
        if climax_consecutive_bars < 1:
            raise ValueError("climax_consecutive_bars must be >= 1")
        if climax_ema_dist_atr <= 0:
            raise ValueError("climax_ema_dist_atr must be > 0")
        if not 0 < strong_trend_retracement_pct < 1:
            raise ValueError("strong_trend_retracement_pct must be in (0, 1)")
        if not 0 < tight_tr_overlap_threshold < 1:
            raise ValueError("tight_tr_overlap_threshold must be in (0, 1)")
        if tr_lookback < 2:
            raise ValueError("tr_lookback must be >= 2")
        if breakout_decay_bars < 1:
            raise ValueError("breakout_decay_bars must be >= 1")

        self.climax_consecutive_bars = climax_consecutive_bars
        self.climax_ema_dist_atr = climax_ema_dist_atr
        self.strong_trend_retracement_pct = strong_trend_retracement_pct
        self.tight_tr_overlap_threshold = tight_tr_overlap_threshold
        self.tr_lookback = tr_lookback
        self.breakout_decay_bars = breakout_decay_bars

        self._last_breakout_idx: Optional[int] = None
        self._last_breakout_side: Optional[str] = None  # "long" | "short"
        self._prev_always_in: str = "neutral"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        features: List[ExtendedBarFeatures],
        structure: MarketStructure,
    ) -> RegimeSnapshot:
        if not features:
            return RegimeSnapshot(
                regime=BrooksRegime.UNKNOWN,
                confidence=0.0,
                reasons=["no_features"],
            )

        last = features[-1]
        current_idx = last.bar_idx

        # Track breakout transitions — BREAKOUT_MODE is about *fresh* shifts
        # in always-in state, not every bar where price extends in the same
        # direction.  Climax bars within an ongoing trend keep firing
        # ``breakout_state=bull_breakout`` but they are not new breakouts.
        if structure.always_in != self._prev_always_in and structure.always_in in ("long", "short"):
            self._last_breakout_idx = current_idx
            self._last_breakout_side = structure.always_in
        self._prev_always_in = structure.always_in

        # --- insufficient history → UNKNOWN ---
        if len(features) < self.tr_lookback:
            return RegimeSnapshot(
                regime=BrooksRegime.UNKNOWN,
                confidence=0.0,
                reasons=[f"insufficient_data ({len(features)} < {self.tr_lookback})"],
                consecutive_trend_bars=_count_consecutive_trend(features),
                ema_atr_distance=last.ema_atr_distance,
            )

        window = features[-self.tr_lookback :]
        atr = max(last.atr14, 1e-9)

        consec_trend = _count_consecutive_trend(features)
        ema_dist = last.ema_atr_distance
        swing_range_atr = (max(f.high for f in window) - min(f.low for f in window)) / atr
        overlap_ratio = _avg_overlap_ratio(window)

        metrics = dict(
            consecutive_trend_bars=consec_trend,
            ema_atr_distance=ema_dist,
            swing_range_atr=swing_range_atr,
            bar_overlap_ratio=overlap_ratio,
        )

        bars_since_breakout: Optional[int] = (
            current_idx - self._last_breakout_idx if self._last_breakout_idx is not None else None
        )

        # --- CLIMAX: strongest overextension signal takes priority ---
        climax = self._check_climax(features, ema_dist)
        if climax is not None:
            regime, reasons, conf = climax
            return RegimeSnapshot(regime=regime, confidence=conf, reasons=reasons, **metrics)

        # --- BREAKOUT_MODE: recently broke out, not yet decayed ---
        if bars_since_breakout is not None and bars_since_breakout < self.breakout_decay_bars:
            reasons = [
                f"bars_since_breakout={bars_since_breakout} < {self.breakout_decay_bars}",
                f"side={self._last_breakout_side}",
            ]
            # Fresh breakout → high confidence; decays toward 0.5 as it ages.
            conf = 1.0 - (bars_since_breakout / self.breakout_decay_bars) * 0.5
            return RegimeSnapshot(regime=BrooksRegime.BREAKOUT_MODE, confidence=conf, reasons=reasons, **metrics)

        # --- Trend regimes (always_in long/short) ---
        if structure.always_in == "long":
            regime, reasons, conf = self._classify_trend(window, "up")
            return RegimeSnapshot(regime=regime, confidence=conf, reasons=reasons, **metrics)

        if structure.always_in == "short":
            regime, reasons, conf = self._classify_trend(window, "down")
            return RegimeSnapshot(regime=regime, confidence=conf, reasons=reasons, **metrics)

        # --- Trading ranges (always_in neutral) ---
        if overlap_ratio > self.tight_tr_overlap_threshold:
            margin = overlap_ratio - self.tight_tr_overlap_threshold
            conf = _confidence_from_margin(margin, scale=2.0)
            return RegimeSnapshot(
                regime=BrooksRegime.TIGHT_TRADING_RANGE,
                confidence=conf,
                reasons=[
                    "always_in=neutral",
                    f"overlap_ratio={overlap_ratio:.2f} > {self.tight_tr_overlap_threshold}",
                ],
                **metrics,
            )

        if swing_range_atr > 2.0:
            conf = _confidence_from_margin(swing_range_atr - 2.0, scale=0.5)
            return RegimeSnapshot(
                regime=BrooksRegime.BROAD_TRADING_RANGE,
                confidence=conf,
                reasons=[
                    "always_in=neutral",
                    f"swing_range_atr={swing_range_atr:.2f} > 2.0",
                ],
                **metrics,
            )

        # Neutral but neither tight nor broad — fall back to UNKNOWN.
        return RegimeSnapshot(
            regime=BrooksRegime.UNKNOWN,
            confidence=0.2,
            reasons=["neutral_without_tr_signature"],
            **metrics,
        )

    # ------------------------------------------------------------------
    # Rule helpers
    # ------------------------------------------------------------------

    def _check_climax(
        self, features: List[ExtendedBarFeatures], ema_dist: float
    ) -> Optional[tuple[BrooksRegime, List[str], float]]:
        n = self.climax_consecutive_bars
        if len(features) < n:
            return None
        tail = features[-n:]
        if not all(f.is_trend and f.body_pct >= 60 for f in tail):
            return None
        bull_climax = all(f.is_bull for f in tail)
        bear_climax = all(not f.is_bull for f in tail)
        if not (bull_climax or bear_climax):
            return None
        if bull_climax and ema_dist <= self.climax_ema_dist_atr:
            return None
        if bear_climax and ema_dist >= -self.climax_ema_dist_atr:
            return None

        side = "bull" if bull_climax else "bear"
        reasons = [
            f"consecutive_trend_bars={n} (all {side}, body≥60%)",
            f"ema_atr_distance={ema_dist:.2f} vs threshold ±{self.climax_ema_dist_atr}",
        ]
        margin = abs(ema_dist) - self.climax_ema_dist_atr
        conf = _confidence_from_margin(margin, scale=1.0)
        return BrooksRegime.CLIMAX, reasons, conf

    def _classify_trend(
        self, window: List[ExtendedBarFeatures], direction: str
    ) -> tuple[BrooksRegime, List[str], float]:
        if direction == "up":
            trend_ratio = sum(1 for f in window if f.is_bull and f.is_trend) / len(window)
            pullback = _leg_pullback_ratio(window, "up")
            strong_regime = BrooksRegime.STRONG_BULL_TREND
            weak_regime = BrooksRegime.WEAK_BULL_TREND
            always_in_label = "long"
        else:
            trend_ratio = sum(1 for f in window if (not f.is_bull) and f.is_trend) / len(window)
            pullback = _leg_pullback_ratio(window, "down")
            strong_regime = BrooksRegime.STRONG_BEAR_TREND
            weak_regime = BrooksRegime.WEAK_BEAR_TREND
            always_in_label = "short"

        reasons = [
            f"always_in={always_in_label}",
            f"trend_bar_ratio={trend_ratio:.2f}",
            f"pullback_ratio={pullback:.2f}",
        ]

        strong = trend_ratio >= 0.6 and pullback < self.strong_trend_retracement_pct
        if strong:
            # Confidence grows as both conditions clear their thresholds.
            ratio_margin = trend_ratio - 0.6
            pullback_margin = self.strong_trend_retracement_pct - pullback
            ratio_conf = _confidence_from_margin(ratio_margin, scale=5.0)
            pullback_conf = _confidence_from_margin(pullback_margin, scale=5.0)
            conf = min(ratio_conf, pullback_conf)
            return strong_regime, reasons + ["strong_trend_criteria_met"], conf

        # Weak trend confidence peaks when structure clearly says always_in
        # but the trend-quality metrics are mediocre (≈ 0.5).  Push it above
        # 0.5 when the picture is clearly trending-but-not-strong.
        weakness = 0.3 * trend_ratio + 0.3 * (1 - min(1.0, pullback))
        conf = max(0.3, min(0.9, 0.45 + weakness))
        return weak_regime, reasons, conf


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _count_consecutive_trend(features: List[ExtendedBarFeatures]) -> int:
    """Count consecutive same-direction trend bars at the tail."""
    count = 0
    direction: Optional[bool] = None  # True=bull, False=bear
    for f in reversed(features):
        if not f.is_trend:
            break
        if direction is None:
            direction = f.is_bull
        elif f.is_bull != direction:
            break
        count += 1
    return count


def _avg_overlap_ratio(window: List[ExtendedBarFeatures]) -> float:
    """Average pairwise bar overlap / combined-range over the window."""
    if len(window) < 2:
        return 0.0
    ratios: List[float] = []
    for a, b in zip(window[:-1], window[1:]):
        lo = max(a.low, b.low)
        hi = min(a.high, b.high)
        overlap = max(0.0, hi - lo)
        combined = max(a.high, b.high) - min(a.low, b.low)
        if combined > 0:
            ratios.append(overlap / combined)
    if not ratios:
        return 0.0
    return sum(ratios) / len(ratios)


def _leg_pullback_ratio(window: List[ExtendedBarFeatures], direction: str) -> float:
    """Deepest retracement within the window, as a fraction of the leg.

    For ``direction="up"`` we walk the window and track the running max of
    the highs seen so far; the retrace at bar *i* is ``running_max - close_i``.
    The deepest such retrace, normalized by the full leg size
    (``max_high − min_low``), is returned.  This avoids penalizing the latest
    bar when it just made a new high (``post`` would otherwise collapse to
    the single bar and count its intra-bar range as a pullback).
    """
    if len(window) < 2:
        return 0.0

    if direction == "up":
        running_peak = -float("inf")
        max_retrace = 0.0
        for f in window:
            if f.high > running_peak:
                running_peak = f.high
            retrace = running_peak - f.close
            if retrace > max_retrace:
                max_retrace = retrace
        leg_size = running_peak - min(f.low for f in window)
        if leg_size <= 0:
            return 0.0
        return max_retrace / leg_size

    # direction == "down"
    running_trough = float("inf")
    max_retrace = 0.0
    for f in window:
        if f.low < running_trough:
            running_trough = f.low
        retrace = f.close - running_trough
        if retrace > max_retrace:
            max_retrace = retrace
    leg_size = max(f.high for f in window) - running_trough
    if leg_size <= 0:
        return 0.0
    return max_retrace / leg_size


def _confidence_from_margin(margin: float, scale: float = 1.0) -> float:
    """Smooth margin → confidence transform.

    At the threshold (``margin=0``) returns ``0.5``.  As ``margin`` grows
    positive the return approaches ``1.0``; if ``margin`` is negative the
    return stays above ``0.0`` but below ``0.5``.  Callers only invoke this
    when a rule already fired, so the contract is "higher margin ⇒ higher
    confidence".
    """
    return 1.0 / (1.0 + math.exp(-margin * scale * 2.0))
