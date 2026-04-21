"""L1 — Extended bar features for Brooks price action.

Builds on `src.analysis.bar_features` (`BarFeatures`) by adding time-series features
required by the higher layers:

- K-bar fractal swing detection (with ``confirmed_as_of_idx`` to prevent look-ahead)
- Breakout flags (N-bar high/low)
- Consecutive bull/bear bar counters
- EMA/ATR distance
- Current leg length
- Gap up / down

Design notes
------------
* Features are computed with a *streaming* ``BarFeatureExtractor`` — internal state is
  an append-only bar buffer; ``on_bar(bar)`` returns the ``ExtendedBarFeatures`` for
  the latest bar plus any newly *confirmed* swing points.
* Swing confirmation lags ``K`` bars by definition: we only know bar ``i`` is a swing
  high once we've seen ``K`` bars after it without exceeding ``high[i]``. Consumers
  (L2/L3) must only use swings whose ``confirmed_at_idx <= current_idx``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Literal, Optional


@dataclass
class ExtendedBarFeatures:
    """Per-bar numeric features (no look-ahead)."""

    bar_idx: int
    timestamp_ns: int  # unix ns — avoid datetime serialization

    open: float
    high: float
    low: float
    close: float

    body_pct: int  # 0..100 — fraction of range taken by body
    is_bull: bool
    is_trend: bool  # body_pct > 50
    is_doji: bool  # body_pct <= 10
    is_inside_bar: bool
    is_outside_bar: bool
    is_reversal_bar: bool  # >= 40% wick
    close_position: Literal["high", "mid", "low"]

    ema20: float
    atr14: float
    ema_atr_distance: float  # (close - ema) / atr
    ema_relation: Literal["above", "at", "below"]

    is_breakout_up_20: bool
    is_breakout_down_20: bool
    consecutive_bull_bars: int
    consecutive_bear_bars: int

    leg_length: int  # # of bars in current same-direction leg
    leg_dir: Literal["up", "down", "flat"]

    gap_up: bool
    gap_down: bool

    def to_dict(self) -> dict:
        return {
            "bar_idx": self.bar_idx,
            "timestamp_ns": self.timestamp_ns,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "body_pct": self.body_pct,
            "is_bull": self.is_bull,
            "is_trend": self.is_trend,
            "is_doji": self.is_doji,
            "is_inside_bar": self.is_inside_bar,
            "is_outside_bar": self.is_outside_bar,
            "is_reversal_bar": self.is_reversal_bar,
            "close_position": self.close_position,
            "ema20": round(self.ema20, 6),
            "atr14": round(self.atr14, 6),
            "ema_atr_distance": round(self.ema_atr_distance, 4),
            "ema_relation": self.ema_relation,
            "is_breakout_up_20": self.is_breakout_up_20,
            "is_breakout_down_20": self.is_breakout_down_20,
            "consecutive_bull_bars": self.consecutive_bull_bars,
            "consecutive_bear_bars": self.consecutive_bear_bars,
            "leg_length": self.leg_length,
            "leg_dir": self.leg_dir,
            "gap_up": self.gap_up,
            "gap_down": self.gap_down,
        }


@dataclass
class SwingPoint:
    """A confirmed (past) swing high or low.

    ``bar_idx`` is the index of the actual pivot bar; ``confirmed_at_idx`` is the
    index of the bar at which this swing became confirmed (always bar_idx + K).
    """

    bar_idx: int
    price: float
    kind: Literal["high", "low"]
    confirmed_at_idx: int
    timestamp_ns: int = 0

    def to_dict(self) -> dict:
        return {
            "bar_idx": self.bar_idx,
            "price": round(self.price, 6),
            "kind": self.kind,
            "confirmed_at_idx": self.confirmed_at_idx,
            "timestamp_ns": self.timestamp_ns,
        }


@dataclass
class _BarCache:
    """Minimal bar record held in the streaming buffer."""

    idx: int
    ts_ns: int
    open: float
    high: float
    low: float
    close: float


class BarFeatureExtractor:
    """Stream features bar-by-bar.

    Call :meth:`on_bar` once per completed bar (never on an in-progress bar).
    """

    def __init__(
        self,
        swing_k: int = 3,
        ema_period: int = 20,
        atr_period: int = 14,
        breakout_lookback: int = 20,
    ):
        if swing_k < 1:
            raise ValueError("swing_k must be >= 1")
        self.swing_k = swing_k
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.breakout_lookback = breakout_lookback

        self._bars: List[_BarCache] = []
        self._ema: Optional[float] = None
        self._atr: Optional[float] = None
        self._atr_warm: Deque[float] = deque(maxlen=atr_period)

        self._consec_bull = 0
        self._consec_bear = 0
        self._leg_dir: Literal["up", "down", "flat"] = "flat"
        self._leg_start_idx = 0

        self.confirmed_swings: List[SwingPoint] = []
        self._last_confirmed_swing_high: Optional[SwingPoint] = None
        self._last_confirmed_swing_low: Optional[SwingPoint] = None

    # ---- public API -----------------------------------------------------

    @property
    def bar_count(self) -> int:
        return len(self._bars)

    def on_bar(self, ts_ns: int, o: float, h: float, l: float, c: float) -> ExtendedBarFeatures:
        """Ingest a bar, return features. Look-ahead free."""
        idx = len(self._bars)
        prev = self._bars[-1] if self._bars else None

        self._update_ema(c)
        self._update_atr(h, l, prev.close if prev else c)

        is_bull = c >= o
        rng = h - l
        body = abs(c - o)
        body_pct = int(body / rng * 100) if rng > 0 else 0
        is_trend = body_pct > 50
        is_doji = body_pct <= 10

        if rng > 0:
            pos = (c - l) / rng
            close_position: Literal["high", "mid", "low"] = "high" if pos >= 0.67 else "low" if pos <= 0.33 else "mid"
        else:
            close_position = "mid"

        is_inside = prev is not None and h <= prev.high and l >= prev.low
        is_outside = prev is not None and h > prev.high and l < prev.low
        if rng > 0:
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            is_reversal = upper_wick >= 0.4 * rng or lower_wick >= 0.4 * rng
        else:
            is_reversal = False

        ema = self._ema if self._ema is not None else c
        atr = self._atr if self._atr is not None else max(rng, 1e-9)
        ema_atr_dist = (c - ema) / atr if atr > 0 else 0.0
        ema_rel: Literal["above", "at", "below"] = (
            "above" if ema_atr_dist > 0.3 else "below" if ema_atr_dist < -0.3 else "at"
        )

        lookback_start = max(0, idx - self.breakout_lookback)
        window = self._bars[lookback_start:idx]  # excludes current bar
        if window:
            prior_high = max(b.high for b in window)
            prior_low = min(b.low for b in window)
            is_breakout_up = h > prior_high
            is_breakout_down = l < prior_low
        else:
            is_breakout_up = False
            is_breakout_down = False

        if is_bull:
            self._consec_bull += 1
            self._consec_bear = 0
        else:
            self._consec_bear += 1
            self._consec_bull = 0

        leg_dir, leg_length = self._update_leg(idx, h, l, prev)

        gap_up = prev is not None and o > prev.high
        gap_down = prev is not None and o < prev.low

        self._bars.append(_BarCache(idx=idx, ts_ns=ts_ns, open=o, high=h, low=l, close=c))
        self._confirm_swings()

        return ExtendedBarFeatures(
            bar_idx=idx,
            timestamp_ns=ts_ns,
            open=o,
            high=h,
            low=l,
            close=c,
            body_pct=body_pct,
            is_bull=is_bull,
            is_trend=is_trend,
            is_doji=is_doji,
            is_inside_bar=is_inside,
            is_outside_bar=is_outside,
            is_reversal_bar=is_reversal,
            close_position=close_position,
            ema20=ema,
            atr14=atr,
            ema_atr_distance=ema_atr_dist,
            ema_relation=ema_rel,
            is_breakout_up_20=is_breakout_up,
            is_breakout_down_20=is_breakout_down,
            consecutive_bull_bars=self._consec_bull,
            consecutive_bear_bars=self._consec_bear,
            leg_length=leg_length,
            leg_dir=leg_dir,
            gap_up=gap_up,
            gap_down=gap_down,
        )

    def bars_since_last_swing(self, kind: Literal["high", "low"], current_idx: int) -> Optional[int]:
        swing = self._last_confirmed_swing_high if kind == "high" else self._last_confirmed_swing_low
        if swing is None or swing.confirmed_at_idx > current_idx:
            return None
        return current_idx - swing.bar_idx

    def last_confirmed_swings(
        self, current_idx: int, n: int = 10, kind: Optional[Literal["high", "low"]] = None
    ) -> List[SwingPoint]:
        out = [s for s in self.confirmed_swings if s.confirmed_at_idx <= current_idx]
        if kind is not None:
            out = [s for s in out if s.kind == kind]
        return out[-n:]

    # ---- internals ------------------------------------------------------

    def _update_ema(self, close: float) -> None:
        alpha = 2.0 / (self.ema_period + 1)
        if self._ema is None:
            self._ema = close
        else:
            self._ema = alpha * close + (1 - alpha) * self._ema

    def _update_atr(self, high: float, low: float, prev_close: float) -> None:
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        self._atr_warm.append(tr)
        if self._atr is None:
            if len(self._atr_warm) >= self.atr_period:
                self._atr = sum(self._atr_warm) / self.atr_period
        else:
            self._atr = (self._atr * (self.atr_period - 1) + tr) / self.atr_period

    def _update_leg(
        self, idx: int, h: float, l: float, prev: Optional[_BarCache]
    ) -> tuple[Literal["up", "down", "flat"], int]:
        if prev is None:
            self._leg_dir = "flat"
            self._leg_start_idx = idx
            return "flat", 1

        going_up = h > prev.high and l >= prev.low
        going_down = l < prev.low and h <= prev.high

        if going_up and self._leg_dir != "up":
            self._leg_dir = "up"
            self._leg_start_idx = idx
        elif going_down and self._leg_dir != "down":
            self._leg_dir = "down"
            self._leg_start_idx = idx
        elif not going_up and not going_down:
            pass  # inside / outside bar — preserve leg direction but don't extend length

        return self._leg_dir, max(1, idx - self._leg_start_idx + 1)

    def _confirm_swings(self) -> None:
        """Promote fractal pivots to confirmed swings once K bars have passed."""
        K = self.swing_k
        if len(self._bars) < 2 * K + 1:
            return
        # Candidate pivot = bar at position (len - K - 1); we now have K bars on both sides.
        pivot_idx = len(self._bars) - K - 1
        pivot = self._bars[pivot_idx]
        left = self._bars[pivot_idx - K : pivot_idx]
        right = self._bars[pivot_idx + 1 : pivot_idx + 1 + K]

        if all(pivot.high >= b.high for b in left) and all(pivot.high > b.high for b in right):
            sp = SwingPoint(
                bar_idx=pivot.idx,
                price=pivot.high,
                kind="high",
                confirmed_at_idx=pivot.idx + K,
                timestamp_ns=pivot.ts_ns,
            )
            self.confirmed_swings.append(sp)
            self._last_confirmed_swing_high = sp

        if all(pivot.low <= b.low for b in left) and all(pivot.low < b.low for b in right):
            sp = SwingPoint(
                bar_idx=pivot.idx,
                price=pivot.low,
                kind="low",
                confirmed_at_idx=pivot.idx + K,
                timestamp_ns=pivot.ts_ns,
            )
            self.confirmed_swings.append(sp)
            self._last_confirmed_swing_low = sp
