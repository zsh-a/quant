"""Four-stage stop ladder.

Walks a position through monotonic stop upgrades:

    initial → break_even → partial_trail → swing_trail

* ``initial``: hold the entry stop until unrealized ≥ 1R.
* ``break_even``: stop moved to entry_px (+/- tick).
* ``partial_trail``: once unrealized ≥ ``partial_trail_trigger_r``, stop tracks
  the min of recent ``partial_trail_lookback`` bar lows (longs) or the max of
  recent highs (shorts).
* ``swing_trail``: each newly confirmed swing (low for longs, high for shorts)
  snaps the stop just inside that swing.

Stops only ever move in the favourable direction — never retreat.
"""

from __future__ import annotations

from typing import List, Optional

from src.brooks.features import SwingPoint
from src.brooks.risk.state import PositionState


class StopLadder:
    def __init__(
        self,
        partial_trail_trigger_r: float = 1.5,
        partial_trail_lookback: int = 10,
        use_swing_trail: bool = True,
        tick_size: float = 0.0,
    ):
        if partial_trail_trigger_r <= 1.0:
            raise ValueError("partial_trail_trigger_r must be > 1.0 (BE is the 1R stage)")
        if partial_trail_lookback < 1:
            raise ValueError("partial_trail_lookback must be >= 1")
        self.partial_trail_trigger_r = partial_trail_trigger_r
        self.partial_trail_lookback = partial_trail_lookback
        self.use_swing_trail = use_swing_trail
        self.tick_size = tick_size

    def update(
        self,
        pos: PositionState,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        confirmed_swings: List[SwingPoint],
    ) -> Optional[float]:
        """Advance the ladder for one bar. Returns a new stop_px if it moved."""
        if pos.qty_open <= 0 or pos.one_r <= 0:
            return None

        self._track_extrema(pos, bar_high, bar_low)
        r = pos.unrealized_r(bar_close)
        if r > pos.max_unrealized_r:
            pos.max_unrealized_r = r

        proposed: Optional[float] = None

        # Stage 1 → 2: initial → break_even at >= 1R.
        if pos.ladder_stage == "initial" and r >= 1.0:
            proposed = self._pick_stop(proposed, pos, pos.entry_px)
            pos.ladder_stage = "break_even"
            pos.breakeven_moved = True

        # Stage 2 → 3: break_even → partial_trail at >= trigger R.
        if pos.ladder_stage == "break_even" and r >= self.partial_trail_trigger_r:
            pos.ladder_stage = "partial_trail"

        # partial_trail: track recent bar extrema.
        if pos.ladder_stage == "partial_trail":
            trail = self._partial_trail_stop(pos)
            if trail is not None:
                proposed = self._pick_stop(proposed, pos, trail)

        # Stage 3 → 4: partial_trail → swing_trail on first usable swing.
        # (Also runs while in swing_trail to keep promoting.)
        if self.use_swing_trail and pos.ladder_stage in ("partial_trail", "swing_trail"):
            swing_stop = self._swing_trail_stop(pos, confirmed_swings)
            if swing_stop is not None:
                proposed = self._pick_stop(proposed, pos, swing_stop)
                pos.ladder_stage = "swing_trail"

        if proposed is None:
            return None

        if self._is_improvement(pos, proposed):
            pos.stop_px = proposed
            return proposed
        return None

    # ---- internals ------------------------------------------------------

    def _track_extrema(self, pos: PositionState, bar_high: float, bar_low: float) -> None:
        pos.highest_since_entry = max(pos.highest_since_entry or bar_high, bar_high)
        pos.lowest_since_entry = min(pos.lowest_since_entry or bar_low, bar_low)

        if pos.side == "long":
            pos.recent_lows.append(bar_low)
            if len(pos.recent_lows) > self.partial_trail_lookback:
                del pos.recent_lows[: len(pos.recent_lows) - self.partial_trail_lookback]
        else:
            pos.recent_highs.append(bar_high)
            if len(pos.recent_highs) > self.partial_trail_lookback:
                del pos.recent_highs[: len(pos.recent_highs) - self.partial_trail_lookback]

    def _partial_trail_stop(self, pos: PositionState) -> Optional[float]:
        if pos.side == "long":
            if not pos.recent_lows:
                return None
            return min(pos.recent_lows) - self.tick_size
        if not pos.recent_highs:
            return None
        return max(pos.recent_highs) + self.tick_size

    def _swing_trail_stop(self, pos: PositionState, confirmed_swings: List[SwingPoint]) -> Optional[float]:
        candidate: Optional[float] = None
        kind = "low" if pos.side == "long" else "high"
        used = pos.used_swing_lows if pos.side == "long" else pos.used_swing_highs
        for s in confirmed_swings:
            if s.kind != kind:
                continue
            if s.bar_idx <= pos.entry_bar_idx:
                continue
            if s.bar_idx in used:
                continue
            used.append(s.bar_idx)
            offset = -self.tick_size if pos.side == "long" else self.tick_size
            level = s.price + offset
            if candidate is None or self._better(pos, level, candidate):
                candidate = level
        return candidate

    @staticmethod
    def _better(pos: PositionState, a: float, b: float) -> bool:
        return a > b if pos.side == "long" else a < b

    def _pick_stop(self, current: Optional[float], pos: PositionState, proposed: float) -> float:
        if current is None:
            return proposed
        return proposed if self._better(pos, proposed, current) else current

    @staticmethod
    def _is_improvement(pos: PositionState, new_stop: float) -> bool:
        if pos.side == "long":
            return new_stop > pos.stop_px
        return new_stop < pos.stop_px
