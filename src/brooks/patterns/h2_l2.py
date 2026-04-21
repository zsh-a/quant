"""H2 / L2 pullback detectors.

H2 ("high two"): in an always-in-long context, price pulls back twice against
the trend before the bull pattern re-asserts. Textbook sequence:

  1. Some bear bars (leg-1 down) make a low
  2. One or more bull bars recover
  3. Another bear push (leg-2 down) takes out (or matches) leg-1 low
  4. A bull reversal bar forms — that's the *signal bar*
  5. Entry = signal_bar.high + 1 tick (stop-buy), stop = signal_bar.low - 1 tick

L2 is the bearish mirror.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from src.brooks.patterns.base import DetectorContext, PatternDetector, PatternSignal
from src.brooks.patterns.registry import PatternRegistry

_TICK = 1e-6


@dataclass
class _PBState:
    state: Literal[
        "IDLE", "PB_LEG1", "PB_LEG1_RECOVERY", "PB_LEG2", "H2_FORMED", "TRIGGERED", "INVALID"
    ] = "IDLE"
    pb_start_idx: int = -1
    leg1_extreme: float = 0.0
    recovery_extreme: float = 0.0
    signal_bar_idx: int = -1
    signal_bar_high: float = 0.0
    signal_bar_low: float = 0.0
    leg2_start_idx: int = -1


class _TwoLeggedPullbackBase(PatternDetector):
    """Shared FSM for H2 and L2 — subclass picks direction-specific wiring."""

    side: Literal["long", "short"]

    def __init__(self, max_leg_bars: int = 10, invalidate_atr_mult: float = 2.0):
        self._s = _PBState()
        self.max_leg_bars = max_leg_bars
        self.invalidate_atr_mult = invalidate_atr_mult

    # ---- public --------------------------------------------------------

    def on_bar(self, ctx: DetectorContext) -> Optional[PatternSignal]:
        if not self._trend_ok(ctx):
            self._s = _PBState()
            return None

        feat = ctx.feat
        s = self._s

        if s.state == "IDLE":
            self._maybe_start_leg1(feat)
            return None

        if s.state == "PB_LEG1":
            self._handle_leg1(ctx)
            return None

        if s.state == "PB_LEG1_RECOVERY":
            self._handle_recovery(ctx)
            return None

        if s.state == "PB_LEG2":
            return self._handle_leg2(ctx)

        if s.state == "H2_FORMED":
            if feat.bar_idx - s.signal_bar_idx > 3:
                self._s = _PBState()
            return None

        return None

    def state_snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "detector": self.name,
            "state": self._s.state,
            "pb_start_idx": self._s.pb_start_idx,
            "leg1_extreme": round(self._s.leg1_extreme, 6),
            "recovery_extreme": round(self._s.recovery_extreme, 6),
            "signal_bar_idx": self._s.signal_bar_idx,
        }

    def mark_triggered(self, bar_idx: int) -> None:
        if self._s.state == "H2_FORMED":
            self._s = _PBState()

    def reset(self) -> None:
        self._s = _PBState()

    # ---- direction-specific hooks (overridden) -------------------------

    def _trend_ok(self, ctx: DetectorContext) -> bool:  # pragma: no cover
        raise NotImplementedError

    def _is_against(self, feat) -> bool:  # pragma: no cover
        raise NotImplementedError

    def _is_with(self, feat) -> bool:  # pragma: no cover
        raise NotImplementedError

    def _extreme_against(self, feat, current: float) -> float:  # pragma: no cover
        raise NotImplementedError

    def _recovery_extreme(self, feat, current: float) -> float:  # pragma: no cover
        raise NotImplementedError

    def _leg2_breaks_leg1(self, feat) -> bool:  # pragma: no cover
        raise NotImplementedError

    def _recovery_broken(self, feat) -> bool:  # pragma: no cover
        raise NotImplementedError

    def _make_signal(self, ctx: DetectorContext) -> PatternSignal:  # pragma: no cover
        raise NotImplementedError

    # ---- FSM step helpers ---------------------------------------------

    def _maybe_start_leg1(self, feat) -> None:
        if self._is_against(feat):
            self._s.state = "PB_LEG1"
            self._s.pb_start_idx = feat.bar_idx
            self._s.leg1_extreme = feat.low if self.side == "long" else feat.high

    def _handle_leg1(self, ctx: DetectorContext) -> None:
        feat = ctx.feat
        s = self._s
        atr = ctx.feat.atr14
        if self.side == "long":
            start_close = (
                ctx.recent_features[-(feat.bar_idx - s.pb_start_idx) - 1].close
                if (feat.bar_idx - s.pb_start_idx) < len(ctx.recent_features)
                else feat.close
            )
            if atr > 0 and start_close - feat.close > self.invalidate_atr_mult * atr:
                self._s = _PBState()
                return
        else:
            start_close = (
                ctx.recent_features[-(feat.bar_idx - s.pb_start_idx) - 1].close
                if (feat.bar_idx - s.pb_start_idx) < len(ctx.recent_features)
                else feat.close
            )
            if atr > 0 and feat.close - start_close > self.invalidate_atr_mult * atr:
                self._s = _PBState()
                return

        if feat.bar_idx - s.pb_start_idx > self.max_leg_bars:
            self._s = _PBState()
            return

        if self._is_against(feat):
            s.leg1_extreme = self._extreme_against(feat, s.leg1_extreme)
        elif self._is_with(feat):
            s.state = "PB_LEG1_RECOVERY"
            s.recovery_extreme = feat.high if self.side == "long" else feat.low

    def _handle_recovery(self, ctx: DetectorContext) -> None:
        feat = ctx.feat
        s = self._s
        if self._recovery_broken(feat):
            self._s = _PBState()
            return
        if self._is_with(feat):
            s.recovery_extreme = self._recovery_extreme(feat, s.recovery_extreme)
        elif self._is_against(feat):
            s.state = "PB_LEG2"
            s.leg2_start_idx = feat.bar_idx

    def _handle_leg2(self, ctx: DetectorContext) -> Optional[PatternSignal]:
        feat = ctx.feat
        s = self._s

        if feat.bar_idx - s.leg2_start_idx > self.max_leg_bars:
            self._s = _PBState()
            return None

        if self._leg2_breaks_leg1(feat):
            if self._is_with(feat) or feat.is_reversal_bar:
                s.state = "H2_FORMED"
                s.signal_bar_idx = feat.bar_idx
                s.signal_bar_high = feat.high
                s.signal_bar_low = feat.low
                return self._make_signal(ctx)
        return None


@PatternRegistry.register("h2")
class H2Detector(_TwoLeggedPullbackBase):
    side: Literal["long", "short"] = "long"

    def _trend_ok(self, ctx: DetectorContext) -> bool:
        return ctx.structure.always_in == "long"

    def _is_against(self, feat) -> bool:
        return not feat.is_bull and feat.body_pct >= 30

    def _is_with(self, feat) -> bool:
        return feat.is_bull and feat.body_pct >= 30

    def _extreme_against(self, feat, current: float) -> float:
        return min(feat.low, current) if current > 0 else feat.low

    def _recovery_extreme(self, feat, current: float) -> float:
        return max(feat.high, current)

    def _leg2_breaks_leg1(self, feat) -> bool:
        return feat.low <= self._s.leg1_extreme

    def _recovery_broken(self, feat) -> bool:
        return False

    def _make_signal(self, ctx: DetectorContext) -> PatternSignal:
        s = self._s
        entry = s.signal_bar_high + _TICK
        stop = s.signal_bar_low - _TICK
        return PatternSignal(
            detector=self.name,
            side="long",
            signal_bar_idx=s.signal_bar_idx,
            entry_px=entry,
            stop_px=stop,
            timestamp_ns=ctx.feat.timestamp_ns,
            reason=(
                f"H2 formed at bar {s.signal_bar_idx}: leg1_low={s.leg1_extreme:.4f} "
                f"recovery_high={s.recovery_extreme:.4f} always_in=long"
            ),
            metadata={
                "leg1_low": s.leg1_extreme,
                "recovery_high": s.recovery_extreme,
                "pb_start_idx": s.pb_start_idx,
            },
        )


@PatternRegistry.register("l2")
class L2Detector(_TwoLeggedPullbackBase):
    side: Literal["long", "short"] = "short"

    def _trend_ok(self, ctx: DetectorContext) -> bool:
        return ctx.structure.always_in == "short"

    def _is_against(self, feat) -> bool:
        return feat.is_bull and feat.body_pct >= 30

    def _is_with(self, feat) -> bool:
        return not feat.is_bull and feat.body_pct >= 30

    def _extreme_against(self, feat, current: float) -> float:
        return max(feat.high, current)

    def _recovery_extreme(self, feat, current: float) -> float:
        return min(feat.low, current) if current > 0 else feat.low

    def _leg2_breaks_leg1(self, feat) -> bool:
        return feat.high >= self._s.leg1_extreme

    def _recovery_broken(self, feat) -> bool:
        return False

    def _make_signal(self, ctx: DetectorContext) -> PatternSignal:
        s = self._s
        entry = s.signal_bar_low - _TICK
        stop = s.signal_bar_high + _TICK
        return PatternSignal(
            detector=self.name,
            side="short",
            signal_bar_idx=s.signal_bar_idx,
            entry_px=entry,
            stop_px=stop,
            timestamp_ns=ctx.feat.timestamp_ns,
            reason=(
                f"L2 formed at bar {s.signal_bar_idx}: leg1_high={s.leg1_extreme:.4f} "
                f"recovery_low={s.recovery_extreme:.4f} always_in=short"
            ),
            metadata={
                "leg1_high": s.leg1_extreme,
                "recovery_low": s.recovery_extreme,
                "pb_start_idx": s.pb_start_idx,
            },
        )
