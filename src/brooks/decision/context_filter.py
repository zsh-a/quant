"""Background → signal context filter.

The aggregator and EV gate accept any signal a detector emits. Brooks'
"background → signal" principle says the **regime / structure** must
permit a signal before it can be traded; otherwise the same set of
detectors will fire constantly in chop and the resulting strategy
trades far too often.

:class:`ContextFilter` sits between the aggregator and the EV gate. It
inspects the combined ``(side, pattern_type set)``, the current
:class:`BrooksRegime`, and the :class:`MarketStructure` snapshot and
returns ``(allow, reason)``. Signals whose background does not permit
them are dropped before they ever reach the EV gate; the strategy keeps
them around as ``failed_signals`` for the Studio panel to render.

The rules below intentionally err on the **conservative** side:

* trends only allow with-trend continuation patterns by default;
  countertrend trades require a strong-reversal package
  (wedge / double / mtr / final-flag / failed-breakout)
* tight ranges reject every breakout follow-through; only reversals
  inside the range are allowed
* broad ranges allow both edges to fade and trend-internal pullbacks
* breakout mode allows the freshly-broken side's continuation only
* climax + unknown reject everything except strong counter-climax
  reversals (climax) or nothing at all (unknown)

Each rule comes with a short reason string that downstream renderers
display next to the failed signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set

from src.brooks.regime import BrooksRegime
from src.brooks.schema import Signal
from src.brooks.structure import MarketStructure

__all__ = [
    "ContextDecision",
    "ContextFilter",
    "PATTERN_TYPE_MAP",
    "REVERSAL_TYPES",
    "CONTINUATION_TYPES",
    "pattern_type_for",
]


# ---------------------------------------------------------------------------
# Pattern → category map
# ---------------------------------------------------------------------------

# Map detector name (``Signal.pattern``) → high-level pattern category.
# Detectors not listed here fall back to ``"unknown"``.
PATTERN_TYPE_MAP: dict[str, str] = {
    # with-trend pullback recoveries (Brooks H1..H4 / L1..L4)
    "h1": "pullback",
    "h2": "pullback",
    "h3": "pullback",
    "h4": "pullback",
    "l1": "pullback",
    "l2": "pullback",
    "l3": "pullback",
    "l4": "pullback",
    # reversal patterns
    "wedge_long": "wedge",
    "wedge_short": "wedge",
    "double_top": "double_top",
    "double_bottom": "double_bottom",
    "mtr_long": "mtr",
    "mtr_short": "mtr",
    "final_flag": "final_flag",
    "failed_breakout": "failed_breakout",
    # breakout continuations
    "bp_long": "breakout_pullback",
    "bp_short": "breakout_pullback",
    "ii_breakout": "breakout",
    "iii_breakout": "breakout",
    "measured_move": "measured_move",
    "micro_channel_long": "micro_channel",
    "micro_channel_short": "micro_channel",
}

REVERSAL_TYPES: Set[str] = {
    "wedge",
    "double_top",
    "double_bottom",
    "mtr",
    "final_flag",
    "failed_breakout",
}

CONTINUATION_TYPES: Set[str] = {
    "pullback",
    "breakout",
    "breakout_pullback",
    "measured_move",
    "micro_channel",
}


def pattern_type_for(pattern: str) -> str:
    """Return the high-level category for a detector's name."""
    return PATTERN_TYPE_MAP.get(pattern, "unknown")


# ---------------------------------------------------------------------------
# Result + filter
# ---------------------------------------------------------------------------


@dataclass
class ContextDecision:
    """Outcome of a :class:`ContextFilter` evaluation."""

    allow: bool
    reason: str
    regime: str = "unknown"
    leg_dir: str = "flat"
    in_trading_range: bool = False
    swing_age_bars: Optional[int] = None
    pattern_types: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "allow": self.allow,
            "reason": self.reason,
            "regime": self.regime,
            "leg_dir": self.leg_dir,
            "in_trading_range": self.in_trading_range,
            "swing_age_bars": self.swing_age_bars,
            "pattern_types": list(self.pattern_types),
        }


class ContextFilter:
    """Background → signal gate that runs before the EV gate.

    Parameters
    ----------
    allow_strong_reversal_in_trend:
        When ``True`` (default), strong reversal packages are permitted
        to take countertrend trades inside trend regimes. Setting to
        ``False`` forces with-trend-only behaviour.
    micro_channel_steepness_atr:
        Reject continuation pullbacks when the leg's micro-channel slope
        per bar exceeds this many ATR units — a too-steep leg is climax
        territory and chasing it usually fails. ``None`` disables the
        check.
    max_swing_age_bars:
        Reject any signal when the latest confirmed swing of either kind
        is older than this many bars (structure is "stale"). ``None``
        disables the check.
    require_reversal_package_for_countertrend:
        Number of distinct reversal-type patterns required for the
        filter to allow a countertrend signal in a trending regime.
        Default ``1`` lets a single wedge/double/mtr through; setting to
        ``2`` enforces confluence on reversal trades.
    """

    def __init__(
        self,
        *,
        allow_strong_reversal_in_trend: bool = True,
        micro_channel_steepness_atr: Optional[float] = 1.0,
        max_swing_age_bars: Optional[int] = 80,
        require_reversal_package_for_countertrend: int = 1,
        weak_trend_pullback_needs_confluence: bool = True,
    ) -> None:
        if require_reversal_package_for_countertrend < 1:
            raise ValueError("require_reversal_package_for_countertrend must be >= 1")
        if micro_channel_steepness_atr is not None and micro_channel_steepness_atr <= 0:
            raise ValueError("micro_channel_steepness_atr must be > 0 or None")
        if max_swing_age_bars is not None and max_swing_age_bars <= 0:
            raise ValueError("max_swing_age_bars must be > 0 or None")
        self._allow_reversal = allow_strong_reversal_in_trend
        self._steep_atr = micro_channel_steepness_atr
        self._max_swing_age = max_swing_age_bars
        self._reversal_n = int(require_reversal_package_for_countertrend)
        self._weak_trend_needs_confluence = bool(weak_trend_pullback_needs_confluence)

    # ------------------------------------------------------------------ check
    def check(
        self,
        side: str,
        signals: Iterable[Signal],
        regime: BrooksRegime,
        structure: MarketStructure,
    ) -> ContextDecision:
        signals = list(signals)
        types = sorted({pattern_type_for(s.pattern) for s in signals})
        type_set = set(types)
        leg_dir = getattr(structure, "current_leg_dir", "flat") or "flat"
        swing_age = _swing_age_bars(structure)
        in_tr = regime in (
            BrooksRegime.TIGHT_TRADING_RANGE,
            BrooksRegime.BROAD_TRADING_RANGE,
        )
        base = dict(
            regime=regime.value if hasattr(regime, "value") else str(regime),
            leg_dir=leg_dir,
            in_trading_range=in_tr,
            swing_age_bars=swing_age,
            pattern_types=types,
        )

        # ---- structure-level guards (apply to all regimes except unknown) ---
        if regime != BrooksRegime.UNKNOWN:
            if (
                self._max_swing_age is not None
                and swing_age is not None
                and swing_age > self._max_swing_age
            ):
                return ContextDecision(
                    allow=False,
                    reason=f"structure_stale: swing_age={swing_age} > {self._max_swing_age}",
                    **base,
                )

        # ---- per-regime rules ----------------------------------------------
        if regime == BrooksRegime.UNKNOWN:
            return ContextDecision(
                allow=False, reason="regime_unknown: insufficient context", **base
            )

        if regime == BrooksRegime.CLIMAX:
            return self._check_climax(side, type_set, structure, base)

        if regime == BrooksRegime.BREAKOUT_MODE:
            return self._check_breakout(side, type_set, structure, base)

        if regime == BrooksRegime.TIGHT_TRADING_RANGE:
            return self._check_tight_range(side, type_set, base)

        if regime == BrooksRegime.BROAD_TRADING_RANGE:
            return self._check_broad_range(side, type_set, base)

        if regime in (
            BrooksRegime.STRONG_BULL_TREND,
            BrooksRegime.WEAK_BULL_TREND,
        ):
            return self._check_trend(
                side, type_set, regime, structure, trend_side="long", base=base
            )

        if regime in (
            BrooksRegime.STRONG_BEAR_TREND,
            BrooksRegime.WEAK_BEAR_TREND,
        ):
            return self._check_trend(
                side, type_set, regime, structure, trend_side="short", base=base
            )

        # Defensive default — should be unreachable; reject so we never trade
        # in a regime we don't recognise.
        return ContextDecision(
            allow=False, reason=f"regime_unhandled: {regime}", **base
        )

    # ------------------------------------------------------------------ trend
    def _check_trend(
        self,
        side: str,
        types: Set[str],
        regime: BrooksRegime,
        structure: MarketStructure,
        *,
        trend_side: str,
        base: dict,
    ) -> ContextDecision:
        with_trend = side == trend_side
        if with_trend:
            with_trend_reversals = (
                {"double_bottom"} if trend_side == "long" else {"double_top"}
            )
            allowed = (CONTINUATION_TYPES & types) | (with_trend_reversals & types)
            if not allowed:
                return ContextDecision(
                    allow=False,
                    reason=(
                        f"with_trend_no_continuation: types={sorted(types)} "
                        f"regime={regime.value}"
                    ),
                    **base,
                )
            # Brooks: a with-trend pullback only makes sense when the
            # current leg is *against* the trend (so price is in a real
            # pullback). When the leg is extending with the trend, an h2/l2
            # here is buying the high / selling the low; when the leg is
            # flat the structure is ambiguous and the pullback signal is
            # likely chop noise.
            leg_dir = (base.get("leg_dir") or "flat")
            in_pullback_leg = (
                (trend_side == "long" and leg_dir == "down")
                or (trend_side == "short" and leg_dir == "up")
            )
            if (allowed <= {"pullback"}) and not in_pullback_leg:
                return ContextDecision(
                    allow=False,
                    reason=(
                        "with_trend_pullback_needs_counter_leg: "
                        f"leg_dir={leg_dir} trend={trend_side}"
                    ),
                    **base,
                )
            # Weak trends are noisy — a lone with-trend pullback is the
            # majority of false signals in chop. Require either a
            # non-pullback continuation pattern (bp_*, measured_move,
            # micro_channel) or a structurally-aligned reversal
            # (double_bottom in bull / double_top in bear).
            if (
                self._weak_trend_needs_confluence
                and regime in (
                    BrooksRegime.WEAK_BULL_TREND,
                    BrooksRegime.WEAK_BEAR_TREND,
                )
                and (allowed <= {"pullback"})
            ):
                return ContextDecision(
                    allow=False,
                    reason=(
                        "weak_trend_pullback_needs_confluence: "
                        f"types={sorted(types)} regime={regime.value}"
                    ),
                    **base,
                )
            steep = self._micro_channel_too_steep(structure, trend_side)
            # Steep micro-channels in strong trends → don't chase pure
            # pullbacks; require breakout-pullback or a reversal package.
            if (
                steep
                and regime == BrooksRegime.STRONG_BULL_TREND
                and types <= {"pullback"}
            ):
                return ContextDecision(
                    allow=False,
                    reason="micro_channel_too_steep_for_pullback",
                    **base,
                )
            if (
                steep
                and regime == BrooksRegime.STRONG_BEAR_TREND
                and types <= {"pullback"}
            ):
                return ContextDecision(
                    allow=False,
                    reason="micro_channel_too_steep_for_pullback",
                    **base,
                )
            return ContextDecision(
                allow=True, reason="with_trend_continuation", **base
            )

        # Counter-trend in a trend regime — require a reversal package.
        if not self._allow_reversal:
            return ContextDecision(
                allow=False,
                reason="counter_trend_blocked: reversals disabled in trend",
                **base,
            )
        reversal_hits = REVERSAL_TYPES & types
        if regime in (BrooksRegime.STRONG_BULL_TREND, BrooksRegime.STRONG_BEAR_TREND):
            needed = max(self._reversal_n, 1)
            # Strong trend needs the full package; a single wedge is not
            # enough to fight an established trend.
            if len(reversal_hits) < max(needed, 2):
                return ContextDecision(
                    allow=False,
                    reason=(
                        f"counter_strong_trend_needs_reversal_package: "
                        f"reversal_types={sorted(reversal_hits)} "
                        f"need>={max(needed, 2)}"
                    ),
                    **base,
                )
            return ContextDecision(
                allow=True,
                reason=(
                    f"counter_strong_trend_with_reversal_package: "
                    f"{sorted(reversal_hits)}"
                ),
                **base,
            )
        # Weak trend — single reversal pattern is sufficient.
        if not reversal_hits:
            return ContextDecision(
                allow=False,
                reason=(
                    f"counter_weak_trend_needs_reversal: types={sorted(types)}"
                ),
                **base,
            )
        return ContextDecision(
            allow=True,
            reason=f"counter_weak_trend_with_reversal: {sorted(reversal_hits)}",
            **base,
        )

    # ------------------------------------------------------------------ ranges
    def _check_tight_range(
        self, side: str, types: Set[str], base: dict
    ) -> ContextDecision:
        # Tight TRs chop violently — Brooks' guidance is "trade out of a
        # tight TR, not in it". Only allow strong-reversal *confluence*
        # (≥ 2 reversal patterns) or an explicit failed-breakout fade.
        if types & CONTINUATION_TYPES:
            return ContextDecision(
                allow=False,
                reason=(
                    f"tight_range_blocks_continuation: types={sorted(types)}"
                ),
                **base,
            )
        reversal_hits = types & REVERSAL_TYPES
        if "failed_breakout" in reversal_hits:
            return ContextDecision(
                allow=True,
                reason=f"tight_range_failed_breakout: {sorted(reversal_hits)}",
                **base,
            )
        if len(reversal_hits) >= 2:
            return ContextDecision(
                allow=True,
                reason=f"tight_range_reversal_confluence: {sorted(reversal_hits)}",
                **base,
            )
        return ContextDecision(
            allow=False,
            reason=f"tight_range_needs_strong_reversal: types={sorted(types)}",
            **base,
        )

    def _check_broad_range(
        self, side: str, types: Set[str], base: dict
    ) -> ContextDecision:
        # Broad TRs allow edge fades and breakout-pullback re-entries but
        # naked breakouts and lone with-range pullbacks usually fail back
        # into the range.
        if types & {"breakout"}:
            return ContextDecision(
                allow=False,
                reason="broad_range_blocks_naked_breakout",
                **base,
            )
        if types & REVERSAL_TYPES:
            return ContextDecision(
                allow=True,
                reason=f"broad_range_reversal: {sorted(types & REVERSAL_TYPES)}",
                **base,
            )
        if "breakout_pullback" in types:
            return ContextDecision(
                allow=True,
                reason="broad_range_breakout_pullback",
                **base,
            )
        return ContextDecision(
            allow=False,
            reason=f"broad_range_needs_reversal_or_bp: types={sorted(types)}",
            **base,
        )

    # ------------------------------------------------------------------ breakout
    def _check_breakout(
        self,
        side: str,
        types: Set[str],
        structure: MarketStructure,
        base: dict,
    ) -> ContextDecision:
        always_in = getattr(structure, "always_in", "neutral") or "neutral"
        # Breakout mode favors continuation in the breakout direction.
        with_breakout = (
            (always_in == "long" and side == "long")
            or (always_in == "short" and side == "short")
        )
        if with_breakout:
            # Brooks: chase the breakout *only* on a confirmed pullback
            # ("BO + BP"). Naked breakout-on-breakout (ii/iii / measured
            # move alone) trades the spike, which usually mean-reverts.
            if "breakout_pullback" in types:
                return ContextDecision(
                    allow=True,
                    reason="breakout_with_direction_bp",
                    **base,
                )
            return ContextDecision(
                allow=False,
                reason=(
                    f"breakout_needs_pullback_re_entry: types={sorted(types)}"
                ),
                **base,
            )
        # Counter-breakout — allow only the explicit failed-breakout package.
        if types & {"failed_breakout", "wedge", "double_top", "double_bottom"}:
            return ContextDecision(
                allow=True,
                reason=(
                    f"breakout_failure_reversal: "
                    f"{sorted(types & (REVERSAL_TYPES | {'failed_breakout'}))}"
                ),
                **base,
            )
        return ContextDecision(
            allow=False,
            reason=(
                f"counter_breakout_needs_failure: types={sorted(types)}"
            ),
            **base,
        )

    # ------------------------------------------------------------------ climax
    def _check_climax(
        self,
        side: str,
        types: Set[str],
        structure: MarketStructure,
        base: dict,
    ) -> ContextDecision:
        # In climax, with-trend continuation is the worst trade Brooks
        # describes — momentum is about to reverse. Reject continuations
        # outright; allow only counter-climax reversal packages.
        with_trend_continuation = bool(types & CONTINUATION_TYPES)
        always_in = getattr(structure, "always_in", "neutral") or "neutral"
        with_trend = (
            (always_in == "long" and side == "long")
            or (always_in == "short" and side == "short")
        )
        if with_trend and with_trend_continuation:
            return ContextDecision(
                allow=False,
                reason=f"climax_blocks_continuation: types={sorted(types)}",
                **base,
            )
        reversal_hits = types & (REVERSAL_TYPES | {"final_flag"})
        if reversal_hits:
            return ContextDecision(
                allow=True,
                reason=f"climax_reversal: {sorted(reversal_hits)}",
                **base,
            )
        return ContextDecision(
            allow=False,
            reason=f"climax_needs_reversal: types={sorted(types)}",
            **base,
        )

    # ------------------------------------------------------------------ helpers
    def _micro_channel_too_steep(
        self, structure: MarketStructure, trend_side: str
    ) -> bool:
        if self._steep_atr is None:
            return False
        atr = float(getattr(structure, "atr14", 0.0) or 0.0)
        if atr <= 0:
            return False
        channel = (
            getattr(structure, "micro_channel_top", None)
            if trend_side == "long"
            else getattr(structure, "micro_channel_bot", None)
        )
        if channel is None:
            return False
        slope = abs(float(getattr(channel, "slope", 0.0)))
        return slope >= self._steep_atr * atr


def _swing_age_bars(structure: MarketStructure) -> Optional[int]:
    """Bars between the latest confirmed swing (high or low) and ``current_idx``.

    Returns ``None`` when no swings have been confirmed yet — callers
    treat that as "unknown" rather than "stale".
    """
    current_idx = getattr(structure, "current_idx", -1)
    if current_idx is None or current_idx < 0:
        return None
    highs = getattr(structure, "confirmed_swing_highs", None) or []
    lows = getattr(structure, "confirmed_swing_lows", None) or []
    last_idx = -1
    if highs:
        last_idx = max(last_idx, highs[-1].bar_idx)
    if lows:
        last_idx = max(last_idx, lows[-1].bar_idx)
    if last_idx < 0:
        return None
    return max(0, current_idx - last_idx)
