"""Expected-value gate.

Replaces the legacy ``min_rr`` hard threshold with a ``min_expected_r``
filter on the Trader's Equation output. Signals whose ``E < min`` are
dropped; the survivors are promoted to :class:`Decision`s with the
TE-derived probability and expected_r baked in.
"""

from __future__ import annotations

from typing import List

from src.brooks.schema import Decision, Signal

from .trader_equation import DEFAULT_REWARD_R, TraderEquation


class EVGate:
    def __init__(self, te: TraderEquation, min_expected_r: float = 0.1):
        self._te = te
        self._min = float(min_expected_r)

    @property
    def min_expected_r(self) -> float:
        return self._min

    def filter(
        self,
        signals: List[Signal],
        regime: str,
        htf_aligned: bool,
        symbol: str = "",
    ) -> List[Decision]:
        out: List[Decision] = []
        for s in signals:
            if s is None:
                continue
            p, e = self._te.score(s, regime, htf_aligned)
            if e < self._min:
                continue
            target_px = s.target_px if s.target_px is not None else _default_target(s, self._te.default_reward_r)
            out.append(
                Decision(
                    symbol=symbol or s.meta.get("symbol", ""),
                    side=s.side,
                    entry_px=s.entry_px,
                    stop_px=s.stop_px,
                    target_px=target_px,
                    probability=p,
                    expected_r=e,
                    regime=regime,
                    htf_aligned=bool(htf_aligned),
                    signals=[s],
                    source=s.source,
                    reasoning=f"p={p:.2f} E={e:.2f}",
                )
            )
        return out


def _default_target(sig: Signal, reward_r: float = DEFAULT_REWARD_R) -> float:
    """Synthesize a target price ``reward_r`` units of risk past the entry."""
    one_r = sig.one_r
    if sig.side == "long":
        return sig.entry_px + reward_r * one_r
    return sig.entry_px - reward_r * one_r


__all__ = ["EVGate"]
