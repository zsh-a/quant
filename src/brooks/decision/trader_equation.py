"""Trader's Equation evaluator.

Brooks' Trader's Equation is the EV of a trade:

    E[R] = p * reward_R - (1 - p) * 1.0 - cost_R

where ``p`` is the probability of reaching the target (in 1R units),
``reward_R`` is the target distance in R, and ``cost_R`` is fees + slippage
expressed in R. The ``1`` on the loss term is the stop distance (1R by
definition).

This module turns a ``Signal`` into ``(p, E)`` by:

* sourcing ``p`` from the historical :class:`HitRateTable` when the bucket
  has enough samples, otherwise falling back to the analyst-supplied
  ``Signal.probability`` prior;
* deriving ``reward_R`` from ``target_px``/``entry_px``/``stop_px`` (or a
  configurable default when the analyst left ``target_px`` unset);
* optionally boosting/dampening ``p`` by an HTF-alignment multiplier
  (Phase 3.5) — ``aligned`` scales by ``htf_aligned_mult`` (default 1.2,
  hard-capped at ``htf_prob_cap``), ``conflict`` by ``htf_conflict_mult``
  (default 0.8), ``neutral`` is a pass-through.
"""

from __future__ import annotations

from typing import Optional, Tuple

from src.brooks.schema import Signal

from .hit_rate import HitRateKey, HitRateTable

DEFAULT_REWARD_R = 2.0
DEFAULT_COST_R = 0.05
DEFAULT_HTF_ALIGNED_MULT = 1.2
DEFAULT_HTF_CONFLICT_MULT = 0.8
DEFAULT_HTF_PROB_CAP = 0.95


class TraderEquation:
    def __init__(
        self,
        hit_rate: HitRateTable,
        cost_r: float = DEFAULT_COST_R,
        default_reward_r: float = DEFAULT_REWARD_R,
        htf_aligned_mult: float = DEFAULT_HTF_ALIGNED_MULT,
        htf_conflict_mult: float = DEFAULT_HTF_CONFLICT_MULT,
        htf_prob_cap: float = DEFAULT_HTF_PROB_CAP,
    ):
        if cost_r < 0:
            raise ValueError("cost_r must be non-negative")
        if default_reward_r <= 0:
            raise ValueError("default_reward_r must be positive")
        if htf_aligned_mult < 1.0:
            raise ValueError("htf_aligned_mult must be >= 1.0")
        if not 0.0 < htf_conflict_mult <= 1.0:
            raise ValueError("htf_conflict_mult must be in (0, 1]")
        if not 0.0 < htf_prob_cap <= 1.0:
            raise ValueError("htf_prob_cap must be in (0, 1]")
        self._ht = hit_rate
        self._cost_r = float(cost_r)
        self._default_reward_r = float(default_reward_r)
        self._htf_aligned_mult = float(htf_aligned_mult)
        self._htf_conflict_mult = float(htf_conflict_mult)
        self._htf_prob_cap = float(htf_prob_cap)

    @property
    def cost_r(self) -> float:
        return self._cost_r

    @property
    def default_reward_r(self) -> float:
        return self._default_reward_r

    def reward_r(self, sig: Signal) -> float:
        """Reward in R units; falls back to ``default_reward_r`` if no target."""
        if sig.target_px is None:
            return self._default_reward_r
        one_r = sig.one_r
        if one_r <= 0:
            return self._default_reward_r
        if sig.side == "long":
            return max(0.0, (sig.target_px - sig.entry_px) / one_r)
        return max(0.0, (sig.entry_px - sig.target_px) / one_r)

    def probability(self, sig: Signal, regime: str, htf_aligned: bool) -> float:
        key = HitRateKey(sig.pattern, regime, bool(htf_aligned), sig.side)
        if self._ht.is_sufficient(key):
            return float(self._ht.lookup(key)["hit_rate_1r"])
        return float(sig.probability)

    def apply_htf_multiplier(self, p: float, htf_alignment: Optional[str]) -> float:
        """Apply the HTF-alignment multiplier to a raw probability.

        ``htf_alignment`` may be ``"aligned"``, ``"conflict"``, ``"neutral"``
        or ``None``. ``"aligned"`` boosts ``p`` by ``htf_aligned_mult`` but
        hard-caps the result at ``htf_prob_cap``; ``"conflict"`` scales by
        ``htf_conflict_mult``; anything else is a pass-through. The
        returned probability is always clipped into ``[0, 1]``.
        """
        if htf_alignment == "aligned":
            p = min(self._htf_prob_cap, p * self._htf_aligned_mult)
        elif htf_alignment == "conflict":
            p = p * self._htf_conflict_mult
        return max(0.0, min(1.0, p))

    def score(
        self,
        sig: Signal,
        regime: str,
        htf_aligned: bool,
        htf_alignment: Optional[str] = None,
    ) -> Tuple[float, float]:
        """Return ``(p, expected_r)`` for the given signal.

        ``htf_aligned`` selects the hit-rate bucket (``True``/``False``).
        ``htf_alignment`` — optional — drives the probability multiplier
        (see :meth:`apply_htf_multiplier`).  When ``htf_alignment`` is
        ``None`` the multiplier is skipped entirely so existing callers
        see identical behaviour.
        """
        p = self.probability(sig, regime, htf_aligned)
        p = self.apply_htf_multiplier(p, htf_alignment)
        reward = self.reward_r(sig)
        e = p * reward - (1.0 - p) * 1.0 - self._cost_r
        return p, e


__all__ = [
    "DEFAULT_COST_R",
    "DEFAULT_REWARD_R",
    "DEFAULT_HTF_ALIGNED_MULT",
    "DEFAULT_HTF_CONFLICT_MULT",
    "DEFAULT_HTF_PROB_CAP",
    "TraderEquation",
]
