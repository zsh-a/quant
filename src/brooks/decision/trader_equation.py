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
  configurable default when the analyst left ``target_px`` unset).
"""

from __future__ import annotations

from typing import Tuple

from src.brooks.schema import Signal

from .hit_rate import HitRateKey, HitRateTable

DEFAULT_REWARD_R = 2.0
DEFAULT_COST_R = 0.05


class TraderEquation:
    def __init__(
        self,
        hit_rate: HitRateTable,
        cost_r: float = DEFAULT_COST_R,
        default_reward_r: float = DEFAULT_REWARD_R,
    ):
        if cost_r < 0:
            raise ValueError("cost_r must be non-negative")
        if default_reward_r <= 0:
            raise ValueError("default_reward_r must be positive")
        self._ht = hit_rate
        self._cost_r = float(cost_r)
        self._default_reward_r = float(default_reward_r)

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

    def score(
        self,
        sig: Signal,
        regime: str,
        htf_aligned: bool,
    ) -> Tuple[float, float]:
        """Return ``(p, expected_r)`` for the given signal."""
        p = self.probability(sig, regime, htf_aligned)
        reward = self.reward_r(sig)
        e = p * reward - (1.0 - p) * 1.0 - self._cost_r
        return p, e


__all__ = ["DEFAULT_COST_R", "DEFAULT_REWARD_R", "TraderEquation"]
