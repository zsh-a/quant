"""Expected-value gate.

Replaces the legacy ``min_rr`` hard threshold with a ``min_expected_r``
filter on the Trader's Equation output. Signals whose ``E < min`` are
dropped; the survivors are promoted to :class:`Decision`s with the
TE-derived probability and expected_r baked in.
"""

from __future__ import annotations

from typing import List, Optional, Union

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
        htf_aligned: Union[bool, str, None] = None,
        symbol: str = "",
        *,
        htf_alignment: Union[str, None] = None,
    ) -> List[Decision]:
        """Score every signal, drop any with ``E < min_expected_r``.

        ``htf_aligned`` accepts either a boolean (legacy: ``True`` →
        "aligned" bucket, ``False`` → non-aligned) or a string tag
        (``"aligned"`` / ``"conflict"`` / ``"neutral"``) produced by
        :meth:`BrooksContext.htf_alignment_for`. Passing the tag also
        activates the Phase-3.5 probability multiplier. For clarity new
        code may use the ``htf_alignment=`` keyword-only variant, which
        takes precedence when both are supplied.
        """
        effective = htf_alignment if htf_alignment is not None else htf_aligned
        aligned_bool, tag = _normalize_alignment(effective)
        out: List[Decision] = []
        for s in signals:
            if s is None:
                continue
            p, e = self._te.score(s, regime, aligned_bool, tag)
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
                    htf_aligned=aligned_bool,
                    signals=[s],
                    source=s.source,
                    reasoning=f"p={p:.2f} E={e:.2f} htf={tag or 'n/a'}",
                )
            )
        return out


def _normalize_alignment(
    value: Union[bool, str, None],
) -> tuple[bool, Optional[str]]:
    """Map the public ``htf_alignment`` argument onto ``(aligned_bool, tag)``.

    ``True``/``False`` are legacy inputs and do not activate the Phase
    3.5 multiplier (``tag`` is ``None``). String tags set both the
    boolean and the multiplier mode.
    """
    if value is None:
        return False, None
    if isinstance(value, bool):
        return value, None
    if value == "aligned":
        return True, "aligned"
    if value == "conflict":
        return False, "conflict"
    if value == "neutral":
        return False, "neutral"
    raise ValueError(f"unknown htf_alignment tag: {value!r}")


def _default_target(sig: Signal, reward_r: float = DEFAULT_REWARD_R) -> float:
    """Synthesize a target price ``reward_r`` units of risk past the entry."""
    one_r = sig.one_r
    if sig.side == "long":
        return sig.entry_px + reward_r * one_r
    return sig.entry_px - reward_r * one_r


__all__ = ["EVGate"]
