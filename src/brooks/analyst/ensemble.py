"""Ensemble analysts — Vote / Router / Critic.

All three wrap one or more base :class:`Analyst` instances and implement
the same :class:`~src.brooks.analyst.base.Analyst` Protocol, so they are
interchangeable via
``BrooksStrategy(analyst="ensemble.vote"|"ensemble.router"|"ensemble.critic")``.

* :class:`VoteAnalyst`   — concurrent fan-out + consensus filter
* :class:`RouterAnalyst` — regime-based routing to a single sub-analyst
* :class:`CriticAnalyst` — producer/critic pipeline with rejection + probability adjustment
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from src.brooks.analyst.base import Analyst, AnalystRegistry
from src.brooks.context import BrooksContext
from src.brooks.regime import BrooksRegime
from src.brooks.schema import Signal

__all__ = ["VoteAnalyst", "RouterAnalyst", "CriticAnalyst"]


# ---------------------------------------------------------------------------
# VoteAnalyst
# ---------------------------------------------------------------------------


@AnalystRegistry.register("ensemble.vote")
class VoteAnalyst:
    """Run several analysts concurrently and keep only consensus signals.

    Signals are grouped by ``(pattern, side)`` with ``signal_bar_idx``
    tolerated within ±1 bar (adjacent pullback/breakout bars frequently
    fire one bar apart across detectors). A group survives only when
    at least ``min_agree_count`` *distinct* analysts contribute to it.

    The surviving signal's numeric fields are weighted averages over the
    group, with weights taken from ``weights[analyst.name]`` (default 1.0).
    """

    name = "ensemble.vote"

    def __init__(
        self,
        analysts: Sequence[Analyst],
        weights: Optional[Dict[str, float]] = None,
        min_agree_count: int = 2,
    ) -> None:
        if not analysts:
            raise ValueError("VoteAnalyst requires at least one analyst")
        if min_agree_count < 1:
            raise ValueError("min_agree_count must be >= 1")
        self._analysts: List[Analyst] = list(analysts)
        self._weights: Dict[str, float] = weights or {a.name: 1.0 for a in self._analysts}
        self._min_agree: int = min_agree_count

    async def analyze(self, ctx: BrooksContext) -> List[Signal]:
        batches = await asyncio.gather(*[a.analyze(ctx) for a in self._analysts])
        tagged: List[Tuple[str, Signal]] = []
        for analyst, batch in zip(self._analysts, batches):
            for sig in batch:
                tagged.append((analyst.name, sig))
        groups = _cluster_signals(tagged)

        kept: List[Signal] = []
        for group in groups:
            voters = {name for name, _ in group}
            if len(voters) < self._min_agree:
                continue
            kept.append(_merge_group(group, self._weights))
        return kept


def _cluster_signals(
    tagged: List[Tuple[str, Signal]],
) -> List[List[Tuple[str, Signal]]]:
    """Cluster by ``(pattern, side)`` with ``signal_bar_idx`` within ±1.

    Iterative single-linkage clustering — a new signal joins the first
    existing cluster whose ``(pattern, side)`` matches and whose nearest
    member is within one bar.
    """
    groups: List[List[Tuple[str, Signal]]] = []
    for item in tagged:
        _, sig = item
        placed = False
        for g in groups:
            head = g[0][1]
            if head.pattern != sig.pattern or head.side != sig.side:
                continue
            if any(abs(gs.signal_bar_idx - sig.signal_bar_idx) <= 1 for _, gs in g):
                g.append(item)
                placed = True
                break
        if not placed:
            groups.append([item])
    return groups


def _merge_group(
    group: List[Tuple[str, Signal]],
    weights: Dict[str, float],
) -> Signal:
    """Weighted-average merge of a consensus group into one :class:`Signal`."""
    ws = [float(weights.get(name, 1.0)) for name, _ in group]
    total_w = sum(ws) or 1.0

    def wavg(attr: str) -> float:
        return sum(w * getattr(s, attr) for w, (_, s) in zip(ws, group)) / total_w

    entry_px = wavg("entry_px")
    stop_px = wavg("stop_px")
    # A merged signal must keep Signal's entry!=stop invariant; if inputs
    # all collapsed to the same price (degenerate case), nudge by epsilon
    # in the stop direction consistent with `side`.
    if entry_px == stop_px:
        eps = max(1e-9, abs(entry_px) * 1e-9)
        side = group[0][1].side
        stop_px = stop_px - eps if side == "long" else stop_px + eps

    targets = [(w, s.target_px) for w, (_, s) in zip(ws, group) if s.target_px is not None]
    if targets:
        tw = sum(w for w, _ in targets) or 1.0
        target_px: Optional[float] = sum(w * tp for w, tp in targets) / tw
    else:
        target_px = None

    voters = sorted({name for name, _ in group})
    reasoning = " | ".join(f"{name}: {s.reasoning}" for name, s in group if s.reasoning)
    template = group[0][1]

    return Signal(
        pattern=template.pattern,
        side=template.side,
        signal_bar_idx=int(round(wavg("signal_bar_idx"))),
        entry_px=entry_px,
        stop_px=stop_px,
        target_px=target_px,
        probability=min(1.0, max(0.0, wavg("probability"))),
        quality=min(1.0, max(0.0, wavg("quality"))),
        reasoning=reasoning,
        source="ensemble.vote",
        meta={
            "voters": voters,
            "votes": len(voters),
            "vote_weights": {n: float(weights.get(n, 1.0)) for n in voters},
        },
    )


# ---------------------------------------------------------------------------
# RouterAnalyst
# ---------------------------------------------------------------------------


@AnalystRegistry.register("ensemble.router")
class RouterAnalyst:
    """Route the call to a single sub-analyst based on the primary regime.

    ``routes`` maps :class:`~src.brooks.regime.BrooksRegime` → analyst.
    When the current regime is missing from ``routes``, or when the
    context lacks a regime altogether, ``default`` handles the call.
    Every returned signal carries ``meta["routed_by"] = <regime.value>``.
    """

    name = "ensemble.router"

    def __init__(
        self,
        routes: Dict[BrooksRegime, Analyst],
        default: Analyst,
    ) -> None:
        if default is None:
            raise ValueError("RouterAnalyst requires a non-None default analyst")
        self._routes: Dict[BrooksRegime, Analyst] = dict(routes)
        self._default: Analyst = default

    async def analyze(self, ctx: BrooksContext) -> List[Signal]:
        regime_snap = ctx.primary.regime
        if regime_snap is None:
            analyst = self._default
            routed_by = BrooksRegime.UNKNOWN.value
        else:
            regime = getattr(regime_snap, "regime", regime_snap)
            analyst = self._routes.get(regime, self._default)
            routed_by = regime.value if hasattr(regime, "value") else str(regime)

        signals = await analyst.analyze(ctx)
        for s in signals:
            s.meta["routed_by"] = routed_by
        return signals


# ---------------------------------------------------------------------------
# CriticAnalyst
# ---------------------------------------------------------------------------


@AnalystRegistry.register("ensemble.critic")
class CriticAnalyst:
    """Producer/critic pipeline — producer proposes, critic confirms.

    ``producer.analyze(ctx)`` returns candidate signals. They are
    attached to the context as ``ctx.candidates`` (plus the optional
    ``ctx.critic_overlay`` prompt snippet) and handed to the critic,
    which returns one signal per candidate it endorses. Each critic
    signal carries ``meta["confirms"] = candidate_idx`` (or ``-1`` for
    "rejected") plus an adjusted ``probability``; the kept candidates
    inherit that probability and the critic's reasoning.
    """

    name = "ensemble.critic"

    def __init__(
        self,
        producer: Analyst,
        critic: Analyst,
        critic_prompt_overlay: Optional[str] = None,
    ) -> None:
        self._producer: Analyst = producer
        self._critic: Analyst = critic
        self._overlay: Optional[str] = critic_prompt_overlay

    async def analyze(self, ctx: BrooksContext) -> List[Signal]:
        candidates = await self._producer.analyze(ctx)
        if not candidates:
            return []
        critic_ctx = _inject_candidates(ctx, candidates, self._overlay)
        critique = await self._critic.analyze(critic_ctx)
        return _apply_critique(candidates, critique)


def _inject_candidates(
    ctx: BrooksContext,
    candidates: List[Signal],
    overlay: Optional[str],
) -> BrooksContext:
    """Return a shallow copy of ``ctx`` carrying ``candidates`` + overlay."""
    new_ctx = replace(ctx)
    new_ctx.candidates = list(candidates)  # type: ignore[attr-defined]
    new_ctx.critic_overlay = overlay  # type: ignore[attr-defined]
    return new_ctx


def _apply_critique(
    candidates: List[Signal],
    critique: List[Signal],
) -> List[Signal]:
    """Promote candidates the critic confirmed; drop the rest."""
    kept: List[Signal] = []
    for c_sig in critique:
        idx = c_sig.meta.get("confirms", -1)
        if idx is None or idx < 0 or idx >= len(candidates):
            continue
        cand = candidates[idx]
        merged_meta = {
            **cand.meta,
            **c_sig.meta,
            "producer_source": cand.source,
            "critic_source": c_sig.source,
        }
        critic_reason = c_sig.reasoning.strip() if c_sig.reasoning else ""
        reasoning = f"{cand.reasoning} | critic: {critic_reason}" if critic_reason else cand.reasoning
        kept.append(
            cand.model_copy(
                update={
                    "probability": c_sig.probability,
                    "reasoning": reasoning,
                    "source": "ensemble.critic",
                    "meta": merged_meta,
                }
            )
        )
    return kept
