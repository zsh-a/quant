"""Rule + multi-LLM consensus auto-labeler.

The :class:`AutoLabeler` walks a bar series, asks every analyst (one rule
analyst plus N LLM analysts) for signals at each candidate bar, and emits
a :class:`~src.brooks.eval.golden.GoldenSample` (``source="silver"``)
whenever at least ``min_agreement`` analysts agree on the same
``(pattern, side)`` setup with similar entry/stop levels.

The consensus is intentionally conservative — silver labels seed the
golden dataset, so it is better to skip ambiguous bars than to embed
disagreement into the corpus.
"""

from __future__ import annotations

import asyncio
import math
import statistics
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from src.brooks.analyst.base import Analyst
from src.brooks.context import Bar, BrooksContext, TFSnapshot
from src.brooks.eval.golden import GoldenSample
from src.brooks.features import BarFeatureExtractor
from src.brooks.regime import BrooksRegimeClassifier, RegimeSnapshot
from src.brooks.schema import Signal
from src.brooks.structure import MarketStructureTracker

__all__ = ["AutoLabeler", "ConsensusVote"]


@dataclass
class ConsensusVote:
    """Internal record describing the agreed-upon setup for a bar."""

    pattern: str
    side: str
    entry: float
    stop: float
    target: Optional[float]
    voters: List[str]
    reasonings: List[str]


class AutoLabeler:
    """Generate silver :class:`GoldenSample`'s from analyst consensus.

    Parameters
    ----------
    rule_analyst:
        Required deterministic analyst — its output anchors each consensus
        bucket. The rule analyst is also counted toward ``min_agreement``.
    llm_analysts:
        Stochastic analysts (typically ``LLMAnalyst`` instances). Set to
        ``[]`` to vote with the rule analyst alone (mostly for tests).
    min_agreement:
        Minimum number of analysts that must agree on ``(pattern, side)``
        before a sample is emitted.
    max_concurrent:
        Cap on concurrent ``analyze`` calls. Used to throttle LLM
        rate limits.
    entry_tolerance:
        Maximum *relative* spread of entry prices (as a fraction of the
        median entry) tolerated within a consensus group. Larger spreads
        cause the bar to be rejected.
    min_bars_for_label:
        Bars before which ``label`` skips the candidate. Detectors need
        history before they can produce signals; this avoids flooding the
        analysts with empty contexts.
    """

    def __init__(
        self,
        rule_analyst: Analyst,
        llm_analysts: Optional[Sequence[Analyst]] = None,
        min_agreement: int = 2,
        max_concurrent: int = 4,
        entry_tolerance: float = 0.005,
        min_bars_for_label: int = 20,
    ) -> None:
        if rule_analyst is None:
            raise ValueError("AutoLabeler requires a rule analyst")
        if min_agreement < 1:
            raise ValueError("min_agreement must be >= 1")
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        if entry_tolerance < 0:
            raise ValueError("entry_tolerance must be non-negative")
        self._rule = rule_analyst
        self._llms: List[Analyst] = list(llm_analysts or [])
        self._analysts: List[Analyst] = [self._rule, *self._llms]
        self._min_agreement = int(min_agreement)
        self._max_concurrent = int(max_concurrent)
        self._entry_tolerance = float(entry_tolerance)
        self._min_bars_for_label = int(min_bars_for_label)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def min_agreement(self) -> int:
        return self._min_agreement

    @property
    def analyst_count(self) -> int:
        return len(self._analysts)

    async def label(
        self,
        bars: Sequence[Bar],
        symbol: str,
        interval: str,
        htf: Optional[dict] = None,
    ) -> List[GoldenSample]:
        """Walk ``bars`` bar-by-bar, returning silver samples on consensus.

        Each candidate bar produces at most one sample (the strongest
        consensus group on that bar wins). Bars before
        ``min_bars_for_label`` are skipped — pattern detectors generally
        need ~20 bars of warmup.
        """
        if len(bars) < max(self._min_bars_for_label, 1):
            return []

        regime_history = _precompute_regimes(bars)
        sem = asyncio.Semaphore(self._max_concurrent)
        out: List[GoldenSample] = []

        for idx in range(self._min_bars_for_label, len(bars)):
            ctx_bars = list(bars[: idx + 1])
            ctx = BrooksContext(
                symbol=symbol,
                primary=TFSnapshot(interval=interval, bars=ctx_bars),
                htf=dict(htf or {}),
            )
            signal_lists = await _gather_signals(self._analysts, ctx, sem)
            vote = _resolve_consensus(
                signal_lists,
                analyst_names=[a.name for a in self._analysts],
                min_agreement=self._min_agreement,
                entry_tolerance=self._entry_tolerance,
            )
            if vote is None:
                continue
            regime_snap = regime_history[idx]
            htf_aligned = bool(htf and ctx.htf_aligned_for(vote.side))
            sample = GoldenSample(
                id=f"silver-{symbol}-{interval}-{idx}-{uuid.uuid4().hex[:8]}",
                symbol=symbol,
                interval=interval,
                bars=ctx_bars,
                target_bar_idx=idx,
                expected_pattern=vote.pattern,
                expected_side=vote.side,  # type: ignore[arg-type]
                expected_entry=vote.entry,
                expected_stop=vote.stop,
                expected_target=vote.target,
                regime=regime_snap.regime.value,
                htf_aligned=htf_aligned,
                source="silver",
                reasoning=" || ".join(r for r in vote.reasonings if r),
                meta={
                    "voters": list(vote.voters),
                    "agreement": len(vote.voters),
                    "regime_confidence": round(regime_snap.confidence, 4),
                },
            )
            out.append(sample)
        return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


async def _gather_signals(
    analysts: Sequence[Analyst],
    ctx: BrooksContext,
    sem: asyncio.Semaphore,
) -> List[List[Signal]]:
    async def _one(a: Analyst) -> List[Signal]:
        async with sem:
            return list(await a.analyze(ctx))

    return await asyncio.gather(*[_one(a) for a in analysts])


def _resolve_consensus(
    signal_lists: List[List[Signal]],
    analyst_names: Sequence[str],
    min_agreement: int,
    entry_tolerance: float,
) -> Optional[ConsensusVote]:
    """Bucket signals by (pattern, side) and return the strongest agreement."""
    buckets: dict[Tuple[str, str], List[Tuple[str, Signal]]] = defaultdict(list)
    for name, sigs in zip(analyst_names, signal_lists):
        seen: set[Tuple[str, str]] = set()
        for s in sigs:
            key = (s.pattern, s.side)
            if key in seen:
                continue
            seen.add(key)
            buckets[key].append((name, s))

    best: Optional[ConsensusVote] = None
    for (pattern, side), entries in buckets.items():
        if len(entries) < min_agreement:
            continue
        signals = [s for _, s in entries]
        if not _entries_compatible(signals, entry_tolerance):
            continue
        entry = statistics.median(s.entry_px for s in signals)
        stop = statistics.median(s.stop_px for s in signals)
        targets = [s.target_px for s in signals if s.target_px is not None]
        target = statistics.median(targets) if targets else None
        vote = ConsensusVote(
            pattern=pattern,
            side=side,
            entry=float(entry),
            stop=float(stop),
            target=float(target) if target is not None else None,
            voters=[name for name, _ in entries],
            reasonings=[s.reasoning for s in signals],
        )
        if best is None or len(vote.voters) > len(best.voters):
            best = vote
    return best


def _entries_compatible(signals: Sequence[Signal], tolerance: float) -> bool:
    if tolerance <= 0:
        return True
    entries = [s.entry_px for s in signals]
    if not entries:
        return False
    base = statistics.median(entries)
    if base == 0:
        return all(math.isclose(e, 0.0) for e in entries)
    spread = (max(entries) - min(entries)) / abs(base)
    return spread <= tolerance


def _precompute_regimes(bars: Sequence[Bar]) -> List[RegimeSnapshot]:
    """Replay every bar through the regime classifier; return per-bar snapshots."""
    ext = BarFeatureExtractor()
    tracker = MarketStructureTracker(ext)
    classifier = BrooksRegimeClassifier()
    history: List = []
    snapshots: List[RegimeSnapshot] = []
    for bar in bars:
        feat = ext.on_bar(bar.timestamp_ns, bar.open, bar.high, bar.low, bar.close)
        struct = tracker.on_features(feat)
        history.append(feat)
        snapshots.append(classifier.classify(history, struct))
    return snapshots
