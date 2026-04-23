"""Replay a :class:`GoldenDataset` through an :class:`Analyst`.

The runner is intentionally narrow: per sample it builds a
:class:`BrooksContext` from the labeled bars, asks the analyst for
signals, and records the prediction (best matching signal vs. ground
truth) plus optional Trader's Equation scoring and 1R/2R hit-rate
checks against the post-target bars.

The output :class:`EvalReport` carries the raw per-sample rows so
:mod:`src.brooks.eval.report` can render them into HTML or hand them
off as a :class:`pandas.DataFrame`.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal, Optional, Sequence

from src.brooks.analyst.base import Analyst
from src.brooks.context import Bar, BrooksContext, TFSnapshot
from src.brooks.decision.trader_equation import TraderEquation
from src.brooks.eval.golden import GoldenDataset, GoldenSample
from src.brooks.schema import Signal

if TYPE_CHECKING:  # pragma: no cover - type-only import to break cycle
    from src.brooks.eval.report import EvalReport

__all__ = ["EvalRunner", "SampleResult"]


HitFlag = Literal["hit", "miss", "unresolved"]


@dataclass
class SampleResult:
    """Per-sample evaluation record consumed by :class:`EvalReport`."""

    sample_id: str
    symbol: str
    interval: str
    expected_pattern: str
    expected_side: str
    regime: str
    htf_aligned: bool
    source: str
    pattern_match: bool
    side_match: bool
    pattern_emitted: List[str] = field(default_factory=list)
    predicted_signal: Optional[Signal] = None
    expected_r: Optional[float] = None
    probability: Optional[float] = None
    realized_r: Optional[float] = None
    hit_1r: HitFlag = "unresolved"
    hit_2r: HitFlag = "unresolved"
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit: bool = False
    error: Optional[str] = None

    def to_row(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "symbol": self.symbol,
            "interval": self.interval,
            "expected_pattern": self.expected_pattern,
            "expected_side": self.expected_side,
            "regime": self.regime,
            "htf_aligned": bool(self.htf_aligned),
            "source": self.source,
            "pattern_emitted": ",".join(self.pattern_emitted),
            "pattern_match": bool(self.pattern_match),
            "side_match": bool(self.side_match),
            "predicted_pattern": self.predicted_signal.pattern if self.predicted_signal else None,
            "predicted_side": self.predicted_signal.side if self.predicted_signal else None,
            "predicted_entry": self.predicted_signal.entry_px if self.predicted_signal else None,
            "predicted_stop": self.predicted_signal.stop_px if self.predicted_signal else None,
            "expected_r": self.expected_r,
            "probability": self.probability,
            "realized_r": self.realized_r,
            "hit_1r": self.hit_1r,
            "hit_2r": self.hit_2r,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_hit": bool(self.cache_hit),
            "error": self.error,
        }


class EvalRunner:
    """Run an analyst against every sample in a dataset."""

    def __init__(
        self,
        analyst: Analyst,
        dataset: GoldenDataset,
        te: Optional[TraderEquation] = None,
        max_concurrent: int = 4,
    ) -> None:
        if analyst is None:
            raise ValueError("EvalRunner requires an analyst")
        if dataset is None:
            raise ValueError("EvalRunner requires a dataset")
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self._analyst = analyst
        self._dataset = dataset
        self._te = te
        self._max_concurrent = int(max_concurrent)

    @property
    def analyst(self) -> Analyst:
        return self._analyst

    @property
    def dataset(self) -> GoldenDataset:
        return self._dataset

    async def run(self) -> "EvalReport":
        """Score every sample. Returns a populated :class:`EvalReport`."""
        from src.brooks.eval.report import EvalReport  # local import avoids cycle

        sem = asyncio.Semaphore(self._max_concurrent)

        async def _score(sample: GoldenSample) -> SampleResult:
            async with sem:
                return await _score_sample(sample, self._analyst, self._te)

        results = await asyncio.gather(*[_score(s) for s in self._dataset])
        return EvalReport(
            results=list(results),
            analyst_name=getattr(self._analyst, "name", "analyst"),
            dataset_size=len(self._dataset),
        )


# ---------------------------------------------------------------------------
# Per-sample scoring
# ---------------------------------------------------------------------------


async def _score_sample(
    sample: GoldenSample,
    analyst: Analyst,
    te: Optional[TraderEquation],
) -> SampleResult:
    ctx_bars = sample.context_bars
    ctx = BrooksContext(
        symbol=sample.symbol,
        primary=TFSnapshot(interval=sample.interval, bars=ctx_bars),
    )
    started = time.perf_counter()
    error: Optional[str] = None
    signals: List[Signal] = []
    try:
        signals = list(await analyst.analyze(ctx))
    except Exception as exc:  # pragma: no cover - propagated via SampleResult.error
        error = f"{type(exc).__name__}: {exc}"
    latency = (time.perf_counter() - started) * 1000.0

    predicted = _pick_signal(signals, sample)
    pattern_match = predicted is not None and predicted.pattern == sample.expected_pattern
    side_match = predicted is not None and predicted.side == sample.expected_side

    result = SampleResult(
        sample_id=sample.id,
        symbol=sample.symbol,
        interval=sample.interval,
        expected_pattern=sample.expected_pattern,
        expected_side=sample.expected_side,
        regime=sample.regime,
        htf_aligned=sample.htf_aligned,
        source=sample.source,
        pattern_match=pattern_match,
        side_match=side_match,
        pattern_emitted=sorted({s.pattern for s in signals}),
        predicted_signal=predicted,
        latency_ms=latency,
        error=error,
    )

    if predicted is not None:
        meta = predicted.meta or {}
        result.input_tokens = int(meta.get("input_tokens", 0) or 0)
        result.output_tokens = int(meta.get("output_tokens", 0) or 0)
        result.cache_hit = bool(meta.get("cache_hit", False))

    if te is not None and predicted is not None:
        prob, expected_r = te.score(
            predicted,
            regime=sample.regime,
            htf_aligned=sample.htf_aligned,
        )
        result.probability = float(prob)
        result.expected_r = float(expected_r)

    if predicted is not None:
        realized, hit_1r, hit_2r = _realized_r(predicted, sample.future_bars)
        result.realized_r = realized
        result.hit_1r = hit_1r
        result.hit_2r = hit_2r

    return result


def _pick_signal(signals: Sequence[Signal], sample: GoldenSample) -> Optional[Signal]:
    """Pick the signal that best matches the ground truth.

    Preference order: exact ``(pattern, side)`` match → same side →
    first emitted signal.
    """
    if not signals:
        return None
    for s in signals:
        if s.pattern == sample.expected_pattern and s.side == sample.expected_side:
            return s
    for s in signals:
        if s.side == sample.expected_side:
            return s
    return signals[0]


def _realized_r(
    signal: Signal,
    future_bars: Sequence[Bar],
) -> tuple[Optional[float], HitFlag, HitFlag]:
    """Walk forward and return (realized_R, hit_1r, hit_2r).

    The simulation is intentionally minimal: we walk forward bar-by-bar
    flagging whether 1R / 2R were touched in either direction. The
    realized R is determined by the first decisive event (stop or 2R
    target). If neither is reached within the window, we report the
    final mark-to-market vs entry as the realized R and resolve the hit
    flags as ``"miss"`` (we observed enough bars to know the signal did
    not work out within the window).
    """
    if not future_bars:
        return None, "unresolved", "unresolved"

    one_r = abs(signal.entry_px - signal.stop_px)
    if one_r <= 0:
        return None, "unresolved", "unresolved"

    side = signal.side
    entry = signal.entry_px
    stop = signal.stop_px
    target_1r = entry + (one_r if side == "long" else -one_r)
    target_2r = entry + (2 * one_r if side == "long" else -2 * one_r)

    hit_1r: HitFlag = "miss"
    hit_2r: HitFlag = "miss"

    for bar in future_bars:
        hit_stop = bar.low <= stop if side == "long" else bar.high >= stop
        hit_2r_now = bar.high >= target_2r if side == "long" else bar.low <= target_2r
        hit_1r_now = bar.high >= target_1r if side == "long" else bar.low <= target_1r

        if hit_1r_now:
            hit_1r = "hit"
        if hit_2r_now:
            hit_1r = "hit"
            hit_2r = "hit"
            realized = 2.0
            return realized, hit_1r, hit_2r
        if hit_stop:
            realized = -1.0
            return realized, hit_1r, hit_2r

    last_close = future_bars[-1].close
    realized = (last_close - entry) / one_r if side == "long" else (entry - last_close) / one_r
    return realized, hit_1r, hit_2r
