"""Rule-based analyst — runs every registered pattern detector once.

The Phase 2 :class:`~src.brooks.context.BrooksContext` does not yet carry
features / structure on its ``TFSnapshot``; the rule analyst rebuilds
them from ``ctx.primary.bars`` so this layer can ship before Phase 3
populates the context. Phase 3 will swap the streaming reconstruction
out for direct field access.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.brooks.analyst.base import AnalystRegistry
from src.brooks.context import BrooksContext
from src.brooks.features import BarFeatureExtractor, ExtendedBarFeatures
from src.brooks.patterns.base import DetectorContext, PatternSignal
from src.brooks.patterns.registry import PatternRegistry
from src.brooks.schema import Signal
from src.brooks.structure import MarketStructure, MarketStructureTracker


@AnalystRegistry.register("rule")
class RuleAnalyst:
    """Run every registered pattern detector and emit unified Signals.

    Parameters
    ----------
    detector_names:
        Optional explicit list of detector names. ``None`` selects every
        detector currently in :class:`PatternRegistry`.
    params:
        Optional ``{detector_name: {param: value, ...}}`` mapping forwarded
        to ``PatternRegistry.build`` when constructing each detector.
    extractor_kwargs:
        Optional kwargs for :class:`BarFeatureExtractor`. Defaults match
        ``brooks_v2.strategy.BrooksStrategyV2`` so signal parity holds.
    structure_kwargs:
        Optional kwargs for :class:`MarketStructureTracker`.
    """

    name = "rule"

    def __init__(
        self,
        detector_names: Optional[List[str]] = None,
        params: Optional[Dict[str, Dict]] = None,
        extractor_kwargs: Optional[Dict] = None,
        structure_kwargs: Optional[Dict] = None,
    ) -> None:
        names = list(detector_names) if detector_names is not None else PatternRegistry.all()
        param_map = params or {}
        self._detectors = [PatternRegistry.build(n, **param_map.get(n, {})) for n in names]
        self._detector_names = names
        self._extractor_kwargs = extractor_kwargs or {}
        self._structure_kwargs = structure_kwargs or {}

    async def analyze(self, ctx: BrooksContext) -> List[Signal]:
        """Replay every bar through each detector's FSM; emit signals
        that arise on the latest bar.

        The Brooks pattern detectors are stateful (H2/L2 build a multi-bar
        pullback FSM, doubles wait for paired swings, etc.), so they must
        observe the full history to decide whether the *current* bar
        completes a setup. We discard signals that fired on earlier bars —
        callers analysing those bars would have invoked the analyst with a
        shorter context at the time.
        """
        bars = ctx.primary.bars
        if not bars:
            return []

        ext = BarFeatureExtractor(**self._extractor_kwargs)
        tracker = MarketStructureTracker(ext, **self._structure_kwargs)
        history: List[ExtendedBarFeatures] = []
        out: List[Signal] = []
        last_idx = len(bars) - 1
        last_struct: Optional[MarketStructure] = None

        for i, bar in enumerate(bars):
            feat = ext.on_bar(bar.timestamp_ns, bar.open, bar.high, bar.low, bar.close)
            last_struct = tracker.on_features(feat)
            history.append(feat)
            detector_ctx = DetectorContext(
                feat=feat,
                structure=last_struct,
                recent_features=history,
                params={},
                htf=ctx.htf,
            )
            for det in self._detectors:
                ps = det.on_bar(detector_ctx)
                if ps is None:
                    continue
                if i == last_idx:
                    out.append(_pattern_signal_to_signal(ps))
        return out


def _pattern_signal_to_signal(ps: PatternSignal) -> Signal:
    """Promote an internal :class:`PatternSignal` to a unified :class:`Signal`."""
    meta = dict(ps.metadata) if ps.metadata else {}
    meta.setdefault("timestamp_ns", ps.timestamp_ns)
    return Signal(
        pattern=ps.detector,
        side=ps.side,
        signal_bar_idx=ps.signal_bar_idx,
        entry_px=ps.entry_px,
        stop_px=ps.stop_px,
        target_px=None,
        probability=0.5,
        quality=0.5,
        reasoning=ps.reason,
        source=f"rule:{ps.detector}",
        meta=meta,
    )
