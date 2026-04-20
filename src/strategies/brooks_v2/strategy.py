"""BrooksStrategyV2 — L4 orchestrator.

On each bar:
  1. Update L1 features (``BarFeatureExtractor``)
  2. Update L2 ``MarketStructure``
  3. Feed each enabled L3 detector → collect signals
  4. ``SignalAggregator.resolve`` with confluence gate
  5. ``BrooksRiskModel``: size + partial + trail
  6. Write decision snapshot to session log (``extra`` field carries the full JSON)

LLM veto is optional and only narrows (never widens) a rule-generated decision.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.analysis.brooks.aggregator import AggregatedDecision, SignalAggregator
from src.analysis.brooks.features import BarFeatureExtractor, ExtendedBarFeatures
from src.analysis.brooks.patterns import (
    DetectorContext,
    DoubleBottomDetector,
    DoubleTopDetector,
    FinalFlagDetector,
    H2Detector,
    L2Detector,
    MeasuredMoveTargeter,
    PatternSignal,
    WedgeDetector,
)
from src.analysis.brooks.structure import MarketStructureTracker
from src.core.base import Bar, Strategy
from src.core.timeframe_resampler import TimeframeResampler
from src.strategies.brooks_v2.risk import BrooksRiskModel, PositionState
from src.strategies.registry import StrategyRegistry


@StrategyRegistry.register(
    "brooks_v2",
    label="Brooks Price Action v2",
    description="Rule-first Brooks price-action with H2/L2/DoubleTB/Wedge/FinalFlag + 1R partial + swing trail",
)
class BrooksStrategyV2(Strategy):
    def __init__(self, db_client=None, session_id: Optional[str] = None, **params):
        super().__init__(session_id)
        self.db_client = db_client
        self.params = self._apply_defaults(params)
        self._rng = np.random.default_rng(self.params["seed"])

        self._extractors: Dict[str, BarFeatureExtractor] = {}
        self._structures: Dict[str, MarketStructureTracker] = {}
        self._detectors: Dict[str, List[Any]] = {}
        self._feature_history: Dict[str, List[ExtendedBarFeatures]] = {}
        self._positions: Dict[str, PositionState] = {}
        self._stop_order_ids: Dict[str, str] = {}
        self._entry_order_ids: Dict[str, str] = {}
        self._pending_decisions: Dict[str, AggregatedDecision] = {}

        self._aggregator = SignalAggregator(confluence_n=self.params["confluence_n"])
        self._risk = BrooksRiskModel(
            risk_pct=self.params["risk_pct"] / 100.0,
            partial_close_1r=self.params["partial_close_1r"],
            swing_trail=self.params["swing_trail"],
            max_concurrent=self.params["max_concurrent"],
        )
        self._targeter = MeasuredMoveTargeter()

        self._resamplers: Dict[str, TimeframeResampler] = {}
        self._higher_tf_structures: Dict[str, Dict[str, MarketStructureTracker]] = {}
        self._higher_tf_extractors: Dict[str, Dict[str, BarFeatureExtractor]] = {}

        # Optional LLM veto (default off)
        self._llm_veto = None
        if self.params["use_llm_veto"]:
            try:
                from src.alpha.llm.backends import get_backend

                self._llm_veto = get_backend(self.params.get("llm_backend", "openai"))
            except Exception as exc:  # pragma: no cover — optional path
                self._log(f"LLM veto disabled ({exc})", level="WARNING")
                self._llm_veto = None

    @classmethod
    def _apply_defaults(cls, user_params: Dict[str, Any]) -> Dict[str, Any]:
        defaults = {
            "seed": 42,
            "swing_k": 3,
            "breakout_lookback": 20,
            "ema_period": 20,
            "atr_period": 14,
            "min_body_pct": 50,
            "min_rr": 2.0,
            "risk_pct": 0.5,  # percent (not fraction) — converted inside
            "confluence_n": 1,
            "max_concurrent": 1,
            "detectors_enabled": ["h2", "l2", "double_top", "double_bottom", "wedge", "final_flag"],
            "use_mtf": False,
            "mtf_intervals": ["15m", "1h"],
            "use_llm_veto": False,
            "partial_close_1r": True,
            "swing_trail": True,
        }
        defaults.update(user_params or {})
        return defaults

    @classmethod
    def get_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "seed": {"type": "int", "default": 42, "description": "Random seed for reproducibility"},
            "swing_k": {"type": "int", "default": 3, "min": 1, "max": 10, "description": "Swing fractal K (bars)"},
            "breakout_lookback": {
                "type": "int",
                "default": 20,
                "min": 5,
                "max": 100,
                "description": "N-bar breakout lookback for always-in",
            },
            "min_rr": {"type": "float", "default": 2.0, "min": 1.0, "max": 5.0, "description": "Minimum RR"},
            "risk_pct": {"type": "float", "default": 0.5, "min": 0.1, "max": 5.0, "description": "Risk per trade (%)"},
            "confluence_n": {
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 5,
                "description": "Min detectors agreeing for entry",
            },
            "detectors_enabled": {
                "type": "list",
                "default": ["h2", "l2", "double_top", "double_bottom", "wedge", "final_flag"],
                "description": "Enabled pattern detectors",
            },
            "use_mtf": {"type": "bool", "default": False, "description": "Filter by higher-TF structure"},
            "mtf_intervals": {"type": "list", "default": ["15m", "1h"], "description": "Higher TFs"},
            "use_llm_veto": {"type": "bool", "default": False, "description": "Route decisions through LLM for veto"},
            "partial_close_1r": {"type": "bool", "default": True, "description": "Close 50% at 1R; move stop to BE"},
            "swing_trail": {"type": "bool", "default": True, "description": "Trail stop to confirmed swings"},
            "max_concurrent": {
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 10,
                "description": "Max simultaneous positions",
            },
        }

    # ---- lifecycle ----------------------------------------------------

    def _ensure_symbol_state(self, symbol: str) -> None:
        if symbol in self._extractors:
            return
        self._extractors[symbol] = BarFeatureExtractor(
            swing_k=self.params["swing_k"],
            ema_period=self.params["ema_period"],
            atr_period=self.params["atr_period"],
            breakout_lookback=self.params["breakout_lookback"],
        )
        self._structures[symbol] = MarketStructureTracker(
            self._extractors[symbol], breakout_lookback=self.params["breakout_lookback"]
        )
        self._feature_history[symbol] = []
        self._detectors[symbol] = self._build_detectors()
        if self.params["use_mtf"]:
            self._resamplers[symbol] = TimeframeResampler(base="5m", higher=self.params["mtf_intervals"])
            self._higher_tf_structures[symbol] = {}
            self._higher_tf_extractors[symbol] = {}
            for h in self.params["mtf_intervals"]:
                ext = BarFeatureExtractor(
                    swing_k=self.params["swing_k"],
                    ema_period=self.params["ema_period"],
                    atr_period=self.params["atr_period"],
                    breakout_lookback=self.params["breakout_lookback"],
                )
                self._higher_tf_extractors[symbol][h] = ext
                self._higher_tf_structures[symbol][h] = MarketStructureTracker(
                    ext, breakout_lookback=self.params["breakout_lookback"]
                )

    def _build_detectors(self) -> List[Any]:
        enabled = set(self.params["detectors_enabled"])
        out: List[Any] = []
        if "h2" in enabled:
            out.append(H2Detector())
        if "l2" in enabled:
            out.append(L2Detector())
        if "double_top" in enabled:
            out.append(DoubleTopDetector())
        if "double_bottom" in enabled:
            out.append(DoubleBottomDetector())
        if "wedge" in enabled:
            out.append(WedgeDetector(direction="long"))
            out.append(WedgeDetector(direction="short"))
        if "final_flag" in enabled:
            out.append(FinalFlagDetector())
        return out

    # ---- main entry ---------------------------------------------------

    def on_bar(self, bars: Dict[str, Bar]):
        for symbol, bar in bars.items():
            self._ensure_symbol_state(symbol)
            self._fill_pending_entry(symbol, bar)
            self._process_symbol_bar(symbol, bar)

    def _process_symbol_bar(self, symbol: str, bar: Bar) -> None:
        ext = self._extractors[symbol]
        tracker = self._structures[symbol]
        history = self._feature_history[symbol]

        ts_ns = int(bar.timestamp.timestamp() * 1_000_000_000)
        feat = ext.on_bar(ts_ns, bar.open, bar.high, bar.low, bar.close)
        struct = tracker.on_features(feat)
        history.append(feat)
        if len(history) > 300:
            del history[:-300]

        # Update any MTF resamplers
        mtf_snapshot: Dict[str, Any] = {}
        if symbol in self._resamplers:
            out = self._resamplers[symbol].update(bar)
            for tf, closed in out.items():
                if closed is None:
                    continue
                tf_ext = self._higher_tf_extractors[symbol][tf]
                tf_tracker = self._higher_tf_structures[symbol][tf]
                tf_ts = int(closed.timestamp.timestamp() * 1_000_000_000)
                tf_feat = tf_ext.on_bar(tf_ts, closed.open, closed.high, closed.low, closed.close)
                tf_struct = tf_tracker.on_features(tf_feat)
                mtf_snapshot[tf] = tf_struct.to_dict()

        # Manage any open position first
        if symbol in self._positions:
            self._manage_position(symbol, bar, struct, feat)

        # Detector fanout
        ctx = DetectorContext(feat=feat, structure=struct, recent_features=history, params=self.params)
        signals: List[PatternSignal] = []
        for det in self._detectors[symbol]:
            sig = det.on_bar(ctx)
            if sig is None:
                continue
            # MTF alignment filter
            if self.params["use_mtf"] and mtf_snapshot:
                if not self._mtf_aligned(sig.side, symbol):
                    continue
            signals.append(sig)

        decision: Optional[AggregatedDecision] = self._aggregator.resolve(signals)

        # Persist decision snapshot (always, even when no signal → easier to audit)
        self._log(
            f"bar {feat.bar_idx} {symbol}: always_in={struct.always_in} signals={[s.detector for s in signals]}",
            level="DEBUG",
            source="strategy",
            extra={
                "bar_idx": feat.bar_idx,
                "features": feat.to_dict(),
                "structure": struct.to_dict(),
                "pattern_states": [d.state_snapshot() for d in self._detectors[symbol]],
                "signals": [s.to_dict() for s in signals],
                "decision": decision.to_dict() if decision else None,
                "mtf": mtf_snapshot,
            },
        )

        if decision is None:
            return
        if symbol in self._positions:
            return  # don't stack
        if not self._risk.can_open(sum(1 for p in self._positions.values() if p.qty_open > 0)):
            return

        # Optional LLM veto
        if self._llm_veto is not None and not self._llm_confirm(symbol, decision, history):
            self._log(f"LLM vetoed {decision.side} decision at bar {feat.bar_idx}", level="INFO")
            return

        self._submit_entry(symbol, bar, decision)

    # ---- MTF ----------------------------------------------------------

    def _mtf_aligned(self, side: str, symbol: str) -> bool:
        structs = self._higher_tf_structures.get(symbol, {})
        if not structs:
            return True
        target = "long" if side == "long" else "short"
        # All higher TFs must either agree or be neutral.
        for tf, tracker in structs.items():
            ai = tracker.state.always_in
            if ai != target and ai != "neutral":
                return False
        return True

    # ---- entry / exit --------------------------------------------------

    def _submit_entry(self, symbol: str, bar: Bar, decision: AggregatedDecision) -> None:
        account = self.engine.broker.get_account_info() if self.engine else {"total_equity": 100000.0}
        equity = account.get("total_equity", 0.0)
        cash = account.get("cash", equity)
        qty = self._risk.sizing(
            equity=equity, entry_px=decision.entry_px, stop_px=decision.stop_px, available_cash=cash
        )
        if qty <= 0:
            self._log(f"sizing=0 at bar {decision.signal_bar_idx} — skipping", level="DEBUG")
            return
        # Round qty: crypto-friendly small decimals
        qty = round(qty, 6)
        if qty <= 0:
            return

        self._pending_decisions[symbol] = decision
        self._log(
            f"ENTRY {decision.side} {symbol} qty={qty} entry={decision.entry_px:.4f} "
            f"stop={decision.stop_px:.4f} via={decision.hit_detectors}",
            level="INFO",
        )

        if decision.side == "long":
            oid = self.engine.submit_order(_StopOrder(symbol, "buy", qty, decision.entry_px))
        else:
            oid = self.engine.submit_order(_StopOrder(symbol, "sell_short", qty, decision.entry_px))
        self._entry_order_ids[symbol] = oid

    def _fill_pending_entry(self, symbol: str, bar: Bar) -> None:
        """Detect if a pending stop-entry triggered during the current bar.

        Uses the engine broker trade record — if our stop price is inside
        [bar.low, bar.high], we assume fill at stop.
        """
        if symbol not in self._pending_decisions:
            return
        decision = self._pending_decisions[symbol]
        triggered = (decision.side == "long" and bar.high >= decision.entry_px) or (
            decision.side == "short" and bar.low <= decision.entry_px
        )
        if not triggered:
            return
        pos = PositionState(
            symbol=symbol,
            side=decision.side,
            entry_px=decision.entry_px,
            stop_px=decision.stop_px,
            qty_initial=0.0,
            qty_open=0.0,
            one_r=abs(decision.entry_px - decision.stop_px),
            entry_bar_idx=decision.signal_bar_idx,
            entry_timestamp_ns=decision.timestamp_ns,
        )
        # Figure out actual filled qty from broker
        if self.engine and hasattr(self.engine.broker, "positions"):
            pos.qty_initial = abs(self.engine.broker.positions.get(symbol, 0.0))
            pos.qty_open = pos.qty_initial
        self._positions[symbol] = pos
        del self._pending_decisions[symbol]

    def _manage_position(self, symbol: str, bar: Bar, structure, feat: ExtendedBarFeatures) -> None:
        pos = self._positions[symbol]

        stop_hit = self._risk.check_stop_hit(pos, bar.high, bar.low)
        if stop_hit and stop_hit.quantity > 0:
            self._emit_order(stop_hit, execution="IMMEDIATE_CLOSE")
            self._log(f"STOP HIT {symbol} {stop_hit.reason}", level="INFO")
            pos.qty_open = 0.0
            del self._positions[symbol]
            return

        actions = self._risk.manage(
            pos,
            bar_high=bar.high,
            bar_low=bar.low,
            bar_close=bar.close,
            confirmed_swing_highs=structure.confirmed_swing_highs,
            confirmed_swing_lows=structure.confirmed_swing_lows,
        )
        for a in actions:
            if a.quantity > 0:
                self._emit_order(a, execution="IMMEDIATE_CLOSE")
            if a.new_stop_px is not None:
                self._log(
                    f"STOP-ADJ {symbol} → {a.new_stop_px:.4f} ({a.reason})",
                    level="INFO",
                )

    def _emit_order(self, action, execution: str = "IMMEDIATE_CLOSE") -> None:
        if self.engine is None or action.quantity <= 0:
            return
        if action.order_type == "buy":
            self.buy(action.symbol, action.quantity, execution_type=execution)
        elif action.order_type == "sell":
            self.sell(action.symbol, action.quantity, execution_type=execution)
        elif action.order_type == "sell_short":
            self.sell_short(action.symbol, action.quantity, execution_type=execution)
        elif action.order_type == "buy_to_cover":
            self.buy_to_cover(action.symbol, action.quantity, execution_type=execution)

    # ---- LLM veto -----------------------------------------------------

    def _llm_confirm(self, symbol: str, decision: AggregatedDecision, history: List[ExtendedBarFeatures]) -> bool:
        if self._llm_veto is None:
            return True
        try:  # pragma: no cover — network path
            payload = {
                "symbol": symbol,
                "proposed": decision.to_dict(),
                "recent_bars": [f.to_dict() for f in history[-20:]],
            }
            resp = self._llm_veto.complete(
                prompt=str(payload),
                temperature=0.0,
                seed=int(self.params["seed"]),
                response_format={"type": "json_object"},
            )
            import json

            obj = json.loads(resp)
            self._log(f"LLM veto resp: {obj}", level="DEBUG", extra={"llm_raw": obj})
            return bool(obj.get("confirm", False))
        except Exception as exc:
            self._log(f"LLM veto error — falling back to confirm: {exc}", level="WARNING")
            return True


class _StopOrder:
    """Lightweight shim — converts to :class:`Order` on submit via Strategy helpers."""

    def __new__(cls, symbol, type_, qty, stop_px):
        from src.core.base import Order

        return Order(symbol=symbol, type=type_, quantity=qty, stop_price=stop_px, execution_type="NEXT_OPEN")
