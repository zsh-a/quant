"""Unified :class:`BrooksStrategy` — the single Phase 3 orchestrator.

One concrete class replaces the three old entry points (``v1`` /
``llm_pipeline`` / ``v2``). It plugs together every existing building
block — context builder, analyst, aggregator, Trader's Equation, EV
gate, portfolio guard, sizer, stop ladder, time stop — and drives them
once per bar on a streaming feed.

Analyst selection is parameterised via the unified
:class:`~src.brooks.analyst.base.AnalystRegistry`:

* ``analyst="rule"``            — run every registered pattern detector
  once per bar (direct equivalent of the legacy rule engine)
* ``analyst="llm:<model>"``     — delegate to :class:`~src.brooks.analyst.llm.LLMAnalyst`
  with the named provider model
* ``analyst="ensemble.<name>"`` — room for a future ensemble analyst
  registered under that exact name

The strategy itself remains deterministic and synchronous; the LLM
analyst runs inside ``asyncio.run`` to fit the ``Strategy.on_bar``
contract. Heavy work (feature extraction, structure, regime) is done
once per bar in ``_process_symbol_bar``; the analyst re-uses the cached
snapshots through :attr:`TFSnapshot.features` / ``.structure`` /
``.regime``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.brooks.analyst.base import AnalystRegistry
from src.brooks.analyst.rule import RuleAnalyst  # noqa: F401 — triggers registration
from src.brooks.context import AccountSnapshot, Bar as BrooksBar, BrooksContext, TFSnapshot
from src.brooks.decision.aggregator import AggregatedDecision, SignalAggregator
from src.brooks.decision.ev_gate import EVGate
from src.brooks.decision.hit_rate import HitRateTable
from src.brooks.decision.trader_equation import TraderEquation
from src.brooks.features import BarFeatureExtractor, ExtendedBarFeatures
from src.brooks.regime import BrooksRegimeClassifier, RegimeSnapshot
from src.brooks.risk.portfolio_guard import PortfolioGuard
from src.brooks.risk.sizer import FixedPercentSizer, KellySizer, Sizer
from src.brooks.risk.state import PositionState
from src.brooks.risk.stop_ladder import StopLadder
from src.brooks.risk.time_stop import TimeStop
from src.brooks.schema import Decision
from src.brooks.structure import MarketStructure, MarketStructureTracker
from src.core.base import Bar, Order, Strategy
from src.core.timeframe_resampler import TimeframeResampler
from src.strategies.registry import StrategyRegistry

__all__ = ["BrooksStrategy"]


_HISTORY_CAP = 400  # cap per-symbol feature history to avoid unbounded growth


@dataclass
class _HTFState:
    """Per-symbol HTF aggregation state (extractor + tracker + regime)."""

    extractor: BarFeatureExtractor
    tracker: MarketStructureTracker
    regime: BrooksRegimeClassifier
    features: List[ExtendedBarFeatures]
    last_feat: Optional[ExtendedBarFeatures] = None
    last_struct: Optional[MarketStructure] = None
    last_regime: Optional[RegimeSnapshot] = None
    recent_bars: List[BrooksBar] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.recent_bars is None:
            self.recent_bars = []


@StrategyRegistry.register(
    "brooks",
    label="Brooks (rule / LLM / ensemble)",
    description=(
        "Unified Brooks price-action strategy. Analyst is pluggable — "
        "'rule' runs every pattern detector, 'llm:<model>' delegates "
        "to the LLM analyst, 'ensemble.<name>' targets a named ensemble."
    ),
)
class BrooksStrategy(Strategy):
    """Phase 3.5 strategy — single class, pluggable analyst, HTF-aware."""

    def __init__(
        self,
        db_client=None,
        session_id: Optional[str] = None,
        **params: Any,
    ) -> None:
        super().__init__(session_id)
        self.db_client = db_client
        self.params = self._apply_defaults(params)

        # --- plug components --------------------------------------------------
        analyst_name: str = self.params["analyst"]
        self._analyst = AnalystRegistry.build(
            analyst_name, **self.params.get("analyst_params", {})
        )
        self._aggregator = SignalAggregator(**self.params.get("aggregator_params", {}))
        self._te = _build_te(self.params.get("te_params", {}))
        self._ev_gate = EVGate(
            self._te,
            min_expected_r=float(self.params.get("min_expected_r", 0.1)),
        )
        self._sizer: Sizer = _build_sizer(self.params.get("sizer_params", {}))
        self._stop_ladder = StopLadder(**self.params.get("stop_ladder_params", {}))
        self._time_stop = TimeStop(**self.params.get("time_stop_params", {}))
        self._portfolio = PortfolioGuard(**self.params.get("portfolio_params", {}))

        # --- per-symbol streaming state ---------------------------------------
        self._extractors: Dict[str, BarFeatureExtractor] = {}
        self._structures: Dict[str, MarketStructureTracker] = {}
        self._regimes: Dict[str, BrooksRegimeClassifier] = {}
        self._feature_history: Dict[str, List[ExtendedBarFeatures]] = {}
        self._primary_bars: Dict[str, List[BrooksBar]] = {}
        self._htf_state: Dict[str, Dict[str, _HTFState]] = {}
        self._resamplers: Dict[str, TimeframeResampler] = {}

        # --- positions & pending entries --------------------------------------
        self._positions: Dict[str, PositionState] = {}
        self._pending_decisions: Dict[str, Decision] = {}
        self._entry_order_ids: Dict[str, str] = {}

    # ------------------------------------------------------------------ defaults

    @classmethod
    def _apply_defaults(cls, user_params: Dict[str, Any]) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {
            "analyst": "rule",
            "analyst_params": {},
            "aggregator_params": {"confluence_n": 1},
            "te_params": {},
            "min_expected_r": 0.1,
            "sizer_params": {"kind": "kelly"},
            "stop_ladder_params": {},
            "time_stop_params": {},
            "portfolio_params": {},
            # Multi-TF
            "base_interval": "5m",
            "mtf_intervals": [],
            "primary_bar_window": 200,
            "htf_bar_window": 120,
            # Context rendering
            "context_budget_tokens": 2000,
            # Misc
            "seed": 42,
        }
        defaults.update(user_params or {})
        return defaults

    @classmethod
    def get_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "analyst": {
                "type": "str",
                "default": "rule",
                "description": "Analyst selector: 'rule', 'llm:<model>', 'ensemble.<name>'.",
            },
            "min_expected_r": {
                "type": "float",
                "default": 0.1,
                "description": "EV gate: drop signals whose Trader's-Equation E < this.",
            },
            "mtf_intervals": {
                "type": "list",
                "default": [],
                "description": "Higher-TF intervals to aggregate (e.g. ['1h','4h']).",
            },
            "base_interval": {
                "type": "str",
                "default": "5m",
                "description": "Primary bar interval for the MTF resampler.",
            },
        }

    # ------------------------------------------------------------------ on_bar

    def on_bar(self, bars: Dict[str, Bar]) -> None:
        for symbol, bar in bars.items():
            self._ensure_symbol_state(symbol)
            self._fill_pending_entry(symbol, bar)
            self._process_symbol_bar(symbol, bar)

    # ------------------------------------------------------------------ setup

    def _ensure_symbol_state(self, symbol: str) -> None:
        if symbol in self._extractors:
            return
        self._extractors[symbol] = BarFeatureExtractor()
        self._structures[symbol] = MarketStructureTracker(self._extractors[symbol])
        self._regimes[symbol] = BrooksRegimeClassifier()
        self._feature_history[symbol] = []
        self._primary_bars[symbol] = []
        self._htf_state[symbol] = {}

        mtf = self.params.get("mtf_intervals") or []
        if mtf:
            self._resamplers[symbol] = TimeframeResampler(
                base=self.params["base_interval"], higher=list(mtf)
            )
            for h in mtf:
                self._htf_state[symbol][h] = _HTFState(
                    extractor=BarFeatureExtractor(),
                    tracker=MarketStructureTracker(BarFeatureExtractor()),
                    regime=BrooksRegimeClassifier(),
                    features=[],
                )
                # rebuild tracker so it shares the freshly-made extractor
                st = self._htf_state[symbol][h]
                st.tracker = MarketStructureTracker(st.extractor)

    # ------------------------------------------------------------------ per-bar

    def _process_symbol_bar(self, symbol: str, bar: Bar) -> None:
        ts_ns = int(bar.timestamp.timestamp() * 1_000_000_000)

        # ---- LTF features/structure/regime for this bar ---------------------
        feat = self._extractors[symbol].on_bar(ts_ns, bar.open, bar.high, bar.low, bar.close)
        struct = self._structures[symbol].on_features(feat)
        history = self._feature_history[symbol]
        history.append(feat)
        if len(history) > _HISTORY_CAP:
            del history[: len(history) - _HISTORY_CAP]
        regime = self._regimes[symbol].classify(history, struct)

        primary_bars = self._primary_bars[symbol]
        primary_bars.append(_bar_to_brooks(bar, ts_ns))
        window = int(self.params.get("primary_bar_window", 200))
        if len(primary_bars) > window:
            del primary_bars[: len(primary_bars) - window]

        # ---- update HTF aggregations ---------------------------------------
        self._update_htf(symbol, bar)

        # ---- manage any open position --------------------------------------
        if symbol in self._positions:
            self._manage_position(symbol, bar, struct, feat)
            if symbol not in self._positions:
                return  # closed this bar; nothing more to do

        # ---- build the analyst context -------------------------------------
        ctx = self._build_context(symbol, feat, struct, regime, primary_bars)

        # ---- analyst → signals ---------------------------------------------
        signals = asyncio.run(self._analyst.analyze(ctx))
        if not signals:
            return

        # ---- HTF alignment for this bar/side (use majority side) -----------
        # The aggregator picks one side; score both here cheaply.
        combined = self._aggregator.resolve(signals)
        if combined is None:
            return
        if symbol in self._positions or symbol in self._pending_decisions:
            return  # don't stack

        htf_tag = ctx.htf_alignment_for(combined.side)
        decisions = self._ev_gate.filter(
            combined.raw_signals,
            regime=regime.regime.value,
            symbol=symbol,
            htf_alignment=htf_tag,
        )
        if not decisions:
            return
        decision = _select_decision(decisions, combined)
        if decision is None:
            return

        # ---- portfolio guard -----------------------------------------------
        equity, cash = self._account_equity_cash()
        qty = self._sizer.size(
            equity=equity,
            entry_px=decision.entry_px,
            stop_px=decision.stop_px,
            probability=decision.probability,
            expected_r=decision.expected_r,
            available_cash=cash,
        )
        qty = round(qty, 6)
        if qty <= 0:
            return
        risk_pct = _risk_pct_from_qty(qty, decision, equity)
        ok, reason = self._portfolio.can_open(
            new_symbol=symbol,
            new_risk_pct=risk_pct,
            open_positions=self._positions,
        )
        if not ok:
            self._log(f"portfolio guard blocked {symbol}: {reason}", level="DEBUG")
            return

        decision.quantity = qty
        self._submit_entry(symbol, decision, risk_pct)

    # ------------------------------------------------------------------ HTF

    def _update_htf(self, symbol: str, bar: Bar) -> None:
        res = self._resamplers.get(symbol)
        if res is None:
            return
        out = res.update(bar)
        for tf, closed in out.items():
            if closed is None:
                continue
            st = self._htf_state[symbol][tf]
            ts_ns = int(closed.timestamp.timestamp() * 1_000_000_000)
            f = st.extractor.on_bar(ts_ns, closed.open, closed.high, closed.low, closed.close)
            s = st.tracker.on_features(f)
            st.features.append(f)
            cap = int(self.params.get("htf_bar_window", 120))
            if len(st.features) > _HISTORY_CAP:
                del st.features[: len(st.features) - _HISTORY_CAP]
            r = st.regime.classify(st.features, s)
            st.last_feat = f
            st.last_struct = s
            st.last_regime = r
            st.recent_bars.append(_bar_to_brooks(closed, ts_ns))
            if len(st.recent_bars) > cap:
                del st.recent_bars[: len(st.recent_bars) - cap]

    def _build_context(
        self,
        symbol: str,
        feat: ExtendedBarFeatures,
        struct: MarketStructure,
        regime: RegimeSnapshot,
        primary_bars: List[BrooksBar],
    ) -> BrooksContext:
        primary = TFSnapshot(
            interval=self.params["base_interval"],
            bars=list(primary_bars),
            features=feat,
            structure=struct,
            regime=regime,
        )
        htf_map: Dict[str, TFSnapshot] = {}
        for tf, st in self._htf_state.get(symbol, {}).items():
            if not st.recent_bars:
                continue
            htf_map[tf] = TFSnapshot(
                interval=tf,
                bars=list(st.recent_bars),
                features=st.last_feat,
                structure=st.last_struct,
                regime=st.last_regime,
            )
        equity, cash = self._account_equity_cash()
        account = AccountSnapshot(
            equity=equity,
            cash=cash,
            open_positions={
                s: p.qty_open for s, p in self._positions.items() if p.qty_open > 0
            },
        )
        return BrooksContext(
            symbol=symbol,
            primary=primary,
            htf=htf_map,
            account=account,
            now_ns=feat.timestamp_ns,
        )

    # ------------------------------------------------------------------ entries

    def _submit_entry(self, symbol: str, decision: Decision, risk_pct: float) -> None:
        qty = decision.quantity
        self._log(
            f"ENTRY {decision.side} {symbol} qty={qty} entry={decision.entry_px:.4f} "
            f"stop={decision.stop_px:.4f} p={decision.probability:.2f} E={decision.expected_r:.2f}",
            level="INFO",
        )
        order_type = "buy" if decision.side == "long" else "sell_short"
        order = Order(
            symbol=symbol,
            type=order_type,
            quantity=qty,
            stop_price=decision.entry_px,
            execution_type="NEXT_OPEN",
        )
        oid = self.engine.submit_order(order) if self.engine else None
        self._pending_decisions[symbol] = decision
        if oid is not None:
            self._entry_order_ids[symbol] = oid
        # Stash the intended risk % on the decision for the portfolio guard later.
        decision.signals[0].meta["risk_pct"] = risk_pct  # type: ignore[index]

    def _fill_pending_entry(self, symbol: str, bar: Bar) -> None:
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
            qty_initial=decision.quantity,
            qty_open=decision.quantity,
            one_r=abs(decision.entry_px - decision.stop_px),
            risk_pct=float(decision.signals[0].meta.get("risk_pct", 0.0)) if decision.signals else 0.0,
            entry_bar_idx=decision.signals[0].signal_bar_idx if decision.signals else -1,
            entry_timestamp_ns=int(bar.timestamp.timestamp() * 1_000_000_000),
        )
        # Reconcile qty against broker fill if available.
        if self.engine and hasattr(self.engine.broker, "positions"):
            filled = abs(self.engine.broker.positions.get(symbol, 0.0))
            if filled > 0:
                pos.qty_initial = filled
                pos.qty_open = filled
        self._positions[symbol] = pos
        del self._pending_decisions[symbol]
        self._entry_order_ids.pop(symbol, None)

    # ------------------------------------------------------------------ management

    def _manage_position(
        self,
        symbol: str,
        bar: Bar,
        structure: MarketStructure,
        feat: ExtendedBarFeatures,
    ) -> None:
        pos = self._positions[symbol]

        # 1) hard stop hit
        if self._stop_hit(pos, bar.high, bar.low):
            self._close_full(pos, reason=f"stop hit @ {pos.stop_px:.4f}")
            return

        # 2) time-stop
        bars_since_entry = feat.bar_idx - pos.entry_bar_idx if pos.entry_bar_idx >= 0 else 0
        if self._time_stop.should_exit(pos, bars_since_entry, pos.max_unrealized_r):
            self._close_full(pos, reason="time stop")
            return

        # 3) ladder advance
        confirmed = (
            structure.confirmed_swing_lows
            if pos.side == "long"
            else structure.confirmed_swing_highs
        )
        new_stop = self._stop_ladder.update(
            pos,
            bar_high=bar.high,
            bar_low=bar.low,
            bar_close=bar.close,
            confirmed_swings=confirmed,
        )
        if new_stop is not None:
            self._log(
                f"STOP-ADJ {symbol} → {new_stop:.4f} stage={pos.ladder_stage}",
                level="INFO",
            )

    def _stop_hit(self, pos: PositionState, bar_high: float, bar_low: float) -> bool:
        if pos.qty_open <= 0:
            return False
        if pos.side == "long":
            return bar_low <= pos.stop_px
        return bar_high >= pos.stop_px

    def _close_full(self, pos: PositionState, *, reason: str) -> None:
        if self.engine is None or pos.qty_open <= 0:
            self._positions.pop(pos.symbol, None)
            return
        order_type = "sell" if pos.side == "long" else "buy_to_cover"
        method = getattr(self, order_type)
        method(pos.symbol, pos.qty_open, execution_type="IMMEDIATE_CLOSE")
        self._log(f"EXIT {pos.side} {pos.symbol} qty={pos.qty_open} reason={reason}", level="INFO")
        pos.qty_open = 0.0
        self._positions.pop(pos.symbol, None)

    # ------------------------------------------------------------------ account

    def _account_equity_cash(self) -> tuple[float, float]:
        if self.engine is None:
            return 100_000.0, 100_000.0
        acct = self.engine.broker.get_account_info()
        equity = float(acct.get("total_equity", 0.0))
        cash = float(acct.get("cash", equity))
        return equity, cash


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_te(te_params: Dict[str, Any]) -> TraderEquation:
    """Construct the TraderEquation, honoring ``hit_rate_path`` if given."""
    params = dict(te_params or {})
    path = params.pop("hit_rate_path", None)
    if path is not None:
        ht = HitRateTable.load(path)
    else:
        ht = HitRateTable.load()
    return TraderEquation(ht, **params)


def _build_sizer(sizer_params: Dict[str, Any]) -> Sizer:
    params = dict(sizer_params or {})
    kind = params.pop("kind", "kelly")
    if kind == "kelly":
        return KellySizer(**params)
    if kind == "fixed":
        return FixedPercentSizer(**params)
    raise ValueError(f"Unknown sizer kind: {kind!r}")


def _bar_to_brooks(bar: Bar, ts_ns: int) -> BrooksBar:
    return BrooksBar(
        timestamp_ns=ts_ns,
        open=float(bar.open),
        high=float(bar.high),
        low=float(bar.low),
        close=float(bar.close),
        volume=float(bar.volume),
    )


def _risk_pct_from_qty(qty: float, decision: Decision, equity: float) -> float:
    """Reverse the sizing math to get the effective risk fraction of equity."""
    if equity <= 0:
        return 0.0
    per_unit = abs(decision.entry_px - decision.stop_px)
    if per_unit <= 0:
        return 0.0
    return (qty * per_unit) / equity


def _select_decision(
    decisions: List[Decision], combined: AggregatedDecision
) -> Optional[Decision]:
    """Pick the decision that matches the aggregator's resolved entry/stop.

    The aggregator combines signals into a single (side, entry, stop)
    tuple; the EV gate operates per-signal, so multiple decisions may
    come back. We prefer the decision whose side matches and whose
    (entry, stop) equals the aggregator's, and fall back to the
    highest-EV decision on the correct side.
    """
    same_side = [d for d in decisions if d.side == combined.side]
    if not same_side:
        return None
    for d in same_side:
        if d.entry_px == combined.entry_px and d.stop_px == combined.stop_px:
            d = d.model_copy(update={"entry_px": combined.entry_px, "stop_px": combined.stop_px})
            return d
    best = max(same_side, key=lambda d: d.expected_r)
    return best.model_copy(update={"entry_px": combined.entry_px, "stop_px": combined.stop_px})
