"""Per-bar pipeline shared by live + replay tasks.

``BrooksCore`` owns the strategy + broker step orchestration, builds the
:class:`BarEvent`-shaped dict from captured per-bar state, and hands it to
the configured sink. Pacing (``next_bar`` cadence) lives in the calling
task — the core just processes whatever bars arrive.

What stays consistent across modes:
    * broker.step → strategy.on_bar → process_same_bar_orders order.
    * BarEvent dict shape (the same loader rebuilds it for both modes).
    * Equity / trade persistence into ``session_db``.

What differs across modes (passed in via constructor):
    * ``sink``                  — live = persist + WS broadcast; replay = persist only.
    * ``clock``                 — live = WallClock; replay = BarClock advanced per bar.
    * ``equity_throttle``       — live = wall-clock min gap; replay = ``None`` (every bar).
    * ``emit``                  — live = async event-bus dispatcher; replay = ``None``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from session_db import SessionDB
from src.api.events import (
    emit_equity_update,
    emit_error,
    emit_strategy_step,
    emit_trade_executed,
)
from src.brooks.runtime.clock import Clock
from src.brooks.runtime.event_sink import BarEventSink
from src.brooks.runtime.regime_capture import RegimeCapturingClassifier
from src.brooks.runtime.throttle import Throttle
from src.brooks.runtime.views import (
    decision_to_dict,
    decision_to_full_dict,
    features_to_view_dict,
    regime_to_view_dict,
    structure_to_view_dict,
)
from src.brooks.strategy import BrooksStrategy
from src.core.base import Bar, Broker


class BrooksCore:
    """Drive one bar through strategy + broker + telemetry."""

    def __init__(
        self,
        *,
        strategy: BrooksStrategy,
        broker: Broker,
        sink: BarEventSink,
        session_id: str,
        session_db: SessionDB,
        analyst_name: str,
        clock: Clock,
        equity_throttle: Optional[Throttle] = None,
        equity_emit_every_n_bars: int = 1,
        equity_points_window: int = 2000,
        emit: Optional[Callable[[Any], None]] = None,
        recent_signals_window: int = 50,
    ):
        self.strategy = strategy
        self.broker = broker
        self.sink = sink
        self.session_id = session_id
        self.session_db = session_db
        self.analyst_name = analyst_name
        self.clock = clock
        self.equity_throttle = equity_throttle
        self.equity_emit_every_n_bars = max(1, int(equity_emit_every_n_bars))
        self._equity_points_window = max(100, int(equity_points_window))
        self._emit = emit
        self._recent_signals_window = max(1, int(recent_signals_window))

        self._step_index = 0
        self._tracked_trades: List[Dict[str, Any]] = []
        self._last_decision: Dict[str, Any] = {}
        self._last_signals: List[Dict[str, Any]] = []
        self._last_regime: Dict[str, Any] = {}
        self._equity_points: List[Dict[str, Any]] = []
        # Per-symbol identity (Python ``id``) of the last decision we
        # already emitted on a BarEvent. Pending stop-entries can linger
        # for many bars before the price triggers them — without this
        # guard the chart shows the same decision marker on every one of
        # those bars, drowning real signals in repetition.
        self._emitted_decision_id: Dict[str, int] = {}

        # Wrap any classifiers that already exist (rare — strategy creates
        # them lazily on first bar) and patch the strategy's setup point so
        # all future symbols come pre-wrapped.
        self._wrap_existing_classifiers()
        self._patch_strategy_ensure_symbol_state()

        # Hook the broker so we can persist decision/order pairing.
        self._install_broker_order_hook()

        # Strategy._submit_entry calls ``self.engine.submit_order(...)`` —
        # without a connected engine all entries silently no-op. The
        # legacy live engine got this wiring for free via ``TradingEngine``;
        # BrooksCore replaces that wrapper, so we provide the same shim.
        strategy.set_engine(_StrategyEngineShim(broker))

    # ------------------------------------------------------------------ public

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def tracked_trades(self) -> List[Dict[str, Any]]:
        return self._tracked_trades

    @property
    def equity_points(self) -> List[Dict[str, Any]]:
        return self._equity_points

    @property
    def last_decision(self) -> Dict[str, Any]:
        return dict(self._last_decision)

    @property
    def last_signals(self) -> List[Dict[str, Any]]:
        return list(self._last_signals)

    @property
    def last_regime(self) -> Dict[str, Any]:
        return dict(self._last_regime)

    def process_bar(self, bars: Dict[str, Bar]) -> Dict[str, Any]:
        """Drive one bar through strategy + broker + telemetry.

        The clock is the caller's responsibility — for replay, advance
        :class:`~src.brooks.runtime.clock.BarClock` to the bar's timestamp
        before calling.
        """
        # 1. Broker step — fills pending NEXT_OPEN orders from the prior bar.
        self.broker.step(bars)

        # 2. Strategy.on_bar. Strategy creates classifiers via the patched
        #    _ensure_symbol_state, so any new classifiers are already wrapped.
        if hasattr(self.strategy, "_update_current_date"):
            self.strategy._update_current_date(bars)
        try:
            self.strategy.on_bar(bars)
        except Exception as exc:
            logger.exception("strategy.on_bar raised: {}", exc)
            if self._emit is not None:
                self._emit(emit_error(self.session_id, str(exc)))

        # 3. Same-bar order processing (IMMEDIATE_*).
        self.broker.process_same_bar_orders(bars, "IMMEDIATE_OPEN")
        self.broker.process_same_bar_orders(bars, "IMMEDIATE_CLOSE")

        # 4. Build the BarEvent and hand it to the sink (persistence + optional WS).
        bar_event = self._build_bar_event(bars)
        self.sink.handle(bar_event)

        # 5. Legacy strategy_step + equity broadcast (live-only when emit is set).
        self._broadcast_strategy_step(bars, bar_event)
        self._maybe_emit_equity(bars)

        # 6. Trade outcome persistence (always; replay benefits from realized R).
        self._persist_decision_outcomes(bars)

        self._step_index += 1
        return bar_event

    # ------------------------------------------------------------------ wrap helpers

    def _wrap_existing_classifiers(self) -> None:
        regimes = getattr(self.strategy, "_regimes", None) or {}
        for sym, cls in list(regimes.items()):
            if not isinstance(cls, RegimeCapturingClassifier):
                regimes[sym] = RegimeCapturingClassifier(cls)
        htf_state = getattr(self.strategy, "_htf_state", None) or {}
        for sym, htf in htf_state.items():
            for tf, st in htf.items():
                if not isinstance(st.regime, RegimeCapturingClassifier):
                    st.regime = RegimeCapturingClassifier(st.regime)

    def _patch_strategy_ensure_symbol_state(self) -> None:
        """Wrap classifiers as the strategy creates them.

        ``BrooksStrategy._ensure_symbol_state`` constructs a fresh
        ``BrooksRegimeClassifier`` per symbol and per HTF the first time it
        sees them. We chain through the original then immediately wrap the
        new classifier so the very first ``classify`` call goes through
        :class:`RegimeCapturingClassifier`, populating ``last_snapshot``
        before the bar event is built.
        """
        original = self.strategy._ensure_symbol_state
        strategy = self.strategy

        def patched(symbol: str) -> None:
            is_new = symbol not in getattr(strategy, "_extractors", {})
            original(symbol)
            if not is_new:
                return
            cls = (getattr(strategy, "_regimes", None) or {}).get(symbol)
            if cls is not None and not isinstance(cls, RegimeCapturingClassifier):
                strategy._regimes[symbol] = RegimeCapturingClassifier(cls)
            htf_for_symbol = (getattr(strategy, "_htf_state", None) or {}).get(symbol, {})
            for tf, st in htf_for_symbol.items():
                if not isinstance(st.regime, RegimeCapturingClassifier):
                    st.regime = RegimeCapturingClassifier(st.regime)

        strategy._ensure_symbol_state = patched  # type: ignore[assignment]

    def _install_broker_order_hook(self) -> None:
        broker = self.broker
        original_on_order_submitted = getattr(broker, "on_order_submitted", None)

        def _on_order(order: Any) -> None:
            if original_on_order_submitted is not None:
                original_on_order_submitted(order)
            self._handle_order_submitted(order)

        broker.on_order_submitted = _on_order  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ bar event

    def _build_bar_event(self, bars: Dict[str, Bar]) -> Dict[str, Any]:
        symbol, bar = next(iter(bars.items()))
        ts_ns = int(bar.timestamp.timestamp() * 1_000_000_000)

        feat_obj = None
        struct_obj = None
        history = (getattr(self.strategy, "_feature_history", None) or {}).get(symbol) or []
        if history:
            feat_obj = history[-1]
        struct_state = (getattr(self.strategy, "_structures", None) or {}).get(symbol)
        if struct_state is not None:
            struct_obj = getattr(struct_state, "state", None)

        bar_idx = int(getattr(feat_obj, "bar_idx", 0)) if feat_obj is not None else 0

        # Regime payload from the capturing wrapper.
        regime_payload: Optional[Dict[str, Any]] = None
        classifier = (getattr(self.strategy, "_regimes", None) or {}).get(symbol)
        snapshot = getattr(classifier, "last_snapshot", None) if classifier else None
        if snapshot is not None:
            try:
                regime_payload = snapshot.to_dict()
            except Exception:
                regime_payload = None
        self._last_regime = regime_payload or {}

        pending = getattr(self.strategy, "_pending_decisions", {}) or {}
        decision = pending.get(symbol)
        decision_dict: Optional[Dict[str, Any]] = None
        signals_list: List[Dict[str, Any]] = []
        # Only render the decision/signals on the bar where they were
        # FIRST submitted. A stop-entry can sit pending for several bars
        # before the price triggers it; carrying the decision payload
        # forward turns one signal into a row of duplicate markers on
        # the chart. Once it fills (and ``_pending_decisions`` clears)
        # or expires, the next genuinely-new decision will pass identity
        # check and emit again.
        decision_id = id(decision) if decision is not None else None
        if decision is not None and self._emitted_decision_id.get(symbol) != decision_id:
            decision_dict = decision_to_full_dict(decision)
            signals_list = [s.model_dump() for s in (decision.signals or [])]
            self._last_decision = decision_to_dict(decision)
            self._last_signals.append(self._last_decision)
            if len(self._last_signals) > self._recent_signals_window:
                del self._last_signals[: -self._recent_signals_window]
            self._emitted_decision_id[symbol] = decision_id
        elif decision is None and symbol in self._emitted_decision_id:
            del self._emitted_decision_id[symbol]

        htf_payload: Dict[str, Dict[str, Any]] = {}
        htf_bars: Dict[str, Dict[str, Any]] = {}
        for tf, st in ((getattr(self.strategy, "_htf_state", None) or {}).get(symbol) or {}).items():
            last_regime = getattr(st, "last_regime", None)
            last_struct = getattr(st, "last_struct", None)
            htf_payload[tf] = {
                "regime": getattr(getattr(last_regime, "regime", None), "value", None) if last_regime else None,
                "always_in": getattr(last_struct, "always_in", None) if last_struct else None,
                "last_swing_idx": (
                    getattr(last_struct, "confirmed_swing_highs", [-1])[-1].bar_idx
                    if last_struct and getattr(last_struct, "confirmed_swing_highs", None)
                    else None
                ),
            }
            recent = getattr(st, "recent_bars", []) or []
            if recent:
                latest = recent[-1]
                htf_bars[tf] = {
                    "timestamp_ns": int(latest.timestamp_ns),
                    "open": float(latest.open),
                    "high": float(latest.high),
                    "low": float(latest.low),
                    "close": float(latest.close),
                    "volume": float(latest.volume),
                }

        return {
            "bar_idx": bar_idx,
            "timestamp_ns": ts_ns,
            "bar": {
                "timestamp_ns": ts_ns,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            },
            "features": features_to_view_dict(feat_obj) if feat_obj is not None else None,
            "structure": structure_to_view_dict(struct_obj) if struct_obj is not None else None,
            "regime": regime_to_view_dict(regime_payload),
            "signals": signals_list,
            "decision": decision_dict,
            "htf": htf_payload,
            "htf_bars": htf_bars,
            "symbol": symbol,
            "analyst": self.analyst_name,
        }

    # ------------------------------------------------------------------ broadcasts

    def _broadcast_strategy_step(self, bars: Dict[str, Bar], bar_event: Dict[str, Any]) -> None:
        """Emit the legacy ``strategy_step`` payload on the per-session channel.

        Studio reads BarEvents from the dedicated channel, but the legacy
        ``/ws/{session_id}`` is still subscribed by older panels for status.
        Replay never emits — the panel pulls from session_db post-hoc.
        """
        if self._emit is None:
            return
        symbol, bar = next(iter(bars.items()))
        positions: Dict[str, Dict[str, Any]] = {}
        for sym, pos in (getattr(self.strategy, "_positions", {}) or {}).items():
            positions[sym] = {
                "side": pos.side,
                "entry_px": pos.entry_px,
                "stop_px": pos.stop_px,
                "qty_open": pos.qty_open,
                "one_r": pos.one_r,
                "ladder_stage": getattr(pos, "ladder_stage", 0),
            }
        legacy_payload = {
            "event": "bar_closed",
            "symbol": symbol,
            "timestamp": bar.timestamp.isoformat(),
            "ohlcv": {
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            },
            "regime": self._last_regime or None,
            "decision": self._last_decision or None,
            "analyst": self.analyst_name,
            "positions": positions,
        }
        self._emit(emit_strategy_step(self.session_id, legacy_payload))

    def _maybe_emit_equity(self, bars: Dict[str, Bar]) -> None:
        # Bar-count gate first (cheap) — replay sets every_n>1 to avoid
        # 1 equity row per bar on multi-day 1m windows. Live keeps every_n=1
        # and uses the wall-clock throttle below.
        if self.equity_emit_every_n_bars > 1 and (self._step_index % self.equity_emit_every_n_bars) != 0:
            return
        if self.equity_throttle is not None:
            now = self.clock.now_seconds()
            if self.equity_throttle.should_skip("equity", now):
                return
            self.equity_throttle.mark_called("equity", now)

        acct = self.broker.get_account_info()
        ts = next(iter(bars.values())).timestamp.isoformat()
        equity_pt = {
            "timestamp": ts,
            "total_equity": float(acct.get("total_equity", 0.0)),
            "cash": float(acct.get("cash", 0.0)),
            "positions": acct.get("detailed_positions", {}),
        }
        self._equity_points.append(equity_pt)
        # Cap the in-memory rolling buffer — long-running live sessions
        # otherwise grow this list unbounded (~17k entries/day at 5s emit).
        # Persistent history lives in ``equity_history`` table, not here.
        if len(self._equity_points) > self._equity_points_window:
            del self._equity_points[: -self._equity_points_window]
        try:
            self.session_db.add_equity_point(
                self.session_id,
                ts,
                equity_pt["total_equity"],
                cash=equity_pt["cash"],
            )
        except Exception as e:
            logger.debug("add_equity_point failed: {}", e)
        if self._emit is not None:
            self._emit(emit_equity_update(self.session_id, equity_pt))

    # ------------------------------------------------------------------ trades

    def _handle_order_submitted(self, order: Any) -> None:
        pending = getattr(self.strategy, "_pending_decisions", {}) or {}
        decision = pending.get(order.symbol)
        if decision is None:
            return
        payload = decision_to_dict(decision)
        payload["order_id"] = order.id
        payload["order_type"] = order.type
        try:
            self.session_db.add_session_log(
                session_id=self.session_id,
                timestamp=datetime.now().isoformat(),
                level="INFO",
                source="brooks_decision",
                message=(
                    f"{payload.get('side')} {order.symbol} @ {payload.get('entry_px')} "
                    f"stop={payload.get('stop_px')} E={payload.get('expected_r')}"
                ),
                extra=payload,
            )
        except Exception as e:
            logger.debug("brooks_decision persist failed: {}", e)

    def _persist_decision_outcomes(self, bars: Dict[str, Bar]) -> None:
        broker_trades = list(getattr(self.broker, "trades", []) or [])
        if len(broker_trades) <= len(self._tracked_trades):
            return
        for trade in broker_trades[len(self._tracked_trades) :]:
            self._tracked_trades.append(trade)
            if self._emit is not None:
                self._emit(emit_trade_executed(self.session_id, trade))
            try:
                self.session_db.add_trade(self.session_id, trade)
            except Exception as e:
                logger.debug("add_trade failed: {}", e)
            if trade.get("type") in ("sell", "buy_to_cover"):
                self._record_realized_outcome(trade)

    def _record_realized_outcome(self, trade: Dict[str, Any]) -> None:
        decision = self._last_decision
        if not decision:
            return
        try:
            entry_px = float(decision.get("entry_px", 0))
            stop_px = float(decision.get("stop_px", 0))
        except (TypeError, ValueError):
            return
        one_r = abs(entry_px - stop_px)
        if one_r == 0:
            return
        try:
            fill_px = float(trade.get("price", 0))
        except (TypeError, ValueError):
            return
        if decision.get("side") == "long":
            realized_r = (fill_px - entry_px) / one_r
        else:
            realized_r = (entry_px - fill_px) / one_r
        try:
            self.session_db.add_session_log(
                session_id=self.session_id,
                timestamp=str(trade.get("timestamp") or datetime.now().isoformat()),
                level="INFO",
                source="brooks_decision_outcome",
                message=f"R={realized_r:.3f} pattern={decision.get('pattern')}",
                extra={
                    "pattern": decision.get("pattern"),
                    "regime": decision.get("regime"),
                    "htf_aligned": bool(decision.get("htf_aligned", False)),
                    "side": decision.get("side"),
                    "realized_r": float(realized_r),
                    "hit_1r": realized_r >= 1.0,
                    "hit_2r": realized_r >= 2.0,
                },
            )
        except Exception as e:
            logger.debug("brooks_decision_outcome persist failed: {}", e)


class _StrategyEngineShim:
    """Minimal engine surface ``BrooksStrategy._submit_entry`` requires.

    BrooksStrategy was built to live inside a :class:`TradingEngine`,
    which exposes ``submit_order`` and a ``broker`` attribute. The
    runtime drives the strategy directly so we synthesise the smallest
    object that satisfies both calls — the alternative (instantiating
    a full TradingEngine) would re-introduce the bar loop we replaced.
    """

    def __init__(self, broker: Broker):
        self.broker = broker

    def submit_order(self, order: Any) -> str:
        return self.broker.submit_order(order)


__all__ = ["BrooksCore"]
