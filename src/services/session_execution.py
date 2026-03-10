from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Optional
import time

from loguru import logger

from session_db import SessionDB
from src.config.settings import get_broker_config, get_data_stream_config
from src.core.backtest_broker import BacktestBroker
from src.core.data_stream import DBDataStream, RealtimeDataStream
from src.core.engine import TradingEngine
from src.core.live_broker import LiveBroker
from src.core.risk_manager import RiskManager
from src.market_data.db import DB
from src.strategies.registry import StrategyRegistry


@dataclass
class SessionExecutionConfig:
    session_id: str
    strategy: str
    symbol: str
    start_date: str
    end_date: Optional[str] = None
    mode: str = "backtest"
    params: Dict[str, Any] = field(default_factory=dict)
    initial_cash: Optional[float] = None
    commission: Optional[float] = None
    slippage: Optional[float] = None
    enable_risk_management: bool = False
    chunk_size_months: Optional[int] = None
    simulation_delay_seconds: float = 1.0


@dataclass
class SessionExecutionResult:
    session_id: str
    status: str
    progress: float
    final_equity: float
    total_trades: int
    positions: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SessionExecutionHooks:
    on_session_started: Optional[Callable[[SessionExecutionConfig], None]] = None
    on_engine_created: Optional[Callable[[TradingEngine, Any], None]] = None
    on_status_change: Optional[
        Callable[[str, float, Optional[str]], None]
    ] = None
    on_progress: Optional[Callable[[float, str], None]] = None
    on_equity_point: Optional[Callable[[Dict[str, Any]], None]] = None
    on_trade: Optional[Callable[[Dict[str, Any]], None]] = None
    on_completed: Optional[Callable[[SessionExecutionResult], None]] = None
    on_failed: Optional[Callable[[str], None]] = None


def _emit(callback: Optional[Callable], *args) -> None:
    if callback is None:
        return
    try:
        callback(*args)
    except Exception as exc:
        logger.exception(f"Session execution callback failed: {exc}")


def _update_session_record(
    session_db: Optional[SessionDB],
    session_id: str,
    status: str,
    progress: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    if session_db is None:
        return

    fields: Dict[str, Any] = {"status": status}
    if progress is not None:
        fields["progress"] = progress
    if error is not None or status in {"running", "completed"}:
        fields["error"] = error

    session_db.update_session(session_id, **fields)


def _persist_runtime_snapshot(
    session_id: str,
    broker: Any,
    session_db: Optional[SessionDB],
    hooks: SessionExecutionHooks,
) -> Dict[str, Any]:
    info = broker.get_account_info()
    clear_buffers = isinstance(broker, BacktestBroker)

    new_equity_points = list(info.get("equity_history", []))
    if new_equity_points and session_db is not None:
        session_db.add_equity_points(session_id, new_equity_points)
    for point in new_equity_points:
        _emit(hooks.on_equity_point, point)
    if clear_buffers and new_equity_points:
        broker.equity_history.clear()

    new_trades = list(info.get("trades", []))
    if new_trades and session_db is not None:
        session_db.add_trades(session_id, new_trades)
    for trade in new_trades:
        _emit(hooks.on_trade, trade)
    if clear_buffers and new_trades:
        broker.trades.clear()

    latest_positions = {}
    if new_equity_points:
        latest_positions = new_equity_points[-1].get("positions", {})
    elif info.get("detailed_positions"):
        latest_positions = info["detailed_positions"]

    return {
        "info": info,
        "positions": latest_positions,
        "equity_points": new_equity_points,
        "trades": new_trades,
    }


def execute_session(
    config: SessionExecutionConfig,
    *,
    session_db: Optional[SessionDB] = None,
    hooks: Optional[SessionExecutionHooks] = None,
) -> SessionExecutionResult:
    hooks = hooks or SessionExecutionHooks()

    broker_config = get_broker_config()
    data_stream_config = get_data_stream_config()

    db_client = DB()
    _update_session_record(session_db, config.session_id, "running", progress=0.0, error=None)
    _emit(hooks.on_status_change, "running", 0.0, None)
    _emit(hooks.on_session_started, config)

    symbols = [config.symbol]
    if config.mode == "live":
        stream = RealtimeDataStream(
            symbols,
            interval_seconds=data_stream_config.realtime.interval_seconds,
            data_source=data_stream_config.realtime.data_source,
            enable_trading_hours_check=True,
        )
        total_bars = 0
    else:
        stream = DBDataStream(
            db_client,
            symbols,
            config.start_date,
            config.end_date,
            chunk_size_months=config.chunk_size_months
            if config.chunk_size_months is not None
            else data_stream_config.chunk_size_months,
        )
        total_bars = getattr(stream, "total_bars", 1)

    if config.mode == "live":
        broker = LiveBroker(server_url=broker_config.live.server_url)
    else:
        broker = BacktestBroker(
            db_client=db_client,
            initial_cash=config.initial_cash
            if config.initial_cash is not None
            else broker_config.backtest.initial_cash,
            commission=config.commission
            if config.commission is not None
            else broker_config.backtest.commission,
            slippage=config.slippage
            if config.slippage is not None
            else broker_config.backtest.slippage,
        )

    risk_manager = None
    if config.enable_risk_management:
        risk_manager = RiskManager(
            initial_capital=config.initial_cash
            if config.initial_cash is not None
            else broker_config.backtest.initial_cash
        )
        if isinstance(broker, BacktestBroker):
            broker.risk_manager = risk_manager

    strategy = StrategyRegistry.create_strategy(
        config.strategy,
        db_client,
        session_id=config.session_id,
        **(config.params or {}),
    )
    if strategy is None:
        error_message = f"Unknown strategy: {config.strategy}"
        _update_session_record(session_db, config.session_id, "failed", error=error_message)
        _emit(hooks.on_status_change, "failed", 0.0, error_message)
        _emit(hooks.on_failed, error_message)
        raise ValueError(error_message)

    last_persist_at = time.time()

    def on_step(_bars):
        nonlocal last_persist_at

        if config.mode in {"backtest", "simulation"} and total_bars > 0:
            progress = (getattr(stream, "idx", 0) / total_bars) * 100
        else:
            progress = 50.0

        current_time = time.time()
        should_persist = current_time - last_persist_at >= 2.0 or progress >= 100
        if should_persist:
            _persist_runtime_snapshot(config.session_id, broker, session_db, hooks)
            _update_session_record(
                session_db,
                config.session_id,
                "running",
                progress=progress,
            )
            _emit(hooks.on_progress, progress, "running")
            _emit(hooks.on_status_change, "running", progress, None)
            last_persist_at = current_time

        if config.mode == "simulation" and config.simulation_delay_seconds > 0:
            time.sleep(config.simulation_delay_seconds)

    engine = TradingEngine(
        strategy=strategy,
        broker=broker,
        data_stream=stream,
        on_step=on_step,
        risk_manager=risk_manager,
    )
    _emit(hooks.on_engine_created, engine, broker)

    try:
        engine.run()
        snapshot = _persist_runtime_snapshot(config.session_id, broker, session_db, hooks)
        final_info = snapshot["info"]

        if session_db is not None:
            stored_equity = session_db.get_equity_history(config.session_id)
            stored_trades = session_db.get_trades(config.session_id)
            final_equity = (
                stored_equity[-1]["total_equity"]
                if stored_equity
                else final_info.get("total_equity", 0.0)
            )
            total_trades = len(stored_trades)
        else:
            final_equity = float(final_info.get("total_equity", 0.0))
            total_trades = len(final_info.get("trades", []))

        positions = snapshot["positions"] or final_info.get("detailed_positions", {})
        result = SessionExecutionResult(
            session_id=config.session_id,
            status="completed",
            progress=100.0,
            final_equity=final_equity,
            total_trades=total_trades,
            positions=positions,
        )
        _update_session_record(session_db, config.session_id, "completed", progress=100.0, error=None)
        _emit(hooks.on_progress, 100.0, "completed")
        _emit(hooks.on_status_change, "completed", 100.0, None)
        _emit(hooks.on_completed, result)
        return result
    except Exception as exc:
        error_message = str(exc)
        _update_session_record(session_db, config.session_id, "failed", error=error_message)
        _emit(hooks.on_status_change, "failed", 0.0, error_message)
        _emit(hooks.on_failed, error_message)
        raise
