"""
Trading Service - Decoupled trading engine service layer.
Provides a clean interface for running backtests and managing trading sessions.
"""

from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from src.core.engine import TradingEngine
from src.core.base import Strategy, Broker, DataStream, Bar
from src.core.risk_manager import RiskManager
from src.analysis.backtest_metrics import calculate_metrics, PerformanceMetrics
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class SessionStatus(Enum):
    """Trading session status"""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SessionConfig:
    """Configuration for a trading session"""

    strategy_name: str
    symbol: str
    start_date: str
    end_date: Optional[str] = None
    mode: str = "backtest"  # backtest, paper, live
    initial_capital: float = 1000000.0
    params: Dict[str, Any] = field(default_factory=dict)
    risk_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "mode": self.mode,
            "initial_capital": self.initial_capital,
            "params": self.params,
            "risk_enabled": self.risk_enabled,
        }


@dataclass
class SessionResult:
    """Result of a completed trading session"""

    session_id: str
    config: SessionConfig
    status: SessionStatus
    metrics: Optional[PerformanceMetrics] = None
    equity_history: List[Dict[str, Any]] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    positions: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "config": self.config.to_dict(),
            "status": self.status.value,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "equity_history": self.equity_history,
            "trades": self.trades,
            "positions": self.positions,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "duration_seconds": self.duration_seconds,
        }


class TradingService:
    """
    High-level trading service that manages trading sessions.
    Decouples the trading engine from API layer.
    """

    def __init__(
        self,
        strategy_factory: Callable[[str, Dict[str, Any]], Strategy],
        broker_factory: Callable[[str, float], Broker],
        data_stream_factory: Callable[[str, str, str, Optional[str]], DataStream],
    ):
        """
        Initialize trading service with factory functions.

        Args:
            strategy_factory: Function(name, params) -> Strategy
            broker_factory: Function(mode, capital) -> Broker
            data_stream_factory: Function(symbol, mode, start, end) -> DataStream
        """
        self.strategy_factory = strategy_factory
        self.broker_factory = broker_factory
        self.data_stream_factory = data_stream_factory

        self._active_sessions: Dict[str, TradingEngine] = {}
        self._results: Dict[str, SessionResult] = {}

    def create_session(self, config: SessionConfig) -> str:
        """
        Create a new trading session.

        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())

        logger.info(
            f"Creating session {session_id[:8]}",
            strategy=config.strategy_name,
            mode=config.mode,
        )

        # Initialize result
        self._results[session_id] = SessionResult(
            session_id=session_id,
            config=config,
            status=SessionStatus.PENDING,
        )

        return session_id

    def run_session(
        self,
        session_id: str,
        on_progress: Optional[Callable[[float, Dict[str, Any]], None]] = None,
    ) -> SessionResult:
        """
        Run a trading session synchronously.

        Args:
            session_id: Session ID from create_session
            on_progress: Optional callback(progress_pct, data)

        Returns:
            SessionResult with metrics and trade history
        """
        result = self._results.get(session_id)
        if not result:
            raise ValueError(f"Session {session_id} not found")

        config = result.config
        result.started_at = datetime.now()
        result.status = SessionStatus.RUNNING

        try:
            # Create components using factories
            strategy = self.strategy_factory(
                config.strategy_name, {**config.params, "session_id": session_id}
            )

            broker = self.broker_factory(config.mode, config.initial_capital)

            data_stream = self.data_stream_factory(
                config.symbol,
                config.mode,
                config.start_date,
                config.end_date,
            )

            # Risk manager
            risk_manager = None
            if config.risk_enabled:
                risk_manager = RiskManager(
                    initial_capital=config.initial_capital, enabled=True
                )

            # Progress tracking
            total_bars = getattr(data_stream, "total_bars", 0)
            current_bar = [0]

            def on_step(bars: Dict[str, Bar]):
                current_bar[0] += 1
                if on_progress and total_bars > 0:
                    progress = current_bar[0] / total_bars * 100
                    on_progress(progress, {"bars": current_bar[0]})

            # Create and run engine
            engine = TradingEngine(
                strategy=strategy,
                broker=broker,
                data_stream=data_stream,
                on_step=on_step,
                risk_manager=risk_manager,
            )

            self._active_sessions[session_id] = engine

            # Run the backtest
            engine.run()

            # Collect results
            account = broker.get_account_info()

            result.equity_history = account.get("equity_history", [])
            result.trades = account.get("trades", [])
            result.positions = account.get("positions", {})

            # Calculate metrics
            if result.equity_history:
                result.metrics = calculate_metrics(result.equity_history, result.trades)

            result.status = SessionStatus.COMPLETED
            result.completed_at = datetime.now()

            logger.info(
                f"Session {session_id[:8]} completed",
                duration=result.duration_seconds,
                trades=len(result.trades),
            )

        except Exception as e:
            result.status = SessionStatus.FAILED
            result.error = str(e)
            result.completed_at = datetime.now()

            logger.error(
                f"Session {session_id[:8]} failed",
                error=str(e),
            )
            raise

        finally:
            self._active_sessions.pop(session_id, None)

        return result

    def stop_session(self, session_id: str) -> bool:
        """Stop a running session."""
        engine = self._active_sessions.get(session_id)
        if engine:
            engine.stop()
            result = self._results.get(session_id)
            if result:
                result.status = SessionStatus.CANCELLED
            return True
        return False

    def get_session_status(self, session_id: str) -> Optional[SessionStatus]:
        """Get current session status."""
        result = self._results.get(session_id)
        return result.status if result else None

    def get_session_result(self, session_id: str) -> Optional[SessionResult]:
        """Get session result."""
        return self._results.get(session_id)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions."""
        return [
            {
                "session_id": sid,
                "status": result.status.value,
                "strategy": result.config.strategy_name,
                "mode": result.config.mode,
            }
            for sid, result in self._results.items()
        ]


def run_backtest(
    strategy_class: type,
    broker_class: type,
    data_stream_class: type,
    config: SessionConfig,
    on_progress: Optional[Callable] = None,
) -> SessionResult:
    """
    Convenience function to run a single backtest.

    Args:
        strategy_class: Strategy class
        broker_class: Broker class
        data_stream_class: DataStream class
        config: Session configuration
        on_progress: Optional progress callback

    Returns:
        SessionResult
    """

    def strategy_factory(name, params):
        # Some strategies require db_client as first argument
        import inspect
        sig = inspect.signature(strategy_class.__init__)
        
        if 'db_client' in sig.parameters:
            from db import DB
            db_client = DB()
            return strategy_class(db_client, **params)
        
        return strategy_class(**params)

    def broker_factory(mode, capital):
        return broker_class(initial_cash=capital)

    def data_stream_factory(symbol, mode, start, end):
        return data_stream_class(
            symbol=symbol,
            start_date=start,
            end_date=end,
        )

    service = TradingService(
        strategy_factory=strategy_factory,
        broker_factory=broker_factory,
        data_stream_factory=data_stream_factory,
    )

    session_id = service.create_session(config)
    return service.run_session(session_id, on_progress)
