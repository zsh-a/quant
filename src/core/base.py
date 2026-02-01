from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
from loguru import logger as loguru_logger

@dataclass
class Bar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    extra: Dict[str, Any] = field(default_factory=dict)

class DataStream(ABC):
    @abstractmethod
    def next_bar(self) -> Optional[Dict[str, Bar]]:
        """Return the next bar for all active symbols, or None if end of stream."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the stream focus to the beginning (for backtesting)."""
        pass

class Order:
    def __init__(self, symbol: str, type: str, quantity: float, price: Optional[float] = None, execution_type: str = "NEXT_OPEN"):
        self.symbol = symbol
        self.type = type  # 'buy' or 'sell'
        self.quantity = quantity
        self.price = price  # None for market order
        self.execution_type = execution_type # 'NEXT_OPEN', 'IMMEDIATE_OPEN', 'IMMEDIATE_CLOSE'
        self.status = "PENDING"
        self.filled_quantity = 0.0
        self.avg_fill_price = 0.0
        self.id = None
        self.created_at = datetime.now()

class Broker(ABC):
    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """Submit an order and return an order ID."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str):
        """Cancel an existing order."""
        pass

    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """Return current cash and positions."""
        pass

    @abstractmethod
    def step(self, bars: Dict[str, Bar]):
        """Advance the broker state based on new market data (for simulation/backtest)."""
        pass

    def process_same_bar_orders(self, bars: Dict[str, Bar], timing: str):
        """Process immediate orders (optional implementation for backtest)."""
        pass

class Strategy(ABC):
    """Base strategy class with built-in session logging support."""
    
    def __init__(self, session_id: Optional[str] = None):
        self.engine = None
        self.session_id = session_id
        self._session_log = None
        self._current_date: Optional[str] = None  # Auto-tracked current backtest date
        
        # Lazy load session logger to avoid circular imports
        if session_id:
            try:
                from src.utils.session_logger import get_session_logger
                self._session_log = get_session_logger(session_id)
            except ImportError:
                pass

    def set_engine(self, engine):
        self.engine = engine
    
    def _update_current_date(self, bars: Dict[str, Bar]):
        """Update current date from bars (called automatically by engine or strategy)."""
        if bars:
            ts = next(iter(bars.values())).timestamp
            self._current_date = str(ts.date())
    
    def _log(self, message: str, level: str = "INFO", source: str = "strategy", 
             include_date: bool = True, **kwargs):
        """Log to both loguru and session logger.
        
        Args:
            message: Log message
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            source: Log source (strategy, broker, engine)
            include_date: Auto-prepend [date] prefix (default True)
            **kwargs: Extra fields to include in session log
        """
        # Auto-prepend date if available and not already included
        if include_date and self._current_date and not message.startswith(f"[{self._current_date}]"):
            message = f"[{self._current_date}] {message}"
            kwargs.setdefault("date", self._current_date)
        
        getattr(loguru_logger, level.lower())(message)
        if self._session_log:
            self._session_log.add(level, source, message, kwargs)

    @abstractmethod
    def on_bar(self, bars: Dict[str, Bar]):
        """Handle a new bar of data."""
        pass

    @classmethod
    def get_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Return the parameter schema for this strategy.
        Format:
        {
            "param_name": {
                "type": "int" | "float" | "str" | "bool" | "list",
                "default": value,
                "description": "User friendly description",
                "min": optional_min,
                "max": optional_max,
                "options": optional_list_of_values (for dropdowns)
            }
        }
        """
        return {}

    def buy(self, symbol: str, quantity: float, price: Optional[float] = None, execution_type: str = "NEXT_OPEN"):
        return self.engine.submit_order(Order(symbol, 'buy', quantity, price, execution_type))

    def sell(self, symbol: str, quantity: float, price: Optional[float] = None, execution_type: str = "NEXT_OPEN"):
        return self.engine.submit_order(Order(symbol, 'sell', quantity, price, execution_type))
