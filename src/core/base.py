from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd

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
    def __init__(self):
        self.engine = None

    def set_engine(self, engine):
        self.engine = engine

    @abstractmethod
    def on_bar(self, bars: Dict[str, Bar]):
        """Handle a new bar of data."""
        pass

    def buy(self, symbol: str, quantity: float, price: Optional[float] = None, execution_type: str = "NEXT_OPEN"):
        return self.engine.submit_order(Order(symbol, 'buy', quantity, price, execution_type))

    def sell(self, symbol: str, quantity: float, price: Optional[float] = None, execution_type: str = "NEXT_OPEN"):
        return self.engine.submit_order(Order(symbol, 'sell', quantity, price, execution_type))
