"""
Enhanced broker interface with risk management integration.
Provides abstract base class and mock implementation for testing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    ERROR = "error"


@dataclass
class Order:
    """Order data structure"""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    stop_price: Optional[float] = None
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    created_at: datetime = None
    updated_at: datetime = None
    commission: float = 0.0

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class Position:
    """Position data structure"""
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_cost': self.avg_cost,
            'current_price': self.current_price,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct
        }


@dataclass
class AccountInfo:
    """Account information data structure"""
    cash: float
    total_equity: float
    positions: Dict[str, Position]
    buying_power: float
    margin_used: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cash': self.cash,
            'total_equity': self.total_equity,
            'positions': {k: v.to_dict() for k, v in self.positions.items()},
            'buying_power': self.buying_power,
            'margin_used': self.margin_used
        }


class BrokerInterface(ABC):
    """Abstract broker interface"""

    @abstractmethod
    def connect(self) -> bool:
        """Connect to broker"""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from broker"""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected"""
        pass

    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """Submit an order, returns order_id"""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status"""
        pass

    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """Get account information"""
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol"""
        pass

    @abstractmethod
    def sync_state(self) -> bool:
        """Sync local state with broker"""
        pass


class MockBroker(BrokerInterface):
    """Mock broker for testing"""

    def __init__(self, initial_cash: float = 1000000.0, commission_rate: float = 0.0001):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_rate = commission_rate

        self.connected = False
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.order_counter = 0

        # Market data simulation
        self.market_prices: Dict[str, float] = {}

        logger.info(f"MockBroker initialized: cash={initial_cash}, commission={commission_rate}")

    def connect(self) -> bool:
        """Connect to mock broker"""
        self.connected = True
        logger.info("MockBroker connected")
        return True

    def disconnect(self) -> bool:
        """Disconnect from mock broker"""
        self.connected = False
        logger.info("MockBroker disconnected")
        return True

    def is_connected(self) -> bool:
        """Check if connected"""
        return self.connected

    def set_market_price(self, symbol: str, price: float):
        """Set market price for simulation"""
        self.market_prices[symbol] = price

    def submit_order(self, order: Order) -> str:
        """Submit an order"""
        if not self.connected:
            raise RuntimeError("Broker not connected")

        # Generate order ID
        self.order_counter += 1
        order.order_id = f"ORD{self.order_counter:06d}"
        order.status = OrderStatus.SUBMITTED
        order.updated_at = datetime.now()

        # Simulate order execution
        self._execute_order(order)

        # Store order
        self.orders[order.order_id] = order

        logger.info(f"Order submitted: {order.order_id} {order.side.value} {order.quantity} {order.symbol}")

        return order.order_id

    def _execute_order(self, order: Order):
        """Simulate order execution"""
        # Get execution price
        if order.order_type == OrderType.MARKET:
            exec_price = self.market_prices.get(order.symbol, order.price or 0)
        else:
            exec_price = order.price or self.market_prices.get(order.symbol, 0)

        if exec_price <= 0:
            order.status = OrderStatus.REJECTED
            logger.warning(f"Order rejected: no price available for {order.symbol}")
            return

        # Calculate commission
        order.commission = order.quantity * exec_price * self.commission_rate

        # Check if we have enough cash for buy orders
        if order.side == OrderSide.BUY:
            total_cost = order.quantity * exec_price + order.commission
            if total_cost > self.cash:
                order.status = OrderStatus.REJECTED
                logger.warning(f"Order rejected: insufficient cash (need ${total_cost:.2f}, have ${self.cash:.2f})")
                return

            # Deduct cash
            self.cash -= total_cost

            # Update position
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                total_qty = pos.quantity + order.quantity
                pos.avg_cost = (pos.avg_cost * pos.quantity + exec_price * order.quantity) / total_qty
                pos.quantity = total_qty
            else:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    avg_cost=exec_price,
                    current_price=exec_price,
                    market_value=order.quantity * exec_price,
                    unrealized_pnl=0.0,
                    unrealized_pnl_pct=0.0
                )

        elif order.side == OrderSide.SELL:
            # Check if we have the position
            if order.symbol not in self.positions:
                order.status = OrderStatus.REJECTED
                logger.warning(f"Order rejected: no position in {order.symbol}")
                return

            pos = self.positions[order.symbol]
            if pos.quantity < order.quantity:
                order.status = OrderStatus.REJECTED
                logger.warning(f"Order rejected: insufficient quantity (need {order.quantity}, have {pos.quantity})")
                return

            # Add cash
            self.cash += order.quantity * exec_price - order.commission

            # Update position
            pos.quantity -= order.quantity
            if pos.quantity <= 0:
                del self.positions[order.symbol]

        # Mark order as filled
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = exec_price
        order.updated_at = datetime.now()

        logger.info(f"Order filled: {order.order_id} @ ${exec_price:.2f}, commission=${order.commission:.2f}")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            return False

        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now()

        logger.info(f"Order cancelled: {order_id}")
        return True

    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status"""
        return self.orders.get(order_id)

    def get_account_info(self) -> AccountInfo:
        """Get account information"""
        # Update position market values
        total_position_value = 0.0
        for symbol, pos in self.positions.items():
            current_price = self.market_prices.get(symbol, pos.avg_cost)
            pos.current_price = current_price
            pos.market_value = pos.quantity * current_price
            pos.unrealized_pnl = (current_price - pos.avg_cost) * pos.quantity
            pos.unrealized_pnl_pct = (current_price - pos.avg_cost) / pos.avg_cost if pos.avg_cost > 0 else 0
            total_position_value += pos.market_value

        total_equity = self.cash + total_position_value

        return AccountInfo(
            cash=self.cash,
            total_equity=total_equity,
            positions=self.positions.copy(),
            buying_power=self.cash  # Simplified: no margin
        )

    def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        return self.positions.copy()

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol"""
        return self.positions.get(symbol)

    def sync_state(self) -> bool:
        """Sync local state with broker"""
        # Mock broker is always in sync
        logger.info("MockBroker state synced")
        return True
