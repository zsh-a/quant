"""
CCXT 实盘 Broker — 基于 CCXT 的多交易所实盘交易接口。

支持交易所: Binance, OKX, Bitget 等 (通过 CCXT 统一接口)。

用法:
    broker = CcxtBroker(exchange="binance", api_key="...", secret="...")
    broker.connect()
    order_id = broker.submit_order(order)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from loguru import logger

from .broker_interface import (
    AccountInfo,
    BrokerInterface,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)


class CcxtBroker(BrokerInterface):
    """基于 CCXT 的多交易所实盘 Broker。"""

    def __init__(
        self,
        exchange: str = "binance",
        api_key: str = "",
        secret: str = "",
        password: str = "",
        sandbox: bool = False,
        market_type: str = "swap",  # spot | swap | future
    ):
        self.exchange_name = exchange
        self.api_key = api_key
        self.secret = secret
        self.password = password
        self.sandbox = sandbox
        self.market_type = market_type
        self._exchange = None
        self._connected = False
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}

    def connect(self) -> bool:
        try:
            import ccxt
            exchange_class = getattr(ccxt, self.exchange_name, None)
            if exchange_class is None:
                logger.error("Unsupported exchange: {}", self.exchange_name)
                return False

            config: dict[str, Any] = {
                "apiKey": self.api_key,
                "secret": self.secret,
                "enableRateLimit": True,
            }
            if self.password:
                config["password"] = self.password
            if self.market_type in ("swap", "future"):
                config["options"] = {"defaultType": self.market_type}

            self._exchange = exchange_class(config)
            if self.sandbox:
                self._exchange.set_sandbox_mode(True)

            self._exchange.load_markets()
            self._connected = True
            logger.info("CcxtBroker connected: {} (sandbox={})", self.exchange_name, self.sandbox)
            return True
        except Exception as exc:
            logger.error("CcxtBroker connect failed: {}", exc)
            self._connected = False
            return False

    def disconnect(self) -> bool:
        self._exchange = None
        self._connected = False
        logger.info("CcxtBroker disconnected")
        return True

    def is_connected(self) -> bool:
        return self._connected and self._exchange is not None

    def submit_order(self, order: Order) -> str:
        if not self.is_connected():
            raise RuntimeError("Broker not connected")

        side = "buy" if order.side == OrderSide.BUY else "sell"
        order_type_map = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP: "stop",
            OrderType.STOP_LIMIT: "stop",
        }
        ccxt_type = order_type_map.get(order.order_type, "market")

        params: dict[str, Any] = {}
        if order.stop_price and order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            params["stopPrice"] = order.stop_price

        try:
            result = self._exchange.create_order(
                symbol=order.symbol,
                type=ccxt_type,
                side=side,
                amount=order.quantity,
                price=order.price,
                params=params,
            )
            order.order_id = result["id"]
            order.status = OrderStatus.SUBMITTED
            self._orders[order.order_id] = order
            logger.info(
                "Order submitted: {} {} {} qty={} price={} id={}",
                self.exchange_name, side, order.symbol, order.quantity,
                order.price, order.order_id,
            )
            return order.order_id
        except Exception as exc:
            order.status = OrderStatus.ERROR
            logger.error("Order submit failed: {}", exc)
            raise

    def cancel_order(self, order_id: str) -> bool:
        if not self.is_connected():
            return False
        try:
            order = self._orders.get(order_id)
            symbol = order.symbol if order else None
            self._exchange.cancel_order(order_id, symbol)
            if order:
                order.status = OrderStatus.CANCELLED
            logger.info("Order cancelled: {}", order_id)
            return True
        except Exception as exc:
            logger.error("Cancel order failed: {}", exc)
            return False

    def get_order_status(self, order_id: str) -> Optional[Order]:
        if not self.is_connected():
            return None
        try:
            order = self._orders.get(order_id)
            if not order:
                return None
            result = self._exchange.fetch_order(order_id, order.symbol)
            status_map = {
                "open": OrderStatus.SUBMITTED,
                "closed": OrderStatus.FILLED,
                "canceled": OrderStatus.CANCELLED,
                "expired": OrderStatus.CANCELLED,
                "rejected": OrderStatus.REJECTED,
            }
            order.status = status_map.get(result.get("status", ""), OrderStatus.PENDING)
            order.filled_quantity = float(result.get("filled", 0))
            order.avg_fill_price = float(result.get("average", 0) or 0)
            order.commission = float(result.get("fee", {}).get("cost", 0) or 0)
            return order
        except Exception as exc:
            logger.error("Fetch order status failed: {}", exc)
            return None

    def get_account_info(self) -> AccountInfo:
        if not self.is_connected():
            return AccountInfo(cash=0, total_equity=0)
        try:
            balance = self._exchange.fetch_balance()
            total = float(balance.get("total", {}).get("USDT", 0))
            free = float(balance.get("free", {}).get("USDT", 0))
            return AccountInfo(
                cash=free,
                total_equity=total,
                positions=self._positions,
                buying_power=free,
            )
        except Exception as exc:
            logger.error("Fetch balance failed: {}", exc)
            return AccountInfo(cash=0, total_equity=0)

    def get_positions(self) -> Dict[str, Position]:
        if not self.is_connected():
            return {}
        try:
            raw_positions = self._exchange.fetch_positions()
            self._positions = {}
            for p in raw_positions:
                qty = float(p.get("contracts", 0) or 0)
                if qty == 0:
                    continue
                symbol = p.get("symbol", "")
                side = p.get("side", "long")
                entry = float(p.get("entryPrice", 0) or 0)
                mark = float(p.get("markPrice", 0) or entry)
                pnl = float(p.get("unrealizedPnl", 0) or 0)

                self._positions[symbol] = Position(
                    symbol=symbol,
                    quantity=qty if side == "long" else -qty,
                    avg_cost=entry,
                    current_price=mark,
                    unrealized_pnl=pnl,
                    market_value=abs(qty) * mark,
                )
            return self._positions
        except Exception as exc:
            logger.error("Fetch positions failed: {}", exc)
            return {}

    def get_position(self, symbol: str) -> Optional[Position]:
        positions = self.get_positions()
        return positions.get(symbol)

    def sync_state(self) -> bool:
        """Sync positions + open orders from exchange."""
        if not self.is_connected():
            return False
        try:
            self.get_positions()
            open_orders = self._exchange.fetch_open_orders()
            for o in open_orders:
                oid = o["id"]
                if oid not in self._orders:
                    self._orders[oid] = Order(
                        symbol=o.get("symbol", ""),
                        side=OrderSide.BUY if o.get("side") == "buy" else OrderSide.SELL,
                        quantity=float(o.get("amount", 0)),
                        price=float(o.get("price", 0) or 0),
                        order_id=oid,
                        status=OrderStatus.SUBMITTED,
                    )
            logger.info("State synced: {} positions, {} open orders",
                        len(self._positions), len(open_orders))
            return True
        except Exception as exc:
            logger.error("Sync state failed: {}", exc)
            return False
