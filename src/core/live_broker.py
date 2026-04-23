"""
A 股实盘 Broker — 通过 HTTP 接口与远端交易服务器通信。

功能:
  - 订单提交/取消/状态轮询
  - 成交价格和手续费追踪
  - 持仓定时同步
  - 错误计数 + 回调通知
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import httpx
from loguru import logger

from .base import Bar, Broker, Order


class LiveBroker(Broker):
    def __init__(
        self,
        server_url: str,
        order_poll_interval: float = 1.0,
        order_timeout: int = 30,
        on_trade: Optional[Callable[[dict], None]] = None,
        on_rejection: Optional[Callable[[Order, str], None]] = None,
    ):
        self.server_url = server_url
        self.order_poll_interval = order_poll_interval
        self.order_timeout = order_timeout
        self.on_trade = on_trade
        self.on_rejection = on_rejection

        self.orders: Dict[str, Order] = {}
        self.filled_orders: List[Order] = []
        self.stock_names: Dict[str, str] = {}
        self.positions: Dict[str, float] = {}
        self.cash: float = 0.0
        self.consecutive_errors: int = 0

        self._http = httpx.Client(base_url=server_url, timeout=10)
        self._load_stock_names()
        self.sync_state()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _load_stock_names(self):
        try:
            from src.market_data.clickhouse import create_clickhouse_client

            client = create_clickhouse_client()
            data = client.query("SELECT code, name FROM stock_data.stock_daily_meta WHERE name != ''")
            for code, name in data.result_rows:
                self.stock_names[code] = name
        except Exception as e:
            logger.warning(f"Failed to load stock names: {e}")

    def sync_state(self):
        """Sync positions and cash from remote broker."""
        try:
            info = self.get_account_info()
            self.positions = info.get("positions", {})
            self.cash = info.get("cash", 0.0)
            self.consecutive_errors = 0
            logger.info("LiveBroker synced: cash={:.2f} positions={}", self.cash, len(self.positions))
        except Exception as e:
            logger.error("LiveBroker sync failed: {}", e)

    # ------------------------------------------------------------------
    # Order lifecycle
    # ------------------------------------------------------------------

    def submit_order(self, order: Order) -> str:
        order.id = str(uuid.uuid4())
        order.status = "SUBMITTED"
        order.created_at = datetime.now()
        self.orders[order.id] = order

        try:
            endpoint = "/buy" if order.type == "buy" else "/sell"
            params: dict[str, Any] = {
                "code": order.symbol,
                "amount": int(order.quantity),
            }
            if order.price:
                params["price"] = order.price

            logger.info(
                "LiveBroker submit: {} {} qty={} price={}", order.type, order.symbol, order.quantity, order.price
            )

            resp = self._http.get(endpoint, params=params)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") == -1:
                order.status = "REJECTED"
                reason = data.get("msg", str(data))
                logger.error("订单被拒: {} — {}", order.symbol, reason)
                if self.on_rejection:
                    self.on_rejection(order, reason)
                self.consecutive_errors += 1
            else:
                # 轮询等待成交
                self._poll_until_filled(order)
                self.consecutive_errors = 0

        except httpx.HTTPError as e:
            logger.error("网络错误: {}", e)
            order.status = "ERROR"
            self.consecutive_errors += 1
        except Exception as e:
            logger.error("下单异常: {}", e)
            order.status = "ERROR"
            self.consecutive_errors += 1

        return order.id

    def _poll_until_filled(self, order: Order):
        """轮询订单状态直到成交、拒绝或超时。"""
        deadline = time.time() + self.order_timeout
        while time.time() < deadline:
            try:
                info = self.get_account_info()
                # 检查持仓变化来确认成交
                new_qty = info.get("positions", {}).get(order.symbol, 0)
                old_qty = self.positions.get(order.symbol, 0)

                if order.type == "buy" and new_qty > old_qty:
                    fill_qty = new_qty - old_qty
                    self._mark_filled(order, info, fill_qty)
                    return
                elif order.type == "sell" and new_qty < old_qty:
                    fill_qty = old_qty - new_qty
                    self._mark_filled(order, info, fill_qty)
                    return
            except Exception as e:
                logger.warning("订单状态轮询异常: {}", e)

            time.sleep(self.order_poll_interval)

        # Timeout — 可能仍在排队，标记为已提交
        logger.warning("订单超时未确认成交: {} {}", order.type, order.symbol)

    def _mark_filled(self, order: Order, account_info: dict, fill_qty: float):
        """标记订单成交并记录成交信息。"""
        order.status = "FILLED"
        order.filled_quantity = fill_qty
        order.avg_fill_price = self._get_fill_price(order.symbol, account_info)
        order.created_at = order.created_at or datetime.now()
        self.filled_orders.append(order)

        # 更新本地持仓
        self.positions = account_info.get("positions", self.positions)
        self.cash = account_info.get("cash", self.cash)

        trade_record = {
            "timestamp": str(datetime.now()),
            "symbol": order.symbol,
            "name": self.stock_names.get(order.symbol, "Unknown"),
            "type": order.type,
            "price": float(order.avg_fill_price),
            "quantity": float(fill_qty),
            "amount": float(fill_qty * order.avg_fill_price),
            "commission": float(fill_qty * order.avg_fill_price * 0.0003),  # 估算手续费
        }

        logger.info(
            "成交: {} {} {} qty={} price={:.2f}",
            order.type,
            order.symbol,
            self.stock_names.get(order.symbol, ""),
            fill_qty,
            order.avg_fill_price,
        )

        if self.on_trade:
            self.on_trade(trade_record)

    def _get_fill_price(self, symbol: str, account_info: dict) -> float:
        """从持仓详情中获取成交均价。"""
        detailed = account_info.get("detailed_positions", {})
        pos_info = detailed.get(symbol, {})
        return float(pos_info.get("avg_cost", 0.0))

    def cancel_order(self, order_id: str):
        """取消挂单。"""
        order = self.orders.get(order_id)
        if not order or order.status != "SUBMITTED":
            return

        try:
            resp = self._http.get("/cancel", params={"order_id": order_id})
            resp.raise_for_status()
            order.status = "CANCELLED"
            logger.info("订单已取消: {}", order_id)
        except Exception as e:
            logger.warning("取消订单失败 ({}): {}", order_id, e)
            # 某些 broker 不支持取消，标记为 WARNING 但不报错
            order.status = "CANCEL_FAILED"

    # ------------------------------------------------------------------
    # Account info
    # ------------------------------------------------------------------

    def get_account_info(self) -> Dict[str, Any]:
        bal_resp = self._http.get("/balance")
        bal_resp.raise_for_status()
        bal_data = bal_resp.json()

        cash = float(bal_data.get("zj", 0) if isinstance(bal_data, dict) else 0)

        pos_resp = self._http.get("/position")
        pos_resp.raise_for_status()
        pos_data = pos_resp.json()

        positions: Dict[str, float] = {}
        detailed: Dict[str, dict] = {}

        if isinstance(pos_data, dict) and "data" in pos_data:
            for p in pos_data["data"]:
                code = p.get("证券代码")
                qty = float(p.get("股票余额", 0))
                if not code or qty <= 0:
                    continue
                positions[code] = qty
                detailed[code] = {
                    "qty": qty,
                    "name": p.get("证券名称", "Unknown"),
                    "price": float(p.get("市价", 0)),
                    "value": float(p.get("市值", 0)),
                    "avg_cost": float(p.get("成本价", 0)),
                    "unrealized_pnl": float(p.get("浮动盈亏", 0)),
                    "pnl_pct": float(p.get("盈亏比例(%)", 0)) / 100.0 if "盈亏比例(%)" in p else 0.0,
                }

        trades_list = [
            {
                "timestamp": str(o.created_at),
                "symbol": o.symbol,
                "name": self.stock_names.get(o.symbol, "Unknown"),
                "type": o.type,
                "price": float(o.avg_fill_price),
                "quantity": float(o.filled_quantity),
                "amount": float(o.filled_quantity * o.avg_fill_price),
                "commission": float(o.filled_quantity * o.avg_fill_price * 0.0003),
            }
            for o in self.filled_orders
        ]

        return {
            "cash": cash,
            "positions": positions,
            "detailed_positions": detailed,
            "total_equity": cash + sum(p["value"] for p in detailed.values()),
            "equity_history": [],
            "trades": trades_list,
            "pending_orders": [vars(o) for o in self.orders.values() if o.status == "SUBMITTED"],
        }

    def step(self, bars: Dict[str, Bar]):
        """Live mode: sync positions from broker each bar."""
        try:
            self.sync_state()
        except Exception as e:
            logger.warning("LiveBroker step sync failed: {}", e)

    def process_same_bar_orders(self, bars: Dict[str, Bar], timing: str):
        """Live mode: immediate orders are submitted directly."""
        pass  # 实盘中 IMMEDIATE 订单在 submit_order 中直接发送


def create_live_broker(mode: str = "paper", **kwargs) -> Broker:
    """Factory for the Phase 4.6 BrooksLive stack.

    ``mode="paper"`` returns a :class:`BacktestBroker` configured to simulate
    fills locally — no real orders are placed. ``mode="live"`` is explicitly
    disabled by Phase 4.6 scope and raises :class:`ValueError`; callers must
    opt-in through a future release once real-order safeguards land.
    """
    if mode == "paper":
        from src.core.backtest_broker import BacktestBroker

        allowed = {
            "initial_cash",
            "commission",
            "slippage",
            "db_client",
            "risk_manager",
            "on_order_submitted",
            "allow_short",
            "session_id",
        }
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return BacktestBroker(**filtered)
    if mode == "live":
        raise ValueError("live mode is disabled in Phase 4.6 (BrooksLive is paper-only). Use mode='paper'.")
    raise ValueError(f"Unknown live broker mode: {mode!r} (expected 'paper' or 'live')")
