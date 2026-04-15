"""
仓位管理器 — 订单监控、持仓同步、动态止盈止损。

从 ta_graph 的 order_monitor / position_sync / position_guard 合并而来。

Usage:
    pm = PositionManager(broker)
    pm.monitor_pending_order(order_id, timeout_bars=1)
    pm.sync_positions()
    pm.trail_stop(symbol, mode="below_prior_bar", bars=bars)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from loguru import logger


class TrailStopMode(str, Enum):
    """动态止盈止损模式。"""
    BELOW_PRIOR_BAR = "below_prior_bar"    # 止损跟踪到前一根 bar 低点 (多头)
    ABOVE_PRIOR_BAR = "above_prior_bar"    # 止损跟踪到前一根 bar 高点 (空头)
    BREAKEVEN = "breakeven"                 # 到达 1R 盈利后移到保本
    ATR_TRAIL = "atr_trail"                 # ATR 倍数跟踪
    NONE = "none"


@dataclass
class OrderMonitorResult:
    """订单监控结果。"""
    order_id: str
    status: str            # filled, cancelled, pending, timeout
    fill_price: float = 0.0
    fill_quantity: float = 0.0
    reason: str = ""


@dataclass
class SyncResult:
    """持仓同步结果。"""
    synced: bool
    local_positions: int
    exchange_positions: int
    discrepancies: list[str]


class PositionManager:
    """仓位生命周期管理器。"""

    def __init__(self, broker: Any = None):
        self.broker = broker

    # ------------------------------------------------------------------
    # 订单监控 — 超时取消
    # ------------------------------------------------------------------

    def monitor_pending_order(
        self,
        order_id: str,
        timeout_bars: int = 1,
        bar_interval_seconds: int = 3600,
    ) -> OrderMonitorResult:
        """监控挂单，超时后自动取消。

        Brooks 原则: setup 必须在 N 根 bar 内触发，否则失效。

        Args:
            order_id: 订单 ID
            timeout_bars: 超时 bar 数 (默认 1 bar)
            bar_interval_seconds: 每根 bar 的秒数

        Returns:
            OrderMonitorResult
        """
        if not self.broker:
            return OrderMonitorResult(order_id, "no_broker", reason="broker not configured")

        import time
        timeout_seconds = timeout_bars * bar_interval_seconds
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            try:
                if hasattr(self.broker, "get_order_status"):
                    order = self.broker.get_order_status(order_id)
                    if order and hasattr(order, "status"):
                        status_val = order.status if isinstance(order.status, str) else order.status.value
                        if status_val.lower() in ("filled", "closed"):
                            return OrderMonitorResult(
                                order_id, "filled",
                                fill_price=getattr(order, "avg_fill_price", 0),
                                fill_quantity=getattr(order, "filled_quantity", 0),
                            )
                        if status_val.lower() in ("cancelled", "rejected", "expired"):
                            return OrderMonitorResult(order_id, "cancelled", reason=status_val)
            except Exception as e:
                logger.warning("订单监控异常: {}", e)

            time.sleep(min(5, timeout_seconds / 10))

        # Timeout — 取消订单
        logger.info("订单超时 ({}s), 取消: {}", timeout_seconds, order_id)
        try:
            self.broker.cancel_order(order_id)
        except Exception:
            pass
        return OrderMonitorResult(order_id, "timeout", reason=f"{timeout_bars} bar(s) 内未成交")

    # ------------------------------------------------------------------
    # 持仓同步 — 本地 vs 交易所对账
    # ------------------------------------------------------------------

    def sync_positions(self) -> SyncResult:
        """同步本地仓位与交易所实际仓位。"""
        discrepancies: list[str] = []

        if not self.broker:
            return SyncResult(False, 0, 0, ["broker not configured"])

        try:
            # 获取交易所仓位
            if hasattr(self.broker, "get_positions"):
                exchange_pos = self.broker.get_positions()
            elif hasattr(self.broker, "positions"):
                exchange_pos = self.broker.positions
            else:
                return SyncResult(False, 0, 0, ["broker has no positions method"])

            # 获取本地仓位
            local_pos = getattr(self.broker, "positions", {})

            local_symbols = set(k for k, v in local_pos.items() if v != 0)
            exchange_symbols = set(k for k, v in (
                exchange_pos.items() if isinstance(exchange_pos, dict) else {}
            ) if (v if isinstance(v, (int, float)) else getattr(v, "quantity", 0)) != 0)

            # 本地有但交易所没有
            for sym in local_symbols - exchange_symbols:
                discrepancies.append(f"本地有但交易所无: {sym}")

            # 交易所有但本地没有
            for sym in exchange_symbols - local_symbols:
                discrepancies.append(f"交易所有但本地无: {sym}")

            if discrepancies:
                for d in discrepancies:
                    logger.warning("仓位不一致: {}", d)

            return SyncResult(
                synced=len(discrepancies) == 0,
                local_positions=len(local_symbols),
                exchange_positions=len(exchange_symbols),
                discrepancies=discrepancies,
            )

        except Exception as e:
            logger.error("仓位同步失败: {}", e)
            return SyncResult(False, 0, 0, [str(e)])

    # ------------------------------------------------------------------
    # 动态止盈止损
    # ------------------------------------------------------------------

    def trail_stop(
        self,
        current_stop: float,
        entry_price: float,
        is_long: bool,
        bars: list[dict],
        mode: TrailStopMode = TrailStopMode.BELOW_PRIOR_BAR,
        atr: float = 0.0,
        atr_multiplier: float = 1.5,
    ) -> float:
        """计算新的止损价格（只能向有利方向移动）。

        Returns:
            新止损价格（如果无变化则返回原值）
        """
        if not bars or mode == TrailStopMode.NONE:
            return current_stop

        new_stop = current_stop
        prev_bar = bars[-2] if len(bars) >= 2 else bars[-1]

        if mode == TrailStopMode.BELOW_PRIOR_BAR and is_long:
            candidate = prev_bar["low"]
            if candidate > current_stop:
                new_stop = candidate

        elif mode == TrailStopMode.ABOVE_PRIOR_BAR and not is_long:
            candidate = prev_bar["high"]
            if candidate < current_stop:
                new_stop = candidate

        elif mode == TrailStopMode.BREAKEVEN:
            # 到达 1R 盈利后移到保本
            risk = abs(entry_price - current_stop)
            current_price = bars[-1]["close"]
            pnl = (current_price - entry_price) if is_long else (entry_price - current_price)
            if pnl >= risk:
                if is_long and entry_price > current_stop:
                    new_stop = entry_price
                elif not is_long and entry_price < current_stop:
                    new_stop = entry_price

        elif mode == TrailStopMode.ATR_TRAIL and atr > 0:
            current_price = bars[-1]["close"]
            if is_long:
                candidate = current_price - atr * atr_multiplier
                if candidate > current_stop:
                    new_stop = candidate
            else:
                candidate = current_price + atr * atr_multiplier
                if candidate < current_stop:
                    new_stop = candidate

        if new_stop != current_stop:
            logger.info("止损调整 ({}) {:.4f} → {:.4f}", mode.value, current_stop, new_stop)

        return new_stop

    def check_followthrough(
        self,
        bars_since_entry: int,
        bars: list[dict],
        entry_price: float,
        is_long: bool,
    ) -> str:
        """检查入场后的 follow-through 质量。

        Brooks: 入场后 1-2 根 bar 最重要。

        Returns:
            "strong" | "weak" | "disappointing" | "too_early"
        """
        if bars_since_entry < 1:
            return "too_early"
        if bars_since_entry > 3:
            return "strong"  # 已经过了关键窗口

        recent = bars[-bars_since_entry:]
        if not recent:
            return "weak"

        # 判断最近 bar 是否朝有利方向发展
        favorable = 0
        for b in recent:
            if is_long and b["close"] > b["open"]:
                favorable += 1
            elif not is_long and b["close"] < b["open"]:
                favorable += 1

        current = bars[-1]["close"]
        pnl_pct = (current - entry_price) / entry_price if is_long else (entry_price - current) / entry_price

        if favorable == len(recent) and pnl_pct > 0:
            return "strong"
        elif pnl_pct < -0.005:  # -0.5% 即为失望
            return "disappointing"
        return "weak"
