"""
交易过滤器 + 权益保护器 — 防止过度交易、控制亏损。

从 ta_graph 迁移并简化，适配 Quent Strategy/Broker 架构。

Usage:
    tf = TradeFilter()
    ok, reasons = tf.check_all(probability=72.0, signal_quality=8)
    if ok:
        broker.submit_order(...)
        tf.record_trade()
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Optional

import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Trade Filter — 防过度交易
# ---------------------------------------------------------------------------

@dataclass
class TradeFilterConfig:
    cooldown_minutes: int = 15
    max_daily_trades: int = 5
    min_probability: float = 60.0
    min_signal_quality: int = 6
    barb_wire_overlap_threshold: float = 0.6  # 60% overlap = barb wire


class TradeFilter:
    """多重交易过滤器，每个检查返回 (passed, reason)。"""

    def __init__(self, config: TradeFilterConfig | None = None):
        self.cfg = config or TradeFilterConfig()
        self.last_trade_time: Optional[datetime] = None
        self.trades_today: int = 0
        self._daily_reset_date: date = date.today()

    def _maybe_reset_daily(self):
        today = date.today()
        if today > self._daily_reset_date:
            self.trades_today = 0
            self._daily_reset_date = today

    def check_cooldown(self) -> tuple[bool, str]:
        if self.last_trade_time is None:
            return True, ""
        elapsed = datetime.now() - self.last_trade_time
        required = timedelta(minutes=self.cfg.cooldown_minutes)
        if elapsed < required:
            remaining = int((required - elapsed).total_seconds() / 60)
            return False, f"冷却中: 还需 {remaining} 分钟 (最少间隔 {self.cfg.cooldown_minutes}m)"
        return True, ""

    def check_daily_limit(self) -> tuple[bool, str]:
        self._maybe_reset_daily()
        if self.trades_today >= self.cfg.max_daily_trades:
            return False, f"日交易上限: {self.trades_today}/{self.cfg.max_daily_trades}"
        return True, ""

    def check_probability(self, probability: float) -> tuple[bool, str]:
        if probability < self.cfg.min_probability:
            return False, f"概率不足: {probability:.1f}% < {self.cfg.min_probability}%"
        return True, ""

    def check_signal_quality(self, quality: int) -> tuple[bool, str]:
        if quality < self.cfg.min_signal_quality:
            return False, f"信号质量不足: {quality}/10 < {self.cfg.min_signal_quality}/10"
        return True, ""

    def check_barb_wire(self, bars: list[dict] | Any) -> tuple[bool, str]:
        """检测 Barb Wire (重叠十字星密集区) — Brooks: 不交易。"""
        if bars is None or len(bars) < 7:
            return True, ""

        recent = bars[-7:]
        overlap_count = 0
        for i in range(1, len(recent)):
            curr = recent[i]
            prev = recent[i - 1]
            body_top = max(curr.get("open", 0), curr.get("close", 0))
            body_bot = min(curr.get("open", 0), curr.get("close", 0))
            if body_top <= prev.get("high", 0) and body_bot >= prev.get("low", 0):
                overlap_count += 1

        total = len(recent) - 1
        if total > 0 and (overlap_count / total) >= self.cfg.barb_wire_overlap_threshold:
            # 检查是否有趋势 (趋势中重叠可接受)
            highs = [b.get("high", 0) for b in recent]
            lows = [b.get("low", 0) for b in recent]
            net_move = max(highs) - min(lows)
            avg_range = np.mean([h - l for h, l in zip(highs, lows)])
            if avg_range > 0 and net_move / avg_range > 4.0:
                return True, ""  # 有趋势，放行
            return False, f"Barb Wire: {overlap_count}/{total} bar 重叠，市场震荡"

        return True, ""

    def check_all(
        self,
        probability: float = 100.0,
        signal_quality: int = 10,
        bars: list[dict] | Any = None,
    ) -> tuple[bool, list[str]]:
        """执行所有过滤器检查。返回 (passed, [reasons])。"""
        failed: list[str] = []
        for passed, reason in [
            self.check_cooldown(),
            self.check_daily_limit(),
            self.check_probability(probability),
            self.check_signal_quality(signal_quality),
            self.check_barb_wire(bars),
        ]:
            if not passed:
                failed.append(reason)

        if failed:
            logger.info("交易被过滤 ({}): {}", len(failed), "; ".join(failed))

        return len(failed) == 0, failed

    def record_trade(self):
        """记录一次交易执行。"""
        self.last_trade_time = datetime.now()
        self.trades_today += 1
        logger.info("交易记录: 今日第 {}/{}", self.trades_today, self.cfg.max_daily_trades)

    def get_status(self) -> dict[str, Any]:
        self._maybe_reset_daily()
        cooldown_remaining = 0
        if self.last_trade_time:
            elapsed = datetime.now() - self.last_trade_time
            remaining = timedelta(minutes=self.cfg.cooldown_minutes) - elapsed
            cooldown_remaining = max(0, int(remaining.total_seconds() / 60))
        return {
            "trades_today": self.trades_today,
            "max_daily_trades": self.cfg.max_daily_trades,
            "cooldown_remaining_minutes": cooldown_remaining,
        }


# ---------------------------------------------------------------------------
# Equity Protector — 权益保护
# ---------------------------------------------------------------------------

@dataclass
class EquityProtectorConfig:
    max_daily_loss_pct: float = 2.0
    max_consecutive_losses: int = 3
    cooldown_hours: int = 2


class EquityProtector:
    """资金保护器 — 日亏损熔断 + 连损冷却。"""

    def __init__(self, config: EquityProtectorConfig | None = None):
        self.cfg = config or EquityProtectorConfig()
        self.daily_pnl: float = 0.0
        self.consecutive_losses: int = 0
        self.trading_enabled: bool = True
        self.cooldown_until: Optional[datetime] = None
        self._last_reset_date: date = date.today()

    def update_trade_result(self, pnl: float, account_balance: float):
        """更新交易结果，检查熔断条件。"""
        self.daily_pnl += pnl

        if pnl < 0:
            self.consecutive_losses += 1
            logger.warning("亏损 #{}: {:.2f}", self.consecutive_losses, pnl)
        else:
            self.consecutive_losses = 0

        # 日亏损熔断
        if self.daily_pnl < 0 and account_balance > 0:
            loss_pct = abs(self.daily_pnl / account_balance * 100)
            if loss_pct >= self.cfg.max_daily_loss_pct:
                self.trading_enabled = False
                logger.critical(
                    "日亏损熔断: {:.2f}% >= {:.1f}% — 交易已禁止",
                    loss_pct, self.cfg.max_daily_loss_pct,
                )

        # 连损冷却
        if self.consecutive_losses >= self.cfg.max_consecutive_losses:
            self.trading_enabled = False
            self.cooldown_until = datetime.now() + timedelta(hours=self.cfg.cooldown_hours)
            logger.warning(
                "连损冷却: {} 次连续亏损 → 暂停 {}h",
                self.consecutive_losses, self.cfg.cooldown_hours,
            )

    def can_trade(self) -> bool:
        # 日重置
        today = date.today()
        if today > self._last_reset_date:
            self.daily_pnl = 0.0
            self.trading_enabled = True
            self._last_reset_date = today

        # 冷却期结束
        if self.cooldown_until and datetime.now() >= self.cooldown_until:
            self.trading_enabled = True
            self.cooldown_until = None
            logger.info("冷却期结束，恢复交易")

        return self.trading_enabled

    def force_enable(self):
        """管理员强制恢复交易。"""
        self.trading_enabled = True
        self.cooldown_until = None
        logger.warning("管理员强制恢复交易")

    def get_status(self) -> dict[str, Any]:
        return {
            "trading_enabled": self.trading_enabled,
            "daily_pnl": self.daily_pnl,
            "consecutive_losses": self.consecutive_losses,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
        }
