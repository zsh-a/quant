from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import httpx
from loguru import logger

from src.config.settings import TelegramConfig
from src.core.base import Order


class TelegramNotifier:
    """Minimal Telegram Bot API notifier."""

    def __init__(self, config: TelegramConfig):
        self.config = config

    def is_enabled(self) -> bool:
        return bool(self.config.enabled and self.config.bot_token)

    def send_message(self, chat_id: str, text: str) -> bool:
        if not self.is_enabled():
            logger.info("Telegram notifier disabled or bot token missing")
            return False
        if not chat_id:
            logger.warning("Telegram chat_id missing, skip notification")
            return False

        url = f"{self.config.api_base_url}/bot{self.config.bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        try:
            response = httpx.post(url, json=payload, timeout=self.config.timeout_seconds)
            response.raise_for_status()
            body = response.json()
            if not body.get("ok", False):
                logger.error("Telegram API rejected message: {}", body)
                return False
            return True
        except Exception as exc:
            logger.exception("Failed to send Telegram notification: {}", exc)
            return False


def build_simulation_trade_message(
    *,
    job_name: str,
    strategy_name: str,
    symbol: str,
    session_id: str,
    run_id: str,
    trade: Dict[str, Any],
) -> str:
    action = "买入" if trade.get("type") == "buy" else "卖出"
    trade_symbol = trade.get("symbol") or symbol
    trade_name = trade.get("name") or "Unknown"
    price = float(trade.get("price", 0.0))
    quantity = float(trade.get("quantity", 0.0))
    amount = float(trade.get("amount", price * quantity))
    commission = float(trade.get("commission", 0.0))
    timestamp = trade.get("timestamp") or datetime.now().isoformat()

    return "\n".join(
        [
            "*模拟交易下单通知*",
            f"*任务*: `{job_name}`",
            f"*策略*: `{strategy_name}`",
            f"*标的*: `{trade_symbol}` {trade_name}",
            f"*方向*: {action}",
            f"*数量*: `{quantity:g}`",
            f"*成交价*: `{price:.4f}`",
            f"*成交额*: `{amount:.2f}`",
            f"*手续费*: `{commission:.2f}`",
            f"*成交时间*: `{timestamp}`",
            f"*Session*: `{session_id}`",
            f"*Run*: `{run_id}`",
        ]
    )


def build_simulation_order_message(
    *,
    job_name: str,
    strategy_name: str,
    session_id: str,
    run_id: str,
    order: Order,
    reference_price: Optional[float] = None,
    stock_name: str = "",
) -> str:
    action = "买入" if order.type == "buy" else "卖出"
    price_text = f"`{float(order.price):.4f}`" if order.price is not None else "市价"
    reference_price_text = (
        f"`{float(reference_price):.4f}`" if reference_price is not None and reference_price > 0 else "N/A"
    )
    stock_display = f"{order.symbol} {stock_name}".strip()

    return "\n".join(
        [
            "*模拟交易下单通知*",
            f"*任务*: `{job_name}`",
            f"*策略*: `{strategy_name}`",
            f"*标的*: `{stock_display}`",
            f"*方向*: {action}",
            f"*数量*: `{float(order.quantity):g}`",
            f"*订单价格*: {price_text}",
            f"*执行方式*: `{order.execution_type}`",
            f"*参考价格*: {reference_price_text}",
            f"*下单时间*: `{order.created_at.isoformat()}`",
            f"*Session*: `{session_id}`",
            f"*Run*: `{run_id}`",
        ]
    )


def get_notification_chat_id(
    notification_config: Optional[Dict[str, Any]],
    telegram_config: TelegramConfig,
) -> str:
    telegram_job_config = (notification_config or {}).get("telegram") or {}
    return str(telegram_job_config.get("chat_id") or telegram_config.default_chat_id or "")


def is_trade_notification_enabled(
    notification_config: Optional[Dict[str, Any]],
    telegram_config: TelegramConfig,
) -> bool:
    telegram_job_config = (notification_config or {}).get("telegram") or {}
    return bool(telegram_config.enabled and telegram_job_config.get("enabled", False))


# ---------------------------------------------------------------------------
# 实盘交易通知
# ---------------------------------------------------------------------------


def build_live_trade_message(*, trade: Dict[str, Any], strategy: str = "") -> str:
    """构建实盘成交通知消息。"""
    action = "买入" if trade.get("type") == "buy" else "卖出"
    return "\n".join([
        "✅ *实盘成交通知*",
        f"*策略*: `{strategy}`" if strategy else "",
        f"*标的*: `{trade.get('symbol', '')}` {trade.get('name', '')}",
        f"*方向*: {action}",
        f"*数量*: `{trade.get('quantity', 0):g}`",
        f"*成交价*: `{trade.get('price', 0):.4f}`",
        f"*成交额*: `{trade.get('amount', 0):.2f}`",
        f"*手续费*: `{trade.get('commission', 0):.2f}`",
        f"*时间*: `{trade.get('timestamp', '')}`",
    ])


def build_live_rejection_message(*, symbol: str, order_type: str, reason: str) -> str:
    """构建实盘拒单通知消息。"""
    action = "买入" if order_type == "buy" else "卖出"
    return "\n".join([
        "❌ *实盘拒单通知*",
        f"*标的*: `{symbol}`",
        f"*方向*: {action}",
        f"*原因*: {reason}",
        f"*时间*: `{datetime.now().isoformat()}`",
    ])


def build_risk_alert_message(*, alert_type: str, message: str) -> str:
    """构建风控告警通知消息。"""
    return "\n".join([
        "⚠️ *风控告警*",
        f"*类型*: {alert_type}",
        f"*详情*: {message}",
        f"*时间*: `{datetime.now().isoformat()}`",
    ])


def build_daily_pnl_message(
    *, strategy: str, date: str, pnl: float, return_pct: float, equity: float,
) -> str:
    """构建每日 P&L 汇总消息。"""
    emoji = "📈" if pnl >= 0 else "📉"
    return "\n".join([
        f"{emoji} *每日 P&L 汇总*",
        f"*策略*: `{strategy}`",
        f"*日期*: `{date}`",
        f"*日盈亏*: `{pnl:+,.2f}` ({return_pct:+.2%})",
        f"*总权益*: `{equity:,.2f}`",
    ])
