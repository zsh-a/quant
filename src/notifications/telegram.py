from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import requests
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
            response = requests.post(url, json=payload, timeout=self.config.timeout_seconds)
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
