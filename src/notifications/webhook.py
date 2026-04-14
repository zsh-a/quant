"""
通用 Webhook 通知器 — 支持 Slack / Discord / 自定义 Webhook。

配置:
  QUANT_NOTIFICATIONS__WEBHOOK__ENABLED=true
  QUANT_NOTIFICATIONS__WEBHOOK__URL=https://hooks.slack.com/services/...
  QUANT_NOTIFICATIONS__WEBHOOK__FORMAT=slack  (slack | discord | plain)
"""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger


class WebhookNotifier:
    """Send notifications to any webhook endpoint (Slack, Discord, custom)."""

    def __init__(self, url: str, format: str = "plain", timeout: int = 10):
        self.url = url
        self.format = format
        self.timeout = timeout

    def is_enabled(self) -> bool:
        return bool(self.url)

    def send(self, text: str, title: str = "") -> bool:
        """Send a message via webhook. Returns True on success."""
        if not self.is_enabled():
            return False
        payload = self._build_payload(text, title)
        try:
            resp = httpx.post(self.url, json=payload, timeout=self.timeout)
            if resp.status_code < 300:
                return True
            logger.warning("Webhook returned {}: {}", resp.status_code, resp.text[:200])
            return False
        except Exception as exc:
            logger.error("Webhook notification failed: {}", exc)
            return False

    def _build_payload(self, text: str, title: str) -> dict[str, Any]:
        if self.format == "slack":
            blocks = []
            if title:
                blocks.append({"type": "header", "text": {"type": "plain_text", "text": title}})
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": text}})
            return {"blocks": blocks}
        elif self.format == "discord":
            content = f"**{title}**\n{text}" if title else text
            return {"content": content}
        else:
            # Generic JSON payload
            payload: dict[str, Any] = {"text": text}
            if title:
                payload["title"] = title
            return payload
