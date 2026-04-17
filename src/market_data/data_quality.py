"""
数据质量检查 — 缺失率、异常值、新鲜度报告。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from loguru import logger

from src.market_data.clickhouse import create_clickhouse_client


class DataQualityChecker:
    """Run quality checks against ClickHouse market data."""

    def __init__(self):
        self.client = create_clickhouse_client()

    def full_report(self) -> dict[str, Any]:
        """Generate a complete data quality report."""
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "stock_daily": self._check_stock_daily(),
            "crypto_futures": self._check_crypto_futures(),
            "freshness": self._check_freshness(),
        }

    # ------------------------------------------------------------------
    # Stock daily data
    # ------------------------------------------------------------------
    def _check_stock_daily(self) -> dict[str, Any]:
        try:
            result = self.client.query(
                "SELECT "
                "  count() AS total_rows, "
                "  countIf(close = 0 OR close IS NULL) AS zero_close, "
                "  countIf(high < low) AS invalid_ohlc, "
                "  countIf(volume = 0) AS zero_volume, "
                "  uniqExact(code) AS unique_codes, "
                "  min(date) AS earliest, "
                "  max(date) AS latest "
                "FROM stock_data.stock_daily"
            )
            row = result.result_rows[0] if result.result_rows else None
            if not row:
                return {"status": "no_data"}

            total = row[0]
            return {
                "status": "ok",
                "total_rows": total,
                "zero_close_count": row[1],
                "zero_close_pct": round(row[1] / total * 100, 2) if total else 0,
                "invalid_ohlc_count": row[2],
                "zero_volume_count": row[3],
                "unique_codes": row[4],
                "date_range": [str(row[5]), str(row[6])],
            }
        except Exception as exc:
            logger.warning("stock_daily quality check failed: {}", exc)
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Crypto futures data
    # ------------------------------------------------------------------
    def _check_crypto_futures(self) -> dict[str, Any]:
        try:
            result = self.client.query(
                "SELECT "
                "  count() AS total_rows, "
                "  countIf(close = 0 OR close IS NULL) AS zero_close, "
                "  countIf(high < low) AS invalid_ohlc, "
                "  countIf(volume = 0) AS zero_volume, "
                "  uniqExact(symbol) AS unique_symbols, "
                "  min(open_time) AS earliest, "
                "  max(open_time) AS latest "
                "FROM crypto_data.futures_5m"
            )
            row = result.result_rows[0] if result.result_rows else None
            if not row:
                return {"status": "no_data"}

            total = row[0]
            return {
                "status": "ok",
                "total_rows": total,
                "zero_close_count": row[1],
                "zero_close_pct": round(row[1] / total * 100, 2) if total else 0,
                "invalid_ohlc_count": row[2],
                "zero_volume_count": row[3],
                "unique_symbols": row[4],
                "time_range": [str(row[5]), str(row[6])],
            }
        except Exception as exc:
            logger.warning("crypto_futures quality check failed: {}", exc)
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Data freshness
    # ------------------------------------------------------------------
    def _check_freshness(self) -> dict[str, Any]:
        """Check how fresh each data source is."""
        checks = {}
        now = datetime.utcnow()

        # Stock daily
        try:
            r = self.client.query("SELECT max(date) FROM stock_data.stock_daily")
            latest = r.result_rows[0][0] if r.result_rows else None
            if latest:
                if hasattr(latest, "date"):
                    latest = latest
                age_days = now.date() - latest.date() if hasattr(latest, "date") else (now - latest).days
                if hasattr(age_days, "days"):
                    age_days = age_days.days
                checks["stock_daily"] = {
                    "latest": str(latest),
                    "age_days": age_days,
                    "stale": age_days > 3,
                }
        except Exception:
            checks["stock_daily"] = {"status": "error"}

        # Crypto futures
        try:
            r = self.client.query("SELECT max(open_time) FROM crypto_data.futures_5m")
            latest = r.result_rows[0][0] if r.result_rows else None
            if latest:
                age = now - latest.replace(tzinfo=None) if latest.tzinfo else now - latest
                checks["crypto_futures"] = {
                    "latest": str(latest),
                    "age_hours": round(age.total_seconds() / 3600, 1),
                    "stale": age > timedelta(hours=6),
                }
        except Exception:
            checks["crypto_futures"] = {"status": "error"}

        return checks
