"""
共享的 Pydantic 校验器 — 日期格式、股票代码、通用参数。
"""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import AfterValidator, Field

# ---------------------------------------------------------------------------
# Date validation
# ---------------------------------------------------------------------------
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")


def _validate_date(v: str) -> str:
    if not _DATE_RE.match(v):
        raise ValueError(f"日期格式必须为 YYYY-MM-DD，收到: {v!r}")
    return v


def _validate_date_or_datetime(v: str) -> str:
    if not _DATE_RE.match(v) and not _DATETIME_RE.match(v):
        raise ValueError(f"日期格式必须为 YYYY-MM-DD 或 ISO datetime，收到: {v!r}")
    return v


DateStr = Annotated[str, AfterValidator(_validate_date)]
DateTimeStr = Annotated[str, AfterValidator(_validate_date_or_datetime)]

# ---------------------------------------------------------------------------
# Stock / symbol code validation
# ---------------------------------------------------------------------------
_STOCK_CODE_RE = re.compile(r"^[a-zA-Z]{2}\.\d{6}$")  # e.g. sh.600000, sz.000001
_CRYPTO_SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,20}(USDT|BUSD|USD|BTC|ETH)$")
_GENERAL_SYMBOL_RE = re.compile(r"^[a-zA-Z0-9._-]{1,30}$")


def _validate_symbol(v: str) -> str:
    if not _GENERAL_SYMBOL_RE.match(v):
        raise ValueError(f"非法的标的代码: {v!r}")
    return v


SymbolStr = Annotated[str, AfterValidator(_validate_symbol)]

# ---------------------------------------------------------------------------
# Mode validation
# ---------------------------------------------------------------------------
_VALID_MODES = {"backtest", "simulation", "live", "paper"}


def _validate_mode(v: str) -> str:
    if v not in _VALID_MODES:
        raise ValueError(f"mode 必须为 {_VALID_MODES} 之一，收到: {v!r}")
    return v


ModeStr = Annotated[str, AfterValidator(_validate_mode)]

# ---------------------------------------------------------------------------
# Market validation
# ---------------------------------------------------------------------------
_VALID_MARKETS = {"a_share", "crypto"}


def _validate_market(v: str) -> str:
    if v not in _VALID_MARKETS:
        raise ValueError(f"market 必须为 {_VALID_MARKETS} 之一，收到: {v!r}")
    return v


MarketStr = Annotated[str, AfterValidator(_validate_market)]

# ---------------------------------------------------------------------------
# Interval validation
# ---------------------------------------------------------------------------
_VALID_INTERVALS = {"1d", "5m", "15m", "1h", "4h"}


def _validate_interval(v: str) -> str:
    if v not in _VALID_INTERVALS:
        raise ValueError(f"interval 必须为 {_VALID_INTERVALS} 之一，收到: {v!r}")
    return v


IntervalStr = Annotated[str, AfterValidator(_validate_interval)]

# ---------------------------------------------------------------------------
# Reusable Field aliases
# ---------------------------------------------------------------------------
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
Ratio = Annotated[float, Field(ge=0.0, le=1.0)]
