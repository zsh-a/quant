"""
Bar 特征提取 — Al Brooks 风格的纯 Python K 线分析（零 API 成本）。

从 ta_graph l0_preprocessor.py 迁移并简化。

提供:
  - BarFeatures: 单根 K 线的特征 (类型/实体比/收盘位/EMA 关系)
  - MarketContext: 市场上下文 (ATR/死市场/区间位置)
  - is_dead_market(): 低波动过滤器
  - extract_features(): 批量提取
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class BarFeatures:
    """单根 K 线的 Brooks 特征。"""

    bar_type: Literal["bull_trend", "bear_trend", "bull_doji", "bear_doji"]
    body_pct: int  # 实体占比 0-100
    close_position: Literal["high", "mid", "low"]
    ema_relation: Literal["above", "at", "below"]
    ema_distance_pct: float  # 与 EMA20 的距离百分比
    is_inside_bar: bool
    is_outside_bar: bool
    is_reversal_bar: bool  # 长影线反转

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MarketContext:
    """市场整体上下文。"""

    atr_14: float
    atr_pct: float  # ATR 相对于价格的百分比
    is_dead_market: bool
    ema20: float
    recent_high: float
    recent_low: float
    price_position_in_range: float  # 0-1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def classify_bar(o: float, h: float, l: float, c: float) -> tuple[str, int]:
    """分类 K 线类型 + 实体占比。"""
    rng = h - l
    body = abs(c - o)
    pct = int(body / rng * 100) if rng > 0 else 0
    is_bull = c >= o
    is_trend = pct > 50
    if is_bull:
        return ("bull_trend" if is_trend else "bull_doji"), pct
    return ("bear_trend" if is_trend else "bear_doji"), pct


def classify_close_position(h: float, l: float, c: float) -> Literal["high", "mid", "low"]:
    rng = h - l
    if rng == 0:
        return "mid"
    pos = (c - l) / rng
    if pos >= 0.67:
        return "high"
    if pos <= 0.33:
        return "low"
    return "mid"


def calculate_atr(bars: list[dict], period: int = 14) -> float:
    """计算 ATR。"""
    if len(bars) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(bars)):
        h, l, pc = bars[i]["high"], bars[i]["low"], bars[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < period:
        return float(np.mean(trs)) if trs else 0.0
    return float(np.mean(trs[-period:]))


def is_dead_market(atr_pct: float, threshold: float = 0.005) -> bool:
    """ATR < 0.5% of price → 死鱼盘。"""
    return atr_pct < threshold


def extract_features(bars: list[dict], ema_period: int = 20) -> tuple[list[BarFeatures], MarketContext]:
    """批量提取 bar 特征 + 市场上下文。

    Args:
        bars: list of {open, high, low, close, volume, ...}

    Returns:
        (bar_features_list, market_context)
    """
    if len(bars) < ema_period + 1:
        return [], MarketContext(0, 0, True, 0, 0, 0, 0.5)

    closes = np.array([b["close"] for b in bars], dtype=np.float64)
    ema = float(np.mean(closes[-ema_period:]))
    atr = calculate_atr(bars)
    current_price = closes[-1]
    atr_pct = atr / current_price if current_price > 0 else 0.0

    recent = bars[-30:] if len(bars) >= 30 else bars
    recent_high = max(b["high"] for b in recent)
    recent_low = min(b["low"] for b in recent)
    rng = recent_high - recent_low
    pos_in_range = (current_price - recent_low) / rng if rng > 0 else 0.5

    ctx = MarketContext(
        atr_14=round(atr, 6),
        atr_pct=round(atr_pct, 6),
        is_dead_market=is_dead_market(atr_pct),
        ema20=round(ema, 4),
        recent_high=recent_high,
        recent_low=recent_low,
        price_position_in_range=round(pos_in_range, 4),
    )

    features = []
    for i in range(max(1, len(bars) - 10), len(bars)):
        b = bars[i]
        o, h, l, c = b["open"], b["high"], b["low"], b["close"]
        prev = bars[i - 1]

        bar_type, body_pct = classify_bar(o, h, l, c)
        close_pos = classify_close_position(h, l, c)

        # EMA relation
        dist = (c - ema) / atr if atr > 0 else 0
        if dist > 0.3:
            ema_rel = "above"
        elif dist < -0.3:
            ema_rel = "below"
        else:
            ema_rel = "at"

        # Inside / outside / reversal
        is_inside = h <= prev["high"] and l >= prev["low"]
        is_outside = h > prev["high"] and l < prev["low"]
        rng_bar = h - l
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        is_reversal = (upper_wick > 0.4 * rng_bar or lower_wick > 0.4 * rng_bar) if rng_bar > 0 else False

        features.append(
            BarFeatures(
                bar_type=bar_type,
                body_pct=body_pct,
                close_position=close_pos,
                ema_relation=ema_rel,
                ema_distance_pct=round(dist, 4),
                is_inside_bar=is_inside,
                is_outside_bar=is_outside,
                is_reversal_bar=is_reversal,
            )
        )

    return features, ctx
