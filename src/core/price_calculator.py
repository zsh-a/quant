"""
价格计算器 — 基于规则的入场/止损/止盈价格推导。

从 ta_graph price_calculator.py 迁移，适配 Quent Bar 数据结构。

规则类型:
  入场: bar_high, bar_low, bar_close, current_price + offset
  止损: bar_low/high, swing_low/high, pattern_low/high + offset
  止盈: risk_multiple, measured_move, key_level
"""

from __future__ import annotations

from typing import Any

from loguru import logger


def _get_tick_size(symbol: str) -> float:
    s = symbol.upper()
    if "BTC" in s:
        return 0.1
    elif "ETH" in s:
        return 0.01
    elif s.startswith("SH.") or s.startswith("SZ."):
        return 0.01  # A 股
    return 0.0001


def calculate_entry_price(
    rule: dict[str, Any],
    bars: list[dict],
    current_price: float,
    symbol: str = "",
) -> float:
    """基于规则计算入场价格。

    rule 结构: {type, barIndex, offset}
    bars: list of {open, high, low, close, ...}
    """
    tick = _get_tick_size(symbol)
    idx = rule.get("barIndex", 0)
    array_idx = len(bars) - 1 + idx

    if array_idx < 0 or array_idx >= len(bars):
        return current_price

    bar = bars[array_idx]
    rtype = rule.get("type", "current_price")

    base = {
        "bar_high": bar.get("high", current_price),
        "bar_low": bar.get("low", current_price),
        "bar_close": bar.get("close", current_price),
        "current_price": current_price,
    }.get(rtype, current_price)

    offset = rule.get("offset", 1 if rtype in ("bar_high", "bar_low") else 0)
    return base + offset * tick if rtype != "bar_low" else base - offset * tick


def calculate_stop_loss(
    rule: dict[str, Any],
    bars: list[dict],
    entry_price: float,
    is_long: bool,
    symbol: str = "",
) -> float:
    """基于规则计算止损价格。

    rule 结构: {type, barIndex, swingStartBar, swingEndBar, offset, offsetPercent}
    """
    tick = _get_tick_size(symbol)
    rtype = rule.get("type", "bar_low" if is_long else "bar_high")

    if rtype in ("bar_low", "bar_high"):
        idx = rule.get("barIndex", -1)
        arr_idx = len(bars) - 1 + idx
        arr_idx = max(0, min(arr_idx, len(bars) - 1))
        bar = bars[arr_idx]
        base = bar.get("low", entry_price) if rtype == "bar_low" else bar.get("high", entry_price)

    elif rtype in ("swing_low", "swing_high", "pattern_low", "pattern_high"):
        start = rule.get("swingStartBar") or rule.get("patternStartBar", -10)
        end = rule.get("swingEndBar") or rule.get("patternEndBar", -1)
        start_idx = max(0, len(bars) - 1 + start)
        end_idx = min(len(bars) - 1, len(bars) - 1 + end)
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        sl = bars[start_idx:end_idx + 1]
        if not sl:
            return entry_price * (0.98 if is_long else 1.02)
        if rtype in ("swing_low", "pattern_low"):
            base = min(b.get("low", entry_price) for b in sl)
        else:
            base = max(b.get("high", entry_price) for b in sl)
    else:
        base = entry_price

    # Apply offset
    if rule.get("offsetPercent") is not None:
        offset_amount = base * rule["offsetPercent"] / 100.0
    elif rule.get("offset") is not None:
        offset_amount = rule["offset"] * tick
    else:
        offset_amount = tick

    return base - offset_amount if is_long else base + offset_amount


def calculate_take_profit(
    rule: dict[str, Any],
    bars: list[dict],
    entry_price: float,
    stop_loss: float,
) -> float:
    """基于规则计算止盈价格。

    rule 结构: {type, riskMultiple, measuredMoveBarStart, measuredMoveBarEnd, keyLevel}
    """
    risk = abs(entry_price - stop_loss)
    is_long = entry_price > stop_loss
    rtype = rule.get("type", "risk_multiple")

    if rtype == "risk_multiple":
        multiple = rule.get("riskMultiple", 2.0)
        return entry_price + risk * multiple if is_long else entry_price - risk * multiple

    elif rtype == "measured_move":
        start = rule.get("measuredMoveBarStart", -20)
        end = rule.get("measuredMoveBarEnd", -10)
        start_idx = max(0, len(bars) - 1 + start)
        end_idx = min(len(bars) - 1, len(bars) - 1 + end)
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        sl = bars[start_idx:end_idx + 1]
        if not sl:
            return entry_price + risk * 2 if is_long else entry_price - risk * 2
        impulse = max(b.get("high", 0) for b in sl) - min(b.get("low", 0) for b in sl)
        return entry_price + impulse if is_long else entry_price - impulse

    elif rtype == "key_level":
        return rule.get("keyLevel", entry_price + risk * 2 if is_long else entry_price - risk * 2)

    return entry_price + risk * 2 if is_long else entry_price - risk * 2


def enforce_min_stop_distance(
    entry: float,
    stop_loss: float,
    atr: float,
    min_atr_mult: float = 1.5,
    min_pct: float = 0.005,
) -> float:
    """确保止损距离不小于 max(min_atr_mult * ATR, min_pct * entry)。"""
    min_dist = max(min_atr_mult * atr, min_pct * entry)
    current_dist = abs(entry - stop_loss)
    if current_dist >= min_dist:
        return stop_loss
    is_long = entry > stop_loss
    adjusted = entry - min_dist if is_long else entry + min_dist
    logger.info("止损距离调整: {:.4f} → {:.4f} (最小距离 {:.4f})", stop_loss, adjusted, min_dist)
    return adjusted


def enforce_min_rr(
    entry: float,
    stop_loss: float,
    take_profit: float,
    min_rr: float = 2.0,
) -> float:
    """确保风险回报比不低于 min_rr，必要时调整止盈。"""
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    if risk == 0:
        return take_profit
    rr = reward / risk
    if rr >= min_rr:
        return take_profit
    is_long = entry > stop_loss
    adjusted = entry + risk * min_rr if is_long else entry - risk * min_rr
    logger.info("RR 调整: {:.4f} → {:.4f} (RR {:.1f} → {:.1f})", take_profit, adjusted, rr, min_rr)
    return adjusted
