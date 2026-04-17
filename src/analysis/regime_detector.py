"""
市场 Regime 检测器 — 基于波动率和趋势判断市场状态。

三种 Regime:
  - bull:     上升趋势 + 低/中波动
  - bear:     下降趋势 + 高波动
  - sideways: 无明显趋势 / 波动率中性

方法: 滚动波动率 z-score + 均线趋势斜率
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger


class Regime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"


@dataclass
class RegimeSnapshot:
    """当前市场状态快照。"""

    regime: Regime
    confidence: float  # 0-1 置信度
    volatility_zscore: float  # 波动率 z-score (>1.5 高波动)
    trend_slope: float  # 趋势斜率 (正=上升)
    current_price: float
    ma_fast: float  # 短期均线
    ma_slow: float  # 长期均线
    annualized_vol: float  # 年化波动率


class RegimeDetector:
    """基于价格序列的 Regime 检测器。"""

    def __init__(
        self,
        fast_window: int = 20,
        slow_window: int = 60,
        vol_window: int = 20,
        vol_lookback: int = 252,
        trend_threshold: float = 0.0002,
        vol_high_threshold: float = 1.2,
    ):
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.vol_window = vol_window
        self.vol_lookback = vol_lookback
        self.trend_threshold = trend_threshold
        self.vol_high_threshold = vol_high_threshold

    def detect(self, prices: np.ndarray | pd.Series) -> RegimeSnapshot:
        """从价格序列检测当前 regime。

        Args:
            prices: 日收盘价序列，至少 slow_window + vol_lookback 个数据点。
        """
        arr = np.asarray(prices, dtype=np.float64)
        n = len(arr)
        min_required = self.slow_window + self.vol_lookback
        if n < min_required:
            logger.warning("regime_detector: 数据不足 ({} < {}), 默认 sideways", n, min_required)
            return RegimeSnapshot(
                regime=Regime.SIDEWAYS,
                confidence=0.0,
                volatility_zscore=0.0,
                trend_slope=0.0,
                current_price=float(arr[-1]) if n > 0 else 0.0,
                ma_fast=0.0,
                ma_slow=0.0,
                annualized_vol=0.0,
            )

        # 均线
        ma_fast = float(np.mean(arr[-self.fast_window :]))
        ma_slow = float(np.mean(arr[-self.slow_window :]))
        current_price = float(arr[-1])

        # 趋势斜率: 短期均线的线性回归斜率 (标准化)
        recent = arr[-self.fast_window :]
        x = np.arange(self.fast_window, dtype=np.float64)
        x_mean = x.mean()
        y_mean = recent.mean()
        slope = float(np.sum((x - x_mean) * (recent - y_mean)) / np.sum((x - x_mean) ** 2))
        norm_slope = slope / y_mean if y_mean != 0 else 0.0  # 相对于均价的斜率

        # 波动率
        log_returns = np.diff(np.log(arr[-self.vol_lookback :]))
        current_vol = float(np.std(log_returns[-self.vol_window :])) * np.sqrt(252)
        hist_vol_mean = (
            float(
                np.mean(
                    [
                        np.std(log_returns[i : i + self.vol_window]) * np.sqrt(252)
                        for i in range(0, len(log_returns) - self.vol_window, self.vol_window)
                    ]
                )
            )
            if len(log_returns) > self.vol_window
            else current_vol
        )
        hist_vol_std = (
            float(
                np.std(
                    [
                        np.std(log_returns[i : i + self.vol_window]) * np.sqrt(252)
                        for i in range(0, len(log_returns) - self.vol_window, self.vol_window)
                    ]
                )
            )
            if len(log_returns) > self.vol_window
            else 1e-6
        )

        vol_zscore = (current_vol - hist_vol_mean) / (hist_vol_std + 1e-9)

        # Regime 判定
        is_uptrend = norm_slope > self.trend_threshold and ma_fast > ma_slow
        is_downtrend = norm_slope < -self.trend_threshold and ma_fast < ma_slow
        is_high_vol = vol_zscore > self.vol_high_threshold

        if is_uptrend and not is_high_vol:
            regime = Regime.BULL
            confidence = min(1.0, abs(norm_slope) / self.trend_threshold * 0.5 + 0.3)
        elif is_downtrend or is_high_vol:
            regime = Regime.BEAR
            confidence = min(1.0, max(abs(norm_slope) / self.trend_threshold, vol_zscore / 2) * 0.5 + 0.2)
        else:
            regime = Regime.SIDEWAYS
            confidence = 0.5

        return RegimeSnapshot(
            regime=regime,
            confidence=round(confidence, 3),
            volatility_zscore=round(vol_zscore, 3),
            trend_slope=round(norm_slope, 6),
            current_price=round(current_price, 4),
            ma_fast=round(ma_fast, 4),
            ma_slow=round(ma_slow, 4),
            annualized_vol=round(current_vol, 4),
        )

    def detect_history(self, prices: np.ndarray | pd.Series, step: int = 1) -> list[dict]:
        """对价格序列逐步检测 regime 历史。"""
        arr = np.asarray(prices, dtype=np.float64)
        min_required = self.slow_window + self.vol_lookback
        results = []
        for i in range(min_required, len(arr), step):
            snap = self.detect(arr[: i + 1])
            results.append(
                {
                    "index": i,
                    "regime": snap.regime.value,
                    "confidence": snap.confidence,
                    "vol_zscore": snap.volatility_zscore,
                    "trend_slope": snap.trend_slope,
                    "price": snap.current_price,
                }
            )
        return results
