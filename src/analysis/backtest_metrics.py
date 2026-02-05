"""
Unified backtest performance metrics calculation.
This module provides consistent metrics calculation for both backend and API responses.
The frontend TypeScript implementation should mirror these calculations.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
import math


@dataclass
class PerformanceMetrics:
    """Complete set of backtest performance metrics"""

    # Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    volatility: float = 0.0  # Annualized
    downside_deviation: float = 0.0

    # Risk-adjusted metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Trade statistics
    total_trades: int = 0
    win_rate: float = 0.0  # Based on daily PnL
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    # Period info
    trading_days: int = 0
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "volatility": self.volatility,
            "downside_deviation": self.downside_deviation,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "trading_days": self.trading_days,
            "start_date": self.start_date,
            "end_date": self.end_date,
        }


RISK_FREE_RATE = 0.02  # 2% annual risk-free rate
TRADING_DAYS_PER_YEAR = 252


def calculate_metrics(
    equity_history: List[Dict[str, Any]],
    trades: Optional[List[Dict[str, Any]]] = None,
    risk_free_rate: float = RISK_FREE_RATE,
) -> PerformanceMetrics:
    """
    Calculate comprehensive backtest performance metrics.

    Args:
        equity_history: List of equity points with 'timestamp', 'total_equity', 'daily_pnl'
        trades: Optional list of trade records
        risk_free_rate: Annual risk-free rate for Sharpe calculation

    Returns:
        PerformanceMetrics dataclass with all calculated metrics
    """
    if not equity_history or len(equity_history) < 2:
        return PerformanceMetrics()

    # Sort by timestamp
    sorted_equity = sorted(equity_history, key=lambda x: x.get("timestamp", ""))

    # Extract values
    equities = [e.get("total_equity", 0) for e in sorted_equity]
    daily_pnls = [e.get("daily_pnl", 0) for e in sorted_equity]
    timestamps = [e.get("timestamp", "") for e in sorted_equity]

    initial_equity = equities[0] if equities[0] > 0 else 1
    final_equity = equities[-1]

    # ========== Return Metrics ==========
    total_return = (final_equity - initial_equity) / initial_equity

    # Trading period
    try:
        start_date = datetime.fromisoformat(timestamps[0].split(" ")[0])
        end_date = datetime.fromisoformat(timestamps[-1].split(" ")[0])
        trading_days = max(1, (end_date - start_date).days)
    except (ValueError, IndexError):
        trading_days = len(sorted_equity)
        start_date = None
        end_date = None

    # Annualized return
    if trading_days > 0:
        annualized_return = (1 + total_return) ** (365 / trading_days) - 1
    else:
        annualized_return = 0.0

    # ========== Daily Returns ==========
    daily_returns = []
    for i in range(1, len(equities)):
        if equities[i - 1] > 0:
            daily_returns.append((equities[i] - equities[i - 1]) / equities[i - 1])

    if not daily_returns:
        return PerformanceMetrics(
            total_return=total_return,
            trading_days=trading_days,
            start_date=timestamps[0] if timestamps else None,
            end_date=timestamps[-1] if timestamps else None,
        )

    # ========== Risk Metrics ==========
    # Volatility (annualized)
    mean_return = sum(daily_returns) / len(daily_returns)
    variance = sum((r - mean_return) ** 2 for r in daily_returns) / len(daily_returns)
    std_dev = math.sqrt(variance)
    volatility = std_dev * math.sqrt(TRADING_DAYS_PER_YEAR)

    # Downside deviation (for Sortino)
    negative_returns = [r for r in daily_returns if r < 0]
    if negative_returns:
        downside_variance = sum(r**2 for r in negative_returns) / len(daily_returns)
        downside_deviation = math.sqrt(downside_variance) * math.sqrt(
            TRADING_DAYS_PER_YEAR
        )
    else:
        downside_deviation = 0.0

    # Max Drawdown
    peak = equities[0]
    max_drawdown = 0.0
    max_dd_duration = 0
    current_dd_start = 0
    in_drawdown = False

    for i, equity in enumerate(equities):
        if equity > peak:
            peak = equity
            if in_drawdown:
                dd_duration = i - current_dd_start
                max_dd_duration = max(max_dd_duration, dd_duration)
            in_drawdown = False
        else:
            if not in_drawdown:
                current_dd_start = i
                in_drawdown = True
            drawdown = (peak - equity) / peak
            max_drawdown = max(max_drawdown, drawdown)

    # Check final drawdown duration
    if in_drawdown:
        dd_duration = len(equities) - current_dd_start
        max_dd_duration = max(max_dd_duration, dd_duration)

    # ========== Risk-Adjusted Metrics ==========
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess_returns = [r - daily_rf for r in daily_returns]

    # Sharpe Ratio
    if std_dev > 0:
        sharpe_ratio = (
            (mean_return - daily_rf) / std_dev * math.sqrt(TRADING_DAYS_PER_YEAR)
        )
    else:
        sharpe_ratio = 0.0

    # Sortino Ratio
    if downside_deviation > 0:
        sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation
    else:
        sortino_ratio = 0.0 if annualized_return <= risk_free_rate else float("inf")

    # Calmar Ratio
    if max_drawdown > 0:
        calmar_ratio = annualized_return / max_drawdown
    else:
        calmar_ratio = 0.0 if annualized_return <= 0 else float("inf")

    # ========== Trade Statistics (based on daily P&L) ==========
    positive_pnls = [p for p in daily_pnls if p > 0]
    negative_pnls = [p for p in daily_pnls if p < 0]

    win_rate = len(positive_pnls) / len(daily_pnls) if daily_pnls else 0.0

    gross_profit = sum(positive_pnls)
    gross_loss = abs(sum(negative_pnls))

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else 0.0

    avg_win = sum(positive_pnls) / len(positive_pnls) if positive_pnls else 0.0
    avg_loss = sum(negative_pnls) / len(negative_pnls) if negative_pnls else 0.0

    # Clamp infinite values for JSON serialization
    profit_factor = min(profit_factor, 999.99)
    sortino_ratio = min(sortino_ratio, 999.99)
    calmar_ratio = min(calmar_ratio, 999.99)

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        max_drawdown=max_drawdown,
        max_drawdown_duration_days=max_dd_duration,
        volatility=volatility,
        downside_deviation=downside_deviation,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        total_trades=len(trades) if trades else 0,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        trading_days=trading_days,
        start_date=timestamps[0] if timestamps else None,
        end_date=timestamps[-1] if timestamps else None,
    )


def calculate_rolling_metrics(
    equity_history: List[Dict[str, Any]], window: int = 30
) -> List[Dict[str, Any]]:
    """
    Calculate rolling metrics over a sliding window.

    Args:
        equity_history: List of equity points
        window: Rolling window size in days

    Returns:
        List of rolling metrics for each day
    """
    if len(equity_history) < window:
        return []

    sorted_equity = sorted(equity_history, key=lambda x: x.get("timestamp", ""))

    rolling_metrics = []

    for i in range(window, len(sorted_equity)):
        window_data = sorted_equity[i - window : i + 1]
        metrics = calculate_metrics(window_data)

        rolling_metrics.append(
            {
                "timestamp": sorted_equity[i].get("timestamp"),
                "rolling_sharpe": metrics.sharpe_ratio,
                "rolling_volatility": metrics.volatility,
                "rolling_return": metrics.total_return,
                "rolling_max_dd": metrics.max_drawdown,
            }
        )

    return rolling_metrics
