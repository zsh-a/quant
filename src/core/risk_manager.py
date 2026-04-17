"""
Risk management system for trading platform.
Handles position limits, stop-loss, take-profit, and risk metrics.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RiskLimits:
    """Risk limit configuration"""

    max_position_pct: float = 0.1  # Max 10% per position
    max_total_position: float = 0.95  # Max 95% total position
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.15  # 15% take profit
    max_daily_loss_pct: float = 0.10  # Max 10% daily loss
    max_drawdown_pct: float = 0.20  # Max 20% drawdown


@dataclass
class RiskMetrics:
    """Risk metrics for a portfolio"""

    total_exposure: float = 0.0
    position_count: int = 0
    largest_position_pct: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: Optional[float] = None
    var_95: Optional[float] = None  # Value at Risk 95%


class RiskManager:
    """Risk management system"""

    def __init__(self, initial_capital: float = 1000000.0, enabled: bool = True):
        self.enabled = enabled

        self.limits = RiskLimits()

        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.daily_start_capital = initial_capital

        # Tracking
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.alerts: List[Dict[str, Any]] = []

        logger.info(
            f"Risk manager initialized: enabled={self.enabled}, initial_capital={initial_capital}, limits={self.limits}"
        )

    def check_position_limit(self, symbol: str, quantity: float, price: float) -> tuple[bool, Optional[str]]:
        """Check if a new position would exceed limits"""
        if not self.enabled:
            return True, None

        position_value = quantity * price
        position_pct = position_value / self.current_capital

        # Check single position limit
        if position_pct > self.limits.max_position_pct:
            reason = f"Position size {position_pct:.1%} exceeds limit {self.limits.max_position_pct:.1%}"
            logger.warning(f"Position limit exceeded: {symbol} - {reason}")
            return False, reason

        # Check total exposure (including new position)
        total_exposure = sum(
            pos["quantity"] * pos.get("current_price", pos["entry_price"]) for pos in self.positions.values()
        )
        new_total_exposure = total_exposure + position_value
        total_exposure_pct = new_total_exposure / self.current_capital

        if total_exposure_pct > self.limits.max_total_position:
            reason = f"Total exposure {total_exposure_pct:.1%} would exceed limit {self.limits.max_total_position:.1%}"
            logger.warning(f"Total exposure limit exceeded: {reason}")
            return False, reason

        return True, None

    def check_stop_loss(self, symbol: str, current_price: float) -> tuple[bool, Optional[str]]:
        """Check if stop loss should be triggered"""
        if not self.enabled or symbol not in self.positions:
            return False, None

        position = self.positions[symbol]
        entry_price = position["entry_price"]

        # Calculate loss percentage
        loss_pct = (entry_price - current_price) / entry_price

        if loss_pct >= self.limits.stop_loss_pct:
            reason = f"Stop loss triggered: {loss_pct:.1%} loss (limit: {self.limits.stop_loss_pct:.1%})"
            logger.warning(f"Stop loss: {symbol} - {reason}")
            self._add_alert("stop_loss", symbol, reason)
            return True, reason

        return False, None

    def check_take_profit(self, symbol: str, current_price: float) -> tuple[bool, Optional[str]]:
        """Check if take profit should be triggered"""
        if not self.enabled or symbol not in self.positions:
            return False, None

        position = self.positions[symbol]
        entry_price = position["entry_price"]

        # Calculate profit percentage
        profit_pct = (current_price - entry_price) / entry_price

        if profit_pct >= self.limits.take_profit_pct:
            reason = f"Take profit triggered: {profit_pct:.1%} profit (target: {self.limits.take_profit_pct:.1%})"
            logger.info(f"Take profit: {symbol} - {reason}")
            self._add_alert("take_profit", symbol, reason)
            return True, reason

        return False, None

    def check_daily_loss_limit(self) -> tuple[bool, Optional[str]]:
        """Check if daily loss limit is exceeded"""
        if not self.enabled:
            return False, None

        daily_pnl = self.current_capital - self.daily_start_capital
        daily_loss_pct = abs(daily_pnl) / self.daily_start_capital if daily_pnl < 0 else 0

        if daily_loss_pct >= self.limits.max_daily_loss_pct:
            reason = f"Daily loss limit exceeded: {daily_loss_pct:.1%} (limit: {self.limits.max_daily_loss_pct:.1%})"
            logger.error(f"Daily loss limit: {reason}")
            self._add_alert("daily_loss_limit", "portfolio", reason)
            return True, reason

        return False, None

    def check_max_drawdown(self) -> tuple[bool, Optional[str]]:
        """Check if maximum drawdown is exceeded"""
        if not self.enabled:
            return False, None

        drawdown = self.peak_capital - self.current_capital
        drawdown_pct = drawdown / self.peak_capital

        if drawdown_pct >= self.limits.max_drawdown_pct:
            reason = f"Max drawdown exceeded: {drawdown_pct:.1%} (limit: {self.limits.max_drawdown_pct:.1%})"
            logger.error(f"Max drawdown: {reason}")
            self._add_alert("max_drawdown", "portfolio", reason)
            return True, reason

        return False, None

    def add_position(self, symbol: str, quantity: float, entry_price: float):
        """Add a new position"""
        self.positions[symbol] = {
            "symbol": symbol,
            "quantity": quantity,
            "entry_price": entry_price,
            "current_price": entry_price,
            "entry_time": datetime.now().isoformat(),
            "pnl": 0.0,
            "pnl_pct": 0.0,
        }
        logger.info(f"Position added: {symbol} qty={quantity} @ {entry_price}")

    def update_position(self, symbol: str, current_price: float):
        """Update position with current price"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        position["current_price"] = current_price

        # Calculate P&L
        pnl = (current_price - position["entry_price"]) * position["quantity"]
        pnl_pct = (current_price - position["entry_price"]) / position["entry_price"]

        position["pnl"] = pnl
        position["pnl_pct"] = pnl_pct

    def close_position(self, symbol: str, exit_price: float) -> Optional[Dict[str, Any]]:
        """Close a position"""
        if symbol not in self.positions:
            return None

        position = self.positions.pop(symbol)

        # Calculate final P&L
        pnl = (exit_price - position["entry_price"]) * position["quantity"]
        pnl_pct = (exit_price - position["entry_price"]) / position["entry_price"]

        # Update capital
        self.current_capital += pnl

        # Update peak capital
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        logger.info(f"Position closed: {symbol} pnl={pnl:.2f} ({pnl_pct:.2%})")

        return {
            "symbol": symbol,
            "quantity": position["quantity"],
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
        }

    def calculate_metrics(self) -> RiskMetrics:
        """Calculate current risk metrics"""
        total_exposure = 0.0
        largest_position_pct = 0.0

        for position in self.positions.values():
            position_value = position["quantity"] * position["current_price"]
            total_exposure += position_value

            position_pct = position_value / self.current_capital
            if position_pct > largest_position_pct:
                largest_position_pct = position_pct

        # Daily P&L
        daily_pnl = self.current_capital - self.daily_start_capital
        daily_pnl_pct = daily_pnl / self.daily_start_capital

        # Drawdown
        drawdown = self.peak_capital - self.current_capital
        drawdown_pct = drawdown / self.peak_capital if self.peak_capital > 0 else 0

        return RiskMetrics(
            total_exposure=total_exposure,
            position_count=len(self.positions),
            largest_position_pct=largest_position_pct,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            max_drawdown=drawdown,
            max_drawdown_pct=drawdown_pct,
        )

    def _add_alert(self, alert_type: str, symbol: str, message: str):
        """Add a risk alert"""
        alert = {
            "type": alert_type,
            "symbol": symbol,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }
        self.alerts.append(alert)

        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]

    def get_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        return self.alerts[-limit:]

    def reset_daily(self):
        """Reset daily tracking"""
        self.daily_start_capital = self.current_capital
        logger.info(f"Daily reset: capital={self.current_capital}")

    def get_status(self) -> Dict[str, Any]:
        """Get risk manager status"""
        metrics = self.calculate_metrics()

        return {
            "enabled": self.enabled,
            "current_capital": self.current_capital,
            "peak_capital": self.peak_capital,
            "limits": {
                "max_position_pct": self.limits.max_position_pct,
                "max_total_position": self.limits.max_total_position,
                "stop_loss_pct": self.limits.stop_loss_pct,
                "take_profit_pct": self.limits.take_profit_pct,
                "max_daily_loss_pct": self.limits.max_daily_loss_pct,
                "max_drawdown_pct": self.limits.max_drawdown_pct,
            },
            "metrics": {
                "total_exposure": metrics.total_exposure,
                "position_count": metrics.position_count,
                "largest_position_pct": metrics.largest_position_pct,
                "daily_pnl": metrics.daily_pnl,
                "daily_pnl_pct": metrics.daily_pnl_pct,
                "max_drawdown": metrics.max_drawdown,
                "max_drawdown_pct": metrics.max_drawdown_pct,
            },
            "positions": list(self.positions.values()),
            "recent_alerts": self.get_alerts(5),
        }
