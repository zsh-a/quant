"""
Attribution Analysis - Analyze return and risk sources.
Simple attribution by asset and sector.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from loguru import logger


@dataclass
class AttributionResult:
    """Attribution analysis result"""
    total_return: float
    by_asset: Dict[str, float]
    by_sector: Dict[str, float]
    by_period: Dict[str, float]  # monthly
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float


class ReturnAttribution:
    """
    Return attribution analysis.

    Decomposes portfolio returns by asset, sector, and time period.
    """

    # Simple sector mapping (can be extended)
    SECTOR_MAP = {
        # Technology
        '300750': 'Technology', '002475': 'Technology', '300059': 'Technology',
        # Consumer
        '000858': 'Consumer', '600519': 'Consumer', '000568': 'Consumer',
        # Finance
        '600036': 'Finance', '601318': 'Finance', '600030': 'Finance',
        # Healthcare
        '300760': 'Healthcare', '600276': 'Healthcare', '000538': 'Healthcare',
        # Industrial
        '601888': 'Industrial', '000333': 'Industrial', '002594': 'Industrial',
        # Energy
        '601857': 'Energy', '600028': 'Energy', '601225': 'Energy',
    }

    def __init__(self, trades: List[Dict], equity_history: List[Dict]):
        self.trades = trades
        self.equity_history = equity_history

    def analyze(self) -> AttributionResult:
        """Run full attribution analysis"""
        logger.info("Running attribution analysis")

        # Calculate returns by asset
        by_asset = self._attribution_by_asset()

        # Calculate returns by sector
        by_sector = self._attribution_by_sector()

        # Calculate returns by period
        by_period = self._attribution_by_period()

        # Calculate trade statistics
        trade_stats = self._calculate_trade_stats()

        # Total return
        if self.equity_history and len(self.equity_history) >= 2:
            initial = self.equity_history[0].get('total_equity', 1)
            final = self.equity_history[-1].get('total_equity', 1)
            total_return = (final - initial) / initial if initial > 0 else 0
        else:
            total_return = 0

        return AttributionResult(
            total_return=total_return,
            by_asset=by_asset,
            by_sector=by_sector,
            by_period=by_period,
            win_rate=trade_stats['win_rate'],
            avg_win=trade_stats['avg_win'],
            avg_loss=trade_stats['avg_loss'],
            profit_factor=trade_stats['profit_factor']
        )

    def _attribution_by_asset(self) -> Dict[str, float]:
        """Calculate P&L contribution by asset"""
        pnl_by_asset = {}

        # Track positions
        positions = {}  # symbol -> [cost_basis, quantity]

        for trade in self.trades:
            symbol = trade.get('symbol', '')
            side = trade.get('type', trade.get('side', ''))
            quantity = trade.get('quantity', 0)
            price = trade.get('price', 0)

            if symbol not in pnl_by_asset:
                pnl_by_asset[symbol] = 0

            if symbol not in positions:
                positions[symbol] = {'cost': 0, 'quantity': 0}

            if side.lower() == 'buy':
                # Add to position
                pos = positions[symbol]
                total_cost = pos['cost'] * pos['quantity'] + price * quantity
                total_qty = pos['quantity'] + quantity
                if total_qty > 0:
                    pos['cost'] = total_cost / total_qty
                pos['quantity'] = total_qty

            elif side.lower() == 'sell':
                # Calculate P&L
                pos = positions[symbol]
                if pos['quantity'] > 0:
                    pnl = (price - pos['cost']) * min(quantity, pos['quantity'])
                    pnl_by_asset[symbol] += pnl
                    pos['quantity'] -= quantity

        return pnl_by_asset

    def _attribution_by_sector(self) -> Dict[str, float]:
        """Calculate P&L contribution by sector"""
        by_asset = self._attribution_by_asset()
        by_sector = {}

        for symbol, pnl in by_asset.items():
            # Extract symbol code (remove exchange prefix)
            code = symbol.split('.')[-1] if '.' in symbol else symbol
            sector = self.SECTOR_MAP.get(code, 'Other')

            if sector not in by_sector:
                by_sector[sector] = 0
            by_sector[sector] += pnl

        return by_sector

    def _attribution_by_period(self) -> Dict[str, float]:
        """Calculate returns by month"""
        by_period = {}

        if len(self.equity_history) < 2:
            return by_period

        # Group equity by month
        monthly_data = {}
        for eq in self.equity_history:
            date = eq.get('date', eq.get('timestamp', ''))[:7]  # YYYY-MM
            if date not in monthly_data:
                monthly_data[date] = []
            monthly_data[date].append(eq.get('total_equity', 0))

        # Calculate monthly returns
        prev_equity = None
        for month in sorted(monthly_data.keys()):
            equities = monthly_data[month]
            if not equities:
                continue

            end_equity = equities[-1]
            if prev_equity is not None and prev_equity > 0:
                monthly_return = (end_equity - prev_equity) / prev_equity
                by_period[month] = monthly_return

            prev_equity = end_equity

        return by_period

    def _calculate_trade_stats(self) -> Dict:
        """Calculate trade statistics"""
        wins = []
        losses = []

        # Pair buy/sell trades
        open_positions = {}  # symbol -> buy_price

        for trade in self.trades:
            symbol = trade.get('symbol', '')
            side = trade.get('type', trade.get('side', ''))
            price = trade.get('price', 0)

            if side.lower() == 'buy':
                open_positions[symbol] = price

            elif side.lower() == 'sell' and symbol in open_positions:
                buy_price = open_positions[symbol]
                pnl_pct = (price - buy_price) / buy_price if buy_price > 0 else 0

                if pnl_pct > 0:
                    wins.append(pnl_pct)
                else:
                    losses.append(pnl_pct)

                del open_positions[symbol]

        n_trades = len(wins) + len(losses)
        win_rate = len(wins) / n_trades if n_trades > 0 else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        # Profit factor
        total_win = sum(wins) if wins else 0
        total_loss = abs(sum(losses)) if losses else 0.001
        profit_factor = total_win / total_loss if total_loss > 0 else 0

        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }


class RiskAttribution:
    """
    Risk attribution analysis.

    Analyzes risk contribution from different positions.
    """

    def __init__(self, equity_history: List[Dict], positions: Dict):
        self.equity_history = equity_history
        self.positions = positions

    def analyze(self) -> Dict:
        """Run risk attribution"""
        logger.info("Running risk attribution")

        # Calculate overall risk metrics
        returns = self._calculate_returns()

        if len(returns) < 5:
            return {
                'volatility': 0,
                'max_drawdown': 0,
                'var_95': 0,
                'cvar_95': 0,
                'sharpe_ratio': 0
            }

        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252)

        # Max drawdown
        max_dd = self._calculate_max_drawdown()

        # VaR and CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean([r for r in returns if r <= var_95])

        # Sharpe ratio
        mean_ret = np.mean(returns)
        sharpe = (mean_ret * 252 - 0.03) / volatility if volatility > 0 else 0

        return {
            'volatility': volatility,
            'max_drawdown': max_dd,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'sharpe_ratio': sharpe
        }

    def _calculate_returns(self) -> List[float]:
        """Calculate daily returns"""
        returns = []
        for i in range(1, len(self.equity_history)):
            prev = self.equity_history[i-1].get('total_equity', 0)
            curr = self.equity_history[i].get('total_equity', 0)
            if prev > 0:
                returns.append((curr - prev) / prev)
        return returns

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.equity_history:
            return 0

        peak = self.equity_history[0].get('total_equity', 1)
        max_dd = 0

        for eq in self.equity_history:
            equity = eq.get('total_equity', 0)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        return max_dd
