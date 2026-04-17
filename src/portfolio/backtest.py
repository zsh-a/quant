"""
Portfolio Backtest Engine - Run backtests with multiple strategies.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

from loguru import logger

from src.core.backtest_broker import BacktestBroker
from src.core.data_stream import DBDataStream
from src.portfolio.portfolio_manager import PortfolioManager


@dataclass
class PortfolioBacktestResult:
    """Portfolio backtest result"""
    portfolio_id: str
    start_date: str
    end_date: str
    initial_capital: float
    final_equity: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    strategy_results: Dict[str, Dict]
    weights_history: List[Dict]
    equity_history: List[Dict]
    trades: List[Dict]


class PortfolioBacktester:
    """
    Portfolio backtesting engine.

    Runs backtests with multiple strategies and combines their results.
    """

    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        initial_capital: float = 1000000.0
    ):
        self.portfolio_manager = portfolio_manager
        self.initial_capital = initial_capital

        # Per-strategy brokers
        self.brokers: Dict[str, BacktestBroker] = {}
        self.equity_history: List[Dict] = []
        self.weights_history: List[Dict] = []
        self.all_trades: List[Dict] = []

    def run(
        self,
        start_date: str,
        end_date: str,
        symbols: List[str]
    ) -> PortfolioBacktestResult:
        """Run portfolio backtest"""
        logger.info(f"Starting portfolio backtest: {start_date} to {end_date}")

        weights = self.portfolio_manager.get_weights()
        len(weights)

        # Initialize brokers with proportional capital
        for strategy_name, weight in weights.items():
            capital = self.initial_capital * weight
            self.brokers[strategy_name] = BacktestBroker(
                initial_capital=capital,
                commission_rate=0.0003
            )
            logger.debug(f"Strategy {strategy_name}: ${capital:,.0f} ({weight:.1%})")

        # Create data stream
        data_stream = DBDataStream(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )

        # Run backtest
        current_date = None

        for bar in data_stream:
            bar_date = bar.get('date', bar.get('timestamp', ''))[:10]

            # Check for rebalance
            if self.portfolio_manager.should_rebalance(bar_date):
                self._rebalance(bar_date)

            # Collect signals from all strategies
            signals = self.portfolio_manager.collect_signals(bar)

            # Process signals for each strategy
            for strategy_name, signal in signals.items():
                if signal is None:
                    continue

                broker = self.brokers.get(strategy_name)
                if broker is None:
                    continue

                # Execute signal
                self._execute_signal(strategy_name, signal, bar, broker)

            # Record equity
            if bar_date != current_date:
                current_date = bar_date
                self._record_equity(current_date)

        # Calculate final results
        return self._calculate_results(start_date, end_date)

    def _execute_signal(
        self,
        strategy_name: str,
        signal: Dict,
        bar: Dict,
        broker: BacktestBroker
    ):
        """Execute a strategy signal"""
        direction = signal.get('direction', 'hold')
        symbol = signal.get('symbol')
        price = bar.get('close', 0)

        if direction == 'hold' or not symbol or price <= 0:
            return

        # Calculate position size (10% of equity per trade)
        equity = broker.get_equity(bar.get('prices', {symbol: price}))
        position_value = equity * 0.10
        quantity = int(position_value / price / 100) * 100  # Round to 100 shares

        if quantity <= 0:
            return

        if direction == 'buy':
            order = broker.submit_order(
                symbol=symbol,
                side='buy',
                quantity=quantity,
                order_type='market',
                price=price,
                timestamp=bar.get('timestamp', '')
            )
            if order and order.get('status') == 'filled':
                trade = order.copy()
                trade['strategy'] = strategy_name
                self.all_trades.append(trade)

        elif direction == 'sell':
            # Close existing position
            position = broker.positions.get(symbol)
            if position and position.quantity > 0:
                order = broker.submit_order(
                    symbol=symbol,
                    side='sell',
                    quantity=position.quantity,
                    order_type='market',
                    price=price,
                    timestamp=bar.get('timestamp', '')
                )
                if order and order.get('status') == 'filled':
                    trade = order.copy()
                    trade['strategy'] = strategy_name
                    self.all_trades.append(trade)

    def _rebalance(self, current_date: str):
        """Rebalance portfolio"""
        self.portfolio_manager.rebalance(current_date)

        # Record weights
        weights = self.portfolio_manager.get_weights()
        self.weights_history.append({
            'date': current_date,
            'weights': weights.copy()
        })

    def _record_equity(self, current_date: str):
        """Record portfolio equity"""
        total_equity = 0
        strategy_equities = {}

        for strategy_name, broker in self.brokers.items():
            # Use last known prices
            equity = broker.get_equity({})
            strategy_equities[strategy_name] = equity
            total_equity += equity

        self.equity_history.append({
            'date': current_date,
            'total_equity': total_equity,
            'strategy_equities': strategy_equities
        })

        # Update strategy returns for weight calculation
        if len(self.equity_history) >= 2:
            prev_equity = self.equity_history[-2]['total_equity']
            if prev_equity > 0:
                for strategy_name, equity in strategy_equities.items():
                    prev_strat_equity = self.equity_history[-2]['strategy_equities'].get(strategy_name, 0)
                    if prev_strat_equity > 0:
                        daily_return = (equity - prev_strat_equity) / prev_strat_equity
                        self.portfolio_manager.update_strategy_return(strategy_name, daily_return)

    def _calculate_results(self, start_date: str, end_date: str) -> PortfolioBacktestResult:
        """Calculate backtest results"""
        if not self.equity_history:
            return PortfolioBacktestResult(
                portfolio_id='',
                start_date=start_date,
                end_date=end_date,
                initial_capital=self.initial_capital,
                final_equity=self.initial_capital,
                total_return=0,
                sharpe_ratio=0,
                max_drawdown=0,
                strategy_results={},
                weights_history=self.weights_history,
                equity_history=self.equity_history,
                trades=self.all_trades
            )

        final_equity = self.equity_history[-1]['total_equity']
        total_return = (final_equity - self.initial_capital) / self.initial_capital

        # Calculate Sharpe ratio
        returns = []
        for i in range(1, len(self.equity_history)):
            prev = self.equity_history[i-1]['total_equity']
            curr = self.equity_history[i]['total_equity']
            if prev > 0:
                returns.append((curr - prev) / prev)

        if returns:
            import numpy as np
            mean_ret = np.mean(returns)
            std_ret = np.std(returns)
            sharpe = (mean_ret * 252 - 0.03) / (std_ret * np.sqrt(252)) if std_ret > 0 else 0
        else:
            sharpe = 0

        # Calculate max drawdown
        peak = self.initial_capital
        max_dd = 0
        for eq in self.equity_history:
            equity = eq['total_equity']
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

        # Per-strategy results
        strategy_results = {}
        for strategy_name, broker in self.brokers.items():
            strategy_results[strategy_name] = {
                'final_equity': broker.get_equity({}),
                'weight': self.portfolio_manager.get_weights().get(strategy_name, 0),
                'trades': len([t for t in self.all_trades if t.get('strategy') == strategy_name])
            }

        return PortfolioBacktestResult(
            portfolio_id=f"portfolio_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_equity=final_equity,
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            strategy_results=strategy_results,
            weights_history=self.weights_history,
            equity_history=self.equity_history,
            trades=self.all_trades
        )
