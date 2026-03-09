"""
Multi-Factor Strategy Backtest Runner

Loads alpha factors from the zoo, runs a backtest on CSI 1000,
and prints a performance report.

Usage:
    python -m src.alpha_mining.backtest_runner
    python -m src.alpha_mining.backtest_runner --start 2023-01-01 --end 2025-01-01 --top_n 10
"""

import argparse
import pandas as pd
import numpy as np
from loguru import logger
from src.market_data.db import DB
from src.config.settings import get_broker_config
from src.core.engine import TradingEngine
from src.core.backtest_broker import BacktestBroker
from src.core.data_stream import DBDataStream
from src.strategies.multi_factor_strategy import MultiFactorStrategy

broker_config = get_broker_config()


def print_report(broker: BacktestBroker, start_date: str, end_date: str):
    """Print a comprehensive backtest performance report."""
    account = broker.get_account_info()
    equity_hist = broker.equity_history
    trades = broker.trades

    if not equity_hist:
        print("\nNo equity history available.")
        return

    # Build equity curve
    eq_df = pd.DataFrame(equity_hist)
    if "date" in eq_df.columns:
        eq_df["date"] = pd.to_datetime(eq_df["date"])
        eq_df = eq_df.set_index("date").sort_index()
    elif "timestamp" in eq_df.columns:
        eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"])
        eq_df = eq_df.set_index("timestamp").sort_index()

    equity_col = "total_equity" if "total_equity" in eq_df.columns else eq_df.columns[0]
    equity = eq_df[equity_col]

    initial_cash = broker.initial_cash
    final_equity = equity.iloc[-1] if len(equity) > 0 else initial_cash

    # Returns
    total_return = (final_equity / initial_cash - 1) * 100
    n_days = len(equity)
    n_years = n_days / 252 if n_days > 0 else 1
    annual_return = ((final_equity / initial_cash) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    # Daily returns
    daily_ret = equity.pct_change().dropna()
    if len(daily_ret) > 0:
        sharpe = (daily_ret.mean() / (daily_ret.std() + 1e-9)) * np.sqrt(252)
        max_dd = ((equity / equity.cummax()) - 1).min() * 100
        calmar = annual_return / abs(max_dd) if abs(max_dd) > 0.01 else 0
        win_rate = (daily_ret > 0).sum() / len(daily_ret) * 100
        volatility = daily_ret.std() * np.sqrt(252) * 100
    else:
        sharpe = max_dd = calmar = win_rate = volatility = 0

    # Trade stats
    n_trades = len(trades)
    buy_trades = [t for t in trades if t.get("type") == "buy"]
    sell_trades = [t for t in trades if t.get("type") == "sell"]

    # Print
    print("\n" + "=" * 70)
    print("       MULTI-FACTOR STRATEGY BACKTEST REPORT")
    print("=" * 70)
    print(f"  Period:           {start_date} → {end_date}")
    print(f"  Trading Days:     {n_days}")
    print(f"  Initial Capital:  ¥{initial_cash:,.2f}")
    print(f"  Final Equity:     ¥{final_equity:,.2f}")
    print("-" * 70)
    print(f"  Total Return:     {total_return:>8.2f}%")
    print(f"  Annual Return:    {annual_return:>8.2f}%")
    print(f"  Sharpe Ratio:     {sharpe:>8.2f}")
    print(f"  Max Drawdown:     {max_dd:>8.2f}%")
    print(f"  Calmar Ratio:     {calmar:>8.2f}")
    print(f"  Volatility:       {volatility:>8.2f}%")
    print(f"  Win Rate (daily): {win_rate:>8.2f}%")
    print("-" * 70)
    print(f"  Total Trades:     {n_trades}")
    print(f"    Buys:           {len(buy_trades)}")
    print(f"    Sells:          {len(sell_trades)}")
    print("-" * 70)

    # Positions
    positions = account.get("positions", {})
    if positions:
        print(f"\n  Current Positions ({len(positions)} stocks):")
        sorted_pos = sorted(positions.items(), key=lambda x: x[1], reverse=True)
        for sym, qty in sorted_pos[:10]:
            print(f"    {sym}: {qty} shares")
        if len(positions) > 10:
            print(f"    ... and {len(positions) - 10} more")

    print("=" * 70)

    # Monthly returns table
    if len(daily_ret) > 20:
        monthly = equity.resample("ME").last().pct_change().dropna() * 100
        if not monthly.empty:
            print("\n  Monthly Returns (%):")
            print("  " + "-" * 50)
            for date, ret in monthly.items():
                marker = "▲" if ret > 0 else "▼"
                print(f"    {date.strftime('%Y-%m')}: {ret:>7.2f}%  {marker}")
            print("  " + "-" * 50)

    print()


def run_backtest(
    start_date: str = "2024-01-01",
    end_date: str = "2025-01-01",
    initial_cash: float = 1_000_000.0,
    top_n: int = 10,
    rebalance_freq: str = "weekly",
    index_code: str = "000852",
    zoo_dir: str = "data/alpha_zoo",
):
    """Run a full multi-factor backtest."""

    logger.info(f"Starting Multi-Factor Backtest: {start_date} → {end_date}")
    logger.info(f"  top_n={top_n}, rebalance={rebalance_freq}, index={index_code}")

    db = DB()

    # 1. Get universe stocks
    stocks = db.get_index_stocks(index_code, start_date)
    if not stocks:
        stocks = db.get_index_stocks(f"sh.{index_code}", start_date)
    if not stocks:
        raise ValueError(f"No stocks found for index {index_code}")
    logger.info(f"Universe: {len(stocks)} stocks")

    # 2. Setup data stream
    stream = DBDataStream(
        db_client=db,
        symbols=stocks,
        start_date=start_date,
        end_date=end_date,
    )

    # 3. Setup broker
    broker = BacktestBroker(
        initial_cash=initial_cash,
        commission=broker_config.backtest.commission,
        slippage=broker_config.backtest.slippage,
        db_client=db,
    )

    # 4. Setup strategy
    strategy = MultiFactorStrategy(
        db_client=db,
        top_n=top_n,
        rebalance_freq=rebalance_freq,
        index_code=index_code,
        zoo_dir=zoo_dir,
    )

    # 5. Progress callback
    total_bars = stream.total_bars
    last_pct = [0]

    def on_step(bars):
        pct = int((stream.global_idx / total_bars) * 100)
        if pct >= last_pct[0] + 10:
            last_pct[0] = pct
            eq = broker.get_account_info()["total_equity"]
            date_str = next(iter(bars.values())).timestamp.strftime("%Y-%m-%d") if bars else "?"
            logger.info(f"  Progress: {pct}%  Date: {date_str}  Equity: ¥{eq:,.0f}")

    # 6. Create engine and run
    engine = TradingEngine(
        strategy=strategy,
        broker=broker,
        data_stream=stream,
        on_step=on_step,
    )

    logger.info("Running backtest...")
    engine.run()

    # 7. Report
    print_report(broker, start_date, end_date)

    return broker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Factor Strategy Backtest")
    parser.add_argument("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--cash", type=float, default=1_000_000, help="Initial cash")
    parser.add_argument("--top_n", type=int, default=10, help="Number of stocks to hold")
    parser.add_argument("--rebalance", default="weekly", choices=["weekly", "biweekly", "monthly"])
    parser.add_argument("--index", default="000852", help="Index code for universe")
    parser.add_argument("--zoo", default="data/alpha_zoo", help="Alpha zoo directory")
    args = parser.parse_args()

    run_backtest(
        start_date=args.start,
        end_date=args.end,
        initial_cash=args.cash,
        top_n=args.top_n,
        rebalance_freq=args.rebalance,
        index_code=args.index,
        zoo_dir=args.zoo,
    )
