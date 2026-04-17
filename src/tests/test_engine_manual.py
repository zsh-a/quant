import logging
import os
import sys

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.core.backtest_broker import BacktestBroker
from src.core.base import Strategy
from src.core.data_stream import CSVDataStream
from src.core.engine import TradingEngine

logging.basicConfig(level=logging.INFO)


class MovingAverageStrategy(Strategy):
    def on_bar(self, bars):
        for symbol, bar in bars.items():
            account = self.engine.broker.get_account_info()
            # Simple logic: buy 100 shares if we have no position
            curr_pos = account["positions"].get(symbol, 0)
            if curr_pos == 0:
                print(f"[{bar.timestamp}] Buying {symbol} at {bar.close}")
                self.buy(symbol, 100)
            elif bar.close > bar.open * 1.05 and curr_pos > 0:
                print(f"[{bar.timestamp}] Selling {symbol} at {bar.close}")
                self.sell(symbol, 100)


def test_backtest():
    # Use 510880.csv from the project qfq data
    data_path = "data/qfq/510880.csv"
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found.")
        return

    stream = CSVDataStream({"510880": data_path}, start_date="2023-01-01")
    broker = BacktestBroker()
    strategy = MovingAverageStrategy()

    engine = TradingEngine(strategy, broker, stream)
    engine.run()

    print("\nBacktest Result:")
    print(broker.get_account_info())


if __name__ == "__main__":
    test_backtest()
