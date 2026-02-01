import pandas as pd
import numpy as np
import talib as ta
from loguru import logger
from src.core.base import Strategy, Bar
import utils.utils as util

PRICE_CHANGE_LIMIT = 0.098
NUM_STOCKS = 6

class JSGStrategy(Strategy):
    def __init__(self, db_client, **kwargs):
        super().__init__()
        self.db_client = db_client
        
        # Load parameters with defaults
        params = self.get_parameters()
        self.max_stocks = kwargs.get('max_stocks', params['max_stocks']['default'])
        self.pool_size = kwargs.get('pool_size', params['pool_size']['default'])
        self.stock_sum = kwargs.get('stock_sum', params['stock_sum']['default'])
        
        self.black_industry_name = {"银行", "煤炭", "采掘", "钢铁"}
        
        # Initialize internal state from original Agent
        self.trad_days = pd.read_csv(
            "marked_trade_datas.csv", index_col="calendar_date", parse_dates=True
        )
        self.pass_month = []

    @classmethod
    def get_parameters(cls) -> dict:
        return {
            "max_stocks": {
                "type": "int",
                "default": NUM_STOCKS,
                "description": "Maximum number of stocks to consider from index",
                "min": 1,
                "max": 100
            },
            "pool_size": {
                "type": "int",
                "default": 20,
                "description": "Size of the candidate pool after financial filtering",
                "min": 5,
                "max": 100
            },
            "stock_sum": {
                "type": "int",
                "default": 10,
                "description": "Maximum number of stocks to hold in portfolio",
                "min": 1,
                "max": 20
            }
        }

    def on_bar(self, bars: dict[str, Bar]):
        # Get some representative bar for timestamp
        if not bars:
            return
        representative_bar = next(iter(bars.values()))
        ts = representative_bar.timestamp
        today_str = str(ts.date())

        # Check if it's the last trading day of the month (rebalance signal)
        if today_str not in self.trad_days.index.strftime('%Y-%m-%d'):
            return
        
        # Original logic check
        if self.trad_days.loc[today_str, "is_last_trading_day"] == 0:
            return

        logger.info(f"JSGStrategy: Rebalancing on {today_str}")

        # 1. Broad market check
        I = self.get_market_breadth(today_str)
        logger.info(f"Market breadth: {I}")

        # 2. Select stocks
        cand_stocks = self.stock_decider(I, today_str)
        
        # 3. Adjust positions
        self.adjust(cand_stocks, today_str)

    def get_market_breadth(self, end_date):
        stocks = self.db_client.get_index_stocks("000985", end_date)
        df = self.db_client.get_price(stocks, end_date, ["close"], 20)
        if len(df) == 0:
            return []

        df["ma20"] = df.groupby(level="code")["close"].transform(lambda x: ta.MA(x, timeperiod=20))
        df.dropna(inplace=True)
        df["bias"] = df["close"] > df["ma20"]

        if not df.empty and "date" in (df.index.names if hasattr(df.index, 'names') else []):
            df.reset_index(level="date", drop=True, inplace=True)

        industry_df = self.db_client.get_stock_industry_sw(df.index.to_list(), end_date)
        df["industry"] = industry_df["industry_name"]
        df = df[(df["industry"] != "")]
        df = df[["bias", "industry"]]

        ratio = (df.groupby("industry")["bias"].mean() * 100).round()
        return ratio.nlargest(1).index.to_list()

    def stock_decider(self, I, date_str):
        if (not self.black_industry_name.intersection(set(I))):
            return self.select_stock(date_str)
        return []

    def filter_basic(self, stocks, date_str):
        df = self.db_client.get_price(stocks, date_str, ["isST"], 1)
        if not df.empty and "date" in (df.index.names if hasattr(df.index, 'names') else []):
            df.reset_index(level="date", drop=True, inplace=True)
        # Assuming tradestatus is also in the fields if needed, 
        # but the original logic only asked for isST
        df = df[(df["isST"] == 0)]
        return df.index.to_list()

    def select_stock(self, date_str):
        stocks = self.db_client.get_index_stocks("399101", date_str)
        stocks = self.filter_basic(stocks, date_str)
        
        # Calculate fin_date as in original logic
        ts = pd.to_datetime(date_str)
        # This part might need the next trading day logic which is tricky in a standalone way
        # For now, we use a slightly simplified date for financial data
        fin_date = (ts - pd.DateOffset(days=1)).date()

        fin_db = self.db_client.get_stock_fincial(
            stocks, fields=["adjusted_profit_diff", "total_shares"], date=str(fin_date)
        )
        fin_db = fin_db[fin_db["adjusted_profit_diff"] > 0]
        
        df = self.db_client.get_price(fin_db.index.to_list(), date_str, ["close"], 1, price_adj=False)
        if not df.empty and "date" in (df.index.names if hasattr(df.index, 'names') else []):
            df.reset_index(level="date", drop=True, inplace=True)
        
        fin_db["close"] = df["close"]
        share_info = self.db_client.get_stock_shares_info(fin_db.index.to_list(), str(fin_date))
        fin_db["total_shares"] = share_info["total_shares"]
        fin_db["market_cap"] = fin_db["close"] * fin_db["total_shares"]
        
        fin_dbContent = fin_db.sort_values(by="market_cap", ascending=True).iloc[:self.pool_size]
        return fin_dbContent.index.to_list()

    def adjust(self, target_stocks, date_str):
        target = target_stocks[: min(len(target_stocks), self.stock_sum)]
        account = self.engine.broker.get_account_info()
        hold_list = list(account['positions'].keys())
        
        logger.info(f"JSG Rebalancing: Target {target}, Holding {hold_list}")
        
        # Sell stocks not in target
        for stock in hold_list:
            if stock not in target:
                qty = account['positions'][stock]
                if qty > 0:
                    self.sell(stock, qty)

        # Buy target stocks
        total_equity = account['total_equity']
        if target:
            val_per_stock = (total_equity * 0.95) / len(target) # 5% cash buffer
            for code in target:
                price_df = self.db_client.get_price(code, date_str, ["close"], 1)
                if price_df.empty:
                    continue
                price = price_df.iloc[0]["close"]
                target_qty = int(val_per_stock / price // 100 * 100) # 100 share lot
                
                curr_qty = account['positions'].get(code, 0)
                delta = target_qty - curr_qty
                if delta > 0:
                    self.buy(code, delta)
                elif delta < 0:
                    self.sell(code, abs(delta))
