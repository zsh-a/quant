import os
import sys
from loguru import logger
import numpy as np
import pandas as pd

from .base_policy import OrderPolicy

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 将外层目录添加到 sys.path
sys.path.append(parent_dir)
import global_var

from account import Account
from market_env import MultiMarketEnv
from order import Order, OrderManager
import indictor
import talib as ta
import utils.utils as util

PRICE_CHANGE_LIMIT = 0.098
MAX_POSITION = 99999999

NUM_STOCKS = 6


class OrderPolicy(OrderPolicy):
    def __init__(self, account, **args) -> None:
        self.last_obs = []
        self.cur_obs = None

        self.account: Account = account

        self.db_client = args["db_client"]
        self.slip = 0.002  # 改为百分比滑点，0.001表示0·.1%
        self.tracking = []

        self.running_in_day = False
        self.current_date = None

    def order_callback(self, order: Order, order_manager: OrderManager):
        pass

    def get_now_price(self, code, key):
        df = self.db_client.get_price(
            [code],
            str(self.current_date.date()),
            [key],
            1,
            str(self.current_date.date()),
        )
        if len(df) == 0:
            logger.error(
                f"can not find {code} price, current_date : {self.current_date}"
            )
            return 0

        df.reset_index(level="date", drop=True, inplace=True)
        open_price = df.loc[code, key]
        return open_price

    def buy_policy(self, order: Order):
        if not self.check_up_down_limit(order.symbol, order.exec_time):
            return (False, 0)

        if (
            len(self.account.get_position_price(str(self.current_date.date())))
            >= NUM_STOCKS
        ):
            return (False, 0)

        trading_price = self.get_now_price(order.symbol, order.exec_time) * (
            1 + self.slip
        )

        action = order.quantity
        min_action = self.account.min_action
        if action > min_action:
            if self.buy_cond(order.symbol):
                logger.info(f"price : {trading_price}")

                action = action // min_action * min_action
                num_stakes = min(
                    self.account.cash // trading_price // min_action * min_action,
                    action,
                )

                if num_stakes >= min_action:
                    amount = trading_price * num_stakes
                    cost = amount * self.account.trading_fee_open

                    while num_stakes > 0 and self.account.cash < amount + cost:
                        num_stakes -= min_action
                        amount = trading_price * num_stakes
                        cost = amount * self.account.trading_fee_open
                    if num_stakes > 0:
                        order.quantity = num_stakes
                        logger.info(
                            f"num_stakes : {num_stakes} {trading_price} : {self.account.cash}"
                        )
                        return (True, trading_price)

        return (False, trading_price)

    def buy_cond(self, code):
        # print(self.last_obs)
        if len(self.last_obs) < 2:
            return False
        # logger.info(
        #     f"buy cond | symbol : {code} | cur high : {self.cur_obs[idx]["high"]} | last high : {self.last_obs[-1][idx]["high"]}"
        # )
        return True
        # return self.cur_obs[idx]['high'] > self.last_obs[-1][idx]["high"]

    def check_up_down_limit(self, code, key):
        df = self.db_client.get_price(
            code, str(self.current_date.date()), ["open", "close"], 2
        )
        df.reset_index(level="code", drop=True, inplace=True)
        if len(df) == 0 or df.index[-1] != self.current_date:
            logger.error(f"{code} 停牌 {self.current_date}")
            return False

        today_price = df.iloc[-1][key]
        prev_close = df.iloc[0]["close"]

        if today_price / prev_close - 1 < -PRICE_CHANGE_LIMIT:
            logger.error(f"{code} 跌停 {self.current_date}")
            return False

        if today_price / prev_close - 1 > PRICE_CHANGE_LIMIT:
            logger.error(f"{code} 涨停 {self.current_date}")
            return False
        return True

    def sell_policy(self, order):
        action = order.quantity

        if not self.check_up_down_limit(order.symbol, order.exec_time):
            return (False, 0)

        trading_price = self.get_now_price(order.symbol, order.exec_time) * (
            1 - self.slip
        )
        min_action = self.account.min_action
        action = action // min_action * min_action
        if self.sell_cond(order.symbol):
            num_stakes = min(self.account.positions[-1][order.symbol].quantity, action)
            if num_stakes > 0:
                order.quantity = num_stakes
                return (True, trading_price)
        else:
            logger.info(
                f"order fail -> tracking | datetime : {self.current_date} | order_id : {order.order_id} | order_type : {order.order_type} "
            )
            if order.order_type == "sell":
                order.status = "tracking"
        return (False, trading_price)

    def sell_cond(self, code):
        if len(self.last_obs) < 2:
            return False
        if not self.account.availables[-1][code]:
            return False
        return True
        # print(self.cur_obs[code])

    def step(self, obs):
        if self.cur_obs:
            self.last_obs.append(self.cur_obs)
        if len(self.last_obs) > 2:
            self.last_obs = self.last_obs[-2:]
        self.cur_obs = obs
        self.current_date = obs[0].name

    def step_in_day(self, obs):
        self.cur_obs = obs
        self.running_in_day = True


class Agent:
    def __init__(self, market_env: MultiMarketEnv, **args) -> None:
        self.market_env = market_env

        self.db_client = args["db_client"]
        self.init_indicators()

        self.current_date = None
        self.pass_month = []
        self.pool_size = 20

        self.stock_sum = 4

        self.black_industry_name = {"银行", "煤炭", "采掘", "钢铁"}

        self.trad_days = pd.read_csv(
            "marked_trade_datas.csv", index_col="calendar_date", parse_dates=True
        )
        print(self.trad_days)

    def get_next_trading_day(self):
        date = pd.to_datetime(self.current_date)
        next_date = date + pd.DateOffset(days=1)
        while self.trad_days.loc[next_date, "is_trading_day"] == 0:
            next_date += pd.DateOffset(days=1)
        return next_date

    def step(self):
        self.current_date = self.market_env.cur_date
        self.next_trading_day = self.get_next_trading_day()

    def signal_indicator(df):
        return df

    def init_indicators(self):
        # self.market_env.add_indicator(Agent.signal_indicator)
        self.market_env.clean_data()

    def select_action(self, code, info):
        ret = 0
        score = 0

        return ret, score

    def get_market_breadth(self, end_date):
        stocks = self.db_client.get_index_stocks("000985", end_date)

        df = self.db_client.get_price(stocks, end_date, ["close"], 20)

        def calculate_ema(group):
            group["ma20"] = ta.MA(group["close"], timeperiod=20)
            return group

        df = df.groupby(level="code", group_keys=False).apply(calculate_ema)

        df.dropna(inplace=True)

        # df_ma20 = df.groupby(level="code", group_keys=False).apply(calculate_ema)
        # df_ma20 = df_ma20.groupby(level=0).tail(1).reset_index("date")['ma20']

        # def check_ma(group):
        #     code = group.index.get_level_values(0).unique().to_list()[0]
        #     group["bias"] = group['close'] > df_ma20[code]
        #     return group

        df["bias"] = df["close"] > df["ma20"]

        df.reset_index(level="date", drop=True, inplace=True)

        industry_df = self.db_client.get_stock_industry_sw(df.index.to_list(), end_date)
        df["industry"] = industry_df["industry_name"]
        # df['industry'] = self.db_client.get_stock_industry(df.index.to_list(), end_date)
        df = df[(df["industry"] != "")]
        df = df[["bias", "industry"]]

        ratio = (
            df.groupby("industry").sum() * 100 / df.groupby("industry").count()
        ).round()
        # if self.get_current_date_str() == '2022-06-02':
        #     ratio.sort_values(by="bias").to_csv("test.csv")

        #     breakpoint()
        # max_bias = ratio["bias"].max()
        # return ratio[ratio["bias"] == max_bias].index.to_list()

        return ratio["bias"].nlargest(1).index.to_list()

    def get_current_date_str(self):
        return str(self.current_date.date())

    def filter_basic(self, stocks):
        df = self.db_client.get_price(stocks, self.get_current_date_str(), ["isST"], 1)
        df.reset_index(level="date", drop=True, inplace=True)
        df = df[(df["tradestatus"] == 1) & (df["isST"] == 0)]

        # df = self.db_client.get_price(df.index.to_list(), self.get_next_trading_day().date(), ["isST"], 1)
        # df.reset_index(level="date", drop=True, inplace=True)
        # df.to_csv("0201.csv")
        # df = df[(df["tradestatus"] == 1)]
        return df.index.to_list()

    def select_stock(self, end_date):
        stocks = self.db_client.get_index_stocks("399101", end_date)
        stocks = self.filter_basic(stocks)
        fin_date = (self.get_next_trading_day() - pd.DateOffset(days=1)).date()

        fin_db = self.db_client.get_stock_fincial(
            stocks, fields=["adjusted_profit_diff","total_shares"],date=fin_date
        )
        fin_db = fin_db[fin_db["adjusted_profit_diff"] > 0]
        fin_db["total_shares_bak"] = fin_db["total_shares"]
        df = self.db_client.get_price(
            fin_db.index.to_list(), end_date, ["close"], 1, price_adj=False
        )
        df.reset_index(level="date", drop=True, inplace=True)

        fin_db["close"] = df["close"]

        df = self.db_client.get_stock_shares_info(fin_db.index.to_list(), fin_date)
        fin_db["total_shares"] = df["total_shares"]
        fin_db["market_cap"] = fin_db["close"] * fin_db["total_shares"]

        # if self.get_current_date_str() == "2025-04-14":
        #     fin_db.sort_values(by="market_cap", ascending=True).to_csv("tmp.csv")
        #     breakpoint()
        fin_db = fin_db.sort_values(by="market_cap", ascending=True).iloc[
            : self.pool_size
        ]

        return fin_db.index.to_list()

    def process_order_value(self, target_value):
        pos_df = self.market_env.account.get_position_price(self.get_current_date_str())
        if len(pos_df) > 0:
            pos_df["value"] = pos_df["position"] * pos_df["close"]

        new_price = self.db_client.get_price(
            list(target_value.keys()), self.get_current_date_str(), ["close"], 1
        )
        new_price.reset_index(level="date", drop=True, inplace=True)
        action_pos = {}
        for code, value in target_value.items():
            if code in pos_df.index:
                new_pos = int(value / pos_df.loc[code, "close"])
                action_pos[code] = new_pos - pos_df.loc[code, "position"]
            else:
                action_pos[code] = int(value / new_price.loc[code, "close"])

        for code, action in action_pos.items():
            self.create_order(code, action)

    def adjust(self, stocks):
        target = stocks[: min(len(stocks), self.stock_sum)]
        hold_list = list(self.market_env.account.positions[-1].keys())
        target_value = {}
        for stock in hold_list:
            if stock not in target:
                target_value[stock] = 0

        total_value = self.market_env.account.get_total_value()

        for code in target:
            if code not in hold_list:
                target_value[code] = total_value / len(target)

        if len(target_value) > 0:
            self.process_order_value(target_value)
        # return target_value

    def action_decider(self, stocks_obs):
        # ts = stocks_obs[0].name
        # self.current_date = ts
        ts = self.market_env.cur_date
        today = str(ts.date())

        if self.trad_days.loc[today, "is_last_trading_day"] == 0:
            return

        I = self.get_market_breadth(end_date=today)
        logger.info(f"market breath : {I}")
        cand_stocks = self.stock_decider(I)
        self.adjust(cand_stocks)

    def is_empty_month(self):
        return self.current_date.month in self.pass_month

    def stock_decider(self, I):
        today = self.get_current_date_str()
        # return self.select_stock(today)
        if (not self.black_industry_name.intersection(I)) and not self.is_empty_month():
            return self.select_stock(today)
        return []

    def run_end(self):
        pos = self.market_env.account.get_position_price(self.get_current_date_str())
        if len(pos) == 0:
            return
        hold_stocks = pos["code"].to_list()
        df = self.db_client.get_price(
            hold_stocks, self.get_current_date_str(), ["close", "open"], 3
        )

        df = df.groupby(level=0, group_keys=False).apply(lambda x: x.head(2))
        # logger.error(f"{df}")

        df["pct"] = df.groupby("code")["close"].pct_change()

        df = df.groupby(level=0).tail(1)
        df.reset_index("date", drop=True, inplace=True)
        banned_stocks = df[df["pct"] >= PRICE_CHANGE_LIMIT].index.to_list()

        logger.info(f"end check banned : {banned_stocks}")
        for stock in banned_stocks:
            self.create_order(stock, -MAX_POSITION, exec_time="close")

    def cancel_order(self, code):
        self.market_env.order_manager.cancel_order(code)

    def create_order(self, code, action, exec_time="open"):
        if action < -self.market_env.min_action:
            self.market_env.order_manager.create_order(
                code, "sell", abs(action), None, exec_time
            )
        if action > self.market_env.min_action:
            self.market_env.order_manager.create_order(
                code, "buy", abs(action), None, exec_time
            )

if __name__ == "__main__":
    agent = Agent(None)
