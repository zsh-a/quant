import os
import sys
from loguru import logger
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import talib
import pickle
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler

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

from models import LeNet, TransformerModel

PRICE_CHANGE_LIMIT = 0.098
MAX_POSITION = 99999999

NUM_STOCKS = 6

factors = [
    "momentum_5",
    "vol_ratio",
    "RSI_14",
    "BB_upper",
    "BB_lower",
    "volatility_20",
    "turnover_avg_20",
    "obv",
    "turnover_5",
]


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
        self.stock_sum = 1

        self.trad_days = pd.read_csv(
            "marked_trade_datas.csv", index_col="calendar_date", parse_dates=True
        )
        # self.etf_pool = [
        #     # 境外
        #     "513100",  # 纳指ETF
        #     # '513500',  # 标普ETF
        #     # '164824',  # 印度基金
        #     # '513050',  # 中概互联
        #     "513520",  # 日经ETF
        #     "513030",  # 德国ETF
        #     # '513080',  # 法国ETF
        #     # 商品
        #     "518880",  # 黄金ETF
        #     "159980",  # 有色ETF
        #     "159985",  # 豆粕ETF
        #     "501018",  # 南方原油
        #     # 债券
        #     # "511010",  # 国债ETF
        #     # "511090",  # 30年国债ETF
        #     # 国内
        #     "513130",  # 恒生科技
        #     # '510050',  # 上证50ETF
        #     # '512100',  # 中证1000ETF
        # ]

        self.etf_pool = self.db_client.get_all_etf_code()

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

    def get_current_date_str(self):
        return str(self.current_date.date())

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

    def calc_factors(df):
        # 动量因子: 过去5日涨跌幅
        df["momentum_5"] = df["close"] / df["close"].shift(5) - 1

        # 成交量因子: (最近5日平均成交量) / (最近10日平均成交量) - 1

        df["vol_ratio"] = (df["volume"].rolling(5).mean()) / (
            df["volume"].rolling(10).mean()
        ) - 1
        # 计算RSI (默认周期14)
        df["RSI_14"] = talib.RSI(df["close"], timeperiod=14)

        # 布林带
        upper, middle, lower = talib.BBANDS(
            df["close"],
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2,
            matype=0,
        )
        df["BB_upper"] = upper
        df["BB_middle"] = middle
        df["BB_lower"] = lower

        # 反转因子 = -动量因子
        df["reversal_5"] = -df["close"].pct_change(periods=5)

        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
        df["volatility_20"] = df["log_ret"].rolling(20).std() * np.sqrt(252)

        df["turnover_avg_20"] = df["turn"].rolling(20).mean()

        df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).cumsum()

        df["turnover_5"] = df["turn"].rolling(5).sum()
        df["future_ret_1d"] = df["close"].pct_change(periods=1)

        return df

    def get_rank(self):
        total_value = self.market_env.account.get_total_value()

        df = self.db_client.get_price(
            self.etf_pool,
            self.get_current_date_str(),
            ["open", "high", "low", "close", "volume", "turn"],
            60,
        )

        fields = ["open", "high", "low", "close", "volume", "turn"]
        df = (
            df.groupby("code", group_keys=False)
            .apply(lambda df: Agent.calc_factors(df).iloc[-1])
            .dropna()
        )
        # load scaler
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        df = df[factors]
        X = scaler.transform(df)
        # load lgb model
        model = lgb.Booster(model_file='lgb_model.txt')
        pred = model.predict(X)
        df['pred_score'] = pred
        
        return df.sort_values('pred_score', ascending=False).index.tolist()

        # all_data = all_data.reset_index("date", drop=True)
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
            logger.info(f"target_value : {target_value}")
            self.process_order_value(target_value)
        # return target_value

    def action_decider(self, stocks_obs):
        # ts = stocks_obs[0].name
        # self.current_date = ts
        ts = self.market_env.cur_date
        today = str(ts.date())

        if self.trad_days.loc[today, "is_last_trading_day"] == 0:
            return

        stocks = self.get_rank()
        self.adjust(stocks)

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
    agent = Agent(None, db_client=None)
