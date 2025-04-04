import os
import sys
from loguru import logger
import numpy as np
from .base_policy import OrderPolicy

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 将外层目录添加到 sys.path
sys.path.append(parent_dir)
import global_var

from account import Account
from market_env import MultiMarketEnv
from order import Order, OrderManager

import indictor


class OrderPolicy(OrderPolicy):
    def __init__(self, account) -> None:
        self.last_obs = []
        self.cur_obs = None

        self.account: Account = account

        self.slip = 0.0015  # 改为百分比滑点，0.001表示0.1%
        self.tracking = []

        self.running_in_day = False

    def order_callback(self, order: Order, order_manager: OrderManager):
        if order.order_type == "buy":
            order_manager.create_order(order.symbol, "stop", order.quantity)
        pass

    def buy_policy(self, order: Order):
        action = order.quantity
        idx = global_var.SYMBOLS.index(order.symbol)
        obs = self.cur_obs[idx]
        trading_price = obs["open"]
        if action > self.account.min_action:
            if self.buy_cond(order.symbol):
                trading_price = (
                    max(self.last_obs[-1][idx]["high"], obs["open"]) * (1 + self.slip)
                )
                logger.info(f"price : {trading_price}")
                num_stakes = min(
                    self.account.capital // trading_price // 100 * 100, action
                )
                if num_stakes > 100 and sum(self.account.positions[-1]) == 0:
                    amount = trading_price * num_stakes
                    cost = amount * self.account.trading_fee_open

                    while num_stakes > 0 and self.account.capital < amount + cost:
                        num_stakes -= 100
                        amount = trading_price * num_stakes
                        cost = amount * self.account.trading_cost_bps
                    if num_stakes > 0:
                        order.quantity = num_stakes
                        return (True, trading_price)

                    logger.info(f"num_stakes : {num_stakes} {trading_price} : {self.account.capital}")
            else:
                # logger.info(
                #     f"order fail -> tracking | datetime : {obs.name} | symbol : {order.symbol} | order_id : {order.order_id} | order_type : {order.order_type} | cur : {self.cur_obs[idx]["close"]} | cur high : {self.cur_obs[idx]["high"]} | last high : {self.last_obs[-1][idx]["high"]}"
                # )
                order.status = "tracking"

        return (False, trading_price)

    def buy_cond(self, code):
        idx = global_var.SYMBOLS.index(code)
        key = "close" if self.running_in_day else "high"
        # print(self.last_obs)
        if len(self.last_obs) < 2:
            return False
        logger.info(
            f"buy cond | symbol : {code} | cur high : {self.cur_obs[idx]["high"]} | last high : {self.last_obs[-1][idx]["high"]}"
        )
        return self.cur_obs[idx]['high'] > self.last_obs[-1][idx]["high"]

    def sell_policy(self, order):
        action = order.quantity
        idx = global_var.SYMBOLS.index(order.symbol)
        obs = self.cur_obs[idx]
        trading_price = obs["open"]
        if self.sell_cond(order.symbol):
            num_stakes = min(self.account.positions[-1][idx], action)
            prev_low = min(self.last_obs[-1][idx]["low"],self.last_obs[-2][idx]["low"])
            if num_stakes > 0:
                trading_price = (
                    prev_low * (1 - self.slip)
                )
                order.quantity = num_stakes
                return (True, trading_price)
        else:
            logger.info(
                f"order fail -> tracking | datetime : {obs.name} | order_id : {order.order_id} | order_type : {order.order_type} "
            )
            if order.order_type == "sell":
                order.status = "tracking"
        return (False, trading_price)

    def sell_cond(self, code):
        if len(self.last_obs) < 2:
            return False
        idx = global_var.SYMBOLS.index(code)
        key = 'low'
        logger.info(
            f"sell cond | avail : {self.account.get_available(code)} | cur low : {self.cur_obs[idx]["low"]} | cur close : {self.cur_obs[idx]["close"]} last low : {self.last_obs[-1][idx]["low"]}, {self.last_obs[-2][idx]["low"]})"
        )
        # print(self.cur_obs[code])
        return self.account.get_available(code) > 0 and (
            self.cur_obs[idx][key]
            < min(self.last_obs[-1][idx]["low"], self.last_obs[-2][idx]["low"])
            # and self.cur_obs[idx][key] < self.cur_obs[idx]["ema10"]
        )

    def step(self, obs):
        if self.cur_obs:
            self.last_obs.append(self.cur_obs)
        if len(self.last_obs) > 2:
            self.last_obs = self.last_obs[-2:]
        self.cur_obs = obs

    def step_in_day(self, obs):
        self.cur_obs = obs
        self.running_in_day = True


class Agent:
    def __init__(self, market_env: MultiMarketEnv) -> None:
        self.market_env = market_env
        self.init_indicators()

    def signal_indicator(df):
        # 计算均线斜率（20日均线）
        df['MA20_slope'] = df['ema_20'].diff().rolling(2).mean()
        # 判断支撑条件
        # df['support_condition'] = (abs(df['close'] - df['ema_20']) / df['ema_20'] < 0.05)  & (df["ema_30"] > df["ema_60"])

        # 计算成交量均量
        df['vol_ma10'] = df['volume'].rolling(10).mean()
        # 缩量条件
        df['low_volume'] = df['volume'] < 0.5 * df['vol_ma10'].shift(5)
        # 综合信号
        df['buy_signal'] = (df['close'] > 0)
        return df

    def init_indicators(self):
        self.market_env.add_indicator(Agent.signal_indicator)
        self.market_env.clean_data()

    def select_action(self, code, info):
        ret = 0
        if info['buy_signal']:
            return (1,0)
        return ret, 0

    def action_decider(self, stocks_obs):
        return [
            {"idx": i, "info": self.select_action(global_var.SYMBOLS[i], stocks_obs[i])}
            for i in range(len(stocks_obs))
        ]

    def stock_decider(self, actions):
        buy_list = [v for v in actions if v["info"][0] == 1]
        sell_list = [v for v in actions if v["info"][0] == -1]
        buy_list = sorted(buy_list, key=lambda x: x["info"][1])
        if len(buy_list) > 0:
            logger.info(f"buy list {buy_list}")
            self.create_order(code=global_var.SYMBOLS[buy_list[0]["idx"]], action=1)
        # if len(sell_list) > 0:
        #     logger.info(f"sell list {sell_list}")
        #     for sell in sell_list:
        #         self.cancel_order(global_var.SYMBOLS[sell["idx"]])

    def cancel_order(self, code):
        self.market_env.order_manager.cancel_order(code)

    def create_order(self, code, action):
        action = int(action * self.market_env.max_stake)
        if action < -self.market_env.min_action:
            self.market_env.order_manager.create_order(code, "sell", abs(action))
        if action > self.market_env.min_action:
            self.market_env.order_manager.create_order(code, "buy", abs(action))

 