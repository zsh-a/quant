import os
import sys
from loguru import logger
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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

        self.slip = 0.002  # 改为百分比滑点，0.001表示0·.1%
        self.tracking = []

        self.running_in_day = False

    def order_callback(self, order: Order, order_manager: OrderManager):
        pass

    def buy_policy(self, order: Order):
        action = order.quantity
        idx = global_var.SYMBOLS.index(order.symbol)
        obs = self.cur_obs[idx]
        trading_price = obs["open"] * (1 + self.slip)

        min_action = self.account.min_action
        if action > min_action:
            if self.buy_cond(order.symbol):
                logger.info(f"price : {trading_price}")
                num_stakes = min(
                    self.account.capital // trading_price // min_action * min_action,
                    action,
                )
                if num_stakes >= min_action and sum(self.account.positions[-1]) == 0:
                    amount = trading_price * num_stakes
                    cost = amount * self.account.trading_fee_open

                    while num_stakes > 0 and self.account.capital < amount + cost:
                        num_stakes -= min_action
                        amount = trading_price * num_stakes
                        cost = amount * self.account.trading_fee_open
                    if num_stakes > 0:
                        order.quantity = num_stakes
                        return (True, trading_price)

                    logger.info(
                        f"num_stakes : {num_stakes} {trading_price} : {self.account.capital}"
                    )
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
        # logger.info(
        #     f"buy cond | symbol : {code} | cur high : {self.cur_obs[idx]["high"]} | last high : {self.last_obs[-1][idx]["high"]}"
        # )
        return True
        # return self.cur_obs[idx]['high'] > self.last_obs[-1][idx]["high"]

    def sell_policy(self, order):
        action = order.quantity
        idx = global_var.SYMBOLS.index(order.symbol)
        obs = self.cur_obs[idx]
        trading_price = obs["open"] * (1 - self.slip)
        if self.sell_cond(order.symbol):
            num_stakes = min(self.account.positions[-1][idx], action)
            if num_stakes > 0:
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
        key = "low"
        # logger.info(
        #     f"sell cond | avail : {self.account.get_available(code)} | cur low : {self.cur_obs[idx]["low"]} | cur close : {self.cur_obs[idx]["close"]} last low : {self.last_obs[-1][idx]["low"]}, {self.last_obs[-2][idx]["low"]})"
        # )
        return True
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
        # RSRS策略实现
        # 参数设置
        N = 18  # 计算最近N天的线性回归斜率
        M = 600  # 计算标准化RSRS指标的历史N日
        S = 5  # 计算斜率的移动平均

        def calculate_beta(df, window=N):
            if df.shape[0] < window:
                return np.nan,np.nan
            x = df['low'].values
            y = df['high'].values
            lr = LinearRegression().fit(x.reshape(-1, 1), y)
            y_pred = lr.predict(x.reshape(-1, 1))
            beta = lr.coef_[0]
            r2 = r2_score(y, y_pred)
            return beta,r2
        tup_list = [calculate_beta(df,window=N) for df in df.rolling(N)]

        df['beta'] = [v[0] for v in tup_list] 
        df['r2'] = [v[1] for v in tup_list] 

        df['std_score'] = (df['beta'] - df['beta'].rolling(M).mean())/df['beta'].rolling(M).std()

        df['mdf_std_score'] = df['r2'] * df['std_score']
        df['rsk_std_score'] = df['beta'] * df['mdf_std_score']

        # 生成买入卖出信号
        # 买入信号：RSRS右偏移动平均大于0.7
        # 卖出信号：RSRS右偏移动平均小于-0.7
        df["signal"] = 0
        df.loc[df["std_score"] > 0.7, "signal"] = 1
        df.loc[df["std_score"] < -0.7, "signal"] = -1
        # df.loc[df["beta"] > 1, "signal"] = 1
        # df.loc[df["beta"] < 0.8, "signal"] = -1
        return df

    def init_indicators(self):
        self.market_env.add_indicator(Agent.signal_indicator)
        self.market_env.clean_data()

    def select_action(self, code, info):
        ret = 0
        score = 0

        # 根据RSRS策略的买入信号生成买入决策
        if info["signal"] == 1:
            ret = 1
            # 使用RSRS右偏值作为买入评分，评分越高优先级越高
            score = info["rsrs_right_ma"] if "rsrs_right_ma" in info else 0

        # 根据RSRS策略的卖出信号生成卖出决策
        elif info["signal"] == -1:
            ret = -1
            # 使用RSRS右偏值的绝对值作为卖出评分
            score = abs(info["rsrs_right_ma"]) if "rsrs_right_ma" in info else 0

        return ret, score

    def action_decider(self, stocks_obs):
        return [
            {"idx": i, "info": self.select_action(global_var.SYMBOLS[i], stocks_obs[i])}
            for i in range(len(stocks_obs))
        ]

    def stock_decider(self, actions):
        buy_list = [v for v in actions if v["info"][0] == 1]
        sell_list = [v for v in actions if v["info"][0] == -1]

        # 按评分排序买入列表和卖出列表
        buy_list = sorted(
            buy_list, key=lambda x: x["info"][1], reverse=True
        )  # 评分高的优先买入
        sell_list = sorted(
            sell_list, key=lambda x: x["info"][1], reverse=True
        )  # 评分高的优先卖出

        # 处理买入信号
        if len(buy_list) > 0:
            logger.info(f"buy list {buy_list}")
            self.create_order(code=global_var.SYMBOLS[buy_list[0]["idx"]], action=1)

        # 处理卖出信号
        if len(sell_list) > 0:
            logger.info(f"sell list {sell_list}")
            for sell in sell_list:
                # 创建卖出订单
                self.create_order(code=global_var.SYMBOLS[sell["idx"]], action=-1)

    def cancel_order(self, code):
        self.market_env.order_manager.cancel_order(code)

    def create_order(self, code, action):
        action = int(action * self.market_env.max_stake)
        if action < -self.market_env.min_action:
            self.market_env.order_manager.create_order(code, "sell", abs(action))
        if action > self.market_env.min_action:
            self.market_env.order_manager.create_order(code, "buy", abs(action))
