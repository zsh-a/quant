from datetime import datetime
import time
from typing import Any, List, SupportsFloat, Tuple, Type
import gymnasium as gym
import numpy as np
from account import Account
from data_source import DBDataSource
from loguru import logger
from order import Order, OrderManager
from policy.base_policy import OrderPolicy
from utils.utils import log2percent
import global_var
import pandas as pd
import akshare as ak
import client as trader
import utils.feichu as msg

INF = 1e9


class Broker:
    def __init__(self) -> None:
        self.last_obs = None

        self.order_policy: OrderPolicy = None
        self.account: Account = None
        self.order_manager: OrderManager = None

    def set_policy(self, order_policy):
        self.order_policy = order_policy

    def get_data(self):
        name = "tmp.csv"
        df = ak.fund_etf_spot_em()
        df.to_csv(name)
        df = pd.read_csv(name)
        all_etf = pd.read_csv("all_etf.csv")

        all_etf.columns = ["代码", "1", "2"]
        # print(all_etf)
        all_etf = all_etf.set_index("代码")
        df = pd.read_csv(name, index_col="代码")
        # df.set_index('代码',inplace=True)
        df = df.loc[all_etf.index.values]
        # df["成交量"] = df["成交量"].astype(int)
        # date = pd.to_datetime(df["数据日期"].unique()[0])
        df = df[
            [
                "最新价",
                # "成交量",
                "成交额",
                "开盘价",
                "最高价",
                "最低价",
                "昨收",
                "更新时间",
            ]
        ]
        df = df.rename(
            columns={
                "最新价": "close",
                # "成交量": "volume",
                "成交额": "amount",
                "开盘价": "open",
                "最高价": "high",
                "最低价": "low",
                "昨收": "pre_close",
                "更新时间": "date",
            }
        )
        df.index.name = "code"
        df.index = df.index.astype(str)
        # df = df.loc[["510880"]]
        df = df.loc[global_var.SYMBOLS]
        df = {row[0]: row[1] for row in df.iterrows()}
        assert self.last_obs is not None
        for idx, v in enumerate(self.last_obs):
            k = global_var.SYMBOLS[idx]
            df[k]["open"] *= v["adj_factor"]
            df[k]["high"] *= v["adj_factor"]
            df[k]["low"] *= v["adj_factor"]
            df[k]["close"] *= v["adj_factor"]
            Value_today = df[k]["close"]

            N = 10
            K = 2 / (N + 1)
            EMA_yesterday = v["ema10"]
            df[k]["ema10"] = (Value_today * K) + (EMA_yesterday * (1 - K))

            N = 20
            K = 2 / (N + 1)
            EMA_yesterday = v["ema20"]
            df[k]["ema20"] = (Value_today * K) + (EMA_yesterday * (1 - K))

            N = 5
            K = 2 / (N + 1)
            EMA_yesterday = v["ema5"]
            df[k]["ema5"] = (Value_today * K) + (EMA_yesterday * (1 - K))
        ret = [df[code] for code in global_var.SYMBOLS]
        return ret

    def step(self, obs):
        # self.order_policy.step(obs)
        self.last_obs = obs
        # print(self.get_data())

    def run(self, orders: List[Type[Order]]):
        self.order_manager.step(None)
        self.account.step(None)
        logger.info("live running")
        # logger.info(f"waiting orders : {self.order_manager.get_waiting_order()}")
        while True:
            current_time = datetime.now().time()
            three_pm = current_time.replace(hour=15, minute=0, second=0, microsecond=0)
            logger.info(f"start process")
            try:
                df = self.get_data()
            except Exception as e:
                logger.error(e)
                time.sleep(5)
                continue
            self.order_policy.step_in_day(df)
            self.match_order(orders)
            if current_time > three_pm:
                logger.info("live runing end")
                return
            time.sleep(60)

    def match_order(self, orders: List[Type[Order]]):
        for order in orders:
            if order.status == "open" or order.status == "tracking":
                if order.order_type == "buy":
                    ok, exec_price = self.order_policy.buy_policy(order)
                    if ok:
                        msg.send_no_except(f"价格向上突破 等待买入{order.symbol}")
                        self.execute_order(order, exec_price)
                elif order.order_type == "sell" or order.order_type == "stop":
                    ok, exec_price = self.order_policy.sell_policy(order)
                    if ok:
                        msg.send_no_except(f"价格向下突破 等待卖出{order.symbol}")
                        self.execute_order(order, exec_price)

    def execute_order(self, order: Order, exec_price):
        logger.info(f"exec order : {order} , exec_price : {exec_price}")
        try:
            if order.order_type == "buy":
                ret = trader.client.buy(order.symbol)
            else:
                ret = trader.client.sell(order.symbol)
        except Exception as e:
            ret = e
            logger.error(ret)


class MultiMarketEnv(gym.Env):
    def __init__(
        self,
        num_step,
        code="510880",
        start_date="20100531",
        end_date=None,
        work_dir="",
        initial_capital=100000,
        max_stake=10000,
        account=Account(),
        order_policy=None,
    ) -> None:
        super().__init__()

        self.code = global_var.SYMBOLS
        self.data_source : List[DBDataSource] = [
            DBDataSource(
                code=c,
                # trading_days=num_step,
                start_date=start_date,
                end_date=end_date,
                work_dir=work_dir,
            )
            for c in self.code
        ]

        self.capital = initial_capital
        self.max_stake = max_stake
        # self.num_step = num_step
        self.tot_values = self.capital
        self.min_action = 100

        self.num_step = len(self.data_source[0].data)
        self.tot_values = np.zeros(self.num_step + 1)
        self.tot_values[0] = self.capital

        self.capitals = np.zeros(self.num_step + 1)
        self.capitals[0] = self.capital
        self.trading_cost_bps = 5e-4
        # state
        self.cur_step = 1
        self.actions = np.zeros(self.num_step + 1)
        self.positions = np.zeros(self.num_step + 1)
        self.trades = np.zeros(self.num_step + 1)
        self.costs = np.zeros(self.num_step + 1)
        self.strategy_returns = np.zeros(self.num_step + 1)
        self.market_returns = np.zeros(self.num_step + 1)

        self.account = account

        self.order_manager = OrderManager(account, order_policy)
        self.broker = Broker()
        self.broker.account = self.account
        self.broker.order_manager = self.order_manager
        self.broker.set_policy(self.order_manager.order_plolicy)
    def add_indicator(self, indicator):
        for ds in self.data_source:
            ds.add_indicator(indicator)

    def clean_data(self):
        for ds in self.data_source:
            ds.clean_data()

    def exec_order(self, obs):
        self.order_manager.match_orders(obs)

    def live(self):
        self.broker.run(self.order_manager.orders)

    def step(self, action: Any) -> Tuple[Any | SupportsFloat | bool | dict[str, Any]]:
        # assert self.action_space.contains(action)

        obs_list = [ds.step() for ds in self.data_source]

        # 如果obs_list中有None,则返回None
        if any(x is None for x in obs_list):
            return None, None

        self.order_manager.step(obs_list)
        self.broker.step(obs_list)
        self.account.step(obs_list)

        self.exec_order(obs_list)

        self.market_returns[self.cur_step] = np.mean(
            [obs["returns"] for obs in obs_list]
        )

        reward = (
            self.account.tot_values[self.cur_step]
            - self.account.tot_values[self.cur_step - 1]
        )

        info = {
            "reward": reward,
            "costs": self.costs[self.cur_step],
        }
        self.cur_step += 1
        return obs_list, info

    def result(self):
        account_res = self.account.result(self.market_returns[self.cur_step - 1])
        market_res = {
            "market_return": log2percent(np.exp(sum(self.market_returns[1:-1]))).astype(
                float
            ),
            "market_value": (
                np.exp(self.market_returns[1:-1].cumsum()) * self.account.capitals[0]
            ).tolist(),
        }
        return account_res | market_res

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> Tuple[Any, dict]:
        self.cur_step = 1
        self.actions.fill(0)
        self.strategy_returns.fill(0)
        self.costs.fill(0)
        self.trades.fill(0)
        self.market_returns.fill(0)
        for ds in self.data_source:
            ds.reset()

        obs_list = [ds.step() for ds in self.data_source]

        # info = {"ori_obs": [v[2] for v in obs]}
        info = {}
        return obs_list, 0, False, info


if __name__ == "__main__":
    # env = MarketEnv(220, start_date="20230401", end_date="20240401")
    # env.reset()
    # print(env.step(1))
    # print(env.step(0.05))
    # print(env.step(0.05))
    # print(env.step(0.05))
    # print(env.step(0.05))
    # print(env.step(0.05))
    # print(env.step(-0.1))

    # env.plot()

    # env = MultiMarketEnv(220, start_date="20230401", end_date="20240401")
    # env.reset()
    # print(env.step(1000))
    # print(env.step(0.05))
    # print(env.step(0.05))
    # print(env.step(0.05))
    # print(env.step(0.05))
    # print(env.step(0.05))
    # print(env.step(-0.1))

    # env.plot()
    bk = Broker()
    print(bk.get_data())
