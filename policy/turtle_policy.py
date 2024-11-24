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


class BaseOrderPolicy(OrderPolicy):
    def __init__(self, account) -> None:
        self.last_obs = []
        self.last_obs
        self.cur_obs = None

        self.account: Account = account

        self.slip = 0.001
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
                    max(self.last_obs[-1][idx]["high"], obs["open"]) + self.slip
                )
                num_stakes = min(
                    self.account.capital // trading_price // 100 * 100, action
                )
                if num_stakes > 100 and sum(self.account.positions[-1]) == 0:
                    amount = trading_price * num_stakes
                    cost = amount * self.account.trading_cost_bps
                    if self.account.capital >= amount + cost:
                        order.quantity = num_stakes
                        return (True, trading_price)
            else:
                logger.info(
                    f"order fail -> tracking | datetime : {obs.name} | symbol : {order.symbol} | order_id : {order.order_id} | order_type : {order.order_type} | cur : {self.cur_obs[idx]["close"]} | cur high : {self.cur_obs[idx]["high"]} | last high : {self.last_obs[-1][idx]["high"]}"
                )
                order.status = "tracking"

        return (False, trading_price)

    def buy_cond(self, code):
        # idx = global_var.SYMBOLS.index(code)
        # key = "close" if self.running_in_day else "high"
        # logger.info(
        #     f"buy cond | symbol : {code} | cur high : {self.cur_obs[idx]["high"]} | last high : {self.last_obs[-1][idx]["high"]}"
        # )
        # return self.cur_obs[idx][key] > self.last_obs[-1][idx]["high"]
        return True

    def sell_policy(self, order):
        action = order.quantity
        idx = global_var.SYMBOLS.index(order.symbol)
        obs = self.cur_obs[idx]
        trading_price = obs["open"]
        # logger.debug(f"pos : {self.account.positions[-1]}")
        if self.sell_cond(order.symbol):
            num_stakes = min(self.account.positions[-1][idx], action)
            if num_stakes > 0:
                trading_price = (
                    min(self.last_obs[-1][idx]["low"], obs["open"]) - self.slip
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
        return True
        # if len(self.last_obs) < 2:
        #     return False
        # idx = global_var.SYMBOLS.index(code)
        # key = "close" if self.running_in_day else "low"
        # logger.info(
        #     f"sell cond | avail : {self.account.get_available(code)} | cur low : {self.cur_obs[idx]["low"]} | cur close : {self.cur_obs[idx]["close"]} last low : {min(self.last_obs[-1][idx]["low"], self.last_obs[-2][idx]["low"])} | ema10 : {self.cur_obs[idx]["ema10"]}"
        # )
        # # print(self.cur_obs[code])
        # return self.account.get_available(code) > 0 and (
        #     self.cur_obs[idx][key]
        #     < min(self.last_obs[-1][idx]["low"], self.last_obs[-2][idx]["low"])
        #     # and self.cur_obs[idx][key] < self.cur_obs[idx]["ema10"]
        # )

    def step(self, obs):
        self.last_obs.append(self.cur_obs)
        if len(self.last_obs) > 2:
            self.last_obs = self.last_obs[-2:]
        self.cur_obs = obs

    def step_in_day(self, obs):
        self.cur_obs = obs
        self.running_in_day = True


class ThreeAgent:
    class HisInfo:
        def __init__(self) -> None:
            self.his_macd_weekly = []
            self.last_force_index = None

    def __init__(self, market_env: MultiMarketEnv) -> None:
        self.market_env = market_env

        self.his_info = {}

    def select_action(self, code, info):
        if info["high"] >= info["up"]:
            return 1
        if info["low"] <= info["down"]:
            return -1
        return 0

    def action_decider(self, stocks_obs):
        # print(stocks_obs)
        return [
            {"idx": i, "info": self.select_action(global_var.SYMBOLS[i], stocks_obs[i])}
            for i in range(len(stocks_obs))
        ]

    def stock_decider(self, actions):
        buy_list = [v for v in actions if v["info"] == 1]
        sell_list = [v for v in actions if v["info"] == -1]
        # logger.debug(f"{buy_list}")
        # buy_list = sorted(buy_list, key=lambda x: x["info"][1])
        
        if len(buy_list) > 0:
            code = global_var.SYMBOLS[buy_list[0]["idx"]]        
            logger.info(f"buy list {buy_list}")
            self.create_order(code=code, action=1)
        if len(sell_list) > 0:
            code = global_var.SYMBOLS[sell_list[0]["idx"]]        
            logger.info(f"sell list {buy_list}")
            self.create_order(code=code, action=-1)

    def cancel_order(self, code):
        self.market_env.order_manager.cancel_order(code)
    def create_order(self, code, action):
        self.cancel_order(code)
        action = int(action * self.market_env.max_stake)
        if action < -self.market_env.min_action:
            self.market_env.order_manager.create_order(code, "sell", abs(action))
        if action > self.market_env.min_action:
            self.market_env.order_manager.create_order(code, "buy", abs(action))
            self.market_env.order_manager.create_order(code, "buy", abs(action))


if __name__ == "__main__":
    account = Account(init_capital=10000)
    order_policy = BaseOrderPolicy(account)

    env = MultiMarketEnv(
        250,
        # code='000001',
        start_date="20210601",
        end_date=None,
        initial_capital=10000,
        max_stake=10000000,
        account=account,
        order_policy=order_policy,
    )
    agent = ThreeAgent(env)
    state, info = env.reset()
    total_reward = 0
    done = False
    live = False
    while not done:
        actions = agent.action_decider(info["ori_obs"])
        # print(actions)
        agent.stock_decider(actions)
        next_state, reward, done, info = env.step(actions)
        # print(reward)
        # agent.buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    if live:
        env.live()
    ret = env.result()
    logger.info("\n" + env.order_manager.get_order_history())
    code_returns = sorted(ret["code_returns"].items(), key=lambda x: x[1])
    # print(code_returns)
    code_returns = [key for key, value in code_returns]
    # print(code_returns)
    # print(ret['market_returns'],ret['strategy_returns'])
    env.plot(code_returns[:3])

    logger.info(f"Total Reward: {total_reward} | {ret}")
