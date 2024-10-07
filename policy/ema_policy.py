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
from market_env import MarketEnv, MultiMarketEnv
from order import Order, OrderManager


class BaseOrderPolicy(OrderPolicy):
    def __init__(self, account) -> None:
        self.last_obs = []
        self.last_obs
        self.cur_obs = None

        self.account: Account = account

        self.slip = 0.001
        self.tracking = []

    def order_callback(self, order: Order, order_manager: OrderManager):
        if order.order_type == "buy":
            order_manager.create_order(order.symbol, "stop", order.quantity)

    def buy_policy(self, order: Order):
        action = order.quantity
        obs = self.cur_obs[order.symbol]

        trading_price = obs["open"]
        if action > self.account.min_action:
            if self.buy_cond(order.symbol):
                trading_price = (
                    max(self.last_obs[-1][order.symbol]["high"], obs["open"])
                    + self.slip
                )
                num_stakes = min(
                    self.account.capital // trading_price // 100 * 100, action
                )
                if num_stakes > 100:
                    amount = trading_price * num_stakes
                    cost = amount * self.account.trading_cost_bps
                    if self.account.capital >= amount + cost:
                        order.quantity = num_stakes
                        return (True, trading_price)
            else:
                logger.info(
                    f"order fail -> tracking | datetime : {obs.name} | symbol : {order.symbol} | order_id : {order.order_id} | order_type : {order.order_type} | cur : {self.cur_obs[order.symbol]["close"]} | cur high : {self.cur_obs[order.symbol]["high"]} | last high : {self.last_obs[-1][order.symbol]["high"]}"
                )
                order.status = "tracking"

        return (False, trading_price)

    def buy_cond(self, code):
        logger.info(
            f"buy cond | symbol : {code} | cur high : {self.cur_obs[code]["high"]} | last high : {self.last_obs[-1][code]["high"]}"
        )
        return self.cur_obs[code]["high"] > self.last_obs[-1][code]["high"]

    def sell_policy(self, order):
        action = order.quantity
        obs = self.cur_obs[order.symbol]
        trading_price = obs["open"]
        # logger.debug(f"pos : {self.account.positions[-1]}")
        idx = global_var.SYMBOLS.index(order.symbol)
        if self.sell_cond(order.symbol):
            num_stakes = min(self.account.positions[-1][idx], action)
            if num_stakes > 0:
                trading_price = (
                    min(self.last_obs[-1][order.symbol]["low"], obs["open"]) - self.slip
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
        logger.info(
            f"sell cond | avail : {self.account.get_available(code)} | cur low : {self.cur_obs[code]["low"]} | cur close : {self.cur_obs[code]["close"]} last low : {min(self.last_obs[-1][code]["low"], self.last_obs[-2][code]["low"])} | ema5 : {self.cur_obs[code]["ema5"]}"
        )
        # print(self.cur_obs[code])
        return self.account.get_available(code) > 0 and (
            self.cur_obs[code]["low"]
            < min(self.last_obs[-1][code]["low"], self.last_obs[-2][code]["low"])
            and self.cur_obs[code]["low"] < self.cur_obs[code]["ema5"]
        )

    def step(self, obs):
        self.last_obs.append(self.cur_obs)
        if len(self.last_obs) > 2:
            self.last_obs = self.last_obs[-2:]
        self.cur_obs = obs

    def step_in_day(self, obs):
        self.cur_obs = obs


class ThreeAgent:
    class HisInfo:
        def __init__(self) -> None:
            self.last_ema13 = None
            self.cur_ema13 = None
            self.last_force_index = None

    def __init__(self, market_env: MarketEnv) -> None:
        self.market_env = market_env

        self.his_info = {}

    def find_monotonic_intervals(series):
        values = series.values
        diffs = np.diff(values)
        trends = np.sign(diffs)

        # Append a zero at the beginning to align trends with the original series length
        trends = np.insert(trends, 0, 0)

        # Find the change points
        change_points = np.where(trends[:-1] != trends[1:])[0] + 1

        intervals = []
        start = 0
        for cp in change_points:
            if trends[start] != 0:
                intervals.append((start, cp - 1, trends[start]))
            start = cp

        # Add the last interval
        if trends[start] != 0:
            intervals.append((start, len(series) - 1, trends[start]))

        return intervals

    def select_action(self, code, info):
        ema13 = info["ema13"]
        force_index = info["force_index_close"]
        ret = 0

        if code not in self.his_info:
            self.his_info[code] = self.HisInfo()
        his = self.his_info[code]
        if his.last_force_index:
            if ema13 > his.last_ema13:
                # trend up
                if (
                    force_index < his.last_force_index
                    and force_index < 0
                    and his.last_force_index > 0
                ):
                    # buy
                    ret = 1
            else:
                # trend down
                if (
                    force_index > his.last_force_index
                    and force_index > 0
                    and his.last_force_index < 0
                ):
                    # sell
                    ret = -1

        his.last_ema13 = ema13
        his.last_force_index = force_index
        return ret, force_index

    def action_decider(self, stocks_obs):
        return {code: self.select_action(code, stocks_obs[code]) for code in stocks_obs}

    def stock_decider(self, actions):
        for code, action in actions.items():
            # print(code,action)
            if action[0] < 0:
                self.market_env.order_manager.cancel_order(code)

        # print(actions)
        buy_list = [(code, v[1]) for code, v in actions.items() if v[0] == 1]
        buy_list = sorted(buy_list, key=lambda x: abs(x[1]))
        # print(buy_list)
        if len(buy_list) > 0:
            logger.info(f"buy list {buy_list}")
            self.create_order(buy_list[0][0], 1)

    def create_order(self, code, action):
        action = int(action * self.market_env.max_stake)
        if action < -self.market_env.min_action:
            self.market_env.order_manager.create_order(code, "sell", abs(action))
        if action > self.market_env.min_action:
            self.market_env.order_manager.create_order(code, "buy", abs(action))

    # seed = 0
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # env.seed(seed)  # 如果环境提供seed()方法


if __name__ == "__main__":
    account = Account(init_capital=10000)
    order_policy = BaseOrderPolicy(account)

    # env = MarketEnv(
    #     220,
    #     start_date="20230401",
    #     end_date="20240401",
    #     initial_capital=10000,
    #     max_stake=10000,
    #     account=account,
    #     order_policy=order_policy,
    # )

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
    # logger.info(
    #     f'Total Reward: {total_reward} | market_return: {ret['market_return'] }% | strategy_return: {ret['strategy_return']}% | max_drawdown : {ret['max_drawdown']} | max_profit : {ret['max_profit']} |  position : {np.array(ret['positions'])} | actions : {np.array(ret['actions'])}'
    # )

    # logger.info(
    #     f'Total Reward: {total_reward} | market_return: {ret['market_return'] }% | strategy_return: {ret['strategy_return']}% | max_drawdown : {ret['max_drawdown']} | max_profit : {ret['max_profit']} |'
    # )

    logger.info(f"Total Reward: {total_reward} | {ret}")