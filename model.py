from collections import deque
from random import sample
import random
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from loguru import logger
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
import os
from market_env import Account, MarketEnv, Order, OrderManager
from torch.utils.tensorboard import SummaryWriter
from market_env import OrderPolicy

writer = SummaryWriter()
from utils import *


def train():
    env = MarketEnv(252, start_date="20101001", end_date="20230401")
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    print(input_size, output_size)
    hidden_dim = 128
    agent = DQNAgent(input_size, hidden_dim, output_size)
    num_episodes = 10000
    market_return_vis = []
    strategy_return_vis = []
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            # print(reward)
            agent.buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.train(episode)
        if episode % 100 == 0:
            agent.save(f"dqn-{episode}")
        tot_ret = env.result()
        market_return_vis.append(tot_ret["market_return"])
        strategy_return_vis.append(tot_ret["strategy_return"])
        rewards.append(total_reward)
        writer.add_scalar("reward", total_reward, episode)
        writer.add_scalar("market_return", tot_ret["market_return"], episode)
        writer.add_scalar("strategy_return", tot_ret["strategy_return"], episode)
        logger.info(
            f'Episode: {episode+1} | loss : {agent.losses[-1]} | Total Reward: {total_reward} | market_return: {tot_ret['market_return'] }% | strategy_return: {tot_ret['strategy_return']}%'
        )
        if episode % 100 == 0:
            valid(f"dqn-{episode}.pth")

    # 创建包含2行1列的子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6))

    ax1.plot(market_return_vis, label="Market Return")
    ax1.plot(strategy_return_vis, label="Strategy Return")
    ax1.legend(loc="best")
    ax1.set_title("Market vs Strategy Returns")
    ax1.set_ylabel("Returns")
    ax1.grid(True)

    ax2.plot(agent.losses, label="loss")
    ax2.set_title("loss")
    ax2.set_ylabel("loss")
    ax2.grid(True)

    ax3.plot(rewards, label="loss")
    ax3.set_title("reward")
    ax3.set_ylabel("reward")
    ax3.grid(True)

    plt.savefig("return_comparison.png", dpi=300, bbox_inches="tight")
    # 显示图形
    # plt.show()


def valid(filename):
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # env.seed(seed)  # 如果环境提供seed()方法

    env = MarketEnv(220, start_date="20230401", end_date="20240401")
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    hidden_dim = 128
    agent = DQNAgent(input_size, hidden_dim, output_size, train=False)
    agent.load(filename)
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        # print(reward)
        # agent.buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    ret = env.result()
    logger.info(
        f'Total Reward: {total_reward} | market_return: {ret['market_return'] }% | strategy_return: {ret['strategy_return']}% | max_drawdown : {ret['max_drawdown']} | max_profit : {ret['max_profit']} |  position : {np.array(ret['positions'])} | actions : {np.array(ret['actions'])}'
    )


def base_policy():
    class BaseAgent:
        def __init__(self) -> None:
            self.last_state = None
            self.last_rsi = None

        def select_action(self, state):
            val = state[2]
            rsi = state[3]
            ret = 1
            if self.last_state is None:
                self.last_state = val
                self.last_rsi = rsi
                return 0

            if val > self.last_state:
                ret = 1
            else:
                ret = -1
            self.last_state = val
            self.last_rsi = rsi
            return ret

    # seed = 0
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # env.seed(seed)  # 如果环境提供seed()方法
    agent = BaseAgent()
    env = MarketEnv(220, start_date="20230401", end_date="20240401")
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        # print(state[0])
        action = agent.select_action(state[0])
        next_state, reward, done, _ = env.step(action)
        # print(reward)
        # agent.buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    ret = env.result()

    # print(ret['market_returns'],ret['strategy_returns'])
    env.plot()
    logger.info(
        f'Total Reward: {total_reward} | market_return: {ret['market_return'] }% | strategy_return: {ret['strategy_return']}% | max_drawdown : {ret['max_drawdown']} | max_profit : {ret['max_profit']} |  position : {np.array(ret['positions'])} | actions : {np.array(ret['actions'])}'
    )


def three_policy():
    class BaseOrderPolicy(OrderPolicy):
        def __init__(self, account) -> None:
            self.last_obs = None
            self.cur_obs = None

            self.account = account

            self.tracking = []

        def order_callback(self, order: Order, order_manager: OrderManager):
            if order.order_type == "buy":
                order_manager.create_order(order.symbol, "stop", order.quantity)

        def buy_policy(self, order):
            action = order.quantity

            trading_price = self.cur_obs["open"]
            if action > self.account.min_action:
                if self.cur_obs["high"] > self.last_obs["high"]:
                    trading_price = max(self.last_obs["high"], self.cur_obs["open"])
                    num_stakes = min(
                        self.account.capital // trading_price // 100 * 100, action
                    )
                    if num_stakes > 0:
                        amount = trading_price * num_stakes
                        cost = amount * self.account.trading_cost_bps
                        if self.account.capital >= amount + cost:
                            order.quantity = num_stakes
                            return (True, trading_price)
                else:
                    logger.info(
                        f"order fail -> tracking | datetime : {self.cur_obs.name} order_id : {order.order_id} | order_type : {order.order_type}"
                    )
                    order.status = "tracking"

            return (False, trading_price)

        def sell_policy(self, order):
            action = order.quantity
            trading_price = self.cur_obs["open"]

            if (
                self.account.available > 0
                and self.cur_obs["low"] < self.last_obs["low"]
            ):
                num_stakes = min(self.account.positions[-1], action)
                if num_stakes > 0:
                    trading_price = min(self.last_obs["low"], self.cur_obs["open"])
                    order.quantity = num_stakes
                    return (True, trading_price)
            else:
                logger.info(
                    f"order fail -> tracking | datetime : {self.cur_obs.name} | order_id : {order.order_id} | order_type : {order.order_type} "
                )
                if order.order_type == "sell":
                    order.status = "tracking"
            return (False, trading_price)

        def step(self, obs):
            self.last_obs = self.cur_obs
            self.cur_obs = obs

    class ThreeAgent:
        def __init__(self, market_env: MarketEnv) -> None:
            self.last_macd_close_weekly = None
            self.cur_macd_close_weekly = None

            self.last_force_index = None

            self.market_env = market_env

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

        def select_action(self, info):
            macd_close_weekly = info["macd_close_weekly"]
            force_index = info["force_index_close"]
            ret = 0
            if (
                self.last_force_index
                and self.cur_macd_close_weekly
                and self.last_force_index
                and self.last_macd_close_weekly
            ):
                if self.cur_macd_close_weekly > self.last_macd_close_weekly:
                    # trend up
                    if (
                        force_index < self.last_force_index
                        and force_index < 0
                        and self.last_force_index > 0
                    ):
                        # buy
                        ret = 1
                else:
                    # trend down
                    if (
                        force_index > self.last_force_index
                        and force_index > 0
                        and self.last_force_index < 0
                    ):
                        # buy
                        ret = -1

            if macd_close_weekly != self.cur_macd_close_weekly:
                self.last_macd_close_weekly = self.cur_macd_close_weekly
                self.cur_macd_close_weekly = macd_close_weekly

            self.last_force_index = force_index
            return ret

        def create_order(self, action):
            action = int(action * self.market_env.max_stake)
            if action < -self.market_env.min_action:
                self.market_env.order_manager.create_order(
                    "510880", "sell", abs(action)
                )
            if action > self.market_env.min_action:
                self.market_env.order_manager.create_order("510880", "buy", abs(action))

    # seed = 0
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # env.seed(seed)  # 如果环境提供seed()方法

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

    env = MarketEnv(
        220 * 2,
        code="510880",
        # code='000001',
        start_date="20220601",
        end_date="20240601",
        initial_capital=10000,
        max_stake=10000,
        account=account,
        order_policy=order_policy,
    )
    agent = ThreeAgent(env)
    state, info = env.reset()
    total_reward = 0
    done = False
    while not done:
        # print(state[0])
        action = agent.select_action(info["ori_obs"])
        agent.create_order(action)
        next_state, reward, done, info = env.step(action)
        # print(reward)
        # agent.buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    ret = env.result()

    # print(ret['market_returns'],ret['strategy_returns'])
    env.plot()
    # logger.info(
    #     f'Total Reward: {total_reward} | market_return: {ret['market_return'] }% | strategy_return: {ret['strategy_return']}% | max_drawdown : {ret['max_drawdown']} | max_profit : {ret['max_profit']} |  position : {np.array(ret['positions'])} | actions : {np.array(ret['actions'])}'
    # )

    # logger.info(
    #     f'Total Reward: {total_reward} | market_return: {ret['market_return'] }% | strategy_return: {ret['strategy_return']}% | max_drawdown : {ret['max_drawdown']} | max_profit : {ret['max_profit']} |'
    # )

    logger.info(f"Total Reward: {total_reward} | {ret}")


if __name__ == "__main__":
    # valid("ddqn-600.pth")
    # train()
    # base_policy()
    three_policy()
