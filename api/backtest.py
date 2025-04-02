from loguru import logger

import os
import sys


# 获取当前文件的绝对路径，并找到上层目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from account import Account
import global_var
from market_env import MultiMarketEnv
from policy.trend_policy import BaseOrderPolicy, ThreeAgent





def run_policy(symbol):
    global_var.SYMBOLS = [symbol]
    account = Account(init_capital=1000000)
    order_policy = BaseOrderPolicy(account)

    env = MultiMarketEnv(
        250,
        # code='000001',
        start_date="20220101",
        # end_date="20220101",
        max_stake=10000000,
        account=account,
        order_policy=order_policy,
    )
    agent = ThreeAgent(env)
    (obs, pos), reward, done, info = env.reset()
    total_reward = 0
    done = False
    while not done:
        # print(state)

        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",obs)
        actions = agent.action_decider(obs)
        # print(actions)
        agent.stock_decider(actions)
        (next_state, pos), reward, done, info = env.step(actions)
        # print(reward)
        # agent.buffer.push(state, action, reward, next_state, done)
        obs = next_state
        total_reward += reward

    ret = env.result()
    logger.info("\n" + env.order_manager.get_order_history())
    ret["buy_sell_points"] = env.order_manager.buy_sell_points

    ret["order_stats"] = env.order_manager.get_order_stats()
    code_returns = sorted(ret["code_returns"].items(), key=lambda x: x[1])
    code_returns = [key for key, value in code_returns]
    return ret


if __name__ == "__main__":
    logger.add("bt.log")
    run_policy("sh.000001")
