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
from policy.trend_policy import OrderPolicy, Agent


logger.remove()  # 这行很关键，先删除logger自动产生的handler，不然会出现重复输出的问题
logger.add(sys.stderr, level='ERROR')  # 只输出警告以上的日志
logger.add("bt.log",level="INFO")

def run_policy(symbol):
    global_var.SYMBOLS = [symbol]
    account = Account(init_capital=1000000)
    order_policy = OrderPolicy(account)

    env = MultiMarketEnv(
        250,
        # code='000001',
        start_date="20231001",
        # end_date="20220101",
        max_stake=10000000,
        account=account,
        order_policy=order_policy,
    )
    agent = Agent(env)
    obs_list, reward, done, info = env.reset()
    total_reward = 0
    while obs_list:
        actions = agent.action_decider(obs_list)
        agent.stock_decider(actions)
        obs_list, info = env.step(actions)
        total_reward += reward

    ret = env.result()
    # logger.info("\n" + env.order_manager.get_order_history())
    ret["buy_sell_points"] = env.order_manager.buy_sell_points

    ret["order_stats"] = env.order_manager.get_order_stats()
    code_returns = sorted(ret["code_returns"].items(), key=lambda x: x[1])
    code_returns = [key for key, value in code_returns]

    logger.info(f"bt res : {ret}")
    return ret


if __name__ == "__main__":
    logger.add("bt.log")
    run_policy("sz.300059")
