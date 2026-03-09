from loguru import logger

import os
import sys
import json

# 获取当前文件的绝对路径，并找到上层目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from account import Account
import global_var
from market_env import MultiMarketEnv
from policy.JSG_policy import OrderPolicy
from policy.JSG_policy import Agent
# from policy.advanced_rotation_policy import AdvancedAgent as Agent
from src.market_data.db import DB

logger.remove()  # 这行很关键，先删除logger自动产生的handler，不然会出现重复输出的问题
logger.add(sys.stderr, level="ERROR")  # 只输出警告以上的日志
logger.add("bt.log", level="INFO")


def run_policy(symbol, start_date, end_date):
    db_client = DB()
    global_var.SYMBOLS = [symbol]
    account = Account(init_capital=1000000, db_client=db_client)
    order_policy = OrderPolicy(account, max_stocks=12, db_client=db_client)
    env = MultiMarketEnv(
        250,
        # code='000001',
        start_date=start_date,
        end_date=end_date,
        max_stake=10000000,
        account=account,
        order_policy=order_policy,
        db_client=db_client,
    )
    env.clean_data()
    agent = Agent(env, db_client=db_client)
    obs_list, reward, done, info = env.reset()
    total_reward = 0
    actions = []
    while True:
        # today
        obs_list, info = env.step(actions)
        if not obs_list:
            break
        agent.step()
        agent.run_end()
        env.run_end()
        env.order_manager.cancel_all()
        actions = agent.action_decider(obs_list)
        total_reward += reward

    ret = env.result()
    logger.info("\n" + env.order_manager.get_order_history())
    ret["buy_sell_points"] = env.order_manager.buy_sell_points

    ret["order_stats"] = env.order_manager.get_order_stats()
    print(ret["order_stats"])
    # code_returns = sorted(ret["code_returns"].items(), key=lambda x: x[1])
    # code_returns = [key for key, value in code_returns]

    logger.info(f"bt res : {ret}")
    print(json.dumps(ret))
    return ret


if __name__ == "__main__":
    # run_policy("sh.000300", "20110101", "20200101")
    # run_policy("sh.000905", "20200101", "20221230")
    # run_policy("sh.000905", "20230101", "20250101")
    run_policy("sh.000905", "20240101", "20261201")
