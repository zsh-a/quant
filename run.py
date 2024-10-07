import os
import sys
import time
from loguru import logger

import schedule
import argparse
from account import Account
from market_env import MultiMarketEnv
from policy.three_policy import BaseOrderPolicy, ThreeAgent
# from policy.ema_policy import BaseOrderPolicy, ThreeAgent


def run_policy(args):
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
        start_date="20211001",
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
    live = args.live
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
    # print(code_returns)re
    code_returns = [key for key, value in code_returns]
    # print(code_returns)
    # print(ret['market_returns'],ret['strategy_returns'])
    env.plot(code_returns)
    logger.warning(f"Total Reward: {total_reward} | {ret}")


if __name__ == "__main__":
    # logger.add(sys.stdout, level="ERROR")  # 设定日志输出的最低级别
    # logger.remove()
    # logger.remove()
    # logger.add(sys.stdout, level="WARNING")  # 设定日志输出的最低级别

    # 创建解析器
    parser = argparse.ArgumentParser(description="策略运行")

    parser.add_argument("--live", action="store_true", help="显示详细信息")
    args = parser.parse_args()

    if not args.live:
        os.environ["INFLUXDB_TOKEN"] = (
            "vH5FD5il70h2n5RNO9zj6i6dRO9TMQihKWL9xDhdbarA7wyXZNM-GOgkc6MKJS3zsmYEOBaW_gylF-XVZBSR0A=="
        )

    os.system("rm -rf gen/*")
    logger.add("logfile.log", rotation="10 MB", retention="100 days", compression="zip")
    run_policy(args)
    if args.live:
        schedule.every().day.at("09:30").do(run_policy)
        while True:
            schedule.run_pending()
            time.sleep(1)
