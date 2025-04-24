from loguru import logger

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
class LeNet(nn.Module):  # 继承于nn.Module这个父类
    def __init__(self):  # 初始化网络结构
        super(LeNet, self).__init__()  # 多继承需用到super函数
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.fc1 = nn.Linear(32 * 14 * 1, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.relu = nn.ReLU()

    def forward(self, x):  # 正向传播过程
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))  # input(1, 60, 6) output(16, 28, 28)
        x = self.pool1(x)  # output(16, 52, 2)
        x = self.relu(self.conv2(x))  # output(32, 28, 1)
        x = self.pool2(x)  # output(32, 14, 1)
        x = x.view(-1, 32 * 14 * 1)  # output(32*14*1)
        x = self.relu(self.fc1(x))  # output(120)
        x = self.relu(self.fc2(x))  # output(84)
        x = self.fc3(x)  # output(1)
        return x



# 获取当前文件的绝对路径，并找到上层目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from account import Account
import global_var
from market_env import MultiMarketEnv
from policy.JSG_policy import OrderPolicy, Agent
from db import DB

logger.remove()  # 这行很关键，先删除logger自动产生的handler，不然会出现重复输出的问题
logger.add(sys.stderr, level="ERROR")  # 只输出警告以上的日志
logger.add("bt.log", level="INFO")


def run_policy(symbol, start_date, end_date):
    db_client = DB()
    global_var.SYMBOLS = [symbol]
    account = Account(init_capital=1000000, db_client=db_client)
    order_policy = OrderPolicy(account, db_client=db_client)

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
    run_policy("sh.000300", "20241226", "20260101")
    # run_policy("sh.000905", "20250101", "20260101")
