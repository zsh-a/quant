import os
import sys
from loguru import logger
import numpy as np
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from data_source import DBDataSource
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns  # 本代码归JayBee黄所有
import talib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from .base_policy import OrderPolicy

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)



parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 将外层目录添加到 sys.path
sys.path.append(parent_dir)
import global_var

from account import Account
from market_env import MultiMarketEnv
from order import Order, OrderManager

# from model import Model

factors = ['momentum_5', 'vol_ratio' ,'RSI_14','BB_upper','BB_lower','volatility_20','turnover_avg_20','obv','turnover_5']  # JayBee黄量化策略

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
        pass

    def buy_policy(self, order: Order):
        action = order.quantity
        idx = global_var.SYMBOLS.index(order.symbol)
        obs = self.cur_obs[idx]

        trading_price = obs["open"]
        if action > self.account.min_action:
            if self.buy_cond(order.symbol):
                trading_price = obs["open"] + self.slip

                num_stakes = min(
                    self.account.capital // trading_price // 100 * 100, action
                )
                if num_stakes > 100 and sum(self.account.positions[-1]) == 0:
                    amount = trading_price * num_stakes
                    cost = amount * self.account.trading_cost_bps
                    if self.account.capital >= amount + cost:
                        order.quantity = num_stakes
                        return (True, trading_price)

        return (False, trading_price)

    def buy_cond(self, code):
        idx = global_var.SYMBOLS.index(code)
        key = "close" if self.running_in_day else "high"
        # logger.info(
        #     f"buy cond | symbol : {code} | cur high : {self.cur_obs[idx]["high"]} | last high : {self.last_obs[-1][idx]["high"]}"
        # )
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
        return (False, trading_price)

    def sell_cond(self, code):
        return True

    def step(self, obs):
        self.last_obs.append(self.cur_obs)
        if len(self.last_obs) > 2:
            self.last_obs = self.last_obs[-2:]
        self.cur_obs = obs

    def step_in_day(self, obs):
        self.cur_obs = obs
        self.running_in_day = True


class ThreeAgent:
    def __init__(self, market_env: MultiMarketEnv) -> None:
        self.market_env = market_env
        self.model = torch.load("model.pth", weights_only=False).to(device)
        self.model.eval()


    def select_action(self, code, info):
        info = info[0] # obs
        with torch.no_grad():
            inputs = torch.tensor(info[factors].values, dtype=torch.float32).to(device)
            outputs = self.model(inputs)
            print(outputs)
            action = outputs.item()
            if action > 0:
                return (1, action)
            elif action < 0:
                return (-1, -action)

        return (0, 0)

    def action_decider(self, stocks_obs):
        return [
            {"idx": i, "info": self.select_action(global_var.SYMBOLS[i], stocks_obs[i])}
            for i in range(len(stocks_obs))
        ]

    def stock_decider(self, actions):
        buy_list = [v for v in actions if v["info"][0] == 1]
        sell_list = [v for v in actions if v["info"][0] == -1]
        # logger.debug(f"{buy_list}")
        buy_list = sorted(buy_list, key=lambda x: x["info"][1])
        sell_list = sorted(sell_list, key=lambda x: x["info"][1])
        if len(buy_list) > 0:
            logger.info(f"buy list {buy_list}")
            self.create_order(code=global_var.SYMBOLS[buy_list[0]["idx"]], action=1)
        if len(sell_list) > 0:
            logger.info(f"sell list {sell_list}")
            self.create_order(code=global_var.SYMBOLS[sell_list[0]["idx"]], action=-11)
            

    def cancel_order(self, code):
        self.market_env.order_manager.cancel_order(code)

    def create_order(self, code, action):
        action = int(action * self.market_env.max_stake)
        if action < -self.market_env.min_action:
            self.market_env.order_manager.create_order(code, "sell", abs(action))
        if action > self.market_env.min_action:
            self.market_env.order_manager.create_order(code, "buy", abs(action))
