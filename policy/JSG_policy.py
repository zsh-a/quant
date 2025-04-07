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
import indictor
import talib as ta


class OrderPolicy(OrderPolicy):
    def __init__(self, account,**args) -> None:
        self.last_obs = []
        self.cur_obs = None

        self.account: Account = account

        self.db_client = args['db_client']
        self.slip = 0.002  # 改为百分比滑点，0.001表示0·.1%
        self.tracking = []

        self.running_in_day = False
        self.current_date = None

    def order_callback(self, order: Order, order_manager: OrderManager):
        pass

    def get_open_price(self, code):
        df = self.db_client.get_price([code], str(self.current_date.date()), ["open"], 1)
        df.reset_index(level="date", drop=True, inplace=True)
        open_price = df.loc[code, "open"]
        return open_price


    def buy_policy(self, order: Order):
        action = order.quantity

        trading_price = self.get_open_price(order.symbol) * (1 + self.slip)

        min_action = self.account.min_action
        if action > min_action:
            if self.buy_cond(order.symbol):
                logger.info(f"price : {trading_price}")

                action = action // min_action * min_action
                num_stakes = min(
                    self.account.cash // trading_price // min_action * min_action,
                    action,
                )
                if num_stakes >= min_action:
                    amount = trading_price * num_stakes
                    cost = amount * self.account.trading_fee_open

                    while num_stakes > 0 and self.account.cash < amount + cost:
                        num_stakes -= min_action
                        amount = trading_price * num_stakes
                        cost = amount * self.account.trading_fee_open
                    if num_stakes > 0:
                        order.quantity = num_stakes
                        logger.info(
                            f"num_stakes : {num_stakes} {trading_price} : {self.account.cash}"
                        )
                        return (True, trading_price)

        return (False, trading_price)

    def buy_cond(self, code):
        # print(self.last_obs)
        if len(self.last_obs) < 2:
            return False
        # logger.info(
        #     f"buy cond | symbol : {code} | cur high : {self.cur_obs[idx]["high"]} | last high : {self.last_obs[-1][idx]["high"]}"
        # )
        return True
        # return self.cur_obs[idx]['high'] > self.last_obs[-1][idx]["high"]

    def sell_policy(self, order):
        action = order.quantity

        min_action = self.account.min_action
        action = action // min_action * min_action
        trading_price = self.get_open_price(order.symbol) * (1 - self.slip)
        if self.sell_cond(order.symbol):
            num_stakes = min(self.account.positions[-1][order.symbol], action)
            if num_stakes > 0:
                order.quantity = num_stakes
                return (True, trading_price)
        else:
            logger.info(
                f"order fail -> tracking | datetime : {self.current_date} | order_id : {order.order_id} | order_type : {order.order_type} "
            )
            if order.order_type == "sell":
                order.status = "tracking"
        return (False, trading_price)

    def sell_cond(self, code):
        if len(self.last_obs) < 2:
            return False
        key = "low"
        # logger.info(
        #     f"sell cond | avail : {self.account.get_available(code)} | cur low : {self.cur_obs[idx]["low"]} | cur close : {self.cur_obs[idx]["close"]} last low : {self.last_obs[-1][idx]["low"]}, {self.last_obs[-2][idx]["low"]})"
        # )
        return True
        # print(self.cur_obs[code])

    def step(self, obs):
        if self.cur_obs:
            self.last_obs.append(self.cur_obs)
        if len(self.last_obs) > 2:
            self.last_obs = self.last_obs[-2:]
        self.cur_obs = obs
        self.current_date = obs[0].name

    def step_in_day(self, obs):
        self.cur_obs = obs
        self.running_in_day = True


class Agent:
    def __init__(self, market_env: MultiMarketEnv,**args) -> None:
        self.market_env = market_env

        self.db_client = args['db_client']
        self.init_indicators()

        self.current_date = None
        self.pass_month = [1,4]
        self.pool_size = 20

        self.stock_sum = 6


    def signal_indicator(df):

        return df

    def init_indicators(self):
        # self.market_env.add_indicator(Agent.signal_indicator)
        self.market_env.clean_data()

    def select_action(self, code, info):
        ret = 0
        score = 0


        return ret, score

    def get_market_breadth(self,end_date):

        stocks = self.db_client.get_index_stocks("000985",end_date)

        count = 20
        df = self.db_client.get_price(stocks,end_date,["close"],20)
        def calculate_ema(group):
            group['ma20'] = ta.MA(group["close"], timeperiod=count)
            return group
        df = df.groupby(level='code',group_keys=False).apply(calculate_ema)
        df.dropna(inplace=True)

        df["bias"] = df['close'] > df['ma20']

        df.reset_index(level="date",drop=True,inplace=True)

        df["industry"] = self.db_client.get_stock_industry(df.index.to_list(),end_date)
        df = df[df['industry'] != '']
        df = df[["bias","industry"]]
        df.to_csv("test.csv")
        ratio = df.groupby("industry").sum() * 100 /  df.groupby('industry').count()
        return ratio["bias"].nlargest(1).index.to_list()


    def select_stock(self,end_date):
        stocks = self.db_client.get_index_stocks("399101",end_date)

        fin_db = self.db_client.get_stock_fincial(stocks,end_date)

        df = self.db_client.get_price(fin_db.index.to_list(),end_date,["close"],1)
        df.reset_index(level="date",drop=True ,inplace=True)
        fin_db["close"] = df["close"]
        fin_db['market_cap'] = fin_db['close'] * fin_db['total_shares']
        fin_db = fin_db[fin_db['adjusted_profit']>0]
        fin_db = fin_db.sort_values(by="market_cap",ascending=True).iloc[:self.pool_size]
        return fin_db.index.to_list()

    def process_order_value(self,target_value):
        pos_df = self.market_env.account.get_position_price(self.current_date)
        if len(pos_df) > 0:
            pos_df['value'] = pos_df['position'] * pos_df['close']

        # print(pos_df)
        new_price = self.db_client.get_price(list(target_value.keys()),self.current_date,["close"],1)
        new_price.reset_index(level="date",drop=True,inplace=True)
        action_pos = {}
        for code,value in target_value.items():
            if code in pos_df.index:
                new_pos = int(value / pos_df.loc[code,'close'])
                action_pos[code] = new_pos - pos_df.loc[code,'position']
            else:
                action_pos[code] = int(value / new_price.loc[code,'close'])
        
        for code,action in action_pos.items():
            self.create_order(code,action)

    def adjust(self,stocks):
        target = stocks[:min(len(stocks),self.stock_sum)]
        hold_list = list(self.market_env.account.positions[-1].keys())

        target_value = {}
        for stock in hold_list:
            if stock not in target:
                target_value[stock] = 0
        
        total_value = self.market_env.account.get_total_value()

        for code in target:
            target_value[code] = total_value / len(target)

        self.process_order_value(target_value)
        # return target_value

    def action_decider(self, stocks_obs):
        ts = stocks_obs[0].name
        if ts.month in self.pass_month:
            return 
        if ts.weekday() != 4:
            return 

        today = str(ts.date())
        self.current_date = today
        I = self.get_market_breadth(end_date=today)
        # print(I)

        black_industries = {"货币", "煤炭", "开采", "黑色金属"}

        for i in I:
            if any(b in i for b in black_industries):
                return

        self.adjust(self.select_stock(today))
    

    def stock_decider(self, actions):
        buy_list = [v for v in actions if v["info"][0] == 1]
        sell_list = [v for v in actions if v["info"][0] == -1]

        # 按评分排序买入列表和卖出列表
        buy_list = sorted(
            buy_list, key=lambda x: x["info"][1], reverse=True
        )  # 评分高的优先买入
        sell_list = sorted(
            sell_list, key=lambda x: x["info"][1], reverse=True
        )  # 评分高的优先卖出

        # 处理买入信号
        if len(buy_list) > 0:
            logger.info(f"buy list {buy_list}")
            self.create_order(code=global_var.SYMBOLS[buy_list[0]["idx"]], action=1)

        # 处理卖出信号
        if len(sell_list) > 0:
            logger.info(f"sell list {sell_list}")
            for sell in sell_list:
                # 创建卖出订单
                self.create_order(code=global_var.SYMBOLS[sell["idx"]], action=-1)

    def cancel_order(self, code):
        self.market_env.order_manager.cancel_order(code)

    def create_order(self, code, action):
        if action < -self.market_env.min_action:
            self.market_env.order_manager.create_order(code, "sell", abs(action))
        if action > self.market_env.min_action:
            self.market_env.order_manager.create_order(code, "buy", abs(action))


if __name__ == "__main__":
    agent = Agent(None)
    print(agent.get_market_breadth("20250101"))