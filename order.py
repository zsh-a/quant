import os
from typing import List, Type
from loguru import logger
import numpy as np
import pandas as pd
from account import Account
import global_var

from pyecharts.components import Table
from pyecharts.options import ComponentTitleOpts


class Order:
    def __init__(self, order_id, symbol, order_type, quantity, price=None):
        self.order_id = order_id
        self.symbol = symbol
        self.order_type = order_type  # 'buy' or 'sell'
        self.quantity = quantity
        self.price = price  # Limit price for limit orders
        self.status = "open"  # 'open', 'filled', 'cancelled'
        self.filled_quantity = 0
        self.timestamp = None  # Time when the order was created
        self.execution_price = None

    def __str__(self) -> str:
        return f"Order({self.order_id}, {self.symbol}, {self.order_type}, {self.quantity}, {self.execution_price}, {self.status}, {self.filled_quantity}, {self.timestamp})"

    def __repr__(self) -> str:
        return self.__str__()


class OrderManager:
    def __init__(self, account: Account, order_policy):
        self.orders: List[Type[Order]] = []
        self.order_id_counter = 1

        self.order_plolicy = order_policy
        self.account = account

        self.buy_sell_points = []
        self.timestamp = None

        self.obs = None

        self.today_traded = False

    def step(self, obs):
        self.obs = obs
        self.today_traded = False
        self.order_plolicy.step(obs)

    def create_order(self, symbol, order_type, quantity, price=None):
        order = Order(self.order_id_counter, symbol, order_type, quantity, price)
        for od in self.orders:
            if od.status == "tracking":
                od.status = "cancelled"
                logger.info(
                    f"cancel order | datetime : {self.get_current_timestamp()} | symbol : {od.symbol} | order_id : {od.order_id}  | order_type : {od.order_type}"
                )

        self.orders.append(order)
        logger.info(
            f"create order | datetime : {self.get_current_timestamp()} | symbol : {symbol} | order_id : {order.order_id}  | order_type : {order_type}"
        )
        self.order_id_counter += 1
        return order

    def cancel_order(self, symbol):
        for od in self.orders:
            if od.status == "tracking" and od.symbol == symbol:
                od.status = "cancelled"
                logger.info(
                    f"cancel order | datetime : {self.get_current_timestamp()} | symbol : {od.symbol} | order_id : {od.order_id}  | order_type : {od.order_type}"
                )

    def get_waiting_order(self):
        return [
            order
            for order in self.orders
            if order.status == "open" or order.status == "tracking"
        ]

    def match_orders(self, market_data):
        """
        Match open orders with the latest market data.
        """

        assert len(market_data) > 0
        ts = market_data[0].name
        self.timestamp = ts

        for order in self.orders:
            if order.status == "open" or order.status == "tracking":
                if order.order_type == "buy":
                    ok, exec_price = self.order_plolicy.buy_policy(order)
                    if ok:
                        self.execute_order(order, exec_price)
                elif order.order_type == "sell" or order.order_type == "stop":
                    ok, exec_price = self.order_plolicy.sell_policy(order)
                    if ok:
                        self.execute_order(order, exec_price)

    def execute_order(self, order, execution_price):
        if self.today_traded:
            return
        order.status = "filled"
        order.filled_quantity = max(
            order.quantity, self.account.get_position(order.symbol)
        )
        order.execution_price = execution_price
        order.timestamp = self.get_current_timestamp()

        self.buy_sell_points.append(
            {
                "timestamp": order.timestamp,
                "symbol": order.symbol,
                "order_type": order.order_type,
            }
        )
        logger.info(
            f"complete order | datetime : {self.get_current_timestamp()} | order_id : {order.order_id} | symbol : {order.symbol} | order_type : {order.order_type} | price : {order.execution_price} | quantity : {order.filled_quantity}"
        )
        self.order_plolicy.order_callback(order, self)
        self.update_account(order)
        self.today_traded = True

    def update_account(self, order):
        """
        Update account balance and positions based on the filled order.
        """
        # Implement account and position update logic

        idx = global_var.SYMBOLS.index(order.symbol)
        amount = order.execution_price * order.filled_quantity
        cost = amount * self.account.trading_cost_bps
        if order.order_type == "buy":
            self.account.capital = self.account.capital - amount - cost
            self.account.cost_price[idx] = order.execution_price
        else:
            self.account.capital = self.account.capital + amount - cost
            self.account.returns[idx] += (
                order.execution_price - self.account.cost_price[idx]
            ) * order.filled_quantity

        self.account.capitals[-1] = self.account.capital
        self.account.actions[-1][idx] = (
            order.filled_quantity
            if order.order_type == "buy"
            else -order.filled_quantity
        )

        self.account.positions[-1][idx] += self.account.actions[-1][idx]
        self.account.tot_values[-1] = self.account.capital + np.sum(
            [
                self.account.positions[-1][idx] * info["close"]
                for idx, info in enumerate(self.obs)
            ]
        )
        self.account.available[-1][idx] = 0

    def get_current_timestamp(self):
        """
        Return the current timestamp in the desired format.
        """
        return self.timestamp

    def get_order_history(self):
        return "\n".join(
            [str(order) for order in self.orders if order.status == "filled"]
        )

    def plot(self):
        order_history = [order for order in self.orders if order.status == "filled"]
        order_history = sorted(order_history, key=lambda x: x.timestamp)
        it = iter(order_history)

        order_returns = []

        for buy, sell in zip(it, it):
            assert (
                buy.symbol == sell.symbol
            ), f"{buy.symbol} {buy.timestamp} != {sell.symbol} {sell.timestamp}"
            order_return = (
                sell.execution_price - buy.execution_price
            ) * sell.filled_quantity
            order_returns.append(
                {
                    "symbol": buy.symbol,
                    "order_revenue": order_return,
                    "return": round(
                        (sell.execution_price / buy.execution_price - 1) * 100,
                        2,
                    ),
                    "buy_time": buy.timestamp,
                    "sell_time": sell.timestamp,
                }
            )
        order_returns.sort(key=lambda x: x["return"], reverse=True)

        # 创建表格
        table = Table()

        # 表头数据
        headers = ["symbol", "order_revenue", "order_return", "buy_time", "sell_time"]

        rows = [
            [
                order["symbol"],
                order["order_revenue"],
                order["return"],
                pd.to_datetime(order["buy_time"]).strftime("%Y-%m-%d"),
                pd.to_datetime(order["sell_time"]).strftime("%Y-%m-%d"),
            ]
            for order in order_returns
        ]

        # 向表格中添加数据
        table.add(headers, rows)

        # 设置表格标题
        table.set_global_opts(title_opts=ComponentTitleOpts(title="交易订单"))

        # 渲染表格为HTML文件
        table.render(os.path.join("gen", "order_table.html"))
