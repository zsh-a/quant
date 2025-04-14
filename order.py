import os
from typing import List, Type
from loguru import logger
import numpy as np
import pandas as pd
from account import Account, PositionInfo
import global_var
import db

from pyecharts.components import Table
from pyecharts.options import ComponentTitleOpts

import utils
import utils.utils


class Order:
    def __init__(
        self, order_id, symbol, name, order_type, quantity, price=None, exec_time="open"
    ):
        self.order_id = order_id
        self.symbol = symbol
        self.name = name
        self.order_type = order_type  # 'buy' or 'sell'
        self.quantity = quantity
        self.price = price  # Limit price for limit orders
        self.status = "open"  # 'open', 'filled', 'cancelled'
        self.filled_quantity = 0
        self.timestamp = None  # Time when the order was created
        self.execution_price = None

        self.exec_time = exec_time

        self.extra_info = {}

    def __str__(self) -> str:
        return f"Order({self.order_id}, {self.symbol}, {self.name}, {self.order_type}, {self.quantity}, {self.execution_price}, {self.status}, {self.filled_quantity}, {self.timestamp})"

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

        self.completed_position = []

    def step(self, obs):
        self.obs = obs
        self.order_plolicy.step(obs)

    def create_order(self, symbol, order_type, quantity, price=None, exec_time="open"):
        name = utils.utils.get_name(symbol=symbol)
        order = Order(
            self.order_id_counter, symbol, name, order_type, quantity, price, exec_time
        )

        self.orders.append(order)
        logger.info(
            f"create order | datetime : {self.get_current_timestamp()} | symbol : {symbol} | order_id : {order.order_id}  | order_type : {order_type} | quantity : {quantity} | exec_time : {exec_time}"
        )
        self.order_id_counter += 1
        return order

    def cancel_order(self, symbol):
        for od in self.orders:
            if od.symbol == symbol and (od.status == "open" or od.status == "tracking"):
                od.status = "cancelled"
                # logger.info(
                #     f"cancel order | datetime : {self.get_current_timestamp()} | symbol : {od.symbol} | order_id : {od.order_id}  | order_type : {od.order_type}"
                # )

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

        buy_waiting_orders = [
            order
            for order in self.orders
            if order.status == "open" and order.order_type == "buy"
        ]
        sell_waiting_orders = [
            order
            for order in self.orders
            if order.status == "open" and order.order_type == "sell"
        ]
        for order in sell_waiting_orders:
            ok, exec_price = self.order_plolicy.sell_policy(order)
            if ok:
                self.execute_order(
                    order,
                    exec_price,
                    {
                        "high": exec_price,
                        "low": exec_price,
                    },
                )
        for order in buy_waiting_orders:
            ok, exec_price = self.order_plolicy.buy_policy(order)
            if ok:
                self.execute_order(
                    order,
                    exec_price,
                    {
                        "high": exec_price,
                        "low": exec_price,
                    },
                )
        logger.info(
            f"match order...\n{self.account.get_position_price(str(ts.date()))}"
        )

    def execute_order(self, order, execution_price, info):
        order.status = "filled"
        order.filled_quantity = order.quantity
        order.execution_price = execution_price
        order.timestamp = self.get_current_timestamp()

        self.buy_sell_points.append(
            {
                "timestamp": str(order.timestamp.date()),
                "symbol": order.symbol,
                "order_type": order.order_type,
                "quantity": float(order.filled_quantity),
                "price": float(order.execution_price),
            }
            | info
        )
        logger.info(
            f"complete order | datetime : {self.get_current_timestamp()} | order_id : {order.order_id} | symbol : {order.symbol} | order_type : {order.order_type} | price : {order.execution_price} | quantity : {order.filled_quantity}"
        )
        self.order_plolicy.order_callback(order, self)
        self.update_account(order)

    def cancel_all(self):
        for od in self.orders:
            if od.status == "open" or od.status == "tracking":
                od.status = "cancelled"

    def update_account(self, order):
        """
        Update account balance and positions based on the filled order.
        """
        # Implement account and position update logic

        amount = order.execution_price * order.filled_quantity
        if order.order_type == "buy":
            cost = amount * self.account.trading_fee_open
            self.account.cash = self.account.cash - amount - cost
        else:
            cost = amount * self.account.trading_fee_close
            self.account.cash = self.account.cash + amount - cost

        if order.symbol not in self.account.positions[-1]:
            self.account.positions[-1][order.symbol] = PositionInfo(
                order.symbol, 0, order.execution_price, str(order.timestamp.date())
            )
        self.account.positions[-1][order.symbol].quantity += (
            order.filled_quantity
            if order.order_type == "buy"
            else -order.filled_quantity
        )

        if order.order_type == "buy":
            self.account.availables[-1][order.symbol] = False

        if self.account.positions[-1][order.symbol].quantity == 0:
            self.completed_position.append(
                {
                    "symbol": order.symbol,
                    "return": round(
                        (
                            (
                                order.execution_price
                                / self.account.positions[-1][order.symbol].cost_price
                            )
                            - 1
                        )
                        * 100,
                        2,
                    ),
                    "open_time": self.account.positions[-1][order.symbol].timestamp,
                    "close_time": str(order.timestamp),
                }
            )
            del self.account.positions[-1][order.symbol]
            del self.account.availables[-1][order.symbol]

    def get_current_timestamp(self):
        """
        Return the current timestamp in the desired format.
        """
        return self.timestamp

    def get_order_history(self):
        return "\n".join(
            [str(order) for order in self.orders if order.status == "filled"]
        )

    def get_order_stats(self):
        # order_history = [order for order in self.orders if order.status == "filled"]
        # order_history = sorted(order_history, key=lambda x: x.timestamp)
        # it = iter(order_history)

        # order_returns = []

        # order_returns.sort(key=lambda x: x["return"], reverse=True)
        trade_infos = self.completed_position

        win = 0
        loss = 0
        order_history = []
        for order in trade_infos:
            if order["return"] > 0:
                win += 1
            else:
                loss += 1

            order_history.append(
                {
                    "symbol": order["symbol"],
                    "order_return": order["return"],
                    "open_time": pd.to_datetime(order["open_time"]).strftime(
                        "%Y-%m-%d"
                    ),
                    "close_time": pd.to_datetime(order["close_time"]).strftime(
                        "%Y-%m-%d"
                    ),
                }
            )

        return {
            "win": win,
            "loss": loss,
            "order_history": order_history,
            # "total_revenue": sum([order["order_revenue"] for order in order_returns]),
        }

        # headers = ["symbol", "order_revenue", "order_return", "buy_time", "sell_time"]

        # rows = [
        #     [
        #         order["symbol"],
        #         order["order_revenue"],
        #         order["return"],
        #         pd.to_datetime(order["buy_time"]).strftime("%Y-%m-%d"),
        #         pd.to_datetime(order["sell_time"]).strftime("%Y-%m-%d"),
        #     ]
        #     for order in order_returns
        # ]
