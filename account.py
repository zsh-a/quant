import os
import numpy as np
import pandas as pd

import global_var

from loguru import logger


class PositionInfo:
    def __init__(self, code, quantity, cost_price, timestamp):
        self.code = code
        self.quantity = quantity
        self.cost_price = cost_price
        self.timestamp = timestamp


class Account:
    def __init__(self, init_capital=10000, **args) -> None:
        self.cash = init_capital
        self.init_cash = init_capital
        self.trading_fee_open = 0.0001
        self.trading_fee_close = 0.0006
        self.min_action = 10

        self.positions = [{}]
        self.availables = [{}]
        self.returns = np.zeros(len(global_var.SYMBOLS))
        self.cost_price = np.zeros(len(global_var.SYMBOLS))
        self.cashs = [init_capital]
        self.tot_values = [init_capital]

        self.dates = [0]

        self.current_date = None

        self.db_client = args["db_client"]

    def step(self, obs_list):
        # self.actions.append(np.zeros(len(global_var.SYMBOLS)))
        # self.costs.append(self.costs[-1])
        self.positions.append(self.positions[-1].copy())
        self.availables.append(self.availables[-1].copy())
        for key, value in self.availables[-1].items():
            self.availables[-1][key] = True
        # self.capitals.append(self.capitals[-1])

        # add deep copy self.positions[-1] to self.available
        # self.available.append(self.positions[-1].copy())
        if obs_list:
            # closes = [obs["close"] for obs in obs_list]
            self.dates.append(obs_list[0].name)
            self.current_date = obs_list[0].name

    def run_end(self):
        self.tot_values.append(self.get_total_value())

    def get_position_price(self, date):
        pos_info = self.positions[-1]
        stocks = list(pos_info.keys())

        codes = []
        posotions = []
        for code, val in pos_info.items():
            codes.append(code)
            posotions.append(val.quantity)

        pos_df = pd.DataFrame(
            {
                "code": codes,
                "position": posotions,
            },
            index=stocks,
        )
        if len(pos_df) == 0:
            return pos_df

        df = self.db_client.get_price(stocks, date, ["close"], 1)
        df.reset_index(level="date", drop=True, inplace=True)
        pos_df["close"] = df["close"]
        pos_df["position_value"] = pos_df["position"] * pos_df["close"]
        return pos_df

    def get_position_value(self, date):
        pos_df = self.get_position_price(date)
        if len(pos_df) == 0:
            return 0

        return np.dot(pos_df["position"], pos_df["close"])

    def get_total_value(self):
        if self.current_date is None:
            return self.cash
        return self.get_position_value(str(self.current_date.date())) + self.cash

    def get_position(self, symbol):
        return self.positions[-1][symbol]

    def get_available(self, symbol):
        return self.available[-1][global_var.SYMBOLS.index(symbol)]

    def result(self, risk_free_rate):
        strategy_return = self.tot_values[-1] / self.tot_values[0] - 1

        x_data = [pd.to_datetime(date).strftime("%Y-%m-%d") for date in self.dates[1:]]
        y_data = np.array(self.tot_values[1:]).astype(float).tolist()

        cum_returns = np.array(self.tot_values) / self.tot_values[0]

        rolling_max = np.maximum.accumulate(cum_returns)

        drawdowns = cum_returns / rolling_max - 1

        # 计算最大回撤
        max_drawdown = np.min(drawdowns)

        # 计算年化收益
        start_date = pd.to_datetime(self.dates[1])
        end_date = pd.to_datetime(self.dates[-1])
        years = (end_date - start_date).days / 365
        logger.info(f"run span : {start_date} - {end_date}")
        annualized_return = (self.tot_values[-1] / self.tot_values[0]) ** (
            1 / years
        ) - 1
        return {
            "strategy_return": f"{strategy_return:.2%}",
            "annualized_return": f"{annualized_return:.2%}",
            "max_drawdown": f"{max_drawdown:.2%}",
            "revenue": {"x": x_data, "y": y_data},
        }

        # 计算滚动最小值
        rolling_min = np.minimum.accumulate(cum_returns)

        # 计算盈利
        gains = cum_returns / rolling_min - 1

        # 计算最大盈利
        max_gain = np.max(gains)

        # print(risk_free_rate, np.mean(strategy_return))
        # return {
        #     "strategy_return": strategy_return,
        #     "max_drawdown": f"{max_drawdown:.2%}",
        #     "max_profit": f"{max_gain:.2%}",
        #     "code_returns": {
        #         code: ret
        #         for code, ret in zip(global_var.SYMBOLS, np.array(self.returns))
        #     },
        #     "sharpe_ratio": (np.mean(strategy_return) - risk_free_rate)
        #     / np.std(strategy_return),
        #     "revenue": {"x": x_data, "y": y_data},
        # }
