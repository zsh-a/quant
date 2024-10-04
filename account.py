import os
import numpy as np
import pandas as pd

import global_var
from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.charts import Kline, Line, Grid, Scatter


class Account:
    def __init__(self, init_capital=10000) -> None:
        self.capital = init_capital
        self.trading_cost_bps = 1e-4
        self.available = [np.zeros(len(global_var.SYMBOLS))]
        self.min_action = 100

        self.actions = [np.zeros(len(global_var.SYMBOLS))]

        self.costs = [0]
        self.positions = [np.zeros(len(global_var.SYMBOLS))]
        self.returns = np.zeros(len(global_var.SYMBOLS))
        self.cost_price = np.zeros(len(global_var.SYMBOLS))
        self.capitals = [init_capital]
        self.tot_values = [init_capital]

        self.dates = [0]

    def step(self, ori_obs):
        self.actions.append(np.zeros(len(global_var.SYMBOLS)))
        self.costs.append(self.costs[-1])
        self.positions.append(self.positions[-1].copy())
        self.capitals.append(self.capitals[-1])
        closes = [info["close"] for info in ori_obs]
        self.tot_values.append(np.dot(self.positions[-1], closes) + self.capitals[-1])

        # add deep copy self.positions[-1] to self.available
        self.available.append(self.positions[-1].copy())
        if ori_obs:
            self.dates.append(ori_obs[0].name)

    def get_position(self, symbol):
        return self.positions[-1][global_var.SYMBOLS.index(symbol)]

    def get_available(self, symbol):
        return self.available[-1][global_var.SYMBOLS.index(symbol)]

    def result(self, risk_free_rate):
        strategy_return = (
            np.array(self.tot_values) - self.tot_values[0]
        ) / self.tot_values[0]
        # print(risk_free_rate, np.mean(strategy_return))
        return {
            "strategy_return": round(strategy_return[-1] * 100, 2),
            "max_drawdown": round(
                (min(self.tot_values) / self.tot_values[0] - 1) * 100, 2
            ),
            "max_profit": round(
                (max(self.tot_values) / self.tot_values[0] - 1) * 100, 2
            ),
            "code_returns": {
                code: ret
                for code, ret in zip(global_var.SYMBOLS, np.array(self.returns))
            },
            "sharpe_ratio": (np.mean(strategy_return) - risk_free_rate)
            / np.std(strategy_return),
        }

    def plot(self):
        x_data = [pd.to_datetime(date).strftime("%Y-%m-%d") for date in self.dates[1:]]

        # 创建折线图
        line = (
            Line()
            .add_xaxis(x_data)  # 添加X轴数据
            .add_yaxis(
                "收益",
                np.array(self.tot_values).astype(float).tolist(),
                label_opts=opts.LabelOpts(is_show=False),
            )  # 添加Y轴数据
            .set_global_opts(
                title_opts=opts.TitleOpts(title="收益曲线"),
                xaxis_opts=opts.AxisOpts(name="月份"),
                yaxis_opts=opts.AxisOpts(name="收益 (元)"),
                tooltip_opts=opts.TooltipOpts(trigger="axis"),
                legend_opts=opts.LegendOpts(is_show=True),
                datazoom_opts=[
                    opts.DataZoomOpts(
                        type_="inside",
                        xaxis_index=[0],  # 影响两个图的x轴
                        range_start=0,
                        range_end=100,
                    ),
                    opts.DataZoomOpts(
                        type_="slider",
                        xaxis_index=[0],  # 滑动缩放器同时影响两个图的x轴
                        range_start=0,
                        range_end=100,
                    ),
                ],
            )
        )

        # 渲染图表到HTML文件
        line.render(os.path.join("gen", "revenue_curve.html"))
