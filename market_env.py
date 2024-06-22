from typing import Any, SupportsFloat, Tuple
import gymnasium as gym
import numpy as np
from data_source import DataSource
import matplotlib.pyplot as plt

from utils import log2percent

INF = 1e9
class MarketEnv(gym.Env):
    def __init__(self,num_step,code='510880',start_date='20100531',end_date='20110901',work_dir='',initial_capital=100000,max_stake=10000) -> None:
        super().__init__()
        
        self.data_source = DataSource(code=code,trading_days=num_step,start_date=start_date,end_date=end_date,work_dir=work_dir)

        self.action_space = gym.spaces.Box(low=-1, high=1, dtype=np.float32)
        self.observation_space = self.observation_space = gym.spaces.Box(low = np.array(self.data_source.min_values),
                                            high = np.array(self.data_source.max_values))
        
        self.capital = initial_capital
        self.max_stake = max_stake
        self.num_step = num_step
        self.tot_values = self.capital
        self.min_action = 100
        
        self.tot_values = np.zeros(self.num_step + 1)
        self.tot_values[0] = self.capital

        self.capitals = np.zeros(self.num_step + 1)
        self.capitals[0] = self.capital
        self.trading_cost_bps = 1e-4
        # state
        self.cur_step = 1
        self.actions = np.zeros(self.num_step + 1)
        self.positions = np.zeros(self.num_step + 1)
        self.navs = np.ones(self.num_step + 1)
        self.trades = np.zeros(self.num_step + 1)
        self.costs = np.zeros(self.num_step + 1)
        self.strategy_returns = np.zeros(self.num_step + 1)
        self.market_returns = np.zeros(self.num_step + 1)


        self.buy_sell_points = []

        self.last_obs = None

        self.ordering = 0

        self.waiting_buy = 0

    def step(self, action: Any) -> Tuple[Any | SupportsFloat | bool | dict[str, Any]]:
        # assert self.action_space.contains(action)
        obs,done,ori_obs = self.data_source.step()
        # if self.ordering != 0 and self.ordering * action >= 0:
        #     action = self.ordering
        # else:
        #     self.ordering = 0
        if action * self.ordering < 0:
            self.ordering = 0
        if self.ordering != 0:
            action = self.ordering

        self.market_returns[self.cur_step] = ori_obs['returns']

        action = int(action * self.max_stake)
        position = self.positions[self.cur_step - 1]
        cost = 0
        trading_price = ori_obs['open']
        if action < -self.min_action:
            if ori_obs['low'] > self.last_obs['low']:
                num_stakes = min(self.positions[self.cur_step - 1],-action)
                if num_stakes > 0:
                    trading_price = min(self.last_obs['low'],ori_obs['open'])
                    position -= num_stakes
                    amount = trading_price * num_stakes
                    cost = amount * self.trading_cost_bps
                    self.capital = self.capital + amount - cost
                    self.ordering = 0
                    self.buy_sell_points.append((ori_obs.name,'sell'))
            else:
                self.ordering = -1

        if action > self.min_action:
            if ori_obs['high'] > self.last_obs['high']:
                trading_price = max(self.last_obs['high'],ori_obs['open'])
                num_stakes = min(self.capital // trading_price // 100 * 100,action)
                if num_stakes > 0:
                    amount = trading_price * num_stakes
                    cost = amount * self.trading_cost_bps
                    if self.capital >= amount + cost:
                        position += num_stakes
                        self.capital = self.capital - amount - cost
                        self.ordering = 0
                        self.buy_sell_points.append((ori_obs.name,'buy'))
            else:
                self.ordering = 1

        self.actions[self.cur_step] = action

        self.costs[self.cur_step] = cost
        self.positions[self.cur_step] = position
        self.capitals[self.cur_step] = self.capital

        self.tot_values[self.cur_step] = self.capital + self.positions[self.cur_step] * ori_obs['close']

        reward = self.tot_values[self.cur_step] - self.tot_values[self.cur_step - 1]
        info = {
            'reward': reward,
            'costs': self.costs[self.cur_step],
            'ori_obs' : ori_obs
        }
        self.cur_step += 1
        self.last_obs = ori_obs
        return (obs.values,position),reward,done,info



    def result(self):
        return {
            "market_return" : log2percent(np.exp(sum(self.market_returns))),
            "strategy_return" : round((self.tot_values[self.cur_step - 1] - self.tot_values[0]) / self.tot_values[0] * 100,2),
            "max_drawdown": log2percent(np.exp(np.min(np.cumsum(self.strategy_returns)))),
            "max_profit": log2percent(np.exp(np.max(np.cumsum(self.strategy_returns)))),
            "actions":self.actions,
            'positions': self.positions,
            'actions':self.actions,
            'strategy_returns': self.strategy_returns,
            'market_returns': self.market_returns
        }

    def plot(self):
        fig, (ax1, ax2, ax3,ax4,ax5,ax6) = plt.subplots(6,1, figsize=(15, 20))
        ax1.plot(self.positions[:self.cur_step], label='Position')
        ax1.set_ylabel("Position/stake")

        ax2.plot(self.tot_values[:self.cur_step], label='Value')
        ax2.set_ylabel("Value/CNY")

        ax3.plot((self.tot_values[:self.cur_step] - self.tot_values[0]) / self.tot_values[0] * 100, label='Strategy Return')
        ax3.set_ylabel("Strategy Returns/%")
        ax4.plot(self.capitals[:self.cur_step], label='Capital')
        ax4.set_ylabel("Capital/CNY")

        ax5.plot(log2percent(np.exp(np.cumsum(self.market_returns))), label='Market Return')
        ax5.set_ylabel("Market Return/%")


        ax1.legend(loc='best')
        # ax1.set_title("Market vs Strategy Returns")``
        # ax1.set_ylabel("Returns/%")
        ax1.grid(True)
        ax2.grid(True)
        ax3.grid(True)
        ax4.grid(True)
        ax5.grid(True)

        self.data_source.plot(self.buy_sell_points)
        plt.show()
        # plt.savefig('trade_result.png', dpi=300, bbox_inches='tight')


    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[Any, dict]:
        self.cur_step = 1
        self.actions.fill(0)
        self.navs.fill(1)
        self.strategy_returns.fill(0)
        self.costs.fill(0)
        self.trades.fill(0)
        self.market_returns.fill(0)
        self.data_source.reset()
        obs,done,ori_obs = self.data_source.step()
        info = {
            "ori_obs":ori_obs
        }
        return (obs.values,0),info


if __name__ == '__main__':
    env = MarketEnv(220,start_date='20230401',end_date='20240401')
    env.reset()
    print(env.step(1))
    print(env.step(0.05))
    print(env.step(0.05))
    print(env.step(0.05))
    print(env.step(0.05))
    print(env.step(0.05))
    print(env.step(-0.1))

    env.plot()