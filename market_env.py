from typing import Any, SupportsFloat, Tuple
import gymnasium as gym
import numpy as np
from data_source import DataSource
import matplotlib.pyplot as plt


from utils import *
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


    def step(self, action: Any) -> Tuple[Any | SupportsFloat | bool | dict[str, Any]]:
        # assert self.action_space.contains(action)
        obs,done,ori_obs = self.data_source.step()

        action = int(action * self.max_stake)
        position = self.positions[self.cur_step - 1]
        cost = 0
        if action < -self.min_action:
            num_stakes = min(self.positions[self.cur_step - 1],-action)
            position -= num_stakes
            amount = ori_obs['open'] * num_stakes
            cost = amount * self.trading_cost_bps
            self.capital = self.capital + amount - cost

        if action > self.min_action:
            num_stakes = min(self.capital // ori_obs['open'],action)
            amount = ori_obs['open'] * num_stakes
            cost = amount * self.trading_cost_bps
            if self.capital >= amount + cost:
                position += num_stakes
                self.capital = self.capital - amount - cost

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
        return (obs.values,position),reward,done,info



    def result(self):
        return {
            "market_return" : log2percent(np.exp(sum(self.market_returns))),
            "strategy_return" : log2percent(np.exp(sum(self.strategy_returns))),
            "max_drawdown": log2percent(np.exp(np.min(np.cumsum(self.strategy_returns)))),
            "max_profit": log2percent(np.exp(np.max(np.cumsum(self.strategy_returns)))),
            "actions":self.actions,
            'positions': self.positions,
            'actions':self.actions,
            'strategy_returns': self.strategy_returns,
            'market_returns': self.market_returns
        }

    def plot(self):
        fig, (ax1, ax2, ax3,ax4) = plt.subplots(4,1, figsize=(10, 6))
        ax1.plot(self.positions[:self.cur_step], label='Position')
        ax1.set_ylabel("Position/stake")

        ax2.plot(self.tot_values[:self.cur_step], label='Value')
        ax2.set_ylabel("Value/CNY")

        ax3.plot(np.diff(self.tot_values[:self.cur_step]) / self.tot_values[:self.cur_step - 1] * 100, label='Value')
        ax3.set_ylabel("Returns/%")
        print(self.capitals[:self.cur_step])
        ax4.plot(self.capitals[:self.cur_step], label='Capital')
        ax4.set_ylabel("Capital/CNY")

        ax1.legend(loc='best')
        # ax1.set_title("Market vs Strategy Returns")``
        # ax1.set_ylabel("Returns/%")
        ax1.grid(True)
        ax2.grid(True)
        ax3.grid(True)
        ax4.grid(True)
        plt.savefig('trade_result.png', dpi=300, bbox_inches='tight')

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
        return (obs.values,0)


if __name__ == '__main__':
    env = MarketEnv(252)
    env.reset()
    print(env.step(1))
    print(env.step(0.05))
    print(env.step(0.05))
    print(env.step(0.05))
    print(env.step(0.05))
    print(env.step(0.05))
    print(env.step(-0.1))

    env.plot()