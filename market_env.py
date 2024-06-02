from typing import Any, SupportsFloat, Tuple
import gymnasium as gym
import numpy as np
from data_source import DataSource

from utils import *
INF = 1e9
class MarketEnv(gym.Env):
    def __init__(self,num_step,code='510880',start_date='20140701',end_date='20160901') -> None:
        super().__init__()
        
        self.data_source = DataSource(code=code,trading_days=num_step,start_date=start_date,end_date=end_date)

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = self.observation_space = gym.spaces.Box(low = np.array(self.data_source.min_values),
                                            high = np.array(self.data_source.max_values))
        
        self.num_step = num_step

        self.trading_cost_bps = 1e-3
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
        assert self.action_space.contains(action)
        self.actions[self.cur_step] = action - 1
        start_position = self.positions[self.cur_step - 1]
        position = start_position + action - 1
        obs,done,ori_obs = self.data_source.step()
        if position >= 0 and position <=1:
            n_trades = action - 1
        else:
            n_trades = 0
        position = np.clip(position,0,1)
        self.positions[self.cur_step] = position
        self.trades[self.cur_step] = n_trades
        trade_cost = abs(n_trades) * self.trading_cost_bps
        self.costs[self.cur_step] = trade_cost
        market_return = ori_obs['returns']
        self.market_returns[self.cur_step] = market_return
        # reward
        # reward = start_position * market_return - self.costs[self.cur_step - 1]
        reward = position * market_return
        if position == 0.:
            reward = -market_return
        
        self.strategy_returns[self.cur_step] = position * market_return
        
        info = {
            'reward': reward,
            'costs': self.costs[self.cur_step],
            'ori_obs' : ori_obs
        }
        self.cur_step += 1
        return obs.values,reward,done,info



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
        return obs.values


if __name__ == '__main__':
    env = MarketEnv(252)
    env.reset()
    print(env.step(2))
    print(env.step(1))
    print(env.step(1))
    print(env.step(0))

    