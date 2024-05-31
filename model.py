from collections import deque
from random import sample
import random
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from loguru import logger
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
import os
from market_env import MarketEnv
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from utils import *

def train():
    env = MarketEnv(252,start_date='20101001',end_date='20230401')
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    print(input_size,output_size)
    hidden_dim = 128
    agent = DQNAgent(input_size, hidden_dim, output_size)
    num_episodes = 10000
    market_return_vis = []
    strategy_return_vis = []
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done,info = env.step(action)
            # print(reward)
            agent.buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.train(episode)
        if episode % 100 == 0:
            agent.save(f'dqn-{episode}')
        tot_ret = env.result()
        market_return_vis.append(tot_ret['market_return'])
        strategy_return_vis.append(tot_ret['strategy_return'])
        rewards.append(total_reward)
        writer.add_scalar("reward", total_reward, episode)
        writer.add_scalar("market_return", tot_ret['market_return'], episode)
        writer.add_scalar("strategy_return", tot_ret['strategy_return'], episode)
        logger.info(f'Episode: {episode+1} | loss : {agent.losses[-1]} | Total Reward: {total_reward} | market_return: {tot_ret['market_return'] }% | strategy_return: {tot_ret['strategy_return']}%')
        if episode % 100 == 0:
            valid(f'dqn-{episode}.pth')

    # 创建包含2行1列的子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6))

    ax1.plot(market_return_vis, label='Market Return')
    ax1.plot(strategy_return_vis, label='Strategy Return')
    ax1.legend(loc='best')
    ax1.set_title("Market vs Strategy Returns")
    ax1.set_ylabel("Returns")
    ax1.grid(True)

    ax2.plot(agent.losses, label='loss')
    ax2.set_title("loss")
    ax2.set_ylabel("loss")
    ax2.grid(True)

    ax3.plot(rewards, label='loss')
    ax3.set_title("reward")
    ax3.set_ylabel("reward")
    ax3.grid(True)
    
    plt.savefig('return_comparison.png', dpi=300, bbox_inches='tight')
    # 显示图形
    # plt.show()


def valid(filename):
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # env.seed(seed)  # 如果环境提供seed()方法

    env = MarketEnv(200,start_date='20230401',end_date='20240401')
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    hidden_dim = 128
    agent = DQNAgent(input_size, hidden_dim, output_size,train=False)
    agent.load(filename)
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done,_ = env.step(action)
        # print(reward)
        # agent.buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    ret = env.result()
    logger.info(f'Total Reward: {total_reward} | market_return: {ret['market_return'] }% | strategy_return: {ret['strategy_return']}% | max_drawdown : {ret['max_drawdown']} | max_profit : {ret['max_profit']} |  position : {np.array(ret['positions'])} | actions : {np.array(ret['actions'])}')

def base_policy():
    class BaseAgent:

        def __init__(self) -> None:
            self.last_state = None

        def select_action(self, state):
            val = state[2]
            if self.last_state is None:
                self.last_state = val
                return 1

            if val > self.last_state:
                return 2
            else:
                return 0

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # env.seed(seed)  # 如果环境提供seed()方法
    agent = BaseAgent()
    env = MarketEnv(240,start_date='20230401',end_date='20240401')
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done,_ = env.step(action)
        # print(reward)
        # agent.buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    ret = env.result()

    # print(ret['market_returns'],ret['strategy_returns'])
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 6))
    ax1.plot(log2percent(np.exp(np.cumsum(ret['market_returns']))), label='Market Return')
    ax1.plot(log2percent(np.exp(np.cumsum(ret['strategy_returns']))), label='Strategy Return')
    ax1.legend(loc='best')
    ax1.set_title("Market vs Strategy Returns")
    ax1.set_ylabel("Returns/%")
    ax1.grid(True)
    plt.savefig('return_comparison.png', dpi=300, bbox_inches='tight')

    logger.info(f'Total Reward: {total_reward} | market_return: {ret['market_return'] }% | strategy_return: {ret['strategy_return']}% | max_drawdown : {ret['max_drawdown']} | max_profit : {ret['max_profit']} |  position : {np.array(ret['positions'])} | actions : {np.array(ret['actions'])}')


if __name__ == '__main__':
    # valid("ddqn-600.pth")
    train()
    # base_policy()