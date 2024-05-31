import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from agent import DDQNAgent
from dqn_agent import DQNAgent
from loguru import logger

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAU = 0.005
TARGET_UPDATE = 30

def train():
    env = gym.make("CartPole-v1")
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    print(input_size,output_size)
    hidden_dim = 128
    agent = DQNAgent(input_size, hidden_dim, output_size)
    num_episodes = 10000
    rewards = []
    tot_step = 0
    # print(env.reset())
    # print(env.step(0))
    for episode in range(num_episodes):
        state = env.reset()
        state = state[0]
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            # print(action)
            next_state, reward, done,_,info = env.step(action)
            # assert next_state is not None
            
            # print(next_state)
            # print(next_state)
            tot_step += 1
            agent.buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.train(episode)
            target_net_state_dict = agent.target_net.state_dict()
            policy_net_state_dict = agent.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            agent.target_net.load_state_dict(target_net_state_dict)
        # tot_ret = env.result()
        # market_return_vis.append(tot_ret['market_return'])
        # strategy_return_vis.append(tot_ret['strategy_return'])
        rewards.append(total_reward)
        # if episode % TARGET_UPDATE == 0:
        #     agent.target_net.load_state_dict(agent.policy_net.state_dict())

        # writer.add_scalar("reward", total_reward, episode)
        # writer.add_scalar("market_return", tot_ret['market_return'], episode)
        # writer.add_scalar("strategy_return", tot_ret['strategy_return'], episode)

        if episode % 100 == 0:
            agent.save(f'dqn-{episode}')
        logger.info(f'Episode: {episode+1} | tot_step : {tot_step} | Total Reward: {total_reward}')
        if episode % 100 == 0:
            print('render')
            # env.render()
            try:
                valid(f'dqn-{episode}.pth')
            except Exception as e:
                print(e)

def valid(filename):
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # env.seed(seed)  # 如果环境提供seed()方法

    env = gym.make("CartPole-v1",render_mode="human")
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    # print(input_size,output_size)
    rewards = []    
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    hidden_dim = 128
    agent = DQNAgent(input_size, hidden_dim, output_size,train=False)
    agent.load(filename)
    total_reward = 0
    done = False
    # for episode in range(num_episodes):
    state = env.reset()
    state = state[0]
    total_reward = 0
    done = False
    while not done:
        action = agent.select_action(state)
        # print(action)
        next_state, reward, done,_,info = env.step(action)
        state = next_state
        total_reward += reward
        # agent.train(episode)
    rewards.append(total_reward)

    logger.info(f'vaild =====   | Total Reward: {total_reward}')

if __name__ == '__main__':
    train()