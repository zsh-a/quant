from collections import deque
from random import sample
import random
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from loguru import logger
import torch.optim as optim

import matplotlib.pyplot as plt


from market_env import MarketEnv

from utils import *

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

BATCH_SIZE = 32

GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4


# class DQN(nn.Module):
#     def __init__(self, n_observations, n_actions):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(n_observations, 128)
#         self.fc2 = nn.Linear(128, n_actions)
    
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)
    
class DQN(nn.Module):
    def __init__(self, n_observations,n_actions) -> None:
        super(DQN,self).__init__()
        self.fc1 = nn.Linear(n_observations,128)
        self.fc2 = nn.Linear(128,128)

        self.fc3 = nn.Linear(128,n_actions)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self,x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.dropout(x,0.1)
        x = self.fc3(x)
        return x

    

# 定义经验回放缓冲区
class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self,state_dim,hidden_dim,num_actions,train=True) -> None:


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQN(state_dim, num_actions).to(self.device)
        self.target_net = DQN(state_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.buffer = ReplayBuffer(capacity=1000)
        self.num_actions = num_actions
        self.total_steps = 0 
        self.gamma = 0.999  # 折扣因子
        self.epsilon = 0.9 if train else 0.0 # 初始ε贪婪值 
        self.epsilon_decay = 0.9 # ε贪婪值衰减率
        self.epsilon_min = 0.05  # 最小ε贪婪值
        self.batch_size = BATCH_SIZE
        self.target_update = 30  # 目标网络更新频率
        self.losses = [0]
        self.train_mode = train

        self.optimizer = optim.RMSprop(self.policy_net.parameters())

    def select_action(self, state):
        # self.total_steps += 1
        eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * self.total_steps / EPS_DECAY)
        # print(eps_threshold)
        self.total_steps +=1 
        print(eps_threshold)
        if self.train_mode and np.random.rand() < eps_threshold:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.max(1)[1].item()
    
    def train(self,episode):
        if len(self.buffer) < self.batch_size:
            return
        transitions = self.buffer.sample(self.batch_size)
        batch = map(np.array,zip(*transitions))
        states, actions, rewards, next_states, done = batch

        state_batch = torch.FloatTensor(states).to(self.device)
        action_batch = torch.LongTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.FloatTensor(next_states).to(self.device)
        
        done_batch = torch.BoolTensor(done).to(self.device)


        # # print(self.policy_net(state_batch).size(),action_batch.size())
        # # print(action_batch.unsqueeze(1))
        # state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # # print(state_action_values)
        # next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        
        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
        #                                   next_state_batch)), dtype=torch.bool)
        # non_final_next_states = torch.cat([s for s in next_state_batch
        #                                         if s is not None])
        # print(non_final_next_states.size())
        # with torch.no_grad():
        #     next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        
        
        # batch = list(zip(*transitions))
        # print(batch[3])
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
        #                                     batch[3])), dtype=torch.bool)
        # non_final_next_states = torch.cat([torch.tensor(s) for s in batch[3]
        #                                             if s is not None])
        
        # non_final_next_states = non_final_next_states.view(-1,4)
        # state_batch = torch.cat([torch.tensor(batch[0])])
        
        # action_batch = torch.cat([torch.tensor(batch[1])])
        # reward_batch = torch.cat([torch.tensor(batch[2])])
        # done_batch = torch.cat([torch.tensor(batch[4])])


        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE)
        with torch.no_grad():
            next_state_values[~done_batch] = self.target_net(next_state_batch[~done_batch]).max(1).values
        
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # criterion = nn.SmoothL1Loss()
        # loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.losses.append(loss.item())
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self,episode):
        if len(self.buffer) < self.batch_size:
            return

        transitions = self.buffer.sample(self.batch_size)
        batch = map(np.array,zip(*transitions))
        states, actions, rewards, next_states, done = batch

        state_batch = torch.FloatTensor(states).to(self.device)
        action_batch = torch.LongTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.FloatTensor(next_states).to(self.device)
        
        done_batch = torch.BoolTensor(done).to(self.device)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        next_state_values = torch.zeros(BATCH_SIZE)
        with torch.no_grad():
            next_state_values[~done_batch] = self.target_net(next_state_batch[~done_batch]).max(1).values
        
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # criterion = nn.SmoothL1Loss()
        # loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.losses.append(loss.item())
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
    def load(self,name):
        self.policy_net = torch.load(name).to(self.device)
        self.target_net = torch.load(name).to(self.device)
                
    def save(self,name):
        torch.save(self.target_net, f'{name}.pth')

