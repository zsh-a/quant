from collections import deque
from random import sample
import random
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from loguru import logger
import matplotlib.pyplot as plt


from market_env import MarketEnv

from utils.utils import *

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

class QNet(nn.Module):
    def __init__(self, state_dim,hidden_dim=128,num_actions=3) -> None:
        super(QNet,self).__init__()
        self.fc1 = nn.Linear(state_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)

        self.fc3 = nn.Linear(hidden_dim,num_actions)

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

class DDQNAgent:
    def __init__(self,state_dim,hidden_dim,num_actions,batch_size,train=True) -> None:


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = QNet(state_dim,hidden_dim,num_actions).to(self.device)
        self.target_net = QNet(state_dim,hidden_dim,num_actions).to(self.device)

        self.buffer = ReplayBuffer(capacity=500)
        self.num_actions = num_actions
        self.total_steps = 0
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0 if train else 0.0 # 初始ε贪婪值 
        self.epsilon_decay = 0.99998  # ε贪婪值衰减率
        self.epsilon_min = 0.2  # 最小ε贪婪值
        self.batch_size = batch_size
        self.target_update = 100  # 目标网络更新频率
        self.losses = []


        self.learning_rate = 0.001
        self.weight_decay = 0.001
        self.mseloss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), 
                                    lr=self.learning_rate,
                                    weight_decay=self.weight_decay)

    def select_action(self, state):
        self.total_steps += 1
        if np.random.rand() < self.epsilon:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.max(1)[1].item()
    
    def train(self,episode):
        if len(self.buffer) < self.batch_size:
            return

        transitions = self.buffer.sample(self.batch_size)
        
        # batch = list(zip(*transitions))
        # print(batch[0])
        batch = map(np.array,zip(*transitions))
        # print(transitions[-1])
        # print(batch.__next__())
        states, actions, rewards, next_states, done = batch
        # print(states,actions,rewards)
        state_batch = torch.FloatTensor(states).to(self.device)
        action_batch = torch.LongTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.FloatTensor(next_states).to(self.device)
        done_batch = torch.BoolTensor(done).to(self.device)

        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        q_eval_next = torch.max(self.policy_net(next_state_batch),1)[1].view(self.batch_size,1)
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch)
            q_e = next_q_values.gather(1,q_eval_next)
            expected_q_values = reward_batch.unsqueeze(1) + self.gamma * q_e * ~done_batch.unsqueeze(1)

        # loss = F.smooth_l1_loss(current_q_values, expected_q_values)
        loss = self.mseloss(current_q_values,expected_q_values)
        self.losses.append(loss.detach().cpu())
        writer.add_scalar("Loss/train", loss, episode)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        writer.add_histogram('fc1 weights', self.policy_net.fc1.weight, episode)
        writer.add_histogram('fc1 biases', self.policy_net.fc1.bias, episode)
        writer.add_histogram('fc1 weight gradients', self.policy_net.fc1.weight.grad, episode)
        writer.add_histogram('fc1 bias gradients', self.policy_net.fc1.bias.grad, episode)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        if self.total_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.total_steps += 1


    def load(self,name):
        self.policy_net = torch.load(name).to(self.device)
        self.target_net = torch.load(name).to(self.device)
                
    def save(self,name):
        torch.save(self.target_net, f'{name}.pth')

