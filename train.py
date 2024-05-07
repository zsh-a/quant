import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 定义双重深度Q网络（DDQN）
class DDQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(DDQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3_adv = nn.Linear(hidden_size, output_size)
        self.fc3_val = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        adv = self.fc3_adv(x)
        val = self.fc3_val(x)
        return val + adv - adv.mean()

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

# 定义DDQN Agent
class DDQNAgent():
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.buffer = ReplayBuffer(capacity=10000)
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 初始ε贪婪值
        self.epsilon_decay = 0.995  # ε贪婪值衰减率
        self.epsilon_min = 0.01  # 最小ε贪婪值
        self.batch_size = 64
        self.target_update = 100  # 目标网络更新频率
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DDQN(input_size, output_size).to(self.device)
        self.target_net = DDQN(input_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.output_size)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.max(1)[1].item()
        
    def train(self):
        if len(self.buffer) < self.batch_size:
            return
        
        transitions = self.buffer.sample(self.batch_size)
        batch = list(zip(*transitions))
        state_batch = torch.FloatTensor(batch[0]).to(self.device)
        action_batch = torch.LongTensor(batch[1]).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).to(self.device)
        next_state_batch = torch.FloatTensor(batch[3]).to(self.device)
        done_batch = torch.BoolTensor(batch[4]).to(self.device)
        
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + self.gamma * next_q_values * ~done_batch
        
        loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        if self.target_update % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_update += 1

# 创建CartPole环境
# env = gym.make('CartPole-v1')
# input_size = env.observation_space.shape[0]
# output_size = env.action_space.n

# # 初始化DDQN Agent
# agent = DDQNAgent(input_size, output_size)

# # 训练
# num_episodes = 1000
# for episode in range(num_episodes):
#     state = env.reset()
#     total_reward = 0
#     done = False
#     while not done:
#         action = agent.select_action(state)
#         next_state, reward, done,_,_ = env.step(action)
#         agent.buffer.push(state, action, reward, next_state, done)
#         state = next_state
#         total_reward += reward
#         agent.train()
#     print(f"Episode: {episode+1}, Total Reward: {total_reward}")


