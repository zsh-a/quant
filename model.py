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

class QNet(nn.Module):
    def __init__(self, state_dim,hidden_dim=128,num_actions=3) -> None:
        super(QNet,self).__init__()
        self.fc1 = nn.Linear(state_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)

        self.fc3 = nn.Linear(hidden_dim,num_actions)

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
    def __init__(self,state_dim,hidden_dim,num_actions) -> None:


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = QNet(state_dim,hidden_dim,num_actions).to(self.device)
        self.target_net = QNet(state_dim,hidden_dim,num_actions).to(self.device)

        self.buffer = ReplayBuffer(capacity=10000)
        self.num_actions = num_actions
        self.total_steps = 0
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 初始ε贪婪值
        self.epsilon_decay = 0.995  # ε贪婪值衰减率
        self.epsilon_min = 0.01  # 最小ε贪婪值
        self.batch_size = 64
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
    
    def train(self):
        if len(self.buffer) < self.batch_size:
            return

        transitions = self.buffer.sample(self.batch_size)
        
        # batch = list(zip(*transitions))
        # print(batch[0])
        batch = map(np.array,zip(*transitions))
        states, actions, rewards, next_states, done = batch

        state_batch = torch.FloatTensor(states).to(self.device)
        action_batch = torch.LongTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.FloatTensor(next_states).to(self.device)
        done_batch = torch.BoolTensor(done).to(self.device)

        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + self.gamma * next_q_values * ~done_batch
        
        loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
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
    # def experience_replay(self):
    #     if self.batch_size > len(self.experience):
    #         return
    #     minibatch = map(np.array, zip(*sample(self.experience, self.batch_size)))
    #     states, actions, rewards, next_states, not_done = minibatch
        
        
    #     #for Pytorch, turn into FloatTensor
    #     states = torch.FloatTensor(states) #[b,5]
    #     actions = torch.tensor(actions).to(torch.long).view(self.batch_size,1) #[b,1]
    #     rewards = torch.FloatTensor(rewards).view(self.batch_size,1) #[b,1]
    #     next_states = torch.FloatTensor(next_states)#[b,5]
        
    #     #calculate next state most probable action
    #     q_eval = self.policy_network(states).gather(1, actions)
    #     q_eval_next = torch.max(self.policy_network(next_states), 1)[1].view(self.batch_size,1) #index(action)
    #     q_next = self.target_network(next_states)
    #     q_e = q_next.gather(1, q_eval_next)
    #     not_done = torch.tensor(not_done).unsqueeze(0)
    #     q_target = rewards + 0.99 * not_done  * q_e
        
    #     loss = self.mseloss(q_eval, q_target)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
        
    #     self.losses.append(loss.data)

    #     if self.total_steps % self.tau == 0:
    #         self.update_target()


def train():
    env = MarketEnv(252)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    print(input_size,output_size)
    hidden_dim = 128
    agent = DDQNAgent(input_size, hidden_dim, output_size)
    num_episodes = 1000
    market_return_vis = []
    strategy_return_vis = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done,_ = env.step(action)
            # print(reward)
            agent.buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.train()
        if episode % 100 == 0:
            agent.save(f'ddqn-{episode}')
        tot_ret = env.result()
        market_return_vis.append(round(tot_ret['market_return']*100 - 100,2))
        strategy_return_vis.append(round(tot_ret['strategy_return']*100 - 100,2))
        logger.info(f'Episode: {episode+1} | Total Reward: {total_reward} | market_return: {round(tot_ret['market_return']*100 - 100,2) }% | strategy_return: {round(tot_ret['strategy_return']*100 - 100,2)}%')

    # 创建折线图
    plt.plot(market_return_vis, label='Market Return')
    plt.plot(strategy_return_vis, label='Strategy Return')

    # 添加图例，位于最佳位置（可选）
    plt.legend(loc='best')

    # 添加标题和轴标签
    plt.title("Market vs Strategy Returns")
    plt.xlabel("Time")
    plt.ylabel("Returns")

    # 显示网格（可选）
    plt.grid(True)
    
    plt.savefig('return_comparison.png', dpi=300, bbox_inches='tight')

    # 显示图形
    # plt.show()


def valid(filename):
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # env.seed(seed)  # 如果环境提供seed()方法

    env = MarketEnv(252,start_date='20220101',end_date='20230501')
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    hidden_dim = 128
    agent = DDQNAgent(input_size, hidden_dim, output_size)
    # agent.load(filename)
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.select_action(state)
        print(action)
        next_state, reward, done,_ = env.step(action)
        # print(reward)
        # agent.buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    tot_ret = env.result()
    logger.info(f'Total Reward: {total_reward} | market_return: {round(tot_ret['market_return']*100 - 100,2) }% | strategy_return: {round(tot_ret['strategy_return']*100 - 100,2)}%')

    
if __name__ == '__main__':
    # valid("ddqn-600.pth")
    train()