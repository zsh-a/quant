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

from utils import *

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
    def __init__(self,state_dim,hidden_dim,num_actions,train=True) -> None:


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = QNet(state_dim,hidden_dim,num_actions).to(self.device)
        self.target_net = QNet(state_dim,hidden_dim,num_actions).to(self.device)

        self.buffer = ReplayBuffer(capacity=10000)
        self.num_actions = num_actions
        self.total_steps = 0
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0 if train else 0.0 # 初始ε贪婪值 
        self.epsilon_decay = 0.99998  # ε贪婪值衰减率
        self.epsilon_min = 0.2  # 最小ε贪婪值
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
    
    def train(self,episode):
        if len(self.buffer) < self.batch_size:
            return

        transitions = self.buffer.sample(self.batch_size)
        
        # batch = list(zip(*transitions))
        # print(batch[0])
        batch = map(np.array,zip(*transitions))
        states, actions, rewards, next_states, done = batch
        # print(states,actions,rewards)
        state_batch = torch.FloatTensor(states).to(self.device)
        action_batch = torch.LongTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.FloatTensor(next_states).to(self.device)
        done_batch = torch.BoolTensor(done).to(self.device)

        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        q_eval_next = torch.max(self.policy_net(next_state_batch),1)[1].view(self.batch_size,1)
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
    env = MarketEnv(252,start_date='20101001',end_date='20230401')
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    print(input_size,output_size)
    hidden_dim = 128
    agent = DDQNAgent(input_size, hidden_dim, output_size)
    num_episodes = 1000
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
            agent.save(f'ddqn-{episode}')
        tot_ret = env.result()
        market_return_vis.append(tot_ret['market_return'])
        strategy_return_vis.append(tot_ret['strategy_return'])
        rewards.append(total_reward)
        writer.add_scalar("reward", total_reward, episode)
        writer.add_scalar("market_return", tot_ret['market_return'], episode)
        writer.add_scalar("strategy_return", tot_ret['strategy_return'], episode)
        logger.info(f'Episode: {episode+1} | loss : {agent.losses[-1]} | Total Reward: {total_reward} | market_return: {tot_ret['market_return'] }% | strategy_return: {tot_ret['strategy_return']}%')
        if episode % 100 == 0:
            valid(f'ddqn-{episode}.pth')

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
    agent = DDQNAgent(input_size, hidden_dim, output_size,train=False)
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