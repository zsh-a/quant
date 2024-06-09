import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

# 选择动作
def select_action(policy_net, state):

    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs = policy_net(state)
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)

# 计算折扣回报
def discount_rewards(rewards, gamma):
    R = 0
    discounted_rewards = []
    for r in rewards[::-1]:
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    return discounted_rewards

# 设置参数
env = gym.make('CartPole-v1')
input_dim = env.observation_space.shape[0]
hidden_dim = 128
output_dim = env.action_space.n
learning_rate = 0.01
gamma = 0.99

policy_net = PolicyNetwork(input_dim, hidden_dim, output_dim).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

# 训练策略网络
num_episodes = 2000
for episode in range(num_episodes):
    state = env.reset()
    state = state[0]
    log_probs = []
    rewards = []
    done = False
    
    while not done:
        action, log_prob = select_action(policy_net, state)
        new_state, reward, done, truncated,_ = env.step(action)
        done = done or truncated
        log_probs.append(log_prob)
        rewards.append(reward)
        
        state = new_state
    
    discounted_rewards = discount_rewards(rewards, gamma)
    discounted_rewards = torch.tensor(discounted_rewards)
    
    # 标准化折扣回报
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
    
    policy_loss = []
    for log_prob, reward in zip(log_probs, discounted_rewards):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()
    
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    
    if episode % 100 == 0:
        print(f'Episode {episode}, Last length: {len(rewards)}')
        # eval_env = gym.make('CartPole-v1',render_mode='human')
        # state = eval_env.reset()
        # state = state[0]
        # done = False
    
        # while not done:
        #     action, log_prob = select_action(policy_net, state)
        #     new_state, reward, done, truncated,_ = eval_env.step(action)
        #     done = done or truncated
        #     state = new_state


env.close()
