import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random
from market_env import MarketEnv
import utils
from loguru import logger

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


# SEED = 1
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# random.seed(SEED)


# class PolicyNetwork(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(PolicyNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.softmax(self.fc2(x), dim=-1)
#         return x

# class PolicyNetwork(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(PolicyNetwork, self).__init__()
#         self.hidden_size = 128
#         self.num_layers = 2
#         self.lstm = nn.LSTM(input_dim, hidden_dim, self.num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.fc(out[:, -1, :])
#         out = torch.softmax(out, dim=-1)
#         return out



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim,hidden_dim, output_size, dropout=0.1):
        super(PolicyNetwork, self).__init__()
        self.model_type = 'Transformer'
        self.input_size = input_dim
        num_encoder_layers = 2
        num_heads = 2
        self.positional_encoding = PositionalEncoding(input_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.Linear(input_dim, output_size)
    
    def forward(self, src):
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :])
        output = torch.softmax(output, dim=-1)
        return output
   
def compute_entropy(probs):
    return -torch.sum(probs * probs, dim=-1).mean()

class PGAgent:
    def __init__(self,state_dim,hidden_dim,num_actions,train=True) -> None:


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = PolicyNetwork(state_dim,hidden_dim,num_actions).to(self.device)
        self.num_actions = num_actions
        self.gamma = 0.99  # 折扣因子
        self.train_mode = train
        self.entropy_coef = 0.01

        self.optimizer = optim.Adam(self.policy_net.parameters(),lr=0.01)

    # 选择动作
    def select_action(self, state):

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.policy_net(state)
        if not self.train_mode:
            print(probs)    
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    

    def discount_rewards(self,rewards):
        R = 0
        discounted_rewards = []
        for r in rewards[::-1]:
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        return discounted_rewards
    
    def train(self,episode,rewards,log_probs):

        discounted_rewards = self.discount_rewards(rewards)

        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)



        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        policy_loss = torch.cat(policy_loss).sum()

        action_probs = torch.cat(log_probs)
        # print(action_probs)
        entropy_loss = compute_entropy(action_probs)
        # print(entropy_loss)
        loss = policy_loss - self.entropy_coef * entropy_loss

    
        self.optimizer.zero_grad()
        loss.backward()
        print(f'episode : {episode} | loss : {policy_loss.item()}')
        self.optimizer.step()
        
    def load(self,name):
        self.policy_net = torch.load(name).to(self.device)
                
    def save(self,name):
        torch.save(self.policy_net, f'{name}.pth')



def valid(name):
    env = MarketEnv(200,start_date='20230401',end_date='20240401')
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    hidden_dim = 128

    agent = PGAgent(input_dim,hidden_dim,output_dim)
    agent.load(name)
    state = env.reset()
    done = False
    log_probs = []
    rewards = []
    actions = []
    with torch.no_grad():
        while not done:
            action, log_prob = agent.select_action(state)
            new_state, reward, done,_ = env.step(action)
            actions.append(action)
            done = done 
            log_probs.append(log_prob)
            rewards.append(reward)
            state = new_state
        ret = env.result()
        logger.info(f'vaild | Total Reward: {sum(rewards)} | market_return: {ret['market_return'] }% | strategy_return: {ret['strategy_return']}% | max_drawdown : {ret['max_drawdown']} | max_profit : {ret['max_profit']} |  position : {np.array(ret['positions'])} | actions : {np.array(ret['actions'])}')

if __name__ == '__main__':
    num_episodes = 10000

    # env = gym.make('CartPole-v1',render_mode='human')
    env = MarketEnv(252,start_date='20100401',end_date='20230401')

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    hidden_dim = 128
    agent = PGAgent(input_dim,hidden_dim,output_dim)

    for episode in range(num_episodes):
        state = env.reset()
        # state = state[0]
        log_probs = []
        rewards = []
        actions = []
        done = False
        
        while not done:
            action, log_prob = agent.select_action(state)
            new_state, reward, done,_ = env.step(action)
            actions.append(action)
            done = done 
            log_probs.append(log_prob)
            rewards.append(reward)
            
            state = new_state
        agent.train(episode,rewards,log_probs)

        if episode % 10 == 0:
            ret = env.result()
            writer.add_scalar("reward", sum(rewards), episode)
            writer.add_scalar("market_return", ret['market_return'], episode)
            writer.add_scalar("strategy_return", ret['strategy_return'], episode)
            writer.add_scalar("max_profit", ret['max_profit'], episode)
            writer.add_scalar("max_drawdown", ret['max_drawdown'], episode)
            logger.info(f'episode : {episode} | Total Reward: {sum(rewards)} | market_return: {ret['market_return'] }% | strategy_return: {ret['strategy_return']}% | max_drawdown : {ret['max_drawdown']} | max_profit : {ret['max_profit']} |  position : {np.array(ret['positions'])} | actions : {np.array(ret['actions'])}')
        # if episode % 100 == 0:
        #     name = f'pg-{episode}'
        #     agent.save(name)
            # valid(f'{name}.pth')

            # print(actions)
            # print(f'Episode {episode}, Last reward: {utils.log2percent(sum(rewards))}%')

    env.close()
