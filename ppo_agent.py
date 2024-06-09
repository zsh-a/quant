import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random
from market_env import MarketEnv
import utils
from torch.distributions import Categorical
import torch.nn.functional as F

from loguru import logger
from torchstat import stat

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# class PolicyNetwork(nn.Module):
#     def __init__(self, input_dim,hidden_dim, output_size, dropout=0.1):
#         super(PolicyNetwork, self).__init__()
#         self.model_type = 'Transformer'
#         self.input_size = input_dim
#         num_encoder_layers = 2
#         num_heads = 2
#         self.positional_encoding = PositionalEncoding(input_dim)
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout,batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
#         self.decoder = nn.Linear(input_dim, output_size)
    
#     def forward(self, src):
#         src = self.positional_encoding(src)
#         output = self.transformer_encoder(src)
#         output = self.decoder(output[:, -1, :])
#         output = torch.softmax(output, dim=-1)
#         return output
    

# 超参数
learning_rate = 0.001
gamma = 0.99
eps_clip = 0.2
K_epoch = 3
T_horizon = 20

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 策略网络和价值网络
class ActorCritic(nn.Module):
    def __init__(self,input_dim,hidden_dim, output_size, dropout=0.1):
        super(ActorCritic, self).__init__()
        self.model_type = 'Transformer'
        self.input_size = input_dim
        num_encoder_layers = 4
        num_heads = 4
        self.positional_encoding = PositionalEncoding(input_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        # self.decoder = nn.Linear(input_dim, output_size)

        self.fc = nn.Linear(input_dim, hidden_dim)

        self.fc_pi = nn.Linear(hidden_dim, output_size)
        self.fc_v = nn.Linear(hidden_dim, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.data = []
        
    def pi(self, x):
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = F.gelu(self.fc(x[:, -1, :]))
        x = self.fc_pi(x)
        prob = torch.softmax(x, dim=-1)
        return prob
    
    def v(self, x):
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = F.gelu(self.fc(x[:, -1, :]))
        v = self.fc_v(x)
        return v
    
    def put_data(self, transition):
        self.data.append(transition)
    
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_lst.append([0 if done else 1])
        
        s, a, r, s_prime, done, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                         torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                         torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done, prob_a
    
    def train_net(self):
        s, a, r, s_prime, done, prob_a = self.make_batch()
        s = s.to(device)
        a = a.to(device)
        r = r.to(device)
        s_prime = s_prime.to(device)
        done = done.to(device)
        prob_a = prob_a.to(device)

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done
            delta = td_target - self.v(s)
            delta = delta.detach()
            
            pi = self.pi(s)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))
            
            surr1 = ratio * delta
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * delta
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

env = MarketEnv(252,start_date='20100401',end_date='20230401')

input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
hidden_dim = 128

# 主函数
model = ActorCritic(input_dim,hidden_dim,output_dim).to(device=device)

# 计算模型参数数量
# num_params = count_parameters(model)
# print("模型参数数量:", num_params)


print_interval = 10

for n_epi in range(10000):
    s = env.reset()
    done = False

    rewards = []
    while not done:
        for t in range(T_horizon):
            
            prob = model.pi(torch.from_numpy(s).float().unsqueeze(0).to(device))
            prob = prob.squeeze(0)
            m = Categorical(prob)
            a = m.sample().item()
            
            s_prime, r, done, info = env.step(a)

            model.put_data((s, a, r, s_prime, prob[a].item(), done))
            
            s = s_prime
            rewards.append(r)
            if done:
                break
        
        model.train_net()
    
    if n_epi % print_interval == 0 and n_epi != 0:
        ret = env.result()
        writer.add_scalar("reward", sum(rewards), n_epi)
        writer.add_scalar("market_return", ret['market_return'], n_epi)
        writer.add_scalar("strategy_return", ret['strategy_return'], n_epi)
        writer.add_scalar("max_profit", ret['max_profit'], n_epi)
        writer.add_scalar("max_drawdown", ret['max_drawdown'], n_epi)
        logger.info(f'n_epi : {n_epi} | Total Reward: {sum(rewards)} | market_return: {ret['market_return'] }% | strategy_return: {ret['strategy_return']}% | max_drawdown : {ret['max_drawdown']} | max_profit : {ret['max_profit']} |  position : {np.array(ret['positions'])} | actions : {np.array(ret['actions'])}')
        # print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
        # score = 0.0


# def valid(name):
#     env = MarketEnv(200,start_date='20230401',end_date='20240401')
#     input_dim = env.observation_space.shape[0]
#     output_dim = env.action_space.n
#     hidden_dim = 128

#     agent = PGAgent(input_dim,hidden_dim,output_dim)
#     agent.load(name)
#     state = env.reset()
#     done = False
#     log_probs = []
#     rewards = []
#     actions = []
#     with torch.no_grad():
#         while not done:
#             action, log_prob = agent.select_action(state)
#             new_state, reward, done,_ = env.step(action)
#             actions.append(action)
#             done = done 
#             log_probs.append(log_prob)
#             rewards.append(reward)
#             state = new_state
#         ret = env.result()
#         logger.info(f'vaild | Total Reward: {sum(rewards)} | market_return: {ret['market_return'] }% | strategy_return: {ret['strategy_return']}% | max_drawdown : {ret['max_drawdown']} | max_profit : {ret['max_profit']} |  position : {np.array(ret['positions'])} | actions : {np.array(ret['actions'])}')

# if __name__ == '__main__':
#     num_episodes = 10000

#     # env = gym.make('CartPole-v1',render_mode='human')
#     env = MarketEnv(252,start_date='20100401',end_date='20230401')

#     input_dim = env.observation_space.shape[0]
#     output_dim = env.action_space.n

#     hidden_dim = 128
#     agent = PGAgent(input_dim,hidden_dim,output_dim)

#     for episode in range(num_episodes):
#         state = env.reset()
#         # state = state[0]
#         log_probs = []
#         rewards = []
#         actions = []
#         done = False
        
#         while not done:
#             action, log_prob = agent.select_action(state)
#             new_state, reward, done,_ = env.step(action)
#             actions.append(action)
#             done = done 
#             log_probs.append(log_prob)
#             rewards.append(reward)
            
#             state = new_state
#         agent.train(episode,rewards,log_probs)

#         if episode % 10 == 0:
#             ret = env.result()
#             writer.add_scalar("reward", sum(rewards), episode)
#             writer.add_scalar("market_return", ret['market_return'], episode)
#             writer.add_scalar("strategy_return", ret['strategy_return'], episode)
#             writer.add_scalar("max_profit", ret['max_profit'], episode)
#             writer.add_scalar("max_drawdown", ret['max_drawdown'], episode)
#             logger.info(f'episode : {episode} | Total Reward: {sum(rewards)} | market_return: {ret['market_return'] }% | strategy_return: {ret['strategy_return']}% | max_drawdown : {ret['max_drawdown']} | max_profit : {ret['max_profit']} |  position : {np.array(ret['positions'])} | actions : {np.array(ret['actions'])}')
#         if episode % 100 == 0:
#             name = f'pg-{episode}'
#             agent.save(name)
#             # valid(f'{name}.pth')

#             # print(actions)
#             # print(f'Episode {episode}, Last reward: {utils.log2percent(sum(rewards))}%')

#     env.close()
