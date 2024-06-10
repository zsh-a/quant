import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random
from market_env import MarketEnv
import utils
from torch.distributions import Categorical
import multiprocessing as mp
import torch.nn.functional as F
from collections import deque

from loguru import logger
from torchstat import stat

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(comment='ppo')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        # Ensure the positional encoding is broadcastable to the input tensor's shape
        x = x + self.pe[:, :seq_len, :]
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
entropy_coef = 0.01
value_coef = 1
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 策略网络和价值网络
class ActorCritic(nn.Module):
    def __init__(self,input_dim,hidden_dim, output_size, dropout=0.1):
        super(ActorCritic, self).__init__()
        self.model_type = 'Transformer'
        self.input_size = input_dim
        num_encoder_layers = 2
        num_heads = 2
        self.positional_encoding = PositionalEncoding(input_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        # self.decoder = nn.Linear(input_dim, output_size)

        self.fc = nn.Linear(input_dim, hidden_dim)

        self.fc_pi = nn.Linear(hidden_dim + 1, output_size)
        self.fc_v = nn.Linear(hidden_dim + 1, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.data = []

        self._initialize_weights()

    def _initialize_weights(self):
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def pi(self, x, pos):
        # print(x.size())
        x = self.positional_encoding(x)
        # print(x.size())
        x = self.transformer_encoder(x)
        x = F.gelu(self.fc(x[:, -1, :]))
        x = torch.cat((x,pos),dim=-1)
        x = self.fc_pi(x)
        prob = torch.softmax(x, dim=-1)
        # print(prob.size())
        return prob
    
    def v(self, x, pos):
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = F.gelu(self.fc(x[:, -1, :]))
        x = torch.cat((x,pos),dim=-1)
        v = self.fc_v(x)
        return v
    
    def put_data(self, transition):
        self.data.append(transition)
    
    def make_batch(self,data):
        s_lst, pos_lst, a_lst, r_lst, s_prime_lst, s_prime_pos_lst, prob_a_lst, done_lst = [], [], [], [], [], [], [], []
        for transition in data:
            (his,pos), a, r, (s_prime,s_prime_pos), prob_a, done = transition
            
            s_lst.append(his)
            pos_lst.append([pos])
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            s_prime_pos_lst.append([s_prime_pos])
            prob_a_lst.append([prob_a])
            done_lst.append([0 if done else 1])
        
        s, a, r, s_prime, done, prob_a = (torch.tensor(np.array(s_lst), dtype=torch.float), torch.tensor(np.array(pos_lst), dtype=torch.float)), torch.tensor(a_lst), \
                                         torch.tensor(r_lst), (torch.tensor(np.array(s_prime_lst), dtype=torch.float),torch.tensor(np.array(s_prime_pos_lst), dtype=torch.float)), \
                                         torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        return s, a, r, s_prime, done, prob_a
    
    def train_net(self,data,n_epi):
        (his,pos), a, r, (s_prime,s_prime_pos), done, prob_a = self.make_batch(data)
        his = his.to(device)
        pos = pos.to(device)
        a = a.to(device)
        r = r.to(device)
        s_prime = s_prime.to(device)
        s_prime_pos = s_prime_pos.to(device)
        done = done.to(device)
        prob_a = prob_a.to(device)

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime,s_prime_pos) * done
            delta = td_target - self.v(his,pos)
            delta = delta.detach()
            
            pi = self.pi(his,pos)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a + 1e-10) - torch.log(prob_a + 1e-10))  # Add small constant to avoid log(0)
            
            surr1 = ratio * delta
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * delta
            entropy = -torch.sum(pi * torch.log(pi + 1e-10), dim=1)
            # print(torch.min(surr1, surr2).mean().item(),entropy.mean().item())
            policy_loss = -torch.min(surr1, surr2)
            value_loss = F.smooth_l1_loss(self.v(his,pos), td_target.detach()) * value_coef
            entropy_loss = entropy_coef * entropy
            
            loss = policy_loss + value_loss + entropy_loss

            writer.add_scalar("policy_loss", policy_loss.mean().item(), n_epi)
            writer.add_scalar("value_loss", value_loss.mean().item(), n_epi)
            writer.add_scalar("entropy_loss", entropy_loss.mean().item(), n_epi)
            writer.add_scalar("loss", loss.mean().item(), n_epi)



            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

# Worker function to collect data
def worker(pid, model, env_name, queue, gamma):
    env = MarketEnv(252,start_date='20100401',end_date='20230401')
    while True:
        state = env.reset()
        done = False
        data = []
        tot_reward = 0
        while not done:
            his,pos = state
            state_tensor = torch.FloatTensor(his).unsqueeze(0).to(device)
            pos_tensor = torch.FloatTensor([pos]).unsqueeze(0).to(device)

            # state_tensor
            logits = model.pi(state_tensor,pos_tensor).squeeze()
            # print(state_tensor.size())
            dist = Categorical(logits=logits)
            action = dist.sample().item()
            # print(logits,action)
            
            next_state, reward, done, info = env.step(action)

            data.append((state, action, reward, next_state, logits[action].item(), done))
            
            tot_reward += reward
            state = next_state
        queue.put(data)

        # if n_epi % 10 == 0 and n_epi != 0:
        # ret = env.result()
        # logger.info(f'Total Reward: {tot_reward} | market_return: {ret['market_return'] }% | strategy_return: {ret['strategy_return']}% | max_drawdown : {ret['max_drawdown']} | max_profit : {ret['max_profit']} |  position : {np.array(ret['positions'])} | actions : {np.array(ret['actions'])}')

num_episodes = 100000
print_interval = 10
num_workers = 4

if __name__ == '__main__':
    mp.set_start_method('spawn')
    env = MarketEnv(252,start_date='20100401',end_date='20230401')

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    hidden_dim = 128

    # 主函数
    model = ActorCritic(input_dim,hidden_dim,output_dim).to(device=device)


    queue = mp.Queue(maxsize=num_workers * 2)


    processes = []
    for pid in range(num_workers):
            p = mp.Process(target=worker, args=(pid, model, 'market', queue, gamma))
            p.start()
            processes.append(p)

    data_buffer = deque(maxlen=num_workers)


    for n_epi in range(num_episodes):
        # Collect data from workers
        data = queue.get()
        # print(queue.qsize())
        model.train_net(data,n_epi)
        # data_buffer.append(data)

        # if len(data_buffer) > 0:
        #     data = data_buffer.popleft()

        # print(len(data_buffer))
        # for _ in range(num_workers):
        #     data.extend(queue.get())

        # Ensure all processes are finished
        # for p in processes:
        #     p.join()
        
        # model.train_net(data)

        if n_epi % 10 == 0:
            env = MarketEnv(200,start_date='20230401',end_date='20240401')
            state = env.reset()
            done = False
            data = []
            tot_reward = 0
            while not done:
                his,pos = state
                state_tensor = torch.FloatTensor(his).unsqueeze(0).to(device)
                pos_tensor = torch.FloatTensor([pos]).unsqueeze(0).to(device)

                # state_tensor
                logits = model.pi(state_tensor,pos_tensor).squeeze()
                # print(state_tensor.size())
                dist = Categorical(logits=logits)
                action = dist.sample().item()
                # print(logits,action)
                
                next_state, reward, done, info = env.step(action)
                tot_reward += reward
                state = next_state
            # if n_epi % 10 == 0 and n_epi != 0:
            ret = env.result()

            writer.add_scalar("reward", tot_reward, n_epi)
            writer.add_scalar("market_return", ret['market_return'], n_epi)
            writer.add_scalar("strategy_return", ret['strategy_return'], n_epi)
            writer.add_scalar("max_profit", ret['max_profit'], n_epi)
            writer.add_scalar("max_drawdown", ret['max_drawdown'], n_epi)
            logger.info(f'n_epi : {n_epi} | Total Reward: {tot_reward} | market_return: {ret['market_return'] }% | strategy_return: {ret['strategy_return']}% | max_drawdown : {ret['max_drawdown']} | max_profit : {ret['max_profit']} |  position : {np.array(ret['positions'])} | actions : {np.array(ret['actions'])}')


    for p in processes:
        p.join()


# for n_epi in range(10000):
#     s = env.reset()
#     done = False

#     rewards = []
#     while not done:
#         for t in range(T_horizon):
            
#             prob = model.pi(torch.from_numpy(s).float().unsqueeze(0).to(device))
#             prob = prob.squeeze(0)
#             m = Categorical(prob)
#             a = m.sample().item()
            
#             s_prime, r, done, info = env.step(a)

#             model.put_data((s, a, r, s_prime, prob[a].item(), done))
            
#             s = s_prime
#             rewards.append(r)
#             if done:
#                 break
        
#         model.train_net()
    
#     if n_epi % print_interval == 0 and n_epi != 0:
#         ret = env.result()
#         writer.add_scalar("reward", sum(rewards), n_epi)
#         writer.add_scalar("market_return", ret['market_return'], n_epi)
#         writer.add_scalar("strategy_return", ret['strategy_return'], n_epi)
#         writer.add_scalar("max_profit", ret['max_profit'], n_epi)
#         writer.add_scalar("max_drawdown", ret['max_drawdown'], n_epi)
#         logger.info(f'n_epi : {n_epi} | Total Reward: {sum(rewards)} | market_return: {ret['market_return'] }% | strategy_return: {ret['strategy_return']}% | max_drawdown : {ret['max_drawdown']} | max_profit : {ret['max_profit']} |  position : {np.array(ret['positions'])} | actions : {np.array(ret['actions'])}')
#         # print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
#         # score = 0.0
