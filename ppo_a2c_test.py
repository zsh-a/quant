import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
import multiprocessing as mp
import numpy as np

# 环境

env_name = 'CartPole-v1'
# 超参数
learning_rate = 0.001
gamma = 0.99
eps_clip = 0.2
K_epoch = 3
T_horizon = 20
entropy_coef = 0.0

# 策略网络和价值网络
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        
        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self._initialize_weights()

    def _initialize_weights(self):
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def pi(self, x, softmax_dim=0):
        x = torch.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = torch.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = torch.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
    
    def put_data(self, transition):
        self.data.append(transition)
    
    def make_batch(self,data):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_lst.append([0 if done else 1])
        
        s, a, r, s_prime, done, prob_a = torch.tensor(np.array(s_lst), dtype=torch.float), torch.tensor(a_lst), \
                                         torch.tensor(r_lst), torch.tensor(np.array(s_prime_lst), dtype=torch.float), \
                                         torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        return s, a, r, s_prime, done, prob_a
    
    def train_net(self,data):
        s, a, r, s_prime, done, prob_a = self.make_batch(data)
        
        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done
            delta = td_target - self.v(s)
            delta = delta.detach()
            
            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a + 1e-10) - torch.log(prob_a + 1e-10))  # Add small constant to avoid log(0)
            
            surr1 = ratio * delta
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * delta
            entropy = -torch.sum(pi * torch.log(pi + 1e-10), dim=1)
            # print(torch.min(surr1, surr2).mean().item(),entropy.mean().item())
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach()) - entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


# Worker function to collect data
def worker(pid, model, env_name, queue, gamma):
    env = gym.make(env_name)
    state = env.reset()[0]
    done = False
    data = []
    
    while not done:
        state_tensor = torch.FloatTensor(state)

        logits = model.pi(state_tensor)
        # print(logits)
        dist = Categorical(logits=logits)
        action = dist.sample().item()
        # print(logits,action)
        
        next_state, reward, done, info, truncated = env.step(action)

        data.append((state, action, reward / 1000, next_state, logits[action].item(), done))
        
        state = next_state
        if truncated:
            break
    queue.put(data)

# 主函数
model = ActorCritic()
score = 0.0
print_interval = 100
num_workers = 16

queue = mp.Queue()

for n_epi in range(10000):
    processes = []
    for pid in range(num_workers):
            p = mp.Process(target=worker, args=(pid, model, env_name, queue, gamma))
            p.start()
            processes.append(p)


    # Collect data from workers
    data = []
    for _ in range(num_workers):
        data.extend(queue.get())

    # Ensure all processes are finished
    for p in processes:
        p.join()
    
    model.train_net(data)
    # Calculate returns and advantages
    # returns = []
    # G = 0
    # for r, done in zip(reversed(rewards), reversed(dones)):
    #     G = r + gamma * G * (1 - done)
    #     returns.insert(0, G)
    # returns = torch.FloatTensor(returns)
    
    # advantages = returns - model.v(states).squeeze().detach()
    
    # # Update policy network
    # for _ in range(4):
    #     logits = model.pi(states)
    #     dist = Categorical(logits=logits)
    #     old_log_probs = dist.log_prob(actions)

    #     new_logits = model.pi(states)
    #     new_dist = Categorical(logits=new_logits)
    #     new_log_probs = new_dist.log_prob(actions)
        
    #     ratio = torch.exp(new_log_probs - old_log_probs)
    #     surr1 = ratio * advantages
    #     surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
    #     loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(model.v(s), td_target.detach())

    
    #     self.optimizer.zero_grad()
    #     loss.mean().backward()
    #     self.optimizer.step()
    
    # print(len(trajectories))
    # print(states)
    
        
    #     model.train_net()
    
    if n_epi % print_interval == 0 and n_epi != 0:
        env = gym.make('CartPole-v1',render_mode='human')
        s = env.reset()[0]
        done = False
        tot_reward = 0
        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                
                s_prime, r, done, info,truncated = env.step(a)
                s = s_prime
                tot_reward += r
                if done or truncated:
                    break
        print("# of episode :{}, tot reward : {:.1f}".format(n_epi, tot_reward))
        score = 0.0

env.close()
