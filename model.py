import torch.nn as nn
import torch
import torch.nn.functional as F

class QNet(nn.Module):
    def __init__(self, state_dim,hidden_dim=128,num_actions=3) -> None:
        super(QNet).__init__()
        self.fc1 = nn.Linear(state_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)

        self.fc3 = nn.Linear(hidden_dim,num_actions)

    def forward(self,x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.dropout(x,0.1)
        x = self.fc3(x)
        return x

    # def epsilon_greedy_policy(self,state):


class DQNAgent:
    def __init__(self,state_dim,hidden_dim,num_actions) -> None:
        
        self.policy_network = QNet(state_dim,hidden_dim,num_actions)
        self.target_network = QNet(state_dim,hidden_dim,num_actions)

