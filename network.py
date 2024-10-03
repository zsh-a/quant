import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random
from market_env import MarketEnv
import utils.utils as utils
from loguru import logger



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

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.hidden_size = 128
        self.num_layers = 2
        self.lstm = nn.LSTM(input_dim, hidden_dim, self.num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = torch.softmax(out, dim=-1)
        return out


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

class StockPriceTransformer(nn.Module):
    def __init__(self, input_dim,hidden_dim, output_size, dropout=0.1):
        super(StockPriceTransformer, self).__init__()
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

if __name__ == '__main__':
    model = StockPriceTransformer(4,128,3)
    x = torch.rand(7,20,4)
    print(model(x))