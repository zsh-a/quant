import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import re,os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math

# torch.autograd.set_detect_anomaly(True)
feature_columns = ['volume','amplitude','returns']
array = np.linspace(0, 0.1, 6)

intervals = []

for i in range(len(array) - 1):
    intervals.append((array[i],array[i + 1]))
labels = np.arange(0,len(array) - 1)


NUM_CLASS = 2

FEATURE_KS = 40
PREDICT_KS = 5

# 定义数据集类
class StockDataset(Dataset):
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx],self.Y[idx]

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self,input_size,seq_len,num_class):
        super(TransformerModel, self).__init__()
        d_model = 256
        self.seq_len = seq_len
        self.fc1 = nn.Linear(input_size,d_model,dtype=float,bias=False)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # 添加一个可学习的CLS头
        # self.pos_embedding = PositionalEncoding(d_model,max_len=120)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, d_model))
        
        self.encoder_layer =  nn.TransformerEncoderLayer(d_model=d_model,dtype=float,nhead=4,activation='gelu',batch_first=True,bias=False)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        # self.transformer = nn.Transformer(d_model=d_model,dtype=float,dim_feedforward=256,activation='gelu',nhead=4,batch_first=True)
        self.fc = nn.Linear(d_model, num_class,dtype=float,bias=False)
    
    def forward(self, x):
        x = self.fc1(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # 将CLS头添加到输入序列的开头
        x += self.pos_embedding[:, :(self.seq_len + 1)]
        x = self.transformer_encoder(x)
        x = self.fc(x[:, 0, :])  # 只使用CLS头的输出进行预测
        return x

def data_preprocess(file_path):
    with open(file_path, 'r',errors="ignore") as file:
        lines = file.readlines()
    match = re.search(r'#(\d+)\.', file_path)
    code = match.group(1)
    lines = lines[2:-1]
    out_path = os.path.join("data",f'{code}.csv')
    with open(out_path, 'w') as file:
        file.writelines(lines)


if __name__ == '__main__':
    for file in os.listdir('tdx_export'):
        data_preprocess(os.path.join('tdx_export',file))