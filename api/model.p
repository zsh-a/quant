import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, output_dim=10, dropout=0.5):
        super(Model, self).__init__()
        # 网络层定义
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 输入层→隐藏层
        self.fc2 = nn.Linear(hidden_dim, output_dim) # 隐藏层→输出层

        self.dropout = nn.Dropout(p=dropout)
        
        # 可选：权重初始化
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        
        # 前向传播
        x = self.fc1(x)
        x = F.relu(x)          # 激活函数
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # 输出层不接激活函数（交叉熵损失函数自带Softmax）

