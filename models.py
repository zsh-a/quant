import torch.nn as nn
import math
import torch

class LeNet(nn.Module):  # 继承于nn.Module这个父类
    def __init__(self):  # 初始化网络结构
        super(LeNet, self).__init__()  # 多继承需用到super函数
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.fc1 = nn.Linear(32 * 14 * 1, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.relu = nn.ReLU()

    def forward(self, x):  # 正向传播过程
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))  # input(1, 60, 6) output(16, 28, 28)
        x = self.pool1(x)  # output(16, 52, 2)
        x = self.relu(self.conv2(x))  # output(32, 28, 1)
        x = self.pool2(x)  # output(32, 14, 1)
        x = x.view(-1, 32 * 14 * 1)  # output(32*14*1)
        x = self.relu(self.fc1(x))  # output(120)
        x = self.relu(self.fc2(x))  # output(84)
        x = self.fc3(x)  # output(1)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim=6, d_model=128, nhead=8, 
                 num_encoder_layers=3, num_decoder_layers=3, 
                 dim_feedforward=512, dropout=0.1, output_steps=20):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        self.decoder_query = nn.Parameter(torch.randn(output_steps, d_model))
        self.output = nn.Linear(d_model, 1)
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):
        batch_size = src.size(0)
        
        # 编码器处理
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        memory = self.encoder(src_emb)
        
        # 解码器处理
        tgt = self.decoder_query.unsqueeze(0).repeat(batch_size, 1, 1)
        tgt = self.pos_decoder(tgt)
        output = self.decoder(tgt, memory)
        
        # 输出预测
        return self.output(output).squeeze(-1)


if __name__ == '__main__':
    model = TransformerModel(output_steps=1)
    x = torch.randn(1,60,6)
    y = model(x)
    print(y.shape)
    print(model)