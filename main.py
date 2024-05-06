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

FEATURE_KS = 30
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
        d_model = 128
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
        # x += self.pos_embedding[:, :(self.seq_len + 1)]
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


def normilize(df):
    for col in feature_columns:
        df.loc[:,col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    df.loc[:,'returns_5d'] = (df['returns_5d'] - df['returns_5d'].min()) / (df['returns_5d'].max() - df['returns_5d'].min())
    # print(df)
    X = []
    Y = []
    for i in range(len(df) - PREDICT_KS - FEATURE_KS):
        X.append(df[feature_columns].iloc[i:i + FEATURE_KS].values)
        # Y.append(df['returns_5d'].iloc[i + FEATURE_KS])
        Y.append(df['label'].iloc[i + FEATURE_KS])
    return np.array(X),np.array(Y)
    
    
def load_csv(file_path):
    df = pd.read_csv(file_path)
    df.columns = ['datetime','open','high','low','close','volume','amount']
    df.set_index('datetime', inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df['2010':]
    df = df[['open','high','low','close','volume']]
    df = df.astype(float)

    # df['change'] = df['close'].pct_change()

    df['amplitude'] = ((df['high'] - df['low']) / df['close'].shift(1))
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    df['returns_5d'] = (df['close'].shift(-PREDICT_KS) - df['close']) / df['close']

    # 定义区间
    bins = np.linspace(0,0.1,NUM_CLASS + 1)
    labels = np.arange(0,NUM_CLASS)

    # 使用 cut 函数生成标签
    # df['label'] = pd.cut(df['returns_5d'], bins=bins, labels=labels)
    df['label'] = (df['returns_5d'] > 0).astype(int)

    df.dropna(inplace=True)

    # df['volume_norm'] = (df['volume'] - df['volume'].min()) / (df['volume'].max() - df['volume'].min())

    train_size = int(len(df) * 0.8)
    train_set,test_set = df.iloc[:train_size],df.iloc[train_size:]
    return normilize(train_set),normilize(test_set)
(train_X,train_Y),(test_X,test_Y) = load_csv('data/510880.csv')
print(train_X)
hist1, bins1 = np.histogram(train_Y, bins=100)
hist2, bins2 = np.histogram(test_Y, bins=100)
# 可视化
# 可视化
plt.bar(bins1[:-1], hist1, width=bins1[1]-bins1[0], alpha=0.5, color='b', label='train')
plt.bar(bins2[:-1], hist2, width=bins2[1]-bins2[0], alpha=0.5, color='r', label='test')

plt.title('Distribution of Float Array')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('histogram.png')
plt.show()
print(train_X.shape,train_Y.shape)

# print(test_X)
train_dataset = StockDataset(train_X,train_Y)
test_dataset = StockDataset(test_X,test_Y)

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(test_dataset, batch_size=batch_size * 4, shuffle=False)
# data = dataset.transform(transform)
# data_preprocess("tdx_export/SZ#000001.txt")



# # 定义神经网络模型
# class SimpleNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x

# 定义训练函数
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for inputs, targets in dataloader:
        # print(inputs,targets)
        # print(inputs)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # print(targets,outputs)
        assert not torch.any(torch.isnan(outputs))
        # print(outputs)
        # print(outputs.squeeze().size(),targets.size())
        loss = criterion(outputs, targets)
        # print(loss.item())
        # break
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == targets).sum().item()
        total_predictions += targets.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct_predictions / total_predictions

    return epoch_loss,accuracy

best_acc = 0

# 定义验证函数
def validate(model, dataloader, criterion, device):
    global best_acc
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # print(targets,outputs)
            # print(outputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

            predicted_classes = torch.argmax(outputs, dim=1)
            # print(predicted_classes)
            
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_predictions += targets.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct_predictions / total_predictions
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model, 'model.pth')
    return epoch_loss,accuracy

# 设置参数
input_size = len(feature_columns)
learning_rate = 0.0001
num_epochs = 50
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerModel(input_size=input_size,seq_len=FEATURE_KS,num_class=NUM_CLASS).to(device)
# for batch in train_loader:
#     print(model(batch[0]).size())
#     break

# 准备数据
# model = SimpleNN(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# # 训练模型
for epoch in range(num_epochs):
    train_loss,train_accuracy = train(model, train_loader, criterion, optimizer, device)
    # train_loss = 0
    valid_loss,valid_accuracy = validate(model,valid_loader,criterion,device)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f} ,Train accuracy : {train_accuracy:.4f} , Valid Loss: {valid_loss:.4f}, Valid accuracy : {valid_accuracy:.4f}")


# model = torch.load('model.pth')
# valid_loss,accuracy = validate(model,valid_loader,criterion,device)

# print(f"Valid Loss: {valid_loss:.4f}, accuracy : {accuracy:.4f}")