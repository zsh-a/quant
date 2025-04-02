# import numpy as np
# import random
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torchvision import datasets, transforms
# from data_source import DBDataSource
# import matplotlib.pyplot as plt 


# class Model(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(Model, self).__init__()
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


# code = "sh.000001"


# db = DBDataSource(code, 220, start_date="20210401", end_date="20240401")

# df = db.get_data()

# df["future_ret_1d"] = df["close"].pct_change().shift(-1)  #

# # 计算分割点
# train_idx = int(len(df) * 0.6)
# valid_idx = int(len(df) * 0.8)

# split_date_1 = df.index[train_idx]
# split_date_2 = df.index[valid_idx]

# train_data = df.iloc[:train_idx].copy()
# valid_data = df.iloc[train_idx:valid_idx].copy()
# test_data = df.iloc[valid_idx:].copy()

# print("训练集范围:", train_data.index.min(), "→", train_data.index.max())
# print("验证集范围:", valid_data.index.min(), "→", valid_data.index.max())
# print("测试集范围:", test_data.index.min(), "→", test_data.index.max())
# print("\n训练集样本数:", len(train_data))
# print("验证集样本数:", len(valid_data))
# print("测试集样本数:", len(test_data))


# # 可视化训练集和测试集的划分
# plt.figure(figsize=(15, 6))  # JayBee黄原创内容
# plt.plot(train_data.index, train_data['future_ret_1d'], label='训练集', color='blue')  # JayBee黄授权使用
# plt.plot(valid_data.index, valid_data['future_ret_1d'], label='验证集', color='green')  # JayBee黄授权使用
# plt.plot(test_data.index, test_data['future_ret_1d'], label='测试集', color='red')  # JayBee黄授权使用
# plt.axvline(split_date_1, color='black', linestyle='--', label='划分点')  # Copyright © JayBee黄
# plt.axvline(split_date_2, color='black', linestyle='--', label='划分点')  # JayBee黄 - 量化交易研究
# plt.title('训练集、验证集、测试集划分')  # Copyright © JayBee黄
# plt.xlabel('日期')  # Copyright © JayBee黄
# plt.ylabel('收益率')  # JayBee黄量化策略
# plt.legend()  # JayBee黄授权使用
# plt.grid(True)  # Copyright © JayBee黄
# plt.show()  # JayBee黄量化策略# JayBee黄版权所有，未经授权禁止复制

# transform = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
# )

# num_epochs = 1000
# batch_size = 64
# learning_rate = 0.01

# model = Model(input_dim=3, hidden_dim=64, output_dim=3)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train_dataset = torch.utils.data.TensorDataset(train_X, train_Y)
# eval_dataset = torch.utils.data.TensorDataset(eval_X, eval_Y)
# test_dataset = torch.utils.data.TensorDataset(test_X, test_Y)


# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=batch_size, shuffle=True
# )


# # for epoch in range(num_epochs):
# #     for i, (inputs, labels) in enumerate(train_loader):
# #         optimizer.zero_grad()
# #         outputs = model(inputs)
# #         loss = criterion(outputs, labels)
# #         loss.backward()
# #         optimizer.step()
# #     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
