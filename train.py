import os
import pickle
from db import DB
from utils import *
import torch
import numpy as np
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler
from models import LeNet, TransformerModel

writer = SummaryWriter()


def build_dataset(index, start_date, end_data):
    cache_file = f"dataset_cache_{index}_{start_date}_{end_data}.pkl"

    # 检查缓存文件是否存在
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    db_client = DB()
    stocks = (
        db_client.get_index_stocks(index)
        if len(index) == 6
        else db_client.get_all_etf_code()
    )
    fields = ["open", "high", "low", "close", "volume", "turn"]
    all_data = db_client.get_price(stocks, end_data, fields, 99999, True, start_date)

    # 预处理所有股票数据
    valid_dfs = []
    for code in stocks:
        try:
            df = all_data.loc[code].reset_index()
            df = df[df["tradestatus"] == 1][fields].dropna()
            if len(df) >= 180:
                valid_dfs.append(df)
        except KeyError:
            continue

    # 批量计算所有样本和标签
    all_samples = []
    all_labels = []
    for df in valid_dfs:
        # 使用rolling窗口批量计算
        for i in range(len(df) - 80):
            sample = df.iloc[i : i + 60]
            future_slice = df.iloc[i + 60 : i + 80]
            change_percentage = (
                (np.mean(future_slice["low"]) - sample["close"].iloc[-1])
                / sample["close"].iloc[-1]
            ) * 100
            if not np.isnan(change_percentage):
                all_samples.append(sample.values)
                all_labels.append(change_percentage)

    # 批量归一化
    scaler = MinMaxScaler()
    normalized_samples = np.array(
        [scaler.fit_transform(sample) for sample in all_samples]
    )

    feature = torch.tensor(normalized_samples, dtype=torch.float32)
    labels = torch.tensor(all_labels, dtype=torch.float32)

    dataset = TensorDataset(feature, labels)

    # 保存缓存
    with open(cache_file, "wb") as f:
        pickle.dump(dataset, f)

    return dataset


class Model(nn.Module):
    def __init__(
        self,
        fc1_size=2000,
        fc2_size=1000,
        fc3_size=100,
        fc1_dropout=0.2,
        fc2_dropout=0.2,
        fc3_dropout=0.2,
        num_of_classes=50,
    ):
        super(Model, self).__init__()

        self.f_model = nn.Sequential(
            nn.Linear(3296, fc1_size),  # 887
            nn.BatchNorm1d(fc1_size),
            nn.ReLU(),
            nn.Dropout(fc1_dropout),
            nn.Linear(fc1_size, fc2_size),
            nn.BatchNorm1d(fc2_size),
            nn.ReLU(),
            nn.Dropout(fc2_dropout),
            nn.Linear(fc2_size, fc3_size),
            nn.BatchNorm1d(fc3_size),
            nn.ReLU(),
            nn.Dropout(fc3_dropout),
            nn.Linear(fc3_size, 1),
        )

        self.conv_layers1 = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=1),
            nn.BatchNorm1d(16),
            nn.Dropout(fc3_dropout),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.Dropout(fc3_dropout),
            nn.ReLU(),
        )

        self.conv_2D = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2),
            nn.BatchNorm2d(16),
            nn.Dropout(fc3_dropout),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.Dropout(fc3_dropout),
            nn.ReLU(),
        )
        hidden_dim = 32
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=4,
            batch_first=True,
            bidirectional=True,
        )
        hidden_dim = 1
        self.l = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=4,
            batch_first=True,
            bidirectional=True,
        )

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )

    def forward(self, x):
        apply = torch.narrow(x, dim=-1, start=0, length=1)[
            :,
            -90:,
        ].squeeze(1)
        redeem = torch.narrow(x, dim=-1, start=1, length=1)[
            :,
            -90:,
        ].squeeze(1)
        apply, _ = self.l(apply)
        redeem, _ = self.l(redeem)
        apply = torch.reshape(apply, (apply.shape[0], apply.shape[1] * apply.shape[2]))
        redeem = torch.reshape(
            redeem, (redeem.shape[0], redeem.shape[1] * redeem.shape[2])
        )

        ZFF = torch.narrow(x, dim=-1, start=2, length=1)[
            :,
            -90:,
        ].squeeze(1)
        HS = torch.narrow(x, dim=-1, start=3, length=1)[
            :,
            -90:,
        ].squeeze(1)
        ZFF, _ = self.l(ZFF)
        HS, _ = self.l(HS)
        ZFF = torch.reshape(ZFF, (ZFF.shape[0], ZFF.shape[1] * ZFF.shape[2]))
        HS = torch.reshape(HS, (HS.shape[0], HS.shape[1] * HS.shape[2]))

        min_vals, _ = torch.min(x, dim=1, keepdim=True)
        max_vals, _ = torch.max(x, dim=1, keepdim=True)
        x = (x - min_vals) / (max_vals - min_vals + 0.00001)

        xx = x.unsqueeze(1)
        xx = self.conv_2D(xx)
        xx = torch.reshape(xx, (xx.shape[0], xx.shape[1] * xx.shape[2] * xx.shape[3]))
        x = x.transpose(1, 2)
        x = self.conv_layers1(x)
        out = x.transpose(1, 2)
        out2, _ = self.lstm(out)
        out2 = torch.reshape(out2, (out2.shape[0], out2.shape[1] * out2.shape[2]))

        IN = torch.cat((xx, out2, apply, redeem, ZFF, HS), dim=1)
        out = self.f_model(IN)
        return out


def train_model(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    device,
    writer,
):
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            output = model(x).squeeze(1)
            loss = criterion(output, y)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        writer.add_scalar("Loss/train", epoch_loss / len(train_loader), epoch)

        with torch.no_grad():
            model.eval()
            valid_loss = 0.0
            for x, y in valid_loader:
                x = x.to(device)
                y = y.to(device)
                output = model(x).squeeze(1)
                loss = criterion(output, y)
                valid_loss += loss.item()
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss / len(train_loader)} Valid Loss: {valid_loss / len(valid_loader)}"
            )
            writer.add_scalar("Loss/valid", valid_loss / len(valid_loader), epoch)
            if epoch % 10 == 0:
                torch.save(model, f"model_{epoch}.pth")

    return model


def k_fold_cross_validation(
    k=5,
    start_date="20100101",
    end_date="20250101",
    num_epochs=300,
    batch_size=4096,
    learning_rate=0.0001,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    full_dataset = build_dataset(start_date, end_date)
    dataset_size = len(full_dataset)
    fold_size = dataset_size // k
    indices = torch.randperm(dataset_size).tolist()

    # Plot and save labels distribution
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.hist(full_dataset.tensors[1].numpy(), bins=50, alpha=0.7)
    plt.title("Labels Distribution")
    plt.xlabel("Change Percentage")
    plt.ylabel("Frequency")
    plt.savefig("labels_distribution.png")
    plt.close()

    fold_results = []

    for i in range(k):
        print(f"\nFold {i + 1}/{k}")
        valid_indices = indices[i * fold_size : (i + 1) * fold_size]
        train_indices = indices[: i * fold_size] + indices[(i + 1) * fold_size :]

        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        valid_dataset = torch.utils.data.Subset(full_dataset, valid_indices)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        model = LeNet().to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

        trained_model = train_model(
            model,
            train_loader,
            valid_loader,
            criterion,
            optimizer,
            scheduler,
            num_epochs,
            device,
            writer,
        )

        # Evaluate on validation set
        with torch.no_grad():
            model.eval()
            valid_loss = 0.0
            for x, y in valid_loader:
                x = x.to(device)
                y = y.to(device)
                output = model(x).squeeze(1)
                plt.figure(figsize=(10, 6))
                plt.hist(output.cpu().numpy(), bins=50, alpha=0.7)
                plt.title("Labels Distribution")
                plt.xlabel("Change Percentage")
                plt.ylabel("Frequency")
                plt.savefig("labels_distribution.png")
                plt.close()
                loss = criterion(output, y)
                valid_loss += loss.item()
            avg_valid_loss = valid_loss / len(valid_loader)
            fold_results.append(avg_valid_loss)
            print(f"Fold {i + 1} Validation Loss: {avg_valid_loss}")
        torch.save(model, f"model_{i + 1}.pth")

    print(
        f"\nAverage Validation Loss across {k} folds: {sum(fold_results) / len(fold_results)}"
    )

    return sum(fold_results) / len(fold_results)


if __name__ == "__main__":
    # k_fold_cross_validation()
    batch_size = 8192
    learning_rate = 0.0001
    num_epochs = 100
    ZZ1000 = "000852"
    ETF = "etf"
    train_dataset = build_dataset(ETF,"20200101", "20230101")

    valid_dataset = build_dataset(ETF,"20230101", "20250101")
    print(f"len(train_dataset): {len(train_dataset)}")
    print(f"len(valid_dataset): {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model = LeNet().to(device)
    model = TransformerModel(output_steps=1).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    trained_model = train_model(
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs,
        device,
        writer,
    )
