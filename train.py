from db import DB
from utils import *
import torch
import numpy as np
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


def build_dataset(start_date, end_data):
    total_samples = []
    total_labels = []
    db_client = DB()

    for code in ["sz.000001", "sz.002594", "sz.300750"]:
        samples = []
        labels = []
        df = db_client.get_kline(code, start_date, end_data)

        df = df[df["tradestatus"] == 1]
        df = df[["open", "high", "low", "close", "amount", "turn"]]

        df = df.dropna()

        # 每列进行标准化
        for col in df.columns:
            df[col] = (df[col] - df[col].mean()) / df[col].std()

        num_samples = len(df)
        if num_samples < 180:
            return

        for i in range(num_samples - 85):
            sample = df.iloc[i : i + 60]
            samples.append(sample.values)
            labels.append(
                (np.mean(df.iloc[i + 60 : i + 85]["low"]) - sample["low"].values[-1])
                / sample["low"].values[-1]
            )

        labels = (np.array(labels) - np.mean(labels)) / np.std(labels) 

        total_samples.extend(samples)
        total_labels.extend(labels)
    feature = torch.tensor(np.array(total_samples), dtype=torch.float32)
    labels = torch.tensor(np.array(total_labels), dtype=torch.float32)

    return TensorDataset(feature, labels)


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


if __name__ == "__main__":
    learning_rate = 0.0001
    num_epochs = 500
    batch_size = 40000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = build_dataset("20100101", "20200101")

    valid_dataset = build_dataset("20200101", "20250101")

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
    criterion = nn.MSELoss()

    model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            # print(x)
            # print(y)
            y = y.to(device)
            output = model(x).squeeze(1)
            loss = criterion(output, y)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader)}")

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

        model.train()
    # torch.save(model.state_dict(), "model.pth")
