import torch
import torch.nn as nn


class MultiscaleSpeckleNet(nn.Module):
    def __init__(self):
        super(MultiscaleSpeckleNet, self).__init__()

        # Encoder
        self.down1 = self.down_sample(1, 32)
        self.down2 = self.down_sample(32, 64)
        self.down3 = self.down_sample(64, 128)

        # Decoder
        self.up3 = self.up_sample(128, 64)
        self.up2 = self.up_sample(128, 32)
        self.up1 = self.up_sample(64, 16)
        self.activation = nn.Sigmoid()
        # self.activation = BinaryActivation()
        # self.activation = BinarySTEActivation()
        # self.activation = SmoothBinaryActivation()

        # グローバルプーリングを追加
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 出力サイズを1に固定

        # Final fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(16, 1024),  # 入力次元を16に変更
            # nn.LeakyReLU(0.2),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 800),
            # nn.LeakyReLU(0.2),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(800, 784),
            # nn.Sigmoid(),
        )

    def down_sample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_channels),
            # nn.ReLU(inplace=True),
            nn.PReLU(),
            nn.Dropout(0.1),
        )

    def up_sample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm1d(out_channels),
            # nn.ReLU(inplace=True),
            nn.PReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        # batch_size=1(1枚分だから)
        # x shape: (batch_size, 500)
        x = x.unsqueeze(1)  # (batch_size, 1, 500)

        # Encoder
        d1 = self.down1(x)  # (batch_size, 32, 250)
        d2 = self.down2(d1)  # (batch_size, 64, 125)
        d3 = self.down3(d2)  # (batch_size, 128, 63)

        # Decoder
        u3 = self.up3(d3)  # (batch_size, 64, 126)
        # 入力サイズに応じてトリミング
        u3 = torch.cat([u3[:, :, : d2.size(2)], d2], dim=1)  # (batch_size, 128, 125)

        u2 = self.up2(u3)  # (batch_size, 32, 250)
        u2 = torch.cat([u2, d1], dim=1)  # (batch_size, 64, 250)

        u1 = self.up1(u2)  # (batch_size, 16, 500) ※入力サイズに応じて変更されます

        # グローバルプーリングで固定サイズに
        pooled = self.global_pool(u1).squeeze(-1)  # (batch_size, 16)

        # Flatten and pass through fully connected layers
        out = self.fc(pooled)  # (batch_size, 64)
        activated_out = self.activation(out)

        return activated_out


if __name__ == "__main__":
    model = MultiscaleSpeckleNet()
    Y = torch.randn(1, 1200)
    res = model(Y)
    print(res.shape)
