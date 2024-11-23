import torch
import torch.nn as nn


class MultiscaleSpeckleNet(nn.Module):
    def __init__(self, outdim):
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
            nn.Linear(16, outdim),  # 入力次元を16に変更
            # nn.LeakyReLU(0.2),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(outdim, outdim),
            # nn.LeakyReLU(0.2),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(outdim, outdim),
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


# import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.prelu(out)
        return out


class NewMultiscaleSpeckleNet(nn.Module):
    def __init__(self, outdim):
        super(NewMultiscaleSpeckleNet, self).__init__()

        # Encoder
        self.down1 = nn.Sequential(self.down_sample(1, 32), ResidualBlock(32))
        self.down2 = nn.Sequential(self.down_sample(32, 64), ResidualBlock(64))
        self.down3 = nn.Sequential(self.down_sample(64, 128), ResidualBlock(128))

        # Decoder
        self.up3 = nn.Sequential(self.up_sample(128, 64), ResidualBlock(64))
        self.conv3 = self.conv_block(128, 64)  # 64 (up3) + 64 (d2) = 128
        self.up2 = nn.Sequential(self.up_sample(64, 32), ResidualBlock(32))
        self.conv2 = self.conv_block(64, 32)  # 32 (up2) + 32 (d1) = 64
        self.up1 = nn.Sequential(self.up_sample(32, 16), ResidualBlock(16))
        self.conv1 = self.conv_block(17, 16)  # 16 (up1) + 1 (x) = 17

        # Global Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Final fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(16, 64),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, outdim),
        )

        # Output activation
        self.activation = nn.Sigmoid()

        # Initialize weights
        self.initialize_weights()

    def down_sample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(),
            nn.Dropout(0.1),
        )

    def up_sample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(),
            nn.Dropout(0.1),
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(),
            nn.Dropout(0.1),
        )

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: (batch_size, 500)
        x = x.unsqueeze(1)  # (batch_size, 1, 500)

        # Encoder
        d1 = self.down1(x)  # (batch_size, 32, 250)
        d2 = self.down2(d1)  # (batch_size, 64, 125)
        d3 = self.down3(d2)  # (batch_size, 128, 63)

        # Decoder
        u3 = self.up3(d3)  # (batch_size, 64, 126)
        u3 = torch.cat([u3[:, :, : d2.size(2)], d2], dim=1)  # (batch_size, 128, 125)
        u3 = self.conv3(u3)  # (batch_size, 64, 125)

        u2 = self.up2(u3)  # (batch_size, 32, 250)
        u2 = torch.cat([u2, d1], dim=1)  # (batch_size, 64, 250)
        u2 = self.conv2(u2)  # (batch_size, 32, 250)

        u1 = self.up1(u2)  # (batch_size, 16, 500)
        u1 = torch.cat([u1, x], dim=1)  # (batch_size, 17, 500)
        u1 = self.conv1(u1)  # (batch_size, 16, 500)

        # Global Pooling
        pooled = self.global_pool(u1).squeeze(-1)  # (batch_size, 16)

        # Fully Connected Layers
        out = self.fc(pooled)  # (batch_size, outdim)
        activated_out = self.activation(out)

        return activated_out


if __name__ == "__main__":
    model = NewMultiscaleSpeckleNet(outdim=1024)
    Y = torch.randn(1, 1024)
    res = model(Y)
    print(res.shape)
