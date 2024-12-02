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
        # x shape: (batch_size, 65536)
        x = x.unsqueeze(1)  # (batch_size, 1, 65536)

        # Encoder
        d1 = self.down1(x)  # (batch_size, 32, 32768)
        d2 = self.down2(d1)  # (batch_size, 64, 16384)
        d3 = self.down3(d2)  # (batch_size, 128, 8192)

        # Decoder
        u3 = self.up3(d3)  # (batch_size, 64, 16384)
        # 入力サイズに応じてトリミング
        u3 = torch.cat([u3[:, :, : d2.size(2)], d2], dim=1)  # (batch_size, 128, 125)

        u2 = self.up2(u3)  # (batch_size, 32, 32768)
        u2 = torch.cat([u2, d1], dim=1)  # (batch_size, 64, 250)

        u1 = self.up1(u2)  # (batch_size, 16, 500) ※入力サイズに応じて変更されます

        # グローバルプーリングで固定サイズに
        pooled = self.global_pool(u1).squeeze(-1)  # (batch_size, 16)

        # Flatten and pass through fully connected layers
        out = self.fc(pooled)  # (batch_size, 64)
        activated_out = self.activation(out)
        return activated_out

class NewMultiscaleSpeckleNet(nn.Module):
    def __init__(self, outdim):
        super(NewMultiscaleSpeckleNet, self).__init__()

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
            nn.Linear(16, 256),  # 入力次元を16に変更
            # nn.LeakyReLU(0.2),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            # nn.LeakyReLU(0.2),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, outdim),
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
        # x shape: (batch_size, 65536)
        x = x.unsqueeze(1)  # (batch_size, 1, 65536)

        # Encoder
        d1 = self.down1(x)  # (batch_size, 32, 32768)
        d2 = self.down2(d1)  # (batch_size, 64, 16384)
        d3 = self.down3(d2)  # (batch_size, 128, 8192)

        # Decoder
        u3 = self.up3(d3)  # (batch_size, 64, 16384)
        # 入力サイズに応じてトリミング
        u3 = torch.cat([u3[:, :, : d2.size(2)], d2], dim=1)  # (batch_size, 128, 125)

        u2 = self.up2(u3)  # (batch_size, 32, 32768)
        u2 = torch.cat([u2, d1], dim=1)  # (batch_size, 64, 250)

        u1 = self.up1(u2)  # (batch_size, 16, 500) ※入力サイズに応じて変更されます

        # グローバルプーリングで固定サイズに
        pooled = self.global_pool(u1).squeeze(-1)  # (batch_size, 16)

        # Flatten and pass through fully connected layers
        out = self.fc(pooled)  # (batch_size, 64)
        activated_out = self.activation(out)
        return activated_out

# import torch.nn as nn

class LargeNet(nn.Module):
    def __init__(self, outdim):
        super(LargeNet, self).__init__()

        # Encoder
        self.down1 = self.down_sample(1, 1024)
        self.down2 = self.down_sample(1024, 2048)
        self.down3 = self.down_sample(2048, 4096)

        # Decoder
        self.up3 = self.up_sample(4096, 2048)
        self.up2 = self.up_sample(4096, 1024)
        self.up1 = self.up_sample(2048, 1024)
        self.activation = nn.Sigmoid()
        # self.activation = BinaryActivation()
        # self.activation = BinarySTEActivation()
        # self.activation = SmoothBinaryActivation()

        # グローバルプーリングを追加
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 出力サイズを1に固定

        # Final fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(1024, 256),  # 入力次元を16に変更
            # nn.LeakyReLU(0.2),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, outdim),
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
        # x shape: (batch_size, 65536)
        x = x.unsqueeze(1)  # (batch_size, 1, 65536)

        # Encoder
        d1 = self.down1(x)  # (batch_size, 32, 32768)
        d2 = self.down2(d1)  # (batch_size, 64, 16384)
        d3 = self.down3(d2)  # (batch_size, 128, 8192)

        # Decoder
        u3 = self.up3(d3)  # (batch_size, 64, 16384)
        # 入力サイズに応じてトリミング
        u3 = torch.cat([u3[:, :, : d2.size(2)], d2], dim=1)  # (batch_size, 128, 125)

        u2 = self.up2(u3)  # (batch_size, 32, 32768)
        u2 = torch.cat([u2, d1], dim=1)  # (batch_size, 64, 250)

        u1 = self.up1(u2)  # (batch_size, 16, 500) ※入力サイズに応じて変更されます

        # グローバルプーリングで固定サイズに
        pooled = self.global_pool(u1).squeeze(-1)  # (batch_size, 16)

        # Flatten and pass through fully connected layers
        out = self.fc(pooled)  # (batch_size, 64)
        activated_out = self.activation(out)
        return activated_out

class SimpleNet(nn.Module):
    def __init__(self, time_length, outdim):
        super(SimpleNet, self).__init__()
        self.down1 = self.down_sample(1, 32)
        self.down2 = self.down_sample(32, 64)
        self.down3 = self.down_sample(64, 128)

        self.up3 = self.up_sample(128, 64)
        self.up2 = self.up_sample(128, 32)
        self.up1 = self.up_sample(64, 16)
        self.compression = nn.Sequential(
            nn.Linear(time_length, 64),
            nn.PReLU(),
        )
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
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 出力サイズを1に固定
        self.activation = nn.Sigmoid()
    def down_sample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(),
        )
    def up_sample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x): # Input size: (1, 1, time_length(default:65536))
        com_x = self.compression(x)
        print("com_x shape:", com_x.shape)
        com_x_3 = com_x.unsqueeze(1)
        # return com_x_3
        d1 = self.down1(com_x_3) # d1 size: (1, 32, 32768)
        print("d1 shape:", d1.shape)
        d2 = self.down2(d1) # d2 size: (1, 64, 16384)
        print("d2 shape:", d2.shape)
        d3 = self.down3(d2) # d3 size: (1, 128, 8192)
        print("d3 shape:", d3.shape)
        u3 = self.up3(d3) # u3 size: (1, 64, 16384)
        print("u3 shape:", u3.shape)
        u3_d2 = torch.cat([u3, d2], dim=1) # u3_d2 size: (1, 128, 16384)
        print("u3_d2 shape:", u3_d2.shape)
        u2 = self.up2(u3_d2) # u2 size: (1, 32, 32768)
        print("u2 shape:", u2.shape)
        u2_d1 = torch.cat([u2, d1], dim=1)
        print("u2_d1 shape:", u2_d1.shape)
        u1 = self.up1(u2_d1)
        print("u1 shape:", u1.shape)
        pooled = self.global_pool(u1).squeeze(-1)
        print("pooled shape:", pooled.shape)
        out = self.fc(pooled)
        activated_out = self.activation(out)
        return activated_out


class TwoLayerNet(nn.Module):
    def __init__(self, outdim):
        super(TwoLayerNet, self).__init__()
        self.down1 = self.down_sample(1, 32)
        self.down2 = self.down_sample(32, 64)
        self.up1 = self.up_sample(64, 32)
        self.up2 = self.up_sample(64, 16)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 出力サイズを1に固定
        self.fc = nn.Sequential(
            nn.Linear(16, 256),  # 入力次元を16に変更
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, outdim),
        )
        self.activation = nn.Sigmoid()
    def down_sample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(),
        )
    def up_sample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(),
        )
    def forward(self, x):
        x = x.unsqueeze(1)
        d1 = self.down1(x)
        # print("d1 shape:", d1.shape)
        d2 = self.down2(d1)
        # print("d2 shape:", d2.shape)
        u1 = self.up1(d2)
        # print("u1 shape:", u1.shape)
        u1_d1 = torch.cat([u1, d1], dim=1)
        # print("u1_d1 shape:", u1_d1.shape)
        u2 = self.up2(u1_d1)
        # print("u2 shape:", u2.shape)
        pooled = self.global_pool(u2).squeeze(-1)
        # print("pooled shape:", pooled.shape)
        out = self.fc(pooled)
        activated_out = self.activation(out)
        return activated_out



class ModifiedMultiscaleSpeckleNet(nn.Module):
    def __init__(self, outdim):
        super(ModifiedMultiscaleSpeckleNet, self).__init__()

        # Encoder
        self.down1 = self.down_sample(1, 64)
        self.down2 = self.down_sample(64, 128)
        self.down3 = self.down_sample(128, 256)

        # Decoder
        self.up3 = self.up_sample(256, 128)
        self.up2 = self.up_sample(256, 64)
        self.up1 = self.up_sample(128, 32)
        self.up0 = self.up_sample(32, 1)
        self.activation = nn.Sigmoid()
        # self.activation = BinaryActivation()
        # self.activation = BinarySTEActivation()
        # self.activation = SmoothBinaryActivation()

        # グローバルプーリングを追加
        self.global_pool = nn.AdaptiveMaxPool1d(2048)  # 出力サイズを1に固定

        # Final fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(2048, 2048),  # 入力次元を16に変更
            # nn.LeakyReLU(0.2),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, outdim),
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
        # x shape: (batch_size, 65536)
        x = x.unsqueeze(1)  # (batch_size, 1, 65536)
        # print("input dimension: ", x.shape)
        # Encoder
        d1 = self.down1(x)  # (batch_size, 32, 32768)
        # print("d1 shape:", d1.shape)
        d2 = self.down2(d1)  # (batch_size, 64, 16384)
        # print("d2 shape:", d2.shape)
        d3 = self.down3(d2)  # (batch_size, 128, 8192)
        # print("d3 shape:", d3.shape)

        # Decoder
        u3 = self.up3(d3)  # (batch_size, 64, 16384)
        # print("u3 shape:", u3.shape)
        # 入力サイズに応じてトリミング
        u3 = torch.cat([u3[:, :, : d2.size(2)], d2], dim=1)  # (batch_size, 128, 125)
        # print("u3(after concated with d2) shape:", u3.shape)
        u2 = self.up2(u3)  # (batch_size, 32, 32768)
        # print("u2 shape:", u2.shape)
        u2 = torch.cat([u2, d1], dim=1)  # (batch_size, 64, 250)
        # print("u2(after concated with d1) shape:", u2.shape)
        u1 = self.up1(u2)  # (batch_size, 16, 500) ※入力サイズに応じて変更されます
        # print("u1 shape:", u1.shape)
        u0 = self.up0(u1)
        # print("u0 shape:", u0.shape)
        # グローバルプーリングで固定サイズに
        pooled = self.global_pool(u0).squeeze(-1)  # (batch_size, 16)
        # print("pooled shape:", pooled.shape)
        # Flatten and pass through fully connected layers
        out = self.fc(pooled)  # (batch_size, 64)
        activated_out = self.activation(out)
        return activated_out


class DeepMultiscaleSpeckleNet(nn.Module):
    def __init__(self, outdim):
        super(DeepMultiscaleSpeckleNet, self).__init__()

        # Encoder
        self.down1 = self.down_sample(1, 64)
        self.down2 = self.down_sample(64, 128)
        self.down3 = self.down_sample(128, 256)

        # Decoder
        self.up3 = self.up_sample(256, 128)
        self.up2 = self.up_sample(256, 64)
        self.up1 = self.up_sample(128, 32)
        self.up0 = self.up_sample(32, 1)
        self.activation = nn.Sigmoid()
        # self.activation = BinaryActivation()
        # self.activation = BinarySTEActivation()
        # self.activation = SmoothBinaryActivation()

        # グローバルプーリングを追加
        self.global_pool = nn.AdaptiveMaxPool1d(4096)  # 出力サイズを1に固定

        # Final fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(4096, 4096),  # 入力次元を16に変更
            # nn.LeakyReLU(0.2),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(4096, outdim),
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
        # x shape: (batch_size, 65536)
        x = x.unsqueeze(1)  # (batch_size, 1, 65536)
        # print("input dimension: ", x.shape)
        # Encoder
        d1 = self.down1(x)  # (batch_size, 32, 32768)
        # print("d1 shape:", d1.shape)
        d2 = self.down2(d1)  # (batch_size, 64, 16384)
        # print("d2 shape:", d2.shape)
        d3 = self.down3(d2)  # (batch_size, 128, 8192)
        # print("d3 shape:", d3.shape)

        # Decoder
        u3 = self.up3(d3)  # (batch_size, 64, 16384)
        # print("u3 shape:", u3.shape)
        # 入力サイズに応じてトリミング
        u3 = torch.cat([u3[:, :, : d2.size(2)], d2], dim=1)  # (batch_size, 128, 125)
        # print("u3(after concated with d2) shape:", u3.shape)
        u2 = self.up2(u3)  # (batch_size, 32, 32768)
        # print("u2 shape:", u2.shape)
        u2 = torch.cat([u2, d1], dim=1)  # (batch_size, 64, 250)
        # print("u2(after concated with d1) shape:", u2.shape)
        u1 = self.up1(u2)  # (batch_size, 16, 500) ※入力サイズに応じて変更されます
        # print("u1 shape:", u1.shape)
        u0 = self.up0(u1)
        # print("u0 shape:", u0.shape)
        # グローバルプーリングで固定サイズに
        pooled = self.global_pool(u0).squeeze(-1)  # (batch_size, 16)
        # print("pooled shape:", pooled.shape)
        # Flatten and pass through fully connected layers
        out = self.fc(pooled)  # (batch_size, 64)
        activated_out = self.activation(out)
        return activated_out

if __name__ == "__main__":
    # model = NewMultiscaleSpeckleNet(outdim=1024)
    model = SimpleNet(time_length=1024, outdim=65536)
    Y = torch.randn(1, 1024)
    res = model(Y)
    print(res.shape)