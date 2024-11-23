import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock2D(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.instnorm1 = nn.InstanceNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.instnorm2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.instnorm1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.instnorm2(out)
        out += residual
        out = self.prelu(out)
        return out


class HybridAutoEncoder(nn.Module):
    def __init__(self, input_size=32, latent_size=128, image_size=64):
        super(HybridAutoEncoder, self).__init__()

        # エンコーダー: 全結合層 + LayerNorm
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, latent_size),
            nn.LayerNorm(latent_size),
            nn.ReLU(),
        )

        # デコーダーの全結合層: 潜在空間から初期画像特徴を生成
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, image_size),
            nn.Sigmoid(),  # ピクセル値を [0, 1] に制限
        )

        # デコーダーの畳み込み層: 初期画像特徴から再構成画像を生成
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(
                1, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # (batch,32,8,8) -> (batch,32,16,16)
            nn.InstanceNorm2d(32),
            nn.PReLU(),
            ResidualBlock2D(32),
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # (batch,16,16,16) -> (batch,16,32,32)
            nn.InstanceNorm2d(16),
            nn.PReLU(),
            ResidualBlock2D(16),
            nn.Conv2d(16, 1, kernel_size=1),  # (batch,1,32,32) -> (batch,1,32,32)
            nn.Sigmoid(),  # ピクセル値を [0, 1] に制限
        )

        # 重みの初期化
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if (
                isinstance(m, nn.Linear)
                or isinstance(m, nn.ConvTranspose2d)
                or isinstance(m, nn.Conv2d)
            ):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, y):
        """
        Args:
            y: Tensor of shape (batch_size, input_size=32)
        Returns:
            x: Tensor of shape (batch_size, image_size=64)
        """
        batch_size = y.size(0)

        # エンコーダーを通過
        latent = self.encoder(y)  # (batch_size, latent_size=128)

        # デコーダーの全結合層を通過
        img_flat = self.decoder_fc(latent)  # (batch_size, image_size=64)

        # 画像を8x8にリシェイプ
        img = img_flat.view(batch_size, 1, 8, 8)  # (batch_size,1,8,8)

        # デコーダーの畳み込み層を通過
        out = self.decoder_conv(img)  # (batch_size,1,32,32)

        # 出力を64ピクセルにフラット化（必要に応じて）
        out = out.view(batch_size, -1)  # (batch_size, 32*32=1024)

        # ここで画像サイズを8x8に戻すために、さらにリシェイプが必要
        # もしくは、出力を適切に処理します。
        # 以下では、出力を8x8にダウンサンプリングします。
        out = F.interpolate(
            out.view(batch_size, 1, 32, 32),
            size=(8, 8),
            mode="bilinear",
            align_corners=False,
        )
        out = out.view(batch_size, -1)  # (batch_size, 64)

        return out


# テスト用コード
if __name__ == "__main__":
    # ダミーマスクパターンの作成
    num_masks = 32  # マスクパターンの数
    height = width = 8  # マスクパターンの空間次元
    mask_patterns = torch.randn(num_masks, height, width)  # ランダムなマスクパターン

    # モデルのインスタンス化
    model = HybridAutoEncoder(input_size=num_masks, latent_size=128, image_size=64)

    # サンプル入力の作成: バッチサイズ1のランダムな入力
    Y = torch.randn(1, 16)  # (1, 32)

    # フォワードパスの実行
    X_prime = model(Y)  # (1, 64)

    print("入力 Y の形状:", Y.shape)
    print("出力 X' の形状:", X_prime.shape)
