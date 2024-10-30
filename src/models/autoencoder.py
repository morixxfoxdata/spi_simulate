import torch
import torch.nn as nn


# エンコーダ
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x


# デコーダ
class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # 0〜1の範囲に正規化
        x = x.view(-1, 28, 28)  # 出力を256x256画像の形状に変換
        return x


# 全体のモデル
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# エンコーダ
class EnhancedEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EnhancedEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim * 4)
        self.fc4 = nn.Linear(hidden_dim * 4, hidden_dim * 4)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim * 2)
        self.layer_norm3 = nn.LayerNorm(hidden_dim * 4)

    def forward(self, x):
        x = torch.relu(self.layer_norm1(self.fc1(x)))
        x = torch.relu(self.layer_norm2(self.fc2(x)))
        x = torch.relu(self.layer_norm3(self.fc3(x)))
        x = torch.relu(self.fc4(x))
        return x


# デコーダ
class EnhancedDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(EnhancedDecoder, self).__init__()
        self.fc1 = nn.Linear(hidden_dim * 4, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim * 4)
        self.layer_norm2 = nn.LayerNorm(hidden_dim * 2)
        self.layer_norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = torch.relu(self.layer_norm1(self.fc1(x)))
        x = torch.relu(self.layer_norm2(self.fc2(x)))
        x = torch.relu(self.layer_norm3(self.fc3(x)))
        x = torch.sigmoid(self.fc4(x))  # 0〜1の範囲に正規化
        x = x.view(-1, 28, 28)  # 出力を256x256画像の形状に変換
        return x


# 全体のモデル
class EnhancedAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EnhancedAutoencoder, self).__init__()
        self.encoder = EnhancedEncoder(input_dim, hidden_dim)
        self.decoder = EnhancedDecoder(hidden_dim, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
