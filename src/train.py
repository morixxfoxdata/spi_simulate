# from utils.speckle_generate import generate_mask_pattern
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from models.autoencoder import Autoencoder

# ====================
# numpy data loaded
# ====================
speckle_num = 628
size = 28
mask_patterns = np.load(f"data/speckle/time{speckle_num}_{size}x{size}.npz")[
    "arr_0"
].astype(np.float32)
model = Autoencoder(input_dim=speckle_num, hidden_dim=128, output_dim=784)
image_data = np.load("data/processed/mnist/mnist_9.npz")["arr_0"].astype(np.float32)

# ====================
# Device setting
# ====================
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using CUDA (GPU)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")


def calculate_Y(X, S, speckle_num=speckle_num):
    X_flat = X.view(-1).float()
    S = S.reshape(speckle_num, -1).float()
    Y = torch.matmul(S, X_flat)
    Y = Y / torch.max(Y)
    return Y


def custom_loss(Y, X_prime, S, speckle_num=speckle_num):
    X_prime_flat = X_prime.view(-1)  # X'をフラット化
    S = S.reshape(speckle_num, -1).float()
    SX_prime = torch.matmul(S, X_prime_flat)
    SX_prime = SX_prime / torch.max(SX_prime)
    loss = torch.mean((Y - SX_prime) ** 2)
    return loss


def main(device=device, mask_patterns=mask_patterns, image_data=image_data):
    image_data = torch.tensor(image_data) / 255.0
    image_data = image_data.to(device)
    mask_patterns = torch.tensor(mask_patterns) / np.max(mask_patterns)
    mask_patterns = mask_patterns.to(device)
    Y = calculate_Y(image_data, mask_patterns)
    # gen1 = generate_mask_pattern(
    #     time_length=156, num_x_pixel_true=28, num_y_pixel_true=28
    # )
    # print(gen1.shape)
    # npz_data_mnist(9)
    # print(mask_patterns.shape)
    # print(max(image_data))
    # print(device)
    print(type(image_data))
    print(mask_patterns.shape)
    print("Y shape: ", Y.shape)
    model = Autoencoder(input_dim=speckle_num, hidden_dim=128, output_dim=784).to(
        device
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # print(model)
    # トレーニングループ
    num_epochs = 10000  # 学習エポック数
    Y = Y.unsqueeze(0).to(device)  # Yを(1, num_masks)の形状に変換
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        X_prime = model(Y)  # Yから再構成画像X'を生成
        loss = custom_loss(Y.squeeze(0), X_prime, mask_patterns)  # 損失を計算

        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

    # 学習後の再構成画像を出力
    model.eval()
    with torch.no_grad():
        X_reconstructed = (
            model(Y).squeeze().cpu().numpy()
        )  # Yから再構成画像X'を生成し、NumPy配列に変換
        X_original = image_data.cpu().numpy()  # 元の画像XをNumPy配列に変換

    # 再構成画像の表示
    plt.imshow(X_reconstructed, cmap="gray")
    plt.title("Reconstructed Image")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main(device, mask_patterns, image_data)
