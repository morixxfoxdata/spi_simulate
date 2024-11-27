# from utils.speckle_generate import generate_mask_pattern
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from skimage.metrics import structural_similarity as ssim

# from models.unet import NewMultiscaleSpeckleNet
# from models.unet import MultiscaleSpeckleNet
from models.autoencoder import Autoencoder
from models.unet import SimpleNet, TwoLayerNet, LargeNet, NewMultiscaleSpeckleNet

PATH = "/home1/komori/spi_simulate"
# PATH = "/Users/komori/Desktop/spi_simulate"
# PATH = "/Users/norikikomori/Desktop/spi_simulate"
# ====================
# numpy data loaded
# ====================
speckle_num = 49152

size = 256
EPOCHS = 10000
LEARNING_RATE = 1e-4
# USE_DATA = "mnist_0"
USE_DATA = "cameraman"
COMPRESSIVE_RATIO = speckle_num / size**2
if f"time{speckle_num}_{size}x{size}.npz" not in os.listdir(f"{PATH}/data/speckle/"):
    # print(os.listdir(f"{PATH}/data/speckle/"))
    print("SPECKLE does not exist!!")


# MASK_PATTERNS = np.load(f"{PATH}/data/speckle/time{speckle_num}_{size}x{size}.npz")[
#     "mask_patterns_normalized"
# ].astype(np.float32)
MASK_PATTERNS = np.load(f"{PATH}/data/speckle/time{speckle_num}_{size}x{size}.npz")[
    "arr_0"
].astype(np.float32)
# MASK_PATTERNS = np.load(f"{PATH}/data/speckle/time{speckle_num}_{size}x{size}.npz")
# model = Autoencoder(input_dim=speckle_num, hidden_dim=16, output_dim=size**2)
number = USE_DATA[-1]
# IMAGE = np.load(f"{PATH}/data/processed/mnist/mnist_{size}x{size}_{number}.npz")[
#     "arr_0"
# ].astype(np.float32)
IMAGE = np.load(f"{PATH}/data/processed/cameraman.npz")["arr_0"].astype(np.float32)
# ====================
# Device setting
# ====================
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print("Using CUDA (GPU)")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# model = MultiscaleSpeckleNet(outdim=size**2).to(DEVICE)
# model = NewMultiscaleSpeckleNet(outdim=size**2).to(DEVICE)

# model = Autoencoder(
#     input_dim=speckle_num, hidden_dim=speckle_num // 256, bottleneck_dim=256, output_dim=size**2
# ).to(DEVICE)
# model = SimpleNet(time_length=speckle_num, outdim=size**2).to(DEVICE)
# model = TwoLayerNet(outdim=size ** 2).to(DEVICE)
# model = LargeNet(outdim=size ** 2).to(DEVICE)
model = NewMultiscaleSpeckleNet(outdim=size ** 2).to(DEVICE)
model_name = model.__class__.__name__


def calculate_Y(X, S, time_length):
    X_flat = X.view(-1).float()
    S = S.reshape(time_length, -1).float()
    Y = torch.matmul(S, X_flat)
    Y = Y / torch.max(Y)
    return Y


def custom_loss(Y, X_prime, S, time_length):
    X_prime_flat = X_prime.view(-1)  # X'をフラット化
    S = S.reshape(time_length, -1).float()
    SX_prime = torch.matmul(S, X_prime_flat)
    SX_prime = SX_prime / torch.max(SX_prime)
    loss = torch.mean((Y - SX_prime) ** 2)
    return loss

# MSE の計算
def calculate_mse(image1, image2):
    # image1 = image1.flatten()
    return np.mean((image1 - image2) ** 2)


# 画像を比較して MSE と SSIM を表示する関数
def display_comparison_with_metrics(
    X_original, X_reconstructed, save_dir=f"{PATH}/data/results"
):
    # MSE 計算
    X_original = X_original.flatten()
    mse_value = calculate_mse(X_original, X_reconstructed)

    # SSIM 計算
    ssim_value, _ = ssim(
        X_original,
        X_reconstructed,
        data_range=X_original.max() - X_original.min(),
        full=True,
    )

    # 画像の表示
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # X_original の表示
    axes[0].imshow(X_original.reshape(size, size), cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    # X_reconstructed の表示
    axes[1].imshow(X_reconstructed.reshape(size, size), cmap="gray")
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")

    # MSE と SSIM を表示
    plt.suptitle(
        f"RATIO: {COMPRESSIVE_RATIO:.4f}, MSE: {mse_value:.6f}, SSIM: {ssim_value:.6f}",
        fontsize=14,
    )
    plt.tight_layout()
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 画像を保存
    save_path = os.path.join(
        save_dir,
        f"{model_name}/{size}{USE_DATA}_sp{speckle_num}_{model_name}_ep{EPOCHS}_{LEARNING_RATE}.png",
    )
    plt.savefig(save_path)
    print(f"Comparison plot saved to {save_path}")
    plt.show()


def main(device=DEVICE, mask_patterns=MASK_PATTERNS, image_data=IMAGE):
    image_data = torch.tensor(image_data)
    image_data = image_data.to(device)
    mask_patterns = torch.tensor(mask_patterns) / np.max(mask_patterns)
    mask_patterns = mask_patterns.to(device)
    Y = calculate_Y(image_data, mask_patterns, time_length=speckle_num)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # print(model)
    # トレーニングループ
    num_epochs = EPOCHS  # 学習エポック数
    Y = Y.unsqueeze(0).to(device)  # Yを(1, num_masks)の形状に変換
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        X_prime = model(Y)  # Yから再構成画像X'を生成
        loss = custom_loss(
            Y.squeeze(0), X_prime, mask_patterns, time_length=speckle_num
        )  # 損失を計算

        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.5f}")

    # 学習後の再構成画像を出力
    model.eval()
    with torch.no_grad():
        X_reconstructed = (
            model(Y).squeeze().cpu().numpy()
        )  # Yから再構成画像X'を生成し、NumPy配列に変換
        X_original = image_data.cpu().numpy()  # 元の画像XをNumPy配列に変換
    # 再構成画像の表示
    display_comparison_with_metrics(
        X_original=X_original, X_reconstructed=X_reconstructed
    )


if __name__ == "__main__":
    # print(list(MASK_PATTERNS.items()))
    main()
    # print(MASK_PATTERNS)
    # print(IMAGE.shape)
