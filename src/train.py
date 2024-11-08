# from utils.speckle_generate import generate_mask_pattern
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from skimage.metrics import structural_similarity as ssim

# from models.unet import MultiscaleSpeckleNet
from models.autoencoder import Autoencoder

# PATH = "/home1/komori/spi_simulate/"
# PATH = "/Users/komori/Desktop/spi_simulate"
PATH = "/Users/norikikomori/Desktop/spi_simulate"
# ====================
# numpy data loaded
# ====================
speckle_num = 16
size = 8
EPOCHS = 10000
USE_DATA = "mnist_0"
COMPRESSIVE_RATIO = speckle_num / size**2
if f"time{speckle_num}_{size}x{size}.npz" not in os.listdir(f"{PATH}/data/speckle/"):
    # print(os.listdir(f"{PATH}/data/speckle/"))
    print("SPECKLE does not exist!!")

MASK_PATTERNS = np.load(f"{PATH}/data/speckle/time{speckle_num}_{size}x{size}.npz")[
    "arr_0"
].astype(np.float32)
# model = Autoencoder(input_dim=speckle_num, hidden_dim=16, output_dim=size**2)
number = USE_DATA[-1]
IMAGE = np.load(f"{PATH}/data/processed/mnist/mnist_{size}x{size}_{number}.npz")[
    "arr_0"
].astype(np.float32)

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

# model = EnhancedAutoencoder(
#     input_dim=speckle_num, hidden_dim=speckle_num // 4, output_dim=size**2
# ).to(DEVICE)
# model = MultiscaleSpeckleNet().to(DEVICE)
# model = BasedDecoder(hidden_dim=speckle_num, output_dim=size**2).to(DEVICE)
model = Autoencoder(
    input_dim=speckle_num, hidden_dim=speckle_num // 4, output_dim=size**2
).to(DEVICE)
# model = ShortAutoencoder(
#     input_dim=speckle_num, hidden_dim=speckle_num // 4, output_dim=size**2
# ).to(DEVICE)
# print(model.__class__.__name__)
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
    return np.mean((image1 - image2) ** 2)


# def display_comparison(size, X_original, X_reconstructed):
#     fig, axes = plt.subplots(1, 2, figsize=(8, 4))

#     # X_original の表示
#     axes[0].imshow(X_original.reshape(size, -1), cmap="gray")
#     axes[0].set_title("Original")
#     axes[0].axis("off")

#     # X_reconstructed の表示
#     axes[1].imshow(X_reconstructed.reshape(size, -1), cmap="gray")
#     axes[1].set_title("Reconstructed")
#     axes[1].axis("off")

#     plt.tight_layout()
#     plt.show()


# 画像を比較して MSE と SSIM を表示する関数
def display_comparison_with_metrics(
    X_original, X_reconstructed, save_dir="data/results"
):
    # MSE 計算
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
        f"RATIO: {COMPRESSIVE_RATIO:.4f}, MSE: {mse_value:.4f}, SSIM: {ssim_value:.4f}",
        fontsize=14,
    )
    plt.tight_layout()
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 画像を保存
    save_path = os.path.join(
        save_dir,
        f"{size}{USE_DATA}_sp{speckle_num}_{model_name}_ep{EPOCHS}.png",
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
    # gen1 = generate_mask_pattern(
    #     time_length=156, num_x_pixel_true=28, num_y_pixel_true=28
    # )
    # print(gen1.shape)
    # npz_data_mnist(9)
    # print(mask_patterns.shape)
    # print(max(image_data))
    # print(device)
    # print(type(image_data))
    # print(mask_patterns.shape)
    # print("Y shape: ", Y.shape)
    # model = EnhancedAutoencoder(input_dim=speckle_num, hidden_dim=16, output_dim=64).to(
    #     device
    # )
    # model = Autoencoder(input_dim=speckle_num, hidden_dim=128, output_dim=784).to(
    #     device
    # )
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
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
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

    # 学習後の再構成画像を出力
    model.eval()
    with torch.no_grad():
        X_reconstructed = (
            model(Y).squeeze().cpu().numpy()
        )  # Yから再構成画像X'を生成し、NumPy配列に変換
        X_original = image_data.cpu().numpy()  # 元の画像XをNumPy配列に変換
    # print(X_reconstructed.shape)
    # print(X_original.shape)
    # 再構成画像の表示
    display_comparison_with_metrics(
        X_original=X_original, X_reconstructed=X_reconstructed
    )
    # plt.imshow(X_reconstructed, cmap="gray")
    # plt.title("Reconstructed Image")
    # plt.axis("off")
    # plt.show()


if __name__ == "__main__":
    main()
    # print("Hello world!")
