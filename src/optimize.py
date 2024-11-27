import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.optim as optim
from skimage.metrics import structural_similarity as ssim

from models.unet import MultiscaleSpeckleNet

# PATH の設定
PATH = "/Users/norikikomori/Desktop/spi_simulate"

# ハイパーパラメータの固定値
speckle_num = 16
size = 8
USE_DATA = "mnist_0"
COMPRESSIVE_RATIO = speckle_num / size**2

# データのロード
if f"time{speckle_num}_{size}x{size}.npz" not in os.listdir(f"{PATH}/data/speckle/"):
    print("SPECKLE does not exist!!")

MASK_PATTERNS = np.load(f"{PATH}/data/speckle/time{speckle_num}_{size}x{size}.npz")[
    "arr_0"
].astype(np.float32)
number = USE_DATA[-1]
IMAGE = np.load(f"{PATH}/data/processed/mnist/mnist_{size}x{size}_{number}.npz")[
    "arr_0"
].astype(np.float32)

# デバイス設定
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print("Using CUDA (GPU)")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")


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


def calculate_mse(image1, image2):
    return np.mean((image1 - image2) ** 2)


def display_comparison_with_metrics(
    X_original,
    X_reconstructed,
    save_dir="data/results",
    speckle_num=16,
    size=8,
    use_data="mnist_0",
    model_name="Model",
    epoch=0,
    lr=1e-3,
):
    mse_value = calculate_mse(X_original, X_reconstructed)
    ssim_value, _ = ssim(
        X_original,
        X_reconstructed,
        data_range=X_original.max() - X_original.min(),
        full=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(X_original.reshape(size, size), cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(X_reconstructed.reshape(size, size), cmap="gray")
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")

    plt.suptitle(
        f"RATIO: {COMPRESSIVE_RATIO:.4f}, MSE: {mse_value:.4f}, SSIM: {ssim_value:.4f}",
        fontsize=14,
    )
    plt.tight_layout()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(
        save_dir,
        f"{size}{use_data}_sp{speckle_num}_{model_name}_ep{epoch}_lr{lr}.png",
    )
    plt.savefig(save_path)
    print(f"Comparison plot saved to {save_path}")
    plt.close()


def objective(trial):
    # ハイパーパラメータの提案
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-3)
    epochs = trial.suggest_int("epochs", 1000, 10000)

    # モデルの初期化
    model = MultiscaleSpeckleNet().to(DEVICE)

    # オプティマイザの設定
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # データの準備
    image_data = torch.tensor(IMAGE).to(DEVICE)
    mask_patterns = torch.tensor(MASK_PATTERNS) / np.max(MASK_PATTERNS)
    mask_patterns = mask_patterns.to(DEVICE)
    Y = calculate_Y(image_data, mask_patterns, time_length=speckle_num)
    Y = Y.unsqueeze(0).to(DEVICE)  # バッチ次元を追加

    # トレーニングループ
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        X_prime = model(Y)
        loss = custom_loss(
            Y.squeeze(0), X_prime, mask_patterns, time_length=speckle_num
        )

        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(
                f"Trial {trial.number} - Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}"
            )

    # 評価
    model.eval()
    with torch.no_grad():
        X_reconstructed = model(Y).squeeze().cpu().numpy()
        X_original = image_data.cpu().numpy()

    mse_value = calculate_mse(X_original, X_reconstructed)
    ssim_value, _ = ssim(
        X_original,
        X_reconstructed,
        data_range=X_original.max() - X_original.min(),
        full=True,
    )

    # MSEを最小化対象とする
    return mse_value


def trial():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)  # 試行回数を適宜調整

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value (MSE): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # 最良のハイパーパラメータで再トレーニングし、結果を表示
    best_lr = trial.params["learning_rate"]
    best_epochs = trial.params["epochs"]

    # モデルの初期化
    best_model = MultiscaleSpeckleNet().to(DEVICE)
    optimizer = optim.Adam(best_model.parameters(), lr=best_lr)

    # データの準備
    image_data = torch.tensor(IMAGE).to(DEVICE)
    mask_patterns = torch.tensor(MASK_PATTERNS) / np.max(MASK_PATTERNS)
    mask_patterns = mask_patterns.to(DEVICE)
    Y = calculate_Y(image_data, mask_patterns, time_length=speckle_num)
    Y = Y.unsqueeze(0).to(DEVICE)  # バッチ次元を追加

    # トレーニングループ
    for epoch in range(best_epochs):
        best_model.train()
        optimizer.zero_grad()

        X_prime = best_model(Y)
        loss = custom_loss(
            Y.squeeze(0), X_prime, mask_patterns, time_length=speckle_num
        )

        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(
                f"Best Model - Epoch [{epoch}/{best_epochs}], Loss: {loss.item():.4f}"
            )

    # 再構成画像の表示
    best_model.eval()
    with torch.no_grad():
        X_reconstructed = best_model(Y).squeeze().cpu().numpy()
        X_original = image_data.cpu().numpy()

    display_comparison_with_metrics(
        X_original=X_original,
        X_reconstructed=X_reconstructed,
        speckle_num=speckle_num,
        size=size,
        use_data=USE_DATA,
        model_name=best_model.__class__.__name__,
        epoch=best_epochs,
        lr=best_lr,
    )


if __name__ == "__main__":
    trial()
