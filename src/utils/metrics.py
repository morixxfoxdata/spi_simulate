import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from skimage.metrics import structural_similarity as ssim
# _PATH = os.getcwd()
PATH = "/home1/komori/spi_simulate"
# res_path = "home1/komori/spi_simulate/data/results"
def calculate_mse(image1, image2):
    # image1 = image1.flatten()
    return np.mean((image1 - image2) ** 2)



# 画像を比較して MSE と SSIM を表示する関数
def display_comparison_with_metrics(
    X_original, X_reconstructed, speckle_num, size, USE_DATA, LOSS_SELECT, EPOCHS, LEARNING_RATE, model_name, save_dir=f"{PATH}/data/results", ratio=1
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
        f"RATIO: {ratio:.4f}, MSE: {mse_value:.6f}, SSIM: {ssim_value:.6f}",
        fontsize=14,
    )
    plt.tight_layout()
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 画像を保存
    save_path = os.path.join(
        save_dir,
        f"{model_name}/{size}{USE_DATA}_sp{speckle_num}_{LOSS_SELECT}_{model_name}_ep{EPOCHS}_{LEARNING_RATE}.png",
    )
    plt.savefig(save_path)
    print(f"Comparison plot saved to {save_path}")

# if __name__ == "__main__":
    # print(res_path)
    # print(PATH)