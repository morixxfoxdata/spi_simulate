# from utils.speckle_generate import generate_mask_pattern
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from skimage.metrics import structural_similarity as ssim
from utils.loss_fun import custom_loss, l1_custom_loss, l1_tv_custom_loss, tv_custom_loss
from utils.metrics import display_comparison_with_metrics
# from models.unet import NewMultiscaleSpeckleNet
# from models.unet import MultiscaleSpeckleNet
from models.autoencoder import Autoencoder
from models.unet import shal_1_DeepMultiscaleSpeckleNet, NewMultiscaleSpeckleNet, ModifiedMultiscaleSpeckleNet, DeepMultiscaleSpeckleNet, NewDeepMultiscaleSpeckleNet

PATH = "/home1/komori/spi_simulate"
# PATH = "/Users/komori/Desktop/spi_simulate"
# PATH = "/Users/norikikomori/Desktop/spi_simulate"
# ====================
# numpy data loaded
# ====================
speckle_num_list = [65536]
LOSS_SELECT = "l1_tv"
# 正則化しない場合でもlambda_regを0に設定する
# lambda_reg = 0.00001
alpha = 0.0
beta = 0.0
size = 256
EPOCHS = 2000
LEARNING_RATE = 1e-4
# USE_DATA = "mnist_0"
USE_DATA = "cameraman"
# COMPRESSIVE_RATIO = speckle_num / size**2
# if f"time{speckle_num}_{size}x{size}.npz" not in os.listdir(f"{PATH}/data/speckle/"):
#     # print(os.listdir(f"{PATH}/data/speckle/"))
#     print("SPECKLE does not exist!!")


# MASK_PATTERNS = np.load(f"{PATH}/data/speckle/time{speckle_num}_{size}x{size}.npz")[
#     "mask_patterns_normalized"
# ].astype(np.float32)
# MASK_PATTERNS = np.load(f"{PATH}/data/speckle/time{speckle_num}_{size}x{size}.npz")[
#     "arr_0"
# ].astype(np.float32)
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
# model = NewMultiscaleSpeckleNet(outdim=size ** 2).to(DEVICE)
# model = ModifiedMultiscaleSpeckleNet(outdim=size ** 2).to(DEVICE)
# model = NewDeepMultiscaleSpeckleNet(outdim=size ** 2).to(DEVICE)
# model_name = model.__class__.__name__


def calculate_Y(X, S, time_length):
    X_flat = X.view(-1).float()
    S = S.reshape(time_length, -1).float()
    Y = torch.matmul(S, X_flat)
    Y = Y / torch.max(Y)
    return Y


def main(speckle_num, model, mask_patterns, image_data=IMAGE, device=DEVICE):
    COMPRESSIVE_RATIO = speckle_num / size**2
    model_name = model.__class__.__name__
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
        if LOSS_SELECT == "l1":
            loss = l1_custom_loss(Y.squeeze(0), X_prime, mask_patterns, time_length=speckle_num, lambda_reg=alpha)
        elif LOSS_SELECT == "tv":
            loss = tv_custom_loss(Y.squeeze(0), X_prime, mask_patterns, time_length=speckle_num, lambda_reg=beta)
        elif LOSS_SELECT == "l1_tv":
            loss = l1_tv_custom_loss(Y.squeeze(0), X_prime, mask_patterns, time_length=speckle_num, alpha=alpha, beta=beta)
        else:
            loss = custom_loss(
                Y.squeeze(0), X_prime, mask_patterns, time_length=speckle_num
                )  # 損失を計算
        # loss = l1_custom_loss(Y.squeeze(0), X_prime, mask_patterns, time_length=speckle_num, lambda_reg=0.0001)

        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.7f}")

    # 学習後の再構成画像を出力
    model.eval()
    with torch.no_grad():
        X_reconstructed = (
            model(Y).squeeze().cpu().numpy()
        )  # Yから再構成画像X'を生成し、NumPy配列に変換
        X_original = image_data.cpu().numpy()  # 元の画像XをNumPy配列に変換
        print(X_reconstructed.shape)
    # 再構成画像の表示
    display_comparison_with_metrics(
        X_original=X_original, X_reconstructed=X_reconstructed, speckle_num=speckle_num, 
        size=size, USE_DATA=USE_DATA, LOSS_SELECT=LOSS_SELECT, EPOCHS=EPOCHS, LEARNING_RATE=LEARNING_RATE,
        model_name=model_name, ratio=COMPRESSIVE_RATIO
    )


if __name__ == "__main__":
    print("EPOCHS:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    for speckle_num in speckle_num_list:
        print(f"Processing speckle_num: {speckle_num}")
        # COMPRESSIVE_RATIO = speckle_num / size**2
        if f"time{speckle_num}_{size}x{size}.npz" not in os.listdir(f"{PATH}/data/speckle/"):
        # print(os.listdir(f"{PATH}/data/speckle/"))
            print("SPECKLE does not exist!!")
        # print(list(MASK_PATTERNS.items()))
        if speckle_num == 65536:
            MASK_PATTERNS = np.load(f"{PATH}/data/speckle/time{speckle_num}_{size}x{size}.npz")[
            "mask_patterns_normalized"].astype(np.float32)
        else:
            MASK_PATTERNS = np.load(f"{PATH}/data/speckle/time{speckle_num}_{size}x{size}.npz")[
            "arr_0"].astype(np.float32)
        # model = Autoencoder(
        # input_dim=speckle_num, hidden_dim=speckle_num // 256, bottleneck_dim=256, output_dim=size**2
        # ).to(DEVICE)
        # model = shal_1_DeepMultiscaleSpeckleNet(outdim=65536).to(DEVICE)
        model = DeepMultiscaleSpeckleNet(outdim=65536).to(DEVICE)
        model_name = model.__class__.__name__
        # main(device=DEVICE, mask_patterns=MASK_PATTERNS, image_data=IMAGE, speckle_num=speckle_num, model=model)
        main(speckle_num=speckle_num, model=model, mask_patterns=MASK_PATTERNS, image_data=IMAGE, device=DEVICE)
        # print(MASK_PATTERNS)
        # print(IMAGE.shape)
