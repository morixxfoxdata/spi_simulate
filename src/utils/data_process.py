import numpy as np

path = "data/processed/mnist/"


# def npz_data_mnist(num):
#     data_all = np.load("data/raw/class10_image10x1000.npz")["arr_0"]
#     data_available_ten = data_all[10:20, :]
#     data_num = data_available_ten[num]
#     print(data_num.shape)
#     np.savez(path + f"mnist_28x28_{num}.npz", data_num / 255)


def npz_data_mnist(num):
    data_all = np.load("data/raw/class10_image10x1000.npz")["arr_0"]
    data_available_ten = data_all[10:20, :]
    data_num = data_available_ten[num]
    print("Original shape:", data_num.shape)

    # データを28x28にリシェイプ（必要な場合）
    if data_num.ndim == 2 and data_num.shape[1] == 784:
        data_num = data_num.reshape(-1, 28, 28)
    elif data_num.ndim == 1 and data_num.shape[0] == 784:
        data_num = data_num.reshape(28, 28)

    # パディングを追加して32x32に拡張
    data_num_padded = np.pad(
        data_num, ((2, 2), (2, 2)), mode="constant", constant_values=0
    )
    print("Padded shape:", data_num_padded.shape)

    # パスを適切に設定してください
    np.savez(path + f"mnist_32x32_{num}.npz", data_num_padded / 255)


def npz_8_8_mnist(num):
    data_all = np.load(
        "data/raw/HP_mosaic_random_size8x8_image64+10+500_alternate.npz"
    )["arr_0"]

    # print(data_all.shape)
    mnist_data_88 = data_all[128:148, :]
    X_mnist = mnist_data_88[0::2, :]
    # data_available_ten = data_all[10:20, :]
    data_num = X_mnist[num]
    print(data_num.shape)
    np.savez(path + f"mnist_8x8_{num}.npz", data_num)


def divide_mask(width):
    mask = np.load(f"data/speckle/time{width**2}_{width}x{width}.npz")["arr_0"]
    indices = np.round(np.linspace(0, len(mask) - 1, (width**2) * 3 // 4)).astype(int)
    # print(indices)
    mask_75 = mask[indices]
    # print(mask_75.shape)
    # print(type(mask_75))
    mask_50 = mask[::2]
    # print(mask_50.shape)
    mask_25 = mask[::4]
    # print(mask_25.shape)
    np.savez(f"data/speckle/time{len(indices)}_{width}x{width}.npz", mask_75)
    np.savez(f"data/speckle/time{(width**2 // 2)}_{width}x{width}.npz", mask_50)
    np.savez(f"data/speckle/time{(width**2 // 4)}_{width}x{width}.npz", mask_25)


def npz_create():
    data_npz = np.load("data/raw/cameraman.npz")["arr_0"]
    # print(data_npz.shape)
    data_true = data_npz.reshape(256 * 256)
    print(data_true.shape)
    np.savez("data/processed/cameraman.npz", data_true / 255)


if __name__ == "__main__":
    npz_data_mnist(0)
    # divide_mask(8)
    # npz_create()
