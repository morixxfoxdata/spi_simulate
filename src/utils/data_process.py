import numpy as np

path = "data/processed/mnist/"


def npz_data_mnist(num):
    data_all = np.load("data/raw/class10_image10x1000.npz")["arr_0"]
    data_available_ten = data_all[10:20, :]
    data_num = data_available_ten[num]
    print(data_num.shape)
    np.savez(path + f"mnist_28x28_{num}.npz", data_num / 255)


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


if __name__ == "__main__":
    # npz_data_mnist(0)
    divide_mask(8)
