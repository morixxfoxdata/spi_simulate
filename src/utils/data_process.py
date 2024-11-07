import numpy as np

path = "data/processed/mnist/"


def npz_data_mnist(num):
    data_all = np.load("data/raw/class10_image10x1000.npz")["arr_0"]
    data_available_ten = data_all[10:20, :]
    data_num = data_available_ten[num]
    print(data_num.shape)
    np.savez(path + f"mnist_{num}.npz", data_num)


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


if __name__ == "__main__":
    npz_8_8_mnist(1)
