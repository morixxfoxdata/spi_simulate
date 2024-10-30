import numpy as np

path = "data/processed/mnist/"


def npz_data_mnist(num):
    data_all = np.load("data/raw/class10_image10x1000.npz")["arr_0"]
    data_available_ten = data_all[10:20, :]
    data_num = data_available_ten[num]
    print(data_num.shape)
    np.savez(path + f"mnist_{num}.npz", data_num)
