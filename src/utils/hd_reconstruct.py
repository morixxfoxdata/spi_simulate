import numpy as np
from matplotlib import pyplot as plt

PATH = "/Users/norikikomori/Desktop/spi_simulate"
size = 8
number = 0
HD_PATTERN = np.load(
    f"{PATH}/data/hadamard/HP_image_{size**2}_size_{size}x{size}_normal.npz"
)["arr_0"].astype(np.float32)
IMAGE = np.load(f"{PATH}/data/processed/mnist/mnist_{size}x{size}_{number}.npz")[
    "arr_0"
].astype(np.float32)
reshape_IMAGE = IMAGE.reshape(size**2, -1)
yy = np.dot(HD_PATTERN, reshape_IMAGE)
ada_sp = np.dot(yy.T, HD_PATTERN.T) / size**2
if __name__ == "__main__":
    print(HD_PATTERN.shape)
    print(reshape_IMAGE.shape)
    print(yy.shape)
    print(ada_sp.shape)
    plt.imshow(ada_sp.reshape(size, size), cmap="gray")

    plt.show()
