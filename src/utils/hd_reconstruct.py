import numpy as np
from matplotlib import pyplot as plt

PATH = "/Users/norikikomori/Desktop/spi_simulate"
# PATH = "/home1/komori/spi_simulate"
size = 32
number = 0
HD_PATTERN = np.load(
    f"{PATH}/data/hadamard/HP_image_{size**2}_size_{size}x{size}_normal.npz"
)["arr_0"].astype(np.float32)
IMAGE = np.load(f"{PATH}/data/processed/mnist/mnist_{size}x{size}_{number}.npz")[
    "arr_0"
].astype(np.float32)
reshape_IMAGE = IMAGE.flatten()
yy = np.dot(HD_PATTERN, reshape_IMAGE)
recon_vec = np.dot(HD_PATTERN.T, yy)
recon_img = recon_vec.reshape(size, size)
reconstructed_image = (recon_img - np.min(recon_img)) / (
    np.max(recon_img) - np.min(recon_img)
)
# ada_sp = np.dot(yy.T, HD_PATTERN.T) / size
# H_inv = np.dot(yy.T, reshape_IMAGE) / size**2
# recon_h = np.dot(yy, H_inv)
if __name__ == "__main__":
    print("HD size: ", HD_PATTERN.shape)
    print("HD min, max: ", HD_PATTERN)
    print("flatten image:", reshape_IMAGE.shape)
    print("yy shape: ", yy.shape)
    # print(recon_h.shape)
    print("max yy: ", max(yy))
    print("max reshape_IMG:", max(reshape_IMAGE))
    plt.imshow(reconstructed_image, cmap="gray")
    plt.show()
    # plt.show()
