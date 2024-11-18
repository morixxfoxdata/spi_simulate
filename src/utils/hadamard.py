"""This script is to create Hadamard Matrix Pattern"""

import matplotlib.pyplot as plt
import numpy as np


def hadamard(n: int) -> np.ndarray:
    """
    args
    n(int) : The size of Hadamard you want

    return
    (ndarray) : (n, n) hadamard matrix
    """
    # nが2のべき乗でない場合はエラーを返す
    if (n & (n - 1)) != 0 or n <= 0:
        raise ValueError("n must be a power of 2")
    # 1次元のアダマール行列
    if n == 1:
        return np.array([[1]])

    # n > 1の場合、行列を再帰的に生成
    h = hadamard(n // 2)
    # 上下左右に行列を配置し、対称的にマイナスを加える
    return np.block([[h, h], [h, -h]])


def hadamard_pattern(n: int) -> np.ndarray:
    """
    args
    n : size of hadamard matrix you want
    Return
    (ndarray): hadamard matrix consists of binary number (-1 → 0)
    and that shape is (n^2, n^2)
    """
    h_matrix = hadamard(n)
    hadamard_pattern_matrix = np.zeros((n * n, n * n))
    count = 0
    for i in range(n):
        for j in range(n):
            row_vector = h_matrix[i, :]
            col_vector = h_matrix[:, j]
            outer_matrix = np.outer(row_vector, col_vector)
            outer_to_vec = outer_matrix.flatten()
            hadamard_pattern_matrix[count, :] = outer_to_vec
            count += 1
    return np.where(hadamard_pattern_matrix > 0, 1, 0)


def hadamard_pattern_alternate(n: int) -> np.ndarray:
    """
    args
    n(int): size of hadamard matrix you want

    return
    h_pattern_alternate(ndarray) :hadamard matrix consists of binary number (-1 → 0)
    and that shape is (2*(n^2), n^2)
    """
    h_normal = hadamard_pattern(n)
    h_reversed = np.where(h_normal > 0, 0, 1)
    h_pattern_alternate = np.empty(
        (2 * h_normal.shape[0], h_normal.shape[1]), dtype=h_normal.dtype
    )
    h_pattern_alternate[0::2, :] = h_normal
    h_pattern_alternate[1::2, :] = h_reversed
    return h_pattern_alternate


def plot_some_img(n: int, rows: int, cols: int, alternate: bool):
    """
    args
    n(int) : size of hadamard you want
    rows(int) : a number of rows of sublplot
    cols(int) : a number of cols of subplot
    alternate(bool) : whether you want to get alternate data or not
    """
    if alternate:
        kind = "alternate"
        hmatrix = hadamard_pattern_alternate(n)
    else:
        kind = "normal"
        hmatrix = hadamard_pattern(n)
    # half_n = int(np.sqrt(n))
    dim = int(np.sqrt(hmatrix.shape[1]))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 8))
    for k in range(rows * cols):
        axes[k // cols, k % cols].imshow(hmatrix[k, :].reshape(dim, dim), cmap="gray")
        # axes[k // 4, k % 4].axis("off")

    plt.suptitle(f"Plotted {rows*cols} out of {dim**2} Hadamard pattern {kind}")
    path = (
        "img/"
        + str(rows * cols)
        + "_outof_"
        + str(dim**2)
        + "Hadamard_pattern_"
        + kind
        + ".png"
    )
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)  # リソースの解放


def save_hadamard_data(n, alternate=True):
    """
    args
    n(int) : size of hadamard you want
    alternate(bool): whether you want to get alternate data or not
    """
    if alternate:
        kind = "alternate"
        saved_data = hadamard_pattern_alternate(n)
    else:
        kind = "normal"
        saved_data = hadamard_pattern(n)
    np.savez(
        "data/hadamard/HP_image_"
        + str(n**2)
        + "_size_"
        + str(n)
        + "x"
        + str(n)
        + "_"
        + kind,
        saved_data,
    )


if __name__ == "__main__":
    save_hadamard_data(32, alternate=False)
