import numpy as np
import logging
import os
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
    mask = np.load(f"/home1/komori/spi_simulate/data/speckle/time{width**2}_{width}x{width}.npz")["mask_patterns_normalized"]
    indices = np.round(np.linspace(0, len(mask) - 1, (width**2) * 3 // 4)).astype(int)
    # print(indices)
    mask_75 = mask[indices]
    # print(mask_75.shape)
    # print(type(mask_75))
    mask_50 = mask[::2]
    # print(mask_50.shape)
    mask_25 = mask[::4]
    # print(mask_25.shape)
    np.savez(f"/home1/komori/spi_simulate/data/speckle/time{len(indices)}_{width}x{width}.npz", mask_75)
    np.savez(f"/home1/komori/spi_simulate/data/speckle/time{(width**2 // 2)}_{width}x{width}.npz", mask_50)
    np.savez(f"/home1/komori/spi_simulate/data/speckle/time{(width**2 // 4)}_{width}x{width}.npz", mask_25)


def npz_create():
    data_npz = np.load("data/raw/cameraman.npz")["arr_0"]
    # print(data_npz.shape)
    data_true = data_npz.reshape(256 * 256)
    print(data_true.shape)
    np.savez("data/processed/cameraman.npz", data_true / 255)

def combine_npz_chunks(input_dir, output_file, prefix='time65536_256x256_chunk', num_chunks=16):
    """
    Combine multiple .npz chunk files into a single .npz file.

    Parameters:
    - input_dir: str
        Directory where the chunk .npz files are located.
    - output_file: str
        Path for the combined output .npz file.
    - prefix: str, optional
        Prefix of the chunk files. Default is 'time65536_256x256_chunk'.
    - num_chunks: int, optional
        Number of chunk files to combine. Default is 16.
    """
    combined_patterns = []
    
    # チャンクファイルの読み込み
    for i in range(1, num_chunks + 1):
        file_name = f"{prefix}{i}.npz"
        file_path = os.path.join(input_dir, file_name)
        
        if not os.path.exists(file_path):
            logging.error(f"File {file_path} does not exist. Skipping.")
            continue
        
        try:
            with np.load(file_path) as data:
                mask_patterns = data['mask_patterns_normalized']
                combined_patterns.append(mask_patterns)
                logging.info(f"Loaded {file_path} with shape {mask_patterns.shape}")
        except KeyError:
            logging.error(f"'mask_patterns_normalized' not found in {file_path}. Skipping.")
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}. Skipping.")
    
    if not combined_patterns:
        logging.error("No data loaded. Exiting without saving.")
        return
    
    # 配列の結合
    try:
        combined_patterns = np.concatenate(combined_patterns, axis=0)
        logging.info(f"Combined shape: {combined_patterns.shape}")
    except Exception as e:
        logging.error(f"Error concatenating arrays: {e}. Exiting without saving.")
        return
    
    # 結合した配列の保存
    try:
        np.savez_compressed(output_file, mask_patterns_normalized=combined_patterns)
        logging.info(f"Combined file saved to {output_file}")
        print(f"Combined file saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving combined file: {e}.")
        print(f"Error saving combined file: {e}.")


if __name__ == "__main__":
    # npz_data_mnist(0)
    divide_mask(256)
    # npz_create()
        # ログの設定
    # logging.basicConfig(
    #     filename='/home1/komori/spi_simulate/data/speckle/combine_chunks.log',
    #     level=logging.DEBUG,
    #     format='%(asctime)s %(levelname)s:%(message)s'
    # )
    
    # input_directory = '/home1/komori/spi_simulate/data/speckle/'  # チャンクファイルが保存されているディレクトリ
    # output_file_path = '/home1/komori/spi_simulate/data/speckle/combined_mask_patterns.npz'  # 結合後のファイルパス
    
    # combine_npz_chunks(
    #     input_dir=input_directory,
    #     output_file=output_file_path,
    #     prefix='time65536_256x256_chunk',
    #     num_chunks=16
    # )
