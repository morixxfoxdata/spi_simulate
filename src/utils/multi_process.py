import numpy as np
from tqdm import tqdm
import logging
import os
import sys
from multiprocessing import Pool, cpu_count
from functools import partial

# グローバル変数
global_params = None

def init_worker(params):
    """
    プロセスプールの各ワーカーで実行される初期化関数。
    ログ設定とグローバルパラメータの設定を行う。
    """
    global global_params
    global_params = params
    
    # 子プロセス内でログ設定を初期化
    logging.basicConfig(
        filename='/home1/komori/spi_simulate/data/speckle/script_debug.log',
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s:%(message)s'
    )
    logging.debug("Workerプロセスが初期化されました。")

#####################################
# Set parameters
#####################################
fiber_length = 20  # Fiber length (m)
width = 200e-6  # Fiber core width  (m)
nref = 1.4611  # Fiber refractive index
na = 0.39  # N.A.

lw0 = 1550.0e-9  # center wavelength (m)
c = 3.0e8  # light velocity (m/s)
omega0 = 2.0 * np.pi * c / lw0  # angular frequency (rad/s)
kw = omega0 / c  # wavenumber (1/m)

####### Input signal ######
dt_w = 0.04e-9  # Time interval of input signal (s)
dt_samp = 0.02e-9  # Sampling time interval (s)
al = 2.0 * np.pi  # Input signal amplitude
num_repeat = int(dt_w / dt_samp)  # number of repeat

######################################
# Fixed parameters
######################################
def prepare_global_parameters(time_length, num_x_pixel_true, num_y_pixel_true):
    num_x_pixel = num_x_pixel_true + 1
    num_y_pixel = num_y_pixel_true + 1
    dx = width / num_x_pixel
    dy = width / num_y_pixel
    x = np.arange(1, num_x_pixel + 1) * dx
    y = np.arange(1, num_y_pixel + 1) * dy

    # ----------- Multimode fiber --------------
    V_number = (np.pi * width / lw0) * na  # normalized frequency
    num_sq_modes = int(np.sqrt(0.5 * V_number * V_number))
    num_modes = num_sq_modes * num_sq_modes  # num. of excited modes

    logging.debug(f"num_sq_modes: {num_sq_modes}, num_modes: {num_modes}")

    # xy wavenumbers
    kx = np.linspace(1, num_sq_modes, num_sq_modes) * (np.pi / width)
    ky = np.linspace(1, num_sq_modes, num_sq_modes) * (np.pi / width)

    # beta wavenumbers & mode patterns
    _beta = np.zeros(num_modes)
    _psi = np.zeros((num_modes, num_x_pixel, num_y_pixel), dtype=np.float32)
    for i in range(num_sq_modes):
        for j in range(num_sq_modes):
            m = num_sq_modes * i + j
            try:
                _beta[m] = np.sqrt(nref**2 * kw**2 - kx[i]**2 - ky[j]**2)
            except ValueError as ve:
                logging.error(f"Mode {m}: sqrt of negative number encountered - {ve}")
                _beta[m] = 0  # 適切なデフォルト値を設定
            _psi[m, :, :] = np.outer(
                np.sin(kx[i] * x),
                np.sin(ky[j] * y)
            ).astype(np.float32)

    # sorting
    beta = np.sort(_beta)
    psi = _psi[np.argsort(_beta), :, :]

    # dispersion, d\beta/d\omega
    disp = ((nref**2) / (c**2)) * omega0 / beta
    # time delay
    time_delay = disp * fiber_length
    # relative time delay
    relative_time_delay = time_delay - time_delay[0]
    # index
    i_delay = (relative_time_delay / dt_samp).astype(int)

    # mode amplitudes
    aa = (np.random.uniform(-1.0, 1.0, num_modes) + 
          1j * np.random.uniform(-1.0, 1.0, num_modes))
    aa = aa * np.exp(1j * beta * fiber_length)

    # Input signal
    _uu = np.random.uniform(-1.0, 1.0, int(time_length / num_repeat))
    uu = al * np.repeat(_uu, num_repeat)

    # time delay
    uu_delay = np.zeros((num_modes, time_length), dtype=np.float32)
    for m in range(num_modes):
        uu_delay[m] = np.roll(uu, i_delay[m])
    # phase modulated input signal
    phase_modulated_input = np.exp(1j * uu_delay)

    return {
        'num_modes': num_modes,
        'psi': psi,
        'aa': aa,
        'phase_modulated_input': phase_modulated_input,
        'num_x_pixel_true': num_x_pixel_true,
        'num_y_pixel_true': num_y_pixel_true
    }

def generate_mask_pattern_chunk(chunk_info, chunk_size, time_length):
    """
    各チャンクを処理する関数。
    """
    chunk_idx, start_n, end_n = chunk_info
    try:
        logging.debug(f"Chunk {chunk_idx + 1}: Processing time steps {start_n} to {end_n - 1}")

        current_chunk_size = end_n - start_n

        # 正しい形状でmask_speckles_chunkを初期化
        num_x_pixel_true = global_params['num_x_pixel_true']
        num_y_pixel_true = global_params['num_y_pixel_true']
        num_x_pixel = num_x_pixel_true + 1
        num_y_pixel = num_y_pixel_true + 1

        mask_speckles_chunk = np.zeros((current_chunk_size, num_x_pixel, num_y_pixel), dtype=np.complex64)

        # Generate speckles for the current chunk
        for local_n, n in enumerate(tqdm(range(start_n, end_n), desc=f"Chunk {chunk_idx + 1}")):
            for m in range(global_params['num_modes']):
                try:
                    mask_speckles_chunk[local_n, :, :] += (
                        global_params['aa'][m] *
                        global_params['psi'][m, :, :] *
                        global_params['phase_modulated_input'][m, n]
                    )
                except Exception as e_inner:
                    logging.error(f"Chunk {chunk_idx + 1}, Mode {m}: エラー発生 - {e_inner}")
                    raise e_inner  # 再度例外を発生させてワーカープロセスを終了させる

        # Compute mask_patterns_normalized for the current chunk
        mask_patterns = np.abs(mask_speckles_chunk) ** 2
        mask_patterns_cropped = mask_patterns[:, :num_x_pixel_true, :num_y_pixel_true].astype(np.float32)
        mask_patterns_normalized = mask_patterns_cropped / np.mean(mask_patterns_cropped)

        # Save mask_patterns_normalized as npz for the current chunk
        save_path = f"/home1/komori/spi_simulate/data/speckle/time{time_length}_{num_x_pixel_true}x{num_y_pixel_true}_chunk{chunk_idx + 1}.npz"
        np.savez_compressed(
            save_path,
            mask_patterns_normalized=mask_patterns_normalized
        )
        logging.debug(f"Chunk {chunk_idx + 1}: Saved to {save_path}")
        print(f"Saved chunk {chunk_idx + 1}")
    except Exception as e:
        logging.error(f"Chunk {chunk_idx + 1}: エラー発生 - {e}")
        print(f"Chunk {chunk_idx + 1}: エラー発生 - {e}")
        # エラーが発生した場合、プロセスを終了させる
        sys.exit(1)

def generate_mask_pattern(
    time_length=65536, 
    num_x_pixel_true=256, 
    num_y_pixel_true=256, 
    chunk_size=4096  # メモリ使用量を抑えるためにチャンクサイズを小さく設定
):
    logging.debug("generate_mask_pattern関数開始")
    try:
        # グローバルパラメータの準備
        params = prepare_global_parameters(time_length, num_x_pixel_true, num_y_pixel_true)

        # チャンクの情報を作成
        num_chunks = (time_length + chunk_size - 1) // chunk_size
        logging.debug(f"Total time_length: {time_length}, chunk_size: {chunk_size}, num_chunks: {num_chunks}")

        chunk_infos = []
        for chunk_idx in range(num_chunks):
            start_n = chunk_idx * chunk_size
            end_n = min((chunk_idx + 1) * chunk_size, time_length)
            chunk_infos.append((chunk_idx, start_n, end_n))
            logging.debug(f"Chunk {chunk_idx + 1}: start_n={start_n}, end_n={end_n}")

        # プロセス数の設定
        num_processes = min(cpu_count(), num_chunks, 4)  # メモリ使用量を抑えるためにプロセス数を4に制限
        logging.debug(f"Using {num_processes} processes for parallel execution")

        # 並列処理の実行
        with Pool(processes=num_processes, initializer=init_worker, initargs=(params,)) as pool:
            func = partial(generate_mask_pattern_chunk, chunk_size=chunk_size, time_length=time_length)
            pool.map(func, chunk_infos)

        logging.debug("generate_mask_pattern関数終了")
        print("All chunks have been processed and saved.")
    except Exception as e:
        logging.error(f"エラー発生: {e}")
        print(f"エラー発生: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # パラメータを適宜調整してください
    generate_mask_pattern(
        time_length=65536, 
        num_x_pixel_true=256, 
        num_y_pixel_true=256, 
        chunk_size=4096  # メモリ使用量を抑えるためにチャンクサイズを4096に設定
    )
