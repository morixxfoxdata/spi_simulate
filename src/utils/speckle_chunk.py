import numpy as np
from tqdm import tqdm
import logging
import os
import sys

# ログの設定
logging.basicConfig(
    filename='/home1/komori/spi_simulate/data/speckle/script_debug.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# グローバルでシードを設定
np.random.seed(42)

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
def generate_mask_pattern(
    time_length=65536, 
    num_x_pixel_true=256, 
    num_y_pixel_true=256, 
    chunk_size=1000  # 一度に処理する時間ステップ数
):
    logging.debug("generate_mask_pattern関数開始")
    try:
        ####### Spatial sampling ###
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
                _beta[m] = np.sqrt(nref**2 * kw**2 - kx[i]**2 - ky[j]**2)
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

        # Prepare for chunked saving
        num_chunks = (time_length + chunk_size - 1) // chunk_size
        logging.debug(f"Total time_length: {time_length}, chunk_size: {chunk_size}, num_chunks: {num_chunks}")

        for chunk_idx in range(num_chunks):
            start_n = chunk_idx * chunk_size
            end_n = min((chunk_idx + 1) * chunk_size, time_length)
            current_chunk_size = end_n - start_n
            logging.debug(f"Processing chunk {chunk_idx + 1}/{num_chunks}: time steps {start_n} to {end_n - 1}")

            # Initialize mask_speckles for the current chunk
            mask_speckles_chunk = np.zeros((current_chunk_size, num_x_pixel, num_y_pixel), dtype=np.complex64)

            # Generate speckles for the current chunk
            for n in tqdm(range(start_n, end_n), desc=f"Generating speckles (chunk {chunk_idx + 1}/{num_chunks})"):
                local_n = n - start_n
                for m in range(num_modes):
                    mask_speckles_chunk[local_n, :, :] += aa[m] * psi[m, :, :] * phase_modulated_input[m, n]

            # Compute mask_patterns_normalized for the current chunk
            mask_patterns = np.abs(mask_speckles_chunk) ** 2
            mask_patterns_cropped = mask_patterns[:, :num_x_pixel_true, :num_y_pixel_true].astype(np.float32)
            mask_patterns_normalized = mask_patterns_cropped / np.mean(mask_patterns_cropped)

            # Save mask_patterns_normalized as npz for the current chunk
            save_path = f"/home1/komori/spi_simulate/data/speckle/time{time_length}_{num_x_pixel_true}x{num_y_pixel_true}_chunk{chunk_idx + 1}.npz"
            logging.debug(f"保存先パス: {save_path}")
            np.savez(
                save_path,
                mask_patterns_normalized=mask_patterns_normalized
            )
            logging.debug(f"ファイル保存完了: {save_path}")
            print(f"Saved chunk {chunk_idx + 1}/{num_chunks}")

        logging.debug("generate_mask_pattern関数終了")
        return

    except Exception as e:
        logging.error(f"エラー発生: {e}")
        print(f"エラー発生: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # パラメータを適宜調整してください
    generate_mask_pattern(
        time_length=256, 
        num_x_pixel_true=256, 
        num_y_pixel_true=256, 
        chunk_size=32  # チャンクサイズを適宜調整
    )