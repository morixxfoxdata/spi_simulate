import numpy as np
from tqdm import tqdm

#####################################
# Set parameters
#####################################
fiber_length = 20  # Fiber length (m)
width = 200e-6  # Fiber core width  (m)
nref = 1.4611  # Fiber refractive index
na = 0.39  # N.A.

lw0 = 1550.0e-9  # center wavelgnth (m)
c = 3.0e8  # light velocity (m/s)
omega0 = 2.0 * np.pi * c / lw0  # angular frequency (rad/s)
kw = omega0 / c  # wavenumber (1/m)

####### Spatial sampling ###
# num_x_pixel = 200
# num_y_pixel = 200
# dx = width / num_x_pixel
# dy = width / num_y_pixel
# x = np.arange(1, num_x_pixel + 1) * dx
# y = np.arange(1, num_y_pixel + 1) * dy

####### Input signal ######
# time_length = 10  # Time length of input signal
dt_w = 0.04e-9  # Time interval of input signal (s)
dt_samp = 0.02e-9  # Sampling time interval (s)
al = 2.0 * np.pi  # Input signal amplitude
num_repeat = int(dt_w / dt_samp)  # number of repeat


######################################
# Fixed parameters
######################################
def generate_mask_pattern(
    time_length=10, num_x_pixel_true=200, num_y_pixel_true=200
) -> np.ndarray:  # (time_length, num_x_pixel, num_y_pixel)
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

    # xy wavenumbers
    kx = np.linspace(1, num_sq_modes, num_sq_modes) * (np.pi / width)
    ky = np.linspace(1, num_sq_modes, num_sq_modes) * (np.pi / width)

    # beta wavenumbers & mode patterns
    _beta = np.zeros(num_modes)
    _psi = np.zeros((num_modes, num_x_pixel, num_y_pixel))
    for i in range(num_sq_modes):
        for j in range(num_sq_modes):
            m = num_sq_modes * i + j
            _beta[m] = np.sqrt(nref * nref * kw * kw - kx[i] * kx[i] - ky[j] * ky[j])
            _psi[m, :, :] = np.matmul(
                np.array(
                    [
                        np.sin(kx[i] * x),
                    ]
                ).T,
                np.array(
                    [
                        np.sin(ky[j] * y),
                    ]
                ),
            )

    # sorting
    beta = np.sort(_beta)
    psi = _psi[np.argsort(_beta), :, :]

    # dispersion, d\beta/d\omega
    disp = ((nref * nref) / (c * c)) * omega0 / beta
    # time delay
    time_delay = disp * fiber_length
    # relative time delay
    relative_time_delay = time_delay - time_delay[0]
    # index
    i_delay = relative_time_delay / dt_samp
    i_delay = i_delay.astype(int)

    # mode amplitudes
    aa = np.random.uniform(-1.0, 1.0, num_modes) + 1j * np.random.uniform(
        -1.0, 1.0, num_modes
    )
    aa = aa * np.exp(1j * beta * fiber_length)

    # Input signal
    _uu = np.random.uniform(-1.0, 1.0, int(time_length / num_repeat))
    uu = al * np.repeat(_uu, num_repeat)

    # time delay
    uu_delay = np.zeros((num_modes, time_length))
    for m in range(num_modes):
        uu_delay[m] = np.roll(uu, i_delay[m])
    # phase modulated input singal
    phase_modulated_input = np.exp(1j * uu_delay)

    # speckle field
    mask_speckles = np.zeros((time_length, num_x_pixel, num_y_pixel)) * 1j
    for n in tqdm(range(time_length)):
        for m in range(num_modes):
            mask_speckles[n, :, :] += aa[m] * psi[m, :, :] * phase_modulated_input[m, n]

    mask_patterns = np.abs(mask_speckles) ** 2
    # oversampling
    # print(mask_patterns.shape)
    mask_patterns_cropped = mask_patterns.astype(float)[
        :, :num_x_pixel_true, :num_y_pixel_true
    ]
    mask_patterns_normalized = mask_patterns_cropped / np.mean(mask_patterns_cropped)

    # save mask_patterns as npz
    np.savez(
        f"data/speckle/time{time_length}_{num_x_pixel_true}x{num_y_pixel_true}.npz",
        mask_patterns_normalized,
    )
    return mask_patterns_normalized
