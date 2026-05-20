from __future__ import annotations

import numpy as np
from scipy import constants # type: ignore
from scipy.special import factorial # type: ignore

def DTabulationToBetas(lambda0: float, data: np.ndarray, poly_order: int, *,
                         data_is_file: bool = False, make_plots: bool = False) -> np.ndarray:
    if data_is_file:
        raise NotImplementedError()
    
    data[:, 1] *= 1e-12 * 1e9 * 1e-3

    c = constants.speed_of_light

    beta2 = -data[:, 1] * (data[:, 0] * 1e-9)**2 / (2.0 * np.pi * c)
    beta2 *= 1e24

    omega_axis = 2.0 * np.pi * c / (data[:, 0] * 1e-9)**2 - (2.0 * np.pi * c) / (lambda0 * 1e-9)
    omega_axis *= 1e-12

    poly_fit_co = np.polyfit(omega_axis, beta2, poly_order)

    betas = poly_fit_co[::-1]

    for i in range(len(betas)):
        betas[i] *= factorial(i)

    if make_plots:
        raise NotImplementedError()

    return betas