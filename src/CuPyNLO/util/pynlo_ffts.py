from __future__ import annotations

import numpy as np
from scipy.fft import ifftshift, ifft, fftshift, fft # type: ignore

def FFT_t(A: np.ndarray, ax: int = 0):
    return ifftshift(ifft(fftshift(A,axes=(ax,)),axis=ax),axes=(ax,))

def IFFT_t(A: np.ndarray, ax: int = 0):
    return ifftshift(fft(fftshift(A,axes=(ax,)),axis=ax),axes=(ax,)) 
