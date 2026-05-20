from __future__ import annotations

import cupy as np
from cupyx.scipy.fft import ifftshift, ifft, fftshift, fft

def FFT_t(A: np.ndarray, ax: int = 0):
    return ifftshift(ifft(fftshift(A,axes=(ax,)),axis=ax),axes=(ax,))

def IFFT_t(A: np.ndarray, ax: int = 0):
    return ifftshift(fft(fftshift(A,axes=(ax,)),axis=ax),axes=(ax,)) 
